"""Gallery region index (one mask-pooled vector per object) and its search index."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .pooling import masked_sum, to_input

CHUNK_EVERY = 100          # images per resumable chunk
FAISS_MAX_BYTES = 8 << 30  # larger galleries are searched from disk instead


# ----------------------------------------------------------------------------
# Building
# ----------------------------------------------------------------------------

def _write_chunk(chunk_dir: Path, cid: int, vecs: List[np.ndarray], meta: List[dict], img_ids: List[int]) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for name, write in (
        (f"u_O_{cid:04d}.npy", lambda p: np.save(p, np.stack(vecs).astype(np.float32))),
        (f"meta_{cid:04d}.json", lambda p: json.dump(meta, open(p, "w"))),
        (f"done_{cid:04d}.json", lambda p: json.dump(img_ids, open(p, "w"))),
    ):
        tmp = chunk_dir / f"tmp_{name}"
        write(tmp)
        shutil.move(str(tmp), str(chunk_dir / name))


def build_region_index(dataset, extractor, target_size: int, mean: List[float], std: List[float],
                       out_dir: Path, batch_size: int = 8) -> Path:
    """Encode every gallery image once and store, per object, the sum of the
    patch features inside its mask (u_O.npy) plus metadata (object_meta.json).

    Resumable: completed chunks are kept and skipped on restart.
    """
    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        print(f"region index already built: {out_dir}")
        return out_dir
    chunk_dir = out_dir / "chunks"
    done = set()
    for f in sorted(chunk_dir.glob("done_*.json")):
        done.update(json.load(open(f)))
    for f in chunk_dir.glob("tmp_*"):
        f.unlink()
    next_chunk = len(list(chunk_dir.glob("done_*.json")))

    todo = [i for i in dataset.image_ids
            if i not in done and dataset.image_path(i) is not None and dataset.image_path(i).exists()]
    print(f"images to encode: {len(todo)} (already done: {len(done)})")

    vecs, meta, img_ids = [], [], []
    n_new = 0
    pbar = tqdm(total=len(todo), desc="region index")
    for start in range(0, len(todo), batch_size):
        batch, tensors = [], []
        for img_id in todo[start:start + batch_size]:
            md = dataset.mask_data(img_id)
            if md is None:
                pbar.update(1)
                continue
            with Image.open(dataset.image_path(img_id)) as im:
                tensors.append(to_input(im.convert("RGB"), target_size, mean, std))
            batch.append((img_id, md))
        if not tensors:
            continue
        feats = extractor.dense_features(torch.stack(tensors)).cpu()
        for (img_id, md), feat in zip(batch, feats):
            for obj, m in dataset.objects(img_id, md):
                u = masked_sum(feat, obj)
                if u is None:
                    continue
                m["n_O"] = int(obj.sum())
                vecs.append(u)
                meta.append(m)
            img_ids.append(img_id)
            n_new += 1
            pbar.update(1)
            if n_new % CHUNK_EVERY == 0 and vecs:
                _write_chunk(chunk_dir, next_chunk, vecs, meta, img_ids)
                next_chunk += 1
                vecs, meta, img_ids = [], [], []
        del feats
        torch.cuda.empty_cache()
    pbar.close()
    if vecs:
        _write_chunk(chunk_dir, next_chunk, vecs, meta, img_ids)

    # Merge chunks in order into one memory-mapped array.
    parts = sorted(chunk_dir.glob("u_O_*.npy"))
    n = sum(np.load(p, mmap_mode="r").shape[0] for p in parts)
    d = np.load(parts[0], mmap_mode="r").shape[1]
    u_all = np.lib.format.open_memmap(str(out_dir / "u_O.npy"), mode="w+", dtype=np.float32, shape=(n, d))
    all_meta, row = [], 0
    for p in tqdm(parts, desc="merging chunks"):
        a = np.load(p)
        u_all[row:row + len(a)] = a
        row += len(a)
        all_meta.extend(json.load(open(chunk_dir / p.name.replace("u_O_", "meta_").replace(".npy", ".json"))))
    u_all.flush()
    del u_all
    json.dump(all_meta, open(out_dir / "object_meta.json", "w"))
    json.dump({"dataset": dataset.name, "split": dataset.split, "cache_id": extractor.cache_id,
               "n_objects": n, "feature_dim": d}, open(out_dir / "config.json", "w"), indent=2)
    shutil.rmtree(chunk_dir)
    print(f"region index: {n} objects, {d}-D -> {out_dir}")
    return out_dir


def load_object_meta(index_dir: Path) -> Tuple[List[Dict], Dict]:
    with open(Path(index_dir) / "config.json") as f:
        config = json.load(f)
    with open(Path(index_dir) / "object_meta.json") as f:
        meta = json.load(f)
    return meta, config


# ----------------------------------------------------------------------------
# Searching
# ----------------------------------------------------------------------------

class MmapIPIndex:
    """Exact inner-product search over an (N, D) .npy read from disk in chunks.

    Same interface as faiss.IndexFlatIP (d, ntotal, search) for galleries
    too large to hold in RAM. Gallery rows are L2-normalized on the fly.
    """

    def __init__(self, npy_path: Path, chunk_size: int = 50_000) -> None:
        self._mmap = np.load(str(npy_path), mmap_mode="r")
        self.ntotal, self.d = self._mmap.shape
        self._chunk = chunk_size

    def search(self, queries: np.ndarray, k: int):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        nq = q.shape[0]
        top_s = np.full((nq, k), -np.inf, dtype=np.float32)
        top_i = np.full((nq, k), -1, dtype=np.int64)
        for s in range(0, self.ntotal, self._chunk):
            e = min(s + self._chunk, self.ntotal)
            db = np.array(self._mmap[s:e], dtype=np.float32)
            db /= np.maximum(np.linalg.norm(db, axis=1, keepdims=True), 1e-12)
            cs = np.concatenate([top_s, q @ db.T], axis=1)
            ci = np.concatenate([top_i, np.broadcast_to(np.arange(s, e), (nq, e - s))], axis=1)
            if cs.shape[1] > k:
                part = np.argpartition(-cs, k, axis=1)[:, :k]
                cs, ci = np.take_along_axis(cs, part, 1), np.take_along_axis(ci, part, 1)
            top_s, top_i = cs, ci
        order = np.argsort(-top_s, axis=1)
        return np.take_along_axis(top_s, order, 1), np.take_along_axis(top_i, order, 1)


def _write_faiss(u_path: str, out_path: str) -> None:
    import faiss

    u = np.load(u_path, mmap_mode="r")
    idx = faiss.IndexFlatIP(u.shape[1])
    for s in range(0, u.shape[0], 50_000):
        c = np.array(u[s:s + 50_000], dtype=np.float32)
        c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
        idx.add(c)
    faiss.write_index(idx, out_path)


def open_search_index(index_dir: Path):
    """FAISS IndexFlatIP over the L2-normalized gallery (built once, then
    memory-mapped), or an MmapIPIndex when it would exceed 8 GB."""
    import faiss

    index_dir = Path(index_dir)
    u_path = index_dir / "u_O.npy"
    n, d = np.load(str(u_path), mmap_mode="r").shape
    if n * d * 4 > FAISS_MAX_BYTES:
        return MmapIPIndex(u_path)
    f_path = index_dir / "faiss_ip.index"
    if not f_path.exists():
        # Separate process so the build's peak memory is released afterwards.
        rc = subprocess.call([sys.executable, "-m", "kap.index", str(u_path), str(f_path)])
        if rc != 0 or not f_path.exists():
            print("FAISS build failed; searching from disk instead")
            return MmapIPIndex(u_path)
    return faiss.read_index(str(f_path), faiss.IO_FLAG_MMAP)


if __name__ == "__main__":
    _write_faiss(sys.argv[1], sys.argv[2])
