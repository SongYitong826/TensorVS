"""Measure fresh Faiss Metal IVF builds from original Apple-hosted base vectors.

Train + Metal upload + add are timed; host I/O, zero padding and serialization
are separate. Training uses the installed Faiss CPU implementation, as does
the native Metal train method. M>64 IVF-PQ uses Faiss CPU because Metal rejects
that configuration; the build record explicitly identifies this execution path.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"glove1.2m": (1193514, 200, 32), "hunyuan": (1000000, 768, 384)}


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=PROFILES, required=True)
    parser.add_argument("--kind", choices=("ivfflat", "ivfpq"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Dataset")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes" / "faiss_metal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result" / "Build")
    parser.add_argument("--faiss-root", type=Path, default=ROOT.parent / "faiss-metal")
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--coarse-iters", type=int, default=20)
    parser.add_argument("--pq-iters", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=.5)
    parser.add_argument("--pq-train-points-per-centroid", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-vectors", type=int, default=0, help="Smoke tests only; zero means full dataset")
    args = parser.parse_args()
    if min(args.nlist, args.threads, args.coarse_iters, args.pq_iters,
           args.pq_train_points_per_centroid, args.batch_size) <= 0 or not 0 < args.train_fraction <= 1:
        parser.error("Counts must be positive and train fraction in (0,1]")
    for name in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["FAISS_METAL_IVF_ALLOW_CPU_FALLBACK"] = "0"
    import faiss
    import numpy as np

    faiss.omp_set_num_threads(args.threads)
    expected, original_dim, pq_m = PROFILES[args.dataset]
    m = pq_m if args.kind == "ivfpq" else 0
    dim = math.ceil(original_dim / m) * m if m else original_dim
    stem = f"MPS_FaissMetal_{args.kind.upper()}_{args.dataset}_nlist{args.nlist}"
    if m:
        stem += f"_M{m}_nbits8_train{args.pq_train_points_per_centroid}"
        if dim != original_dim:
            stem += f"_pad{dim}"
    if args.max_vectors:
        stem += f"_smoke{args.max_vectors}"
    csv_path = args.output_dir / f"{stem}_build.csv"
    meta_path = csv_path.with_suffix(".json")
    index_path = args.index_dir / f"{stem}.index"
    for path in (csv_path, meta_path, index_path):
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing measurement/index: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.data_root / args.dataset / f"{args.dataset}_base.fvecs"
    source_stat = base_path.stat()
    read_started = time.perf_counter()
    raw = np.memmap(base_path, dtype="<i4", mode="r")
    if int(raw[0]) != original_dim or raw.size % (original_dim + 1):
        raise ValueError("Base vector shape is incorrect")
    records = raw.reshape(-1, original_dim + 1)
    if len(records) != expected or not np.all(records[:, 0] == original_dim):
        raise ValueError("Base vector count/dimension headers do not match the dataset")
    count = min(expected, args.max_vectors) if args.max_vectors else expected
    base = np.ascontiguousarray(records[:count, 1:].view("<f4"))
    if not np.isfinite(base).all():
        raise ValueError("Base vectors contain nonfinite values")
    read_seconds = time.perf_counter() - read_started
    pad_started = time.perf_counter()
    if dim != original_dim:
        base = np.pad(base, ((0, 0), (0, dim - original_dim)))
    pad_seconds = time.perf_counter() - pad_started
    del raw, records
    quantizer = faiss.IndexFlatL2(dim)
    cpu = (faiss.IndexIVFPQ(quantizer, dim, args.nlist, m, 8) if m else
           faiss.IndexIVFFlat(quantizer, dim, args.nlist, faiss.METRIC_L2))
    coarse_max_points = math.ceil(count * args.train_fraction / args.nlist)
    cpu.cp.niter = args.coarse_iters
    cpu.cp.max_points_per_centroid = coarse_max_points
    cpu.cp.seed = args.seed
    if m:
        cpu.pq.cp.niter = args.pq_iters
        cpu.pq.cp.max_points_per_centroid = args.pq_train_points_per_centroid
        cpu.pq.cp.seed = args.seed
    on_metal = not m or m <= 64
    print(f"{args.dataset} {args.kind}: {count:,} vectors, D={original_dim}->{dim}, "
          f"nlist={args.nlist}, M={m}; training CPU, add {'Metal' if on_metal else 'CPU (M>64)'}", flush=True)
    state = {
        "status": "running", "stage": "train", "dataset": args.dataset,
        "kind": args.kind, "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(meta_path, state)
    started = time.perf_counter()
    cpu.train(base)
    train_seconds = time.perf_counter() - started
    print(f"Training finished in {train_seconds:.2f}s", flush=True)
    write_json(meta_path, dict(state, stage="add", train_seconds=train_seconds))
    transfer_started = time.perf_counter()
    resources = faiss.StandardGpuResources() if on_metal else None
    index = faiss.index_cpu_to_gpu(resources, 0, cpu) if on_metal else cpu
    if on_metal:
        faiss.gpu_sync_all_devices()
    transfer_seconds = time.perf_counter() - transfer_started
    add_seconds = 0.0
    for first in range(0, count, args.batch_size):
        add_started = time.perf_counter()
        index.add(base[first:first + args.batch_size])
        if on_metal:
            faiss.gpu_sync_all_devices()
        add_seconds += time.perf_counter() - add_started
        print(f"Added {min(first + args.batch_size, count):,}/{count:,}", flush=True)
    # Exclude status checkpoint I/O and progress printing between timed phases.
    build_seconds = train_seconds + transfer_seconds + add_seconds
    if int(index.ntotal) != count or not index.is_trained:
        raise RuntimeError("Index is incomplete")
    write_json(meta_path, dict(state, stage="save", train_seconds=train_seconds, build_seconds=build_seconds))
    save_started = time.perf_counter()
    serializable = faiss.index_gpu_to_cpu(index) if on_metal else index
    temporary = index_path.with_suffix(".index.tmp")
    faiss.write_index(serializable, str(temporary))
    save_seconds = time.perf_counter() - save_started
    size_bytes = temporary.stat().st_size
    temporary.replace(index_path)
    checks = base[:8].copy()
    # Release the full base/index buffers before independently loading the saved index.
    del serializable, index, cpu, quantizer, resources, base
    gc.collect()
    validation_started = time.perf_counter()
    restored = faiss.read_index(str(index_path))
    if int(restored.ntotal) != count or int(restored.d) != dim or int(restored.nlist) != args.nlist:
        raise RuntimeError("Saved index metadata is incorrect")
    if m and (int(restored.pq.M) != m or int(restored.pq.nbits) != 8):
        raise RuntimeError("Saved PQ configuration is incorrect")
    if sum(int(restored.invlists.list_size(i)) for i in range(args.nlist)) != count:
        raise RuntimeError("Saved lists do not contain every vector")
    restored.nprobe = min(32, args.nlist)
    distances, ids = restored.search(checks, 10)
    if not np.isfinite(distances).all() or np.any(ids < 0) or np.any(ids >= count):
        raise RuntimeError("Saved index failed search validation")
    validation_seconds = time.perf_counter() - validation_started
    if base_path.stat().st_size != source_stat.st_size or base_path.stat().st_mtime_ns != source_stat.st_mtime_ns:
        raise RuntimeError("Source base file changed during the build")
    source_files = (args.faiss_root / "faiss/gpu_metal/MetalIndexIVFFlat.mm",
                    args.faiss_root / "faiss/gpu_metal/MetalIndexIVFPQ.mm")
    row = {
        "dataset": args.dataset, "method": "Faiss Metal", "index_type": args.kind,
        "nlist": args.nlist, "n_vectors": count, "dim": original_dim,
        "working_dim": dim, "M": m, "nbits": 8 if m else 0,
        "build_seconds": build_seconds, "train_seconds": train_seconds,
        "metal_upload_seconds": transfer_seconds, "add_seconds": add_seconds,
        "save_seconds": save_seconds, "index_size_bytes": size_bytes,
        "index_size_mib": size_bytes / 1024**2, "index_size_gib": size_bytes / 1024**3,
        "build_device": "CPU train + Metal upload/add" if on_metal else "CPU",
    }
    metadata = {
        **row, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": "Apple M5", "threads": args.threads,
        "faiss_version": faiss.__version__, "faiss_module": faiss.__file__,
        "base_path": str(base_path), "base_size_bytes": source_stat.st_size,
        "base_mtime_ns": source_stat.st_mtime_ns, "index_path": str(index_path),
        "read_seconds": read_seconds, "pad_seconds": pad_seconds,
        "coarse_niter": args.coarse_iters, "coarse_max_points_per_centroid": coarse_max_points,
        "coarse_train_fraction_target": args.train_fraction, "pq_niter": args.pq_iters if m else None,
        "pq_train_points_per_centroid": args.pq_train_points_per_centroid if m else None,
        "seed": args.seed, "add_batch_size": args.batch_size,
        "peak_build_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "timing": "Native Faiss CPU train + Metal upload (if supported) + full add; excludes file I/O, zero padding, serialization and validation",
        "padding": f"{dim-original_dim} trailing zero dimensions" if dim != original_dim else "none",
        "execution_note": "Metal training delegates to CPU; Metal IVF-PQ rejects M>64, so Hunyuan M384 uses Faiss CPU train/add",
        "serialization": "faiss.index_gpu_to_cpu + faiss.write_index" if on_metal else "faiss.write_index",
        "validation": {"saved_count": count, "sum_list_sizes": count, "search_queries": 8,
                       "search_k": 10, "validation_seconds": validation_seconds},
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "smoke_test": bool(args.max_vectors),
    }
    with csv_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    write_json(meta_path, metadata)
    print(f"COMPLETE build={build_seconds:.3f}s, saved={size_bytes / 1024**3:.6f}GiB; {meta_path}", flush=True)


if __name__ == "__main__":
    main()
