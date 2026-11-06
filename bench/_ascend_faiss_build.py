"""Build and persist a Faiss Ascend index from raw vectors (no TensorVS index)."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def pad_vectors(vectors, dim=256):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] > dim:
        raise ValueError("Invalid input/working dimensions")
    if vectors.shape[1] == dim:
        return np.ascontiguousarray(vectors)
    result = np.zeros((len(vectors), dim), dtype=np.float32)
    result[:, :vectors.shape[1]] = vectors
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def create_residual_ip_index(faiss, resources, options):
    return faiss.NpuIndexIVFPQ(resources, 256, 1024, 32, 8,
                             faiss.METRIC_INNER_PRODUCT, True, options)


def exact_ip_ground_truth(faiss, base_path, queries, topk, cache_path):
    """Cache exhaustive raw-IP neighbors; never use the dataset's L2 truth."""
    from tensorvs import load_dataset

    base_path, cache_path = Path(base_path), Path(cache_path)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    signature = dict(metric="inner_product", normalization="none", topk=topk,
        base=str(base_path.resolve()), size_bytes=base_path.stat().st_size,
        mtime_ns=base_path.stat().st_mtime_ns,
        queries_sha256=hashlib.sha256(queries.tobytes()).hexdigest(),
        queries_shape=list(queries.shape), implementation="Faiss CPU IndexFlatIP exhaustive search")
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            ids = cached["ids"]
            if json.loads(str(cached["signature"])) == signature:
                if ids.shape == (len(queries), topk) and (ids >= 0).all():
                    return ids, dict(signature=signature, path=str(cache_path), cached=True)
    print("Preparing exact IP ground truth on CPU (outside benchmark timing)", flush=True)
    started = time.perf_counter()
    base = load_dataset(base_path).numpy()
    if base.ndim != 2 or base.shape[1] != queries.shape[1] or len(base) < topk:
        raise ValueError("Invalid base/query dimensions or K for IP ground truth")
    flat = faiss.IndexFlatIP(base.shape[1])
    for offset in range(0, len(base), 65536):
        flat.add(np.ascontiguousarray(base[offset:offset + 65536]))
    ids = np.concatenate([flat.search(queries[offset:offset + 32], topk)[1]
                          for offset in range(0, len(queries), 32)])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, ids=ids, signature=json.dumps(signature, sort_keys=True))
    temporary.replace(cache_path)
    return ids, dict(signature=signature, path=str(cache_path), cached=False,
                     preparation_seconds=time.perf_counter() - started)


def build_or_load(faiss, resources, options, args, base_path):
    """NPU train + full add; cache original build timings for resumed retrieval."""
    from tensorvs import load_dataset

    base_path = Path(base_path)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    path = args.index_dir / "Ascend_Faiss_IVFPQ_glove1.2m_nlist1024_M32_nbits8_train512.faiss"
    manifest_path = path.with_suffix(".build.json")
    signature = dict(base=str(base_path.resolve()), size_bytes=base_path.stat().st_size,
        mtime_ns=base_path.stat().st_mtime_ns, nlist=1024, m=32, nbits=8,
        original_dim=200, working_dim=256, metric="inner_product", by_residual=True,
        normalization="none", preprocessing="zero padding; inner products unchanged",
        train_device="NPU", seed=args.seed, coarse_niter=args.coarse_niter, pq_niter=args.pq_niter,
        coarse_max_points_per_centroid=512, pq_max_points_per_centroid=512,
        train_size=args.train_size, add_batch_size=args.add_batch_size,
        lut_dtype=args.lut_dtype, faiss_version=faiss.__version__,
        faiss_source_revision=os.environ.get("FAISS_ASCEND_SOURCE_REVISION"),
        faiss_source_patch_sha256=os.environ.get("FAISS_ASCEND_SOURCE_PATCH_SHA256"), npu_id=args.npu_id,
        numpy_version=np.__version__, cann_home=os.environ.get("ASCEND_HOME_PATH"),
        faiss_library_sha256={p.name: digest(p) for p in Path(faiss.__file__).parent.glob("_swigfaiss*.so")},
        builder_sha256=digest(__file__))
    if (path.exists() or manifest_path.exists()) and not args.rebuild_index:
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError("Incomplete index cache; use --rebuild-index")
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature or manifest["index_sha256"] != digest(path):
            raise ValueError("Cached index settings/data/content differ; use --rebuild-index")
        cpu = faiss.read_index(str(path))
        index = faiss.NpuIndexIVFPQ(resources, cpu, options)
        resources.getResources().syncDefaultStream(args.npu_id)
        build = manifest["build"]
        loaded = True
        print(f"Loaded Faiss-built index: {path} (original build time retained)", flush=True)
    else:
        prepare_start = time.perf_counter()
        base = load_dataset(base_path).numpy()
        if base.ndim != 2 or base.shape[1] != 200 or len(base) < 1024:
            raise ValueError("Expected GloVe base vectors of dimension 200, at least 1024 rows")
        base = pad_vectors(base)
        train = base
        if args.train_size and args.train_size < len(base):
            selection = np.random.default_rng(args.seed).choice(len(base), args.train_size, replace=False)
            train = np.ascontiguousarray(base[selection])
        if len(train) < 1024:
            raise ValueError("At least nlist=1024 training vectors are required")
        preparation_seconds = time.perf_counter() - prepare_start
        started = time.perf_counter()
        # Explicit residual mode, native NPU training. No TensorVS centroids,
        # codebooks, rotations, codes or assignments are loaded.
        index = create_residual_ip_index(faiss, resources, options)
        index.pq.cp.niter = args.pq_niter
        index.pq.cp.seed = args.seed
        index.pq.cp.nredo = 1
        index.pq.cp.max_points_per_centroid = 512
        index.verbose = True
        resources.getResources().syncDefaultStream(args.npu_id)
        train_start = time.perf_counter()
        print(f"Faiss Ascend train: {len(train)} input rows, 200 -> 256 dimensions, M32", flush=True)
        index.train(train)
        resources.getResources().syncDefaultStream(args.npu_id)
        train_seconds = time.perf_counter() - train_start
        add_start = time.perf_counter()
        for offset in range(0, len(base), args.add_batch_size):
            index.add(base[offset:offset + args.add_batch_size])
        resources.getResources().syncDefaultStream(args.npu_id)
        add_seconds = time.perf_counter() - add_start
        build_seconds = time.perf_counter() - started
        if not index.is_trained or int(index.ntotal) != len(base):
            raise RuntimeError("Incomplete Faiss index build")
        # Serialization and D2H export are outside build_time_seconds.
        serialization_start = time.perf_counter()
        quantizer = faiss.IndexFlatIP(256)
        cpu = faiss.IndexIVFPQ(quantizer, 256, 1024, 32, 8, faiss.METRIC_INNER_PRODUCT)
        index.copyTo(cpu)
        cpu.referenced_objects = [quantizer]
        temporary = path.with_suffix(".faiss.tmp")
        faiss.write_index(cpu, str(temporary))
        temporary.replace(path)
        serialize_seconds = time.perf_counter() - serialization_start
        size = path.stat().st_size
        build = dict(train_seconds=train_seconds, add_seconds=add_seconds,
            build_time_seconds=build_seconds, preparation_seconds=preparation_seconds,
            build_with_preparation_seconds=preparation_seconds + build_seconds,
            serialize_seconds=serialize_seconds, index_size_bytes=size,
            index_size_mb=size / (1 << 20), index_size_unit="MiB",
            index_path=str(path.resolve()), vectors=int(index.ntotal), metric="inner_product",
            training_input_rows=len(train), coarse_training_rows=min(len(train), 1024 * 512),
            pq_training_rows=min(len(train), 256 * 512), training_backend="Faiss Ascend useNpuTrain=True",
            timing_definition="constructor + train + full add + NPU stream synchronization; excludes input preparation and serialization",
            size_definition="serialized CPU-portable Faiss index including centroids, codebooks, vector IDs and PQ codes; not NPU HBM usage")
        manifest = dict(signature=signature, build=build, index_sha256=digest(path))
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
        loaded = False
        print(f"Build: {build_seconds:.3f} s (train {train_seconds:.3f}, add {add_seconds:.3f}); "
              f"index {size / (1 << 20):.3f} MiB", flush=True)
    counts = np.array([index.getListLength(i) for i in range(1024)], dtype=np.int64)
    if cpu.metric_type != faiss.METRIC_INNER_PRODUCT or not cpu.by_residual:
        raise RuntimeError("Expected residual inner-product index")
    if counts.sum() != index.ntotal or cpu.ntotal != index.ntotal:
        raise RuntimeError("Faiss list counts/vector totals differ")
    return index, cpu, counts, build, loaded
