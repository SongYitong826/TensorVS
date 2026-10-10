"""Benchmark Faiss Metal's actual IVF-PQ CPU search against existing Torch CSVs.

Run with conda run -n tensorvs python -u bench/faiss_metal_ivfpq_cpu_bench.py
The Metal wrapper stores codes on GPU but delegates every query to CPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
ROOT = Path(__file__).resolve().parents[1]
FAISS_ROOT = ROOT.parent / "faiss-metal"
sys.path.insert(0, str(FAISS_ROOT / "local_bench"))
import benchmark_sift as common
import faiss
import numpy as np

TORCH_STEM = "torch_mps_IVFPQ_sift_nlist1024_M16_nbits8_train256"
STEM = "faiss_metal_cpu_IVFPQ_sift_nlist1024_M16_nbits8_train256"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_index(source):
    cache_dir = FAISS_ROOT / "local_bench/indexes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "sift_ivfpq_from_tensorvs.faiss"
    manifest = cache.with_suffix(".json")
    signature = common.source_signature(source)
    exporter = ROOT / "bench/export_ivfpq_to_numpy.py"
    if cache.exists() and manifest.exists():
        previous = json.loads(manifest.read_text())
        if (previous["source"] == signature and previous["exporter_sha256"] == digest(exporter)
                and previous["cache_sha256"] == digest(cache)):
            print(f"Loading converted index: {cache}", flush=True)
            return faiss.read_index(str(cache)), previous
    with tempfile.TemporaryDirectory(prefix="export-pq-", dir=cache_dir) as directory:
        export = Path(directory)
        subprocess.run([sys.executable, str(exporter), str(source), str(export)], check=True)
        centers, books, codes, ids, counts, offsets = (
            np.load(export / f"{name}.npy", mmap_mode="r")
            for name in ("centroids", "codebooks", "codes", "ids", "counts", "offsets"))
        assert centers.shape == (1024,128) and books.shape == (16,256,8)
        assert codes.shape == (1000000,16) and codes.dtype == np.uint8
        assert ids.dtype == np.int64 and np.array_equal(np.sort(ids), np.arange(len(ids)))
        quantizer = faiss.IndexFlatL2(128)
        quantizer.add(centers)
        index = faiss.IndexIVFPQ(quantizer,128,1024,16,8,faiss.METRIC_L2)
        faiss.copy_array_to_vector(np.ascontiguousarray(books).reshape(-1), index.pq.centroids)
        index.by_residual = True
        index.is_trained = True
        for list_id, count in enumerate(counts):
            count, start = int(count), int(offsets[list_id])
            if count:
                list_ids = ids[start:start+count]
                list_codes = codes[start:start+count].reshape(-1)
                index.invlists.add_entries(list_id,count,faiss.swig_ptr(list_ids),faiss.swig_ptr(list_codes))
                np.testing.assert_array_equal(faiss.rev_swig_ptr(index.invlists.get_ids(list_id),count),list_ids)
                np.testing.assert_array_equal(faiss.rev_swig_ptr(index.invlists.get_codes(list_id),count*16),list_codes)
        index.ntotal = len(ids)
        np.testing.assert_array_equal(index.quantizer.reconstruct_n(0,1024), centers)
        np.testing.assert_array_equal(faiss.vector_to_array(index.pq.centroids).reshape(books.shape), books)
        assert all(index.invlists.list_size(i) == int(counts[i]) for i in range(1024))
        index.precompute_table()
        temporary = cache.with_suffix(".faiss.tmp")
        faiss.write_index(index,str(temporary))
        temporary.replace(cache)
        conversion = dict(source=signature, cache=str(cache), cache_sha256=digest(cache),
            exporter_sha256=digest(exporter), dimension=128, nlist=1024, M=16, nbits=8,
            vectors=len(ids), rotation="identity", pq_mode="residual",
            training="reused original coarse centroids and PQ codebooks; no retraining",
            assignments="exact original PQ codes, vector IDs and inverted-list assignments",
            all_imported_arrays_verified_exact=True)
        common.write_json(manifest,conversion)
    return index, conversion


def run_batch(batch_size, cpu, wrapper, conversion, queries, truth, threads):
    folder = ROOT / "results" / f"qbatch{batch_size}"
    reference = folder / f"{TORCH_STEM}.csv"
    if not reference.exists():
        reference = ROOT / "results/ivfpq_gpu_schedule" / f"qbatch{batch_size}" / f"{TORCH_STEM}.csv"
    reference_meta = json.loads(reference.with_suffix(".json").read_text())
    assert reference_meta["status"] == "complete" and reference_meta["q_batch"] == batch_size
    assert reference_meta["cpu_operator_fallback"] is False
    assert reference_meta["query_count"] == len(queries) == 1024
    assert reference_meta["warmup_scope"] == "all-batches"
    assert conversion["source"] == dict(path=reference_meta["index"],
        size_bytes=reference_meta["index_size_bytes"],mtime_ns=reference_meta["index_mtime_ns"])
    with reference.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(common.FIELDS)
        reference_rows = list(reader)
    assert len(reference_rows) == 14
    path = folder / f"{STEM}.csv"
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        raise FileExistsError(path)
    reference_hash = digest(reference)
    warmup, repetitions = reference_meta["warmup_iterations"], reference_meta["repetitions"]
    counts = np.array([cpu.invlists.list_size(i) for i in range(cpu.nlist)],dtype=np.int64)
    batches = [queries[start:start+batch_size] for start in range(0,len(queries),batch_size)]
    source = FAISS_ROOT / "faiss/gpu_metal/MetalIndexIVFPQ.mm"
    verification = []
    for probe in (5,150):
        params = faiss.SearchParametersIVF()
        params.nprobe = probe
        values, ids = wrapper.search(queries[:8],10,params=params)
        cpu_values, cpu_ids = cpu.search(queries[:8],10,params=params)
        np.testing.assert_allclose(values,cpu_values,rtol=0,atol=0)
        np.testing.assert_array_equal(ids,cpu_ids)
        verification.append(dict(nprobes=probe,queries=8,cpu_delegate_matches_cpu_index_exactly=True))
    metadata = dict(status="running",algorithm="ivfpq",backend="faiss_metal_cpu",device="cpu",
        method="Faiss Metal IVF-PQ (CPU search)",faiss=faiss.__version__,faiss_import_path=faiss.__file__,
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=FAISS_ROOT,text=True).strip(),
        query_execution_device="cpu",gpu_query_execution=False,gpu_index_storage=True,
        cpu_execution="MetalIndexIVFPQ.search directly delegates to cpuIndex_->search",
        search_source=str(source),search_source_sha256=digest(source),
        benchmark_source_sha256=digest(Path(__file__)),memory_helper_sha256=digest(FAISS_ROOT/"local_bench/benchmark_sift.py"),
        metal_device=common.METAL.metal_device_name().decode(),omp_threads=threads,
        reference=str(reference),reference_sha256=reference_hash,index_conversion=conversion,
        query_count=len(queries),query_subset="same first 1024 queries as Torch reference",q_batch=batch_size,
        K=10,warmup_iterations=warmup,warmup_scope="all-batches",repetitions=repetitions,
        expected_rows=len(reference_rows),verification=verification,
        lut_dtype="fp32",precision_comparison="Faiss CPU uses FP32 lookup tables; existing Torch MPS uses FP16 LUT storage and FP32 accumulation",
        timing="median of 3 full-query sweeps; sum of synchronous per-batch Faiss Python API wall time; resident index and prepared host query batches; excludes recall and memory sampling",
        timing_comparison_limit=f"Faiss search runs on CPU with host input/output and {threads} OpenMP threads; Torch search runs on Apple GPU with resident MPS input/output. This compares the available library execution paths, not equal-device algorithms.",
        coarse_precision_comparison="Original centroids and list assignments are identical. CPU and MPS FP32 coarse-distance rounding/tie order can select a different list near the nprobe cutoff; measured visited-vector counts are reported independently.",
        visited_vectors_definition="sum of original IVF list sizes selected by the imported coarse quantizer, averaged over queries; excludes padding",
        memory_unit="MiB",memory_metric="sampled process Metal driver allocation in an independent sweep; includes GPU index storage and caches. Query computation is CPU; this is not CPU RAM usage or GPU search workspace usage.",results=[])
    common.write_csv(path,[])
    common.write_json(meta_path,metadata)
    rows = []
    started = time.perf_counter()
    try:
        for reference_row in reference_rows:
            k, probe = int(reference_row["K"]),int(reference_row["nprobes"])
            assert k == 10 and int(reference_row["q_batch"]) == batch_size
            params = faiss.SearchParametersIVF()
            params.nprobe = probe
            _, lists = cpu.quantizer.search(queries,probe)
            visited = float(counts[lists].sum(axis=1).mean())
            reference_visited=float(reference_row["visited_vectors_per_query"])
            assert math.isclose(visited,reference_visited,rel_tol=1e-4), "Unexpected coarse-selection discrepancy"
            for _ in range(warmup):
                for batch in batches:
                    wrapper.search(batch,k,params=params)

            def sweep(retain=False,sampler=None):
                elapsed, saved = 0.0, []
                for batch in batches:
                    begin = time.perf_counter()
                    values, ids = wrapper.search(batch,k,params=params)
                    elapsed += time.perf_counter()-begin
                    if sampler is not None:
                        sampler.sample()
                    if retain:
                        assert np.isfinite(values).all() and (ids>=0).all() and (ids<cpu.ntotal).all()
                        assert all(len(set(row)) == k for row in ids)
                        saved.append(ids)
                return elapsed,saved

            samples=[]
            for repetition in range(repetitions):
                elapsed,saved=sweep(retain=repetition==repetitions-1)
                samples.append(elapsed)
            ids=np.concatenate(saved)
            recall=float(common.recall_at_k(ids,truth,k))
            with common.MemorySampler() as sampler:
                sweep(sampler=sampler)
            row=dict(nprobes=probe,q_batch=batch_size,K=k,qps=len(queries)/statistics.median(samples),
                recall=recall,visited_vectors_per_query=visited,peak_gpu_memory=sampler.peak/(1<<20))
            assert all(math.isfinite(float(v)) for v in row.values())
            rows.append(row)
            metadata["results"].append(dict(row,samples_seconds=samples,memory_samples=sampler.samples,
                reference_qps=float(reference_row["qps"]),reference_recall=float(reference_row["recall"]),
                reference_visited_vectors_per_query=reference_visited,
                visited_vectors_difference=visited-reference_visited))
            common.write_csv(path,rows)
            common.write_json(meta_path,metadata)
            print(f"QBatch={batch_size} {len(rows)}/14 nprobe={probe} CPU QPS={row['qps']:.2f} recall={recall:.6f}",flush=True)
        assert digest(reference)==reference_hash
        assert common.source_signature(Path(conversion["source"]["path"]))==conversion["source"]
        metadata.update(status="complete",wall_seconds=time.perf_counter()-started)
        common.write_json(meta_path,metadata)
    except BaseException as error:
        metadata.update(status="failed",error=repr(error),wall_seconds=time.perf_counter()-started)
        common.write_json(meta_path,metadata)
        raise
    mirror=FAISS_ROOT/"results"/f"qbatch{batch_size}"
    mirror.mkdir(parents=True,exist_ok=True)
    for artifact in (path,meta_path):
        target=mirror/artifact.name
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(artifact,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-batches",type=int,nargs="+",default=[256,512])
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    assert args.threads>0 and faiss.__version__=="1.15.1"
    faiss.omp_set_num_threads(args.threads)
    common.METAL=common.memory_api()
    source=Path("/Users/yitongsong/Dataset/sift/GPUIndex/TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin")
    cpu,conversion=import_index(source)
    resources=faiss.StandardGpuResources()
    wrapper=faiss.index_cpu_to_gpu(resources,0,cpu)
    assert wrapper.ntotal==cpu.ntotal==1000000
    print(f"Metal IVF-PQ ready; queries execute on CPU; OpenMP threads={faiss.omp_get_max_threads()}",flush=True)
    dataset=source.parent.parent
    queries=common.read_vectors(dataset/"sift_query.fvecs",np.float32)[:1024]
    truth=common.read_vectors(dataset/"sift_groundtruth.ivecs",np.int64)[:1024]
    for batch in args.q_batches:
        run_batch(batch,cpu,wrapper,conversion,queries,truth,args.threads)


if __name__=="__main__":
    main()
