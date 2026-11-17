"""Export original single-device CUDA index statistics; import them into Build/.

conda run -n tensorvs python bench/organize_cuda_build_results.py --ssh-host GPU-yitong
No index is rebuilt. Raw statistics, exact file sizes, index metadata, and
query-memory provenance are retained in JSON. Unknown measurements stay empty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys

ROOT = Path.cwd() if __file__ == "<stdin>" else Path(__file__).resolve().parents[1]
DATASETS = ("sift1m", "sift10m", "glove1.2m", "deep1m", "tiny5m", "hunyuan", "gist", "openai1536")


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def match_statistics(rows, kind, method, size):
    candidates = []
    for row in rows:
        if normalized(row.get("Index Type", "")) != kind or normalized(row.get("Method Name", "")) != method:
            continue
        measured = float(row["Index Size (MB)"])
        errors = {"MiB": abs(measured - size / 2**20), "MB": abs(measured - size / 1e6)}
        unit = min(errors, key=errors.get)
        candidates.append(dict(row=row, size_error=errors[unit], recorded_size_unit=unit))
    if not candidates:
        return None, []
    best = min(item["size_error"] for item in candidates)
    matching = [item for item in candidates if abs(item["size_error"] - best) < 1e-7 and best <= .05]
    timings = {float(item["row"]["Build Time (s)"]) for item in matching}
    return (next(iter(timings)) if len(timings) == 1 else None), matching


def export(data_root, repository):
    import torch
    prefixes = {"TenVS": ("TensorVS_Triton", "TensorVS", repository / "result"),
                "Faiss": ("FaissGPU", "faiss", Path("/home/songyitong/faiss/result")),
                "cuVS": ("cuVS", "cuVS", Path("/home/songyitong/cuvs/python/cuvs/cuvs/result"))}
    entries = []
    for dataset in DATASETS:
        folder = data_root / dataset / "GPUIndex"
        stats_path = folder / "statistic.csv"
        with stats_path.open(newline="") as file:
            statistics = list(csv.DictReader(file))
        base = data_root / dataset / f"{dataset}_base.fvecs"
        with base.open("rb") as file:
            dim = struct.unpack("<i", file.read(4))[0]
        n_vectors, remainder = divmod(base.stat().st_size, 4 * (dim + 1))
        if remainder:
            raise ValueError(f"Invalid fvecs size: {base}")
        for index in sorted(folder.iterdir()):
            match = re.fullmatch(r"(TenVS|Faiss|cuVS)_(IVFFlat|IVFPQ(?:Residual)?)_nlist(\d+)(.*)\.(bin|index)", index.name)
            if not match:
                continue
            prefix, family, nl, suffix, _ = match.groups()
            kind = "ivfflat" if family == "IVFFlat" else "ivfpq"
            method, label, query_root = prefixes[prefix]
            m = bits = 0
            if kind == "ivfpq":
                pq = re.search(r"_(?:m|M|pqdim)(\d+)_(?:nbits|pqbits)(\d+)", suffix)
                if not pq:
                    raise ValueError(index)
                m, bits = map(int, pq.groups())
            metadata = {}
            if prefix == "TenVS":
                payload = torch.load(index, map_location="cpu", mmap=True, weights_only=False)
                metadata = payload["metadata"]
                assert int(metadata["nlist"]) == int(nl)
                assert int(metadata["dim"]) == dim and int(metadata["num_vectors"]) == n_vectors
                del payload
            size = index.stat().st_size
            build, candidates = match_statistics(statistics, kind, normalized(label), size)
            if metadata.get("build_time_seconds") is not None:
                build = metadata["build_time_seconds"]
            stem = f"CUDA_{method}_{kind.upper()}_{dataset}_nlist{nl}"
            tail = ""
            if kind == "ivfpq":
                stem += f"_M{m}_nbits{bits}"
                tail = (f"_M{m}_nbits{bits}" if prefix != "cuVS" else f"_pqdim{m}_pqbits{bits}")
            query_candidates = []
            query_pattern = f"{kind.upper()}_{dataset}_nlist{nl}{tail}*.csv"
            patterns = (["CUDA_TensorVS_Triton_" + query_pattern, query_pattern]
                        if prefix == "TenVS" else [query_pattern])
            for pattern in patterns:
                query_candidates = [p for p in sorted(query_root.glob(pattern))
                                    if "_QBatch" not in p.name]
                if query_candidates:
                    break
            if len(query_candidates) > 1:
                raise ValueError(f"Ambiguous query configuration: {query_candidates}")
            query = query_candidates[0] if query_candidates else None
            if prefix == "TenVS" and kind == "ivfpq":
                train = re.search(r"_train(\d+)", query.stem if query else index.stem)
                stem += f"_train{train[1] if train else metadata.get('pq_max_points_per_centroid', 256)}"
            memory = {}
            query_info = None
            if query:
                with query.open(newline="") as file:
                    for row in csv.DictReader(file):
                        if int(float(row.get("K", row.get("k", -1)))) != 10 or int(float(row["q_batch"])) != 128:
                            continue
                        probe = str(int(float(row["nprobes"])))
                        value = float(row["peak_gpu_memory"])
                        if probe in memory and memory[probe] != value:
                            raise ValueError(f"Duplicate query memory: {query}/{probe}")
                        memory[probe] = value
                query_info = dict(source_path=str(query), sha256=hashlib.sha256(query.read_bytes()).hexdigest(),
                                  K=10, q_batch=128, peak_gpu_memory_mib_by_nprobe=memory)
            record = dict(dataset=dataset, method=label, nlist=int(nl), n_vectors=n_vectors, dim=dim,
                          build_seconds=build, save_seconds=None, index_size_bytes=size,
                          index_size_mib=size / 2**20, peak_build_gpu_memory_mib=None)
            extra = dict(status="source_index_metadata", index_type=kind, M=m, nbits=bits,
                         index_path=str(index), index_mtime_ns=index.stat().st_mtime_ns,
                         build_device=metadata.get("device") if prefix == "TenVS" else "cpu" if prefix == "Faiss" else "cuda",
                         source_index_metadata=metadata, statistics_source=str(stats_path),
                         statistics_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                         matching_statistics=candidates,
                         statistics_rows=[r for r in statistics if normalized(r.get("Index Type", "")) == kind and normalized(r.get("Method Name", "")) == normalized(label)],
                         query_statistics=query_info,
                         measurement_notes=["Build time comes from saved index metadata or uniquely size-matched legacy statistics.",
                                            "Index size is the exact serialized file size in bytes; legacy MB/MiB units are checked.",
                                            "Serialization time and build peak GPU memory were not separately measured; empty means unknown.",
                                            "Query memory is kept separately in query_statistics and is never used as build peak memory.",
                                            "Multiple equally matched legacy timings are retained as candidates; an ambiguous build time stays empty."])
            entries.append(dict(filename=stem + "_build.csv", record=record, metadata=extra))
    return dict(host=socket.gethostname(), dataset_root=str(data_root), entries=entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("/storage/vector_data"))
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--ssh-host")
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "result" / "Build")
    args = parser.parse_args()
    if args.export:
        print(json.dumps(export(args.data_root, args.repository)))
        return
    if args.ssh_host:
        command = "cd /home/songyitong/TensorVS && /home/songyitong/miniconda3/bin/conda run --no-capture-output -n tensorvs python - --export"
        completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.ssh_host, command],
                                   input=Path(__file__).read_text(), capture_output=True, text=True, check=True)
        data = json.loads(completed.stdout)
        snapshot = ROOT / "tmp" / "cuda_build_source.json"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(data, indent=2) + "\n")
    elif args.source_json:
        data = json.loads(args.source_json.read_text())
    else:
        parser.error("Provide --ssh-host or --source-json")
    sys.path.insert(0, str(ROOT))
    from bench._cuda_build_results import write_build
    from plot.ivf_resource_table import DATASETS as TABLE_DATASETS, PQ_SPECS, METHODS
    legacy_path = ROOT / "plot" / "ivf_cuda_resources.json"
    legacy_rows = json.loads(legacy_path.read_text())["records"] if legacy_path.exists() else []
    legacy_hash = hashlib.sha256(legacy_path.read_bytes()).hexdigest() if legacy_rows else None
    legacy_map = {}
    for old in legacy_rows:
        dataset = next(item for item in TABLE_DATASETS if item.label == old["dataset"])
        label = "TensorVS" if old["method"] == "TensorVS (Triton)" else old["method"]
        method = next(item for item in METHODS if item.label == label)
        index_name = method.flat_index_name if old["kind"] == "ivfflat" else PQ_SPECS[dataset.key][label].index_name
        legacy_map[dataset.key, old["kind"], index_name] = dict(
            peak_gpu_memory_gib=None if old["peak"] == "N/S" else float(old["peak"]),
            K=10, q_batch=128, source_snapshot_sha256=legacy_hash,
            original_rounded_record=old,
            measurement_notes="Historical query peak rounded to 0.01 GiB; retained separately from newer logical query-allocation counters and build peaks.")
    names = [entry["filename"] for entry in data["entries"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate destination filenames")
    for entry in data["entries"]:
        path = args.build_dir / entry["filename"]
        previous = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
        metadata = dict(entry["metadata"], source_host=data["host"])
        key = entry["record"]["dataset"], metadata["index_type"], Path(metadata["index_path"]).name
        legacy = legacy_map.get(key, previous.get("legacy_table_statistics"))
        if legacy is not None:
            metadata["legacy_table_statistics"] = legacy
        write_build(path, entry["record"], metadata)
    print(f"Wrote {len(names)} CUDA single-device build CSV/JSON pairs to {args.build_dir}")
    missing = [entry["filename"] for entry in data["entries"] if entry["record"]["build_seconds"] is None]
    print(f"Build time unavailable/ambiguous for {len(missing)} records: {missing}")


if __name__ == "__main__":
    main()
