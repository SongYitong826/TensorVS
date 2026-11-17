"""Single-device CUDA build records, using the existing AMD/CUDA CSV schema."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "result" / "Build"
BUILD_FIELDS = ("dataset", "method", "nlist", "n_vectors", "dim", "build_seconds",
                "save_seconds", "index_size_bytes", "index_size_mib", "peak_build_gpu_memory_mib")


def write_build(path: Path, record: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=BUILD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(record)
    temporary.replace(path)
    sidecar = path.with_suffix(".json")
    payload = dict(metadata, **record, schema_version=1, platform="CUDA",
                   csv_columns=list(BUILD_FIELDS),
                   csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(sidecar)


def load_build(dataset: str, method: str, kind: str, nlist: int, *, m=None, nbits=None,
               index_name=None, build_dir=None) -> tuple[Path, dict]:
    directory = Path(build_dir) if build_dir is not None else BUILD_DIR
    pattern = f"CUDA_{method}_{kind.upper()}_{dataset}_nlist{nlist}*_build.json"
    matches = []
    for path in sorted(directory.glob(pattern)):
        if "MultiGPU" in path.name:
            continue
        data = json.loads(path.read_text())
        if (data.get("dataset"), data.get("index_type"), int(data.get("nlist", -1))) != (dataset, kind, nlist):
            continue
        if kind == "ivfpq" and (int(data.get("M", -1)), int(data.get("nbits", -1))) != (m, nbits):
            continue
        if index_name is not None and Path(data["index_path"]).name != index_name:
            continue
        csv_path = path.with_suffix(".csv")
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != data["csv_sha256"]:
            raise ValueError(f"CUDA build CSV/JSON checksum mismatch: {csv_path}")
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            records = list(reader)
            if tuple(reader.fieldnames or ()) != BUILD_FIELDS or len(records) != 1:
                raise ValueError(f"Invalid CUDA build CSV schema: {csv_path}")
        for field in BUILD_FIELDS:
            value = data.get(field)
            if records[0][field] != ("" if value is None else str(value)):
                raise ValueError(f"CUDA build CSV/JSON value mismatch: {csv_path}: {field}")
        matches.append((csv_path, data))
    if len(matches) != 1:
        raise ValueError(f"Expected one CUDA build record for {dataset}/{method}/{kind}/{nlist}/{m}, got {len(matches)}")
    return matches[0]


def save_tensorvs_build(index: Path, dataset: str, kind: str, *, build_seconds=None,
                        query_path=None, build_dir=None) -> Path:
    """Save original index metadata; never substitute query memory for build memory."""
    import torch
    index = Path(index)
    metadata = torch.load(index, map_location="cpu", mmap=True, weights_only=False)["metadata"]
    nlist = int(metadata["nlist"])
    m, bits = int(metadata.get("m", 0)), int(metadata.get("nbits", 0))
    stem = f"CUDA_TensorVS_Triton_{kind.upper()}_{dataset}_nlist{nlist}"
    if kind == "ivfpq":
        train = int(metadata.get("pq_max_points_per_centroid") or 256)
        stem += f"_M{m}_nbits{bits}_train{train}"
    if query_path is not None:
        query_stem = Path(query_path).stem
        query_stem = re.sub(r"^(?:CUDA_TensorVS_Triton_)?", "", query_stem)
        if query_stem.startswith(f"{kind.upper()}_{dataset}_nlist{nlist}"):
            stem = "CUDA_TensorVS_Triton_" + query_stem
    path = Path(build_dir or BUILD_DIR) / f"{stem}_build.csv"
    previous = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
    record = dict(dataset=dataset, method="TensorVS", nlist=nlist,
                  n_vectors=int(metadata["num_vectors"]), dim=int(metadata["dim"]),
                  build_seconds=build_seconds if build_seconds is not None else metadata.get("build_time_seconds"),
                  save_seconds=None, index_size_bytes=index.stat().st_size,
                  index_size_mib=index.stat().st_size / 2**20, peak_build_gpu_memory_mib=None)
    write_build(path, record, dict(previous, index_type=kind, M=m, nbits=bits,
                                 index_path=str(index.resolve()), build_device=metadata.get("device"),
                                 source_index_metadata=metadata, status="source_index_metadata",
                                 measurement_notes=["Original saved index build time; serialization time and build peak memory were not measured."]))
    return path
