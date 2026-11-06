"""AMD-compatible query CSVs with full Apple measurements in JSON sidecars."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
METHOD_NAMES = {
    "TensorVS": "TensorVS_Tensor",
    "Faiss Metal": "FaissMetal",
    "TensorVS CPU": "TensorVS_CPU",
    "Faiss CPU": "FaissCPU",
}
QUERY_FIELDS = ("nprobes", "q_batch", "K", "qps", "recall",
                "visited_vectors_per_query", "peak_gpu_memory")


def index_stem(index: Path, dataset: str, kind: str, method: str) -> str:
    nlist = re.search(r"_nlist(\d+)", index.name, re.IGNORECASE)
    if nlist is None:
        raise ValueError(f"Missing nlist in index name: {index}")
    stem = f"MPS_{METHOD_NAMES[method]}_{kind.upper()}_{dataset}_nlist{nlist[1]}"
    if kind.lower() == "ivfpq":
        pq = re.search(r"_m(\d+)_nbits(\d+)_train(\d+)", index.name, re.IGNORECASE)
        if pq is None:
            raise ValueError(f"Missing PQ configuration in index name: {index}")
        stem += f"_M{pq[1]}_nbits{pq[2]}_train{pq[3]}"
    return stem


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        compact = tuple(reader.fieldnames or ()) == QUERY_FIELDS
    if not compact:
        return rows
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("csv_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"CSV/JSON checksum mismatch: {path}")
    records = metadata.get("records", [])
    if len(records) != len(rows):
        raise ValueError(f"CSV/JSON row count mismatch: {path}")
    records = [{key: "" if value is None else str(value) for key, value in record.items()}
               for record in records]
    for row, record in zip(rows, records):
        if any(row[field] != record.get(field, "") for field in QUERY_FIELDS):
            raise ValueError(f"CSV/JSON measurement mismatch: {path}")
    return records


def rows_sha256(rows: list[dict]) -> str:
    normalized = sorted(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
    return hashlib.sha256("\n".join(normalized).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_result(path: Path, rows: list[dict], *, complete=True, provenance=None) -> None:
    """Preserve every original field; metadata never invents missing measurements."""
    if not rows:
        raise ValueError(f"No measurements to save: {path}")
    record_fields = list(dict.fromkeys(key for row in rows for key in row))
    comparison = all(row.get("index_type") in ("ivfflat", "ivfpq") and "recall" in row
                     for row in rows)
    fields = list(QUERY_FIELDS) if comparison else record_fields
    previous = path.with_suffix(".json")
    previous_meta = json.loads(previous.read_text()) if previous.exists() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    config = {}
    for field in ("dataset", "index_type", "method", "execution_device", "nlist", "dim",
                  "M", "nbits", "pq_working_dim", "index_path", "index_size_bytes"):
        values = list(dict.fromkeys(str(row[field]) for row in rows if row.get(field) is not None))
        if len(values) == 1:
            config[field] = values[0]
    for field, label in (("q_batch", "q_batches"), ("nprobes", "nprobes"), ("K", "topk")):
        config[label] = sorted({int(row[field]) for row in rows})
    write_json(previous, {**previous_meta,
        "schema_version": 2,
        "status": "complete" if complete else "in_progress",
        "platform": "MPS", "host_family": "Apple Silicon",
        "completed_rows": len(rows), "config": config, "csv_columns": fields,
        "record_fields": record_fields, "records": rows,
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "provenance": provenance if provenance is not None else previous_meta.get("provenance"),
        "measurement_notes": [
            "execution_device records the actual device; Faiss Metal IVF-PQ uses CPU search.",
            "peak_gpu_memory, when recorded, is sampled Metal-driver allocation in MiB, not CUDA VRAM.",
            "CPU-memory experiments report RSS in MiB; missing metrics were not measured.",
            "Empty visited_vectors_per_query cells mean the original run did not measure this metric.",
            "Full per-point timing, device, RSS, and provenance fields are retained in records.",
        ],
    })


def split_comparison(source: Path, result_dir: Path, *, archive=False, cpu_memory=False):
    rows = read_rows(source)
    if not rows:
        raise ValueError(f"Empty comparison: {source}")
    groups = {}
    for row in rows:
        kind = row.get("index_type", "ivfpq")
        stem = index_stem(Path(row["index_path"]), row["dataset"], kind, row["method"])
        stem += "_cpu_memory" if cpu_memory else ""
        groups.setdefault(stem, []).append(row)
    outputs = []
    for stem, measurements in groups.items():
        directory = result_dir / "Build" if cpu_memory else result_dir / "Archive" if archive else result_dir
        output = directory / f"{stem}.csv"
        if output.exists():
            raise FileExistsError(output)
        write_result(output, measurements, provenance={
            "source_csv": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_rows": len(rows), "method_rows": len(measurements),
        })
        # Check round-trip identity before the migration may remove the source.
        if read_rows(output) != measurements:
            raise ValueError(f"Measurements changed while splitting {source}")
        outputs.append(output)
    return outputs


def comparison_paths(result_dir: Path, dataset: str, kind: str, *, archive=False) -> list[Path]:
    directory = result_dir / "Archive" if archive else result_dir
    paths = []
    for method in ("TensorVS_Tensor", "FaissMetal"):
        pattern = f"MPS_{method}_{kind.upper()}_{dataset}_nlist*.csv"
        candidates = []
        for path in sorted(directory.glob(pattern)):
            # Archives, incomplete runs, and memory checks stay in separate directories.
            tail = r"_M\d+_nbits\d+_train\d+" if kind.lower() == "ivfpq" else ""
            name = rf"MPS_{method}_{kind.upper()}_{re.escape(dataset)}_nlist\d+{tail}\.csv"
            if re.fullmatch(name, path.name):
                metadata = json.loads(path.with_suffix(".json").read_text())
                if metadata.get("status") != "complete":
                    continue
                candidates.append(path)
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous MPS result configuration: {candidates}")
        paths.extend(candidates)
    if not paths:
        raise FileNotFoundError(f"No complete MPS {dataset} {kind} results in {directory}")
    return paths


def comparison_rows(result_dir: Path, dataset: str, kind: str, *, archive=False) -> list[dict[str, str]]:
    rows = []
    for path in comparison_paths(result_dir, dataset, kind, archive=archive):
        measurements = read_rows(path)
        if any(row["dataset"] != dataset or row["index_type"] != kind for row in measurements):
            raise ValueError(f"Misidentified comparison rows: {path}")
        rows.extend(measurements)
    return rows


def save_source_index(index: Path, dataset: str, kind: str, result_dir: Path) -> Path:
    """Archive source build metadata, preserving its actual original build device."""
    import torch

    metadata = torch.load(index, map_location="cpu", mmap=True, weights_only=False)["metadata"]
    stem = index_stem(index, dataset, kind, "TensorVS") + "_build"
    output = result_dir / "Build" / f"{stem}.csv"
    record = {
        "dataset": dataset, "method": "TensorVS", "index_type": kind,
        "benchmark_platform": "MPS", "build_device": metadata.get("device", ""),
        "nlist": metadata["nlist"], "n_vectors": metadata["num_vectors"], "dim": metadata["dim"],
        "M": metadata.get("m", ""), "nbits": metadata.get("nbits", ""),
        "build_seconds": metadata.get("build_time_seconds", ""),
        "index_size_bytes": index.stat().st_size, "index_size_mib": index.stat().st_size / (1 << 20),
        "index_path": str(index.resolve()), "status": "source_index_metadata",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    write_json(output.with_suffix(".json"), {
        **record, "source_index_metadata": metadata,
        "timing": "Original saved index build time and device; no Apple build was timed during migration.",
        "faiss_index": "Faiss comparisons import this TensorVS index without retraining; no separate Faiss build measurement.",
    })
    return output
