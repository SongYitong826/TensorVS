"""Migrate legacy Apple benchmark CSVs to result/ and source index records to Build/."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import tempfile

try:
    from ._mps_results import ROOT, RESULT_DIR, QUERY_FIELDS, read_rows, rows_sha256, save_source_index, split_comparison, write_json, write_result
except ImportError:
    from _mps_results import ROOT, RESULT_DIR, QUERY_FIELDS, read_rows, rows_sha256, save_source_index, split_comparison, write_json, write_result


def migrate(source_dir: Path, result_dir: Path):
    sources = sorted(source_dir.glob("*.csv"))
    sources += sorted((source_dir / "mps_kernel").glob("*.csv"))
    sources += sorted((source_dir / "cpu_memory").glob("*.csv"))
    if not sources:
        raise FileNotFoundError(f"No legacy Apple result CSVs in {source_dir}")
    # Keep exact input bytes outside the repo as a rollback copy.
    backup = Path(tempfile.mkdtemp(prefix="tensorvs-mps-migration-"))
    mappings, indexes = [], {}
    for source in sources:
        rows = read_rows(source)
        relative = source.relative_to(source_dir)
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, saved)
        for row in rows:
            indexes[Path(row["index_path"])] = (row["dataset"], row.get("index_type", "ivfpq"))
        outputs = split_comparison(source, result_dir,
            archive=source.parent == source_dir,
            cpu_memory=source.parent.name == "cpu_memory")
        source_name = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
        mappings.append({"source": source_name, "rows": len(rows), "rows_sha256": rows_sha256(rows),
                         "outputs": [str(p.relative_to(result_dir)) for p in outputs]})
    build_outputs = [save_source_index(index, *config, result_dir) for index, config in indexes.items()]
    # No original is deleted until every measurement and source build record is saved.
    for source in sources:
        source.unlink()
    for directory in (source_dir / "mps_kernel", source_dir / "cpu_memory", source_dir):
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    manifest = {"status": "complete", "sources": mappings,
                "source_rows": sum(item["rows"] for item in mappings),
                "build_records": [str(p.relative_to(result_dir)) for p in build_outputs], "backup_dir": str(backup)}
    write_json(result_dir / "MPS_migration_manifest.json", manifest)
    print(f"Migrated {len(sources)} source CSVs ({manifest['source_rows']} rows), "
          f"wrote {sum(len(item['outputs']) for item in mappings)} result pairs and {len(build_outputs)} build pairs")
    print(f"Exact original CSV backup: {backup}")
    return manifest


def normalize(result_dir: Path):
    """Normalize existing MPS pairs without discarding original measurements."""
    sources = sorted(result_dir.glob("MPS_*.csv"))
    if not sources:
        raise FileNotFoundError(f"No Apple result CSVs in {result_dir}")
    backup = Path(tempfile.mkdtemp(prefix="tensorvs-mps-format-"))
    manifest_path = result_dir / "MPS_migration_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest_path.exists():
        shutil.copy2(manifest_path, backup / manifest_path.name)
    plan = []
    for source in sources:
        metadata = json.loads(source.with_suffix(".json").read_text())
        measurements = read_rows(source)
        name = re.sub(r"_QBatch\d+(?:_\d+)*", "", source.name)
        if source.stem.endswith("_cpu_memory"):
            directory = result_dir / "Build"
        elif metadata.get("status") != "complete":
            directory = result_dir / "Archive" / "incomplete"
        elif "_QBatch" in source.stem:
            directory = result_dir / "Archive"
        else:
            directory = result_dir
        destination = directory / name
        if destination != source and (destination.exists() or destination.with_suffix(".json").exists()):
            raise FileExistsError(destination)
        for path in (source, source.with_suffix(".json")):
            shutil.copy2(path, backup / path.name)
        plan.append((source, destination, metadata, measurements))
    mappings = {}
    for source, destination, metadata, measurements in plan:
        if destination != source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.with_suffix(".json"), destination.with_suffix(".json"))
        write_result(destination, measurements, complete=metadata.get("status") == "complete")
        if read_rows(destination) != measurements:
            raise ValueError(f"Measurements changed while normalizing {source}")
        if not source.stem.endswith("_cpu_memory"):
            with destination.open(newline="") as file:
                assert tuple(next(csv.reader(file))) == QUERY_FIELDS
        mappings[str(source.relative_to(result_dir))] = str(destination.relative_to(result_dir))
    for record in manifest.get("sources", []):
        record["outputs"] = [mappings.get(name, name) for name in record["outputs"]]
        recovered = [row for name in record["outputs"] for row in read_rows(result_dir / name)]
        if len(recovered) != record["rows"] or rows_sha256(recovered) != record["rows_sha256"]:
            raise ValueError(f"Original migration record changed: {record['source']}")
    manifest.update(format_version=2, query_csv_columns=list(QUERY_FIELDS))
    manifest.setdefault("format_updates", []).append({
        "backup_dir": str(backup), "paths": mappings,
        "rows": sum(len(measurements) for _, _, _, measurements in plan),
    })
    write_json(manifest_path, manifest)
    # Remove superseded paths only after every new pair and original row hash verifies.
    for source, destination, _, _ in plan:
        if destination != source:
            source.unlink()
            source.with_suffix(".json").unlink()
    print(f"Normalized {len(plan)} result pairs, preserving {manifest['format_updates'][-1]['rows']} rows")
    print(f"Exact pre-normalization backup: {backup}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--normalize", action="store_true", help="Normalize existing result/ MPS pairs")
    args = parser.parse_args()
    normalize(args.result_dir) if args.normalize else migrate(args.source_dir, args.result_dir)
