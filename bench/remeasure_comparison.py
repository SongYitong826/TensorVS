"""Remeasure timing outliers and neighbors without editing performance values.

Uses seven samples of >=0.4 timed seconds each, with alternating config order.
Only the four final CSVs are retained. Selection plans/exports are temporary.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from compare_mps_faiss import write_csv

ROOT = Path(__file__).resolve().parents[1]


def select_points(rows):
    selected = {}
    for method in ("Faiss Metal", "TensorVS"):
        for batch in (128, 256, 512):
            curve = sorted((r for r in rows if r["method"] == method and int(r["q_batch"]) == batch),
                           key=lambda row: int(row["nprobes"]))
            flagged = set()
            for i, row in enumerate(curve):
                samples = [float(row[f"sample{j}_seconds"]) for j in range(1, int(row["repetitions"]) + 1)]
                if (max(samples) - min(samples)) / statistics.median(samples) > .12:
                    flagged.add(i)
            for i in range(1, len(curve)):
                if float(curve[i]["qps"]) > float(curve[i - 1]["qps"]) * 1.025:
                    flagged.update((i - 1, i))
            for i in range(1, len(curve) - 1):
                a, b, c = curve[i - 1:i + 2]
                fraction = (int(b["nprobes"]) - int(a["nprobes"])) / (int(c["nprobes"]) - int(a["nprobes"]))
                neighbor_rate = math.exp(math.log(float(a["qps"])) * (1 - fraction) + math.log(float(c["qps"])) * fraction)
                if float(b["qps"]) < neighbor_rate * .9:
                    flagged.add(i)
            neighbors = {j for i in flagged for j in (i - 1, i, i + 1) if 0 <= j < len(curve)}
            for i in neighbors:
                selected.setdefault(method, set()).add((batch, int(curve[i]["nprobes"])))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()
    for path in sorted((ROOT / "results").glob("*.csv")):
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        dataset, kind = rows[0]["dataset"], rows[0]["index_type"]
        if args.datasets and dataset not in args.datasets:
            continue
        # Support the dataset's canonical filenames after a prefix rename.
        migrated = False
        for field, suffix in (("query_path", "query.fvecs"), ("groundtruth_path", "groundtruth.ivecs")):
            stored = Path(rows[0][field])
            if not stored.is_file():
                candidate = stored.parent / f"{dataset}_{suffix}"
                if not candidate.is_file():
                    raise FileNotFoundError(stored)
                for row in rows:
                    row[field] = str(candidate.resolve())
                migrated = True
        if migrated:
            write_csv(path, rows)
        selections = select_points(rows)
        print(f"{path.name}: " + ", ".join(f"{method} {len(points)} points" for method, points in selections.items()), flush=True)
        with tempfile.TemporaryDirectory(prefix="tensorvs-retime-") as directory:
            folder = Path(directory)
            index = Path(rows[0]["index_path"])
            if "Faiss Metal" in selections:
                subprocess.run([sys.executable, str(ROOT / "bench/export_comparison_index.py"),
                                kind, str(index), str(folder / "index")], check=True)
            for method, points in selections.items():
                plan = folder / "points.csv"
                with plan.open("w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(("q_batch", "nprobes"))
                    writer.writerows(sorted(points))
                subprocess.run([sys.executable, "-u", str(ROOT / "bench/compare_mps_faiss.py"),
                                "--dataset", str(index.parent.parent), "--kind", kind,
                                "--worker", "faiss" if method == "Faiss Metal" else "tensorvs",
                                "--index", str(index), "--queries", rows[0]["query_path"],
                                "--groundtruth", rows[0]["groundtruth_path"],
                                "--nq", rows[0]["query_count"], "--output", str(path),
                                "--export-dir", str(folder / "index"), "--remeasure-points", str(plan),
                                "--repetitions", "7", "--min-sample-seconds", ".4"], check=True)


if __name__ == "__main__":
    main()
