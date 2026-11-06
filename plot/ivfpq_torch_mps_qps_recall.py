"""Plot measured TensorNative IVF-PQ GPU curves for QBatch=256 and 512.

Faiss 1.15.1's Metal IVF-PQ search delegates to CPU, so no Faiss GPU curve
is fabricated. Run: conda run -n tensorvs python plot/ivfpq_torch_mps_qps_recall.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench._mps_results import comparison_paths, read_rows
from plot.qbatch_comparison import read_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plot/Figures")
    args = parser.parse_args()
    read_results(args.results_dir, "sift1m", "ivfpq")
    paths = comparison_paths(args.results_dir, "sift1m", "ivfpq", archive=True)
    path = next(path for path in paths if read_rows(path)[0]["method"] == "TensorVS")
    sources = []
    figure, axis = plt.subplots(figsize=(9.2, 6.2))
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    styles = {256: dict(color="#2474B5", marker="o", linestyle="-"),
              512: dict(color="#D66A28", marker="s", linestyle="--", markerfacecolor="white")}
    maxima, minima, query_counts, configurations = [], [], set(), []
    for batch, style in styles.items():
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["status"] == "complete"
        selected = [row for row in read_rows(path) if int(row["q_batch"]) == batch]
        assert all(row["execution_device"] == "mps" for row in selected)
        with path.open(newline="") as file:
            points = [dict(nprobes=int(r["nprobes"]), recall=float(r["recall"]), qps=float(r["qps"]))
                      for r in csv.DictReader(file) if int(r["K"]) == 10 and int(r["q_batch"]) == batch]
        points.sort(key=lambda p: p["nprobes"])
        assert len(points) == len({p["nprobes"] for p in points}) == 14
        assert all(math.isfinite(p["qps"]) and p["qps"] > 0 and
                   math.isfinite(p["recall"]) and 0 <= p["recall"] <= 1 for p in points)
        configurations.append([p["nprobes"] for p in points])
        query_counts.update(int(row["query_count"]) for row in selected)
        axis.plot([p["recall"] for p in points], [p["qps"] for p in points],
                  label=f"Torch MPS — QBatch = {batch}", linewidth=2.1,
                  markersize=5.7, markeredgewidth=1.2, **style)
        maxima.extend(p["qps"] for p in points)
        minima.extend(p["recall"] for p in points)
        sources.append(dict(q_batch=batch, csv=str(path.resolve()),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), points=points))
    assert configurations[0] == configurations[1] and query_counts == {1024}
    figure.subplots_adjust(left=.115, right=.97, top=.82, bottom=.20)
    figure.suptitle("IVF-PQ: TensorVS Torch GPU", fontsize=18, y=.96)
    figure.text(.5, .897, "SIFT1M  |  nlist = 1024, M = 16, nbits = 8  |  Apple M5  |  K = 10",
                ha="center", fontsize=10.7, color="#555555")
    axis.set_xlabel("Recall@10", fontsize=13, labelpad=9)
    axis.set_ylabel("QPS", fontsize=13, labelpad=9)
    axis.set_xlim(min(minima) - .008, max(minima) + .008)
    axis.set_ylim(0, max(maxima) * 1.12)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=7))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.tick_params(labelsize=11)
    axis.legend(loc="upper right", frameon=False, fontsize=11)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(True, linestyle="--", linewidth=.65, alpha=.30)
    figure.text(.115, .075, "14 measured nprobe settings per curve; full-query warmup; median of saved sweeps over 1,024 queries.",
                fontsize=9, color="#555555")
    figure.text(.115, .035, "Faiss Metal 1.15.1 IVF-PQ search delegates to CPU; no Faiss GPU curve is available.",
                fontsize=9, color="#666666")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "SIFT1M_IVFPQ_TorchMPS_QPSRecall_QBatch256_512_K10"
    outputs = []
    for ext, options in (("png", {"dpi": 220}), ("pdf", {}), ("svg", {})):
        target = args.output_dir / f"{stem}.{ext}"
        figure.savefig(target, **options)
        outputs.append(str(target.resolve()))
    plt.close(figure)
    report = dict(K=10, x_axis="Recall@10", y_axis="QPS", y_scale="linear",
                  method="TensorVS Torch / MPS / TensorNative", sources=sources,
                  points_per_curve=14, warmup_scope="all-batches", no_smoothing=True,
                  faiss_gpu_search_available=False, outputs=outputs)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="complete", outputs=outputs), indent=2))


if __name__ == "__main__":
    main()
