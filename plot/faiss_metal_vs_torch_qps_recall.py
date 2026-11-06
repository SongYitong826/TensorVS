"""Plot the current Faiss Metal / TensorNative Torch SIFT1M Recall@10 curves.

Run: conda run -n tensorvs python plot/faiss_metal_vs_torch_qps_recall.py
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
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench._mps_results import comparison_paths, read_rows
from plot.qbatch_comparison import read_results

K = 10


def read_points(path, q_batch):
    with path.open(newline="") as file:
        points = [dict(nprobes=int(row["nprobes"]), recall=float(row["recall"]),
                       qps=float(row["qps"]))
                  for row in csv.DictReader(file)
                  if int(row["K"]) == K and int(row["q_batch"]) == q_batch]
    points.sort(key=lambda p: p["nprobes"])
    if not points or len({p["nprobes"] for p in points}) != len(points):
        raise ValueError(f"Missing or duplicate K=10/QBatch={q_batch} measurements: {path}")
    if not all(math.isfinite(p["qps"]) and p["qps"] > 0 and
               math.isfinite(p["recall"]) and 0 <= p["recall"] <= 1 for p in points):
        raise ValueError(f"Invalid plotted measurements: {path}")
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--q-batch", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plot/Figures")
    args = parser.parse_args()
    if args.q_batch <= 0:
        parser.error("q-batch must be positive")
    args.results_dir = args.results_dir or ROOT / "result"
    read_results(args.results_dir, "sift1m", "ivfflat")
    paths = comparison_paths(args.results_dir, "sift1m", "ivfflat", archive=True)
    specifications = (
        ("Faiss Metal",
         dict(color="#2474B5", marker="o", linestyle="-", markerfacecolor="#2474B5")),
        ("TensorVS Torch (TensorNative)",
         dict(color="#D66A28", marker="s", linestyle="--", markerfacecolor="white")),
    )
    series, sources, metadata = [], [], []
    for name, style in specifications:
        method = "Faiss Metal" if name == "Faiss Metal" else "TensorVS"
        path = next(path for path in paths if read_rows(path)[0]["method"] == method)
        points = read_points(path, args.q_batch)
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["status"] != "complete":
            raise ValueError(f"Incomplete benchmark: {path}")
        series.append((name, points, style))
        metadata.append(next(row for row in read_rows(path) if int(row["q_batch"]) == args.q_batch))
        sources.append(dict(method=name, csv=str(path.resolve()),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            metadata_sha256=hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest(),
                            points=points))
    faiss, torch = metadata
    assert faiss["execution_device"] == "metal" and torch["execution_device"] == "mps"
    assert int(faiss["query_count"]) == int(torch["query_count"]) == 1024
    assert faiss["warmup_scope"] == torch["warmup_scope"] == "all-batches"
    for field in ("index_path", "index_size_bytes", "index_mtime_ns"):
        assert faiss[field] == torch[field]
    probes = [[p["nprobes"] for p in points] for _, points, _ in series]
    assert probes[0] == probes[1] and len(probes[0]) == 14

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "savefig.facecolor": "white"})
    figure, axis = plt.subplots(figsize=(9.2, 6.2))
    figure.subplots_adjust(left=.115, right=.97, top=.82, bottom=.20)
    figure.suptitle("Faiss Metal vs. TensorVS Torch", fontsize=18, y=.96)
    figure.text(.5, .897, f"SIFT1M  |  IVF-Flat, nlist = 1024  |  Apple M5  |  K = 10  |  QBatch = {args.q_batch}",
                ha="center", fontsize=10.7, color="#555555")

    zoom = axis.inset_axes([.61, .52, .35, .27])
    for name, points, style in series:
        recalls, rates = [p["recall"] for p in points], [p["qps"] for p in points]
        axis.plot(recalls, rates, label=name, linewidth=2.1, markersize=5.7,
                  markeredgewidth=1.2, **style)
        zoom.plot(recalls, rates, linewidth=1.8, markersize=4.3,
                  markeredgewidth=1.0, **style)
    axis.set_xlabel("Recall@10", fontsize=13, labelpad=9)
    axis.set_ylabel("QPS", fontsize=13, labelpad=9)
    axis.set_xlim(.72, 1.008)
    maximum_qps = max(p["qps"] for _, points, _ in series for p in points)
    axis.set_ylim(0, max(30500, math.ceil(maximum_qps / 5000) * 5000 + 500))
    axis.xaxis.set_major_locator(MultipleLocator(.05))
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.legend(loc="upper right", frameon=False, fontsize=11, handlelength=2.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(True, linewidth=.65, linestyle="--", alpha=.30)
    zoom.set_title("High-recall detail", fontsize=10, pad=7)
    zoom.set_xlim(.973, 1.001)
    zoom_maximum = max(p["qps"] for _, points, _ in series for p in points if p["recall"] >= .973)
    zoom.set_ylim(0, max(6500, math.ceil(zoom_maximum / 1000) * 1000 + 500))
    zoom.set_xticks([.98, .99, 1.00])
    zoom.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    zoom.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    zoom.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    zoom.tick_params(labelsize=9)
    zoom.grid(True, linewidth=.55, linestyle="--", alpha=.30)
    for spine in zoom.spines.values():
        spine.set_edgecolor("#B8B8B8")
    figure.text(.115, .075, "14 measured nprobe settings (5–150) per method; median of saved sweeps over 1,024 queries.",
                fontsize=9.5, color="#555555")
    figure.text(.115, .035, "Faiss uses host input/output; Torch uses resident MPS input/output. No smoothing or point removal.",
                fontsize=9, color="#666666")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"SIFT1M_IVFFlat_FaissMetal_vs_Torch_QPSRecall_QBatch{args.q_batch}_K10"
    outputs = []
    for ext, options in (("png", {"dpi": 220}), ("pdf", {}), ("svg", {})):
        path = args.output_dir / f"{stem}.{ext}"
        figure.savefig(path, **options)
        outputs.append(str(path.resolve()))
    plt.close(figure)
    report = dict(K=K, q_batch=args.q_batch, x_axis="Recall@10", y_axis="QPS",
                  y_scale="linear", source_csvs=sources, points_per_method=14,
                  high_recall_inset=True, no_smoothing=True, no_point_removal=True,
                  warmup_scope=torch.get("warmup_scope", "first-batch"),
                  measurement_limit="Faiss includes host/device transfers; Torch uses resident MPS tensors.", outputs=outputs)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="complete", points_per_method=14, outputs=outputs), indent=2))


if __name__ == "__main__":
    main()
