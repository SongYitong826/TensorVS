"""Plot measured Triton/Torch QPS--Recall curves at QBatch=128.

Each figure fixes the index type and K, with one panel per dataset. All
measured nprobe values are retained, including equal/nonmonotonic recalls.

Run from the repository root:
    conda run -n tensorvs python plot/search_backends_qps_recall.py
    conda run -n tensorvs python plot/search_backends_qps_recall.py --k 10
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "result"
DEFAULT_OUTPUT = PROJECT_ROOT / "plot" / "Figures"
Q_BATCH = 128
DATASETS = {
    "sift1m": ("SIFT1M", 128),
    "glove1.2m": ("GloVe1.2M", 200),
    "deep1m": ("Deep1M", 256),
    "tiny5m": ("Tiny5M", 384),
    "gist": ("GIST1M", 960),
    "openai1536": ("OpenAI1536", 1536),
}
STYLES = {
    "Triton": dict(color="#2474B5", marker="o", linestyle="-", markerfacecolor="#2474B5"),
    "Torch": dict(color="#D66A28", marker="s", linestyle="--", markerfacecolor="white"),
}
NAMES = {"IVFFLAT": "IVF-Flat", "IVFPQ": "IVF-PQ"}


def read_results(path: Path) -> dict[tuple[int, int], dict[str, float | int]]:
    """Keep only QBatch=128 and reject duplicate or invalid measurements."""
    results = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"nprobes", "q_batch", "K", "qps", "recall"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing plot columns in {path}")
        for row in reader:
            if int(row["q_batch"]) != Q_BATCH:
                continue
            k, probe = int(row["K"]), int(row["nprobes"])
            qps, recall = float(row["qps"]), float(row["recall"])
            if k <= 0 or probe <= 0 or not math.isfinite(qps) or qps <= 0:
                raise ValueError(f"Invalid QPS or parameters in {path}: {row}")
            if not math.isfinite(recall) or not 0 <= recall <= 1:
                raise ValueError(f"Invalid Recall in {path}: {row}")
            if (k, probe) in results:
                raise ValueError(f"Duplicate K={k}, nprobe={probe} in {path}")
            results[k, probe] = dict(K=k, nprobe=probe, qps=qps, recall=recall)
    if not results:
        raise ValueError(f"No QBatch={Q_BATCH} rows in {path}")
    return results


def collect_groups(result_dir: Path, index_type: str, k_filter: int | None):
    groups, sources = {}, []
    for torch_path in sorted(result_dir.glob("torch_*.csv")):
        match = re.fullmatch(r"torch_(IVFFlat|IVFPQ)_(.+)_nlist(\d+)(.*)\.csv",
                             torch_path.name, re.IGNORECASE)
        if match is None:
            continue
        algorithm, dataset, nlist, suffix = match.groups()
        algorithm = algorithm.upper()
        if index_type != "both" and algorithm != index_type.upper():
            continue
        reference = result_dir / f"{algorithm}_{dataset}_nlist{nlist}{suffix}.csv"
        triton, torch = read_results(reference), read_results(torch_path)
        metadata_path = torch_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        for measured in metadata.get("results", []):
            key = measured.get("K"), measured.get("nprobe")
            samples = measured.get("samples_seconds", [])
            if key in torch and len(samples) >= 2:
                rates = [metadata["query_count"] / elapsed for elapsed in samples]
                quartiles = statistics.quantiles(rates, n=4, method="inclusive")
                torch[key]["qps_interval_low"], torch[key]["qps_interval_high"] = quartiles[0], quartiles[2]
        if triton.keys() != torch.keys():
            raise ValueError(f"Triton/Torch parameter sets differ: {reference}, {torch_path}")
        for k in sorted({key[0] for key in torch}):
            if k_filter is not None and k != k_filter:
                continue
            group = groups.setdefault((algorithm, k), {})
            if dataset in group:
                raise ValueError(f"Multiple index configurations for {dataset}, {algorithm}, K={k}")
            keys = sorted((key for key in torch if key[0] == k), key=lambda key: key[1])
            group[dataset] = {"Triton": [triton[key] for key in keys],
                              "Torch": [torch[key] for key in keys]}
        for method, path, rows in (("Triton", reference, triton), ("Torch", torch_path, torch)):
            source = dict(method=method, algorithm=algorithm, dataset=dataset,
                          path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          selected_rows=sum(k_filter is None or key[0] == k_filter for key in rows))
            metadata_path = path.with_suffix(".json")
            if method == "Torch" and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                source["measurement"] = {key: metadata.get(key) for key in
                                         ("flat_scan", "repetitions", "timing", "device", "cuda_visible_devices")}
                source["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            sources.append(source)
    if not groups:
        raise ValueError(f"No paired QBatch={Q_BATCH} results in {result_dir}")
    return groups, sources


def format_qps(value: float, _position: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def plot_group(algorithm, k, datasets, output_dir, torch_label="Torch (eager)"):
    order = sorted(datasets, key=lambda name: (DATASETS.get(name, (name, math.inf))[1], name))
    columns = min(3, len(order))
    rows = math.ceil(len(order) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.8 * columns, 3.35 * rows + 0.85),
                                squeeze=False)
    handles = {}
    plotted_points = 0
    for axis, dataset in zip(axes.flat, order):
        methods = datasets[dataset]
        for method, style in STYLES.items():
            points = methods[method]
            (line,) = axis.plot([p["recall"] for p in points], [p["qps"] for p in points],
                                linewidth=2.0, markersize=4.7, markeredgewidth=1.1,
                                markeredgecolor=style["color"], **style)
            handles.setdefault(method, line)
            plotted_points += len(points)
            if all("qps_interval_low" in point for point in points):
                axis.fill_between([p["recall"] for p in points],
                                  [p["qps_interval_low"] for p in points],
                                  [p["qps_interval_high"] for p in points],
                                  color=style["color"], alpha=0.12, linewidth=0)
        recalls = [p["recall"] for points in methods.values() for p in points]
        qps = [p["qps"] for points in methods.values() for p in points]
        qps.extend(p[key] for points in methods.values() for p in points
                   for key in ("qps_interval_low", "qps_interval_high") if key in p)
        margin = max(0.006, 0.08 * (max(recalls) - min(recalls)))
        axis.set_xlim(max(0, min(recalls) - margin), min(1.006, max(recalls) + margin))
        axis.set_ylim(min(qps) / 1.30, max(qps) * 1.30)
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        precision = 3 if max(recalls) - min(recalls) < 0.08 else 2
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _, digits=precision: f"{value:.{digits}f}"))
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=10))
        axis.yaxis.set_major_formatter(FuncFormatter(format_qps))
        axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="major", linestyle="--", linewidth=0.65, alpha=0.35)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10)
        axis.set_xlabel(f"Recall@{k}", fontsize=11)
        axis.set_ylabel("QPS (log scale)", fontsize=11)
        label, dimension = DATASETS.get(dataset, (dataset, None))
        axis.set_title(label if dimension is None else f"{label}  (D={dimension})",
                       fontsize=12, pad=10)
    for axis in list(axes.flat)[len(order):]:
        axis.set_visible(False)
    figure.suptitle(f"{NAMES[algorithm]}  |  QBatch = {Q_BATCH}  |  K = {k}",
                   fontsize=16, y=0.985)
    figure.legend([handles[method] for method in STYLES], ["Triton", torch_label],
                  loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2,
                  frameon=False, fontsize=12, handlelength=3, columnspacing=2.5)
    has_intervals = any("qps_interval_low" in point for dataset in datasets.values()
                        for points in dataset.values() for point in points)
    if has_intervals:
        figure.text(0.5, 0.01, "Torch shaded band: 25th–75th percentile of measured QPS",
                    ha="center", fontsize=9, color="#666666")
    figure.tight_layout(rect=(0.005, 0.04 if has_intervals else 0.005, 0.995, 0.89), h_pad=2.2, w_pad=2.5)
    stem = f"{'IVFFlat' if algorithm == 'IVFFLAT' else 'IVFPQ'}_Triton_vs_Torch_QPSRecall_QBatch{Q_BATCH}_K{k}"
    outputs = []
    for extension, options in (("pdf", {}), ("png", {"dpi": 180})):
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, bbox_inches="tight", **options)
        outputs.append(str(path))
    plt.close(figure)
    return dict(algorithm=algorithm, K=k, datasets=order, panels=len(order),
                plotted_points=plotted_points, outputs=outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index-type", choices=("both", "ivfflat", "ivfpq"), default="both")
    parser.add_argument("--k", type=int, help="Only plot this K; default: all available K values")
    args = parser.parse_args()
    groups, sources = collect_groups(args.result_dir, args.index_type, args.k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.titleweight": "normal", "savefig.facecolor": "white"})
    reports = []
    for (algorithm, k), datasets in sorted(groups.items()):
        measurements = [source.get("measurement", {}) for source in sources
                        if source["method"] == "Torch" and source["algorithm"] == algorithm]
        grouped = algorithm == "IVFFLAT" and all(m.get("flat_scan") == "grouped" for m in measurements)
        repetitions = {m.get("repetitions") for m in measurements}
        label = "Torch (eager, IVF lists)" if grouped else "Torch (eager)"
        if len(repetitions) == 1 and None not in repetitions:
            label += f", median (at least {repetitions.pop()} runs)"
        report = plot_group(algorithm, k, datasets, args.output_dir, label)
        reports.append(report)
        print(f"Saved {algorithm}, K={k}: {report['panels']} panels, "
              f"{report['plotted_points']} measured points; PDF + PNG", flush=True)
    manifest = dict(q_batch=Q_BATCH, x_axis="Recall@K", y_axis="QPS (log scale)",
                    curve_order="Increasing measured nprobe; no interpolation or point filtering",
                    torch_uncertainty="25th–75th percentile of measured sweep QPS; all samples retained",
                    figures=reports, sources=sources)
    (args.output_dir / f"Triton_vs_Torch_QPSRecall_QBatch{Q_BATCH}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
