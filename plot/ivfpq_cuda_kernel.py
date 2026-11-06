"""Plot the 3x6 IVFPQ CUDA-kernel QPS/Recall figure.

Rows are QBatch=32/128/512 and columns are the six paper datasets.  Each
dataset uses one Recall range shared across all three rows; this is necessary
for IVFPQ because its Recall ranges differ substantially by dataset.

Run from the TensorVS repository root with::

    conda run -n tensorvs python plot/ivfpq_cuda_kernel.py --require-complete
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plot" / "Figures"
QBATCHES = (32, 128, 512)


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str


@dataclass(frozen=True)
class Method:
    label: str
    color: str
    marker: str
    linestyle: str


@dataclass(frozen=True)
class ResultFile:
    path: Path
    filename_topk: int | None


DATASETS = (
    Dataset("sift10m", "SIFT"),
    Dataset("glove1.2m", "GloVe"),
    Dataset("tiny5m", "Tiny"),
    Dataset("hunyuan", "Hunyuan"),
    Dataset("gist", "GIST"),
    Dataset("openai1536", "OpenAI"),
)

METHODS = {
    "tensorvs": Method("TensorVS", "#1f77b4", "o", "-"),
    "cuvs": Method("cuVS", "#ff7f0e", "s", "--"),
    "faiss": Method("Faiss GPU", "#2ca02c", "^", "-."),
}
LEGEND_ORDER = ("cuvs", "faiss", "tensorvs")

DEFAULT_RESULT_DIRS = {
    "tensorvs": (PROJECT_ROOT / "result", PROJECT_ROOT / "results"),
    "cuvs": (
        Path("/home/songyitong/cuvs/python/cuvs/cuvs/result"),
        Path("/home/songyitong/cuvs/python/cuvs/cuvs/results"),
    ),
    "faiss": (
        Path("/home/songyitong/faiss/result"),
        Path("/home/songyitong/faiss/results"),
    ),
}

FILENAME_PATTERN = re.compile(
    r"^(?:TensorVS_Tensor_)?IVFPQ_(?P<dataset>.+?)_nlist\d+.*?(?:_K(?P<topk>\d+))?\.csv$",
    re.IGNORECASE,
)


def _selected_dirs(
    supplied: list[Path] | None, defaults: tuple[Path, ...]
) -> tuple[Path, ...]:
    return tuple(supplied) if supplied else defaults


def discover_result_files(
    result_dirs: dict[str, tuple[Path, ...]],
) -> tuple[dict[str, dict[str, ResultFile]], list[str]]:
    requested = {dataset.key.casefold() for dataset in DATASETS}
    discovered: dict[str, dict[str, ResultFile]] = {method: {} for method in METHODS}
    warnings: list[str] = []
    for method in METHODS:
        for result_dir in result_dirs[method]:
            if not result_dir.is_dir():
                continue
            for path in sorted(result_dir.glob("*.csv")):
                match = FILENAME_PATTERN.match(path.name)
                if match is None:
                    continue
                dataset = match.group("dataset").casefold()
                if dataset not in requested:
                    continue
                if dataset in discovered[method]:
                    warnings.append(
                        f"Ignoring duplicate {METHODS[method].label}/{dataset}: {path}"
                    )
                    continue
                encoded_topk = match.group("topk")
                discovered[method][dataset] = ResultFile(
                    path=path,
                    filename_topk=int(encoded_topk) if encoded_topk else None,
                )
    return discovered, warnings


def _column_name(fieldnames: list[str], *accepted: str) -> str | None:
    by_casefold = {name.casefold(): name for name in fieldnames}
    return next(
        (by_casefold[name.casefold()] for name in accepted if name.casefold() in by_casefold),
        None,
    )


def read_result_rows(result_file: ResultFile) -> list[dict[str, float]]:
    with result_file.path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        q_batch_column = _column_name(fieldnames, "q_batch", "qbatch")
        topk_column = _column_name(fieldnames, "K", "topk")
        nprobe_column = _column_name(fieldnames, "nprobes", "nprobe")
        qps_column = _column_name(fieldnames, "qps")
        recall_column = _column_name(fieldnames, "recall")
        missing = [
            label
            for label, column in (
                ("q_batch", q_batch_column),
                ("nprobes", nprobe_column),
                ("qps", qps_column),
                ("recall", recall_column),
            )
            if column is None
        ]
        if topk_column is None and result_file.filename_topk is None:
            missing.append("K")
        if missing:
            raise ValueError(
                f"{result_file.path} is missing columns: {', '.join(missing)}"
            )

        rows: list[dict[str, float]] = []
        for csv_row in reader:
            try:
                row = {
                    "q_batch": float(csv_row[q_batch_column]),
                    "topk": (
                        float(csv_row[topk_column])
                        if topk_column is not None
                        else float(result_file.filename_topk)
                    ),
                    "nprobe": float(csv_row[nprobe_column]),
                    "qps": float(csv_row[qps_column]),
                    "recall": float(csv_row[recall_column]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if (
                all(math.isfinite(value) for value in row.values())
                and row["qps"] > 0
                and 0.0 <= row["recall"] <= 1.0
            ):
                rows.append(row)
    return rows


def collect_curves(
    result_files: dict[str, dict[str, ResultFile]],
    q_batches: tuple[int, ...],
    topk: int,
) -> tuple[
    dict[str, dict[int, dict[str, list[dict[str, float]]]]],
    list[str],
]:
    curves = {
        dataset.key: {q_batch: {} for q_batch in q_batches} for dataset in DATASETS
    }
    warnings: list[str] = []
    for method_key, method in METHODS.items():
        for dataset in DATASETS:
            result_file = result_files[method_key].get(dataset.key)
            if result_file is None:
                warnings.append(f"{method.label}/{dataset.label}: file missing")
                continue
            try:
                rows = read_result_rows(result_file)
            except (OSError, ValueError) as error:
                warnings.append(str(error))
                continue
            for q_batch in q_batches:
                selected = [
                    row
                    for row in rows
                    if int(row["q_batch"]) == q_batch and int(row["topk"]) == topk
                ]
                selected.sort(key=lambda row: (row["recall"], -row["qps"]))
                if selected:
                    curves[dataset.key][q_batch][method_key] = selected
                else:
                    warnings.append(
                        f"{method.label}/{dataset.label}: no QBatch={q_batch}, K={topk} rows"
                    )
    return curves, warnings


def _all_dataset_values(
    dataset_curves: dict[int, dict[str, list[dict[str, float]]]], field: str
) -> list[float]:
    return [
        point[field]
        for method_curves in dataset_curves.values()
        for curve in method_curves.values()
        for point in curve
    ]


def _recall_limits(
    dataset_curves: dict[int, dict[str, list[dict[str, float]]]],
) -> tuple[float, float] | None:
    values = _all_dataset_values(dataset_curves, "recall")
    if not values:
        return None
    low, high = min(values), max(values)
    span = max(high - low, 0.01)
    margin = max(0.005, span * 0.06)
    lower = max(0.0, math.floor((low - margin) * 100.0) / 100.0)
    upper = min(1.0, math.ceil((high + margin) * 100.0) / 100.0)
    if upper - lower < 0.02:
        lower = max(0.0, lower - 0.01)
        upper = min(1.0, upper + 0.01)
    return lower, upper


def _qps_limits(
    dataset_curves: dict[int, dict[str, list[dict[str, float]]]],
) -> tuple[float, float] | None:
    values = _all_dataset_values(dataset_curves, "qps")
    if not values:
        return None
    low, high = min(values), max(values)
    return low / 1.25, high * 1.25


def plot_figure(
    curves: dict[str, dict[int, dict[str, list[dict[str, float]]]]],
    q_batches: tuple[int, ...],
    topk: int,
    output_dir: Path,
    output_stem: str,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        len(q_batches), len(DATASETS), figsize=(18.0, 8.0), squeeze=False
    )
    legend_handles: dict[str, object] = {}

    for column, dataset in enumerate(DATASETS):
        recall_limits = _recall_limits(curves[dataset.key])
        qps_limits = _qps_limits(curves[dataset.key])
        for row_index, q_batch in enumerate(q_batches):
            axis = axes[row_index][column]
            method_curves = curves[dataset.key][q_batch]
            for method_key, method in METHODS.items():
                curve = method_curves.get(method_key)
                if not curve:
                    continue
                (line,) = axis.plot(
                    [point["recall"] for point in curve],
                    [point["qps"] for point in curve],
                    color=method.color,
                    marker=method.marker,
                    linestyle=method.linestyle,
                    linewidth=2.0,
                    markersize=4.5,
                    markeredgewidth=0.6,
                    label=method.label,
                )
                legend_handles.setdefault(method_key, line)

            if recall_limits is not None:
                axis.set_xlim(*recall_limits)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            axis.yaxis.set_minor_locator(
                LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12)
            )
            axis.yaxis.set_minor_formatter(NullFormatter())
            if qps_limits is not None:
                axis.set_ylim(*qps_limits)
            axis.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
            axis.tick_params(axis="both", labelsize=12.5)
            axis.set_xlabel("Recall", fontsize=14)
            axis.set_ylabel("QPS" if column == 0 else "", fontsize=14)
            axis.text(
                0.04,
                0.06,
                dataset.label,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=13.5,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
            axis.text(
                0.96,
                0.94,
                f"QBatch = {q_batch}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=12.5,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
            if not method_curves:
                axis.text(
                    0.5,
                    0.5,
                    "Results pending",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="#777777",
                )

    legend_keys = [key for key in LEGEND_ORDER if key in legend_handles]
    if legend_keys:
        figure.legend(
            [legend_handles[key] for key in legend_keys],
            [METHODS[key].label for key in legend_keys],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(legend_keys),
            frameon=True,
            fontsize=16,
            handlelength=3.0,
            columnspacing=2.2,
        )
    figure.subplots_adjust(
        left=0.052,
        right=0.995,
        bottom=0.09,
        top=0.91,
        wspace=0.27,
        hspace=0.32,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{output_stem}.pdf"
    png_path = output_dir / f"{output_stem}.png"
    figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the 3x6 IVFPQ CUDA-kernel QPS/Recall figure."
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--q-batches", type=int, nargs="+", default=list(QBATCHES))
    parser.add_argument("--tensorvs-dir", action="append", type=Path)
    parser.add_argument("--cuvs-dir", action="append", type=Path)
    parser.add_argument("--faiss-dir", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_batches = tuple(args.q_batches)
    result_dirs = {
        "tensorvs": _selected_dirs(args.tensorvs_dir, DEFAULT_RESULT_DIRS["tensorvs"]),
        "cuvs": _selected_dirs(args.cuvs_dir, DEFAULT_RESULT_DIRS["cuvs"]),
        "faiss": _selected_dirs(args.faiss_dir, DEFAULT_RESULT_DIRS["faiss"]),
    }
    result_files, discovery_warnings = discover_result_files(result_dirs)
    curves, curve_warnings = collect_curves(result_files, q_batches, args.k)
    warnings = discovery_warnings + curve_warnings
    if warnings:
        print("Incomplete result entries:")
        for warning in warnings:
            print(f"  - {warning}")
    if args.require_complete and warnings:
        raise RuntimeError(f"{len(warnings)} requested result entries are incomplete")

    output_stem = args.output_stem or (
        f"IVFPQ_CUDA_QBatch{'_'.join(map(str, q_batches))}_K{args.k}"
    )
    pdf_path, png_path = plot_figure(
        curves,
        q_batches=q_batches,
        topk=args.k,
        output_dir=args.output_dir,
        output_stem=output_stem,
    )
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
