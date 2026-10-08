"""Plot IVFPQ QPS/recall curves for TensorVS, cuVS, and Faiss.

The script scans the TensorVS, cuVS, and Faiss result directories for
``IVFPQ_*.csv`` files.
Each distinct ``(QBatch, K)`` combination is written to a separate figure,
while the datasets available for that combination are arranged in one row.
When run without arguments, only K=10 figures are generated.

Run from the project root with::

    conda run -n tensorvs python plot/ivfpq_plot.py

For example, a result figure is written to
``plot/Figures/IVFPQ_QBatch128_K10.pdf``.
"""

# python ivfpq_plot.py 全部q-batch的K=10图
# python ivfpq_plot.py --q-batch 128
# python ivfpq_plot.py --q-batch 128 --k 20

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plot" / "Figures"

RESULT_DIRS = {
    "TensorVS": PROJECT_ROOT / "result",
    "cuVS": Path("/home/songyitong/cuvs/python/cuvs/cuvs/result"),
    "Faiss": Path("/home/songyitong/faiss/result"),
}

STYLES = {
    # Keep TensorVS above the other methods where the curves overlap.
    "TensorVS": {"color": "#1f77b4", "marker": "o", "zorder": 3},
    "cuVS": {"color": "#ff7f0e", "marker": "s", "zorder": 2},
    "Faiss": {"color": "#2ca02c", "marker": "^", "zorder": 2},
}

# Add a small margin around each dataset's observed Recall range so markers at
# the extrema do not sit directly on the axes borders.
RECALL_AXIS_MARGIN_FRACTION = 0.05
RECALL_AXIS_MIN_MARGIN = 0.01

# Keep the figure wide enough for a row of dataset panels, but relatively
# short so it fits compactly into a paper column or page layout.
FIGURE_WIDTH_PER_DATASET = 5.2
FIGURE_HEIGHT = 3.6
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 11
DATASET_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 13
LINE_WIDTH = 2.4
MARKER_SIZE = 8

# Dimensions of the datasets used by the IVF-PQ benchmarks.  The result CSV
# files contain performance data but do not store the vector dimension, so the
# plotting order is defined from the benchmark dataset names here.
DATASET_DIMENSIONS = {
    "deep1m": 256,
    "sift1m": 128,
    "glove1.2m": 200,
    "tiny5m": 384,
    "gist": 960,
    "openai1536": 1536,
}


def dataset_sort_key(dataset: str) -> tuple[float, str]:
    """Sort known datasets by vector dimension, then unknown names naturally."""
    dimension = DATASET_DIMENSIONS.get(dataset.casefold())
    return (dimension if dimension is not None else math.inf, dataset.casefold())


def dataset_display_name(dataset: str) -> str:
    """Return the label shown above a dataset subplot."""
    if dataset.casefold() == "deep1m":
        return "Deep"
    if dataset.casefold() == "glove1.2m":
        return "GloVe"
    if dataset.casefold() == "tiny5m":
        return "Tiny5M"
    if dataset.casefold() == "openai1536":
        return "OpenAI"
    return dataset.upper()


@dataclass(frozen=True)
class ResultFile:
    """One IVFPQ result file and the metadata encoded in its filename."""

    method: str
    dataset: str
    topk: int | None
    path: Path


def result_metadata_from_filename(path: Path) -> tuple[str, int | None] | None:
    """Extract dataset and optional legacy K metadata from a result filename."""

    prefix = "IVFPQ_"
    if not path.stem.upper().startswith(prefix):
        return None

    dataset_and_config = path.stem[len(prefix) :]
    topk_match = re.search(r"_K(?P<topk>\d+)$", dataset_and_config, re.IGNORECASE)
    if topk_match is None:
        dataset_and_nlist = dataset_and_config
        topk = None
    else:
        dataset_and_nlist = dataset_and_config[: topk_match.start()]
        topk = int(topk_match.group("topk"))
    dataset = dataset_and_nlist.split("_nlist", maxsplit=1)[0]
    if not dataset:
        return None
    return dataset, topk


def discover_result_files() -> list[ResultFile]:
    """Find and de-duplicate all available IVFPQ result files."""

    discovered: dict[tuple[str, str, int | None], ResultFile] = {}
    for method, result_dir in RESULT_DIRS.items():
        if not result_dir.is_dir():
            continue

        for path in sorted(result_dir.glob("*.csv")):
            metadata = result_metadata_from_filename(path)
            if metadata is None:
                continue
            dataset, topk = metadata
            key = (method, dataset, topk)
            if key in discovered:
                print(
                    f"Warning: multiple {method} files for {dataset}, K={topk}; "
                    f"using {discovered[key].path}"
                )
                continue
            discovered[key] = ResultFile(method, dataset, topk, path)
    return list(discovered.values())


def read_results(path: Path, filename_topk: int | None) -> list[dict[str, float]]:
    """Read all valid rows from one benchmark CSV file."""

    if not path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"nprobes", "q_batch", "qps", "recall"}
        actual_columns = set(reader.fieldnames or ())
        missing = required_columns - actual_columns
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing columns: {missing_columns}")

        k_column = next(
            (column for column in (reader.fieldnames or ()) if column.casefold() == "k"),
            None,
        )
        has_k_column = k_column is not None
        if not has_k_column and filename_topk is None:
            raise ValueError(f"{path} is missing the K column")

        rows = []
        for row in reader:
            rows.append(
                {
                    "nprobes": float(row["nprobes"]),
                    "q_batch": float(row["q_batch"]),
                    "K": float(row[k_column]) if has_k_column else float(filename_topk),
                    "qps": float(row["qps"]),
                    "recall": float(row["recall"]),
                }
            )

    rows = [row for row in rows if row["qps"] > 0]
    rows.sort(key=lambda row: row["recall"])
    return rows


def adaptive_recall_bounds(
    rows: list[dict[str, float]],
) -> tuple[float, float] | None:
    """Return an axis range derived from the observed Recall values."""
    recalls = [
        row["recall"]
        for row in rows
        if math.isfinite(row["recall"]) and 0.0 <= row["recall"] <= 1.0
    ]
    if not recalls:
        return None

    observed_min = min(recalls)
    observed_max = max(recalls)
    observed_span = observed_max - observed_min
    margin = max(
        observed_span * RECALL_AXIS_MARGIN_FRACTION,
        RECALL_AXIS_MIN_MARGIN,
    )
    lower = max(0.0, observed_min - margin)
    upper = min(1.0, observed_max + margin)
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12):
        lower = max(0.0, observed_min - RECALL_AXIS_MIN_MARGIN)
        upper = min(1.0, observed_max + RECALL_AXIS_MIN_MARGIN)
    return lower, upper


def select_recall_range(
    rows: list[dict[str, float]],
    recall_min: float,
    recall_max: float,
) -> list[dict[str, float]]:
    """Keep rows inside one dataset's adaptive Recall axis range."""
    selected_rows = [
        row
        for row in rows
        if recall_min <= row["recall"] <= recall_max
    ]
    selected_rows.sort(key=lambda row: row["recall"])
    return selected_rows


def collect_parameter_groups(
    result_files: list[ResultFile],
    q_batch_filter: int | None = None,
    topk_filter: int | None = None,
) -> dict[tuple[int, int], dict[str, dict[str, list[dict[str, float]]]]]:
    """Group rows by ``(QBatch, K)`` and then by dataset and method."""

    groups: dict[
        tuple[int, int], dict[str, dict[str, list[dict[str, float]]]]
    ] = {}
    for result_file in result_files:
        try:
            rows = read_results(result_file.path, result_file.topk)
        except (OSError, ValueError) as error:
            print(f"Warning: skipping {result_file.path}: {error}")
            continue

        q_batches = sorted({int(row["q_batch"]) for row in rows})
        for q_batch in q_batches:
            if q_batch_filter is not None and q_batch != q_batch_filter:
                continue
            q_batch_rows = [row for row in rows if int(row["q_batch"]) == q_batch]
            for topk in sorted({int(row["K"]) for row in q_batch_rows}):
                if topk_filter is not None and topk != topk_filter:
                    continue
                parameter_group = groups.setdefault((q_batch, topk), {})
                dataset_group = parameter_group.setdefault(result_file.dataset, {})
                if result_file.method in dataset_group:
                    print(
                        f"Warning: duplicate {result_file.method}/{result_file.dataset}/"
                        f"QBatch{q_batch}/K{topk}; keeping the first file"
                    )
                    continue
                dataset_group[result_file.method] = [
                    row for row in q_batch_rows if int(row["K"]) == topk
                ]
    return groups


def plot_parameter_group(
    q_batch: int,
    topk: int,
    grouped_results: dict[str, dict[str, list[dict[str, float]]]],
    output_dir: Path,
) -> Path | None:
    """Plot one ``(QBatch, K)`` group as one row of dataset subplots."""

    dataset_results: dict[str, dict[str, list[dict[str, float]]]] = {}
    dataset_recall_bounds: dict[str, tuple[float, float]] = {}
    for dataset, method_rows in grouped_results.items():
        all_rows = [row for rows in method_rows.values() for row in rows]
        recall_bounds = adaptive_recall_bounds(all_rows)
        if recall_bounds is None:
            continue

        valid_method_rows: dict[str, list[dict[str, float]]] = {}
        for method, rows in method_rows.items():
            selected_rows = select_recall_range(rows, *recall_bounds)
            if selected_rows:
                valid_method_rows[method] = selected_rows
        if valid_method_rows:
            dataset_results[dataset] = valid_method_rows
            dataset_recall_bounds[dataset] = recall_bounds

    if not dataset_results:
        return None

    datasets = sorted(dataset_results, key=dataset_sort_key)
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(FIGURE_WIDTH_PER_DATASET * len(datasets), FIGURE_HEIGHT),
        squeeze=False,
    )
    axes = axes[0]
    legend_handles: dict[str, object] = {}

    for axis, dataset in zip(axes, datasets):
        for method in RESULT_DIRS:
            rows = dataset_results[dataset].get(method)
            if rows is None:
                continue
            style = STYLES[method]
            (line,) = axis.plot(
                [row["recall"] for row in rows],
                [row["qps"] for row in rows],
                label=method,
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                markeredgewidth=0.8,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                **style,
            )
            legend_handles.setdefault(method, line)

        axis.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE)
        axis.set_ylabel("QPS", fontsize=AXIS_LABEL_FONTSIZE)
        axis.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        axis.set_xlim(*dataset_recall_bounds[dataset])
        axis.set_yscale("log")
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.text(
            0.02,
            0.03,
            dataset_display_name(dataset),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=DATASET_LABEL_FONTSIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    legend_methods = [method for method in RESULT_DIRS if method in legend_handles]
    figure.legend(
        [legend_handles[method] for method in legend_methods],
        legend_methods,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        borderaxespad=0.2,
        ncol=len(legend_methods),
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.4,
        columnspacing=1.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"IVFPQ_QBatch{q_batch}_K{topk}.pdf"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to: {output}")
    return output


def plot_all(
    q_batch_filter: int | None,
    topk_filter: int | None,
    output_dir: Path,
) -> list[Path]:
    """Generate one figure for every available parameter combination."""

    result_files = discover_result_files()
    groups = collect_parameter_groups(result_files, q_batch_filter, topk_filter)
    outputs = []
    for (q_batch, topk), grouped_results in sorted(groups.items()):
        output = plot_parameter_group(q_batch, topk, grouped_results, output_dir)
        if output is not None:
            outputs.append(output)

    if not outputs:
        filters = []
        if q_batch_filter is not None:
            filters.append(f"QBatch={q_batch_filter}")
        if topk_filter is not None:
            filters.append(f"K={topk_filter}")
        filter_description = ", ".join(filters) if filters else "the available parameters"
        raise ValueError(f"No valid IVFPQ plots found for {filter_description}.")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot IVFPQ QPS/recall curves grouped by QBatch and K."
    )
    parser.add_argument(
        "--q-batch",
        type=int,
        default=None,
        help="Only generate this QBatch figure; default: generate all QBatch values.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Only generate this K figure; default: K=10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_all(
        q_batch_filter=args.q_batch,
        topk_filter=args.k,
        output_dir=args.output_dir,
    )
