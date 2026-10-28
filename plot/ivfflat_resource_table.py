#!/usr/bin/env python3
"""Generate the IVF-Flat index-resource LaTeX table.

The script intentionally uses only the Python standard library.  Index sizes
are read from the actual index files.  CUDA peak memory is the largest value
reported for K=10 and QBatch=128 over nprobe values shared by all methods.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path("/storage/vector_data")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tables" / "ivfflat_resource_statistics.tex"


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str


@dataclass(frozen=True)
class Method:
    label: str
    stats_name: str
    result_root: Path
    index_name: str


DATASETS = (
    Dataset("sift10m", "SIFT"),
    Dataset("glove1.2m", "GloVe"),
    Dataset("tiny5m", "Tiny"),
    Dataset("hunyuan", "Hunyuan"),
    Dataset("gist", "GIST"),
    Dataset("openai1536", "OpenAI"),
)

METHODS = (
    Method(
        "cuVS",
        "cuvs",
        Path("/home/songyitong/cuvs/python/cuvs/cuvs/result"),
        "cuVS_IVFFlat_nlist1024.bin",
    ),
    Method(
        "Faiss GPU",
        "faiss",
        Path("/home/songyitong/faiss/result"),
        "Faiss_IVFFlat_nlist1024.index",
    ),
    Method(
        "TensorVS",
        "tensorvs",
        Path("/home/songyitong/TensorVS/result"),
        "TenVS_IVFFlat_nlist1024.bin",
    ),
)

TOPK = 10
Q_BATCH = 128


def normalize(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def build_times(dataset: Dataset) -> dict[str, float]:
    stats_path = DATA_ROOT / dataset.key / "GPUIndex" / "statistic.csv"
    values: dict[str, float] = {}
    with stats_path.open(newline="") as source:
        for row in csv.DictReader(source):
            if normalize(row["Index Type"]) != "ivfflat":
                continue
            values[normalize(row["Method Name"])] = float(row["Build Time (s)"])
    missing = {method.stats_name for method in METHODS} - values.keys()
    if missing:
        raise ValueError(f"Missing IVF-Flat build times in {stats_path}: {sorted(missing)}")
    return values


def peak_memory_by_nprobe_gib(dataset: Dataset, method: Method) -> dict[int, float]:
    result_path = method.result_root / f"IVFFLAT_{dataset.key}_nlist1024.csv"
    candidates: dict[int, float] = {}
    with result_path.open(newline="") as source:
        for row in csv.DictReader(source):
            k = int(float(row.get("K", row.get("k", "-1"))))
            q_batch = int(float(row["q_batch"]))
            if k == TOPK and q_batch == Q_BATCH:
                nprobe = int(float(row["nprobes"]))
                if nprobe in candidates:
                    raise ValueError(
                        f"Duplicate K={TOPK}, QBatch={Q_BATCH}, nprobe={nprobe} "
                        f"row in {result_path}"
                    )
                candidates[nprobe] = float(row["peak_gpu_memory"]) / 1024.0
    if not candidates:
        raise ValueError(
            f"No K={TOPK}, QBatch={Q_BATCH} rows in {result_path}"
        )
    return candidates  # benchmark CSVs report MiB


def index_size_gib(dataset: Dataset, method: Method) -> float:
    path = DATA_ROOT / dataset.key / "GPUIndex" / method.index_name
    return path.stat().st_size / (1024.0**3)


def highlighted(value: float, minimum: float, maximum: float) -> str:
    rendered = f"{value:.2f}"
    if value == maximum:
        return rf"\cellcolor{{tablemax}}{rendered}"
    if value == minimum:
        return rf"\cellcolor{{tablemin}}{rendered}"
    return rendered


def generate_table() -> str:
    rows: list[str] = []
    for dataset_index, dataset in enumerate(DATASETS):
        times = build_times(dataset)
        memory_by_method = {
            method.label: peak_memory_by_nprobe_gib(dataset, method)
            for method in METHODS
        }
        common_nprobes = set.intersection(
            *(set(values) for values in memory_by_method.values())
        )
        if not common_nprobes:
            raise ValueError(
                f"No common K={TOPK}, QBatch={Q_BATCH} nprobe for {dataset.key}"
            )
        metrics = {
            method.label: {
                "size": index_size_gib(dataset, method),
                "time": times[method.stats_name],
                "memory": max(
                    memory_by_method[method.label][nprobe]
                    for nprobe in common_nprobes
                ),
            }
            for method in METHODS
        }
        extrema = {
            metric: (
                min(values[metric] for values in metrics.values()),
                max(values[metric] for values in metrics.values()),
            )
            for metric in ("size", "time", "memory")
        }

        for method_index, method in enumerate(METHODS):
            prefix = rf"\multirow{{3}}{{*}}{{\texttt{{{dataset.label}}}}}" if method_index == 0 else ""
            values = metrics[method.label]
            cells = [
                highlighted(values[metric], extrema[metric][0], extrema[metric][1])
                for metric in ("size", "time", "memory")
            ]
            rows.append(
                f"    {prefix} & {method.label} & {cells[0]} & {cells[1]} & "
                f"{cells[2]} & -- & -- & -- \\\\" 
            )
        if dataset_index != len(DATASETS) - 1:
            rows.append("    \\midrule")

    body = "\n".join(rows)
    return rf"""\documentclass{{article}}
\usepackage[margin=0.45in]{{geometry}}
\usepackage{{array}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage[table]{{xcolor}}
\usepackage{{multirow}}

% Per-dataset maximum and minimum among cuVS, Faiss GPU, and TensorVS.
\definecolor{{tablemax}}{{HTML}}{{FADBD8}} % light red
\definecolor{{tablemin}}{{HTML}}{{D6EAF8}} % light blue

% Replace these three labels when the remaining kernel names are decided.
\newcommand{{\KernelTwo}}{{Kernel 2}}
\newcommand{{\KernelThree}}{{Kernel 3}}
\newcommand{{\KernelFour}}{{Kernel 4}}

\begin{{document}}
\begin{{table*}}[t]
  \centering
  \caption{{IVF-Flat index and GPU resource statistics. Index sizes are
  calculated from the index files. CUDA peak memory is the maximum observed
  over nprobe values shared by all methods with $K={TOPK}$ and
  $\mathrm{{QBatch}}={Q_BATCH}$. Within each dataset
  and metric, \colorbox{{tablemax}}{{maximum}} and
  \colorbox{{tablemin}}{{minimum}} values are highlighted.}}
  \label{{tab:ivfflat-resource-statistics}}
  \setlength{{\tabcolsep}}{{4.2pt}}
  \renewcommand{{\arraystretch}}{{1.10}}
  \resizebox{{\textwidth}}{{!}}{{%
  \begin{{tabular}}{{>{{\centering\arraybackslash}}p{{2.0cm}}lcc*{{4}}{{c}}}}
    \toprule
    \multirow{{2}}{{*}}{{Dataset}} & \multirow{{2}}{{*}}{{Method}}
      & \multirow{{2}}{{*}}{{\shortstack{{Index Size\\(GiB)}}}}
      & \multirow{{2}}{{*}}{{\shortstack{{Build Time\\(s)}}}}
      & \multicolumn{{4}}{{c}}{{Peak GPU Memory (GiB)}} \\
    \cmidrule(lr){{5-8}}
      & & & & CUDA & \KernelTwo & \KernelThree & \KernelFour \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}}}
\end{{table*}}
\end{{document}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_table())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
