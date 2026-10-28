#!/usr/bin/env python3
"""Generate a SIGMOD-width IVF-Flat/IVF-PQ resource table."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/storage/vector_data")
DEFAULT_OUTPUT = PROJECT_ROOT / "tables" / "ivf_resource_statistics.tex"
TOPK = 10
Q_BATCH = 128


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str


@dataclass(frozen=True)
class Method:
    label: str
    stats_name: str
    result_root: Path
    flat_index_name: str


@dataclass(frozen=True)
class PQSpec:
    nlist: int
    m: int
    bits: int
    index_name: str
    result_name: str | None

    @property
    def config(self) -> tuple[int, int, int]:
        return self.nlist, self.m, self.bits

    @property
    def config_text(self) -> str:
        return f"{self.nlist}/{self.m}/{self.bits}"


@dataclass(frozen=True)
class Record:
    config: tuple[int, int | None, int | None]
    config_text: str
    size_gib: float
    build_seconds: float
    peak_memory_gib: float | None


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
        PROJECT_ROOT / "result",
        "TenVS_IVFFlat_nlist1024.bin",
    ),
)

PQ_SPECS: dict[str, dict[str, PQSpec]] = {
    "sift10m": {
        "cuVS": PQSpec(
            1024, 16, 8,
            "cuVS_IVFPQ_nlist1024_pqdim16_pqbits8.bin",
            "IVFPQ_sift10m_nlist1024_pqdim16_pqbits8.csv",
        ),
        "Faiss GPU": PQSpec(
            1024, 16, 8,
            "Faiss_IVFPQ_nlist1024_M16_nbits8.index",
            "IVFPQ_sift10m_nlist1024_M16_nbits8.csv",
        ),
        "TensorVS": PQSpec(
            1024, 16, 8,
            "TenVS_IVFPQResidual_nlist1024_m16_nbits8_train256.bin",
            "IVFPQ_sift10m_nlist1024_M16_nbits8_train256.csv",
        ),
    },
    "glove1.2m": {
        "cuVS": PQSpec(
            1024, 25, 8,
            "cuVS_IVFPQ_nlist1024_pqdim25_pqbits8.bin",
            "IVFPQ_glove1.2m_nlist1024_pqdim25_pqbits8.csv",
        ),
        "Faiss GPU": PQSpec(
            1024, 25, 8,
            "Faiss_IVFPQ_nlist1024_M25_nbits8.index",
            "IVFPQ_glove1.2m_nlist1024_M25_nbits8.csv",
        ),
        "TensorVS": PQSpec(
            1024, 25, 8,
            "TenVS_IVFPQResidual_nlist1024_m25_nbits8_train256.bin",
            "IVFPQ_glove1.2m_nlist1024_M25_nbits8_train512.csv",
        ),
    },
    "tiny5m": {
        "cuVS": PQSpec(
            2048, 64, 8,
            "cuVS_IVFPQ_nlist2048_pqdim64_pqbits8.bin",
            "IVFPQ_tiny5m_nlist2048_pqdim64_pqbits8.csv",
        ),
        "Faiss GPU": PQSpec(
            2048, 64, 8,
            "Faiss_IVFPQ_nlist2048_M64_nbits8.index",
            "IVFPQ_tiny5m_nlist2048_M64_nbits8.csv",
        ),
        "TensorVS": PQSpec(
            2048, 64, 8,
            "TenVS_IVFPQResidual_nlist2048_m64_nbits8_train256.bin",
            "IVFPQ_tiny5m_nlist2048_M64_nbits8_train256.csv",
        ),
    },
    "hunyuan": {
        "cuVS": PQSpec(
            1024, 384, 8,
            "cuVS_IVFPQ_nlist1024_pqdim384_pqbits8.bin",
            "IVFPQ_hunyuan_nlist1024_pqdim384_pqbits8.csv",
        ),
        # The CPU index was built, but Faiss GPU cannot launch M=384 on the
        # RTX 4090 because the kernel requires more shared memory per block
        # than the device supports.
        "Faiss GPU": PQSpec(
            1024, 384, 8,
            "Faiss_IVFPQ_nlist1024_M384_nbits8.index",
            None,
        ),
        "TensorVS": PQSpec(
            1024, 384, 8,
            "TenVS_IVFPQResidual_nlist1024_m384_nbits8_train256.bin",
            "IVFPQ_hunyuan_nlist1024_M384_nbits8_train256.csv",
        ),
    },
    "gist": {
        "cuVS": PQSpec(
            1024, 120, 8,
            "cuVS_IVFPQ_nlist1024_pqdim120_pqbits8.bin",
            "IVFPQ_gist_nlist1024_pqdim120_pqbits8.csv",
        ),
        "Faiss GPU": PQSpec(
            1024, 96, 8,
            "Faiss_IVFPQ_nlist1024_M96_nbits8.index",
            "IVFPQ_gist_nlist1024_M96_nbits8.csv",
        ),
        "TensorVS": PQSpec(
            1024, 120, 8,
            "TenVS_IVFPQResidual_nlist1024_m120_nbits8_train256.bin",
            "IVFPQ_gist_nlist1024_M120_nbits8_train256.csv",
        ),
    },
    "openai1536": {
        "cuVS": PQSpec(
            1024, 192, 8,
            "cuVS_IVFPQ_nlist1024_pqdim192_pqbits8.bin",
            "IVFPQ_openai1536_nlist1024_pqdim192_pqbits8.csv",
        ),
        "Faiss GPU": PQSpec(
            1024, 96, 8,
            "Faiss_IVFPQ_nlist1024_M96_nbits8.index",
            "IVFPQ_openai1536_nlist1024_M96_nbits8.csv",
        ),
        "TensorVS": PQSpec(
            1024, 192, 8,
            "TenVS_IVFPQResidual_nlist1024_m192_nbits8_train256.bin",
            "IVFPQ_openai1536_nlist1024_M192_nbits8_train256.csv",
        ),
    },
}


def normalize(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def index_path(dataset: Dataset, filename: str) -> Path:
    return DATA_ROOT / dataset.key / "GPUIndex" / filename


def matching_build_time(
    dataset: Dataset,
    method: Method,
    index_type: str,
    path: Path,
) -> float:
    """Match a statistic row to the actual file despite legacy MB units."""
    stats_path = DATA_ROOT / dataset.key / "GPUIndex" / "statistic.csv"
    candidates: list[tuple[float, float]] = []
    # Accept both the legacy decimal-MB rows and the current MiB rows.
    actual_sizes = (
        path.stat().st_size / (1024 * 1024),
        path.stat().st_size / 1_000_000,
    )
    with stats_path.open(newline="") as source:
        for row in csv.DictReader(source):
            if normalize(row["Index Type"]) != normalize(index_type):
                continue
            if normalize(row["Method Name"]) != method.stats_name:
                continue
            recorded_size = float(row["Index Size (MB)"])
            size_error = min(abs(recorded_size - size) for size in actual_sizes)
            candidates.append((size_error, float(row["Build Time (s)"])))
    if not candidates:
        raise ValueError(
            f"Missing {index_type} build statistics for {method.label} in {stats_path}"
        )
    size_error, build_seconds = min(candidates)
    if size_error > 0.05:
        raise ValueError(
            f"No {index_type} statistic matches {path}; closest size error={size_error:.4f} MB"
        )
    return build_seconds


def peak_memory_by_nprobe_gib(result_path: Path) -> dict[int, float]:
    candidates: dict[int, float] = {}
    with result_path.open(newline="") as source:
        for row in csv.DictReader(source):
            k = int(float(row.get("K", row.get("k", "-1"))))
            q_batch = int(float(row["q_batch"]))
            if k != TOPK or q_batch != Q_BATCH:
                continue
            nprobe = int(float(row["nprobes"]))
            if nprobe in candidates:
                raise ValueError(
                    f"Duplicate K={TOPK}, QBatch={Q_BATCH}, nprobe={nprobe} in {result_path}"
                )
            candidates[nprobe] = float(row["peak_gpu_memory"]) / 1024.0
    if not candidates:
        raise ValueError(f"No K={TOPK}, QBatch={Q_BATCH} rows in {result_path}")
    return candidates


def collect_records(
    dataset: Dataset,
    kind: str,
) -> dict[str, Record]:
    if kind == "IVF-PQ" and dataset.key not in PQ_SPECS:
        return {}

    paths: dict[str, Path] = {}
    configs: dict[str, tuple[int, int | None, int | None]] = {}
    config_texts: dict[str, str] = {}
    memory_rows: dict[str, dict[int, float]] = {}
    for method in METHODS:
        if kind == "IVF-Flat":
            path = index_path(dataset, method.flat_index_name)
            result_path = method.result_root / f"IVFFLAT_{dataset.key}_nlist1024.csv"
            configs[method.label] = (1024, None, None)
            config_texts[method.label] = "1024/--/--"
        else:
            spec = PQ_SPECS[dataset.key][method.label]
            path = index_path(dataset, spec.index_name)
            configs[method.label] = spec.config
            config_texts[method.label] = spec.config_text
        paths[method.label] = path
        if kind == "IVF-Flat":
            memory_rows[method.label] = peak_memory_by_nprobe_gib(result_path)
        elif spec.result_name is not None:
            result_path = method.result_root / spec.result_name
            memory_rows[method.label] = peak_memory_by_nprobe_gib(result_path)

    common_nprobes = set.intersection(
        *(set(values) for values in memory_rows.values())
    )
    if not common_nprobes:
        raise ValueError(
            f"No common K={TOPK}, QBatch={Q_BATCH} nprobe for {dataset.key} {kind}"
        )

    return {
        method.label: Record(
            config=configs[method.label],
            config_text=config_texts[method.label],
            size_gib=paths[method.label].stat().st_size / (1024.0**3),
            build_seconds=matching_build_time(
                dataset, method, kind, paths[method.label]
            ),
            peak_memory_gib=(
                max(memory_rows[method.label][nprobe] for nprobe in common_nprobes)
                if method.label in memory_rows
                else None
            ),
        )
        for method in METHODS
    }


def highlighted(
    value: float | None,
    metric: str,
    method_label: str,
    records: dict[str, Record],
) -> str:
    if value is None:
        return r"\ding{55}"
    rendered = f"{value:.2f}"
    peers = [
        candidate
        for record in records.values()
        if (candidate := getattr(record, metric)) is not None
    ]
    if not peers:
        return rendered
    if min(peers) == max(peers):
        return rendered
    if value == max(peers):
        return rf"\cellcolor{{tablemax}}{rendered}"
    if value == min(peers):
        return rf"\cellcolor{{tablemin}}{rendered}"
    return rendered


def data_rows(dataset: Dataset) -> list[str]:
    flat = collect_records(dataset, "IVF-Flat")
    pq = collect_records(dataset, "IVF-PQ")
    rows: list[str] = []

    for method_index, method in enumerate(METHODS):
        flat_record = flat[method.label]
        dataset_cell = (
            rf"\multirow{{3}}{{*}}{{\texttt{{{dataset.label}}}}}"
            if method_index == 0 else ""
        )
        flat_size = highlighted(flat_record.size_gib, "size_gib", method.label, flat)
        flat_build = highlighted(
            flat_record.build_seconds, "build_seconds", method.label, flat
        )
        flat_memory = highlighted(
            flat_record.peak_memory_gib, "peak_memory_gib", method.label, flat
        )
        if pq:
            pq_record = pq[method.label]
            pq_size = highlighted(pq_record.size_gib, "size_gib", method.label, pq)
            pq_build = highlighted(
                pq_record.build_seconds, "build_seconds", method.label, pq
            )
            pq_memory = highlighted(
                pq_record.peak_memory_gib, "peak_memory_gib", method.label, pq
            )
            pq_cells = f"{pq_size} & {pq_build} & {pq_memory} & -- & -- & --"
        else:
            pq_cells = "-- & -- & -- & -- & -- & --"
        rows.append(
            f"    {dataset_cell} & {method.label} "
            f"& {flat_size} & {flat_build} & {flat_memory} & -- & -- & -- "
            f"& {pq_cells} \\\\"
        )
    return rows


def generate_table() -> str:
    body_rows: list[str] = []
    for dataset_index, dataset in enumerate(DATASETS):
        body_rows.extend(data_rows(dataset))
        if dataset_index != len(DATASETS) - 1:
            body_rows.append("    \\midrule")
    body = "\n".join(body_rows)

    return rf"""\documentclass[sigconf,nonacm]{{acmart}}
\usepackage{{array}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage[table]{{xcolor}}
\usepackage{{multirow}}
\usepackage{{pifont}}

\definecolor{{tablemax}}{{HTML}}{{FADBD8}} % light red
\definecolor{{tablemin}}{{HTML}}{{D6EAF8}} % light blue

% Replace these labels when the remaining kernel names are decided.
\newcommand{{\KernelTwo}}{{Kernel 2}}
\newcommand{{\KernelThree}}{{Kernel 3}}
\newcommand{{\KernelFour}}{{Kernel 4}}

\begin{{document}}
\begin{{table*}}[t]
  \centering
  \caption{{IVF index and GPU resource statistics ($K={TOPK}$,
  $\mathrm{{QBatch}}={Q_BATCH}$). ``--'' means not yet available. CUDA peak
  memory is the absolute peak process GPU memory, including the resident
  index, maximized over nprobe values shared by methods with successful GPU
  runs. \ding{{55}} indicates that the index was built, but GPU search could
  not run because of GPU resource limits. Within each dataset and index type,
  \colorbox{{tablemax}}{{maximum}} and
  \colorbox{{tablemin}}{{minimum}} values are highlighted.}}
  \label{{tab:ivf-resource-statistics}}
  \scriptsize
  \setlength{{\tabcolsep}}{{2.2pt}}
  \renewcommand{{\arraystretch}}{{1.00}}
  \resizebox{{\textwidth}}{{!}}{{%
  \begin{{tabular}}{{>{{\centering\arraybackslash}}p{{1.45cm}}l*{{2}}{{cc*{{4}}{{c}}}}}}
    \toprule
    \multirow{{3}}{{*}}{{Dataset}} & \multirow{{3}}{{*}}{{Method}}
      & \multicolumn{{6}}{{c}}{{IVF-Flat}}
      & \multicolumn{{6}}{{c}}{{IVF-PQ}} \\
    \cmidrule(lr){{3-8}} \cmidrule(lr){{9-14}}
      & & \multirow{{2}}{{*}}{{\shortstack{{Index Size\\(GiB)}}}}
      & \multirow{{2}}{{*}}{{\shortstack{{Build Time\\(s)}}}}
      & \multicolumn{{4}}{{c}}{{Peak GPU Memory (GiB)}}
      & \multirow{{2}}{{*}}{{\shortstack{{Index Size\\(GiB)}}}}
      & \multirow{{2}}{{*}}{{\shortstack{{Build Time\\(s)}}}}
      & \multicolumn{{4}}{{c}}{{Peak GPU Memory (GiB)}} \\
    \cmidrule(lr){{5-8}} \cmidrule(lr){{11-14}}
      & & & & CUDA & \KernelTwo & \KernelThree & \KernelFour
      & & & CUDA & \KernelTwo & \KernelThree & \KernelFour \\
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
