"""Platforms/methods across columns; dataset/metrics down rows, Flat/PQ groups.

The cross-platform main table contains GloVe/Hunyuan. The four additional CUDA
datasets are retained in a CUDA-only companion, without empty platform columns.
Run this script to compile tables/ivf_resources.pdf and tables/cuda_resources.pdf
and save their .tex sources. Build intermediates stay under tmp/tables/.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot.ivf_resource_table_compact import collect
from plot.ivf_resource_table_example import PLATFORMS, cell

PLATFORM_NAMES = ("NVIDIA GPU", "AMD GPU", "Apple Silicon GPU", "Huawei Ascend NPU",
                  "Cambricon MLU", "Kunlunxin XPU")
PAPER_EXPERIMENTS = Path("/Users/yitongsong/Desktop/TensorVS_SIGMOD27/6-experiments.tex")
PAPER_TABLE_LABEL = r"\label{tab:ivf-cuda-resource-statistics}"
HEADERS = {
    "TensorVS (Triton)": r"\shortstack{TensorVS\\(Triton)}",
    "TensorVS (Tensor)": r"\shortstack{TensorVS\\(Tensor)}",
    "cuVS": "cuVS", "Faiss GPU": r"\shortstack{Faiss\\GPU}",
    "Faiss Metal": r"\shortstack{Faiss\\Metal}",
    "Faiss ROCm": r"\shortstack{Faiss\\ROCm}", "hipVS": "hipVS",
    "Faiss Ascend": r"\shortstack{Faiss\\Ascend}",
}
COLORS = r"""\definecolor{e8qrow}{HTML}{EEF3F8}
\definecolor{pending}{HTML}{929AA4}
\definecolor{tensorvsbest}{HTML}{1F77B4}
"""
PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs,multirow,array,tabularx,graphicx}
\usepackage[table]{xcolor}
\begin{document}
"""


def best_tensorvs_cells(records):
    """Mark TensorVS values within 5% of each group minimum."""
    groups = {}
    for (platform, dataset, kind, method), entry in records.items():
        for metric in ("build", "size", "peak"):
            try:
                value = Decimal(str(entry[metric]))
            except InvalidOperation:
                continue
            if value.is_finite():
                groups.setdefault((platform, dataset, kind, metric), []).append((method, value))
    best = set()
    for (platform, dataset, kind, metric), values in groups.items():
        minimum = min(value for _, value in values)
        best.update((platform, dataset, kind, method, metric)
                    for method, value in values
                    if method.startswith("TensorVS (") and value <= minimum * Decimal("1.05"))
    return best


def resource_cell(records, best, platform, dataset, kind, method, metric):
    value = cell(records[platform, dataset, kind, method][metric])
    if (platform, dataset, kind, method, metric) in best:
        return r"\textcolor{tensorvsbest}{\textbf{\textit{" + value + "}}}"
    return value


def table(records, platforms, datasets):
    best = best_tensorvs_cells(records)
    columns = [(p, method) for p in platforms for method in PLATFORMS[p][2]]
    count = len(columns) + 2
    header_size = "4.8" if len(columns) > 3 else "7.0"
    # Equal method-column widths keep all neighboring values evenly spaced.
    specs_parts = []
    for group_index, platform in enumerate(platforms):
        if group_index:
            gap = "10pt" if (len(PLATFORMS[platform][2]) == 1
                             and len(PLATFORMS[platforms[group_index - 1]][2]) == 1) else "2.1pt"
            specs_parts.append(r"@{\hspace{" + gap + "}}")
        specs_parts.extend(r">{\centering\arraybackslash}X" for _ in PLATFORMS[platform][2])
    specs = "".join(specs_parts)
    parts = [r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{1.05pt}
\setlength{\aboverulesep}{0.20ex}
\setlength{\belowrulesep}{0.20ex}
\renewcommand{\arraystretch}{.88}
\renewcommand{\tabularxcolumn}[1]{m{#1}}
""", r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash\fontsize{6.3}{7}\selectfont}m{28pt}>{\raggedleft\arraybackslash\fontsize{6.3}{7}\selectfont}m{34pt}"
       + specs + r"@{}}",
             r"\toprule"]
    # Account for the two-line method labels and cmidrule spacing in the header.
    row = [r"\multirow{2}{*}[-4pt]{\textbf{Dataset}}",
           r"\multirow{2}{*}[-4pt]{\textbf{Metric}}"]
    rules = []
    start = 3
    for platform in platforms:
        methods = PLATFORMS[platform][2]
        name = rf"\fontsize{{{header_size}}}{{5.6}}\selectfont\textbf{{{PLATFORM_NAMES[platform]}}}"
        if PLATFORM_NAMES[platform] == "Cambricon MLU":
            name = r"\hspace{-2pt}" + name
        elif PLATFORM_NAMES[platform] == "Kunlunxin XPU":
            name = r"\hspace{2pt}" + name
        # A platform label must not widen its method columns (notably Kunlunxin).
        row.append(rf"\multicolumn{{{len(methods)}}}{{c}}{{\makebox[0pt]{{{name}}}}}")
        rules.append(rf"\cmidrule(lr){{{start}-{start + len(methods) - 1}}}")
        start += len(methods)
    method_headers = []
    method_size = "5.5" if len(columns) > 3 else header_size
    for _, method in columns:
        label = HEADERS[method]
        weight = r"\bfseries " if method.startswith("TensorVS (") else ""
        method_headers.append(r"{\fontsize{" + method_size + r"}{7.4}\selectfont " + weight + label + "}")
    parts.extend((" & ".join(row) + r" \\", "".join(rules),
                  " & & " + " & ".join(method_headers) + r" \\",
                  r"\midrule"))
    for kind_index, kind in enumerate(("ivfflat", "ivfpq")):
        if kind_index:
            parts.append(r"\midrule")
        name = "IVF-Flat" if kind == "ivfflat" else "IVF-PQ"
        parts.append(rf"\multicolumn{{{count}}}{{@{{}}l}}{{\textbf{{\textit{{{name}}}}}}} \\")
        for dataset_index, dataset in enumerate(datasets):
            for metric_index, (metric, label) in enumerate((
                    ("build", "Build (s)"), ("size", "Size (GB)"), ("peak", "Peak (GB)"))):
                if metric_index in (0, 2):
                    parts.append(r"\rowcolor{e8qrow}")
                # Anchor the label in the last row so alternating backgrounds
                # do not paint over the multirow text.
                metric_cell = r"{\fontsize{6.3}{7}\selectfont\mbox{" + label + "}}"
                values = [rf"\multirow{{-3}}{{*}}{{{dataset}}}" if metric_index == 2 else "", metric_cell]
                values.extend(resource_cell(records, best, platform, dataset, kind, method, metric)
                              for platform, method in columns)
                parts.append(" & ".join(values) + r" \\")
            if dataset_index != len(datasets) - 1:
                parts.append(rf"\cmidrule[.15pt](lr){{1-{count}}}")
    parts.append(r"\bottomrule\end{tabularx}\endgroup")
    return "\n".join(parts)


def cuda_transposed(records, datasets):
    best = best_tensorvs_cells(records)
    parts = [r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{1.5pt}
\setlength{\aboverulesep}{.20ex}
\setlength{\belowrulesep}{.20ex}
\renewcommand{\arraystretch}{1.0}
\renewcommand{\tabularxcolumn}[1]{m{#1}}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}m{40pt}>{\raggedright\arraybackslash}m{37pt}*{4}{>{\centering\arraybackslash}X}@{}}
\toprule
\multicolumn{2}{@{}l}{\textbf{NVIDIA / CUDA}} & \multicolumn{4}{c}{\textbf{\textit{IVF-Flat / IVF-PQ}}} \\
\cmidrule(lr){3-6}
Method & \textbf{Metric} & """
             + " & ".join(datasets) + r" \\", r"\midrule"]
    for method_index, method in enumerate(PLATFORMS[0][2]):
        for metric_index, (metric, label) in enumerate((
                ("build", "Build (s)"), ("size", "Size (GB)"), ("peak", "Peak (GB)"))):
            if metric_index in (0, 2):
                parts.append(r"\rowcolor{e8qrow}")
            heading = (r"{\fontsize{6.3}{7}\selectfont " + HEADERS[method] + "}") if metric_index == 1 else ""
            # The label lives in the unshaded middle row; the TensorVS header
            # itself occupies two lines centered across the metric triplet.
            values = [heading, r"{\fontsize{6.0}{7}\selectfont\mbox{" + label + "}}"]
            for dataset in datasets:
                flat = resource_cell(records, best, 0, dataset, "ivfflat", method, metric)
                pq = resource_cell(records, best, 0, dataset, "ivfpq", method, metric)
                values.append(r"\mbox{\fontsize{6.5}{7}\selectfont "
                              + flat + r"\,/\," + pq + "}")
            parts.append(" & ".join(values) + r" \\")
        if method_index < 2:
            parts.append(r"\cmidrule[.15pt](lr){1-6}")
    parts.append(r"\bottomrule\end{tabularx}\endgroup")
    return "\n".join(parts)


def save(output, records, platforms, datasets):
    cuda_only = platforms == (0,)
    content = cuda_transposed(records, datasets) if cuda_only else table(records, platforms, datasets)
    width = "3.5in" if cuda_only else "7in"
    preview = (PREAMBLE + COLORS + r"\begin{minipage}{" + width + "}" + "\n"
               + content + r"\end{minipage}\end{document}")
    output.write_text(preview)
    snippet = ("% Requires booktabs, multirow, array, tabularx, graphicx, xcolor[table].\n" + COLORS
               + (r"\begin{table}[t]\centering" if cuda_only else r"\begin{table*}[t]\centering") + "\n"
               + content + "\n" + (r"\end{table}" if cuda_only else r"\end{table*}") + "\n")
    output.with_suffix(".table.tex").write_text(snippet)
    audit = [dict(platform=PLATFORMS[p][0], dataset=dataset, kind=kind, method=method, **entry)
             for (p, dataset, kind, method), entry in records.items() if p in platforms and dataset in datasets]
    output.with_suffix(".sources.json").write_text(json.dumps(audit, indent=2) + "\n")
    return len(audit)


def sync_paper_table(records, path=PAPER_EXPERIMENTS):
    """Replace only the resource table body, retaining its paper caption/label."""
    if not path.exists():
        return
    original = path.read_text()
    floats = [match for match in re.finditer(
        r"\\begin\{table\*\}.*?\\end\{table\*\}", original, flags=re.DOTALL)
        if PAPER_TABLE_LABEL in match.group()]
    if len(floats) != 1 or original.count(PAPER_TABLE_LABEL) != 1:
        raise ValueError(f"Expected exactly one resource table in {path}")
    match = floats[0]
    label_end = match.start() + match.group().index(PAPER_TABLE_LABEL) + len(PAPER_TABLE_LABEL)
    body_end = match.end() - len(r"\end{table*}")
    body = "\n" + COLORS + table(records, tuple(range(6)), ("GloVe", "Hunyuan")) + "\n"
    updated = original[:label_end] + body + original[body_end:]
    if updated != original:
        path.write_text(updated)
        print(f"Synced resource table to {path}")


def main():
    records = collect()
    work_dir = ROOT / "tmp" / "tables"
    output_dir = ROOT / "tables"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_tex = work_dir / "ivf_resources.tex"
    extra = work_dir / "cuda_resources.tex"
    first = save(main_tex, records, tuple(range(6)), ("GloVe", "Hunyuan"))
    second = save(extra, records, (0,), ("SIFT", "Tiny", "GIST", "OpenAI"))
    assert first == 52 and second == 24
    compiler = shutil.which("pdflatex") or "/Library/TeX/texbin/pdflatex"
    for source in (main_tex, extra):
        result = subprocess.run(
            [compiler, "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={work_dir}", str(source)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        for suffix in (".pdf", ".tex"):
            artifact = source.with_suffix(suffix)
            destination = output_dir / artifact.name
            shutil.copy2(artifact, destination)
            print(f"Saved {destination}")
    sync_paper_table(records)
    print(f"Wrote main table ({first} records) and CUDA companion ({second} records); total 76 preserved")


if __name__ == "__main__":
    main()
