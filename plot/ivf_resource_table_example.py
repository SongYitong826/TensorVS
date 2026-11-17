"""Generate a two-page, seven-inch-wide preview of all platform resource tables.

Uses measured local summaries where available; xxx means pending. CUDA resource
statistics are read from the canonical CSV/JSON pairs in result/Build/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench._mps_results import comparison_paths, read_rows as mps_rows
from plot.plot_amd_ivf_compare import DATASETS as AMD_DATASETS, result_path
from plot.plot_ascend_ivf_compare import DATASETS as ASCEND_DATASETS
from plot.ivf_resource_table import DATASETS as CUDA_DATASETS, collect_records

RESULT = ROOT / "result"
DATASETS = ("SIFT", "GloVe", "Tiny", "Hunyuan", "GIST", "OpenAI")
KEYS = {"GloVe": "glove1.2m", "Hunyuan": "hunyuan"}
PLATFORMS = (
    ("NVIDIA GPU (CUDA)", DATASETS, ("TensorVS (Triton)", "cuVS", "Faiss GPU")),
    ("Apple Silicon (MPS / Metal)", ("GloVe", "Hunyuan"), ("TensorVS (Tensor)", "Faiss Metal")),
    ("AMD GPU (ROCm)", ("GloVe", "Hunyuan"),
     ("TensorVS (Triton)", "TensorVS (Tensor)", "Faiss ROCm", "hipVS")),
    ("Ascend NPU", ("GloVe", "Hunyuan"), ("TensorVS (Tensor)", "Faiss Ascend")),
    ("Cambricon MLU", ("GloVe", "Hunyuan"), ("TensorVS (Tensor)", "Baseline (TBD)")),
    ("Kunlunxin XPU", ("GloVe", "Hunyuan"), ("TensorVS (Tensor)",)),
)


def load_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def number(value):
    return f"{float(value):.2f}"


def pending():
    return dict(build="xxx", size="xxx", peak="xxx", reused=False, sources=[])


def cuda_snapshot():
    records = {}
    for dataset in CUDA_DATASETS:
        for kind in ("ivfflat", "ivfpq"):
            measured = collect_records(dataset, "IVF-Flat" if kind == "ivfflat" else "IVF-PQ")
            for method, record in measured.items():
                label = "TensorVS (Triton)" if method == "TensorVS" else method
                records[kind, dataset.label, label] = dict(
                    build=number(record.build_seconds), size=number(record.size_gib * 1024**3 / 1e9),
                    peak="N/S" if record.peak_memory_gib is None else number(record.peak_memory_gib * 1024**3 / 1e9),
                    reused=False, sources=[str(record.source_path.relative_to(ROOT))])
    assert len(records) == 36, f"Expected 36 CUDA index/method/dataset records, got {len(records)}"
    return records


def amd_records(dataset, kind):
    key = KEYS[dataset]
    config = next(config for name, _, config in AMD_DATASETS if name == key)
    records, queries = {}, {}
    for method in PLATFORMS[2][2]:
        record = pending()
        records[method] = record
        if kind == "ivfpq" and key == "hunyuan" and method == "Faiss ROCm":
            record.update(build="N/S", size="N/S", peak="N/S")
            continue
        path = result_path(RESULT, method, kind, key, config)
        if not path.exists():
            continue
        meta = load_json(path.with_suffix(".json"))
        assert meta["status"] == "complete"
        queries[method] = rows(path)
        record["sources"].append(str(path.relative_to(ROOT)))
        record["size"] = number(int(meta["config"]["index_size_bytes"]) / 1e9)
        index_meta = meta.get("index_metadata", {})
        if method.startswith("TensorVS"):
            record["build"] = number(index_meta["build_time_seconds"])
            record["reused"] = True
            record["shared_index"] = (meta["config"]["index"], meta["config"]["index_size_bytes"],
                                      index_meta["build_time_seconds"])
        elif method == "hipVS":
            # Use the build time of the exact index used by the query run.
            record["build"] = number(index_meta["build_seconds"])
        else:
            build_paths = [path for path in (RESULT / "Build").glob(f"AMD_FaissROCm_{kind.upper()}_{key}_*.json")
                           if load_json(path).get("index_path") == meta["config"]["index"]]
            assert len(build_paths) == 1
            build = load_json(build_paths[0])
            assert build["index_size_bytes"] == meta["config"]["index_size_bytes"]
            record["build"] = number(build["build_seconds"])
            record["sources"].append(str(build_paths[0].relative_to(ROOT)))
    add_peak(records, queries)
    assert records["TensorVS (Triton)"]["shared_index"] == records["TensorVS (Tensor)"]["shared_index"]
    return records


def ascend_records(dataset, kind):
    key = KEYS[dataset]
    pq_config = next(config for name, _, config in ASCEND_DATASETS if name == key)
    suffix = f"_{pq_config}" if kind == "ivfpq" else ""
    records, queries = {}, {}
    for method, prefix in (("TensorVS (Tensor)", "Ascend_TensorVS_Tensor"),
                           ("Faiss Ascend", "Ascend_Faiss")):
        record = pending()
        records[method] = record
        if method == "Faiss Ascend" and (kind == "ivfflat" or key == "hunyuan"):
            reason = ("Faiss IVF-Flat search on the Ascend host runs on CPU"
                      if kind == "ivfflat" else "Faiss Ascend IVF-PQ does not support M=384")
            record.update(build="N/S", size="N/S", peak="N/S", unsupported_reason=reason)
            continue

        path = RESULT / f"{prefix}_{kind.upper()}_{key}_nlist1024{suffix}.csv"
        metadata_path = path.with_suffix(".json")
        if not path.is_file():
            continue
        meta = load_json(metadata_path)
        measurements = rows(path)
        if (meta.get("status") != "complete"
                or meta.get("completed_rows") != len(measurements)
                or not meta.get("device", "").startswith("npu:")
                or meta.get("memory_unit") != "MiB"):
            raise ValueError(f"Expected a complete Ascend NPU benchmark: {path}")
        if method == "TensorVS (Tensor)":
            build = meta["index_metadata"]["build_time_seconds"]
            size = meta["config"]["index_size_bytes"]
            record["reused"] = True
        else:
            build = meta["build"]["build_time_seconds"]
            size = meta["build"]["index_size_bytes"]
        record.update(build=number(build), size=number(size / 1e9),
                      sources=[str(p.relative_to(ROOT)) for p in (path, metadata_path)])
        queries[method] = measurements
    if queries:
        add_peak(records, queries)
    return records


def apple_records(dataset, kind):
    key = KEYS[dataset]
    records = {method: pending() for method in PLATFORMS[1][2]}
    queries = {}
    for path in comparison_paths(RESULT, key, kind):
        measurements = mps_rows(path)
        method = "TensorVS (Tensor)" if measurements[0]["method"] == "TensorVS" else "Faiss Metal"
        queries[method] = measurements
        records[method]["sources"].append(str(path.relative_to(ROOT)))
        if method.startswith("TensorVS"):
            build_path = RESULT / "Build" / f"{path.stem}_build.json"
            build = load_json(build_path)
            records[method].update(build=number(build["build_seconds"]),
                size=number(build["index_size_bytes"] / 1e9), reused=True)
            records[method]["sources"].append(str(build_path.relative_to(ROOT)))
    add_peak(records, queries)
    if kind == "ivfpq":
        record = records["Faiss Metal"]
        cpu_paths = sorted((RESULT / "Build").glob(f"MPS_FaissCPU_IVFPQ_{key}_*_cpu_memory.csv"))
        if len(cpu_paths) > 1:
            raise ValueError(f"Ambiguous Faiss CPU memory results: {cpu_paths}")
        source_index = queries["TensorVS (Tensor)"][0]
        if cpu_paths:
            cpu_path = cpu_paths[0]
            measurements = [row for row in rows(cpu_path)
                            if int(row["K"]) == 10 and int(row["q_batch"]) == 128]
            for row in measurements:
                for field in ("index_path", "nlist", "M", "nbits"):
                    if row[field] != source_index[field]:
                        raise ValueError(f"Faiss CPU memory index mismatch ({field}): {cpu_path}")
            if not measurements:
                raise ValueError(f"Missing K=10, QBatch=128 Faiss CPU memory rows: {cpu_path}")
            record.update(
                peak=number(max(float(row["peak_rss_mib"]) for row in measurements) * 1024**2 / 1e9),
                peak_device="cpu", peak_memory_metric="current-process RSS",
                memory_nprobes=sorted({int(row["nprobes"]) for row in measurements}),
            )
            record["sources"].append(str(cpu_path.relative_to(ROOT)))
        else:
            measurements = [row for row in queries.get("Faiss Metal", [])
                            if row["execution_device"] == "cpu"
                            and int(row["K"]) == 10 and int(row["q_batch"]) == 128]
            if measurements:
                record.update(
                    peak=number(max(float(row["peak_rss_memory"]) for row in measurements) * 1024**2 / 1e9),
                    peak_device="cpu", peak_memory_metric="process high-water RSS",
                    memory_nprobes=sorted({int(row["nprobes"]) for row in measurements}),
                )
    # Native build experiments use fresh indexes; keep their build/size separate
    # from the older imported-index search memory measurements above.
    target = queries["TensorVS (Tensor)"][0]
    native_builds = []
    for path in (RESULT / "Build").glob(f"MPS_FaissMetal_{kind.upper()}_{key}_*_build.json"):
        build = load_json(path)
        if (build.get("status") == "complete" and not build.get("smoke_test", False)
                and int(build["nlist"]) == int(target["nlist"])
                and int(build["dim"]) == int(target["dim"])
                and int(build["M"]) == int(target["M"])
                and int(build["nbits"]) == int(target["nbits"])):
            native_builds.append((path, build))
    if len(native_builds) > 1:
        raise ValueError(f"Ambiguous native Faiss Metal builds: {[path for path, _ in native_builds]}")
    if native_builds:
        path, build = native_builds[0]
        records["Faiss Metal"].update(
            build=number(build["build_seconds"]), size=number(build["index_size_bytes"] / 1e9),
            build_device=build["build_device"],
            build_search_runs="Build/size: fresh native index; Peak: prior imported-index search experiment",
        )
        records["Faiss Metal"]["sources"].append(str(path.relative_to(ROOT)))
    return records


def add_peak(records, queries):
    selected = {method: {int(row["nprobes"]): float(row["peak_gpu_memory"])
                         for row in measurements
                         if int(row["K"]) == 10 and int(row["q_batch"]) == 128
                         and row.get("execution_device", "gpu") != "cpu"}
                for method, measurements in queries.items()}
    selected = {method: values for method, values in selected.items() if values}
    assert selected
    common = set.intersection(*(set(values) for values in selected.values()))
    assert common
    for method, values in selected.items():
        records[method]["peak"] = number(max(values[probe] for probe in common) * 1024**2 / 1e9)
        records[method]["memory_nprobes"] = sorted(common)


def cell(value, reused=False):
    if value == "xxx":
        return r"\textcolor{pending}{\textit{xxx}}"
    if value in ("N/S", "CPU"):
        return rf"\textsf{{{value}}}"
    return value + (r"\textsuperscript{r}" if reused else "")


def method_label(method):
    if method.startswith("TensorVS"):
        return method.replace("TensorVS", r"\textbf{TensorVS}")
    return method


def make_table(kind, snapshot, audit):
    body = []
    for platform_index, (platform, datasets, methods) in enumerate(PLATFORMS):
        if platform_index:
            body.append(r"\addlinespace[3pt]")
        body.append(r"\rowcolor{platformbg}")
        body.append(rf"\multicolumn{{5}}{{@{{\hspace{{4pt}}}}l@{{}}}}{{\strut\textbf{{{platform}}}"
                    rf"\quad\textcolor{{secondary}}{{\scriptsize {len(datasets)} datasets}}}} \\")
        body.append(r"\addlinespace[1.5pt]")
        for dataset_index, dataset in enumerate(datasets):
            if platform_index == 0:
                records = {method: snapshot[kind, dataset, method] for method in methods}
            elif platform_index == 1:
                records = apple_records(dataset, kind)
            elif platform_index == 2:
                records = amd_records(dataset, kind)
            elif platform_index == 3:
                records = ascend_records(dataset, kind)
            else:
                records = {method: pending() for method in methods}
            audit.extend(dict(platform=platform, dataset=dataset, kind=kind, method=method, **record)
                         for method, record in records.items())
            for method_index, method in enumerate(methods):
                record = records[method]
                dataset_cell = (rf"\multirow{{{len(methods)}}}{{*}}{{{dataset}}}" if len(methods) > 1 else dataset) if method_index == 0 else ""
                build_cell = cell(record["build"], record["reused"])
                size_cell = cell(record["size"])
                if platform_index == 2 and method_index == 0:
                    build_cell = rf"\multirow{{2}}{{*}}{{{build_cell}}}"
                    size_cell = rf"\multirow{{2}}{{*}}{{{size_cell}}}"
                elif platform_index == 2 and method_index == 1:
                    build_cell = size_cell = ""
                body.append(f"{dataset_cell} & {method_label(method)} & {build_cell} & {size_cell} & {cell(record['peak'])} " + r"\\")
            if dataset_index != len(datasets) - 1:
                body.append(r"\cmidrule(lr){1-5}")
    return "\n".join(body)


def generate(output):
    snapshot, audit = cuda_snapshot(), []
    parts = [r"""\documentclass[letterpaper]{article}
\usepackage[textwidth=7in,top=0.55in,bottom=0.55in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs,tabularx,multirow,array}
\usepackage[table]{xcolor}
\definecolor{platformbg}{HTML}{EAF0F6}
\definecolor{pending}{HTML}{939BA5}
\definecolor{secondary}{HTML}{55616E}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.0}
\begin{document}
"""]
    for page, kind in enumerate(("ivfflat", "ivfpq"), 1):
        display = "IVF-Flat" if kind == "ivfflat" else "IVF-PQ"
        if page > 1:
            parts.append(r"\newpage")
        parts.append(rf"{{\large\bfseries Table {page}. {display}: index and search resources}}\par")
        parts.append(r"\vspace{3pt}{\small\textcolor{secondary}{Layout preview\enspace|\enspace all platforms and methods\enspace|\enspace existing measurements + pending entries}}\par\vspace{7pt}")
        parts.append(r"""\begingroup\fontsize{8.3}{9.6}\selectfont
\begin{tabularx}{\textwidth}{@{}p{0.12\textwidth}p{0.30\textwidth}*{3}{>{\centering\arraybackslash}X}@{}}
\toprule
Dataset & Method & \shortstack{Build time\\(s)} & \shortstack{Index size\\(GB)} & \shortstack{Peak memory\\(GB)} \\
\midrule
""")
        parts.append(make_table(kind, snapshot, audit))
        parts.append(r"""\bottomrule
\end{tabularx}\endgroup
\vspace{5pt}
{\fontsize{7.1}{8.5}\selectfont\color{secondary}
\textbf{Reading the table.} Platform bands group methods; only CUDA includes all six datasets.
The two TensorVS backends on AMD share merged build-time and index-size cells; search peaks remain separate.\par
\vspace{2pt}
\textit{xxx}: data pending; \textsf{N/S}: unsupported; \textsf{CPU}: CPU search, so no device-search peak is reported.
\textsuperscript{r}Original source-index build time, not a build measured on this platform.
Apple Faiss build/serialized-index size are pending; its CSV records the imported TensorVS source index.\par
\vspace{2pt}
Peak memory: search at $K=10$, QBatch$=128$, maximized over common nprobe settings within each platform/dataset.
Memory counters differ by platform; compare methods within a platform. CUDA uses the existing rounded table values.
PQ configurations follow each source run; this preview does not normalize M or training settings.
}\par
""")
    parts.append(r"\end{document}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts))
    output.with_suffix(".sources.json").write_text(json.dumps(audit, indent=2) + "\n")
    assert len(audit) == 80
    print(f"Wrote {output} ({len(audit)} method/dataset/index records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tables" / "ivf_resource_statistics_multiplatform_example.tex")
    generate(parser.parse_args().output)
