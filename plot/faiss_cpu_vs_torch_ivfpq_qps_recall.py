"""Plot Faiss Metal IVF-PQ CPU search vs Torch MPS at matching batch sizes.

Run: conda run -n tensorvs python plot/faiss_cpu_vs_torch_ivfpq_qps_recall.py
Each figure fixes QBatch and provides linear and logarithmic QPS panels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = "IVFPQ_sift_nlist1024_M16_nbits8_train256"


def read_series(path, batch):
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["status"] == "complete" and metadata["algorithm"] == "ivfpq"
    assert metadata["q_batch"] == batch and metadata["query_count"] == 1024
    assert metadata["warmup_scope"] == "all-batches" and metadata["warmup_iterations"] == 2
    assert metadata["repetitions"] == 3
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 14
    assert all(int(row["q_batch"]) == batch and int(row["K"]) == 10 for row in rows)
    points = sorted((dict(nprobes=int(r["nprobes"]),recall=float(r["recall"]),qps=float(r["qps"]))
                     for r in rows),key=lambda r:r["nprobes"])
    assert len({p["nprobes"] for p in points}) == 14
    assert all(math.isfinite(p["qps"]) and p["qps"] > 0 and 0 <= p["recall"] <= 1 for p in points)
    return points, metadata, dict(csv=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest(),points=points)


def make_figure(batch, results_dir, faiss_results_dir, output_dir, compiled_results_dir=None,
                gpu_scheduled_results_dir=None):
    folder = results_dir / f"qbatch{batch}"
    cpu_points,cpu,cpu_source = read_series(faiss_results_dir/f"qbatch{batch}"/f"faiss_metal_cpu_{SUFFIX}.csv",batch)
    gpu_points,gpu,gpu_source = read_series(folder/f"torch_mps_{SUFFIX}.csv",batch)
    assert cpu["backend"] == "faiss_metal_cpu" and cpu["query_execution_device"] == "cpu"
    assert cpu["gpu_query_execution"] is False
    assert gpu["backend"] == "torch" and gpu["device"] == "mps" and gpu["cpu_operator_fallback"] is False
    assert cpu["index_conversion"]["source"] == dict(path=gpu["index"],
        size_bytes=gpu["index_size_bytes"],mtime_ns=gpu["index_mtime_ns"])
    assert [p["nprobes"] for p in cpu_points] == [p["nprobes"] for p in gpu_points]
    # A new Torch implementation may be compared with the existing measured
    # CPU curve. Verify its original reference without rewriting provenance.
    reference = Path(cpu["reference"])
    if reference.exists():
        assert cpu["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
        original = json.loads(reference.with_suffix(".json").read_text())
    else:
        # Authorized cleanup removed historical tables. Their original hash
        # and configuration were verified and captured before deletion.
        provenance = cpu["reference_provenance"]
        assert provenance["original_csv_deleted"] is True
        assert provenance["verified_before_cleanup"] is True
        assert provenance["csv_sha256"] == cpu["reference_sha256"]
        original = provenance["configuration"]
    for key in ("index", "index_size_bytes", "index_mtime_ns", "query_count", "q_batch",
                "topks", "nprobes", "warmup_iterations", "warmup_scope", "repetitions"):
        assert original[key] == gpu[key]
    gpu_label = "Torch compiled (MPS GPU)" if gpu["use_compile"] else "Torch eager (MPS GPU)"
    if gpu["search_options"].get("torch_pq_scan") == "gpu":
        gpu_label = "Torch GPU scheduling + compiled" if gpu["use_compile"] else "Torch GPU scheduling + eager"
        if gpu.get("gpu_scan_adaptation") and gpu["use_compile"]:
            gpu_label = "Torch adaptive + compiled (MPS GPU)"
    if gpu["search_options"].get("torch_pq_scan") == "bucketed":
        gpu_label = "Torch CPU length buckets + compiled (MPS GPU)" if gpu["use_compile"] else "Torch CPU length buckets (MPS GPU)"
        if gpu.get("adc_access", {}).get("pq_codes_per_word") == 4:
            gpu_label = "Torch length buckets + int32 ADC (MPS GPU)"
        if gpu.get("topk_selection", {}).get("exact_distances"):
            gpu_label = "Torch length buckets + region-min Top-K (MPS GPU)"
    series = [(f"Faiss Metal (CPU, {cpu['omp_threads']} threads)",cpu_points,
               dict(color="#2474B5",marker="o",linestyle="-")),
              (gpu_label,gpu_points,
               dict(color="#D66A28",marker="s",linestyle="--",markerfacecolor="white"))]
    sources = [cpu_source, gpu_source]
    if compiled_results_dir is not None:
        compiled_points,compiled,compiled_source = read_series(
            compiled_results_dir/f"qbatch{batch}"/f"torch_mps_{SUFFIX}.csv",batch)
        assert compiled["use_compile"] is True and compiled["device"] == "mps"
        assert compiled["backend"] == "torch" and compiled["cpu_operator_fallback"] is False
        for key in ("index", "index_size_bytes", "index_mtime_ns", "query_count", "q_batch",
                    "topks", "nprobes", "warmup_iterations", "warmup_scope", "repetitions", "source_sha256"):
            assert compiled[key] == gpu[key]
        series.append(("Torch compiled (MPS GPU)",compiled_points,
                       dict(color="#218C6C",marker="^",linestyle="-.",markerfacecolor="white")))
        sources.append(compiled_source)
    if gpu_scheduled_results_dir is not None:
        scheduled_points,scheduled,scheduled_source = read_series(
            gpu_scheduled_results_dir/f"qbatch{batch}"/f"torch_mps_{SUFFIX}.csv",batch)
        assert scheduled["use_compile"] is True and scheduled["device"] == "mps"
        assert scheduled["backend"] == "torch" and scheduled["cpu_operator_fallback"] is False
        assert scheduled["query_dependent_cpu_scheduling"] is False
        assert scheduled["query_tensor_host_readback"] is False
        assert scheduled["search_options"]["torch_pq_scan"] == "gpu"
        for key in ("index", "index_size_bytes", "index_mtime_ns", "query_count", "q_batch",
                    "topks", "nprobes", "warmup_iterations", "warmup_scope", "repetitions"):
            assert scheduled[key] == gpu[key]
        series.append(("Torch GPU scheduling + compiled",scheduled_points,
                       dict(color="#8553AC",marker="D",linestyle=":",markerfacecolor="white")))
        sources.append(scheduled_source)
    figure,axes = plt.subplots(1,2,figsize=(13.6,5.9),sharex=True)
    figure.subplots_adjust(left=.073,right=.98,bottom=.22,top=.78,wspace=.25)
    if len(series)>3:
        figure.subplots_adjust(top=.73)
    figure.suptitle("Faiss Metal (CPU search) vs. TensorVS Torch (GPU)",fontsize=18,y=.97)
    figure.text(.5,.905,f"SIFT1M  |  IVF-PQ: nlist = 1024, M = 16, nbits = 8  |  Apple M5  |  K = 10  |  QBatch = {batch}",
                ha="center",fontsize=10.5,color="#555555")
    all_points = [p for _, points, _ in series for p in points]
    lo,hi = min(p["recall"] for p in all_points),max(p["recall"] for p in all_points)
    for axis in axes:
        for label,points,style in series:
            axis.plot([p["recall"] for p in points],[p["qps"] for p in points],
                      label=label,linewidth=2,markersize=4.7,markeredgewidth=1.1,**style)
        axis.set_xlabel("Recall@10",fontsize=12,labelpad=8)
        axis.set_ylabel("QPS",fontsize=12,labelpad=8)
        axis.set_xlim(lo-.004,hi+.004)
        axis.xaxis.set_major_locator(MultipleLocator(.01))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        axis.tick_params(labelsize=10)
        axis.spines[["top","right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(True,linestyle="--",linewidth=.6,alpha=.3)
    axes[0].set_title("Linear QPS scale",fontsize=11,pad=12)
    axes[0].set_ylim(0,max(p["qps"] for p in all_points)*1.12)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=6,steps=[1,2,2.5,5,10]))
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[1].set_title("Logarithmic QPS scale",fontsize=11,pad=12)
    axes[1].set_yscale("log")
    axes[1].set_ylim(min(p["qps"] for p in all_points)/1.4,max(p["qps"] for p in all_points)*1.6)
    handles,labels=axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.874),ncol=2 if len(series)>3 else len(series),
                  frameon=False,fontsize=10)
    figure.text(.073,.11,"Same saved centroids, PQ codebooks, codes and IVF lists; 14 nprobe settings (5–150); 1,024 queries.",
                fontsize=9.5,color="#555555")
    figure.text(.073,.065,"2 full-query warmup sweeps; median of 3 timed sweeps. Faiss: CPU / FP32 LUT; Torch: GPU / FP16 LUT, FP32 accumulation.",
                fontsize=9.2,color="#555555")
    footer = "Available library paths on this machine; the Faiss Metal IVF-PQ search implementation delegates to CPU."
    if gpu["use_compile"] or compiled_results_dir is not None:
        footer = "Faiss Metal IVF-PQ delegates to CPU. Torch compiled: PyTorch Inductor; compilation excluded by warmup."
    figure.text(.073,.023,footer,
                fontsize=9.2,color="#666666")
    stem=f"SIFT1M_IVFPQ_FaissMetalCPU_vs_TorchGPU_QPSRecall_QBatch{batch}_K10"
    outputs=[]
    for extension,options in (("png",dict(dpi=220)),("pdf",{}),("svg",{})):
        target=output_dir/f"{stem}.{extension}"
        figure.savefig(target,**options)
        outputs.append(str(target.resolve()))
    plt.close(figure)
    manifest=dict(status="complete",q_batch=batch,K=10,x_axis="Recall@10",y_axis="QPS",
        y_scales=["linear","log"],same_q_batch=True,points_per_method=14,
        no_smoothing=True,no_point_removal=True,source_csvs=sources,
        torch_modes=["compiled" if gpu["use_compile"] else "eager"] + (["compiled"] if compiled_results_dir else []),
        gpu_scheduled_curve=gpu["search_options"].get("torch_pq_scan") == "gpu" or gpu_scheduled_results_dir is not None,
        query_dependent_cpu_scheduling=gpu.get("query_dependent_cpu_scheduling", False),
        torch_adaptation=gpu.get("gpu_scan_adaptation"),
        adc_access=gpu.get("adc_access"),
        topk_selection=gpu.get("topk_selection"),
        lut_preparation=gpu.get("lut_preparation"),
        execution_devices=dict(faiss="cpu",torch="mps"),cpu_threads=cpu["omp_threads"],
        measurement_limit=cpu["timing_comparison_limit"],precision_limit=cpu["precision_comparison"],outputs=outputs)
    (output_dir/f"{stem}.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-batches",nargs="+",type=int,default=[256,512])
    parser.add_argument("--results-dir",type=Path,default=ROOT/"results/ivfpq_gpu_schedule")
    parser.add_argument("--faiss-results-dir",type=Path,default=ROOT/"results")
    parser.add_argument("--compiled-results-dir",type=Path)
    parser.add_argument("--gpu-scheduled-results-dir",type=Path)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"plot/Figures")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.faiss_results_dir = args.faiss_results_dir or args.results_dir
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,
                         "svg.fonttype":"none","savefig.facecolor":"white"})
    for batch in args.q_batches:
        manifest=make_figure(batch,args.results_dir,args.faiss_results_dir,args.output_dir,args.compiled_results_dir,
                             args.gpu_scheduled_results_dir)
        print(json.dumps(dict(q_batch=batch,status=manifest["status"],outputs=manifest["outputs"]),indent=2))


if __name__=="__main__":
    main()
