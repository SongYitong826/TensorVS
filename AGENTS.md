# Development Environment

Use the Conda environment `tensorvs` for all Python execution, tests, and
dependency-management commands in this project. Prefer the explicit form:

```bash
conda run -n tensorvs <command>
```

# Paper Figure Synchronization

After updating either AMD comparison figure, also copy its PDF to
`/Users/yitongsong/Desktop/TensorVS_SIGMOD27/expFig/`, keeping the filename:

- `IVFFLAT_AMD_QBatch32_128_512_K10.pdf`
- `IVFPQ_AMD_QBatch32_128_512_K10.pdf`

Sync PDFs only; do not copy PNGs. `plot/plot_amd_ivf_compare.py` automatically
syncs generated PDFs when the local paper figure directory exists.

# Paper Table Synchronization

`tables/ivf_resources.pdf` corresponds to Table 5 in
`/Users/yitongsong/Desktop/TensorVS_SIGMOD27/6-experiments.tex`, identified by
`\label{tab:ivf-cuda-resource-statistics}`. After changing this table, also
update that LaTeX table, preserving its caption, label, and surrounding text.
`plot/ivf_resource_table_platform_columns.py` automatically syncs the table body
when the local paper file exists. Keep `tables/ivf_resources.tex` up to date too.

# MPS Result Organization

MPS query result filenames omit QBatch. Query CSVs use the same seven columns as
AMD/Ascend: `nprobes,q_batch,K,qps,recall,visited_vectors_per_query,peak_gpu_memory`.
Use `bench._mps_results` to write/read these pairs; full original measurements
and actual execution devices are retained in JSON `records`. Leave unmeasured
metrics empty. Current runs live in `result/`, older runs in `result/Archive/`,
and incomplete runs in `result/Archive/incomplete/`. Memory-only measurements
belong in `result/Build/`. Never mix different runs or plot incomplete curves.
