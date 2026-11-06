# hipVS on AMD MI300X

The server has hipVS in `/root/hipvs-venv`. Run the benchmark through
`bench/hipvs_python.sh` so the ROCm libraries resolve correctly. This uses
hipVS IVF-Flat and IVF-PQ directly. It does not convert TensorVS indexes or
reconstruct their vectors.

## Supply original base vectors

The existing query and ground-truth files are already on the server. Place
the original base vectors at one of these paths:

```
/root/vector_data/glove1.2m/glove1.2m_base.fvecs
/root/vector_data/hunyuan/hunyuan_base.fvecs
```

A 2D `float32` `.npy` file with the same stem also works. If the files are
elsewhere, pass `--base /absolute/path/to/base.fvecs`. Both datasets use
squared Euclidean distance and the matching query/ground-truth files.

## Run

From `/root/TensorVS`, start a separate process for each dataset and index:

```bash
cd /root/TensorVS
bash bench/hipvs_python.sh -u bench/hipvs_amd_bench.py --dataset glove1.2m --kind ivfflat
bash bench/hipvs_python.sh -u bench/hipvs_amd_bench.py --dataset glove1.2m --kind ivfpq
bash bench/hipvs_python.sh -u bench/hipvs_amd_bench.py --dataset hunyuan --kind ivfflat
bash bench/hipvs_python.sh -u bench/hipvs_amd_bench.py --dataset hunyuan --kind ivfpq
```

Each command builds and saves a hipVS index, then measures all default probe
counts and query batch sizes. The GloVe IVF-PQ setting is `M=25`, 8 bits; the
Hunyuan setting is `M=384`, 8 bits. Defaults are `nlist=1024`, `K=10`,
`q_batch=32,128,512`, `NQ=min(1024, query count)`, two warmups, and three
timed repetitions. To split a long job, use `--stage build` followed by
`--stage bench`; to continue an interrupted result, add `--resume` using
the same options. Use `--overwrite` to replace results or
`--overwrite-index` to rebuild an existing index.

Saved indexes and their metadata are in each dataset's `GPUIndex/hipVS/`
directory. Results are in `/root/TensorVS/result/` as
`hipVS_IVFFLAT_<dataset>_nlist1024.csv` or
`hipVS_IVFPQ_<dataset>_nlist1024_M<M>_nbits8_train<label>.csv`, plus a JSON sidecar.
The CSV has exactly the same seven columns and order as the TensorVS AMD
results:

```
nprobes,q_batch,K,qps,recall,visited_vectors_per_query,peak_gpu_memory
```

`peak_gpu_memory` is MiB sampled for this process with AMD-SMI in a separate
search sweep. `visited_vectors_per_query` is estimated using the original
vectors' assignments to the saved hipVS coarse centers; ties and hipVS
internal assignment details can cause small differences from its internal
list sizes. Build time, index size and the measurement definitions are in the
JSON files. The original base is not needed after building the hipVS index.

The installed hipVS wheel targets ROCm 7.2.3, while this server exposes ROCm
10.1. The provided launcher resolves the libraries on this server. Small
IVF-Flat and IVF-PQ GPU build/search/save/load tests passed on MI300X; full
dataset runs still need the original base files.
