"""Small synthetic residual-IP correctness check; no GloVe experiment."""
import faulthandler
import json

import faiss
import numpy as np

from ascend_faiss_ivfpq_bench import search_batch


def main():
    faulthandler.enable()
    faiss.omp_set_num_threads(1)
    resources = faiss.StandardNpuResources()
    config = faiss.NpuIndexIVFPQConfig()
    config.useNpuTrain = True
    config.allowCpuCoarseQuantizer = True
    config.useFloat16LookupTables = True
    config.cp.niter = 2
    config.cp.nredo = 1
    config.cp.max_points_per_centroid = 4
    config.cp.min_points_per_centroid = 1
    config.cp.spherical = True
    print("check: residual IP constructor", flush=True)
    index = faiss.NpuIndexIVFPQ(resources, 256, 1024, 32, 8,
                               faiss.METRIC_INNER_PRODUCT, True, config)
    x = np.zeros((4096, 256), dtype=np.float32)
    x[:, :200] = np.random.default_rng(42).normal(0, .2, (len(x), 200))
    index.pq.cp.niter = 2
    index.pq.cp.nredo = 1
    index.pq.cp.max_points_per_centroid = 4
    index.pq.cp.min_points_per_centroid = 1
    print("check: NPU train", flush=True)
    index.train(x)
    print("check: NPU add", flush=True)
    index.add(x)
    resources.getResources().syncDefaultStream(0)
    print("check: CPU export", flush=True)
    cpu = faiss.IndexIVFPQ(faiss.IndexFlatIP(256), 256, 1024, 32, 8,
                          faiss.METRIC_INNER_PRODUCT)
    index.copyTo(cpu)
    assert cpu.ntotal == index.ntotal == len(x)
    assert cpu.by_residual and cpu.metric_type == faiss.METRIC_INNER_PRODUCT
    assert sum(index.getListLength(i) for i in range(1024)) == len(x)
    flat_config = faiss.NpuIndexFlatConfig()
    flat_config.storageType = faiss.NpuFlatStorageFloat16
    print("check: NPU IP coarse index", flush=True)
    coarse = faiss.NpuIndexFlatIP(resources, faiss.downcast_index(cpu.quantizer), flat_config)
    index.nprobe = cpu.nprobe = 50
    print("check: NPU IP search", flush=True)
    distances, ids, lists = search_batch(faiss, index, coarse, resources, 0, x[:32, :200], 256, 10)
    assert np.isfinite(distances).all() and (ids >= 0).all() and (ids < len(x)).all()
    centroids = cpu.quantizer.reconstruct_n(0, 1024)
    biases = np.einsum("qd,qpd->qp", x[:32], centroids[lists]).astype("float32")
    reference_d, reference_i = cpu.search_preassigned(x[:32], 10, lists, biases)
    overlap = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, reference_i)]))
    assert overlap >= .98, f"CPU/NPU neighbor agreement unexpectedly low: {overlap}"
    np.testing.assert_allclose(distances, reference_d, rtol=.05, atol=.2)
    for batch in (32, 128, 512):
        for probes in (50, 400):
            print(f"check: query batch={batch}, nprobe={probes}", flush=True)
            index.nprobe = probes
            d, i, selected = search_batch(faiss, index, coarse, resources, 0, x[:batch, :200], 256, 10)
            assert d.shape == i.shape == (batch, 10)
            assert np.isfinite(d[i >= 0]).all() and (i >= -1).all() and (i < len(x)).all(), dict(
                finite=int(np.isfinite(d).sum()), ids_min=int(i.min()), ids_max=int(i.max()),
                lists_min=int(selected.min()), lists_max=int(selected.max()),
                failed_queries=np.flatnonzero((~np.isfinite(d) | (i < 0) | (i >= len(x))).any(1)).tolist()[:20])
            cpu.nprobe = probes
            bias = np.einsum("qd,qpd->qp", x[:batch], centroids[selected]).astype("float32")
            _, expected = cpu.search_preassigned(x[:batch], 10, selected, bias)
            agreement = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(i, expected)])
            print(f"  CPU/NPU overlap={agreement:.5f}, missing slots={int((i == -1).sum())}", flush=True)
            assert agreement >= .98
    print(json.dumps(dict(metric="inner_product", native_npu_train=True,
        native_npu_search=True, cpu_export=True, cpu_npu_topk_overlap=overlap,
        vectors=len(x), nlist=1024, q_batches=[32, 128, 512], nprobes_checked=[50, 400],
        glove_experiment_executed=False)), flush=True)


if __name__ == "__main__":
    main()
