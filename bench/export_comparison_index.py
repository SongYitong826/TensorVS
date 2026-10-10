"""Export original IVF lists for Faiss in a separate Torch/OpenMP process."""
from pathlib import Path
import sys

import numpy as np
from tensorvs import load_ivfflat_index, load_ivfpq_index


def main():
    kind, source, destination = sys.argv[1:]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    loader = load_ivfflat_index if kind == "ivfflat" else load_ivfpq_index
    index = loader(Path(source), load_lists=True)
    lists = index.logical_lists
    arrays = dict(centroids=index.centroids, data=lists.data, ids=lists.ids,
                  counts=lists.counts, offsets=lists.offsets)
    if kind == "ivfpq":
        if index.metadata["pq_mode"] != "residual" or index.metadata["nbits"] != 8:
            raise ValueError("Comparison requires residual, 8-bit PQ")
        arrays["codebooks"] = index.pq_codebooks
        if index.pq_rotation_matrix is not None:
            arrays["rotation"] = index.pq_rotation_matrix
            arrays["pq_centroids"] = index.pq_centroids
    for name, tensor in arrays.items():
        np.save(destination / f"{name}.npy", tensor.numpy())
    print(f"Exported {kind}: {len(lists.ids):,} original entries; no retraining", flush=True)


if __name__ == "__main__":
    main()
