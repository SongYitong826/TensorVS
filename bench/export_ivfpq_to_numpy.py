"""Export an existing IVF-PQ index in a process separate from Faiss/OpenMP."""
from pathlib import Path
import sys

import numpy as np
from tensorvs import load_ivfpq_index


def main():
    source, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    index = load_ivfpq_index(source, load_lists=True)
    assert index.pq_rotation_matrix is None, "Only identity PQ rotation is supported"
    assert index.metadata["pq_mode"] == "residual" and index.metadata["nbits"] == 8
    lists = index.logical_lists
    for name, tensor in (("centroids", index.centroids), ("codebooks", index.pq_codebooks),
                         ("codes", lists.data), ("ids", lists.ids),
                         ("counts", lists.counts), ("offsets", lists.offsets)):
        np.save(output / f"{name}.npy", tensor.numpy())
    print(f"Exported {len(lists.ids)} original PQ codes and IDs without retraining", flush=True)


if __name__ == "__main__":
    main()
