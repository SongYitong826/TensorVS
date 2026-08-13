from __future__ import annotations

from dataclasses import dataclass

import torch

from flash_kmeans import FlashKMeans


_STAGE_SEED_STRIDE = 1_000_003
_MAX_TORCH_SEED = (1 << 63) - 1
_KMEANS_CPU_CHUNK_BYTES = 256 << 20
_INERTIA_CHUNK_BYTES = 64 << 20
_MAX_KMEANS_CPU_CHUNK_ROWS = 1 << 20


@dataclass(frozen=True)
class KMeansTrainingResult:
    """Best final-label KMeans result selected from multiple initializations."""

    labels: torch.Tensor
    centroids: torch.Tensor
    inertia: float
    seed: int


def derive_kmeans_seed(base_seed: int, stage: int, restart: int) -> int:
    """Derive a stable, non-overlapping seed for a training stage and restart."""
    if stage < 0:
        raise ValueError(f"stage must be non-negative, got {stage}")
    if restart < 0:
        raise ValueError(f"restart must be non-negative, got {restart}")
    return (int(base_seed) + stage * _STAGE_SEED_STRIDE + restart) % _MAX_TORCH_SEED


def _rows_for_byte_budget(
    d: int,
    *,
    byte_budget: int,
    max_rows: int | None = None,
) -> int:
    """Return a positive row chunk whose FP32 payload fits a byte budget."""
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    rows = max(1, int(byte_budget) // (int(d) * torch.float32.itemsize))
    if max_rows is not None:
        rows = min(rows, int(max_rows))
    return rows


def _kmeans_cpu_chunk_rows(d: int) -> int:
    """Bound FlashKMeans' CPU-to-GPU chunk by bytes as well as row count."""
    return _rows_for_byte_budget(
        d,
        byte_budget=_KMEANS_CPU_CHUNK_BYTES,
        max_rows=_MAX_KMEANS_CPU_CHUNK_ROWS,
    )


def _assigned_inertia(
    data: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int | None = None,
) -> float:
    """Compute sum of squared distances without materializing an N-by-K matrix."""
    if chunk_size is None:
        chunk_size = _rows_for_byte_budget(
            int(data.shape[1]),
            byte_budget=_INERTIA_CHUNK_BYTES,
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    total = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, data.shape[0], chunk_size):
        end = min(start + chunk_size, data.shape[0])
        data_chunk = data[start:end].to(dtype=torch.float32, device=device, copy=False)
        label_chunk = labels[start:end].to(dtype=torch.int64, device=device, copy=False)
        # Reuse the gathered-centroid allocation for subtraction and squaring.
        # The former ``residual * residual`` expression allocated a second
        # N-by-D tensor; on GIST-1M that single expression requested 7.15 GiB.
        residual = centroids[label_chunk]
        residual.sub_(data_chunk).square_()
        total.add_(residual.sum(dtype=torch.float64))
    return float(total.item())


def fit_best_kmeans(
    data: torch.Tensor,
    *,
    d: int,
    k: int,
    niter: int,
    use_triton: bool,
    device: torch.device | str,
    seed: int,
    n_init: int,
    stage: int,
) -> KMeansTrainingResult:
    """Run KMeans multiple times and retain the lowest-inertia final quantizer."""
    if data.ndim != 2:
        raise ValueError(f"KMeans data must be 2D, got shape {tuple(data.shape)}")
    if data.shape[1] != d:
        raise ValueError(f"KMeans data dimension {data.shape[1]} does not match d={d}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if data.shape[0] < k:
        raise ValueError(f"KMeans requires at least k={k} rows, got {data.shape[0]}")
    if niter <= 0:
        raise ValueError(f"niter must be positive, got {niter}")
    if n_init <= 0:
        raise ValueError(f"n_init must be positive, got {n_init}")

    target_device = torch.device(device)
    best: KMeansTrainingResult | None = None

    for restart in range(int(n_init)):
        run_seed = derive_kmeans_seed(seed, stage, restart)
        kmeans = FlashKMeans(
            d=d,
            k=k,
            niter=niter,
            use_triton=use_triton,
            seed=run_seed,
            device=target_device,
            chunk_size_data_cpu=_kmeans_cpu_chunk_rows(d),
        )
        kmeans.fit(data)
        centroids = kmeans.centroids_b.squeeze(0).to(
            dtype=torch.float32,
            device=target_device,
            copy=False,
        )

        # FlashKMeans' fit labels precede its last centroid update. Always predict
        # again so labels, inertia, residuals, and serialized lists use one quantizer.
        labels = kmeans.predict(data)
        inertia = _assigned_inertia(
            data,
            labels,
            centroids,
            device=target_device,
        )
        if best is None or inertia < best.inertia:
            best = KMeansTrainingResult(
                labels=labels.clone(),
                centroids=centroids.clone(),
                inertia=inertia,
                seed=run_seed,
            )

    if best is None:  # pragma: no cover - guarded by n_init validation
        raise RuntimeError("KMeans training produced no result")
    return best
