"""Working-space transforms and query lookup tables for residual IVF-PQ."""

from __future__ import annotations

import torch


_LUT_STORAGE_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


def _normalize_lut_dtype(lut_dtype: str) -> str:
    """Validate the shared Tensor/Triton IVF-PQ LUT precision switch."""
    normalized = str(lut_dtype).lower()
    if normalized not in _LUT_STORAGE_DTYPES:
        raise ValueError(
            f"Unsupported LUT dtype {lut_dtype!r}; expected one of {tuple(_LUT_STORAGE_DTYPES)}"
        )
    return normalized


def _pq_code_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.uint8
    if nbits <= 16:
        return torch.int16
    return torch.int32


def _pq_working_dim(dim: int, m: int) -> int:
    """Return the cuVS-style PQ working dimension for ``m`` subspaces."""
    if dim <= 0 or m <= 0:
        raise ValueError(f"dim and m must be positive, got dim={dim}, m={m}")
    return ((dim + m - 1) // m) * m


def _make_pq_rotation_matrix(
    dim: int,
    working_dim: int,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor | None:
    """Make the tall orthogonal PQ transform used when ``dim % m != 0``.

    cuVS maps a non-divisible input dimension into the next dimension that is
    divisible by ``pq_dim`` with a random orthogonal transform.  A tall matrix
    with orthonormal columns preserves all L2 distances while making the
    padded working space suitable for equal-width PQ subspaces.  A divisible
    shape needs no transform and returns ``None`` to preserve the existing
    fast path and on-disk format.
    """
    if working_dim < dim:
        raise ValueError(
            f"PQ working dimension must be >= input dimension, got {working_dim} < {dim}"
        )
    if working_dim == dim:
        return None

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        (working_dim, dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    q, r = torch.linalg.qr(matrix, mode="reduced")
    # Canonicalize QR column signs so a fixed seed produces a stable transform.
    signs = torch.where(torch.diagonal(r) < 0.0, -1.0, 1.0).to(q.dtype)
    return (q * signs.unsqueeze(0)).contiguous()


def _apply_pq_rotation(
    vectors: torch.Tensor,
    rotation_matrix: torch.Tensor | None,
) -> torch.Tensor:
    """Map original-space vectors into the PQ working space when required."""
    if rotation_matrix is None:
        return vectors
    if vectors.shape[1] != rotation_matrix.shape[1]:
        raise ValueError(
            "PQ rotation/input dimensions do not match: "
            f"{vectors.shape[1]} != {rotation_matrix.shape[1]}"
        )
    return (vectors @ rotation_matrix.transpose(0, 1)).contiguous()


def _build_global_query_luts_bmm(
    queries: torch.Tensor,
    pq_codebooks: torch.Tensor,
    pq_codebook_norms: torch.Tensor,
    *,
    m: int,
    subdim: int,
    use_tf32: bool,
    storage_dtype: torch.dtype | None = None,
    max_m_chunk: int = 128,
    max_fp32_scratch_bytes: int = 32 << 20,
) -> torch.Tensor:
    """Build ``||p_mk||^2 - 2 q_m^T p_mk`` as a global LUT.

    The returned Global LUT uses ``[m, q, k]`` storage for the Triton TopK
    scanner.

    The omitted ``||q||^2`` is intentionally carried by neither this table
    nor the probe bias.  For a candidate in list ``l`` the direct scanner
    evaluates ``probe_distance(q, l) + gamma[i] + sum_m G[q, m, code_m]``;
    this is algebraically identical to the residual-PQ distance because the
    former decomposition's query-norm terms cancel exactly.  Keeping them out
    avoids two query-norm reductions and their temporary global-memory tensors.
    Individual entries of this table may be negative, so unlike a squared-L2
    LUT it must not be clamped before ADC accumulation.
    """
    num_queries = queries.shape[0]
    q_sub_by_m = (
        queries.view(num_queries, m, subdim)
        .permute(1, 0, 2)
        .contiguous()
    )
    codebooks = pq_codebooks.transpose(1, 2)
    norm_base = pq_codebook_norms.unsqueeze(1)

    output_dtype = torch.float32 if storage_dtype is None else storage_dtype
    if output_dtype not in (torch.float32, torch.float16):
        raise ValueError(
            f"Global LUT storage must be FP32 or FP16, got {output_dtype}"
        )

    # FP16 is the benchmark/search storage format, but BMM accumulation stays
    # in FP32.  Materializing the complete FP32 table and then converting it
    # keeps both full tables alive at once.  Fill the sole FP16 table from a
    # bounded FP32 subspace scratch instead.  Every entry uses the same FP32
    # baddbmm expression as the unchunked path, so this changes only tensor
    # lifetime, not the ADC formula or storage precision.
    chunk_m = int(m)
    if output_dtype == torch.float16:
        if max_m_chunk <= 0 or max_fp32_scratch_bytes <= 0:
            raise ValueError("Global LUT chunk limits must be positive")
        bytes_per_subspace = max(
            1,
            int(num_queries) * int(pq_codebooks.shape[1]) * torch.float32.itemsize,
        )
        chunk_m = min(
            int(m),
            int(max_m_chunk),
            max(1, int(max_fp32_scratch_bytes) // bytes_per_subspace),
        )

    if output_dtype == torch.float32 or chunk_m >= int(m):
        # Keep the original single-BMM path for FP32 and for small FP16 LUTs.
        # The latter still uses a full FP32 temporary, but avoids unnecessary
        # launch fragmentation when the table is already small.
        if not use_tf32 or not queries.is_cuda:
            values = torch.baddbmm(
                norm_base,
                q_sub_by_m,
                codebooks,
                beta=1.0,
                alpha=-2.0,
            )
        else:
            previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                values = torch.baddbmm(
                    norm_base,
                    q_sub_by_m,
                    codebooks,
                    beta=1.0,
                    alpha=-2.0,
                )
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        return values if output_dtype == torch.float32 else values.to(output_dtype)

    ksub = int(pq_codebooks.shape[1])
    values = torch.empty(
        (int(m), int(num_queries), ksub),
        dtype=output_dtype,
        device=queries.device,
    )
    fp32_scratch = torch.empty(
        (chunk_m, int(num_queries), ksub),
        dtype=torch.float32,
        device=queries.device,
    )

    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32 if queries.is_cuda else None
    try:
        if queries.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
        for start in range(0, int(m), chunk_m):
            end = min(start + chunk_m, int(m))
            scratch_view = fp32_scratch[: end - start]
            torch.baddbmm(
                norm_base[start:end],
                q_sub_by_m[start:end],
                codebooks[start:end],
                beta=1.0,
                alpha=-2.0,
                out=scratch_view,
            )
            values[start:end].copy_(scratch_view)
    finally:
        if previous_allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32

    return values
