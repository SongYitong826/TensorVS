"""Global merge, provenance, metrics and bounded residual correctness."""
from pathlib import Path
import csv
import json

import pytest
import torch

from bench.sift100m_multigpu_bench import (partitions, merge_topk, stem,
    parse_args, atomic_csv, atomic_json, FIELDS, AggregateMemorySampler)
from tensorvs.index_ivfpq import _cpu_residual_vectors


@pytest.mark.parametrize('count', [3, 10, 100_000_000])
def test_partitions_cover_all_ids(count):
    bounds = partitions(count)
    assert bounds[0][0] == 0 and bounds[-1][1] == count
    assert all(bounds[i][1] == bounds[i+1][0] for i in range(2))
    lengths = [end-start for start, end in bounds]
    assert max(lengths)-min(lengths) <= 1


def test_global_merge_matches_exhaustive_disjoint_shards():
    gen = torch.Generator().manual_seed(819)
    vectors = torch.randn(101, 7, generator=gen)
    queries = torch.randn(11, 7, generator=gen)
    distances = (queries[:, None]-vectors).square().sum(-1)
    local = []
    for start, end in partitions(len(vectors)):
        values, ids = distances[:, start:end].topk(10, largest=False)
        local.append(dict(distances=values.numpy(), ids=(ids+start).numpy()))
    actual, ids = merge_topk(local, 10)
    expected, expected_ids = distances.topk(10, largest=False)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(ids, expected_ids)


def test_merge_never_turns_missing_ids_into_valid_results():
    results = [dict(distances=[[float('inf'), 2.]], ids=[[-1, 101]]),
               dict(distances=[[1., 3.]], ids=[[0, 200]])]
    values, ids = merge_topk(results, 4)
    assert ids.tolist() == [[0, 101, 200, -1]]
    assert torch.isinf(values[0, -1])


@pytest.mark.parametrize('chunk', [1, 7, 65536])
def test_chunked_cpu_residuals_are_bitwise_identical(chunk):
    gen = torch.Generator().manual_seed(90)
    vectors = torch.randn(101, 128, generator=gen)
    centers = torch.randn(5, 128, generator=gen)
    labels = torch.randint(0, 5, (101,), generator=gen)
    assert torch.equal(_cpu_residual_vectors(vectors, centers, labels, chunk),
                       vectors-centers[labels])


def test_default_parameters_and_names():
    args = parse_args(['--kind', 'ivfpq'])
    assert args.gpus == [0, 1, 2] and args.nlist == 1024
    assert (args.pq_dim, args.pq_bits, args.pq_iters, args.pq_restarts) == (16, 8, 50, 2)
    assert args.coarse_train_fraction == .5 and args.coarse_iters == 25
    assert args.q_batches == [32, 128, 512] and args.topk == 10
    assert stem(args) == 'CUDA_TensorVS_Triton_MultiGPU3_IVFPQ_sift100m_nlist1024_M16_nbits8_train512'
    assert stem(args, build=True).endswith('_build')
    assert args.output_dir.name == 'result'


def test_result_pair_uses_amd_seven_columns(tmp_path):
    row = dict(zip(FIELDS, [5, 32, 10, 2000., .8, 12345., 5678.]))
    detail = dict(row, memory={'peak_per_gpu_mib': {'0': 1900., '1': 1900., '2': 1900.}})
    atomic_csv(tmp_path/'result.csv', [detail], FIELDS)
    atomic_json(tmp_path/'result.json', dict(results=[detail]))
    with (tmp_path/'result.csv').open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(FIELDS)
        assert len(list(reader)) == 1
    assert json.loads((tmp_path/'result.json').read_text())['results'][0]['memory']


def test_aggregate_peak_uses_simultaneous_sum(monkeypatch):
    from bench import sift100m_multigpu_bench as script
    samples = iter([10, 80, 10, 70, 10, 10])
    observed_pids = []
    def memory(device, *, pid):
        observed_pids.append(pid)
        return next(samples)
    monkeypatch.setattr(script, 'process_gpu_memory_bytes', memory)
    sampler = AggregateMemorySampler([0, 1, 2], [100, 101, 102])
    sampler.sample()
    sampler.sample()
    assert sampler.peak_bytes == 100  # Not sum(per-GPU maxima) == 160.
    assert sampler.per_gpu == {'0': 70, '1': 80, '2': 10}
    assert observed_pids == [100, 101, 102]*2


@pytest.mark.parametrize('options', [
    ['--gpus', '0', '0', '2'], ['--nprobes', '1025'],
    ['--expected-vectors', '999'], ['--pq-dim', '3'],
])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(SystemExit):
        parse_args(['--kind', 'ivfpq', *options])
