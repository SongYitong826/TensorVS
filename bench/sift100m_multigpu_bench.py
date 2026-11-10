"""Build independent SIFT100M shards and benchmark merged global Top-K on 3 GPUs.

Each shard has its own IVF quantizer/PQ codebooks and nlist=1024 by default.
This is NOT a shared-codebook, global-nlist=1024 experiment.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import statistics
import threading
import time
import traceback

import numpy as np
import torch

from tensorvs import (build_ivfflat_index, build_ivfpq_index, load_ivfflat_index,
    load_ivfpq_index, save_ivfflat_index, save_ivfpq_index,
    search_ivfflat_index, search_ivfpq_index, compute_recall_at_k)
try:
    from ._benchmark_utils import (_load_headered_binary, load_benchmark_vectors,
        load_benchmark_groundtruth, _make_padded_batch, probe_visit_statistics,
        process_gpu_memory_bytes)
except ImportError:
    from _benchmark_utils import (_load_headered_binary, load_benchmark_vectors,
        load_benchmark_groundtruth, _make_padded_batch, probe_visit_statistics,
        process_gpu_memory_bytes)

ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20
FIELDS = ('nprobes', 'q_batch', 'K', 'qps', 'recall',
          'visited_vectors_per_query', 'peak_gpu_memory')
BUILD_FIELDS = ('dataset', 'method', 'nlist', 'n_vectors', 'dim', 'build_seconds',
                'save_seconds', 'index_size_bytes', 'index_size_mib',
                'peak_build_gpu_memory_mib')
DEFAULT_PROBES = (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150)


def progress(message):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}', flush=True)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    temporary.replace(path)


def partitions(count, shards=3):
    if count < shards:
        raise ValueError('Each shard must contain at least one vector')
    return [(count * rank // shards, count * (rank + 1) // shards)
            for rank in range(shards)]


def merge_topk(results, k):
    """Merge squared-L2 shard results; invalid local IDs remain invalid."""
    distances = torch.cat([torch.as_tensor(item['distances']) for item in results], 1)
    ids = torch.cat([torch.as_tensor(item['ids']) for item in results], 1)
    distances = distances.masked_fill(ids < 0, float('inf'))
    values, order = distances.topk(k, largest=False, sorted=True)
    return values, ids.gather(1, order).masked_fill(~torch.isfinite(values), -1)


def stem(args, *, build=False):
    route = '' if build else ('Triton_' if args.backend == 'triton' else 'Tensor_')
    name = f'CUDA_TensorVS_{route}MultiGPU3_{args.kind.upper()}_{args.dataset_label}_nlist{args.nlist}'
    if args.kind == 'ivfpq':
        name += f'_M{args.pq_dim}_nbits{args.pq_bits}_train{args.pq_train_points}'
    if getattr(args, 'flat_storage', 'float32') == 'uint8':
        name += '_UInt8'
    return name + ('_build' if build else '')


def fingerprint(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def source_hashes():
    paths = [Path(__file__), ROOT / 'bench/_benchmark_utils.py']
    paths += sorted((ROOT / 'src/tensorvs').glob('*.py'))
    paths += sorted((ROOT / 'flash-kmeans/flash_kmeans').glob('*.py'))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def release_host():
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except AttributeError:
        pass


class AggregateMemorySampler:
    """Sample the simultaneous sum for these worker PIDs, excluding other jobs."""
    def __init__(self, devices, pids, interval=.01):
        self.devices, self.pids, self.interval = devices, pids, interval
        self.peak_bytes = 0
        self.per_gpu = {str(gpu): 0 for gpu in devices}
        self.samples = 0
        self.error = None
        self.stop_event = threading.Event()

    def sample(self):
        values = [process_gpu_memory_bytes(torch.device(f'cuda:{gpu}'), pid=pid)
                  for gpu, pid in zip(self.devices, self.pids)]
        self.peak_bytes = max(self.peak_bytes, sum(values))
        for gpu, value in zip(self.devices, values):
            self.per_gpu[str(gpu)] = max(self.per_gpu[str(gpu)], value)
        self.samples += 1

    def loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.sample()
            except BaseException as error:
                self.error = error
                return

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError('NVML sampling failed') from self.error
        self.sample()

    def metrics(self):
        return dict(peak_total_mib=self.peak_bytes / MIB,
                    peak_per_gpu_mib={key: value / MIB for key, value in self.per_gpu.items()},
                    samples=self.samples, interval_seconds=self.interval)


def worker(connection, args, rank, bounds, index_path, build_config, serialize_lock):
    """One persistent process owns exactly one GPU and one contiguous shard."""
    try:
        torch.set_num_threads(args.cpu_threads)
        device = torch.device(f'cuda:{args.gpus[rank]}')
        torch.cuda.set_device(device)
        torch.empty(1, device=device)  # Initialize this worker's CUDA context.
        begin_id, end_id = bounds
        load = load_ivfflat_index if args.kind == 'ivfflat' else load_ivfpq_index
        search = search_ivfflat_index if args.kind == 'ivfflat' else search_ivfpq_index
        save = save_ivfflat_index if args.kind == 'ivfflat' else save_ivfpq_index
        meta_path = index_path.with_suffix('.json')
        own_config = dict(build_config, rank=rank, begin_id=begin_id, end_id=end_id)
        exists = index_path.exists() or meta_path.exists()
        if exists:
            if not args.resume and args.phase != 'query':
                raise FileExistsError(f'Index exists; use --resume: {index_path}')
            if not index_path.is_file() or not meta_path.is_file():
                raise ValueError(f'Incomplete shard index: {index_path}')
            metadata = json.loads(meta_path.read_text())
            if metadata['config'] != own_config or metadata['status'] != 'complete':
                raise ValueError(f'Shard source/settings changed: {index_path}')
            if metadata['index'] != fingerprint(index_path):
                raise ValueError(f'Shard index was modified: {index_path}')
            base = None
        else:
            if args.phase == 'query':
                raise FileNotFoundError(index_path)
            started = time.perf_counter()
            mapped = _load_headered_binary(args.dataset_dir / 'base.bin', 'u1')
            storage_dtype = torch.uint8 if args.flat_storage == 'uint8' else torch.float32
            base = mapped[begin_id:end_id].to(dtype=storage_dtype)
            del mapped
            read_seconds = time.perf_counter() - started
            progress(f'GPU {args.gpus[rank]} read shard {rank}: IDs [{begin_id}, {end_id}); {read_seconds:.1f}s')
        connection.send(dict(state='ready', reused=exists, pid=os.getpid(),
                             gpu_name=torch.cuda.get_device_name(device)))
        index = None
        batches = []
        while True:
            command = connection.recv()
            action = command['action']
            if action == 'stop':
                return
            if action == 'build':
                if exists:
                    connection.send(dict(state='built', reused=True, metrics=metadata['metrics']))
                    continue
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                progress(f'GPU {args.gpus[rank]} building {args.kind} shard {rank}')
                common = dict(vectors=base, nlist=args.nlist, niter=args.coarse_iters,
                              use_triton=True, device=device, seed=args.seed, n_init=1)
                if args.kind == 'ivfflat':
                    built = build_ivfflat_index(**common, max_points_per_centroid=256,
                                                store_uint8=args.flat_storage == 'uint8')
                else:
                    built = build_ivfpq_index(**common, m=args.pq_dim, nbits=args.pq_bits,
                        coarse_niter=args.coarse_iters, coarse_n_init=1,
                        coarse_trainset_fraction=args.coarse_train_fraction,
                        coarse_kmeans_method='balanced', pq_niter=args.pq_iters,
                        pq_n_init=args.pq_restarts, pq_max_points_per_centroid=args.pq_train_points,
                        pq_kmeans_method='balanced', cpu_residual_chunk_size=65536)
                del common
                torch.cuda.synchronize(device)
                metrics = dict(rank=rank, gpu=args.gpus[rank], begin_id=begin_id,
                    end_id=end_id, n_vectors=end_id-begin_id, read_seconds=read_seconds,
                    build_seconds=time.perf_counter()-started,
                    peak_build_torch_allocated_mib=torch.cuda.max_memory_allocated(device)/MIB,
                    peak_build_torch_reserved_mib=torch.cuda.max_memory_reserved(device)/MIB)
                release_host()
                progress(f'GPU {args.gpus[rank]} built shard {rank} in {metrics["build_seconds"]:.1f}s')
                connection.send(dict(state='built', reused=False, metrics=metrics))
            elif action == 'save':
                if not exists:
                    # Sorting/interleaving can need multiple host copies of a shard.
                    # Serialize one shard at a time while GPU construction stays parallel.
                    with serialize_lock:
                        progress(f'GPU {args.gpus[rank]} serializing shard {rank}')
                        index_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = index_path.with_name(index_path.stem + '.partial.bin')
                        if temporary.exists():
                            raise FileExistsError(f'Interrupted build exists: {temporary}')
                        started = time.perf_counter()
                        save(built, temporary, dataset_path=args.dataset_dir / 'base.bin',
                             build_time_seconds=metrics['build_seconds'])
                        temporary.replace(index_path)
                        metrics.update(save_seconds=time.perf_counter()-started,
                                       index_size_bytes=index_path.stat().st_size)
                        metadata = dict(status='complete', config=own_config,
                                        metrics=metrics, index=fingerprint(index_path))
                        atomic_json(meta_path, metadata)
                        del built, base
                        base = None
                        release_host()
                        torch.cuda.empty_cache()
                        progress(f'GPU {args.gpus[rank]} saved shard {rank}: {metrics["index_size_bytes"]/MIB:.1f} MiB')
                connection.send(dict(state='saved', metrics=metadata['metrics'],
                                     index=metadata['index']))
            elif action == 'prepare':
                progress(f'GPU {args.gpus[rank]} loading shard {rank} for querying')
                index = load(index_path, load_lists=True)
                if int(index.metadata['num_vectors']) != end_id-begin_id:
                    raise ValueError('Loaded shard has the wrong row count')
                index.reside_on_device(device, backend=args.backend, persist_layout=False)
                if args.flat_storage == 'uint8' and index.list_vectors_interleaved.dtype != torch.uint8:
                    raise ValueError('Expected UInt8 vectors in the resident IVF-Flat index')
                query_path = args.dataset_dir / 'query.bin'
                queries = (_load_headered_binary(query_path, 'u1')
                           if args.flat_storage == 'uint8' else load_benchmark_vectors(query_path))
                queries = queries[:args.actual_nq]
                torch.cuda.synchronize(device)
                connection.send(dict(state='prepared'))
            elif action == 'configure':
                batches.clear()
                q_batch, probe = command['q_batch'], command['nprobe']
                batches = [_make_padded_batch(queries, start, q_batch, device, dtype=queries.dtype)
                           for start in range(0, len(queries), q_batch)]
                options = dict(device=device, backend=args.backend, persist_layout=False)
                if args.backend == 'torch':
                    options.update(torch_query_batch_size=q_batch, torch_memory_budget_mb=128)
                if args.kind == 'ivfpq':
                    options['lut_dtype'] = args.lut_dtype
                    if args.backend == 'triton':
                        options.update(use_preallocated_workspace=True, use_cuda_graph=True,
                                       triton_autotune=True)
                bytes_per_vector = (128 * index.list_vectors_interleaved.element_size()
                                    if args.kind == 'ivfflat' and args.backend == 'triton'
                                    else 128 * 4 if args.kind == 'ivfflat' else args.pq_dim)
                visits = probe_visit_statistics(index, queries, probe, device, bytes_per_vector)
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                connection.send(dict(state='configured', visits=visits))
            elif action == 'search':
                batch, valid = batches[command['batch']]
                with torch.inference_mode():
                    distances, ids = search(index, batch, topk=args.topk, nprobe=probe, **options)
                    distances, ids = distances[:valid].cpu(), ids[:valid].cpu()
                    ids = torch.where(ids >= 0, ids + begin_id, ids)
                connection.send(dict(distances=distances.numpy(), ids=ids.numpy()))
            elif action == 'reset_memory':
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                connection.send(dict(state='reset'))
            elif action == 'memory':
                torch.cuda.synchronize(device)
                connection.send(dict(gpu=args.gpus[rank],
                    peak_torch_allocated_mib=torch.cuda.max_memory_allocated(device)/MIB,
                    peak_torch_reserved_mib=torch.cuda.max_memory_reserved(device)/MIB))
            else:
                raise ValueError(f'Unknown command: {action}')
    except BaseException:
        try:
            connection.send(dict(error=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def receive_all(connections, processes):
    pending = {connection: rank for rank, connection in enumerate(connections)}
    results = [None] * len(connections)
    while pending:
        for connection in wait(list(pending), timeout=1):
            rank = pending.pop(connection)
            try:
                value = connection.recv()
            except EOFError as error:
                raise RuntimeError(f'GPU worker {rank} exited unexpectedly') from error
            if 'error' in value:
                raise RuntimeError(f'GPU worker {rank}:\n{value["error"]}')
            results[rank] = value
        for connection, rank in list(pending.items()):
            if not processes[rank].is_alive() and not connection.poll():
                raise RuntimeError(f'GPU worker {rank} exited with {processes[rank].exitcode}')
    return results


def benchmark(args):
    args.dataset_dir = args.dataset_dir.resolve()
    mapped = _load_headered_binary(args.dataset_dir / 'base.bin', 'u1')
    count, dimension = mapped.shape
    del mapped
    if (count, dimension) != (args.expected_vectors, 128):
        raise ValueError(f'Expected {args.expected_vectors} x 128 vectors, got {count} x {dimension}')
    truth = load_benchmark_groundtruth(args.dataset_dir / 'groundtruth.bin')
    queries = load_benchmark_vectors(args.dataset_dir / 'query.bin')
    if len(queries) != len(truth) or queries.shape[1] != 128 or truth.shape[1] < args.topk:
        raise ValueError('Queries and ground truth have inconsistent dimensions')
    args.actual_nq = min(args.nq, len(queries))
    truth = truth[:args.actual_nq, :args.topk]
    if (truth < 0).any() or (truth >= count).any():
        raise ValueError('Ground truth contains IDs outside this dataset')
    bounds = partitions(count)
    if min(end-start for start, end in bounds) < max(args.nlist, 1 << args.pq_bits):
        raise ValueError('Too few vectors per shard for the requested quantizers')
    hashes = source_hashes()
    build_config = dict(dataset=args.dataset_label, base=fingerprint(args.dataset_dir/'base.bin'),
        kind=args.kind, shards=3, independent_quantizers=True, global_nlist=3*args.nlist,
        nlist_per_shard=args.nlist, coarse_iters=args.coarse_iters, seed=args.seed,
        flat_max_points_per_centroid=256, coarse_train_fraction=args.coarse_train_fraction,
        pq_dim=args.pq_dim, pq_bits=args.pq_bits, pq_iters=args.pq_iters,
        pq_restarts=args.pq_restarts, pq_train_points=args.pq_train_points,
        cpu_residual_chunk_size=65536, source_hashes=hashes)
    if args.flat_storage == 'uint8':
        build_config.update(vector_storage_dtype='uint8', query_dtype='float32')
    index_dir = args.index_dir or args.dataset_dir/'GPUIndex'/'TensorVS_MultiGPU3'/stem(args, build=True)
    index_paths = [index_dir/f'shard{rank}of3.bin' for rank in range(3)]
    query_csv = args.output_dir/(stem(args)+'.csv')
    query_json = query_csv.with_suffix('.json')
    build_csv = args.output_dir/'Build'/(stem(args, build=True)+'.csv')
    build_json = build_csv.with_suffix('.json')
    build_manifest = index_dir/'build.json'
    if args.phase == 'query':
        recorded_build = json.loads(build_manifest.read_text())['build_config']
        # Query-only runs retain the index's original build provenance.
        # Current search sources are recorded independently for result resume checks.
        build_inputs = {key: value for key, value in build_config.items() if key != 'source_hashes'}
        recorded_inputs = {key: value for key, value in recorded_build.items() if key != 'source_hashes'}
        if build_inputs != recorded_inputs:
            raise ValueError('Existing index build settings do not match the requested configuration')
        build_config = recorded_build
    config = dict(build=build_config, gpus=args.gpus, backend=args.backend,
        q_batches=args.q_batches, nprobes=args.nprobes, nq=args.actual_nq,
        topk=args.topk, warmup=args.warmup, repetitions=args.repetitions,
        lut_dtype=args.lut_dtype, cpu_threads=args.cpu_threads,
        queries=fingerprint(args.dataset_dir/'query.bin'),
        truth=fingerprint(args.dataset_dir/'groundtruth.bin'),
        index_paths=[str(path) for path in index_paths],
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if args.phase == 'query':
        config['query_source_hashes'] = hashes
    if args.flat_storage == 'uint8':
        config['query_dtype'] = 'uint8'
    # Normalize tuples/Paths before comparison with a saved JSON document.
    config = json.loads(json.dumps(config))
    metadata = dict(status='running', config=config, results=[],
        nlist_semantics='1024 per shard by default; three independently trained quantizers',
        probe_semantics='nprobes is applied independently to each shard',
        memory_unit='MiB', memory_metric='NVML worker-process memory; maximum simultaneous sum across 3 GPUs, sampled in a separate search sweep',
        timing='median end-to-end full-query wall time, including IPC, GPU search, GPU-to-CPU result transfers and CPU global Top-K merge; excludes recall, visit statistics, query upload and memory sweep')
    if args.phase != 'build' and (query_csv.exists() or query_json.exists()):
        if not args.resume or not query_csv.is_file() or not query_json.is_file():
            raise FileExistsError(f'Results exist; use --resume: {query_csv}')
        previous = json.loads(query_json.read_text())
        if previous['config'] != config:
            raise ValueError('Cannot resume with changed data, sources, or settings')
        with query_csv.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDS):
                raise ValueError('Unexpected CSV schema')
            csv_rows = list(reader)
        details = previous['results']
        if [tuple(str(item[key]) for key in FIELDS) for item in details] != [tuple(item[key] for key in FIELDS) for item in csv_rows]:
            raise ValueError('CSV/JSON result pair is inconsistent')
        metadata['results'] = details
    completed = {(item['q_batch'], item['nprobes']) for item in metadata['results']}
    context = mp.get_context('spawn')
    lock = context.Lock()
    connections, processes = [], []
    try:
        for rank in range(3):
            parent, child = context.Pipe()
            process = context.Process(target=worker,
                args=(child, args, rank, bounds[rank], index_paths[rank], build_config, lock))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)

        def command(action, **kwargs):
            for connection in connections:
                connection.send(dict(action=action, **kwargs))
            return receive_all(connections, processes)

        ready = receive_all(connections, processes)
        metadata['workers'] = ready
        pids = [item['pid'] for item in ready]
        all_reused = all(item['reused'] for item in ready)
        if any(item['reused'] for item in ready) and not all_reused:
            raise ValueError('Partial shard set: keep it separate and rebuild all three together')
        if args.phase == 'query' or all_reused:
            command('build')  # Return the original stored build metrics; do not retrain.
            saved = command('save')
            if not build_manifest.is_file():
                raise FileNotFoundError('Original aggregate build measurements are required for index reuse')
            original = json.loads(build_manifest.read_text())
            if original['build_config'] != build_config or original['indexes'] != [item['index'] for item in saved]:
                raise ValueError('Original build measurements do not match the shard indexes')
            atomic_json(build_json, original)
            atomic_csv(build_csv, [original['row']], BUILD_FIELDS)
        else:
            with AggregateMemorySampler(args.gpus, pids) as sampler:
                started = time.perf_counter()
                built = command('build')
                build_seconds = time.perf_counter()-started
                started = time.perf_counter()
                saved = command('save')
                save_seconds = time.perf_counter()-started
            size = sum(item['metrics']['index_size_bytes'] for item in saved)
            row = dict(dataset=args.dataset_label, method='TensorVS-MultiGPU3-'+args.kind,
                nlist=args.nlist, n_vectors=count, dim=dimension,
                build_seconds=build_seconds, save_seconds=save_seconds,
                index_size_bytes=size, index_size_mib=size/MIB,
                peak_build_gpu_memory_mib=sampler.peak_bytes/MIB)
            build_document = dict(status='complete', build_config=build_config,
                row=row, indexes=[item['index'] for item in saved],
                shards=[item['metrics'] for item in saved], memory=sampler.metrics(),
                build_time_definition='parallel train + full-row assignment/encoding; excludes input conversion and serialization',
                save_time_definition='wall time for three serialized host sorting/interleaving/writes',
                memory_definition='sampled simultaneous sum of worker NVML memory across construction and serialization',
                nlist_per_shard=args.nlist, global_nlist=3*args.nlist)
            atomic_json(build_manifest, build_document)
            atomic_json(build_json, build_document)
            atomic_csv(build_csv, [row], BUILD_FIELDS)
            progress(f'All {args.kind} shards built: {build_seconds:.1f}s parallel construction; {size/MIB:.1f} MiB serialized')
        metadata['indexes'] = [item['index'] for item in saved]
        if args.phase == 'build':
            print(f'Built all three shards. Measurements: {build_csv}', flush=True)
            return
        command('prepare')

        def save_results():
            metadata['completed_rows'] = len(metadata['results'])
            metadata['expected_rows'] = len(args.q_batches)*len(args.nprobes)
            atomic_json(query_json, metadata)
            atomic_csv(query_csv, metadata['results'], FIELDS)

        save_results()
        for q_batch in args.q_batches:
            for probe in args.nprobes:
                if (q_batch, probe) in completed:
                    continue
                visits = command('configure', q_batch=q_batch, nprobe=probe)

                def sweep():
                    all_ids = []
                    started = time.perf_counter()
                    for batch in range(math.ceil(args.actual_nq/q_batch)):
                        results = command('search', batch=batch)
                        distances, ids = merge_topk(results, args.topk)
                        all_ids.append(ids)
                    return time.perf_counter()-started, torch.cat(all_ids)

                for _ in range(args.warmup):
                    sweep()
                samples = []
                for _ in range(args.repetitions):
                    elapsed, ids = sweep()
                    samples.append(elapsed)
                if (ids < 0).any() or (ids >= count).any():
                    raise ValueError('Global Top-K has missing/out-of-range IDs')
                recall = float(compute_recall_at_k(ids, truth, args.topk))
                command('reset_memory')
                with AggregateMemorySampler(args.gpus, pids) as sampler:
                    sweep()
                torch_memory = command('memory')
                row = dict(nprobes=probe, q_batch=q_batch, K=args.topk,
                    qps=args.actual_nq/statistics.median(samples), recall=recall,
                    visited_vectors_per_query=sum(item['visits']['visited_vectors_per_query'] for item in visits),
                    peak_gpu_memory=sampler.peak_bytes/MIB)
                if not all(math.isfinite(float(value)) for value in row.values()):
                    raise ValueError('Non-finite result metric')
                metadata['results'].append(dict(row, samples_seconds=samples,
                    memory=sampler.metrics(), torch_memory_per_gpu=torch_memory,
                    visit_statistics_per_shard=[item['visits'] for item in visits]))
                save_results()
                print(f'{args.kind} QBatch={q_batch} nprobe/shard={probe} '
                      f'QPS={row["qps"]:.2f} recall={recall:.6f} '
                      f'3-GPU peak={row["peak_gpu_memory"]:.1f} MiB', flush=True)
        metadata['status'] = 'complete'
        save_results()
        print(f'Saved {query_csv}', flush=True)
    except BaseException as error:
        if args.phase != 'build' and 'indexes' in metadata:
            metadata.update(status='failed', error=repr(error))
            atomic_json(query_json, metadata)
        raise
    finally:
        for connection, process in zip(connections, processes):
            if process.is_alive():
                try:
                    connection.send(dict(action='stop'))
                except (BrokenPipeError, OSError):
                    pass
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in connections:
            connection.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('ivfflat', 'ivfpq'), required=True)
    parser.add_argument('--phase', choices=('build', 'query', 'all'), default='all')
    parser.add_argument('--dataset-dir', type=Path, default=Path('/storage/vector_data/sift100M'))
    parser.add_argument('--dataset-label', default='sift100m')
    parser.add_argument('--expected-vectors', type=int, default=100_000_000)
    parser.add_argument('--gpus', type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument('--index-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'result')
    parser.add_argument('--backend', choices=('triton', 'torch'), default='triton')
    parser.add_argument('--flat-storage', choices=('float32', 'uint8'), default='float32')
    parser.add_argument('--nlist', type=int, default=1024)
    parser.add_argument('--pq-dim', type=int, default=16)
    parser.add_argument('--pq-bits', type=int, choices=(4, 8), default=8)
    parser.add_argument('--coarse-iters', type=int, default=25)
    parser.add_argument('--coarse-train-fraction', type=float, default=.5)
    parser.add_argument('--pq-iters', type=int, default=50)
    parser.add_argument('--pq-restarts', type=int, default=2)
    parser.add_argument('--pq-train-points', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--nq', type=int, default=1024)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--q-batches', type=int, nargs='+', default=[32, 128, 512])
    parser.add_argument('--nprobes', type=int, nargs='+', default=list(DEFAULT_PROBES))
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--lut-dtype', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.flat_storage == 'uint8' and (args.kind != 'ivfflat' or args.backend != 'triton'):
        parser.error('--flat-storage uint8 requires IVFFlat and the Triton backend')
    positive = [args.nlist, args.pq_dim, args.coarse_iters, args.pq_iters,
                args.pq_restarts, args.pq_train_points, args.nq, args.topk,
                args.repetitions, args.cpu_threads, args.expected_vectors,
                *args.q_batches, *args.nprobes]
    if min(positive) <= 0 or args.warmup < 0 or len(set(args.gpus)) != 3 or min(args.gpus) < 0:
        parser.error('Counts must be positive; specify three distinct nonnegative GPU IDs')
    if not 0 < args.coarse_train_fraction <= 1 or 128 % args.pq_dim:
        parser.error('Train fraction must be in (0,1]; PQ M must divide 128')
    if max(args.nprobes) > args.nlist:
        parser.error('nprobe cannot exceed nlist per shard')
    if len(set(args.q_batches)) != len(args.q_batches) or len(set(args.nprobes)) != len(args.nprobes):
        parser.error('Duplicate batch/probe values')
    if args.dataset_label == 'sift100m' and args.expected_vectors != 100_000_000:
        parser.error('Small correctness runs must use a distinct --dataset-label')
    if not args.dataset_label.replace('_', '').isalnum():
        parser.error('Dataset labels may contain letters, digits and underscores only')
    return args


if __name__ == '__main__':
    import fcntl
    arguments = parse_args()
    lock_dir = arguments.dataset_dir/'GPUIndex'/'TensorVS_MultiGPU3'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir/('gpus' + '-'.join(map(str, sorted(arguments.gpus))) + '.lock')
    with lock_path.open('a') as run_lock:
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'These GPUs already have a SIFT100M run: {lock_path}')
        benchmark(arguments)
