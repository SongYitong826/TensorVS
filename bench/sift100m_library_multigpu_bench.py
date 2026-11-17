"""Native cuVS/Faiss SIFT100M build + query benchmarks on three GPUs.

The coordinator uses tensorvs; isolated native workers use the already
installed library environments. Indexes and results never mix libraries.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import socket
import statistics
import subprocess
import tempfile
import time

import numpy as np
import torch
from tensorvs import compute_recall_at_k
try:
    from .sift100m_multigpu_bench import (AggregateMemorySampler, partitions,
        merge_topk, atomic_csv, atomic_json, fingerprint, FIELDS, BUILD_FIELDS,
        DEFAULT_PROBES, progress)
    from ._sift100m_library_protocol import (send_frame, receive_frame,
        read_binary_matrix, unpack_results)
except ImportError:
    from sift100m_multigpu_bench import (AggregateMemorySampler, partitions,
        merge_topk, atomic_csv, atomic_json, fingerprint, FIELDS, BUILD_FIELDS,
        DEFAULT_PROBES, progress)
    from _sift100m_library_protocol import (send_frame, receive_frame,
        read_binary_matrix, unpack_results)

ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20
LABELS = {'cuvs': 'cuVS', 'faiss': 'FaissGPU'}


def method_label(args):
    if args.library == 'faiss' and getattr(args, 'flat_storage', None) == 'uint8':
        return 'FaissGPU_SQ8Direct'
    return LABELS[args.library]


def stem(args, build=False):
    name = f'CUDA_{method_label(args)}_MultiGPU3_{args.kind.upper()}_{args.dataset_label}_nlist{args.nlist}'
    if args.kind == 'ivfflat' and (args.library == 'cuvs' or getattr(args, 'flat_storage', None) == 'uint8'):
        name += '_UInt8'
    if args.kind == 'ivfpq':
        name += f'_M{args.pq_dim}_nbits{args.pq_bits}_train{args.pq_train_points}'
        if not build and args.lut_dtype=='fp16':
            name += '_LUTfp16'
    if not build and args.query_input == 'host':
        name += '_HostInput'
    return name + ('_build' if build else '')


def sources():
    paths = [Path(__file__), ROOT/'bench/_sift100m_library_worker.py',
             ROOT/'bench/_sift100m_library_protocol.py',
             ROOT/'bench/sift100m_multigpu_bench.py', ROOT/'bench/_benchmark_utils.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class NativePool:
    def __init__(self, args, config, log_dir):
        self.processes, self.connections, self.logs = [], [None]*3, []
        self.temporary = tempfile.TemporaryDirectory(prefix='sift100m-native-')
        directory = Path(self.temporary.name)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.address = directory/'worker.sock'
        self.listener.bind(str(self.address))
        self.listener.listen(3)
        self.listener.settimeout(1)
        config_path = directory/'config.json'
        config_path.write_text(json.dumps(config))
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            for rank, gpu in enumerate(args.gpus):
                log = (log_dir/f'gpu{gpu}.log').open('w')
                self.logs.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS=str(args.cpu_threads),
                           MKL_NUM_THREADS=str(args.cpu_threads),
                           OPENBLAS_NUM_THREADS=str(args.cpu_threads),
                           PYTHONUNBUFFERED='1')
                process = subprocess.Popen([str(args.conda), 'run', '--no-capture-output',
                    '-n', args.worker_env or args.library, 'python', '-u',
                    str(ROOT/'bench/_sift100m_library_worker.py'), '--socket', str(self.address),
                    '--config', str(config_path), '--rank', str(rank)], env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.processes.append(process)
            deadline = time.monotonic()+120
            while any(c is None for c in self.connections):
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Workers did not connect; see {log_dir}')
                try:
                    connection, _ = self.listener.accept()
                except socket.timeout:
                    self.check_processes()
                    continue
                message, payload = receive_frame(connection)
                rank = int(message['rank'])
                if message.get('state')!='connected' or rank not in range(3) or self.connections[rank]:
                    connection.close()
                    raise ValueError('Invalid worker handshake')
                self.connections[rank] = connection
            self.ready = self.collect(list(range(3)))
        except BaseException:
            self.close()
            raise

    def check_processes(self):
        for rank, process in enumerate(self.processes):
            if process.poll() is not None:
                raise RuntimeError(f'Native worker {rank} exited ({process.returncode}); inspect worker logs')

    def collect(self, ranks):
        results = {}
        with selectors.DefaultSelector() as selector:
            for rank in ranks:
                selector.register(self.connections[rank], selectors.EVENT_READ, rank)
            while selector.get_map():
                ready = selector.select(timeout=1)
                if not ready:
                    self.check_processes()
                    continue
                for key, _ in ready:
                    rank = key.data
                    message, payload = receive_frame(key.fileobj)
                    if 'error' in message:
                        raise RuntimeError(f'Native worker {rank}:\n{message["error"]}')
                    results[rank] = unpack_results(message, payload) if 'rows' in message else message
                    selector.unregister(key.fileobj)
        return [results[rank] for rank in ranks]

    def call(self, action, *, ranks=None, payload=b'', **parameters):
        ranks = list(range(3)) if ranks is None else list(ranks)
        for rank in ranks:
            send_frame(self.connections[rank], dict(action=action, **parameters), payload)
        return self.collect(ranks)

    def close(self):
        import signal
        for connection in self.connections:
            if connection is not None:
                try:
                    send_frame(connection, dict(action='stop'))
                except (OSError, EOFError):
                    pass
                connection.close()
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Each conda wrapper/worker has its own process group.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        self.listener.close()
        for log in self.logs:
            log.close()
        self.temporary.cleanup()


def benchmark(args):
    torch.set_num_threads(args.cpu_threads)
    args.dataset_dir = args.dataset_dir.resolve()
    base_path = args.dataset_dir/'base.bin'
    base = read_binary_matrix(base_path, 'u1')
    count, dimension = base.shape
    del base
    if (count, dimension) != (args.expected_vectors, 128):
        raise ValueError(f'Expected {args.expected_vectors} x 128 vectors, got {count} x {dimension}')
    queries = read_binary_matrix(args.dataset_dir/'query.bin', 'u1').astype(np.float32)
    truth = read_binary_matrix(args.dataset_dir/'groundtruth.bin', '<u4').astype(np.int64)
    if len(queries)!=len(truth) or queries.shape[1]!=128 or truth.shape[1]<args.topk:
        raise ValueError('Query/ground-truth dimensions are inconsistent')
    actual_nq = min(args.nq, len(queries))
    queries, truth = queries[:actual_nq], truth[:actual_nq, :args.topk]
    if np.any(truth<0) or np.any(truth>=count):
        raise ValueError('Ground-truth IDs are outside the dataset')
    bounds = partitions(count)
    if min(end-start for start,end in bounds) < max(args.nlist, 1<<args.pq_bits):
        raise ValueError('Not enough rows per shard for the requested quantizers')
    index_dir = args.index_dir or args.dataset_dir/'GPUIndex'/(LABELS[args.library]+'_MultiGPU3')/stem(args, True)
    suffix = '.index' if args.library=='faiss' else '.bin'
    index_paths = [index_dir/f'shard{rank}of3{suffix}' for rank in range(3)]
    query_csv = args.output_dir/(stem(args)+'.csv')
    query_json = query_csv.with_suffix('.json')
    build_csv = args.output_dir/'Build'/(stem(args,True)+'.csv')
    build_json = build_csv.with_suffix('.json')
    manifest_path = index_dir/'build.json'
    worker_config = dict(library=args.library, kind=args.kind, dataset_dir=str(args.dataset_dir),
        bounds=bounds, gpus=args.gpus, index_paths=[str(p) for p in index_paths],
        nlist=args.nlist, pq_dim=args.pq_dim, pq_bits=args.pq_bits,
        coarse_iters=args.coarse_iters, train_fraction=args.train_fraction,
        pq_iters=args.pq_iters, pq_restarts=args.pq_restarts,
        pq_train_points=args.pq_train_points, seed=args.seed,
        add_batch_size=args.add_batch_size, actual_nq=actual_nq, topk=args.topk,
        query_input=args.query_input, lut_dtype=args.lut_dtype, cpu_threads=args.cpu_threads,
        faiss_temp_memory_mib=args.faiss_temp_memory_mib)
    faiss_sq8_direct = method_label(args) == 'FaissGPU_SQ8Direct'
    worker_config['faiss_sq8_direct'] = faiss_sq8_direct
    build_config = dict(library=args.library, kind=args.kind, base=fingerprint(base_path),
        dataset=args.dataset_label, shards=3, nlist_per_shard=args.nlist,
        global_nlist=3*args.nlist, independent_quantizers=True,
        dtype='uint8' if args.kind=='ivfflat' and (args.library=='cuvs' or faiss_sq8_direct) else 'float32',
        metric='squared L2', bounds=bounds,
        coarse_iters=args.coarse_iters, train_fraction=args.train_fraction,
        pq_dim=args.pq_dim, pq_bits=args.pq_bits, pq_train_points=args.pq_train_points,
        actual_pq_iters=args.coarse_iters if args.library=='cuvs' else args.pq_iters,
        pq_restarts='native library default' if args.library=='cuvs' else args.pq_restarts,
        seed='native library default (API has no seed parameter)' if args.library=='cuvs' else args.seed,
        add_batch_size=args.add_batch_size, faiss_use_cuvs=False if args.library=='faiss' else None,
        source_hashes=sources())
    if faiss_sq8_direct:
        build_config.update(faiss_index_type='GpuIndexIVFScalarQuantizer',
                            faiss_quantizer_type='QT_8bit_direct', by_residual=False, query_dtype='float32')
    build_config = json.loads(json.dumps(build_config))
    config = dict(build=build_config, gpus=args.gpus, worker_env=args.worker_env or args.library,
        queries=fingerprint(args.dataset_dir/'query.bin'), truth=fingerprint(args.dataset_dir/'groundtruth.bin'),
        nq=actual_nq, topk=args.topk, q_batches=args.q_batches, nprobes=args.nprobes,
        warmup=args.warmup, repetitions=args.repetitions, query_input=args.query_input,
        lut_dtype=args.lut_dtype, cpu_threads=args.cpu_threads,
        faiss_temp_memory_mib=args.faiss_temp_memory_mib, index_paths=[str(p) for p in index_paths])
    config = json.loads(json.dumps(config))
    metadata = dict(status='running', config=config, results=[],
        timing='median full-query wall time: dispatch to three workers, GPU search, output device-to-host copies, response IPC, CPU global Top-K and final concatenation; excludes warmup, recall, visit statistics and memory sweep',
        query_input=args.query_input,
        query_transfer=('queries pre-uploaded before timing' if args.query_input=='resident-gpu' else 'CPU query payload dispatch and H2D upload included in timing'),
        memory_unit='MiB', memory_metric='maximum sampled simultaneous sum of the three native-worker NVML process memories in a separate sweep',
        nlist_semantics='1024 per independent shard by default; global list count is 3072',
        probe_semantics='nprobes applies independently to each shard')
    if args.phase!='build' and (query_csv.exists() or query_json.exists()):
        if not args.resume or not query_csv.is_file() or not query_json.is_file():
            raise FileExistsError(f'Results exist; use --resume: {query_csv}')
        previous = json.loads(query_json.read_text())
        if previous['config'] != config:
            raise ValueError('Cannot resume with different settings, data, or source code')
        import csv
        with query_csv.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDS):
                raise ValueError('CSV schema differs')
            rows = list(reader)
        details = previous['results']
        expected = [tuple('' if r.get(k) is None else str(r[k]) for k in FIELDS) for r in details]
        if expected != [tuple(r[k] for k in FIELDS) for r in rows]:
            raise ValueError('CSV/JSON pair is inconsistent')
        metadata['results'] = details
    present = [path.exists() for path in index_paths]
    reuse = any(present) or manifest_path.exists()
    if reuse:
        if not all(present) or not manifest_path.is_file():
            raise ValueError('Incomplete native index set; keep it separate and rebuild all shards')
        if not args.resume and args.phase!='query':
            raise FileExistsError('Native indexes exist; use --resume or --phase query')
        original = json.loads(manifest_path.read_text())
        if original.get('status')!='complete' or original['build_config'] != build_config:
            raise ValueError('Index build settings/sources do not match')
        if original['indexes'] != [fingerprint(p) for p in index_paths]:
            raise ValueError('Native index files changed since their build')
    elif args.phase=='query':
        raise FileNotFoundError('Query-only phase requires all three saved native indexes')
    run_label = time.strftime('%Y%m%d_%H%M%S')
    log_dir = args.output_dir/'logs'/f'{args.library}_{args.kind}_workers_{run_label}_{os.getpid()}'
    progress(f'{args.library} {args.kind}: starting native workers on GPUs {args.gpus}; logs={log_dir}')
    pool = NativePool(args, worker_config, log_dir)
    try:
        metadata['workers'] = pool.ready
        software = [{k: item[k] for k in ('library_version','python','environment',
            'cuda_runtime_version','cuda_runtime_library')} for item in pool.ready]
        metadata['software'] = software
        pids = [item['pid'] for item in pool.ready]
        if reuse:
            if original['software'] != software:
                raise ValueError('Library environment differs from original build')
            pool.call('load')
            atomic_json(build_json, original)
            atomic_csv(build_csv, [original['row']], BUILD_FIELDS)
        else:
            progress('Reading and converting each contiguous third to FP32')
            reads = pool.call('read')
            progress('Constructing the three native indexes concurrently (train + batch add)')
            with AggregateMemorySampler(args.gpus, pids) as sampler:
                started = time.perf_counter()
                built = pool.call('build')
                build_seconds = time.perf_counter()-started
                progress(f'Parallel native build complete in {build_seconds:.2f}s; saving indexes')
                started = time.perf_counter()
                saved = []
                for rank in range(3):
                    result = pool.call('save', ranks=[rank])[0]
                    saved.append(result)
                    progress(f'Saved shard {rank}: {result["metrics"]["index_size_bytes"]/MIB:.1f} MiB')
                save_seconds = time.perf_counter()-started
            size = sum(item['metrics']['index_size_bytes'] for item in saved)
            row = dict(dataset=args.dataset_label, method=method_label(args)+'-MultiGPU3-'+args.kind,
                nlist=args.nlist, n_vectors=count, dim=dimension, build_seconds=build_seconds,
                save_seconds=save_seconds, index_size_bytes=size, index_size_mib=size/MIB,
                peak_build_gpu_memory_mib=sampler.peak_bytes/MIB)
            original = dict(status='complete', build_config=build_config, software=software,
                row=row, indexes=[fingerprint(p) for p in index_paths],
                shards=[dict(item['metrics'],read_seconds=reads[rank]['read_seconds'],
                             begin_id=bounds[rank][0],end_id=bounds[rank][1]) for rank,item in enumerate(saved)],
                memory=sampler.metrics(), global_nlist=3*args.nlist,
                build_time_definition='parallel wall time for native training plus adding every shard vector; includes native allocations/transfers, excludes host input conversion and serialization',
                save_time_definition='wall time for serialized native saves, including Faiss GPU-to-CPU cloning',
                memory_definition='sampled simultaneous sum of native-worker NVML memory during construction and saving')
            atomic_json(manifest_path, original)
            atomic_json(build_json, original)
            atomic_csv(build_csv, [row], BUILD_FIELDS)
            for rank,path in enumerate(index_paths):
                atomic_json(path.with_suffix('.json'), dict(status='complete', config=build_config,
                    software=software[rank], metrics=original['shards'][rank], index=fingerprint(path)))
        metadata['indexes'] = original['indexes']
        if args.phase=='build':
            return

        def save_results():
            metadata['completed_rows'] = len(metadata['results'])
            metadata['expected_rows'] = len(args.q_batches)*len(args.nprobes)
            atomic_json(query_json, metadata)
            atomic_csv(query_csv, metadata['results'], FIELDS)

        save_results()
        completed = {(row['q_batch'],row['nprobes']) for row in metadata['results']}
        for q_batch in args.q_batches:
            for probe in args.nprobes:
                if (q_batch,probe) in completed:
                    continue
                visits = pool.call('configure', q_batch=q_batch, nprobe=probe)
                host_payloads = []
                if args.query_input=='host':
                    for start in range(0,actual_nq,q_batch):
                        padded = np.zeros((q_batch,128),dtype='<f4')
                        part = queries[start:start+q_batch]
                        padded[:len(part)] = part
                        host_payloads.append(padded.tobytes())

                def sweep():
                    saved_ids = []
                    started = time.perf_counter()
                    for batch in range(math.ceil(actual_nq/q_batch)):
                        results = pool.call('search', batch=batch,
                            payload=host_payloads[batch] if host_payloads else b'')
                        _, ids = merge_topk(results,args.topk)
                        saved_ids.append(ids)
                    ids = torch.cat(saved_ids)
                    return time.perf_counter()-started, ids

                for _ in range(args.warmup):
                    sweep()
                samples = []
                for _ in range(args.repetitions):
                    elapsed, ids = sweep()
                    samples.append(elapsed)
                if ids.shape!=(actual_nq,args.topk) or (ids<0).any() or (ids>=count).any():
                    raise ValueError('Invalid global Top-K IDs')
                recall = float(compute_recall_at_k(ids,torch.from_numpy(truth),args.topk))
                pool.call('synchronize')
                with AggregateMemorySampler(args.gpus,pids) as sampler:
                    sweep()
                values = [item['visits']['visited_vectors_per_query'] for item in visits]
                visited = None if any(v is None for v in values) else sum(values)
                row = dict(nprobes=probe,q_batch=q_batch,K=args.topk,
                    qps=actual_nq/statistics.median(samples),recall=recall,
                    visited_vectors_per_query=visited,peak_gpu_memory=sampler.peak_bytes/MIB)
                if not all(v is None or math.isfinite(float(v)) for v in row.values()):
                    raise ValueError('Non-finite measurement')
                metadata['results'].append(dict(row,samples_seconds=samples,memory=sampler.metrics(),
                    visit_statistics_per_shard=[item['visits'] for item in visits]))
                save_results()
                progress(f'{args.library} {args.kind} QBatch={q_batch} nprobe/shard={probe}: '
                         f'QPS={row["qps"]:.2f} recall={recall:.6f}, peak={row["peak_gpu_memory"]:.1f} MiB')
        metadata['status'] = 'complete'
        save_results()
        progress(f'Complete: {query_csv}')
    except BaseException as error:
        if args.phase!='build' and 'indexes' in metadata:
            metadata.update(status='failed',error=repr(error))
            atomic_json(query_json,metadata)
        raise
    finally:
        pool.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',choices=('cuvs','faiss'),required=True)
    parser.add_argument('--kind',choices=('ivfflat','ivfpq'),required=True)
    parser.add_argument('--flat-storage',choices=('uint8',),help='Use lossless Faiss SQ8Direct storage for IVF-Flat')
    parser.add_argument('--phase',choices=('build','query','all'),default='all')
    parser.add_argument('--dataset-dir',type=Path,default=Path('/storage/vector_data/sift100M'))
    parser.add_argument('--dataset-label',default='sift100m')
    parser.add_argument('--expected-vectors',type=int,default=100_000_000)
    parser.add_argument('--gpus',type=int,nargs=3,default=[0,1,2])
    parser.add_argument('--conda',type=Path,default=Path('/home/songyitong/miniconda3/bin/conda'))
    parser.add_argument('--worker-env')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'result')
    parser.add_argument('--index-dir',type=Path)
    parser.add_argument('--nlist',type=int,default=1024)
    parser.add_argument('--pq-dim',type=int,default=16)
    parser.add_argument('--pq-bits',type=int,choices=(4,8),default=8)
    parser.add_argument('--coarse-iters',type=int,default=25)
    parser.add_argument('--train-fraction',type=float,default=.5)
    parser.add_argument('--pq-iters',type=int,default=50)
    parser.add_argument('--pq-restarts',type=int,default=2)
    parser.add_argument('--pq-train-points',type=int,default=512)
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--add-batch-size',type=int,default=65536)
    parser.add_argument('--faiss-temp-memory-mib',type=int,default=128)
    parser.add_argument('--query-input',choices=('resident-gpu','host'),default='resident-gpu')
    parser.add_argument('--lut-dtype',choices=('fp16','fp32'),default='fp32')
    parser.add_argument('--nq',type=int,default=1024)
    parser.add_argument('--topk',type=int,default=10)
    parser.add_argument('--q-batches',type=int,nargs='+',default=[32,128,512])
    parser.add_argument('--nprobes',type=int,nargs='+',default=list(DEFAULT_PROBES))
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(argv)
    if args.flat_storage is not None and (args.kind != 'ivfflat' or args.library != 'faiss'):
        parser.error('--flat-storage uint8 is for Faiss IVF-Flat; cuVS IVF-Flat already uses UInt8')
    counts = [args.expected_vectors,args.nlist,args.pq_dim,args.coarse_iters,args.pq_iters,
        args.pq_restarts,args.pq_train_points,args.add_batch_size,args.faiss_temp_memory_mib,
        args.nq,args.topk,args.repetitions,args.cpu_threads,*args.q_batches,*args.nprobes]
    if min(counts)<=0 or args.warmup<0 or len(set(args.gpus))!=3 or min(args.gpus)<0:
        parser.error('Counts must be positive and GPU IDs distinct/nonnegative')
    if not 0<args.train_fraction<=1 or 128%args.pq_dim or max(args.nprobes)>args.nlist:
        parser.error('Invalid training fraction, PQ dimension or nprobe')
    if len(set(args.q_batches))!=len(args.q_batches) or len(set(args.nprobes))!=len(args.nprobes):
        parser.error('Duplicate query batch or nprobe')
    if not args.dataset_label.replace('_','').isalnum():
        parser.error('Dataset labels allow letters, digits and underscores only')
    if args.dataset_label=='sift100m' and args.expected_vectors!=100_000_000:
        parser.error('Small correctness runs must use a distinct dataset label')
    return args


if __name__=='__main__':
    arguments = parse_args()
    lock_dir = arguments.dataset_dir/'GPUIndex'/'TensorVS_MultiGPU3'
    lock_dir.mkdir(parents=True,exist_ok=True)
    lock_path = lock_dir/('gpus'+'-'.join(map(str,sorted(arguments.gpus)))+'.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'A three-GPU benchmark already owns these GPUs: {lock_path}')
        benchmark(arguments)
