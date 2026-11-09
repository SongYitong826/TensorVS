"""Query all three saved SIFT100M IVF-PQ shards sequentially on one GPU."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

from _sift100m_library_protocol import read_binary_matrix, receive_frame, send_frame

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path('/storage/vector_data/sift100M')
METHODS = {'tensorvs': 'TensorVS_Triton', 'cuvs': 'cuVS', 'faiss': 'FaissGPU'}
BOUNDS = [(0, 33333333), (33333333, 66666666), (66666666, 100000000)]


def worker(endpoint):
    import numpy as np

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(endpoint)
        try:
            config, _ = receive_frame(connection)
            method, lut_dtype = config['method'], config['lut_dtype']
            query = read_binary_matrix(DATASET/'query.bin', 'u1')[:config['nq']].astype(np.float32)
            truth = read_binary_matrix(DATASET/'groundtruth.bin', '<u4')[:config['nq'], :config['topk']]
            topk = config['topk']
            if method == 'tensorvs':
                import torch
                from tensorvs import load_ivfpq_index, search_ivfpq_index
                from _benchmark_utils import probe_visit_statistics

                torch.set_num_threads(config['cpu_threads'])
                torch.set_grad_enabled(False)
                device = torch.device('cuda:0')
                version = str(torch.__version__)

                def load(path, count):
                    index = load_ivfpq_index(path, load_lists=True)
                    if int(index.metadata['num_vectors']) != count or len(index.list_count) != 1024:
                        raise ValueError('TensorVS shard dimensions do not match the saved run')
                    return index.reside_on_device(device, backend='triton', persist_layout=False)

                def prepare(padded):
                    return torch.from_numpy(padded).to(device)

                def search(index, q, probe, valid, *, use_packed_codes=True):
                    with torch.inference_mode():
                        distances, ids = search_ivfpq_index(
                            index, q, topk=topk, nprobe=probe, device=device, backend='triton',
                            persist_layout=False, lut_dtype=lut_dtype, use_preallocated_workspace=True,
                            use_cuda_graph=True, triton_autotune=True, use_packed_16b_codes=use_packed_codes)
                        return distances[:valid].cpu().numpy(), ids[:valid].cpu().numpy()

                def visits(index, probe, count):
                    return probe_visit_statistics(index, torch.from_numpy(query), probe, device, 16)
            else:
                from _sift100m_library_worker import CudaRuntime, CuVSBackend, FaissBackend

                runtime = CudaRuntime()
                backend_class = CuVSBackend if method == 'cuvs' else FaissBackend
                backend_config = dict(kind='ivfpq', lut_dtype=lut_dtype, nlist=1024,
                                      cpu_threads=config['cpu_threads'], faiss_temp_memory_mib=128)

                def load(path, count):
                    backend = backend_class(backend_config, runtime)
                    backend.load(path)
                    return backend

                def prepare(padded):
                    shape = (len(padded), topk)
                    return (runtime.array(padded.shape, np.float32, padded),
                            runtime.array(shape, np.float32), runtime.array(shape, np.int64))

                def search(index, buffers, probe, valid):
                    q, distances, ids = buffers
                    index.search(q, distances, ids, topk)
                    return distances.download()[:valid], ids.download()[:valid]

                def visits(index, probe, count):
                    return index.visits(query, probe, count)

            indexes = []
            for rank, path in enumerate(config['paths']):
                print(f'Loading shard {rank} on the same GPU: {path}', flush=True)
                first, last = BOUNDS[rank]
                indexes.append(load(path, last-first))
            if method != 'tensorvs':
                version = indexes[0].version
            send_frame(connection, dict(pid=os.getpid(), library_version=version, python=sys.version,
                                        environment=sys.prefix))
            batches = []
            batch_size = None
            visit_cache = {}
            while True:
                request, _ = receive_frame(connection)
                action = request['action']
                if action == 'stop':
                    return
                if action == 'configure':
                    size, probe = request['q_batch'], request['nprobe']
                    if size != batch_size:
                        batches.clear()
                        for offset in range(0, len(query), size):
                            part = query[offset:offset+size]
                            padded = np.empty((size, 128), dtype=np.float32)
                            padded[:len(part)] = part
                            padded[len(part):] = part[-1]
                            batches.append((prepare(padded), len(part)))
                        batch_size = size
                    if method != 'tensorvs':
                        for index in indexes:
                            index.configure(probe)
                    if probe not in visit_cache:
                        visit_cache[probe] = [visits(index, probe, last-first)
                                              for index, (first, last) in zip(indexes, BOUNDS)]
                    send_frame(connection, dict(visits=visit_cache[probe]))
                    continue
                if action not in ('timed', 'memory'):
                    raise ValueError(f'Unknown action: {action}')

                def sweep(*, original=False):
                    begin = time.perf_counter()
                    all_ids = []
                    all_distances = []
                    for batch, valid in batches:
                        shard_distances, shard_ids = [], []
                        for index, (offset, _) in zip(indexes, BOUNDS):
                            if original:
                                distances, ids = search(index, batch, probe, valid, use_packed_codes=False)
                            else:
                                distances, ids = search(index, batch, probe, valid)
                            shard_distances.append(distances)
                            shard_ids.append(np.where(ids >= 0, ids+offset, -1))
                        distances = np.concatenate(shard_distances, axis=1)
                        ids = np.concatenate(shard_ids, axis=1)
                        distances[ids < 0] = np.inf
                        order = np.argsort(distances, axis=1, kind='stable')[:, :topk]
                        all_ids.append(np.take_along_axis(ids, order, axis=1))
                        all_distances.append(np.take_along_axis(distances, order, axis=1))
                    merged = np.concatenate(all_ids)
                    return time.perf_counter()-begin, merged, all_distances

                if action == 'memory':
                    sweep()
                    send_frame(connection, {})
                    continue
                reference = sweep(original=True) if config.get('verify_original') else None
                for _ in range(config['warmup']):
                    sweep()
                samples = [sweep() for _ in range(config['repetitions'])]
                ids = samples[-1][1]
                if (ids.shape != truth.shape or (ids < 0).any() or (ids >= BOUNDS[-1][1]).any()
                        or any(not np.isfinite(values).all() for values in samples[-1][2])):
                    raise ValueError('Invalid global Top-K IDs or distances')
                recall = float(np.any(ids[:, :, None] == truth[:, None, :], axis=2).mean())
                response = dict(samples_seconds=[sample[0] for sample in samples], recall=recall)
                if reference is not None:
                    np.testing.assert_array_equal(ids, reference[1])
                    for observed, expected in zip(samples[-1][2], reference[2], strict=True):
                        np.testing.assert_array_equal(observed, expected)
                    response['identical_to_original'] = True
                send_frame(connection, response)
        except BaseException:
            send_frame(connection, dict(error=traceback.format_exc()))
            raise


def run_method(args, method, run_dir):
    from sift100m_multigpu_bench import AggregateMemorySampler, FIELDS, atomic_csv, atomic_json, fingerprint

    stem = f'CUDA_{METHODS[method]}_MultiGPU3_IVFPQ_sift100m_nlist1024_M16_nbits8_train512'
    old_json = args.result_dir/(stem+'.json')
    old = json.loads(old_json.read_text())
    if old['status'] != 'complete' or old['completed_rows'] != 42:
        raise ValueError(f'Expected a complete source experiment: {old_json}')
    previous_rows = {(row['q_batch'], row['nprobes']): row for row in old['results']}
    config = copy.deepcopy(old['config'])
    paths = [item['path'] for item in old['indexes']]
    if len(paths) != len(BOUNDS) or [fingerprint(Path(path)) for path in paths] != old['indexes']:
        raise ValueError(f'Saved index fingerprints changed: {old_json}')
    for field, name in (('queries', 'query.bin'), ('truth', 'groundtruth.bin')):
        if config[field] != fingerprint(DATASET/name):
            raise ValueError(f'Saved dataset fingerprint changed: {name}')
    config.update(gpus=[args.gpu], gpu_count=1, execution='three resident shards searched sequentially')
    settings = dict(method=method, paths=paths, cpu_threads=config['cpu_threads'],
                    lut_dtype=config['lut_dtype'], nq=config['nq'], topk=config['topk'],
                    warmup=config['warmup'], repetitions=config['repetitions'])
    if args.verify_original and method == 'tensorvs':
        settings['verify_original'] = config['verify_original'] = True
    config['worker_env'] = 'tensorvs' if method == 'tensorvs' else method
    report = dict(status='running', config=config, results=[], indexes=old['indexes'],
        expected_rows=42, completed_rows=0, memory_unit='MiB',
        timing='median full-query wall time on one GPU; three sequential shard searches, output device-to-host copies and CPU global Top-K; excludes index load, query upload, warmup and metric sweeps',
        memory_metric='sampled peak NVML process memory on one GPU, including all three resident shards',
        query_transfer='pre-uploaded queries; final batch padded with the last valid query',
        nlist_semantics=old['nlist_semantics'], probe_semantics=old['probe_semantics'],
        result_filename_note='Legacy filename retained; measured query used one GPU',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    csv_path, json_path = run_dir/(stem+'.csv'), run_dir/(stem+'.json')
    with tempfile.TemporaryDirectory(prefix='sift-onegpu-') as directory:
        endpoint = str(Path(directory)/'worker.sock')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener, (run_dir/f'{method}.log').open('w') as log:
            listener.bind(endpoint)
            listener.listen(1)
            listener.settimeout(1)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            child = subprocess.Popen([str(args.conda), 'run', '--no-capture-output', '-n',
                config['worker_env'], 'python', '-u', str(Path(__file__).resolve()), '--worker', endpoint],
                stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                deadline = time.monotonic()+120
                while True:
                    try:
                        connection, _ = listener.accept()
                        break
                    except TimeoutError:
                        if child.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(f'Worker did not connect. See {log.name}')
                with connection:
                    connection.settimeout(600)

                    def read_response():
                        response, _ = receive_frame(connection)
                        if 'error' in response:
                            raise RuntimeError(response['error'])
                        return response

                    def call(action, **options):
                        send_frame(connection, dict(action=action, **options))
                        return read_response()

                    send_frame(connection, settings)
                    report['worker'] = read_response()
                    report['software'] = [report['worker']]
                    for batch_size in config['q_batches']:
                        for probe in config['nprobes']:
                            visits = call('configure', q_batch=batch_size, nprobe=probe)['visits']
                            timed = call('timed')
                            with AggregateMemorySampler([args.gpu], [report['worker']['pid']]) as sampler:
                                call('memory')
                            qps = config['nq']/statistics.median(timed['samples_seconds'])
                            row = dict(nprobes=probe, q_batch=batch_size, K=config['topk'], qps=qps,
                                recall=timed['recall'],
                                visited_vectors_per_query=sum(v['visited_vectors_per_query'] for v in visits),
                                peak_gpu_memory=sampler.metrics()['peak_total_mib'])
                            if row['peak_gpu_memory'] <= 0 or row['qps'] <= 0:
                                raise ValueError('Invalid query measurement')
                            original = previous_rows[batch_size, probe]
                            if abs(row['recall']-original['recall']) > 0.005:
                                raise ValueError(f'Recall changed by more than 0.5 points at {batch_size=}, {probe=}')
                            row_record = dict(row, samples_seconds=timed['samples_seconds'],
                                              memory=sampler.metrics(), visit_statistics_per_shard=visits)
                            if settings.get('verify_original'):
                                row_record['identical_to_original'] = timed['identical_to_original']
                            report['results'].append(row_record)
                            report['completed_rows'] = len(report['results'])
                            atomic_json(json_path, report)
                            atomic_csv(csv_path, report['results'], FIELDS)
                            print(f'{method} {report["completed_rows"]}/42 QBatch={batch_size} '
                                  f'nprobe={probe} QPS={qps:.2f} recall={row["recall"]:.6f}', flush=True)
                    if report['completed_rows'] != report['expected_rows']:
                        raise ValueError('Incomplete result')
                    send_frame(connection, dict(action='stop'))
                if child.wait(timeout=60) != 0:
                    raise RuntimeError(f'Worker did not exit cleanly. See {log.name}')
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
    report['status'] = 'complete'
    atomic_json(json_path, report)
    return csv_path, json_path


def publish(args, run_dir, files):
    archive = args.result_dir/'Archive'/f'{run_dir.name}_previous'
    archive.mkdir(parents=True, exist_ok=False)
    for source in files:
        shutil.copy2(args.result_dir/source.name, archive/source.name)
    staged = []
    try:
        for source in files:
            target = args.result_dir/source.name
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f'.{target.name}.', delete=False) as file:
                temp = Path(file.name)
            staged.append((temp, target))
            shutil.copy2(source, temp)
        for temp, target in staged:
            temp.replace(target)
    finally:
        for temp, _ in staged:
            temp.unlink(missing_ok=True)
    (run_dir/'published.json').write_text(json.dumps(dict(archive=str(archive),
        updated=[str(args.result_dir/file.name) for file in files]), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--result-dir', type=Path, default=ROOT/'result')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--conda', type=Path, default=Path('/home/songyitong/miniconda3/bin/conda'))
    parser.add_argument('--method', choices=METHODS, help='Run only this method')
    parser.add_argument('--verify-original', action='store_true',
                        help='Compare TensorVS results against an untimed original scan')
    parser.add_argument('--replace-existing', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    # NVML receives physical GPU IDs in the coordinator. Only the child sees
    # CUDA_VISIBLE_DEVICES restricted to the selected single GPU.
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    logs = args.result_dir/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir or Path(tempfile.mkdtemp(prefix='sift100m_ivfpq_single_gpu_', dir=logs))
    run_dir.mkdir(parents=True, exist_ok=True)
    status = run_dir/'status'
    status.write_text('running\n')
    lock_file = DATASET/'GPUIndex/TensorVS_MultiGPU3/gpus0-1-2.lock'
    try:
        with lock_file.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            files = []
            for method in ([args.method] if args.method else METHODS):
                (run_dir/'current_method').write_text(method+'\n')
                files.extend(run_method(args, method, run_dir))
            if args.replace_existing:
                publish(args, run_dir, files)
        status.write_text('complete\n')
    except BaseException:
        status.write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
