"""Native cuVS/Faiss worker; run in the library's existing Conda environment.

This file intentionally has no Torch/TensorVS dependency. CUDA input/output
buffers use the public CUDA array interface and Faiss's public pointer API.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import gc
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import traceback

import numpy as np
from _sift100m_library_protocol import (send_frame, receive_frame, read_binary_matrix,
    pack_results, read_cuvs_flat_list_sizes)


class CudaRuntime:
    def __init__(self):
        paths = sorted((Path(sys.prefix)/'lib').glob('libcudart.so*'))
        if not paths:
            raise RuntimeError('The library environment has no CUDA runtime')
        self.lib = ct.CDLL(str(paths[0]))
        self.lib.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
        self.lib.cudaFree.argtypes = [ct.c_void_p]
        self.lib.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
        self.lib.cudaSetDevice.argtypes = [ct.c_int]
        self.lib.cudaGetErrorString.argtypes = [ct.c_int]
        self.lib.cudaGetErrorString.restype = ct.c_char_p
        self.check(self.lib.cudaSetDevice(0))
        version = ct.c_int()
        self.lib.cudaRuntimeGetVersion.argtypes = [ct.POINTER(ct.c_int)]
        self.check(self.lib.cudaRuntimeGetVersion(ct.byref(version)))
        self.version = version.value
        self.library_path = str(paths[0])

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.cudaGetErrorString(status).decode())

    def synchronize(self):
        self.check(self.lib.cudaDeviceSynchronize())

    def memory_info(self):
        free, total = ct.c_size_t(), ct.c_size_t()
        self.lib.cudaMemGetInfo.argtypes = [ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)]
        self.check(self.lib.cudaMemGetInfo(ct.byref(free),ct.byref(total)))
        return dict(free_bytes=free.value,total_bytes=total.value,
                    used_bytes=total.value-free.value)

    def array(self, shape, dtype, host=None):
        result = CudaArray(self, shape, dtype)
        if host is not None:
            result.upload(host)
        return result

    def to_host(self, value):
        if hasattr(value, 'copy_to_host'):
            return np.asarray(value.copy_to_host())
        interface = value.__cuda_array_interface__
        result = np.empty(interface['shape'], dtype=np.dtype(interface['typestr']))
        self.check(self.lib.cudaMemcpy(result.ctypes.data, interface['data'][0], result.nbytes, 2))
        return result


class CudaArray:
    def __init__(self, runtime, shape, dtype):
        self.runtime = runtime
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.nbytes = math.prod(self.shape)*self.dtype.itemsize
        self.pointer = ct.c_void_p()
        runtime.check(runtime.lib.cudaMalloc(ct.byref(self.pointer), self.nbytes))

    @property
    def __cuda_array_interface__(self):
        return dict(shape=self.shape, strides=None, typestr=self.dtype.str,
                    data=(self.pointer.value, False), version=3, stream=1)

    def upload(self, array):
        array = np.asarray(array, dtype=self.dtype, order='C')
        if array.shape != self.shape:
            raise ValueError('CUDA upload shape mismatch')
        self.runtime.check(self.runtime.lib.cudaMemcpy(self.pointer, array.ctypes.data, self.nbytes, 1))

    def download(self):
        result = np.empty(self.shape, dtype=self.dtype)
        self.runtime.check(self.runtime.lib.cudaMemcpy(result.ctypes.data, self.pointer, self.nbytes, 2))
        return result

    def close(self):
        if self.pointer.value:
            self.runtime.check(self.runtime.lib.cudaFree(self.pointer))
            self.pointer = ct.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CuVSBackend:
    def __init__(self, config, runtime):
        import cuvs
        from cuvs.common import Resources
        from cuvs.neighbors import ivf_flat, ivf_pq
        self.module = ivf_flat if config['kind'] == 'ivfflat' else ivf_pq
        self.config, self.runtime = config, runtime
        self.version = getattr(cuvs, '__version__', 'unknown')
        self.index = None
        self.resource_type = Resources
        self.resources = Resources()
        self.flat_counts = None
        self.occupancy_error = None

    def build(self, base):
        c = self.config
        preparation_started = time.perf_counter()
        options = dict(n_lists=c['nlist'], metric='sqeuclidean',
            kmeans_n_iters=c['coarse_iters'], kmeans_trainset_fraction=c['train_fraction'],
            add_data_on_build=False, conservative_memory_allocation=True)
        if c['kind'] == 'ivfflat':
            options['adaptive_centers'] = False
            # The installed Flat C API requires a device matrix. Reproduce
            # its regular-stride training sample first, train-only with all
            # sampled rows, then release that GPU matrix before bulk add.
            target = max(int(c['train_fraction']*len(base)), c['nlist'])
            ratio = max(1, len(base)//target)
            train_rows = len(base)//ratio
            train = self.runtime.array((train_rows,128), np.uint8,
                                       base[::ratio][:train_rows])
            options['kmeans_trainset_fraction'] = 1.0
            input_data = train
        else:
            options.update(pq_dim=c['pq_dim'], pq_bits=c['pq_bits'], codebook_kind='subspace',
                force_random_rotation=False, max_train_points_per_pq_code=c['pq_train_points'])
            input_data = base
        params = self.module.IndexParams(**options)
        input_prepare_seconds = time.perf_counter()-preparation_started
        started = time.perf_counter()
        training_resources = self.resource_type()
        self.index = self.module.build(params, input_data, resources=training_resources)
        training_resources.sync()
        self.runtime.synchronize()
        if c['kind']=='ivfflat':
            train.close()
            del train, input_data
        del training_resources
        gc.collect()
        self.runtime.synchronize()
        after_train_memory = self.runtime.memory_info()
        print(f'After training/input release: {after_train_memory}',flush=True)
        checkpoint_seconds = 0.0
        if c['kind']=='ivfflat':
            # Keep only the trained state across the training/add boundary.
            # Native serialization is lossless FP32 and contains zero rows
            # here. Unloading the old index also releases any allocator state
            # retained by training before the large inverted lists are grown.
            checkpoint_started = time.perf_counter()
            centers_before = self.runtime.to_host(self.index.centers)
            with tempfile.TemporaryDirectory(prefix='cuvs-trained-flat-') as temporary:
                checkpoint = Path(temporary)/'trained.bin'
                self.module.save(str(checkpoint),self.index,include_dataset=True,resources=self.resources)
                self.resources.sync()
                counts = read_cuvs_flat_list_sizes(checkpoint,0,c['nlist'],dtype='uint8')
                if counts.any():
                    raise ValueError('Training checkpoint unexpectedly contains vectors')
                self.index = None
                gc.collect()
                self.runtime.synchronize()
                self.index = self.module.load(str(checkpoint),resources=self.resources)
                self.resources.sync()
            centers_after = self.runtime.to_host(self.index.centers)
            if centers_before.dtype != centers_after.dtype or centers_before.tobytes()!=centers_after.tobytes():
                raise ValueError('Native trained checkpoint changed coarse centroids')
            checkpoint_seconds = time.perf_counter()-checkpoint_started
            print(f'After trained-state reload: {self.runtime.memory_info()}',flush=True)
        train_seconds = time.perf_counter()-started
        started = time.perf_counter()
        add_memory = []
        for start in range(0, len(base), c['add_batch_size']):
            end = min(start+c['add_batch_size'], len(base))
            ids = np.arange(start, end, dtype=np.int64)
            gpu_vectors = self.runtime.array((end-start,128),np.uint8 if c['kind']=='ivfflat' else np.float32,base[start:end])
            gpu_ids = self.runtime.array(ids.shape,np.int64,ids)
            self.index = self.module.extend(self.index, gpu_vectors, gpu_ids, resources=self.resources)
            self.resources.sync()
            self.runtime.synchronize()
            gpu_vectors.close()
            gpu_ids.close()
            if start==0 or (start//c['add_batch_size'])%32==0 or end==len(base):
                snapshot = dict(added_rows=end,**self.runtime.memory_info())
                add_memory.append(snapshot)
                print(f'Add progress: {snapshot}',flush=True)
        self.runtime.synchronize()
        return dict(train_seconds=train_seconds, add_seconds=time.perf_counter()-started,
            input_prepare_seconds=input_prepare_seconds,
            native_build_options=options, actual_coarse_iterations=c['coarse_iters'],
            actual_pq_iterations=c['coarse_iters'] if c['kind']=='ivfpq' else None,
            coarse_train_target_fraction=c['train_fraction'],
            external_regular_train_rows=train_rows if c['kind']=='ivfflat' else None,
            external_regular_train_stride=ratio if c['kind']=='ivfflat' else None,
            resource_policy='dedicated training Resources released before add; one persistent Resources for add/search',
            device_memory_after_training=after_train_memory,
            trained_checkpoint_seconds=checkpoint_seconds,
            trained_checkpoint_centroids_bitwise_verified=c['kind']=='ivfflat',
            add_device_memory_snapshots=add_memory,
            pq_iteration_note='cuVS exposes one kmeans_n_iters for both quantizers')

    def save(self, path):
        self.module.save(str(path), self.index, include_dataset=True, resources=self.resources)
        self.resources.sync()
        self.runtime.synchronize()
        self.read_occupancy(path)

    def load(self, path):
        self.index = self.module.load(str(path), resources=self.resources)
        self.resources.sync()
        self.runtime.synchronize()
        self.read_occupancy(path)

    def read_occupancy(self, path):
        if self.config['kind']=='ivfflat':
            rank = self.config['gpus'].index(int(os.environ['CUDA_VISIBLE_DEVICES']))
            start,end = self.config['bounds'][rank]
            try:
                self.flat_counts = read_cuvs_flat_list_sizes(path,end-start,self.config['nlist'],dtype='uint8')
            except (ValueError, EOFError, OSError) as error:
                self.occupancy_error = repr(error)

    def configure(self, probe):
        options = dict(n_probes=probe)
        if self.config['kind'] == 'ivfpq':
            options.update(lut_dtype=np.float16 if self.config['lut_dtype']=='fp16' else np.float32,
                           internal_distance_dtype=np.float32, coarse_search_dtype=np.float32)
        self.params = self.module.SearchParams(**options)

    def search(self, query, distances, ids, k):
        self.module.search(self.params, self.index, query, k, neighbors=ids, distances=distances,
                           resources=self.resources)
        self.resources.sync()
        self.runtime.synchronize()

    def list_sizes(self):
        if self.flat_counts is not None:
            return self.flat_counts
        if hasattr(self.index, 'list_sizes'):
            return self.runtime.to_host(self.index.list_sizes).astype(np.int64).reshape(-1)
        return None

    def visits(self, queries, probe, expected_rows):
        counts = self.list_sizes()
        if counts is None:
            if probe == self.config['nlist']:
                return dict(visited_vectors_per_query=float(expected_rows), source='all IVF lists probed')
            return dict(visited_vectors_per_query=None,
                        unavailable_reason='No supported native occupancy reader; not estimated',
                        occupancy_error=self.occupancy_error)
        if len(counts) != self.config['nlist'] or int(counts.sum()) != expected_rows:
            raise ValueError('Native cuVS list occupancy disagrees with the shard size')
        from cuvs.neighbors import brute_force
        centers = self.index.centers_rot if self.config['kind']=='ivfpq' else self.index.centers
        coarse = brute_force.build(centers, metric='sqeuclidean', resources=self.resources)
        q = self.runtime.array(queries.shape, np.float32, queries)
        out_d = self.runtime.array((len(queries), probe), np.float32)
        out_i = self.runtime.array((len(queries), probe), np.int64)
        brute_force.search(coarse, q, probe, neighbors=out_i, distances=out_d,resources=self.resources)
        self.resources.sync()
        labels = out_i.download()
        if (labels < 0).any() or (labels >= len(counts)).any():
            raise ValueError('Invalid coarse-list IDs')
        visited = counts[labels].sum(axis=1)
        return dict(visited_vectors_per_query=float(visited.mean()),
                    minimum=int(visited.min()), maximum=int(visited.max()),
                    source='native list occupancy and cuVS brute-force nearest coarse centers')


class FaissBackend:
    def __init__(self, config, runtime):
        import faiss
        if not hasattr(faiss, 'GpuIndexIVFFlat'):
            raise RuntimeError('The deployed Faiss is not a GPU build')
        self.faiss, self.config, self.runtime = faiss, config, runtime
        faiss.omp_set_num_threads(config['cpu_threads'])
        self.version = faiss.__version__
        self.resources = faiss.StandardGpuResources()
        self.resources.setTempMemory(config['faiss_temp_memory_mib']*(1<<20))
        self.resources.setDefaultNullStreamAllDevices()
        self.index = None

    def new_index(self):
        f, c = self.faiss, self.config
        if c.get('faiss_sq8_direct'):
            params = f.GpuIndexIVFScalarQuantizerConfig()
            params.device = 0
            params.indicesOptions = f.INDICES_64_BIT
            params.use_cuvs = False
            index = f.GpuIndexIVFScalarQuantizer(self.resources, 128, c['nlist'],
                f.ScalarQuantizer.QT_8bit_direct, f.METRIC_L2, False, params)
            index.cp.niter, index.cp.nredo, index.cp.seed = c['coarse_iters'], 1, c['seed']
            return index
        if c['kind'] == 'ivfflat':
            params = f.GpuIndexIVFFlatConfig()
            params.useFloat16 = False
        else:
            params = f.GpuIndexIVFPQConfig()
            params.useFloat16LookupTables = c['lut_dtype']=='fp16'
        params.device, params.indicesOptions, params.use_cuvs = 0, f.INDICES_64_BIT, False
        if c['kind'] == 'ivfflat':
            index = f.GpuIndexIVFFlat(self.resources, 128, c['nlist'], f.METRIC_L2, params)
        else:
            index = f.GpuIndexIVFPQ(self.resources, 128, c['nlist'], c['pq_dim'], c['pq_bits'], f.METRIC_L2, params)
            index.pq.cp.niter, index.pq.cp.nredo = c['pq_iters'], c['pq_restarts']
            index.pq.cp.max_points_per_centroid, index.pq.cp.seed = c['pq_train_points'], c['seed']
        index.cp.niter, index.cp.nredo, index.cp.seed = c['coarse_iters'], 1, c['seed']
        return index

    def build(self, base):
        c = self.config
        self.index = self.new_index()
        self.index.cp.max_points_per_centroid = math.ceil(len(base)*c['train_fraction']/c['nlist'])
        started = time.perf_counter()
        self.index.train(base)
        self.runtime.synchronize()
        train_seconds = time.perf_counter()-started
        started = time.perf_counter()
        reserved = hasattr(self.index,'reserveMemory')
        if reserved:
            self.index.reserveMemory(len(base))
        for start in range(0, len(base), c['add_batch_size']):
            self.index.add(base[start:start+c['add_batch_size']])
        reclaimed = hasattr(self.index,'reclaimMemory')
        if reclaimed:
            self.index.reclaimMemory()
        self.runtime.synchronize()
        if int(self.index.ntotal) != len(base):
            raise ValueError('Native Faiss did not add every shard vector')
        return dict(train_seconds=train_seconds, add_seconds=time.perf_counter()-started,
            actual_coarse_iterations=c['coarse_iters'],
            actual_pq_iterations=c['pq_iters'] if c['kind']=='ivfpq' else None,
            use_cuvs=False, indices='INDICES_64_BIT', metric='squared L2',
            max_points_per_coarse_centroid=int(self.index.cp.max_points_per_centroid),
            faiss_scratch_mib=c['faiss_temp_memory_mib'],
            reserved_rows=len(base) if reserved else None,
            reclaimed_after_add=reclaimed,
            lookup_table_dtype=c['lut_dtype'] if c['kind']=='ivfpq' else None)

    def save(self, path):
        cpu = self.faiss.index_gpu_to_cpu(self.index)
        self.check_storage(cpu)
        self.faiss.write_index(cpu, str(path))
        del cpu

    def check_storage(self, cpu):
        if self.config.get('faiss_sq8_direct') and (
            not isinstance(cpu, self.faiss.IndexIVFScalarQuantizer)
            or cpu.by_residual
            or cpu.sq.qtype != self.faiss.ScalarQuantizer.QT_8bit_direct
            or cpu.code_size != cpu.d
        ):
            raise ValueError('Expected lossless direct UInt8 storage in the Faiss index')

    def load(self, path):
        cpu = self.faiss.read_index(str(path))
        self.check_storage(cpu)
        options = self.faiss.GpuClonerOptions()
        options.use_cuvs = False
        options.indicesOptions = self.faiss.INDICES_64_BIT
        options.useFloat16 = self.config['kind']=='ivfpq' and self.config['lut_dtype']=='fp16'
        self.index = self.faiss.index_cpu_to_gpu(self.resources, 0, cpu, options)
        del cpu
        self.runtime.synchronize()

    def configure(self, probe):
        self.index.nprobe = probe

    def search(self, query, distances, ids, k):
        f = self.faiss
        self.index.search_c(query.shape[0], f.cast_integer_to_float_ptr(query.pointer.value), k,
            f.cast_integer_to_float_ptr(distances.pointer.value), f.cast_integer_to_idx_t_ptr(ids.pointer.value))
        self.runtime.synchronize()

    def visits(self, queries, probe, expected_rows):
        counts = np.array([self.index.getListLength(i) for i in range(self.config['nlist'])], dtype=np.int64)
        if int(counts.sum()) != expected_rows:
            raise ValueError('Faiss list occupancy disagrees with the shard size')
        quantizer = self.faiss.downcast_index(self.index.quantizer)
        _, labels = quantizer.search(queries, probe)
        visited = counts[labels].sum(axis=1)
        return dict(visited_vectors_per_query=float(visited.mean()), minimum=int(visited.min()),
                    maximum=int(visited.max()), source='native Faiss getListLength and index quantizer search')


def release_host():
    gc.collect()
    try:
        ct.CDLL(None).malloc_trim(0)
    except AttributeError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rank = args.rank
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(args.socket)
    send_frame(connection, dict(state='connected', rank=rank, pid=os.getpid()))
    try:
        runtime = CudaRuntime()
        backend = (CuVSBackend if config['library']=='cuvs' else FaissBackend)(config, runtime)
        start_id, end_id = config['bounds'][rank]
        index_path = Path(config['index_paths'][rank])
        base = None
        batches = []
        send_frame(connection, dict(state='ready', rank=rank, pid=os.getpid(),
            library_version=backend.version, python=sys.version, environment=sys.prefix,
            cuda_runtime_version=runtime.version, cuda_runtime_library=runtime.library_path,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        while True:
            message, payload = receive_frame(connection)
            action = message['action']
            if action == 'stop':
                return
            if action == 'read':
                started = time.perf_counter()
                matrix = read_binary_matrix(Path(config['dataset_dir'])/'base.bin', 'u1')
                input_dtype = np.uint8 if config['library']=='cuvs' and config['kind']=='ivfflat' else np.float32
                base = matrix[start_id:end_id].astype(input_dtype)
                del matrix
                send_frame(connection, dict(state='read', read_seconds=time.perf_counter()-started))
            elif action == 'build':
                started = time.perf_counter()
                metrics = backend.build(base)
                runtime.synchronize()
                metrics.update(build_seconds=time.perf_counter()-started, rank=rank,
                               gpu=config['gpus'][rank], n_vectors=end_id-start_id)
                send_frame(connection, dict(state='built', metrics=metrics))
            elif action == 'save':
                index_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = index_path.with_name(index_path.name+'.partial')
                if temporary.exists():
                    raise FileExistsError(temporary)
                started = time.perf_counter()
                backend.save(temporary)
                temporary.replace(index_path)
                metrics.update(save_seconds=time.perf_counter()-started,
                               index_size_bytes=index_path.stat().st_size)
                del base
                base = None
                release_host()
                send_frame(connection, dict(state='saved', metrics=metrics))
            elif action == 'load':
                backend.load(index_path)
                send_frame(connection, dict(state='loaded'))
            elif action == 'configure':
                batches.clear()
                release_host()
                probe, q_batch = message['nprobe'], message['q_batch']
                backend.configure(probe)
                queries = read_binary_matrix(Path(config['dataset_dir'])/'query.bin', 'u1')[:config['actual_nq']].astype(np.float32)
                visits = backend.visits(queries, probe, end_id-start_id)
                for start in range(0, len(queries), q_batch):
                    valid = min(q_batch, len(queries)-start)
                    query_dtype = np.uint8 if config['library']=='cuvs' and config['kind']=='ivfflat' else np.float32
                    padded = np.zeros((q_batch, 128), dtype=query_dtype)
                    padded[:valid] = queries[start:start+valid]
                    q = runtime.array(padded.shape, query_dtype,
                        padded if config['query_input']=='resident-gpu' else None)
                    d = runtime.array((q_batch, config['topk']), np.float32)
                    ids = runtime.array((q_batch, config['topk']), np.int64)
                    batches.append((q, d, ids, valid))
                runtime.synchronize()
                send_frame(connection, dict(state='configured', visits=visits))
            elif action == 'search':
                q, d, ids, valid = batches[message['batch']]
                if config['query_input']=='host':
                    host = np.frombuffer(payload, dtype='<f4').reshape(q.shape)
                    q.upload(host)
                elif payload:
                    raise ValueError('Unexpected resident-query payload')
                backend.search(q, d, ids, config['topk'])
                host_d = d.download()[:valid]
                host_ids = ids.download()[:valid]
                if np.any((host_ids>=0) & ~np.isfinite(host_d)):
                    raise FloatingPointError(f'Native {config["library"]} returned non-finite scores '
                        f'with LUT={config["lut_dtype"]}; raw SIFT inputs may overflow FP16 LUTs. '
                        'Use --lut-dtype fp32 for the original 0-255 input range.')
                # cuVS may use an unsigned -1 sentinel; preserve invalid IDs.
                host_ids = np.where(host_ids >= 0, host_ids+start_id, -1).astype(np.int64)
                metadata, data = pack_results(host_d, host_ids)
                send_frame(connection, metadata, data)
            elif action == 'synchronize':
                runtime.synchronize()
                send_frame(connection, dict(state='synchronized'))
            else:
                raise ValueError(f'Unknown native-worker command: {action}')
    except BaseException:
        try:
            send_frame(connection, dict(error=traceback.format_exc(), rank=rank))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


if __name__ == '__main__':
    main()
