"""Cross-environment wire format and native benchmark result conventions."""
import json
import socket
import struct
import threading

import numpy as np
import pytest

from bench._sift100m_library_protocol import (send_frame, receive_frame,
    pack_results, unpack_results, read_binary_matrix, MAX_FRAME, read_cuvs_flat_list_sizes)
from bench.sift100m_library_multigpu_bench import parse_args, stem


def test_protocol_preserves_signed_ids_and_distances():
    distances = np.array([[0., 2.5], [3., float('inf')]], dtype=np.float32)
    ids = np.array([[0, 99_999_999], [42, -1]], dtype=np.int64)
    metadata, payload = pack_results(distances, ids)
    left, right = socket.socketpair()
    try:
        send_frame(left, metadata, payload)
        header, raw = receive_frame(right)
        actual = unpack_results(header, raw)
    finally:
        left.close()
        right.close()
    np.testing.assert_array_equal(actual['distances'], distances)
    np.testing.assert_array_equal(actual['ids'], ids)


def test_fragmented_socket_reads():
    left, right = socket.socketpair()
    header = json.dumps({'state':'ready','rank':2}).encode()
    frame = struct.pack('!II',len(header),3)+header+b'abc'
    def send_fragments():
        for value in frame:
            left.sendall(bytes([value]))
        left.close()
    thread = threading.Thread(target=send_fragments)
    thread.start()
    try:
        metadata, payload = receive_frame(right)
    finally:
        right.close()
        thread.join()
    assert metadata == {'state':'ready','rank':2} and payload==b'abc'


def test_truncated_result_is_rejected():
    with pytest.raises(ValueError):
        unpack_results({'rows':2,'k':10}, b'123')


def test_oversized_frame_is_rejected():
    left,right = socket.socketpair()
    try:
        left.sendall(struct.pack('!II',MAX_FRAME+1,0))
        with pytest.raises(ValueError):
            receive_frame(right)
    finally:
        left.close()
        right.close()


def test_binary_reader_preserves_uint8_values(tmp_path):
    path = tmp_path/'base.bin'
    values = np.array([[0,128,255],[21,42,250]],dtype='u1')
    with path.open('wb') as stream:
        np.array(values.shape,dtype='<u4').tofile(stream)
        values.tofile(stream)
    result = read_binary_matrix(path,'u1')
    np.testing.assert_array_equal(result,values)


def test_native_cuvs_occupancy_reader_uses_serialized_counts(tmp_path):
    path = tmp_path/'native.bin'
    with path.open('wb') as stream:
        stream.write(b'<f4\0')
        for value in (np.int32(4), np.int64(10), np.uint32(128), np.uint32(3),
                      np.int32(0), np.bool_(False), np.bool_(True),
                      np.zeros((3,128),dtype=np.float32), np.bool_(True),
                      np.zeros(3,dtype=np.float32), np.array([1,2,7],dtype=np.uint32)):
            np.save(stream,value,allow_pickle=False)
        stream.write(b'vector payload must not be read')
    np.testing.assert_array_equal(read_cuvs_flat_list_sizes(path,10,3), [1,2,7])
    with pytest.raises(ValueError):
        read_cuvs_flat_list_sizes(path,11,3)


@pytest.mark.parametrize('library,label',[('cuvs','cuVS'),('faiss','FaissGPU')])
@pytest.mark.parametrize('kind',['ivfflat','ivfpq'])
def test_native_result_names_and_defaults(library,label,kind):
    args = parse_args(['--library',library,'--kind',kind])
    assert stem(args).startswith(f'CUDA_{label}_MultiGPU3_{kind.upper()}_sift100m_nlist1024')
    assert stem(args,True).endswith('_build')
    assert args.gpus==[0,1,2] and args.q_batches==[32,128,512]
    assert args.pq_dim==16 and args.pq_bits==8 and args.topk==10
    assert args.output_dir.name=='result' and args.query_input=='resident-gpu'
    assert args.lut_dtype=='fp32'
    assert len(args.nprobes)*len(args.q_batches)==42


def test_host_input_results_get_a_distinct_filename():
    args = parse_args(['--library','cuvs','--kind','ivfpq','--query-input','host'])
    assert stem(args).endswith('_HostInput')
    assert '_HostInput' not in stem(args,True)


def test_half_lut_has_a_distinct_result_name():
    args = parse_args(['--library','cuvs','--kind','ivfpq','--lut-dtype','fp16'])
    assert stem(args).endswith('_LUTfp16')
    assert '_LUTfp16' not in stem(args,True)


@pytest.mark.parametrize('options',[
    ['--gpus','0','0','2'], ['--nprobes','2048'],
    ['--train-fraction','0'], ['--expected-vectors','123']])
def test_invalid_native_config_rejected(options):
    with pytest.raises(SystemExit):
        parse_args(['--library','cuvs','--kind','ivfflat',*options])


def test_cuvs_multi_batch_resources_on_three_free_gpus(tmp_path):
    """Exercise real native reallocation and lossless trained-state reload.

    Skip outside the configured server or while its benchmark GPUs are busy.
    The existing experiment lock prevents overlapping the full experiment.
    """
    import fcntl
    from pathlib import Path
    import subprocess
    import torch
    from bench.sift100m_library_multigpu_bench import benchmark

    environment=Path('/home/songyitong/miniconda3/envs/cuvs')
    lock_path=Path('/storage/vector_data/sift100M/GPUIndex/TensorVS_MultiGPU3/gpus0-1-2.lock')
    if not environment.is_dir() or not lock_path.is_file() or not torch.cuda.is_available():
        pytest.skip('requires the configured cuVS server and CUDA')
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'],text=True)
    free={int(i.strip()):int(v.strip()) for i,v in (line.split(',') for line in output.splitlines())}
    if any(free.get(gpu,0)<20000 for gpu in (0,1,2)):
        pytest.skip('three free benchmark GPUs are required')
    with lock_path.open('r+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pytest.skip('the full benchmark owns the GPUs')
        rng=np.random.default_rng(916)
        base=rng.integers(0,256,size=(1536,128),dtype=np.uint8)
        queries=rng.integers(0,256,size=(33,128),dtype=np.uint8)
        distances=((queries.astype(np.float32)[:,None]-base.astype(np.float32))**2).sum(-1)
        truth=np.argsort(distances,axis=1)[:,:10].astype('<u4')
        for name,data in [('base.bin',base),('query.bin',queries),('groundtruth.bin',truth)]:
            with (tmp_path/name).open('wb') as stream:
                np.array(data.shape,dtype='<u4').tofile(stream)
                data.tofile(stream)
        for kind in ('ivfflat','ivfpq'):
            cli=['--library','cuvs','--kind',kind,'--dataset-dir',str(tmp_path),
                 '--dataset-label','synthetic1536','--expected-vectors','1536',
                 '--output-dir',str(tmp_path/'result'),'--nlist','4','--pq-dim','16',
                 '--coarse-iters','2','--pq-iters','2','--pq-restarts','1',
                 '--pq-train-points','64','--add-batch-size','32','--q-batches','16',
                 '--nprobes','1','4','--nq','33','--warmup','1','--repetitions','1','--cpu-threads','1']
            args=parse_args(cli)
            benchmark(args)
            report=json.loads((tmp_path/'result'/(stem(args)+'.json')).read_text())
            assert report['status']=='complete' and report['completed_rows']==2
            assert report['results'][-1]['visited_vectors_per_query']==1536
            if kind=='ivfflat':
                assert report['results'][-1]['recall']==1.0
                build=json.loads((tmp_path/'result'/'Build'/(stem(args,True)+'.json')).read_text())
                assert all(s['trained_checkpoint_centroids_bitwise_verified'] for s in build['shards'])
            benchmark(parse_args(cli+['--phase','query','--resume']))
