"""Small version-independent wire protocol for Python 3.11/3.12 workers."""
from __future__ import annotations
import json
import struct

MAX_FRAME = 16 << 20


def send_frame(connection, metadata, payload=b''):
    header = json.dumps(metadata, allow_nan=False).encode()
    if len(header) > MAX_FRAME or len(payload) > MAX_FRAME:
        raise ValueError('Native-worker frame exceeds its size limit')
    connection.sendall(struct.pack('!II', len(header), len(payload)) + header + payload)


def _read_exact(connection, count):
    chunks = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError('Native worker disconnected')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def receive_frame(connection):
    header_size, payload_size = struct.unpack('!II', _read_exact(connection, 8))
    if header_size > MAX_FRAME or payload_size > MAX_FRAME:
        raise ValueError('Native-worker frame exceeds its size limit')
    metadata = json.loads(_read_exact(connection, header_size))
    payload = _read_exact(connection, payload_size)
    return metadata, payload


def read_binary_matrix(path, dtype):
    import numpy as np
    from pathlib import Path
    path = Path(path)
    header = np.fromfile(path, dtype='<u4', count=2)
    if len(header) != 2 or min(header) <= 0:
        raise ValueError(f'Invalid binary matrix header: {path}')
    shape = tuple(map(int, header))
    if path.stat().st_size != 8 + shape[0]*shape[1]*np.dtype(dtype).itemsize:
        raise ValueError(f'Binary matrix payload length disagrees with its header: {path}')
    return np.memmap(path, mode='c', offset=8, dtype=dtype, shape=shape)


def pack_results(distances, ids):
    import numpy as np
    distances = np.asarray(distances, dtype='<f4', order='C')
    ids = np.asarray(ids, dtype='<i8', order='C')
    if distances.ndim != 2 or ids.shape != distances.shape:
        raise ValueError('Distances and IDs must have the same matrix shape')
    return dict(rows=len(ids), k=ids.shape[1]), distances.tobytes() + ids.tobytes()


def unpack_results(metadata, payload):
    import numpy as np
    shape = (int(metadata['rows']), int(metadata['k']))
    count = shape[0]*shape[1]
    if len(payload) != count*12:
        raise ValueError('Native result payload has an invalid length')
    distances = np.frombuffer(payload, dtype='<f4', count=count).reshape(shape).copy()
    ids = np.frombuffer(payload, dtype='<i8', count=count, offset=count*4).reshape(shape).copy()
    return dict(distances=distances, ids=ids)


def read_cuvs_flat_list_sizes(path, expected_rows, nlist, dimension=128, dtype='float32'):
    """Read actual list counts from cuVS v4's documented RAFT/NumPy stream.

    Source: cuVS cpp/src/neighbors/ivf_flat/ivf_flat_serialize.cuh. Only the
    small prefix is read; vector payloads are never loaded by this helper.
    Unknown formats fail closed rather than supplying inferred occupancy.
    """
    import numpy as np
    with open(path, 'rb') as stream:
        prefix = stream.read(4).rstrip(b'\0').decode()
        if np.dtype(prefix) != np.dtype(dtype):
            raise ValueError(f'Expected a native {dtype} cuVS IVF-Flat index')
        def scalar():
            return np.load(stream, allow_pickle=False).item()
        version = int(scalar())
        if version != 4:
            raise ValueError(f'Unsupported cuVS IVF-Flat serialization version: {version}')
        rows, dim, lists = int(scalar()), int(scalar()), int(scalar())
        scalar()  # Native distance metric enum.
        scalar()  # adaptive_centers.
        scalar()  # conservative_memory_allocation.
        if (rows, dim, lists) != (expected_rows, dimension, nlist):
            raise ValueError('Native cuVS header does not match the configured shard')
        centers = np.load(stream, allow_pickle=False)
        if centers.shape != (nlist, dimension) or centers.dtype != np.float32:
            raise ValueError('Unexpected native coarse-center matrix')
        has_norms = bool(scalar())
        if has_norms:
            norms = np.load(stream, allow_pickle=False)
            if norms.shape != (nlist,):
                raise ValueError('Unexpected native center-norm shape')
        counts = np.load(stream, allow_pickle=False)
        if counts.shape != (nlist,) or counts.dtype.kind != 'u' or int(counts.sum()) != expected_rows:
            raise ValueError('Native cuVS list occupancy is inconsistent')
        return counts.astype(np.int64)
