"""CPU Faiss IVF baselines; Linux process RSS is sampled independently per phase."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
import numpy as np
import faiss
from ascend_faiss_cpu_ivfflat_bench import (read_vectors, write_csv, write_json, RssSampler,
    make_batches, account_visits, cpu_information, file_signature, NPROBES, MIB)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('nprobes','q_batch','K','qps','recall','visited_vectors_per_query','peak_gpu_memory')
BUILD_FIELDS = ('dataset','method','nlist','n_vectors','dim','cpu_threads','build_time_seconds',
    'train_seconds','add_seconds','serialize_seconds','index_size_bytes','index_size_mb',
    'peak_build_cpu_memory_mib')

# Hunyuan CPU baselines (ivfflat and ivfpq): 512 queries, 16 query threads and
# a ten-point nprobe sweep. Scoped to this dataset only; other baselines keep
# the shared defaults. Explicit CLI flags still take precedence.
HUNYUAN_CPU_PROFILE = {
    'nq': 512,
    'query_threads': 16,
    'nprobes': (350, 380, 400, 450, 500, 550, 600, 700, 800, 900),
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=('glove1.2m','hunyuan'))
    p.add_argument('--kind', required=True, choices=('ivfflat','ivfpq'))
    p.add_argument('--data-root', type=Path, default=Path('/storage/vector_data'))
    p.add_argument('--output-dir', type=Path, default=ROOT/'result')
    p.add_argument('--index-dir', type=Path)
    p.add_argument('--build-threads', type=int, default=64)
    p.add_argument('--query-threads', type=int, default=None,
                   help='Faiss search threads (default: 1; hunyuan: 16)')
    p.add_argument('--nlist', type=int, default=1024)
    p.add_argument('--pq-m', type=int)
    p.add_argument('--nbits', type=int, default=8)
    p.add_argument('--pq-train-points', type=int)
    p.add_argument('--coarse-niter', type=int, default=25)
    p.add_argument('--pq-niter', type=int, default=25)
    p.add_argument('--train-fraction', type=float, default=.5)
    p.add_argument('--add-batch-size', type=int, default=65536)
    p.add_argument('--nq', type=int, default=None, help='number of queries (default: 512)')
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--q-batches', type=int, nargs='+', default=[32,128,512])
    p.add_argument('--nprobes', type=int, nargs='+')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repetitions', type=int, default=3)
    a=p.parse_args()
    profile = HUNYUAN_CPU_PROFILE if a.dataset=='hunyuan' else None
    if a.nq is None: a.nq = profile['nq'] if profile else 512
    if a.query_threads is None: a.query_threads = profile['query_threads'] if profile else 1
    if a.nprobes is None and profile: a.nprobes = list(profile['nprobes'])
    if min(a.build_threads,a.query_threads,a.nlist,a.nq,a.topk,a.coarse_niter,a.pq_niter,a.add_batch_size,a.repetitions,*a.q_batches)<=0 or a.warmup<0 or not 0<a.train_fraction<=1:
        p.error('Invalid size, iteration count or training fraction')
    probes=a.nprobes or NPROBES[a.dataset]
    if min(probes)<1 or max(probes)>a.nlist: p.error('nprobes must be within nlist')
    m=a.pq_m or (32 if a.dataset=='glove1.2m' else 384)
    train_points=a.pq_train_points or (512 if a.dataset=='glove1.2m' else 256)
    if min(m,train_points,a.nbits)<1: p.error('Invalid PQ settings')
    folder=a.data_root/a.dataset
    base_path=folder/f'{a.dataset}_base.fvecs'
    query_path=folder/f'{a.dataset}_query.fvecs'
    truth_path=folder/f'{a.dataset}_groundtruth.ivecs'
    faiss.omp_set_num_threads(a.build_threads)
    stem=f'CPU_Faiss_IVF{a.kind[3:].upper()}_{a.dataset}_nlist{a.nlist}'
    if a.kind=='ivfpq': stem+=f'_M{m}_nbits{a.nbits}_train{train_points}'
    csv_path=a.output_dir/(stem+'.csv'); json_path=csv_path.with_suffix('.json')
    build_path=a.output_dir/'Build'/(stem+'_build.csv')
    index_dir=a.index_dir or folder/'CPUIndex'/'Faiss'
    index_path=index_dir/(stem+'.faiss')
    queries=read_vectors(query_path,limit=a.nq)
    truth=read_vectors(truth_path,integer=True,limit=a.nq)
    if len(queries)==0 or len(queries)!=len(truth) or truth.shape[1]<a.topk: raise ValueError('Invalid queries or truth')
    orig_dim=queries.shape[1]
    dim=math.ceil(orig_dim/m)*m if a.kind=='ivfpq' else orig_dim
    queries=np.pad(queries,((0,0),(0,dim-orig_dim))).astype(np.float32,copy=False)
    print(f'CPU Faiss {a.kind} {a.dataset} build_threads={a.build_threads} query_threads={a.query_threads} dimension={orig_dim}->{dim}',flush=True)
    loaded_index = index_path.exists()
    if loaded_index:
        if not build_path.with_suffix('.json').is_file():
            raise ValueError('Existing index lacks build metadata; cannot validate its dataset and parameters')
        build=json.loads(build_path.with_suffix('.json').read_text())
        expected=dict(dataset=a.dataset,nlist=a.nlist,original_dim=orig_dim,padded_dim=dim,
                      pq_m=m if a.kind=='ivfpq' else None,nbits=a.nbits if a.kind=='ivfpq' else None,
                      pq_train_points=train_points if a.kind=='ivfpq' else None,metric='squared_l2')
        if any(build.get(k)!=v for k,v in expected.items()) or build.get('base')!=file_signature(base_path):
            raise ValueError('Existing index build metadata does not match the dataset or parameters')
        if build.get('index_size_bytes') != index_path.stat().st_size:
            raise ValueError('Existing index size does not match build metadata')
        print(f'Loading existing index: {index_path}',flush=True)
        index=faiss.read_index(str(index_path))
        expected_type = 'IndexIVFFlat' if a.kind=='ivfflat' else 'IndexIVFPQ'
        if (type(index).__name__!=expected_type or index.d!=dim or index.nlist!=a.nlist or
                index.ntotal!=build['n_vectors'] or index.metric_type!=faiss.METRIC_L2 or not index.is_trained):
            raise ValueError('Existing index content does not match build metadata')
        if a.kind=='ivfpq' and (index.pq.M!=m or index.pq.nbits!=a.nbits):
            raise ValueError('Existing PQ settings differ')
        print(f'Preserving original build metrics (threads={build.get("cpu_threads", "unknown")})',flush=True)
    else:
        with RssSampler() as memory:
            prep_start=time.perf_counter()
            base=read_vectors(base_path)
            if base.shape[1]!=orig_dim or len(base)<a.nlist: raise ValueError('Invalid base shape')
            if dim!=orig_dim: base=np.pad(base,((0,0),(0,dim-orig_dim)))
            prep=time.perf_counter()-prep_start
            start=time.perf_counter()
            quantizer=faiss.IndexFlatL2(dim)
            index=(faiss.IndexIVFFlat(quantizer,dim,a.nlist,faiss.METRIC_L2) if a.kind=='ivfflat' else
                   faiss.IndexIVFPQ(quantizer,dim,a.nlist,m,a.nbits,faiss.METRIC_L2))
            index.cp.niter=a.coarse_niter; index.cp.seed=1234; index.cp.nredo=1
            index.cp.max_points_per_centroid=max(1,math.ceil(len(base)*a.train_fraction/a.nlist))
            if a.kind=='ivfpq':
                index.pq.cp.niter=a.pq_niter; index.pq.cp.seed=1234; index.pq.cp.nredo=1
                index.pq.cp.max_points_per_centroid=train_points
            print(f'Training {len(base):,} vectors, nlist={a.nlist}',flush=True)
            t=time.perf_counter(); index.train(base); train_time=time.perf_counter()-t
            t=time.perf_counter()
            for first in range(0,len(base),a.add_batch_size):
                index.add(base[first:first+a.add_batch_size])
                print(f'Added {min(first+a.add_batch_size,len(base)):,}/{len(base):,}',flush=True)
            add_time=time.perf_counter()-t; build_time=time.perf_counter()-start
            if index.ntotal!=len(base): raise RuntimeError('Incomplete index')
            index_dir.mkdir(parents=True,exist_ok=True)
            t=time.perf_counter(); temporary=index_path.with_suffix('.faiss.tmp')
            faiss.write_index(index,str(temporary)); temporary.replace(index_path)
            save_time=time.perf_counter()-t
        build=dict(dataset=a.dataset,method='Faiss-CPU',nlist=a.nlist,n_vectors=int(index.ntotal),dim=dim,
            original_dim=orig_dim,padded_dim=dim,pq_m=m if a.kind=='ivfpq' else None,
            nbits=a.nbits if a.kind=='ivfpq' else None,pq_train_points=train_points if a.kind=='ivfpq' else None,
            cpu_threads=a.build_threads,build_threads=a.build_threads,query_threads=a.query_threads,build_time_seconds=build_time,train_seconds=train_time,add_seconds=add_time,
            serialize_seconds=save_time,index_size_bytes=index_path.stat().st_size,index_size_mb=index_path.stat().st_size/MIB,
            peak_build_cpu_memory_mib=memory.peak_bytes/MIB,preparation_seconds=prep,index_path=str(index_path),
            memory_definition='10ms sampled process RSS including base loading, padding, training, add and serialization; may miss shorter peaks',
            timing_definition='index creation + training + add, excluding input load and serialization',cpu_information=cpu_information(),
            faiss_version=faiss.__version__,base=file_signature(base_path),metric='squared_l2')
        write_csv(build_path,[{k:build[k] for k in BUILD_FIELDS}],BUILD_FIELDS)
        write_json(build_path.with_suffix('.json'),build)
        del base
    faiss.omp_set_num_threads(a.query_threads)
    print(f'Querying with {faiss.omp_get_max_threads()} Faiss CPU thread(s)',flush=True)
    counts=np.array([index.invlists.list_size(i) for i in range(a.nlist)],dtype=np.int64)
    if counts.sum()!=index.ntotal: raise RuntimeError('Invalid inverted list sizes')
    meta=dict(status='running',build=build,config=dict(dataset=a.dataset,kind=a.kind,build_threads=build.get('build_threads',build.get('cpu_threads')),query_threads=a.query_threads,index_loaded_from_cache=loaded_index,
        nq=len(queries),topk=a.topk,q_batches=a.q_batches,nprobes=probes,warmup=a.warmup,repetitions=a.repetitions,
        query=file_signature(query_path),truth=file_signature(truth_path)),results=[],
        timing='median of full-query sweeps; sum of per-batch index.search wall times, excluding statistics',
        memory='CPU process RSS in JSON per query; peak_gpu_memory=0 because CPU-only')
    rows=[]
    def save():
        write_json(json_path,meta); write_csv(csv_path,rows,FIELDS)
    save()
    try:
        for size in a.q_batches:
            batches=make_batches(queries,size)
            for probe in probes:
                index.nprobe=probe
                def sweep(account=False):
                    elapsed=0.; found=[]; visits=[]
                    for batch,valid in batches:
                        if account: faiss.cvar.indexIVF_stats.reset()
                        t=time.perf_counter(); distances,ids=index.search(batch,a.topk); elapsed+=time.perf_counter()-t
                        if account:
                            ndis=int(faiss.cvar.indexIVF_stats.ndis)
                            found.append(ids[:valid].copy()); visits.append(account_visits(faiss,index,counts,batch,valid,ndis))
                    return elapsed,found,visits
                for _ in range(a.warmup): sweep()
                times=[sweep()[0] for _ in range(a.repetitions)]
                with RssSampler() as query_memory: _,found,visits=sweep(True)
                found=np.concatenate(found); visits=np.concatenate(visits)
                recall=float(np.mean([len(set(row)&set(gt[:a.topk]))/a.topk for row,gt in zip(found,truth)]))
                row=dict(nprobes=probe,q_batch=size,K=a.topk,qps=len(queries)/statistics.median(times),recall=recall,
                         visited_vectors_per_query=float(visits.mean()),peak_gpu_memory=0.)
                rows.append(row); meta['results'].append(dict(row,samples_seconds=times,peak_rss_memory_mib=query_memory.peak_bytes/MIB))
                meta['completed_rows']=len(rows); save()
                print(f'QBatch={size} nprobe={probe} QPS={row["qps"]:.2f} recall={recall:.5f}',flush=True)
        meta['status']='complete'; save()
    except BaseException as e:
        meta['status']='failed'; meta['error']=repr(e); save(); raise

if __name__=='__main__': main()
