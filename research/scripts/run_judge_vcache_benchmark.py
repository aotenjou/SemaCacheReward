#!/usr/bin/env python3
"""Offline judge-cache pilot using the real vCache APIs and public label oracles.

First prepare traces with prepare_judge_traces.py. Clone the official repo into
work/vCache at commit 50694ee and apply vcache_compat.patch if optional adapters
are unavailable. No inference API key is required. No training speed is measured.

python research/scripts/run_judge_vcache_benchmark.py --stage embed
python research/scripts/run_judge_vcache_benchmark.py --stage benchmark
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA = ROOT / 'work' / 'data'
OUT = ROOT / 'research' / 'results'
EMBED = ROOT / 'work' / 'embeddings'
MODEL_ID = 'BAAI/bge-small-en-v1.5'
MODEL_REVISION = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
MODEL_PATH = ROOT / 'work' / 'models' / 'bge-small-en-v1.5'
DATASETS = ['pku', 'helpsteer2', 'ultrafeedback']
THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
DELTAS = [0.01, 0.05, 0.10]
CHUNK_TOKENS = 480
SEED = 42


def json_text(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_for_callbacks(policy, timeout=30):
    deadline = time.monotonic() + timeout
    while policy.callback_queue.unfinished_tasks:
        if not policy.callback_queue.worker_thread.is_alive():
            raise RuntimeError('vCache callback worker exited with unfinished updates')
        if time.monotonic() >= deadline:
            raise TimeoutError('vCache callback updates did not finish within 30 seconds')
        time.sleep(.001)


def read_rows(name):
    with (DATA / f'{name}.jsonl').open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        assert json.loads(row['prompt']) == {k: row[k] for k in ('rubric', 'context', 'instruction', 'candidates')}
        assert 'oracle' not in json.loads(row['prompt'])
    return rows


def complete_rows(name):
    raw = read_rows(name)
    selected = [i for i, row in enumerate(raw) if all(v is not None for v in row['oracle'].values())]
    rows = [dict(raw[i], source_trace_index=i) for i in selected]
    return rows, selected, len(raw)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def embed_all(device, batch_size):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    if MODEL_PATH.exists() and (MODEL_PATH/'model.safetensors').exists():
        model = SentenceTransformer(str(MODEL_PATH), device=device, local_files_only=True)
    else:
        model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=device)
    model.eval()
    tokenizer = model.tokenizer
    info = {}
    EMBED.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        rows = read_rows(name)
        started = time.perf_counter()
        chunk_ids, owners, lengths, token_counts = [], [], [], []
        query_ids, query_owners, query_lengths = [], [], []
        for i, row in enumerate(rows):
            for text, ids_out, owners_out, lengths_out, full in [
                (row['prompt'], chunk_ids, owners, lengths, True),
                (row['instruction'], query_ids, query_owners, query_lengths, False),
            ]:
                ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
                if full:
                    token_counts.append(len(ids))
                for start in range(0, max(1, len(ids)), CHUNK_TOKENS):
                    seq = ids[start:start+CHUNK_TOKENS]
                    ids_out.append(tokenizer.build_inputs_with_special_tokens(seq))
                    owners_out.append(i)
                    lengths_out.append(max(1, len(seq)))

        def encode_and_reduce(seqs, indexes, weights):
            result = np.zeros((len(rows), model.get_sentence_embedding_dimension()), dtype=np.float32)
            weight_sum = np.zeros(len(rows), dtype=np.float32)
            for start in range(0, len(seqs), batch_size):
                end = min(start+batch_size, len(seqs))
                features = tokenizer.pad({'input_ids':seqs[start:end]}, padding=True, return_tensors='pt')
                features = {k:v.to(device) for k,v in features.items()}
                with torch.inference_mode():
                    vectors = model(features)['sentence_embedding']
                    vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=1).cpu().numpy()
                for j, vec in enumerate(vectors, start):
                    result[indexes[j]] += vec * weights[j]
                    weight_sum[indexes[j]] += weights[j]
                if start % (batch_size*20) == 0:
                    print(f'embed {name}: {end}/{len(seqs)} chunks', flush=True)
            result /= weight_sum[:,None]
            result /= np.linalg.norm(result, axis=1, keepdims=True).clip(1e-10)
            assert np.isfinite(result).all()
            return result

        full = encode_and_reduce(chunk_ids, owners, lengths)
        query = encode_and_reduce(query_ids, query_owners, query_lengths)
        np.savez_compressed(EMBED/f'{name}.npz', full=full, question_only=query)
        counts = np.asarray(token_counts)
        info[name] = {
            'requests':len(rows), 'trace_sha256':sha(DATA/f'{name}.jsonl'),
            'full_chunks':len(chunk_ids), 'question_chunks':len(query_ids),
            'tokens_min':int(counts.min()), 'tokens_median':float(np.median(counts)),
            'tokens_p95':float(np.quantile(counts,.95)), 'tokens_max':int(counts.max()),
            'requests_over_510_tokens':int((counts>510).sum()),
            'elapsed_seconds':time.perf_counter()-started,
            'embedding_sha256':sha(EMBED/f'{name}.npz'),
        }
        print('embedding complete', name, info[name], flush=True)
        save_json(EMBED/'embedding_manifest.json', {'model':MODEL_ID,'revision':MODEL_REVISION,
            'chunk_tokens':CHUNK_TOKENS,'dimension':384,'device':device,
            'method':'Normalize each CLS chunk embedding, token-count-weighted mean, normalize. No truncation; no instruction prefix.',
            'datasets':info})
    save_json(OUT/'embedding_manifest.json',json.loads((EMBED/'embedding_manifest.json').read_text()))


def evaluation(target, cached):
    exact = target == cached
    complete = all(v is not None for v in target.values()) and all(v is not None for v in cached.values())
    valid = {k:abs(v-cached[k]) for k,v in target.items()
             if isinstance(v,(int,float)) and isinstance(cached.get(k),(int,float))}
    return exact, complete, valid


def summarize(name, order, policy, parameter, key_mode, rows, events, entries, elapsed, observations):
    hits = [x for x in events if x['hit']]
    complete = [x for x in hits if x['complete_comparison']]
    same_q = [x for x in hits if x['same_instruction']]
    dims = list(rows[0]['oracle'])
    dim_agreement, dim_mae = {}, {}
    for dim in dims:
        errors = [x[f'error_{dim}'] for x in hits if x[f'error_{dim}'] is not None]
        dim_agreement[dim] = sum(e == 0 for e in errors)/len(errors) if errors else None
        dim_mae[dim] = sum(errors)/len(errors) if errors else None
    flat_errors = [x[f'error_{d}'] for x in hits for d in dims if x[f'error_{d}'] is not None]
    observed = Counter(observations)
    return {
        'dataset':name,'order':order,'policy':policy,'parameter':parameter,'key_mode':key_mode,
        'requests':len(events),'hits':len(hits),'hit_rate':len(hits)/len(events),
        'exact_input_hits':sum(x['exact_input'] for x in hits),
        'semantic_hits':sum(not x['exact_input'] for x in hits),
        'wrong_hits':sum(not x['oracle_equal'] for x in hits),
        'hit_label_agreement':sum(x['oracle_equal'] for x in hits)/len(hits) if hits else None,
        'wrong_hit_rate_all_requests':sum(not x['oracle_equal'] for x in hits)/len(events),
        'complete_case_hits':len(complete),
        'complete_case_hit_agreement':sum(x['oracle_equal'] for x in complete)/len(complete) if complete else None,
        'hit_mae_valid_dimensions':sum(flat_errors)/len(flat_errors) if flat_errors else None,
        'dimension_agreement':json_text(dim_agreement),'dimension_mae':json_text(dim_mae),
        'same_instruction_hits':len(same_q),
        'different_candidate_hits':sum(x['same_instruction'] and not x['same_candidates'] for x in hits),
        'swapped_pair_hits':sum(x.get('swapped_pair',False) for x in hits),
        'first_500_hit_rate':sum(x['hit'] for x in events[:500])/min(500,len(events)),
        'after_500_hit_rate':sum(x['hit'] for x in events[500:])/len(events[500:]) if len(events)>500 else None,
        'cache_entries':entries,'real_observations_per_entry':json_text(dict(sorted(observed.items()))),
        'entries_with_at_least_4_observations':sum(x>=4 for x in observations),
        'replay_seconds_excluding_embedding':elapsed,
    }


def run_exact(name, rows, order_name, indexes):
    cache, events = {}, []
    started = time.perf_counter()
    for step, i in enumerate(indexes):
        row = rows[i]
        origin = cache.get(row['prompt'])
        hit = origin is not None
        if not hit:
            cache[row['prompt']] = i
            origin = i
        cached = rows[origin]
        exact, complete, errors = evaluation(row['oracle'],cached['oracle'])
        events.append({'hit':int(hit),'oracle_equal':exact,'complete_comparison':complete,
            'exact_input':bool(hit),'same_instruction':bool(hit),'same_candidates':bool(hit),
            **{f'error_{d}':errors.get(d) for d in row['oracle']}})
    return summarize(name,order_name,'exact_hash',None,'full_request',rows,events,len(cache),
                     time.perf_counter()-started,[])


def run_vcache(name, rows, vectors, order_name, indexes, kind, parameter, key_mode):
    import numpy as np
    from vcache import (VCache,VCacheConfig,BenchmarkEmbeddingEngine,BenchmarkInferenceEngine,
        HNSWLibVectorDB,InMemoryEmbeddingMetadataStorage,NoEvictionPolicy,
        StringComparisonSimilarityEvaluator,BenchmarkStaticDecisionPolicy,VerifiedDecisionPolicy)
    random.seed(SEED)
    np.random.seed(SEED)
    policy = BenchmarkStaticDecisionPolicy(parameter) if kind=='static' else VerifiedDecisionPolicy(parameter)
    engine, embed = BenchmarkInferenceEngine(), BenchmarkEmbeddingEngine()
    store = InMemoryEmbeddingMetadataStorage()
    vector_db = HNSWLibVectorDB(max_capacity=len(rows)+10)
    cache = VCache(VCacheConfig(inference_engine=engine, embedding_engine=embed,vector_db=vector_db,
        embedding_metadata_storage=store,eviction_policy=NoEvictionPolicy(),
        similarity_evaluator=StringComparisonSimilarityEvaluator()), policy)
    pending, last_knn = [], []
    if kind=='verified':
        submit = policy.executor.submit
        def record_submit(*args, **kwargs):
            future = submit(*args, **kwargs)
            pending.append(future)
            return future
        policy.executor.submit = record_submit
    get_knn = vector_db.get_knn
    def capture_knn(*args, **kwargs):
        result = get_knn(*args, **kwargs)
        last_knn[:] = result
        return result
    vector_db.get_knn = capture_knn
    events=[]
    started=time.perf_counter()
    try:
        for step,i in enumerate(indexes):
            row=rows[i]
            embed.set_next_embedding(vectors[i].tolist())
            engine.set_next_response(json_text(row['oracle']))
            hit,response,returned,nearest=cache.infer_with_cache_info(
                row['prompt'],id_set=i)
            for future in pending:
                future.result(timeout=30)
            pending.clear()
            if kind=='verified':
                wait_for_callbacks(policy)
                assert policy.callback_queue.unfinished_tasks == 0
            cached=json.loads(response)
            if not hit:
                assert cached==row['oracle'], 'Benchmark miss returned incorrect oracle'
            origin=returned.id_set if hit else -1
            other=rows[origin] if hit else None
            exact,complete,errors=evaluation(row['oracle'],cached)
            events.append({'step':step,'trace_index':i,'source_index':row['source_index'],
                'source_trace_index':row['source_trace_index'],
                'completion_index':row.get('completion_index'), 'hit':int(hit),'origin_trace_index':origin,
                'nearest_similarity':float(last_knn[0][0]) if last_knn else None,
                'oracle_equal':exact,'complete_comparison':complete,
                'exact_input':bool(hit and row['prompt']==other['prompt']),
                'same_instruction':bool(hit and row['instruction']==other['instruction']),
                'same_candidates':bool(hit and row['candidates']==other['candidates']),
                'swapped_pair':bool(hit and len(row['candidates'])==2 and row['candidates']==other['candidates'][::-1]),
                'oracle':json_text(row['oracle']),'returned':json_text(cached),
                **{f'error_{d}':errors.get(d) for d in row['oracle']}})
            if (step+1)%500==0:
                print(f'{name} {order_name} {kind} {parameter} {key_mode}: {step+1}/{len(rows)}, hits={sum(x["hit"] for x in events)}',flush=True)
        assert vector_db.size()==len(store.get_all_embedding_metadata_objects())
        observations=[len(m.observations)-2 for m in store.get_all_embedding_metadata_objects()] if kind=='verified' else []
        result=summarize(name,order_name,kind,parameter,key_mode,rows,events,vector_db.size(),
                         time.perf_counter()-started,observations)
        file=f'{name}_{order_name}_{kind}_{parameter}_{key_mode}.csv'
        write_csv(OUT/'events'/file,events)
        result['events_file']='events/'+file
        print('RESULT',json_text(result),flush=True)
        return result
    finally:
        if kind=='verified':
            policy.executor.shutdown(wait=True)
            if policy.callback_queue.worker_thread.is_alive():
                policy.callback_queue.put(None)
                policy.callback_queue.worker_thread.join(timeout=30)
                if policy.callback_queue.worker_thread.is_alive():
                    raise TimeoutError('vCache callback worker did not shut down')


def benchmark(orders, resume):
    import numpy as np
    sys.path.insert(0,str(ROOT/'work'/'vCache'))
    log=OUT/'benchmark.log'
    logging.basicConfig(filename=log,level=logging.WARNING,force=True)
    warnings.filterwarnings('ignore',category=FutureWarning)
    metadata=json.loads((EMBED/'embedding_manifest.json').read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'work'/'vCache',text=True).strip()
    git_diff=subprocess.check_output(['git','diff','--binary'],cwd=ROOT/'work'/'vCache')
    signature_payload={'runner_sha256':sha(Path(__file__)),
        'prepare_script_sha256':sha(SCRIPT_DIR/'prepare_judge_traces.py'),
        'compat_patch_sha256':sha(OUT/'vcache_compat.patch'),
        'vcache_commit':commit,'vcache_diff_sha256':hashlib.sha256(git_diff).hexdigest(),
        'trace_metadata_sha256':sha(DATA/'metadata.json'),
        'embedding_manifest_sha256':sha(EMBED/'embedding_manifest.json'),
        'static_thresholds':THRESHOLDS,'verified_deltas':DELTAS,'orders':orders,'seed':SEED,
        'workload_filter':'exclude incomplete oracle rows for every policy'}
    run_signature=hashlib.sha256(json_text(signature_payload).encode()).hexdigest()
    signature_path=OUT/'run_signature.json'
    results=[]
    path=OUT/'summary.json'
    if resume and path.exists():
        if not signature_path.exists() or json.loads(signature_path.read_text())['signature']!=run_signature:
            raise RuntimeError('Result signature mismatch; use --no-resume for a fresh experiment')
        results=json.loads(path.read_text())
    save_json(signature_path,{'signature':run_signature,'inputs':signature_payload})
    done={(r['dataset'],r['order'],r['policy'],r['parameter'],r['key_mode']) for r in results}
    for name in DATASETS:
        rows, selected, original_count=complete_rows(name)
        assert metadata['datasets'][name]['trace_sha256']==sha(DATA/f'{name}.jsonl'), 'Trace changed; rerun embeddings'
        assert metadata['datasets'][name]['embedding_sha256']==sha(EMBED/f'{name}.npz'), 'Embedding changed'
        with np.load(EMBED/f'{name}.npz') as arrays:
            full,question=arrays['full'][selected],arrays['question_only'][selected]
        for order_name in orders:
            indexes=list(range(len(rows)))
            if order_name=='shuffle42':
                random.Random(SEED).shuffle(indexes)
            configs=[('exact_hash',None,'full_request')]
            configs += [('static',t,'full_request') for t in THRESHOLDS]
            configs += [('verified',d,'full_request') for d in DELTAS]
            if order_name=='source':
                configs += [('static',.95,'question_only_negative_control')]
            for kind,param,key in configs:
                if (name,order_name,kind,param,key) in done:
                    continue
                result=run_exact(name,rows,order_name,indexes) if kind=='exact_hash' else run_vcache(
                    name,rows,question if key.startswith('question_only') else full,
                    order_name,indexes,kind,param,key)
                result.setdefault('events_file',None)
                result['original_trace_requests']=original_count
                result['excluded_incomplete_oracle_requests']=original_count-len(rows)
                results.append(result)
                save_json(path,results)
                write_csv(OUT/'summary.csv',results)
    labels={}
    for name in DATASETS:
        rows, _, original_count=complete_rows(name)
        counts=Counter(json_text(r['oracle']) for r in rows)
        n=len(rows)
        labels[name]={'requests':n,'original_trace_requests':original_count,
            'excluded_incomplete_oracle_requests':original_count-n,
            'exact_unique_requests':len({r['prompt'] for r in rows}),
            'unique_instructions':len({r['instruction'] for r in rows}),
            'complete_oracles':sum(all(v is not None for v in r['oracle'].values()) for r in rows),
            'majority_vector_fraction':max(counts.values())/n,
            'independent_random_vector_agreement':sum((v/n)**2 for v in counts.values()),
            'most_common_vectors':counts.most_common(10)}
    save_json(OUT/'label_statistics.json',labels)
    versions={p:importlib.metadata.version(p) for p in ['numpy','torch','sentence-transformers',
        'transformers','hnswlib','scikit-learn','statsmodels','scipy']}
    save_json(OUT/'run_manifest.json',{'python':sys.version,'platform':sys.platform,'versions':versions,
        'repository':'https://github.com/vcache-project/vCache','commit':commit,
        'seed':SEED,'orders':orders,'static_thresholds':THRESHOLDS,'verified_deltas':DELTAS,
        'cache_capacity':'unbounded for this trace (N+10)',
        'workload_filter':'Only rows with every oracle dimension present, for all strategies; UltraFeedback excludes 83/2000 rows.',
        'oracle_comparison':'StringComparisonSimilarityEvaluator on canonical numeric JSON, equivalent to full label-vector equality; no null present',
        'feedback_timing':'serialized immediate feedback; future.result then bounded callback drain after every request',
        'labels':'public dataset oracle, not new live judge outputs','embedding':metadata,
        'trace_manifest_sha256':sha(DATA/'metadata.json'),'run_signature':run_signature,
        'code_signatures':signature_payload})
    save_json(OUT/'trace_metadata.json',json.loads((DATA/'metadata.json').read_text()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['embed','benchmark','all'],default='all')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--orders',nargs='+',choices=['source','shuffle42'],default=['source','shuffle42'])
    parser.add_argument('--no-resume',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['MKL_NUM_THREADS']='4'
    if args.stage in ['embed','all']:
        embed_all(args.device,args.batch_size)
    if args.stage in ['benchmark','all']:
        benchmark(args.orders,not args.no_resume)


if __name__=='__main__':
    main()
