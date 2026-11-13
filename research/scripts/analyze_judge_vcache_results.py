#!/usr/bin/env python3
"""Validate completed replay outputs and build the Chinese report and a plot."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research' / 'results'
NAMES = {'pku': 'PKU-SafeRLHF', 'helpsteer2': 'HelpSteer2', 'ultrafeedback': 'UltraFeedback'}


def pct(x):
    return '—' if x is None else f'{100*x:.2f}%'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    results = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    assert len(results) == 63, f'Expected 63 completed configurations, got {len(results)}'
    lookup = {(r['dataset'], r['order'], r['policy'], r['parameter'], r['key_mode']):r for r in results}
    assert len(lookup) == 63
    traces = {}
    for name in NAMES:
        with (ROOT/'work'/'data'/f'{name}.jsonl').open(encoding='utf-8') as f:
            raw = [json.loads(line) for line in f]
        traces[name] = [dict(r, source_trace_index=i) for i,r in enumerate(raw)
                        if all(v is not None for v in r['oracle'].values())]
    diagnostics = []
    total_events = 0
    for r in results:
        assert r['requests'] == (1917 if r['dataset']=='ultrafeedback' else 2000)
        assert r['complete_case_hits'] == r['hits']
        assert r['hits'] == r['exact_input_hits'] + r['semantic_hits']
        assert abs(r['hit_rate'] - r['hits']/r['requests']) < 1e-12
        assert abs(r['wrong_hit_rate_all_requests'] - r['wrong_hits']/r['requests']) < 1e-12
        if not r['events_file']:
            continue
        with (OUT/r['events_file']).open(encoding='utf-8-sig', newline='') as stream:
            events = list(csv.DictReader(stream))
        assert len(events) == r['requests']
        total_events += len(events)
        trace = traces[r['dataset']]
        for event in events:
            current = trace[int(event['trace_index'])]
            assert current['source_trace_index'] == int(event['source_trace_index'])
            assert current['source_index'] == int(event['source_index'])
            assert json.loads(event['oracle']) == current['oracle']
            if event['hit']=='1':
                origin = trace[int(event['origin_trace_index'])]
                assert json.loads(event['returned']) == origin['oracle']
                assert (event['exact_input']=='True') == (current['prompt']==origin['prompt'])
                assert (event['same_instruction']=='True') == (current['instruction']==origin['instruction'])
                assert (event['same_candidates']=='True') == (current['candidates']==origin['candidates'])
            else:
                assert json.loads(event['returned']) == current['oracle']
        hits = [x for x in events if x['hit']=='1']
        assert len(hits) == r['hits']
        assert sum(x['oracle_equal']=='False' for x in hits) == r['wrong_hits']
        assert all(x['complete_comparison']=='True' for x in hits)
        groups = {
            'same_instruction_different_candidates':[x for x in hits if x['same_instruction']=='True' and x['same_candidates']=='False'],
            'different_instruction':[x for x in hits if x['same_instruction']=='False'],
            'exact_full_input':[x for x in hits if x['exact_input']=='True'],
        }
        for group, xs in groups.items():
            diagnostics.append({**{k:r[k] for k in ['dataset','order','policy','parameter','key_mode']},
                'group':group,'hits':len(xs),'correct':sum(x['oracle_equal']=='True' for x in xs),
                'agreement':sum(x['oracle_equal']=='True' for x in xs)/len(xs) if xs else None})
    with (OUT/'hit_group_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(diagnostics[0])); w.writeheader(); w.writerows(diagnostics)

    labels = json.loads((OUT/'label_statistics.json').read_text(encoding='utf-8'))
    manifest = json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    embed = json.loads((OUT/'embedding_manifest.json').read_text(encoding='utf-8'))
    package_names=['requests','numpy','torch','sentence-transformers','transformers','hnswlib',
                   'scikit-learn','statsmodels','scipy','matplotlib','openai','faiss-cpu','psutil','typing-extensions']
    versions={name:importlib.metadata.version(name) for name in package_names}
    (OUT/'requirements-observed.txt').write_text('\n'.join(f'{name}=={version}' for name,version in versions.items())+'\n',encoding='utf-8')
    model_path=ROOT/'work'/'models'/'bge-small-en-v1.5'/'model.safetensors'
    environment={'observed_versions':versions,'gpu':'NVIDIA GeForce RTX 4060 Laptop GPU',
        'gpu_memory_mib':8188,'driver_version':'566.26',
        'model_weights_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.exists() else None,
        'dependency_caveat':'Observed installed environment, not the upstream locked/supported dependency set. vCache imported from work/vCache via sys.path; Windows optional-import compatibility patch applied.'}
    (OUT/'environment_snapshot.json').write_text(json.dumps(environment,indent=2)+'\n',encoding='utf-8')
    def get(name, policy, param, order='source', key='full_request'):
        return lookup[(name,order,policy,param,key)]
    def row(r):
        label = 'Exact hash' if r['policy']=='exact_hash' else (f"Static τ={r['parameter']:.2f}" if r['policy']=='static' else f"Verified δ={r['parameter']:.2f}")
        return [NAMES[r['dataset']],label,f"{r['hits']}/{r['requests']}",pct(r['hit_rate']),
            pct(r['hit_label_agreement']),f"{r['wrong_hits']} ({pct(r['wrong_hit_rate_all_requests'])})"]
    main_rows = [row(get(name,policy,param)) for name in NAMES
                 for policy,param in [('exact_hash',None),('static',.95),('static',.98),('verified',.01),('verified',.05),('verified',.1)]]
    static_rows = [[NAMES[name],f'{t:.2f}',
        *[f"{pct(get(name,'static',t,order)['hit_rate'])} / {pct(get(name,'static',t,order)['hit_label_agreement'])}" for order in ['source','shuffle42']]]
        for name in NAMES for t in [.8,.85,.9,.95,.98,.99]]
    negative_rows = []
    for name in NAMES:
        r=get(name,'static',.95,key='question_only_negative_control')
        negative_rows.append([NAMES[name],pct(r['hit_rate']),pct(r['hit_label_agreement']),
                              f"{r['different_candidate_hits']}/{r['hits']}",pct(r['wrong_hit_rate_all_requests'])])
    verified_rows = []
    for name in NAMES:
        for d in [.01,.05,.1]:
            a,b=get(name,'verified',d),get(name,'verified',d,'shuffle42')
            verified_rows.append([NAMES[name],f'{d:.2f}',
                f"{pct(a['hit_rate'])} / {pct(a['hit_label_agreement'])}",
                f"{pct(b['hit_rate'])} / {pct(b['hit_label_agreement'])}",
                f"{a['entries_with_at_least_4_observations']}/{a['cache_entries']}"])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(15,4.4),sharey=True)
    for ax,name in zip(axes,NAMES):
        for order,style in [('source','-'),('shuffle42','--')]:
            rs=[get(name,'static',t,order) for t in [.8,.85,.9,.95,.98,.99]]
            ax.plot([100*r['hit_rate'] for r in rs],[100*r['hit_label_agreement'] if r['hit_label_agreement'] is not None else math.nan for r in rs],
                    style,marker='o',color='#2463a6',alpha=1 if order=='source' else .5,label=f'Static ({order})')
        rs=[get(name,'verified',d) for d in [.01,.05,.1]]
        ax.scatter([100*r['hit_rate'] for r in rs],[100*r['hit_label_agreement'] if r['hit_label_agreement'] is not None else math.nan for r in rs],
                   marker='s',s=55,color='#e07b32',label='Verified (source)',zorder=3)
        for j,r in enumerate(rs):
            if r['hit_label_agreement'] is not None:
                if name=='pku':
                    ax.annotate(f"δ={r['parameter']:.2f}",(100*r['hit_rate'],100*r['hit_label_agreement']),xytext=(4,5),textcoords='offset points',fontsize=8)
                else:
                    ax.text(.37,.94-j*.065,f"δ={r['parameter']:.2f}: hit {100*r['hit_rate']:.2f}%, agree {100*r['hit_label_agreement']:.1f}%",transform=ax.transAxes,fontsize=8,color='#a74e16')
        neg=get(name,'static',.95,key='question_only_negative_control')
        ax.scatter(100*neg['hit_rate'],100*neg['hit_label_agreement'],marker='x',s=75,color='#c13f4b',label='Question only, τ=.95',zorder=4)
        ax.set(title=NAMES[name],xlabel='Cache hit rate (%)',xlim=(-1,100),ylim=(-2,104))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Exact full-label agreement among hits (%)')
    axes[0].legend(loc='lower left',fontsize=8)
    fig.suptitle('Offline judge-label replay: more semantic hits do not imply faithful label reuse',fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT/'hit_rate_vs_label_agreement.png',dpi=180)
    fig.savefig(OUT/'hit_rate_vs_label_agreement.svg')
    plt.close(fig)

    baseline_rows=[[NAMES[n],labels[n]['requests'],labels[n]['unique_instructions'],
        labels[n]['exact_unique_requests'],pct(labels[n]['majority_vector_fraction']),
        pct(labels[n]['independent_random_vector_agreement'])] for n in NAMES]
    dim_rows=[]
    for n in NAMES:
        for policy,param in [('static',.95),('verified',.01)]:
            r=get(n,policy,param)
            for dim,mae in json.loads(r['dimension_mae']).items():
                dim_rows.append([NAMES[n],f'{policy} {param}',dim,
                    pct(json.loads(r['dimension_agreement'])[dim]),'—' if mae is None else f'{mae:.3f}'])
    token_rows=[[NAMES[n],embed['datasets'][n]['tokens_median'],embed['datasets'][n]['tokens_p95'],
                 embed['datasets'][n]['tokens_max'],embed['datasets'][n]['requests_over_510_tokens']] for n in NAMES]
    report=f'''# vCache：三套后训练 judge workload 的命中率实验

本轮已完成 **63 个配置**，每个配置从空缓存开始，使用官方 vCache 的静态策略及 Verified 动态策略。它是一次公开评分标签的离线重放 pilot；衡量可以省下多少次 oracle 查询以及复用标签是否相同，未调用新的 LLM judge、未运行训练，也未测训练加速比。

## 数据集与判定任务

- **PKU-SafeRLHF**：双回答 helpfulness / harmlessness 偏好比较，输出两个 0/1 候选编号；2,000 请求。
- **HelpSteer2**：单回答 helpfulness、correctness、coherence、complexity、verbosity 五维 0–4 评分；2,000 请求。
- **UltraFeedback**：GPT-4 的 instruction_following、truthfulness、honesty、helpfulness 四维 1–5 评分；从 500 指令的 2,000 候选中，统一排除 83 条含缺失 honesty 的记录，所有策略使用相同 **1,917** 请求。保留原始 trace，不补分数；一个原始空回答保留。

PKU 和 HelpSteer2 的 oracle 为人工标签，可构造后训练 judge 的典型输入；UltraFeedback 的 oracle 来自 GPT-4。不能将三套都称为已采集的真实 LLM judge 调用日志。rubric 为固定的简化规范，未复原原始 annotation/API prompt；自由文本 critique 不属于本轮被缓存的输出。

抽样在整个 train split 等距取 100 行的页，非 IID。UltraFeedback 覆盖五类来源，各 100 指令。详细来源、索引、缺失项、许可证及快照哈希见 [dataset_selection.md](dataset_selection.md) 与 [trace_metadata.json](trace_metadata.json)。抽样可影响数值，不能当作生产流量命中率估计。

## 主结果：完整请求，原始顺序

命中标签一致率要求 **全部维度同时相同**，无缺失值参与训练或评估。错误命中占全部请求 = 错误复用次数 / 全部请求；命中标签一致率 = 正确复用次数 / 命中次数。二者分母不同。零命中时一致率记为“—”。

{table(['数据集','策略','命中/请求','命中率','命中标签一致率','错误命中数（占全部请求）'],main_rows)}

静态 τ=0.95 是预设展示点，未据结果调参；全部阈值结果如下。Verified 的 δ 是误差控制参数，不是余弦阈值。[官方 benchmark](https://github.com/vcache-project/vCache/blob/{manifest['commit']}/benchmarks/_plotter_helper.py#L287-L308) 采用“错误命中数 / 全部请求”口径，不能直接把 δ 解读为命中集合上的错误率上限。例如 PKU δ=.01 的错误命中为 7/2,000=0.35%，命中内错误为 7/30=23.33%；分母不同。本报告不据短流的实测结果宣称理论保证成立或被推翻。

![命中率与复用质量](hit_rate_vs_label_agreement.png)

## 静态阈值与请求顺序

每格为“命中率 / 命中标签一致率”。shuffle42 对相同请求做固定种子随机重排；两种顺序各自从空缓存开始，不合并结果。

{table(['数据集','余弦阈值 τ','source','shuffle42'],static_rows)}

## Verified：请求顺序与冷启动

{table(['数据集','δ','source 命中率 / 一致率','shuffle42 命中率 / 一致率','source 有≥4条真实反馈的条目/缓存条目'],verified_rows)}

官方实现每条目有两条初始伪观测，累计至少四条真实反馈后才进入参数估计和动态决策。2,000 级别的流中大量条目仍未成熟，因此低命中率同时反映请求稀疏性与冷启动；不能外推为长期稳态。反馈在每次请求后立即串行完成，未模拟线上异步反馈延迟。

## 负对照：只嵌入指令

此处仍传完整 prompt 给 vCache，但检索向量只由 instruction 计算；静态 τ=0.95，source 顺序。它刻意忽略候选答案与上下文，不能用于实际 judge 缓存。

{table(['数据集','命中率','命中标签一致率','同指令不同回答的命中/全部命中','错误命中占全部请求'],negative_rows)}

命中率升高本身不能证明缓存有效。同题的候选答案会改变质量评分或成对胜者；缓存判定必须包含答案，成对任务还必须保留候选顺序。具体分组结果见 [hit_group_diagnostics.csv](hit_group_diagnostics.csv)，个例见 [error_analysis.md](error_analysis.md)。

## 标签分布与逐维误差

{table(['数据集','评估请求','不同指令','不同完整输入','众数全标签占比','独立随机抽取两标签的一致率'],baseline_rows)}

上述分布仅是解释指标难度的参照，未作为缓存策略。PKU 比较的是候选编号；不同题的编号同为 0，并不表示答案或推理语义相同。多维严格相等比单维指标更苛刻；逐维一致率和 MAE 帮助区分小分差与严重误复用，不能据“严格相等较低”就断定所有差异同样有害。

{table(['数据集','策略/参数','维度','命中逐维一致率','命中 MAE'],dim_rows)}

PKU 的 MAE 是 0/1 编号差，HelpSteer2、UltraFeedback 为各自原始评分尺度；不把跨数据集 MAE 当作可直接比较的奖励损失。

## 实现与复现

- 仓库：[vcache-project/vCache](https://github.com/vcache-project/vCache)，commit `{manifest['commit']}`。调用真实 `VCache.infer_with_cache_info`、`BenchmarkStaticDecisionPolicy`、`VerifiedDecisionPolicy`、HNSW 与 `NoEvictionPolicy`，未改决策算法。Windows 可选依赖导入兼容修改见 [vcache_compat.patch](vcache_compat.patch)。
- 输入为 canonical JSON：`rubric + context + instruction + ordered candidates`，不含 oracle、rating、critique 或 rationale。UltraFeedback 保留每个回答自己的生成 system prompt。
- embedding 为固定 revision 的 `BAAI/bge-small-en-v1.5`，384 维；每 480 个内容 token 一块，逐块 CLS 向量归一化、按 token 数加权均值后再归一化。没有截掉长回答；这是本实验的自定义 embedding 适配，不等同于 vCache 的默认 OpenAI embedding，也不等同于论文原始设置。
- 使用预计算向量注入 `BenchmarkEmbeddingEngine`、公开完整标签注入 `BenchmarkInferenceEngine`；无新模型推理成本。静态策略只在 miss 加入缓存；Verified 在探索后使用完整标签相等反馈。每请求等待 future 与 callback 更新完毕，避免预计算向量被下一条请求覆盖。
- 缓存容量 N+10、无淘汰；原始顺序与 seed=42 重排；静态阈值 .80/.85/.90/.95/.98/.99，Verified δ=.01/.05/.10。仅一个 seed，没有总体置信区间或统计显著性结论。
- 所有配置相同缺失过滤规则；签名绑定 trace、embedding、脚本、代码 diff 和配置，防止 resume 混入旧结果。完整软件版本及签名在 [run_manifest.json](run_manifest.json)。
- 本机为 Windows / Python 3.12 / RTX 4060 Laptop（8 GB），使用已有安装环境；部分版本超出仓库声明的依赖范围，不能称为官方锁定环境复现。实际依赖版本见 [requirements-observed.txt](requirements-observed.txt)，模型权重 SHA256 与硬件在 [environment_snapshot.json](environment_snapshot.json)。
- 分析脚本独立复核 57 份事件文件的全部 {total_events:,} 条记录，逐条核对请求索引、原始索引、oracle 与命中来源标签，汇总计数与事件一致；检查记录见 [protocol_audit.md](protocol_audit.md)。

原始 2,000 条/库的输入长度（BGE tokenizer，未过滤缺失行）：

{table(['数据集','中位 token','P95 token','最大 token','超过510 token的请求'],token_rows)}

```powershell
# 在本任务目录执行；work/vCache 是上述 commit，先应用兼容补丁并安装依赖
python research/scripts/prepare_judge_traces.py --sampling spread
python research/scripts/run_judge_vcache_benchmark.py --stage embed
python research/scripts/run_judge_vcache_benchmark.py --stage benchmark --no-resume
python research/scripts/analyze_judge_vcache_results.py
```

原始请求和 raw 页、来源元数据打包在 [judge_trace_snapshot.zip](judge_trace_snapshot.zip)，便于无需再次下载 HF 数据而精确复现；解压目录中的 `data/` 放回 `work/data/`。原始数据的各自许可证仍适用。已有完整 embedding 不必重算；第二次 benchmark 可默认 resume，但输入签名变化时会拒绝混跑。

## 对后训练缓存适配的含义

1. 将 judge model 版本、rubric 版本、system context、完整回答与候选顺序作为缓存命名空间与输入契约；只有题目相似不足以复用奖励。
2. 用候选敏感的检索与验证：指令、上下文、回答分别编码，避免长题目或模板稀释短答案差异；涉及数值、否定、关键事实或成对顺序变化时优先回退真实 judge。是否优于当前整体向量需另做受控实验。
3. 根据训练目标定义可接受误差，例如偏好翻转率、每维 MAE、奖励偏差及排序保持率；在独立校准流上选择阈值，再报告在质量约束下的命中率，不能在同一评估流上挑最佳阈值。
4. 下一轮使用真实训练 rollout 顺序：同一指令的多候选、重复 epoch 和策略版本变化，以及真实 judge 延迟/费用。分别评估精确缓存与语义增益、训练奖励漂移及端到端吞吐。当前重放时间只表示本地策略执行耗时，不能称为训练加速比。

本轮结果支持把该方向继续拆成“候选敏感的安全复用”和“满足奖励保真约束时能获得多少命中”；这些实验尚未证明通用语义缓存能直接加速后训练。
'''
    (OUT/'judge_vcache_report.md').write_text(report,encoding='utf-8')

    with zipfile.ZipFile(OUT/'judge_trace_snapshot.zip','w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for path in sorted((ROOT/'work'/'data').rglob('*')):
            if path.is_file():
                z.write(path,Path('data')/path.relative_to(ROOT/'work'/'data'))
    deliverables={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
                  if p.is_file() and p.name!='deliverable_hashes.json'}
    (OUT/'deliverable_hashes.json').write_text(json.dumps(deliverables,indent=2)+'\n',encoding='utf-8')
    print(f'Validated 63 configurations and all 57 event files ({total_events:,} events). Report, plot, and data snapshot saved.')


if __name__=='__main__':
    main()
