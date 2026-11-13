# MVR-Cache 用于 LLM Judge 后训练缓存的实验报告

## 结论

本轮将 [PKU-SDS-lab/MVR-Cache](https://github.com/PKU-SDS-lab/MVR-Cache) 的 MaxSim 分段缓存适配到三类 LLM Judge 输入，并在本机 RTX 4060 Laptop GPU 上训练了三套数据集专属的分段模型。固定配置为 `delta=0.02`、`candidate_k=10`、`multivector_top_k`、最多 8 个边界动作、`separate` policy、标点分割、full embedding 和 full-cos 混合。8 个动作可能重复或落到文本尾部，所以实际片段数通常少于 8。

在按 instruction 分组留出的测试流上，MVR 的命中非常少，而且两个命中都是错误复用：PKU 为 `1/303=0.330%`，HelpSteer2 为 `1/302=0.331%`，UltraFeedback 为 `0/295=0%`。三套数据的 `TP` 均为 0；因此当前结果没有证明可以安全复用 Judge 结果，也不能把回放耗时解释成训练加速比。

## 数据集与切分

三套数据集均来自已经保存的本地 trace 快照，oracle 只用于离线评估和训练 pair 标签，没有放入缓存输入：

| 数据集 | Judge 任务 | 完整 oracle 行 | instruction 组 | test 行 | test 唯一 prompt |
|---|---|---:|---:|---:|---:|
| PKU-SafeRLHF | 两个回答的 helpfulness / harmlessness 偏好，两个 0/1 选择 | 2,000 | 1,929 | 303 | 303 |
| HelpSteer2 | 单回答 helpfulness、correctness、coherence、complexity、verbosity，五个 0–4 分 | 2,000 | 999 | 302 | 302 |
| UltraFeedback | instruction_following、truthfulness、honesty、helpfulness，四个 1–5 分 | 1,917 | 494 | 295 | 294 |

切分按 `(dataset, stripped instruction)` 分组，比例为 70%/15%/15%，seed 为 42，保证同一 instruction 的候选不会跨 train/validation/test。训练 pair 为每套 `10,000/1,000/1,000`（train/val/test），正负样本各半；正例要求同一 instruction 下完整 oracle tuple 相同，负例要求不同。详细 manifest 在 `work/mvr_training_data/*_manifest.json` 和 `work/mvr_eval_data/*_test_manifest.json`。

## 本机 GPU 训练

三套分段模型均在本机 CUDA GPU 上完成训练，使用本地 `BAAI/bge-base-en-v1.5` 权重，768 维 embedding。训练和推理保持一致：

- `max_segments=8`，`policy_mode=separate`；
- 仅允许标点作为候选边界（`punctuation_only`）；
- `batch_size=2`，`accumulate_grad_batches=4`，`lr=5e-5`，`max_epochs=5`；
- `bce_auto_balance=True`，pair 训练模式；pair 文件规模为 train/val/test `10,000/1,000/1,000`，但 RL 每个 epoch 实际采样 `512` 条 train pair，batch 为 2、累积 4 步，即每 epoch 64 个 optimizer steps，5 个 epoch 共 320 steps；
- 文本 tokenizer 使用 `max_length=512`、右侧截断，包含 `[CLS]`/`[SEP]`；
- 设备为 `cuda:0`，硬件为 NVIDIA GeForce RTX 4060 Laptop GPU（8 GB）。

Checkpoint：

- PKU：`work/mvr_checkpoints/pku/epoch=4-step=320.ckpt`（最终 checkpoint）
- HelpSteer2：`work/mvr_checkpoints/helpsteer2/epoch=0-step=64.ckpt`（best monitored checkpoint；最终 checkpoint 为 `last.ckpt`，epoch 4 / step 320）
- UltraFeedback：`work/mvr_checkpoints/ultrafeedback/epoch=3-step=256.ckpt`（best monitored checkpoint；最终 checkpoint 为 `last.ckpt`，epoch 4 / step 320）

HelpSteer2 训练结束时 Windows 控制台在 Rich progress bar 清理阶段出现 Unicode 编码异常，但 checkpoint 已写入并用于后续评测。

## MVR 回放结果

评测入口为 `work/MVR-Cache/mvr-cache/benchmarks/eval_sembenchmark_verified_splitter.py`。本轮使用 MVR-Cache 仓库 commit `8cfaa0b86c4cb6a650cb207caa053fc1932bda7b` 加上本地 Windows / parent-aware HNSW 与 CUDA 适配；评测参数和逐请求结果均写入下列 JSON。

命中定义为缓存策略返回已有结果。命中且完整 oracle tuple 相同为 `TP`，命中但不同为 `FP`。miss 时，当前请求与最近缓存项标签相同记为 `FN`，不同记为 `TN`；后两项是“潜在复用机会”的近似统计，不是独立构造的标准二分类 ground truth。

| 数据集 | 请求数 | 命中 | 命中率 | TP | FP | TN | FN | 命中标签一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PKU-SafeRLHF | 303 | 1 | 0.330% | 0 | 1 | 295 | 7 | 0/1 = 0% |
| HelpSteer2 | 302 | 1 | 0.331% | 0 | 1 | 283 | 18 | 0/1 = 0% |
| UltraFeedback | 295 | 0 | 0% | 0 | 0 | 273 | 22 | — |

PKU 的唯一命中是一个关于解雇残障员工的请求，复用了另一条种族歧视请求，属于跨 instruction 错误复用；HelpSteer2 的唯一命中属于同一 instruction/对话，但候选回答不同且完整 oracle 标签不同，说明相同任务本身不足以安全复用。UltraFeedback 的测试流保留了一条同 instruction 的精确重复 prompt，但本配置仍未命中。详细 trace 分析见 [MVR 低命中诊断](mvr_failure_analysis.md)。

原始结果：

- [PKU JSON](C:\Users\AzrMedit0x\Documents\Codex\2026-10-20\llm-judge-llm-tvcache-llm-judge\outputs\mvr_pku_delta002_k10.json) / [命中样本](C:\Users\AzrMedit0x\Documents\Codex\2026-10-20\llm-judge-llm-tvcache-llm-judge\outputs\mvr_pku_delta002_k10_hits.jsonl)
- [HelpSteer2 JSON](C:\Users\AzrMedit0x\Documents\Codex\2026-10-20\llm-judge-llm-tvcache-llm-judge\outputs\mvr_helpsteer2_delta002_k10.json) / [命中样本](C:\Users\AzrMedit0x\Documents\Codex\2026-10-20\llm-judge-llm-tvcache-llm-judge\outputs\mvr_helpsteer2_delta002_k10_hits.jsonl)
- [UltraFeedback JSON](C:\Users\AzrMedit0x\Documents\Codex\2026-10-20\llm-judge-llm-tvcache-llm-judge\outputs\mvr_ultrafeedback_delta002_k10.json)

## 截断影响

CSV 和训练 pair 都保存完整 canonical prompt；截断发生在 BGE tokenizer / MaxSim splitter 的 `truncation=True, max_length=512`，方向为 right。按完整测试 prompt 的含特殊 token 长度统计，确定被截掉内容的比例为：

| 数据集 | 确定被截断的测试 prompt | 比例 | 最大长度 |
|---|---:|---:|---:|
| PKU-SafeRLHF | 10/303 | 3.30% | 1,544 |
| HelpSteer2 | 157/302 | 51.99% | 1,837 |
| UltraFeedback | 148/295 | 50.17% | 2,609 |

因此 HelpSteer2 和 UltraFeedback 约一半请求的候选答案、instruction 或 rubric 尾部没有进入 MVR 的语义编码。这个限制可能直接压低了候选敏感检索质量，下一轮应比较 512、1024 或分字段编码，而不是只调 `delta`。

## 时间分解

计时包含本地 MVR replay 的 embedding、候选检索、RL splitter 和 MaxSim 阶段；每条请求都等待后台索引更新完成：

| 数据集 | 总耗时 | embedding | retrieval | RL splitter | MaxSim | 平均每请求 |
|---|---:|---:|---:|---:|---:|---:|
| PKU-SafeRLHF | 22.54 s | 3.866 s | 0.579 s | 4.934 s | 2.502 s | 74.4 ms |
| HelpSteer2 | 24.77 s | 4.824 s | 0.816 s | 4.673 s | 2.577 s | 82.0 ms |
| UltraFeedback | 23.20 s | 4.616 s | 0.959 s | 4.399 s | 2.159 s | 78.6 ms |

这是缓存决策回放耗时，不包含真实 LLM Judge 调用，也没有测量训练 rollout 的端到端吞吐、GPU 显存压力或节省的 API 成本，所以不能称为训练加速比。

## 与已有 vCache 结果的关系

已有 vCache pilot 在完整原始流上报告：PKU static `tau=.95` 为 `67/2000=3.35%`、Verified `delta=.01` 为 `30/2000=1.50%`；HelpSteer2 分别为 `534/2000=26.70%` 和 `4/2000=0.20%`；UltraFeedback 分别为 `790/1917=41.21%` 和 `1/1917=0.052%`。这些数字只能作参考，不能作为严格 head-to-head：

- vCache 跑完整 2,000/2,000/1,917 条原始顺序，MVR 只跑按 instruction 留出的 303/302/295 条 test 流；
- vCache 使用 BGE-small（384 维）和 480-token chunk 的完整请求向量，MVR 使用 BGE-base（768 维）、最多 8 个边界动作（去重后最多形成 9 段）以及 512-token tokenizer 截断；
- vCache 的 Verified `delta` 与 MVR 的 MaxSim splitter 校准并非同一个分数空间，`delta=.01` 与 `delta=.02` 不能直接比较；
- vCache 旧结果的 replay 计时口径和 MVR 的 GPU 分段计时不同。

要做公平比较，应在相同 held-out rows、相同顺序、相同完整输入和统一标签误差定义上重跑 vCache 与 MVR，并同时报告命中率、错误命中率、偏好翻转率、逐维 MAE 和端到端真实 Judge 成本。

## 当前限制与下一步

本轮只使用一个 seed、一组 `delta/k`（0.02/10），且没有调用新的 LLM Judge API；公开标签被当作 oracle replay。测试流主要由独立 instruction 组成，不能代表真实后训练 rollout 中同一 prompt 跨 epoch、同一 instruction 多候选、策略版本变化和批量并发的请求分布。

下一轮建议优先做：

1. 在相同 test rows 上 sweep `delta` 和 `candidate_k=1/5/10/20`，并比较 `top_k` 与 `multivector_top_k`。
2. 构造真实训练顺序：同 instruction 多候选、重复 epoch、策略 checkpoint 演化和并发请求。
3. 将 rubric、instruction、context、candidate 分字段编码，或提升 tokenizer 长度，单独测量长答案截断带来的误差。
4. 接入真实 Judge API / 本地 Judge 模型，报告端到端延迟、调用节省和奖励漂移，而不是只报告离线 replay 命中率。
