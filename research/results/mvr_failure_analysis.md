# MVR 低命中与错误复用诊断

本分析使用三套测试集的实际 MVR checkpoint 和线上缓存路径重跑单文本 greedy segmentation，并核对原始请求、重排后的 replay trace、标签和策略代码。完整逐条切分（含原 prompt、token 长度、截断标志、8 个边界动作和解码片段）在 [mvr_segmentation_diagnostics.jsonl](mvr_segmentation_diagnostics.jsonl)；复现脚本在 `work/diagnose_mvr_segments.py`。

## 主要原因：Verified 冷启动门槛让测试集里的正确复用机会无法命中

这比单纯的 splitter 检索质量更直接。缓存条目默认带 2 个合成观测 `(0,0)` 和 `(1,1)`；MVR 复用的 Verified 算法在累计观测少于 6 条时无条件返回 `EXPLORE`，也就是至少要给同一个缓存条目积累 4 条真实反馈才允许任何命中。miss 但标签一致时，代码只把这次结果记为该条目的正反馈，不插入一条新缓存；因此同一 label 的重复请求必须反复遇到同一缓存条目，才能走出 warm-up。

但当前 held-out 测试流中，每个 instruction 下最多同标签请求数分别为：PKU 5、HelpSteer2 2、UltraFeedback 4。HelpSteer2 每个同标签缓存条目最多只能积累 1 条真实反馈；UltraFeedback 最多 3 条。两者都达不到需要的 4 条。只有 PKU 一个 group-label 在样本内理论上够门槛，其余大多数 group 也不够。

这与 trace 完全吻合：HelpSteer2 有 18 条 miss 的最近缓存项标签与当前完整 oracle 一致（`FN=18`），UltraFeedback 有 22 条（`FN=22`），但两套数据 `TP=0`。UltraFeedback 的一条精确重复 prompt 在打乱后的第 44 和第 91 个请求出现；第 91 条最近邻标签正确却仍然 miss（`FN=1`）。这说明至少这条重复没有卡在语义检索，而是被 Bayesian warm-up 门槛挡住。

相关代码：`EmbeddingMetadataObj` 初始化两个先验观测（`work/MVR-Cache/mvr-cache/vcache/vcache_core/cache/embedding_store/embedding_metadata_storage/embedding_metadata_obj.py:40-42`）；`_Algorithm.select_action` 在少于 6 个观测时直接 explore（`work/MVR-Cache/mvr-cache/vcache/vcache_policy/strategies/verified.py:535`）；离线标签一致性由评测器在 miss 时记为 FN（`work/MVR-Cache/mvr-cache/benchmarks/eval_sembenchmark_verified_splitter.py:219-229`）。因此当前数据规模和策略的冷启动设计不匹配，不能靠增大 `candidate_k` 解决。

## 切分中间结果：8 个动作常重复或落在 JSON 尾部

推理输入是一整段 canonical JSON，键顺序为 `candidates → context → instruction → rubric`，模型以 BGE tokenizer 的 512-token 右截断处理；它不是分别对 instruction、回答、rubric 做分段。模型每条输入固定输出 8 个边界动作，重复位置会被去重；因此 `max_segments=8` 实际表示最多 8 个边界动作，可形成最多 9 个片段。

| 数据集 | 实际片段数均值 | 实际片段数分布 | 8 个动作中的重复位置占比 | 最终片段为 `" }` |
|---|---:|---|---:|---:|
| PKU | 3.69 | 2–9 段；303 条中 66 条只有 2 段 | 54.0% | 293/303 (96.7%) |
| HelpSteer2 | 5.39 | 2–9 段 | 34.0% | 142/302 (47.0%) |
| UltraFeedback | 6.99 | 2–9 段；295 条中 134 条有至少 8 段 | 15.8% | 118/295 (40.0%) |

最后一片 `" }` 是 JSON 结束符，不含 Judge 语义。HelpSteer2 的另一条代表性 trace（原始 test `sample_index=2`）有 7 个有效片段，但最后两片变成 rubric 字段前的 JSON 标记和只剩开头几个词的 instruction：`" ], " context " : { }, " instruction " :` / `" define signal discuss its`。这不是一个完整 Judge 任务单元。UltraFeedback 里 rubric 也常被拆成可复用的小片段，例如 `higher means stronger performance.` 出现 98 次、`the context preserves the system prompt...` 出现 98 次；这种共有短片段会进入 multivector 检索，却不能区分候选是否会得到相同评分。

## 512-token 截断直接丢失任务条件

请求 JSON 把长候选回答放在 instruction 和 rubric 前面；右截断会先丢掉后面的任务条件。按本地 BGE tokenizer 的 offset mapping 检查：

| 数据集 | 超过 512 token | instruction 未完整进入 | rubric 未完整进入 |
|---|---:|---:|---:|
| PKU | 10/303 (3.3%) | 6/303 (2.0%) | 9/303 (3.0%) |
| HelpSteer2 | 157/302 (52.0%) | 122/302 (40.4%) | 157/302 (52.0%) |
| UltraFeedback | 148/295 (50.2%) | 133/295 (45.1%) | 147/295 (49.8%) |

HelpSteer2 有 53 条测试请求完全看不到 instruction，另有 69 条只看到一部分；UltraFeedback 对应为 69 条完全缺失、64 条部分缺失。此时向量比较主要由回答开头和 JSON 结构决定，无法按 judge 的实际条件判断相似。当前 MVR 数据准备没有预先截短文本，问题发生在 splitter/model tokenizer 输入阶段（`MaxSimSplitter.py:418-423`）。

同一问题也影响训练：以 pair 端点出现次数为分母，训练 pair 的右截断比例为 PKU `491/20,000=2.46%`、HelpSteer2 `10,392/20,000=51.96%`、UltraFeedback `10,134/20,000=50.67%`。HelpSteer2/UltraFeedback 的监督标签由完整 prompt 的 oracle 生成，但分段策略训练时约一半端点只看到了前 512 tokens，instruction/rubric 不完整会让“同 instruction 同标签 / 不同标签”的监督变得不可辨。

## 误命中显示“语义相近”不等于“可复用”

- PKU 第 271 个 replay 请求把“解雇残障员工”的候选回答匹配到“希望不与黑人共同生活”的缓存项。两者 instruction 不同，完整 oracle tuple 也不同。这是跨题目的语义近邻误复用。
- HelpSteer2 第 269 个 replay 请求和缓存项是同一段 Azure 借贷市场建模对话，但模型生成的回答不同，五维评分向量也不同。共享 instruction/context 被切分器赋予了足够相似度，Verified policy 最终错误复用了评分。

所以这里的目标不是单纯提高同题检索率。单回答评分依赖候选全文；偏好比较依赖候选内容和顺序；rubric/system context 也可能改变分数。复用必须通过候选敏感、标签误差受控的判定。

## 其他会放大问题的因素

训练 pair 文件每套有 10,000 条 train pair，但本轮 `train_data_size=512`、5 epochs，训练总计 2,560 个 pair exposure（每 epoch 64 个 optimizer steps）。模型因此只见到 pair 文件的一部分；同时测试文本字段序列化顺序与现有按句子相似度训练目标之间没有字段级约束。该点可能降低 splitter 的泛化，但现有证据不能单独证明它是主要命中率瓶颈。

`candidate_k=10` 是“每个 query segment 找 10 个缓存向量邻居，再 union parent ID”，不等于全请求只看 10 条缓存。trace 中的 18 和 22 个同标签最近邻 miss 说明已有不少正确候选经过候选召回和 MaxSim 重排到达决策层；对这些请求，首要瓶颈是 Verified warm-up。其余正标签重复还可能受切分、512 截断和 multivector HNSW 候选召回影响。

## 建议的验证顺序

1. 先做决策层消融：在固定 candidate/MaxSim 分数下，比较原始 6-observation warm-up、较低 warm-up、校准好的固定阈值，以及 exact-match fallback。必须单独报告 FP，不能只看 hit rate。
2. 在相同 held-out rows 上记录每次请求的候选数、top candidate ID、MaxSim/full-cos 分数、`t_hat/t_prime`、观测数和 explore/exploit 动作。当前逐请求 JSON 只留 hit/TN/FN，缺了这些诊断列。
3. 修改输入表示：将 rubric、system context、instruction、候选分别编码；偏好任务显式保留 candidate 0/1 顺序；长回答按块编码或采用层级聚合，不要把 instruction/rubric 静默截掉。
4. 再扩大训练 pair 覆盖，增加重复轮次并报告多 seed。先验证在本轮标签定义下能召回同 instruction、同 oracle 的已知正例，再做真实训练 rollout 和端到端吞吐测量。

当前最确定的解释是：HelpSteer2/UltraFeedback 的正确复用机会几乎无法通过 Verified 的冷启动门槛；切分无字段感知和 512-token 截断让相似度质量更差；跨任务或候选错误复用已经出现，因此直接放宽门槛会增加奖励污染风险。
