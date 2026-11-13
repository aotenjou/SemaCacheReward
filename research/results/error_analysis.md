# vCache 命中来源与错误复用分析

分析对象为本次公开评分标签离线重放的 source 顺序，从空缓存开始。这里的“正确”仅指缓存返回的**全部 oracle 维度与目标记录一致**；没有重新调用 LLM 或人工复评。PKU、HelpSteer2 的 oracle 来自人工标注，UltraFeedback 来自 GPT-4。不同维度数、评分尺度与标签分布会影响严格相等率，因此不能直接把跨数据集的一致率差异当作奖励损害大小。

## 命中主要发生在哪里

下表每格为“标签一致的命中数 / 该类全部命中数（一致率）”。“同题”要求 instruction 字符串相同；“同候选”要求完整 candidates 数组相同，保留顺序。分类互斥且覆盖全部命中。“同题同候选”仍可能有不同的 context，例如 UltraFeedback 的生成 system prompt，因此不自动等于完整输入精确重复。

| 策略、检索输入 | 数据集 | 同题、不同候选 | 不同题 | 同题、同候选 |
| --- | --- | --- | --- | --- |
| Static τ=.95，完整请求 | PKU | 21/31 (67.74%) | 21/36 (58.33%) | — |
| Static τ=.95，完整请求 | HelpSteer2 | 64/494 (12.96%) | 2/40 (5.00%) | — |
| Static τ=.95，完整请求 | UltraFeedback | 61/696 (8.76%) | 4/78 (5.13%) | 15/16 (93.75%) |
| Static τ=.95，仅 instruction，负对照 | PKU | 48/70 (68.57%) | 15/34 (44.12%) | — |
| Static τ=.95，仅 instruction，负对照 | HelpSteer2 | 88/978 (9.00%) | 3/44 (6.82%) | — |
| Static τ=.95，仅 instruction，负对照 | UltraFeedback | 66/1320 (5.00%) | 2/120 (1.67%) | 13/15 (86.67%) |
| Verified δ=.01，完整请求 | PKU | — | 23/30 (76.67%) | — |
| Verified δ=.01，完整请求 | HelpSteer2 | — | 0/4 (0.00%) | — |
| Verified δ=.01，完整请求 | UltraFeedback | — | 0/1 (0.00%) | — |

完整请求的静态配置中，HelpSteer2 的同题异答占命中 494/534=92.51%，UltraFeedback 为 696/790=88.10%。其中整向量标签分别只有 12.96% 和 8.76% 相同，因此同题并不足以复用 reward。仅嵌入问题进一步提高这类命中，并降低标签保真度。**本次 BGE-small-en-v1.5、480-token 分块加权均值的完整输入编码**也没有消除问题，候选差异仍可能被共同指令、长上下文、评分模板稀释；这不代表所有 embedding 或候选敏感编码都必然如此。

Verified 的 HelpSteer2/UltraFeedback 命中只有 4/1 条，不能用这些小分母估计长期的命中质量；短流冷启动与实际流量分布还需要单独评估。δ 与条件于命中的错误率分母不同，不能根据上述几条命中宣称误差控制理论被推翻。

## 具体错误复用

以下均来自 Static τ=.95、完整请求向量。source index 为原始 Hugging Face split 中的零基行号；completion index 为该行内的零基候选编号。similarity 为本实验完整请求 embedding 的余弦相似度。片段用于说明碰撞，不代表我们复核并认可了所有原始评分。

### HelpSteer2：同一礼物排序题，错误答案继承高评分

命中方向为 source **14974 → 14975**，余弦相似度 **0.9556478262**，相同 instruction、不同候选。

题目给出多位朋友收到的礼物，要求按总数量升序、相同时按字母顺序排列；其中 Leia 和 Anna 的数量没有给定。

- 缓存中的候选指出：“Only six of the friends can be sorted properly because the number of gifts Leia and Anna received is unknown.” 随后按已知数量列出 Elsa 3、Aurora/Belle/Jasmine 4、Ariel/Merida 6。
- 当前候选自行补出 Anna 3、Leia 5，先后给出彼此冲突的排名，最后也未按升序列出。

按 **(coherence, complexity, correctness, helpfulness, verbosity)** 顺序，原始目标标签为 **(0,1,0,0,4)**，缓存却返回 **(4,1,4,4,2)**。correctness 和 helpfulness 均偏差 4 分，已经覆盖该维度 0–4 的整个范围；这不只是轻微措辞或单分差异。

### UltraFeedback：长阅读上下文中，短答案的一个实体差异被合并

命中方向为 source **31954 completion 0 → completion 2**，余弦相似度 **0.9884274602**，相同 instruction、不同候选。

阅读题问：“What is Macy's favorite room?”

- 缓存中的候选：**“The Recreation Room.”**
- 当前候选：**“The Media Room.”**

按 **(helpfulness, honesty, instruction_following, truthfulness)** 顺序，原始目标标签为 **(4,5,5,5)**，缓存返回 **(1,1,1,1)**。两段短答案对同一长题目的最终选择不同，本次整体向量仍达到约 .988 相似度。此例说明：在当前 BGE 配置下，即使保留完整候选，固定一个较高的整体余弦阈值也不能保证公开 judge 标签可复用。

同一源行的 completion 3（“The answer is ‘The Media Room’.”）也命中了 completion 0，相似度 **0.9905096889**，原始标签同样为 (4,5,5,5)。升到 .99 仍不能拦住这一具体错误。

### HelpSteer2：不同用户意图的通用开场白发生碰撞

命中方向为 source **1188 → 20274**，余弦相似度 **0.9551978111**，instruction 不同。

- 缓存请求：“Hey buddy! Can you help me shell out a business idea today?” 候选：“Sure, I can help. What kind of business are you interested in starting?”
- 当前请求：“Hello”。候选：“Hello, I'm Open Assistant. Is there anything I can help you with?”

按 HelpSteer2 上述五维顺序，原始目标标签为 **(4,1,4,4,1)**，缓存返回 **(3,0,0,0,0)**。这也提示评分目标与具体用户意图绑定：相似的问候语气不能决定回答是否满足请求。原始标签可能存在主观性，本实验衡量的是相对这些公开标签的保真度，而非独立裁决两个答案的真实质量。

## 对下一轮实验的含义

1. 缓存键与命名空间应明确 judge/rubric 版本、system context、完整候选和顺序；对相同问题下的多候选不能直接共享评分。
2. 将 instruction/context 与候选分别表示，增加候选敏感的校验，尤其关注短答案实体、数值、否定和成对顺序。仅提高当前整体向量阈值仍可能放过严重差异；这些改进尚需受控实验验证。
3. 除整向量相等率，继续报告每维 MAE、偏好翻转率、奖励偏差与排序保持率，并在独立校准流上选阈值。对于后训练，最终还要验证训练质量和端到端吞吐。

可核查的逐请求明细位于 `outputs/events/*_source_*.csv`；完整分组统计位于 `hit_group_diagnostics.csv`。本文件仅诊断已完成的离线重放，不据此宣称已测得训练加速。
