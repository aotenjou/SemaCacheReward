# LLM Judge 缓存实验的数据集与复现

本轮选择三套与后训练直接相关的公开数据，覆盖候选比较、单回答奖励评分、合成质量反馈。`work/data` 中是固定抽样的离线请求重放 pilot；它不代表真实在线训练调用序列，也不用于估计生产环境命中率。

| 数据集 | 选择依据与 oracle | 本轮规模 | 许可证 |
| --- | --- | ---: | --- |
| [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | 两个候选的 helpfulness / harmlessness 人工偏好，oracle 为 `better_response_id`、`safer_response_id` | 2,000 个双答比较请求；标签全部完整 | CC-BY-NC-4.0 |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | 用于 reward model 训练的人工评分；oracle 为 helpfulness、correctness、coherence、complexity、verbosity 五个 0–4 分值 | 2,000 个单答评分请求；标签全部完整 | CC-BY-4.0 |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | GPT-4 细粒度反馈，可直接复用公开 judge 标签；oracle 为 instruction_following、truthfulness、honesty、helpfulness 四个 1–5 分值 | 500 条指令 × 4 个候选 = 2,000 请求；1,917 条四维完整 | MIT |

PKU/HelpSteer2 是人工 oracle；将其转为规范化 judge 请求，并不意味着已采集原始 LLM judge 调用。UltraFeedback 虽有 GPT-4 反馈，本脚本也没有声称复原其原始完整 API prompt。三套均使用简洁、固定版本的 rubric，适合比较缓存策略的覆盖与标签复用误差。

## 当前抽样

默认 `--sampling spread`：通过 HF datasets-server 获取 `default/train` 的全 split 行数，在整个行范围内等距取页，每页 100 行。PKU、HelpSteer2 各取 20 页，UltraFeedback 取 5 页后展开候选，均以原始全局 `source_index` 排序。页起点对齐 100 行边界，保证不越界且不重复。这是 systematic page sample，**不是 IID 随机抽样**。

| 数据集 | 全 train 行数 | 选定原始页起点 |
| --- | ---: | --- |
| PKU | 73,907 | 0, 3900, 7800, 11700, 15500, 19400, 23300, 27200, 31100, 35000, 38800, 42700, 46600, 50500, 54400, 58300, 62200, 66000, 69900, 73800 |
| HelpSteer2 | 20,324 | 0, 1100, 2100, 3200, 4300, 5300, 6400, 7500, 8500, 9600, 10600, 11700, 12800, 13800, 14900, 16000, 17000, 18100, 19200, 20200 |
| UltraFeedback | 63,967 | 0, 16000, 31900, 47900, 63800 |

UltraFeedback 抽中的指令来源为 `evol_instruct`、`flan_v2_niv2`、`flan_v2_p3`、`sharegpt`、`ultrachat` 各 100 条（每个来源展开为 400 个请求）。这是本轮样本的实际分布，不代表全库分布，也没有覆盖全库所有来源。

UltraFeedback 中 83 条请求的 honesty `Rating` 原值为 `N/A`，对应字段保留为 JSON `null`；不推断、不补零、不删除整条样本。计算复用误差时应跳过缺失维度并报告有效维度数量，或另报 complete-case 覆盖率。原始 `source_index=77, completion_index=0` 的回答为空字符串，原样保留；metadata 记录其位置。其余候选均非空。三套完整 prompt 的精确唯一数量依次为 2,000、2,000、1,998。

正式评分实验统一使用完整标签子集：PKU、HelpSteer2 各 2,000 条，UltraFeedback 1,917 条；所有缓存策略和两种顺序都排除相同 83 条缺失记录，避免 Verified 学习 `null == null`。上述“保留”指原始 trace 快照，评分 workload 通过索引筛选，不改写原始数据。筛选后 UltraFeedback 有 494 条不同指令、1,915 个不同完整输入，五类来源不再保证各 400 个评估请求；完整案例筛选可能引入选择偏差。

旧的前缀抽样快照已原样保存在 `work/data_prefix`，当前实验应读取 `work/data`。

## 请求格式与输入边界

每行 JSONL 包含 `dataset`、`source_index`、可选的 `completion_index`、`instruction`、`candidates`、`context`、`rubric`、`oracle`、`prompt`。`prompt` 是仅包含 `rubric/context/instruction/candidates` 四个字段的 canonical JSON（UTF-8、键排序、紧凑分隔符）。UltraFeedback 的 `context.custom_system_prompt` 保留生成候选时的 system prompt；评分、偏好标签、critique、rationale 均不放入输入。这里的 rubric/context 版本必须和候选一起参与缓存判定。

读 JSONL 使用逐行迭代，不要先对文本调用 `splitlines()`，避免原始候选中的 Unicode 行分隔字符被拆开。脚本也转义了 U+2028/U+2029。

## 复现命令

在任务根目录执行（Python 3.10+，依赖 `requests`）：

```powershell
python outputs/prepare_judge_traces.py --sampling spread
```

输出在 `work/data/{pku,helpsteer2,ultrafeedback}.jsonl`，完整来源、修订号、所有原始索引、raw 页 SHA、缺失位置与统计在 `work/data/metadata.json`。脚本使用 `/rows` API、4 个下载线程、100 行分页、失败重试；遇 429 时等待后重试。相同 revision/采样计划的已存 raw 页可复用以恢复中断下载。

完全离线重建并核验已保存快照：

```powershell
python outputs/prepare_judge_traces.py --offline
```

需要复现最初的前缀抽样，可指定单独目录：

```powershell
python outputs/prepare_judge_traces.py --sampling prefix --output-dir work/data_prefix_rebuilt
```

HF `/rows` 返回当前已处理的数据视图，无法保证按 Git revision 固定读取；脚本记录下载前后 Hub revision 并要求二者一致，同时保存所有原始页以支持精确离线复现。Hub revision 一致也不是处理视图 revision 一致的强证明。本次下载日期为 2026-10-20；成功恢复执行耗时 9.545 秒（复用了已有页，不能当作 fresh 下载耗时），离线重建耗时 0.550 秒。

## 当前快照身份

| 数据集 | HF revision | Trace SHA-256 |
| --- | --- | --- |
| PKU | `9421ffafec3fa40a1f1a7d567b4d525079477ecb` | `b86b3a5090b491e6183c34e65d3f1a65aa2d93fbbd0f016b1aa3cde99c8a75f3` |
| HelpSteer2 | `990b2711a36180dd19d9c94b8627844866f8982a` | `7469f20d0b293a2d90a2f869fe3c1f93cd5dd8338315ba1d11b463c8b2edcb8b` |
| UltraFeedback | `40b436560ca83a8dba36114c22ab3c66e43f6d5e` | `2619aeeec160c4ad3875982fabe1e01a3b2b4da3c2c0b1fb1b7e3052a11eafe3` |

已验证：三套各 2,000 行可完整解析；请求字段与 `prompt` 四字段逐条一致；原始索引单调、与 metadata 一致；offline 重建后三份 SHA-256 不变。所有 oracle 字段均没有进入 `prompt`。
