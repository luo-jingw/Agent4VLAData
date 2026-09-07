# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

## 基本信息
- paper_id: paper_008
- 年份: 2023（NeurIPS 2023 Datasets and Benchmarks Track；arXiv:2306.05685v4，2023-12-24）
- 作者: Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica（UC Berkeley / UC San Diego / CMU / Stanford / MBZUAI）
- 来源: NeurIPS 2023，Track on Datasets and Benchmarks

## 研究问题
现有基准（MMLU、HELM）无法区分经过对齐的模型与基座模型，与用户感知存在偏差；人类评估是金标准但慢且贵。论文系统研究能否用强 LLM（如 GPT-4）作为"judge"评测聊天助手在开放、多轮问题上的回答，并考察其与人类偏好的一致性及其局限（position/verbosity/self-enhancement bias、数学推理评测能力不足），同时引入 MT-bench 与 Chatbot Arena 两个带人类偏好的基准。

## 方法
- 三种 judge 形式（Section 3.1）：pairwise comparison（选 A/B 或平局）、single answer grading（1–10 分）、reference-guided grading（先给参考解）。
- MT-bench：80 道多轮（两轮）问题，8 类（writing、roleplay、reasoning、math、coding、extraction、STEM、humanities），每类 10 题；6 个模型（GPT-4、GPT-3.5、Claude-v1、Vicuna-13B、Alpaca-13B、LLaMA-13B），58 名专家级标注者（多为研究生），共约 3K 票。
- Chatbot Arena：众包匿名对战平台，运行一个月收集约 30K 票，随机采样 3K 单轮票（2114 个唯一 IP）与 LLM judge 对比。
- 一致性（agreement）定义为两名随机抽取的不同 judge 在随机问题上投相同答案的概率；另有 human-majority 计算方式。
- 缓解方法（Section 3.4）：交换位置（保守策略：两次评判一致才算赢，否则平局）、few-shot judge、CoT judge、reference-guided judge、微调 judge（Vicuna-13B）。

## 关键发现
- 一致性数字（Section 4.2）：MT-bench 上 GPT-4 pairwise 与人类专家在非平局设定 S2 下达成 85%（第一轮 859 票、第二轮 864 票），高于人类之间的一致性 81%（第一轮）/82%（第二轮）；含平局的 S1 设定下为 66%。Chatbot Arena 上 GPT-4 与人类 S2 一致性 87%（1944 票），S1 为 64%（3066 票）。摘要称"超过 80% agreement，与人类之间一致水平相当"。
- 当人类与 GPT-4 意见不同时，人类认为 GPT-4 判断合理的比例是 75%，并愿意改票的比例是 34%。
- 一致性随模型对实力差距增大而上升，从约 70% 到接近 100%（Figure 2）。
- Position bias（Section 3.3，Table 2）：交换答案顺序后判定一致率 GPT-4 65.0%、GPT-3.5 46.2%、Claude-v1 23.8%；多数 judge 偏好第一个位置（GPT-4 偏向第一 30%、GPT-3.5 50%、Claude 75%）；Claude 还表现出对名字"Assistant A"的偏好。position bias 在写作/STEM/人文等开放类更明显（如 humanities 一致率仅 36%），当两模型实力差距大时几乎消失（GPT-3.5 vs LLaMA-13B 一致率 98.8%）。few-shot 将 GPT-4 一致率从 65.0% 提升至 77.5%。
- Verbosity bias（Section 3.3，Table 3）："repetitive list"攻击（23 个含编号列表的答案，用无新增信息的冗长改写加在列表前）下失败率：Claude-v1 91.3%、GPT-3.5 91.3%、GPT-4 8.7%。
- Self-enhancement bias（Section 3.3）：GPT-4 给自己 +10% win rate，Claude-v1 +25%，但 GPT-3.5 不偏向自己；因数据有限，论文明确表示"无法确定模型是否存在 self-enhancement bias"。
- 数学/推理评测（Section 3.3–3.4，Table 4）：GPT-4 即使能独立解对题，也会被给定答案误导；10 道数学题的误判率：default 14/20、CoT 6/20、reference-guided 3/20（即 70%→15%）。
- 多轮评测（Section 3.5）：把两轮拆成两个 prompt 会导致 judge 定位不到上一轮回答而出错；把完整对话放进单个 prompt 并提示关注第二轮，可显著缓解。

## 证据等级
- 等级: direct
- 理由: 论文直接、系统性地度量了 LLM-as-judge 与人类评分的一致性（85% / 87% / 81%）及三类偏差的量化证据，是"agent/LLM 用于质量评分"的直接经验证据。

## 引用
- 动机与基准：Section 1–2；一致性数字：Section 4.2、Table 4/5/6（第 7–8 页）
- Position bias：Section 3.3、Table 2（第 5 页）；Verbosity bias：Table 3；Self-enhancement：Section 3.3（第 5 页）
- 数学评测失败率：Table 4（第 5 页）；多轮评测：Section 3.5（第 6 页）

## 不确定项
- Self-enhancement bias 未定论（文中明言数据有限无法确定）。
- 论文只覆盖 helpfulness，未覆盖安全/诚实维度。
- 一致性数值高度依赖投票包含/剔除平局的口径（S1 vs S2），且 human-human 一致性可能被低估（Appendix D.3 有说明）。
- 论文评测的是"模型输出质量"，未直接处理"数据/数据集本身的质量评分或策展"场景。
- Position bias 的来源（训练数据还是自回归架构）未研究，留作未来工作。
