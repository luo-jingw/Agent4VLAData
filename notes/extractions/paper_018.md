# Toolformer: Language Models Can Teach Themselves to Use Tools

## 基本信息
- paper_id: paper_018
- 年份: 2023（arXiv:2302.04761v1, 2023-02-09）
- 作者: Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom（Meta AI Research；Dessì 属 Universitat Pompeu Fabra）
- 来源: arXiv 预印本

## 研究问题
能否让 LM 以自监督方式学会"是否调用、何时调用、传什么参数、如何把 API 结果融入后续 token 预测"，弥补算术、事实性/时效信息、低资源语言、时间意识等固有缺陷，同时不损害通用语言建模能力（Abstract、Section 1）。

## 方法
- 总体（Section 2, Fig. 2）：给定纯文本语料 C，构造带 API 调用的 C* 并用标准 LM 目标微调 M。API 调用线性化：e(c)="<API> ac(ic) </API>"；带结果 e(c,r)="<API> ac(ic) → r </API>"；实际用 " [", "]", "->" 三个 token，不改词表（脚注 1）。
- 三步流程：(1) Sampling：对位置 i 计算 p_i=P_M(<API>|P(x),x_{1:i-1})，保留 p_i>τ_s 的前 k 个位置，每个位置从 M 采样至多 m 个调用；P(x) 为每工具手写的含少量示例的提示（Fig. 3、Appendix A.2）。(2) Executing：真实执行 API 得结果 r_i。(3) Filtering：加权交叉熵 L_i(z)=−Σ_j w_{j−i}·log P_M(x_j|z,x_{1:j−1})，令 L_i^+=L_i(e(c_i,r_i))、L_i^−=min(L_i(ε), L_i(e(c_i,ε)))，仅保留 L_i^−−L_i^+≥τ_f 的调用；权重 w̃_t=max(0,1−0.2t) 归一化，使调用靠近其真正有用的位置。
- 微调与推理：把保留的调用插回原文（x* = x_{1:i-1}, e(c_i,r_i), x_{i:n}）得 C*；除插入的 API 调用外 C* 与 C 文本完全相同。推理时解码到 "→" 即中断、执行对应 API、插入结果后继续（Section 2 "Model Finetuning"/"Inference"）。
- 五个工具（Section 3, Table 1）：QA（Atlas，在 Natural Questions 上微调的检索增强 LM）、Calculator（仅 +−*/ 四则运算，结果保留两位小数）、Wikipedia Search（BM25 检索 KILT Wikipedia dump）、Machine Translation（NLLB 600M，支持 200 语言，fastText 自动识别源语言，目标固定英语）、Calendar（无输入，返回当前日期）。
- 设置（Section 4.1, Appendix A/B）：C 为 CCNet 子集，M 为 GPT-J（6.7B）；默认 τ_s=0.05、τ_f=1.0、k=5、m=5；calculator 与 MT 因启发式过滤后样本少，改为 τ_s=0.0、k=20、m=10、τ_f=0.5；微调 batch 128、lr 1e-5、前 10% 线性 warmup、每工具至多 25k 例、序列长 1024、8×A100 40GB BF16 + ZeRO-3、至多 2k 步。Table 2 给出 C* 各工具调用数（τ_f=0.5/1.0/2.0）：QA 51,987/18,526/5,135；WikiSearch 207,241/60,974/13,944；Calculator 3,680/994/138；Calendar 61,811/20,587/3,007；MT 3,156/1,034/229。
- 解码改动（Section 4.2）：<API> 只要位于 top-k（k=10）就触发调用，而非仅贪心最优；每输入最多 1 次调用防死循环。

## 关键发现
- LAMA（Table 3）：Toolformer 33.8/11.5/53.5（SQuAD/Google-RE/T-REx），vs GPT-J 17.8/4.9/31.9；相对最佳基线 +11.7/+5.2/+18.6 分；超过 OPT 66B（21.6/2.9/30.1）与 GPT-3 175B（26.8/7.0/39.8）；98.1% 样本自主调用 QA 工具（0.7% 用其他、1.2% 不用）。
- 数学（Table 4）：ASDiv 40.4 vs GPT-J 7.5；SVAMP 29.4 vs 5.2；MAWPS 44.0 vs 9.9；均超 GPT-3（14.0/10.0/19.8）；97.9% 样本调用 calculator。禁用 API 时仍超 GPT-J（如 ASDiv 14.8），作者推测微调数据本身提升了数学能力。
- QA 数据集（Table 5）：WebQS 26.3、NQ 17.7、TriviaQA 48.8，超同规模基线但低于 GPT-3（29.0/22.6/65.9）；99.3% 调用 WikiSearch；作者归因于检索器简单且模型无法改写查询/浏览多个结果。
- MLQA（Table 6）：调用 MT 对各语言均有提升（使用率 63.8%–94.9%，Hindi 仅 7.3%），但因 CCNet 微调对部分语言有损，未稳定超过 GPT-J。
- 时间任务（Table 7）：TEMPLAMA 16.3 vs GPT-J 13.7（calendar 仅用 0.2%，提升主要来自 WikiSearch/QA）；DATESET 27.3 vs GPT-J 3.9（calendar 用 54.8%）。
- 语言建模（Table 8）：禁用 API 时 PPL 与 GPT-J+CC 相同（WikiText 10.3、CCNet 10.5），"无代价"。
- 缩放（Section 4.4, Fig. 4）：工具使用能力约在 775M 参数才涌现；GPT-2 124M/355M 用不用工具差别不大；模型越大，用与不用 API 的差距仍很大。
- 解码校准（Table 9）：k=1 时模型对"不用 API 表现差"的样本有校准（不调用样本得 44.3/19.9，高于全禁用 34.9/18.9），k 增大后校准消失。
- 局限（Section 7）：不能链式调用（各调用独立采样）；不能交互式使用工具（如改写搜索查询、浏览多结果）；对输入措辞敏感；样本低效（处理超 100 万文档仅得几千条 calculator 调用）；调用时不计 API 计算成本。

## 候选相关性（总体）
- 总体相关性: high
- 理由: 论文是 Toolformer 自监督训练机制与实验数字的一手来源，三步流程、公式、超参与各基准数字均可直接引用；数字为论文自报。

## 引用
- 三步机制与公式：Section 2（Sampling/Executing/Filtering/Finetuning/Inference）、Fig. 2、Fig. 3；权重函数：Section 4.1。
- 工具与超参：Section 3、Table 1；Appendix A（τ_s/τ_f/k/m 默认值与各工具实现）、Appendix B（训练配置）。
- 实验数字：Section 4.2.1–4.2.5（Table 3–7）、Section 4.3（Table 8）、Section 4.4（Fig. 4）、Section 5（Table 9–10）。
- 局限：Section 7。

## 不确定项
- 本文自述来源仅为 arXiv；paper_017 的参考文献将其列为 NeurIPS（Adv. Neural Inf. Process. Syst. 36），该信息来自他文。
- 实际微调步数未给定（"至多 2k 步，每 500 步验 PPL 选最优 checkpoint"）。
- 各下游任务数字未报告方差或多次运行结果；"few demonstrations"的具体数量与编写方式未量化。
- Table 2 的 τ_f 取值（0.5/1.0/2.0）与正文默认 τ_f=1.0、calculator/MT 用 0.5 的对应关系需读者自行对应，无统一说明。
- DATESET 为本文自建数据集（Appendix D，9,400 条模板化样本），无独立公开评测记录。
