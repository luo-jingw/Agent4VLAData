# MUTEX: Learning Unified Policies from Multimodal Task Specifications
## 基本信息
- paper_id: paper_020
- 年份: 2023（CoRL 2023；arXiv:2309.14320v1，2023-09-25）
- 作者: Rutav Shah, Roberto Martín-Martín, Yuke Zhu（University of Texas at Austin）
- 来源: 7th Conference on Robot Learning (CoRL 2023)
## 研究问题
如何学习一个统一策略，使其能接受六种模态中的任意一种（或组合）作为任务规范：视频示范（V）、目标图像（v）、文本目标（l）、文本指令（L）、语音目标（s）、语音指令（S）；并让跨模态训练提升每一单模态的执行鲁棒性。目标策略 π(a|t,o)。
## 方法
- 两阶段训练（均与行为克隆 BC 联合）：
  1. Masked modeling：每次迭代随机采样一个或多个模态，mask 部分 token/特征，要求模型借其他模态预测被 mask 部分——文本（mask 单词，CLIP 语言模型特征，交叉熵损失）、视觉（mask CLIP 中间特征，L1 损失）、语音（mask Whisper 特征，L1 损失）；促使文本/语音引导视觉特征提取、视觉为文本/语音提供 grounding；
  2. Cross-modal matching：用 ℓ2 损失把 L/l/S/s/v 的表示拉向信息密度最高的视频表示 V（梯度不回传到视频编码器）。
- 架构：预训练模态编码器（文本/视觉用 CLIP ViT-L/14，语音用 Whisper-Small）+ 投影层（MLP 或 Transformer+mean pooling）→ Perceiver 风格 policy encoder（机器人观测 token 作 query、任务规范 token 作 key/value 的交叉注意力）→ Perceiver 风格 policy decoder（可学习 query，输出动作与 masked token 预测）；动作用 5 分量 GMM + NLL 损失。
- 数据/标注：模拟 100 任务（LIBERO-100，每任务 50 条 SpaceMouse 遥操作轨迹）；真实世界 50 任务、17 个物体、每任务 30 条轨迹（共 1,500 条，Franka Panda + SpaceMouse，平均 horizon 242）；每任务每模态 11 个任务规范。
- 标注管线：文本规范由 ChatGPT/GPT-4 生成 11 个变体后人工过滤；语音规范用文本规范经 LLM 改写后由 Amazon Polly（多说话人）合成；模拟中的视频示范用机器人遥操作（避免 real2sim 差异）、真实世界的视频示范为人类手部示范。
- 评估：80%/20% 任务划分，未见过的任务规范 + 新物体初始位置；模拟每任务 20 trials×100 任务×3 seeds（6,000 trials），真实 10 trials×5 任务（600 trials）。
## 关键发现
- 模拟（Table 1）：MUTEX 六模态成功率 50.1/53.0/61.6/63.2/40.9/46.0 vs 单模态基线 41.7/39.9/58.7/62.0/22.3/28.4，平均 +10.3%；真实（Table 2）64/58/62/64/50/60 vs 52/48/42/52/32/46，平均 +14.3%。
- 失败归因：因未理解任务规范导致的失败从 85/240（35.4%）降至 40/240（16.7%）。
- 训练消融：去掉 masked modeling 掉点最多（43.8/46.0/63.2 中的部分模态）；joint training、无 cross-modal matching 均差于两阶段顺序训练。
- 多模态组合推理：TG+SG 50.1、TG+IG 59.2、SI+VD 59.6，全模态 60.1——组合未带来额外增益（甚至低于单用 IG/VD）。
- 表示对比（Table 3）：MUTEX 50.1/53.0/61.6/63.2/59.2 优于 T5（40.0/44.0）、R3M（59.5/44.7）、VIMA（47.0）。
## 候选相关性（总体）
- 总体相关性: high
- 理由: 直接研究多模态任务规范的标注与策略学习；自建带六模态标注的数据集（含 LLM 生成文本、TTS 生成语音的标注管线），在模拟与真实机器人上给出六模态量化对比与消融。
## 引用
- §3.3 Multimodal Task Specification Dataset（数据集与标注管线，ChatGPT/Polly）；§3.1 训练两阶段；§4 Experimental Evaluation（Table 1/2/3，错误归因 35.4%→16.7%）；§5 局限（配对全模态、合成语音、同工作区视觉规范）；Appendix 6.4（GPT-4 prompt、Polly 说话人列表、11 规范/模态）。
## 不确定项
- 正文称文本规范由 ChatGPT 生成（§3.3），附录 6.4 的 prompt 标注为 "GPT4 Prompts"——所用 LLM 具体版本不一致。
- 训练用视频示范在模拟为机器人遥操作、真实世界为人类手部视频，两种来源差异对 matching 的影响未量化。
- 每任务 11 个规范在训练时如何参与采样（是否全部用于 masked modeling）仅见伪代码，无消融。
- 组合推理的退化（全模态 60.1 < 单模态最优 63.2）只给出假设，未进一步实验。
- 未报告真实世界 50 任务中哪 5 个任务用于测试（80/20 应为 40/10，文中写 "10 trials × 5 tasks"，数量与 20% 不符，未解释）。
