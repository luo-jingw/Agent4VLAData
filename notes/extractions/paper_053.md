# Learning to Act Anywhere with Task-centric Latent Actions（UniVLA）

## 基本信息
- 作者：Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, Hongyang Li（The University of Hong Kong / OpenDriveLab / AgiBot）
- arXiv：2505.06111v3（cs.RO），2025-11-03
- 代码：github.com/OpenDriveLab/UniVLA

## 研究问题
- 能否学习统一的、与 embodiment 无关的动作表示（task-centric latent actions），从无动作标签的跨 embodiment 视频（机器人 + 人类）训练通用 VLA，并高效解码部署到具体机器人
- 对 LAPA/Genie 类方法的批评：纯重建目标会编码任务无关动态（非自我 agent 运动、相机晃动、新物体出现），噪声表示损害策略预训练

## 方法
### Latent action 的定义与学习
- 定义：成对帧 {ot, ot+k} 的逆动力学推出的量化离散动作 token az∈R^{N×d}（N=4；帧间隔 k 按各数据集频率校准为约 1 秒）；量 = 帧间状态变化（约 1 秒粒度），可解码为 action chunk
- 隐动作模型：IDM 编码器 I(at|ot, ot+k)（时空 transformer，causal 掩码，接可学习动作 token 组）+ FDM 解码器 F(ot+k|ot, at)（空间 transformer，仅用量化 token 预测未来帧，不喂历史帧防记忆/过拟合）；VQ-VAE 目标，码本 |C|
- 特征空间：在 DINOv2 patch 特征空间学习（JEPA 式），最小化 ‖Ôt+k − Ot+k‖2（嵌入重构），避开像素空间的纹理/光照噪声
- Task-centric 解耦（两阶段）：
  - Stage 1：T5 语言指令嵌入 ℓ 同时给编码器与解码器；语言已提供任务语义，受码本容量约束，量化动作 ãTI 只编码任务无关动态（新物体、外部 agent、相机运动）
  - Stage 2：冻结任务无关 codebook，新 codebook VQTC 学任务中心动作 ãTC；解码 F([Ot; ãTI; ãTC])；aTC 供策略训练
- 通用策略（Stage 2）：Prismatic-7B VLM（SigLIP+DINOv2 融合视觉编码器 + LLaMA-2）；词汇表扩展 |C| 个特殊 token {ACT_1..ACT_C}；自回归最小化 next-latent-action NLL；N=4；|C|=16 时动作空间 164（vs OpenVLA 2567）
- 解码（Stage 3）：post-training 轻量 action decoder（多头注意力池化：视觉嵌入 E'v 作 query，与 latent action 嵌入交叉注意力提取 E'a，线性投影到目标动作空间）；action head 12.6M 参数，LoRA 后总可训练约 123M；解码为 action chunk（实践 chunk=12，可按 embodiment 定制）；历史 latent action（上一时间步 4 token）附加进指令作 in-context 反馈（CoT 启发）

### 与 raw action 的关系
- 预训练完全不用动作标签：机器人数据集的 action 与本体感知在预训练阶段被排除，仅用帧 + 文本指令；部署时由 action decoder 完成 latent→raw 映射（每 embodiment 一个轻量 head）

### 性质要求
- 论文未形式化定义 controllability / consistency；要求的性质为"task-centric"（与任务相关动态对齐、与任务无关动态正交），验证方式：同一 latent action 跨数据源/embodiment 的语义一致性（定性图）+ 任务无关 codebook 与任务中心 codebook 的下游成功率对比（定量，见关键发现）

## 关键发现
- LIBERO 平均成功率：UniVLA Full 95.2（Spatial 96.5 / Object 96.8 / Goal 95.6 / Long 92.0）vs OpenVLA 76.5（+18.5）、LAPA* 65.7、MaIL 83.5、Octo 75.1；Bridge-only 92.5、Human-only 88.7（超 OpenVLA 12.2 个百分点）
- R2R（VLN-CE）oracle 成功率 47.1% vs OpenVLA 17.5%、Seq2Seq 8.1%、CMA 10.8%、LLaVA-Nav 14%、NaVid 49.1%（NaVid 需全部历史帧，UniVLA 仅单帧 RGB + 历史 latent action）
- 真机（7-DoF Piper，第三视角 Orbecc DABAI，4 任务）：UniVLA Full 平均成功率 81.7% / 平均分 2.63 vs OpenVLA 38.3%/1.63、LAPA(OXE) 45%/1.95、Diffusion Policy 33.3%/1.45；+36.7%、+0.68
- 真机泛化（平均 68.9% / 2.49）：Lightning 66.7%、Visual Distractor 53.3%、Novel Object 86.7%（LAPA 28.9%、OpenVLA 20.0%）
- CALVIN ABC→D：5 任务连续完成 56.5%（OpenVLA 43.5%，+13%）、平均连续完成长度 3.80（OpenVLA 3.27）
- SimplerEnv：任务成功率 42.7%（OpenVLA 4.1%、Octo-Base 16.0%）；仅训 decoder 35.4%
- 数据效率：10% 演示数据 LIBERO-Goal 86.3% vs OpenVLA 全量 79.2%；LIBERO-Long 用 50% 数据即新 SOTA
- 数据扩展：LIBERO Bridge 92.5 → +OpenX 94.2 → +人类视频 95.2；真机平均分 +0.3（OpenX）、再 +0.28（人类视频）
- 隐动作性质消融（仅 Ego4D 人类视频预训练，LIBERO）：task-centric 88.7 > Genie 式（全视觉变化）82.3（+6.4%，Goal +13%、Long +9.8%）>> task-irrelevant 56.5（Long 仅 0.2）；task-irrelevant 的 token 预测精度也低
- 语义一致性：同一 latent action 跨 Bridge/RT-1/LIBERO 数据源呈现语义一致动作（如"Pick up things"组）；隐动作模型未训 LIBERO 数据即可标注其动作；对齐腕部视角操作与自我中心导航运动（Group C）；注意力热图集中于末端执行器与目标物体
- 预训练算力 960 A100-hours（batch 1024、32×A100、约 30h、20k 步）vs OpenVLA 21,500 A100-hours（<1/20）；Human/Bridge 预训练约 200 A100 GPU-hours；真机推理 10Hz（RTX 4090，chunk 12）
- 消融：action decoder 带视觉查询 95.2 vs 无视觉 92.5 vs 自回归（OpenVLA 式）73.6（Long：92.0 / 86.0 / 49.0）；历史 latent action 4 token：R2R +16.5%、LIBERO-Long +3.9%

## 证据等级
- 中：arXiv 预印本（v3），未经同行评审；4 个仿真基准（LIBERO / CALVIN / SimplerEnv / R2R）+ 真机，结果均为作者自报；LIBERO 每套 500 次试验、3 种子；真机含步进评分制但 rollout 次数未明示

## 引用
- Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, H. Li. "Learning to Act Anywhere with Task-centric Latent Actions." arXiv:2505.06111, 2025.

## 不确定项
- 码本大小 |C|：正文仅出现 |C|=16 一例（动作空间 164），各实验的实际 |C| 值文中未统一说明；"164"的计算方式未展开
- 帧间隔 k 按数据集而异，具体取值文中未逐一列出
- Full 预训练数据集总时长/轨迹数文中未说明（附录仅列各子集混合权重）
- 真机评估每组任务的 rollout 次数文中未明确给出（仅说明微调用 20-80 条轨迹）
- OpenVLA 在真机的动作分块推理延迟 0.18s/0.68s 为文中数值，但 UniVLA 对应的单步延迟未给出（仅 10Hz 整体频率）
