# Latent Action Pretraining from Videos（LAPA）

## 基本信息
- 作者：Seonghyeon Ye、Joel Jang（共同一作）、Byeongguk Jeon、Sejune Joo、Jianwei Yang、Baolin Peng、Ajay Mandlekar、Reuben Tan、Yu-Wei Chao、Yuchen Lin、Lars Liden、Kimin Lee、Jianfeng Gao、Luke Zettlemoyer、Dieter Fox、Minjoon Seo（KAIST / UW / Microsoft Research / NVIDIA / AI2）
- 会议：ICLR 2025
- arXiv：2410.11758v2（cs.RO），2025-05-15
- 项目页：latentactionpretraining.github.io

## 研究问题
- 能否在无 ground-truth 机器人动作标签的前提下预训练 VLA（首个此类无监督方法），利用互联网级视频（含人类操作视频），克服跨环境、跨 embodiment 差距
- 三个子问题：Q1 跨任务/环境/embodiment 差距下表现；Q2 是否优于用真值动作预训练的多 embodiment 先验；Q3 能否仅靠人类操作视频

## 方法
### Latent action 的定义与学习
- 定义：视频相邻两帧 xt 与 xt+H（固定窗口 H）之间的离散量化隐动作 zt；量 = 帧间状态变化（tokenized atomic action，类比 BPE），非计划；不需要预定义动作先验（末端位姿/关节层级），端到端由"相邻帧的 delta"优化得出
- 量化模型（Stage 1）：编码器 (xt, xt+H)→zt（作 inverse dynamics model），解码器 (zt, xt)→重构 xt+H（作 forward dynamics/world model）；VQ-VAE 目标；C-ViViT tokenizer 变体（编码器含空间+时间 transformer，解码器仅空间 transformer，仅两帧输入）；相对 GENIE 用 cross attention（而非 additive embedding）接入 zt；NSVQ 防梯度坍塌 + 解码时对 xt patch embedding stop-gradient + 早期 codebook replacement；latent action 为 s 个序列、|C| 码本词汇
- Latent pretraining（Stage 2）：7B Large World Model（LWM-Chat-1M）VLM 行为克隆：给定语言指令 + 当前帧 xt 预测 zt；新挂单层 MLP latent action head（vocab |C|）；冻结视觉编码器、解冻语言模型；动作生成空间 84（vs OpenVLA 2567）
- Action finetuning（Stage 3）：丢弃 latent action head，换动作 head 输出 delta 末端动作（逐维等频分箱离散化，沿用 Kim et al. 2024 / Brohan et al. 2023）；小标注集：每任务 150 条轨迹

### 与 raw action 的关系
- latent action 完全从观测导出，不经过真值动作（区别于 Lynch 2020 / Jiang 2023 / Lee 2024 / Mete 2024 的"真值动作转 latent"路线）；只有微调阶段建立 latent→raw 映射；VQ 解码器可作世界模型做纯神经闭环 rollout

### 性质要求
- controllability / consistency 无形式化定义；定性证据：同一 latent action 跨 embodiment/环境在解码器中产生相似重建动作（共享表示空间）；每个 latent action 对应一个语义动作（如 [1,1,3,2] 对应向左下移动）；环境中心性（含物体与相机运动）

## 关键发现
- 真机 3 任务平均成功率（每模型 54 rollouts）：LAPA(Open-X) 50.1% > OpenVLA(Open-X) 43.9%；LAPA(Bridge) 36.8% > ActionVLA(Bridge) 32.6% > OpenVLA(Bridge) 30.8% > Scratch 21.2%；LAPA(Human) 34.0% > OpenVLA(Bridge)
- 超 OpenVLA +6.22%；弱点在抓取（pick 任务早抓），但 reaching 83.33% vs OpenVLA 66.67%
- Language Table 平均成功率%（±StdErr）：in-domain LAPA 62.0±8.7(seen)/49.6±9.5(unseen)，ActionVLA 77.0±3.5/58.8±6.6；cross-task 73.2/54.8；cross-env 33.6/29.6（Scratch 15.6/15.2）
- SIMPLER（仅人类视频预训练）：LAPA 52.1% > VPT 45.8% > Scratch 34.4% > UniPi 0.7%
- 预训练效率：272 H100-hours（8×H100，34h，batch 128）vs OpenVLA 21,500 A100-hours，约 30-40 倍；LAPA 单 epoch 即最优（ActionVLA 3 epochs、OpenVLA 30 epochs）
- 数据：Something-Something v2 220K 人类操作视频；Bridgev2 181k 轨迹（in-domain 预训练）、440k 真机轨迹（cross-env 预训练）
- Scaling 消融（SIMPLER）：模型 30→300M、数据 10%→100%、latent 序列长度 1→9、vocab 2→8 均提升；主实验 latent 生成空间固定 84；最优 latent 空间规模依赖预训练数据动作维度复杂度
- 可视化：LAPA 输出 + VQ 解码器可闭环预测轨迹（"take the broccoli out of the pot"），潜力作通用机器人世界模型 / 测试时扩展

## 证据等级
- 中-高：ICLR 2025 同行评审；仿真（Language Table、SIMPLER）+ 真机（Franka）；主要结果带 StdErr；与用真值标签的 SOTA（OpenVLA）直接对比；缩放消融系统

## 引用
- S. Ye, J. Jang, B. Jeon, S. Joo, J. Yang, B. Peng, A. Mandlekar, R. Tan, Y.-W. Chao, Y. Lin, L. Liden, K. Lee, J. Gao, L. Zettlemoyer, D. Fox, M. Seo. "Latent Action Pretraining from Videos." ICLR 2025. arXiv:2410.11758.

## 不确定项
- 窗口 H、码本大小 |C|、序列长度 s 的具体数值正文未给出（标注在附录，所给文本未见）
- 图 5(a) 模型尺寸轴 30/75/150/300 的单位正文未标（应理解为 M 参数，原文未写明）
- "84"动作生成空间的构成（|C| 与 s 的组合）文中未展开
- 人类视频预训练真机实验所用 SSv2 子集规模与筛选方式文中未说明
- "第一个无监督 VLA 预训练方法"的原创性声明依赖与同期工作（IGOR 等）的时间关系，文中未对比细节
