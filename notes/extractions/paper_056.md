# CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Action Models from Human Videos

## 基本信息
- 作者：Chubin Zhang、Jianan Wang（共同一作）、Zifeng Gao、Yue Su、Tianru Dai、Cai Zhou、Jiwen Lu、Yansong Tang（通讯）；清华、Astribot、香港大学、MIT
- arXiv：2601.04061v2（cs.RO），2026-06-14
- 代码：https://github.com/LinShan-Bin/OpenCLAP
- 类型：预印本；真机实验（Astribot S1 双臂）+ LIBERO 仿真

## 研究问题
- 机器人数据稀缺；LAM 用无标注人视频学习潜在动作，但潜在空间不与机器人物理动作空间显式对齐
- 既有 LAM 缺陷（本文指出的批判）：1) 潜在空间只从视觉动态（逆动力学 + 重建）学习，与机器人物理动作空间不对齐；2) 视觉纠缠（visual entanglement）——编码背景移动、物体形变等无关视觉因素而非操作技能；3) 需复杂 post-hoc 训练把视觉潜在映射到机器人控制，限制人视频→机器人执行的直接技能迁移
- 目标：把人视频视觉转移映射到物理有根基（physically grounded）、与机器人命令同构（isomorphic）的潜在动作空间

## 方法
### Act-VAE（动作量化）
- VQ-VAE 量化连续动作块（chunk 32）为动作 token 码本；选定 Nq=16、K=256（rate-distortion 拐点，PSNR 40.00 dB）；每臂 8 个 code，15/15 层编码/解码
- 动作空间 A⊆R14：双臂各含末端位置 p∈R3、欧拉角 θ∈R3、夹爪开度 g∈R1
### VD-VAE（视觉-动态对齐）
- 冻结 DINOv3 视觉骨干提取 ft、ft+H；逆动力学编码器把转移 (ot, ot+H) 分解为 action-relevant 流 zv,a 与 action-irrelevant 流 zv,i；前向解码器在特征空间重建未来特征
- zv,a 用冻结 Act-VAE 码本 Cact 量化；zv,i 用可学习环境码本 Cenv 量化；对 zv,i 施加 L1 稀疏正则（λreg=0.5）
- SigLIP 对比损失（λcon=0.1）对齐 zv,a 与参考 z+a：机器人样本 z+a=sg(Eact(agt))；人视频用 EMA 教师编码器构造 stop-gradient 伪正样本
- 总损失：LVD = Lrec + λvq·LVQ + λcon·Lcontrastive + λreg·∥zv,i∥1
### CLAP-NTP（token 空间策略）
- Qwen3VL-4B 骨干，自回归预测 subtask + action token 序列；机器人真值 za 与人视频伪 token ẑa 联合训练；150,000 步，峰值 lr 5×10⁻⁵
### 后训练（CLAP-RF + KM）
- CLAP-RF：DiT 型 Rectified Flow 动作头，cross-attend 到 NTP 的 KV cache，预测连续动作块（低延迟）
- KM（Knowledge Matching）：对冻结 NTP 参考模型施加 reverse KL 正则 LKL，防止灾难性遗忘
- 两阶段：Stage 1 全数据 NTP 适配 3 epochs（lr 2×10⁻⁵）；Stage 2 仅机器人轨迹训 RF+KM 5 epochs（lr 1×10⁻⁴）

## 关键发现
- 真机 5 任务任务均值：CLAP-RF 62.7% > π0.5 60.0% > CLAP-NTP 58.7% > π0 54.0% > UniVLA 35.0%
- 扰动鲁棒性（背景/光照/新物体）均值：CLAP-RF 70.0% vs π0.5 56.7% vs π0 46.7% vs UniVLA 16.7%
- 人视频微调泛化：Make Bouquets (OOD) 10%→45%；PnP (OOD) 70%→85%；对比 UniVLA 5%→10% 与 50%→55%
- LIBERO 泛化模型（单模型训 4 套件）：CLAP-RF 平均 97.2%，超过 π0.5/FLOWER（各 96.9%）、π0（86.0%）、SmolVLA（88.8%），接近 X-VLA（98.1%），略超 specialist OpenVLA-OFT（97.1%）
- 失败模式：加入人视频后 attempt failures 4→1（PnP OOD）、8→0（Make Bouquets OOD），20 次尝试/设置
- 消融：去人视频平均降 12.5%（60.0%→47.5%），Make Bouquets (OOD) 崩至 5%；去对比损失 60.0%→53.8%；WiLoR 估计手姿态直接映射动作不可靠（失败增加）
- 推理延迟（H100，224×224 输入）：CLAP-NTP 382.0 ms（4.5B 参数），CLAP-RF 70.1 ms（6.0B），π0 45.8 ms，OpenVLA 219.8 ms

## 证据等级
- 低-中：arXiv 预印本 v2，未经同行评审；真机（5 任务）+ LIBERO（500 trials/套件）+ 消融，结果均为作者自报，未见第三方复现
- 观察性证据：成功率、失败模式分类、t-SNE、3D 轨迹投影（仅 Astribot 数据，因其他数据集缺相机外参）

## 引用
- Chubin Zhang, Jianan Wang, Zifeng Gao, Yue Su, Tianru Dai, Cai Zhou, Jiwen Lu, Yansong Tang. "CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Action Models from Human Videos." arXiv preprint arXiv:2601.04061, 2026.

## 不确定项
- 未报告置信区间/统计显著性检验；真机每任务仅 10–20 次试验
- CLAP-NTP 训练时长约 3,800 小时（8×A100 估计），文中标注为估算
- 引文 [59] 标注为"AgiBot World Colosseo"，正文数据集章节写"AgiBot World Beta"，两者关系文中未说明
- 各损失权重 λvq 数值文中未说明（仅 λcon=0.1、λreg=0.5、β=1.0）
