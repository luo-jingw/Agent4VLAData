# DiLA: Disentangled Latent Action World Models

## 基本信息
- paper_id: paper_039
- 年份: 2026（arXiv v1：2026-05-15，Preprint）
- 作者: Tianqiu Zhang、Muyang Lyu（共同一作）、Yufan Zhang、Fang Fang、Si Wu（通讯）；Peking University（清华-北京大学生命科学联合中心、前沿交叉学科研究院、IDG/麦戈文脑科学研究所、定量生物学中心、心理与认知科学学院）
- 来源: arXiv:2605.15725v1 [cs.CV]

## 研究问题
潜在动作模型（LAM）从无标签视频中推断抽象动作（IDM 逆动力学模型 + FDM 前向动力学模型），面临根本困境 "LAM Trade-off"：动作抽象（需 VQ/变分等强信息瓶颈）↔ 生成保真（强瓶颈扭曲动作潜空间固有流形、过简化；放松瓶颈则动作纠缠视觉细节）。传统解法弃 FDM、用独立预训练扩散模型生成，或借光流/深度做输入限定制。DiLA 主张 content–structure 分离（结构 = 动力学相关空间布局、内容 = 视觉外观纹理），并提出两个问题：① 脱纠缠学习与潜在动作学习能否共优化（co-evolve）；② 如何同时获得抽象潜在动作与高保真预测。对齐路线：动作潜空间的内容-结构分解与语义对齐（无监督、连续、可流形解释）。

## 方法
- 特征提取：冻结 DINOv2 编码器取 e0:t，经空间-时间 Transformer 精修（空间注意力建模全局空间依赖、时间注意力带因果掩码 + RoPE 限制历史信息流）。
- 结构通路：structure encoder 压缩 token → 结构嵌入 s0:t；IDM 以结构时差 Δs0:t−1 为输入（3D 卷积块：空间核取平移不变全局特征、时间核聚合双向上下文），输出连续潜在动作 z0:t−1（dz=256）；FDM 为轻量时空 Transformer + AdaLN-zero，残差更新 ŝ1:t = s0:t−1 + FDM(s0:t−1, z0:t−1)。时差 Δs 双重功能：信息瓶颈（强制抽象）+ 脱纠缠驱动力（预测结构 s 而非全视觉 e → 模型只把纯动力学蒸馏进潜在动作、把静态高熵细节逐出 s）。
- 内容通路：content encoder → c0:t；Mamba（慢特征分析原则）维护历史：c^mem_t = Mamba(c_t, h_{t−1})；承载 POMDP 信念更新（暂时遮挡区域、新场景未观测区域推断），使结构通路专注物理动力学。
- 融合解码器：空间注意力 Transformer 双重 cross-attention——以 ŝ 为 query，第一层以内容记忆 c^mem 为 K/V，第二层以初始嵌入 e0 为 K/V（e0 供应压缩中丢失的高频细节）；Decθ(ŝ_{t+1}, e0, c^mem_t) = ê_{t+1}；用冻结 RAE decoder 可视化。
- 潜在 rollout：直接在结构潜空间自回归 ŝ_{t+1} = ŝ_t + FDM(ŝ_t, z_t)，再融合解码（区别于在观测空间 rollout 的 AdaWorld 等）。
- 训练目标（全部自监督、teacher-forcing，无动作标签）：Ltotal = λe·Le + λs·Ls + λz·Lz + λreg·Lreg；Le=‖e_t−ê_t‖²、Ls=‖Δs_t−FDM(s_t,z_t)‖²、Lz=‖IDM(s_{t+1}−s_t)−IDM(ŝ_{t+1}−s_t)‖²、Lreg=‖z_t‖² + 正反向余弦项 + 0.5·exp(−10·σ_z)。正反向 z 余弦（组对称原理）强制逆时序转移互为相反向量；mask mt=1[‖s_{t+1}−s_t‖>τ] 滤静态帧；norms/variance 正则保持紧凑流形与信息熵。
- 数据与训练：SSv2、RT-1、RECON、LoopNav；30k 教师强制迭代 + 1k（文中另述"30k + 1k"）潜在 rollout 微调，batch 32、16 帧 clip。

## 关键发现
- 生成质量（16 帧自回归 rollout，Table 1，SSIM↑/LPIPS↓）：SSv2 0.660±0.037 / 0.356±0.027；RT-1 0.774±0.010 / 0.206±0.013。对比：LAPA 0.637/0.565 与 0.491/0.595；MOTO 0.555/0.593 与 0.762/0.284；AdaWorld(FDM) 0.625/0.576 与 0.554/0.549；AdaWorld（借外部扩散模型）0.674/0.521 与 0.634/0.429；villa-X 0.636/0.515 与 0.576/0.477；DiLA w/o content 0.594/0.450 与 0.647/0.258。作者结论：优于多数基线（SSv2 SSIM 略低于 AdaWorld 0.674）。
- 消融（Table 2，SSv2+RT-1 混合，LPIPS）：DiLA rollout 0.263±0.027 / cycle transfer 0.343±0.022 / MSE(10k iter) 0.216；w/o content（结构嵌入维 32→128 以公平容量）0.344/0.451/0.249（参数量 646.17M vs DiLA 123M）；Discrete z（NSVQ 码本 8、量化维 32，dz=512）0.334/0.442/0.262；Gaussian z（β-VAE，β=10⁻⁴，dz=256）0.346/0.434/0.265。结论：VQ/VAE 刚性先验过强，破坏潜动作空间固有低维流形。
- 线性探测（Table 3，OOD 未训练基准，MSE↓）：Franka Kitchen 0.073、Block Pushing 0.037、Push-T 0.009、LIBERO Goal 0.119；Discrete z 0.098/0.061/0.023/0.160；Gaussian z 0.125/0.102/0.041/0.190——DiLA 潜空间与真值连续控制信号对齐最好。
- 流形分析（OOD OmniObject3D 原生仿射变换，Fig. 6）：平移 → 2D 平面（小位移近原点、大位移远端，与物理运动拓扑同构）；缩放 → 关于恒等点对称；旋转 → 连续角度谱；组合 Translation+Scaling → 保留缩放拓扑的簇；Translation+Rotation 流形重叠（平移在视觉信号占主导）。导航：RECON 相对 yaw 连续谱、LoopNav 前移/转弯离散簇。
- 动作转移（Sec.4.1 + Fig. 3）：人→机器人跨实施例（含"拾起"语义转移、大视角变化）、域内转移、导航虚拟↔真机；量化用 action cycle transfer 指标（源视频推断→转移至目标 rollout→再推断并回传源，误差增量小即语义保留）。
- 内容-结构分离验证（Fig. 4）：rebinding——结构来自序列 i、内容来自序列 j，输出保留 i 的空间布局 + j 的外观（类似风格迁移）；冻结 s 于 s0 时序列完全静止 → 内容通路不含运动信息。消融（Fig. 5）：去掉 IDM+FDM 后结构嵌入含冗余内容细节、rebinding 出现纹理泄漏 → 预测瓶颈是脱纠缠的必要驱动力。
- 视觉规划（VP2 基准，MPPI，Table 4，4 次 run 平均）：DiLA 聚合 41.44 vs AdaWorld 21.54；Robosuite Push 68.00% vs 63.50%、Open Slide 15.00% vs 5.83%、Blue Button 78.33% vs 29.17%、Green Button 35.83% vs 10.83%、Red Button 20.83% vs 10.00%、Upright Block 3.33% vs 5.00%（基线为真值仿真器归一化，聚合相对值）。流程：预训练 DiLA 不变，仅训轻量 action MLP 把真值动作标签映射进潜空间代替 IDM，不作潜在动作学习。
- 行为现象（Sec.5）：目标场景不支持源动作时 rollout 强行复现源动力学（墙变大变模糊、未持物机械臂把夹爪当"投掷物"）；导航动作转移至 RT-1 场景时运动载体在臂移动与视角变化间切换。

## 证据等级
- 等级: A（一手实验）
- 理由: 直接目标论文；生成质量（4 基线 + 自自消融）、动作转移、OOD 线性探测（4 基准）、流形分析（可控 OOD OmniObject3D + 导航数据集）、VP2 视觉规划均有量化结果；关键消融（w/o IDM+FDM、w/o content、VQ、Gaussian）直接支撑其核心主张；局限：全部为仿真/合成或静态数据集，无真机遥操/物理执行；论文为 preprint（May 18, 2026），无第三方同行评审或独立复现；对比基线数量有限（LAM 领域仅 LAPA/MOTO/AdaWorld/villa-X）。

## 引用
- 问题与动机：Sec.1（p.1–2）；相关工作：Sec.2（p.2–3）；方法：Sec.3（p.3–5）；训练细节/数据：Sec.4 开头（p.5）；动作转移：Sec.4.1（p.5–6）、Fig. 3；生成质量：Sec.4.2、Table 1（p.6）；脱纠缠验证：Sec.4.3（p.6）、Fig. 4；消融：Sec.4.4、Table 2/3（p.6–8）；流形分析：Sec.4.5（p.8）、Fig. 6；视觉规划：Sec.4.6、Table 4（p.8–9）；局限与讨论：Sec.5（p.9）。

## 不确定项
- 动作潜空间与真值动作的"对齐"机制：训练完全无监督；真值对齐仅出现在 OOD 线性探测（MSE 指标）与 VP2 的 action MLP 映射中。论文未说明潜动作与物理动作的坐标/量纲对应关系。
- "低维流形"无量化维度：仅有 dz=256（Gaussian z 同），未报告有效自由度/本征维度;"紧凑流形"为定性。
- 训练超参（λe/λs/λz/λreg 具体值、τ、variance 项权重）正文未给出（Appendix A，本抽取未核对）。
- 表 1 中 DiLA SSv2 SSIM（0.660）低于 AdaWorld（0.674），作者结论写"outperforms majority"而非全部，摘要主声明需注意。
- RAE decoder 来源（Zheng et al., 2025）与 DINOv2 具体型号正文未给出；VP2 基线的"ground-truth simulator baseline 归一化聚合"口径（21.54/41.44 为相对值）未充分展开。
- action cycle transfer 指标采用 Garrido et al. 2026 的同期工作版本，其阈值/统计细节在该文，本论文未独立验证该指标可靠性。
