# Unified Video Action Model

## 基本信息
- paper_id: paper_037
- 年份: 2025（arXiv v3：2025-04-24）
- 作者: Shuang Li、Yihuai Gao、Dorsa Sadigh、Shuran Song；Stanford University
- 来源: arXiv:2503.00200v3 [cs.RO]

## 研究问题
如何统一视频生成与动作预测：视频为动作提供丰富场景信息，动作揭示视频中的动力学信息。难点：动作建模要求高时间频率（稠密精细动作），视频生成要求高空间分辨率，二者冲突；先视频后动作的分层方法（UniPi 类）慢（24.07s/轨迹）且生成误差传播进动作预测；动作-only 策略缺少观测监督，易过拟合动作历史并对视觉扰动敏感。UVA 目标：联合优化视频与动作预测，既达到动作-only 精度与速度，又获得视频监督的泛化。对齐路线：视频与动作在联合潜在空间中的表示对齐（非离散量化）。

## 方法
- 联合 video-action 潜在表示：历史观测经预训练 VAE 编码器（kl-f16）→ latent map（w×h×c）→ 展平 + FC 投影为每图 N 个 d 维 token；历史动作以更高采样频率（每张图对应 L 个动作的 chunk A_t，形状 L×m），重复对齐视觉 token 数后 FC 映射为 N 个动作 token。
- Masked autoencoder 视频预测：未来观测 h 帧同样 VAE+FC → 每帧 N 个 token，训练中随机 mask（所有帧同一位置同时 mask，防跨帧信息泄漏）；历史视觉 token + 历史动作 token + 掩码未来视觉 token 通道级拼接，h 个时间步再时序拼接为 N×h 序列，送入 Transformer 融合 → 联合潜在 {Z_{t+1},…,Z_{t+h}}（每个含 N 个 token）。语言指令：CLIP 文本编码器 → 1 个 d 维 token，重复 M 次后 append（共 N×h+M 个 token），经 cross-attention 融合，取前 N×h 输出为 Z。
- 解耦 video-action 扩散（关键设计）：两个轻量扩散解码头（base size，来自相关生成工作[27]），训练中在联合潜在上联合噪声预测 L = Laction + Lvideo（按 h 求和）；动作头把 Z_{t+1} 的全部 token 经 conv + MLP 聚合为动作条件 latent，生成动作 chunk A_t；视频头逐 token 预测帧 patch，经 VAE decoder 重建帧。推理时按需跳过：策略只走动作头，视频生成只走视频头（多步自回归 token 生成）。
- 遮罩训练灵活性（5 种任务）：policy、video generator、forward dynamics、inverse dynamics、policy+planner；未用输入/输出用可学习 mask token 替换，按任务选择性施加 action/video loss，支持无动作标签视频数据。
- 实验设定：h=h'；仿真 100 denoise 步、真机 16 步；每轨迹预测 16 步、执行 8 步（OpenVLA 逐动作 8 次以对齐推理时间）；UVA 约 0.5B 参数，只用第三人称图像（无腕部视角、无 proprioception）。

## 关键发现
- Policy 仿真（Table I，50 rollout 平均）：UVA PushT 0.98、Tool 0.88、PushT-M 0.88、Libero10 0.90（0.23s）；DP-C 0.91/0.95/0.68/0.53（0.50s）；π0 仅 Libero10 0.85（0.09s）；UVA-action 消融 0.45/0.62/0.46/0.86（0.22s）。超出最强基线：PushT-M +20%、Libero10 +5%（正文）。对比模型规模（π0 3.3B，UVA 0.5B，且 π0 用更多输入模态）。
- 真机 UMI 数据集（Table II，OOD：未见环境/物体/机器人/绿色未见过夹爪）：单任务 Cup UVA 0.85 vs DP-UMI 0.95；OOD 多任务 UVA Cup 0.65、Towel 0.70、Mouse 0.80 vs DP-UMI 0.50、0.70、0.40（每任务 60 rollout）；UVA 在 Cup +15%、Mouse +40%（绝对）；速度 95ms vs 70ms。
- 视觉扰动（PushT，Table III）：UVA 0.98 → BgColor 0.35、BgObject 0.31、GoalColor 0.64；UniPi 0.42→0.31/0.36/0.40；OpenVLA 0.35→0.17/0.13/0.32；DP-C 0.91→0.12/0.21/0.17。
- 视频生成（FVD，Table IV，500 视频）：Libero10 UVA(1 步) 89.36、UVA(8 步) 51.10 vs UniPi 56.55；Cup Arrange 51.34/29.72 vs 71.37。
- 前向动力学规划（双块推挤，Table V）：DP-C 38% → UVA 引导 60% → GT 仿真器上界 75%（MPC：采样 100 条轨迹、选预测帧奖励最高、执行前 6 步）。
- 逆向动力学（UMI Cup，Mocap 真值，Table VI）：UVA 位置 0.75cm/旋转 1.11°（16 步同时预测、时间连贯）vs UniPi 逆动力学 1.92cm/2.21° vs 视觉惯性 SLAM 0.41cm/0.30°。
- 历史长度（PushT-M，Fig. 6）：历史 proprioception 加长（1→16）DP-C 性能下降，UVA 保持稳健；UVA 自述"注意力模块占推理一半时间"。
- 端到端联合建模的收益：去掉视频生成（UVA-action）在所有仿真设置全面下降；视频为联合潜在提供更细场景信息，但速度由轻量解码头保住。

## 证据等级
- 等级: A（一手实验）
- 理由: 直接目标论文；同一随机种子/初始状态、7 个公开基准（PushT、Toolhang、PushT-M、Libero10、UMI 真机、视频生成 FVD、Mcp 逆动力学），公开数据无新增采集；局限：无独立第三方复现，真机单任务低于 DP-UMI，作者给出机制解释（恢复数据 vs 长历史记忆）但未验证；UVA 未使用大规模无动作视频数据（自述 limitation）。

## 引用
- 方法与架构：Sec.III-A/III-B/III-C/III-D（p.3–5）；仿真 policy 结果：Sec.V-A、Table I（p.5）；真机：Sec.V-B、Table II（p.6）；视频生成：Sec.VI、Table IV（p.8）；前向动力学：Sec.VII、Table V（p.9）；逆向动力学：Sec.VIII、Table VI（p.10）；综合讨论与限制：Sec.IX（p.10）。

## 不确定项
- 动作 token 对齐的具体重复次数："重复 action chunk M 次以匹配视觉 token 数"与图 2 标注"Flatten as N tokens"的 M/N 关系文中未统一说明；L（每图动作数）、N、d 的具体数值正文未给出。
- 对齐粒度：UVA 不做动作离散量化，无"低秩"或流形建模表述；动作以连续 L×m chunk 重复成 token 与视觉 token 通道拼接，属实例级对齐而非语义动作语言；"动作 latent"维度未说明。
- 速度对比口径不完全一致：π0/π0-FAST 的 denoise 步数未列出；UVA 速度为其 0.5B 模型注意力瓶颈所致（自述 Flash Attention 可改进）。
- Table II 中 DP-UMI 的 Cup 单任务 0.95 使用了改善的预训练视觉编码器（CLIP ViT-B/16），UVA 无，公平性由作者声明保证。
- FVD 评估生成的视频长度与步数细节在补充材料（Supplementary §X-D/§X-E），本抽取未核对其内容。
