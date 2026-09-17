# VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model

## 基本信息
- 作者：Jingwen Sun、Wenyao Zhang（共同一作）、Zekun Qi、Shaojie Ren、Zezhi Liu、Hanxin Zhu、Guangzhong Sun、Xin Jin（通讯）、Zhibo Chen（通讯）；中科大、中关村学院、上海交大、清华、东方理工、国科大、南开
- arXiv：2602.10098v2（cs.RO），2026-02-14
- 代码：https://github.com/ginwind/VLA-JEPA/；项目页：https://ginwind.github.io/VLA-JEPA/
- 类型：预印本；LIBERO / LIBERO-Plus / SimplerEnv 仿真 + 真机（Franka 臂）

## 研究问题
- 既有 latent-action 预训练目标"学错东西"：锚定像素变化而非动作相关状态转移。四大失败模式（本文指出的批判）：
  1. 像素级目标使表征偏向外观（纹理、光照、背景杂波、视点）——高方差但低可控，与策略需掌握的可控自由度弱相关
  2. 真实人视频/野外视频中相机运动与非因果背景变化强于交互引发的状态变化，帧差式 latent action 目标被激励去编码这些主导信号，latent action 沦为"delta-frame encoder of nuisance motion"
  3. 信息泄漏：当前+未来观测输入同一模块、或未来上下文影响动作变量 → latent action 直接编码未来帧本身，坍塌为语义空洞的捷径（semantically empty）
  4. 多阶段（≥3 阶段）训练管线复杂脆弱，引入阶段间不一致
- 目标：leakage-free state prediction——预测未来 latent state 而非像素，未来帧仅作监督目标、绝不作为输入；两阶段配方（JEPA 预训练 + action head 微调）

## 方法
- 骨干 Qwen3-VL-2B（SigLIP-2 视觉编码器）；新增可学习 token ⟨latenti⟩ 与 ⟨action⟩
- 世界状态编码器：冻结 V-JEPA2 编码器 + 多视角拼接 sti = ∥v F(Iv,ti)；世界模型用其输出作监督目标（stop-gradient）
- Latent world model：自回归 Transformer（12 层 8 head，token dim 2048，每时间步 256 图像 token）+ time-causal attention（时间步内双向全注意力，跨时间步严格因果）；⟨latenti⟩ 重复 K=24/T 次（T=8 时 K=3）
- 世界建模损失 LWM = Σ(ŝtk − stk)（teacher-forcing，潜空间重建，无像素重建）；ELBO 视角下因目标编码器确定性嵌入 KL 项消失
- 机器人数据联合目标：L = LFM + β·LWM；条件 flow-matching 动作头 DiT-B（16 层 12 head，token dim 1024，去噪步数 4），⟨action⟩ 重复 32 次作条件；动作：末端 delta 位置 + delta 轴角（7 维），夹爪二值化
- 数据：预训练 Something-Something-v2（220K 人视频）+ Droid（76K 轨迹）联合 50K 步；LIBERO 微调约 2K demos（30K 步）；SimplerEnv 用 Fractal/BridgeV2；真机 100 demos/3 任务（20K 步）；8×A100；VLM/世界模型 lr 1e-5，动作头 1e-4

## 关键发现
- LIBERO：VLA-JEPA 平均 97.2%（4 套件中 2 个第一），超过 OpenVLA-OFT 97.1%、π0.5 96.9%，且用更少训练数据
- SimplerEnv：Google Robot 平均 65.2（最高）；WidowX 平均 57.3（第二，villa-x 最高）；使用不足 villa-x 1% 的训练数据
- LIBERO-Plus（7 扰动维度）：5/7 维度最佳，平均 79.5 vs OpenVLA-OFT 69.6、π0 53.6、UniVLA 42.9；Language/Light/Background/Layout 优势显著
- 人视频消融（w/o human videos）：LIBERO 96.1（几乎不降）、SimplerEnv Google 78.4（反升）→ ID/real-to-sim 场景高质量专家演示比人视频更关键；LIBERO-Plus 62.9（大降）→ 人视频主要增强鲁棒性与技能稳定性，不直接教授动作轨迹物理动力学
- 未来视界 T∈{4,8,16}：T=8 平均最高（96.1）；latent action token 数 = 帧数 − 1
- 注意力可视化（预训练 checkpoint，未微调）：LAPA latent action 关注过密视觉信息（桌面无关物体）→ 泄漏使其退化为目标图像压缩表示；UniVLA 靠文本引导缓解但过度强调语义、关注无关背景（钢笔、桌布纹理）；VLA-JEPA 聚焦机械臂/手/被操作物体
- 真机：ID 与 object-layout OOD 均最优（相对 π0/π0.5），task OOD 第二；π0.5 指令跟随更准但常违反臂安全边界，VLA-JEPA 执行更稳定；VLA-JEPA 习得重复抓取（re-grasp），π0/π0.5 无（归因于人视频含重复抓取行为）；香蕉任务 π0.5 与 VLA-JEPA 均约 50%

## 证据等级
- 低-中：arXiv 预印本 v2，未经同行评审；3 个仿真基准 + 真机（每任务 10 次试验）+ 消融与注意力可视化，结果均为作者自报，未见第三方复现
- 观察性证据：成功率（50 episodes/任务，500/套件）、注意力矩阵、行为定性分析

## 引用
- Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, Zhibo Chen. "VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model." arXiv preprint arXiv:2602.10098, 2026.

## 不确定项
- 未报告置信区间/统计显著性检验；真机每任务仅 10 次试验
- 真机 ID/OOD 具体成功率数值仅在 Figure 4 图中呈现，正文未列数字表
- 联合损失权重 β 的具体数值文中未说明
- 正文 Table 1 讨论中将 VLA-JEPA 误写为"A"（应为排版笔误）
- LIBERO 训练是否含 LIBERO-Plus 增强数据：文中明确"without incorporating the augmented dataset from LIBERO-Plus"
- 推理延迟未报告
