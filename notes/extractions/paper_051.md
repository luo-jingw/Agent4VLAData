# GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation

## 基本信息
- 作者：Chi-Lam Cheang 等 13 人（按字母序），ByteDance Research
- arXiv：2410.06158v1（cs.RO），2024-10-08
- 项目页：https://gr2-manipulation.github.io
- 类型：技术报告（预印本）；真机实验为主，含 CALVIN 仿真基准

## 研究问题
- 语言条件（language-conditioned）视觉机器人操作：单一通用策略 π 以语言指令 l、观测序列 o_{t-h:t}、机器人状态序列 s_{t-h:t} 为输入，端到端输出动作轨迹 a_{t:t+k}：a_{t:t+k} = π(l, o_{t-h:t}, s_{t-h:t})
- 用大规模视频生成预训练迁移世界动态知识，缓解机器人数据稀缺，提升动作预测与泛化

## 方法
### 视频-动作联合模态的具体形式
- 单一 GPT 式 transformer 统一文本 + 图像 token + 机器人状态三路输入
- 微调阶段同时（in tandem）输出未来图像与动作轨迹：π(l, o_{t-h:t}, s_{t-h:t}) → o_{t+1}, a_{t:t+k}
- 动作轨迹由条件 VAE（cVAE）生成；输出轨迹而非单步动作（对轨迹平滑与实时性关键）
- 视频生成充当动作生成的"planner"：先生成视觉轨迹，再据此推断动作轨迹；预测动作被描述为在预测视频中"重放"轨迹
- 多视角：机器人数据含多视角（腕部 + 头部相机），模型输出每视角未来图像 + 动作轨迹；预训练 web 视频仅单视角

### 文本与视频-动作的模态划分/对齐结构
- 文本：冻结文本编码器 [6] tokenize 语言指令；仅作输入条件，无文本输出
- 图像/视频：VQGAN [7] 将每帧转离散 token；VQGAN 在互联网数据 + 域内机器人数据上训练并保持冻结；未来帧由 VQGAN 解码器解码
- 机器人状态：末端位置、旋转 + 二值夹爪状态，经线性层编码（仅微调阶段可训练）
- 两阶段对齐：1) 视频生成预训练——给定文本描述 + 视频帧，自回归预测后续帧（文本条件→视觉时序生成）；2) 机器人微调——同权重输出视频与动作

### 预训练数据（与模态对齐的关系）
- 38M 视频片段、>50B tokens
- 人类活动公开数据：Howto100M（36M）、Ego4D（1.2M）、Kinetics-700（121k）、SSV2（46k）、EPIC-KITCHENS（46k）
- 机器人数据混入：RT-1（82k）、Bridge（25k）
- 处理管线：手部过滤（MediaPipe [13]）+ 重标注（Open-Sora [14]）
- 相对 GR-1：预训练视频 0.8M → 38M

### 部署
- 7-DoF Kinova Gen3 + Robotiq 2F-85 夹爪；双相机（静态头部 + 末端安装）
- 整身控制 WBC：轨迹优化平滑 + 200 Hz 关节级执行（含碰撞约束与可操作度）

## 关键发现
- 多任务学习：105 个桌面任务（8 类技能），~40,000 条轨迹（平均 400 条/任务）；Simple 设置成功率 97.7%
- 数据稀缺：约 1/8 数据（≈50 轨迹/任务）时 Simple 成功率 73.9%，三个泛化设置全面超过 GR-1
- 泛化：Unseen Backgrounds 71.4%、Unseen Environments 71.7%（约为 GR-1 两倍）；加数据增强后 Unseen Environments 87.0%、三设置平均 74.7%；Unseen Manipulation 55.8%
- 端到端 bin picking：训练 ~94,000 轨迹/55 物体；评估 122 物体（55 seen + 67 unseen）；平均成功率 79.0%（GR-1 为 33.3%），Unseen 与两个 Cluttered 设置与 Seen 相当；可处理透明/可变形/反光物体
- CALVIN（34 任务，ABCD-D，>20,000 演示，1,000 序列 × 连续 5 任务）：单任务成功率 98.6%（GR-1 94.9%）、五任务 85.9%（GR-1 73.1%）、平均完成长度 4.64（GR-1 4.21）
- 扩展性：默认 230M 总参数（95M 可训练）；4 个尺寸（可训练 30M/95M/312M/719M），验证损失与成功率随尺寸单调改善
- 生成视频与真实 rollout 高度一致；视频预测与动作预测强相关

## 证据等级
- 低-中：arXiv 预印本 v1，未经同行评审；真机大规模实验 + CALVIN 仿真，量化结果均为作者自报，文中未见第三方复现
- 观察性证据充分（成功率、验证损失、GT/Pred rollout 对比图）

## 引用
- Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, et al. "GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation." arXiv preprint arXiv:2410.06158, 2024.

## 不确定项
- 文本编码器具体模型名文中未写明（仅引用 [6]，即 CLIP 论文）
- 观测窗口 h 与动作窗口 k 的具体数值文中未说明
- 视频 token 化分辨率、VQGAN 码本大小文中未说明
- 预训练/微调算力与训练时长文中未说明
- Unseen Manipulation 评估任务数量文中未说明
- 引言"仅 5,000 轨迹学 >100 任务"与正文"40,000 轨迹、1/8≈50/任务"的对应关系文中未直接声明（5,000 = 40,000/8，需读者换算）
- CALVIN 各基线（RT-1、MT-ACT、HULC、RoboFlamingo）的具体成功率数值文中未列出（仅图 10）
