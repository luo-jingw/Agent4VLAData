# Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware（ALOHA / ACT）

## 基本信息
- paper_id: paper_003
- 年份: 2023（arXiv v1，2023-04-23）
- 作者: Tony Z. Zhao（Stanford）、Vikash Kumar（Meta）、Sergey Levine（UC Berkeley）、Chelsea Finn（Stanford）
- 来源: arXiv:2304.13705（文中未标注会议名）

## 研究问题
- 低成本、低精度硬件能否通过端到端模仿学习完成精细（fine-grained）双手操作任务？
- 两个核心障碍：模仿学习的误差累积（compounding errors）；人类示范的非平稳、多模态特性。

## 方法
### 数据采集（ALOHA 遥操管线）
- 双臂 ViperX 6-DoF 从臂（约 $5600/台）+ WidowX 引导臂（$3300），关节空间映射（joint-space mapping）遥操；系统总成本 <$20k（含 3D 打印件）。
- 4 台 Logitech C922x 摄像头，480×640 RGB，分别装在双腕、前部、顶部；遥操与数据记录均 50Hz。
- 记录 leader 关节位置作为 action（而非 follower 位置，因施力隐含在两者差值中、经底层 PID 执行）；observation = follower 当前关节位置 + 4 路图像。
- 每任务 50 条示范（Thread Velcro 为 100 条）；单条 episode 8-14 秒 = 400-700 时间步（50Hz）；每任务约 10-20 分钟数据，墙上时间 30-60 分钟（含重置与误操作）。
### ACT 算法
- 条件 VAE（CVAE）：encoder（BERT 式 transformer）输入 [CLS]+关节位置+目标动作序列（共 k+2 个 token），预测风格变量 z 的均值/方差（对角高斯）；encoder 训练时省略图像输入；测试时 z 置零（先验均值）。
- decoder（策略）：ResNet18 将每张 480×640×3 图像编码为 15×20×512 特征图，展平为 300×512，4 张共 1200×512，加 2D 正弦位置编码；拼接关节位置（14→512 线性投影）与 z（32→512），transformer encoder 输入共 1202×512；transformer decoder 经交叉注意力输出 k×512，MLP 投影到 k×14（双臂各 7 DoF 绝对关节目标位置）。
- 动作分块（action chunking）：每 k 步观察一次、一次预测 k 步动作序列，有效视界降 k 倍；k=100（Appendix D，Table III）。
- 时序集成（temporal ensembling）：每步都查询策略，同一时间步的多个重叠预测按 wi = exp(−m·i)（w0 为最旧动作的权重）加权平均；m 越小新观察融合越快。
- 损失：L1 重构 + β·KL 正则，β=10；约 80M 参数；单卡 RTX 2080 Ti（11G）训练约 5 小时，推理约 0.01s。

## 关键发现
- 6 个真实任务达 80-90% 成功率，仅约 10 分钟示范。
- Table I（子任务逐步成功率，格式 脚本数据|人类数据）：Slide Ziploc 97|82、Slot Battery 90|60、Transfer Cube 86|50、Bimanual Insertion 93|76；正文 V-C 另称 Slide Ziploc 与 Slot Battery 最终成功率 88% 与 96%（与表中 82/60 不一致，见"不确定项"）。
- Table II 末级子任务：Open Cup 84%、Thread Velcro 20%、Prep Tape 64%、Put On Shoe 92%；BeT 全部为 0。
- 消融：k=1 成功率 1%，k=100 时 44%（无 TE，4 个设定平均）；TE 使 ACT +3.3%、BC-ConvMLP +4%、VINN −20%；去掉 CVAE：脚本数据几乎无变化，人类数据 35.3%→2%。
- 用户研究（6 人）：遥操频率 50Hz→5Hz 任务耗时增加 62%（穿扎带 33s→20s；拆杯 16s→10s），p<0.001。

## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 一手来源，直接描述"遥操数据采集（记录内容、频率、维度）→训练"完整管线；但数据清洗、归一化、增强等预处理操作基本未涉及（见"不确定项"）。

## 引用
- 遥操硬件、50Hz、4 相机配置：Section III
- leader 位置作 action、observation 组成：Section IV 开头
- chunking 与 temporal ensembling 定义：Section IV-A（Figure 5、Algorithm 1/2）
- CVAE 结构：Section IV-B、Figure 4
- 维度细节（1200×512、k×14、15×20×512 等）：Section IV-C
- 数据采集数量与时长：Section V-B
- 消融数字：Section VI-A、VI-B；超参：Appendix D（Table III）
- 用户研究：Section VI-C、Appendix E

## 不确定项
- 图像预处理（resize、归一化、随机增强等）未说明；仅描述 ResNet18 将 480×640×3 变为 15×20×512 特征图。
- 正文 V-C 称 Slide Ziploc / Slot Battery 最终成功率 88% / 96%，与 Table I 末项 82 / 60 不一致，文中未解释口径。
- Section III 称记录 50Hz，但 Appendix B 称相机 30fps，两处不一致。
- 关节位置/动作是否归一化：文中未说明。
- 真实世界示范是否包含失败片段、有无清洗/过滤：文中未说明（模拟任务明确记录 50 条成功示范；真实任务仅说 50 条示范，墙上时间含"误操作"）。
- temporal ensembling 的 m 具体取值未给出。
- 每任务数据量与成功率的关系（scaling）未报告。
