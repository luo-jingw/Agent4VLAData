# LeRobot: An Open-Source Library for End-to-End Robot Learning

## 基本信息
- paper_id: paper_004
- 年份: 2026（arXiv v1，2026-02-26；页眉标注 "Published as a conference paper at ICLR 2026"）
- 作者: Remi Cadene、Simon Alibert、Francesco Capuano（牛津大学，工作完成于 Hugging Face）、Michel Aractingi、Adil Zouitine、Pepijn Kooijmans、Jade Choghari、Martino Russi、Caroline Pascal、Steven Palma、Mustafa Shukor、Jess Moss、Alexander Soare、Dana Aubakirova、Quentin Lhoest、Quentin Gallouédec、Thomas Wolf（Hugging Face）
- 来源: ICLR 2026（arXiv:2602.22818）

## 研究问题
- 机器人学习生态碎片化（middleware 平台绑定、数据集格式不一、算法框架难复现）抬高进入门槛；论文交付 LeRobot：统一的开源端到端栈，覆盖中间件→数据采集/存储/流式→训练/推理。

## 方法
### 数据集格式（Section 3.2、Appendix C）
- LeRobotDataset：统一多模态 schema，自包含元数据：任务文本描述（用于过滤与语言条件策略）、机器人本体信息、采集 FPS、传感器类型等。
- 存储组织：表格记录（.parquet）+ 压缩视频（.mp4）+ 轻量元数据文件；元数据全量下载，视频/控制流按需取用。
- StreamingLeRobotDataset：IterableDataset 接口 + torchcodec 在线视频解码；远程流式，内存占用与数据集大小无关；稳态时序性能与本地加载相当（Figure 10）。
- delta_timestamps 参数：按时间偏移取相邻多帧（如 observation.images.wrist_camera: [-0.2, -0.1, 0.0]），返回 tensor 多一维（[3, C, H, W]）。
- 字段示例：observation.state、action、observation.images.<camera>（Appendix C.2 代码示例）。
### 数据层操作（论文显式提及的）
- 过滤：任务文本描述元数据"用于过滤和语言条件策略"（Section 3.2，仅此一处）。
- 归一化统计：dataset_metadata.stats 传入 make_pre_post_processors 构造 pre/post processor（Appendix D.1、D.2 示例），即存在 norm stats 接口但计算细节未展开。
- 流式取帧：parquet+mp4 分区 + 按需解码（Appendix C.1）。
- "分片""重采样""去重"等操作：正文未显式描述。
### 数据收集与上传流程
- 共享 middleware：读 leader 配置写入 follower（遥操），或直接以策略控制 follower；直接对接 FeeTech、Dynamixel SDK。
- 社区上传：截至 2025-09，16K+ 数据集、2.2K+ 贡献者；50%+ 数据集来自 SO-10X 平台（Figure 5d）；下载量前四：Panda 1,878,395 次/588 数据集/926,776 episode；xArm 1,107,329/74/450,329（Table 1b）。
### 训练与推理
- 模型：ACT（实测 52M 参数）、Diffusion Policy（263M）、π0（3.5B）、SmolVLA（450M）、VQ-BET、HIL-SERL、TD-MPC；纯 PyTorch 实现。
- 推理栈：物理解耦（远程服务器推理）+ 逻辑解耦（异步生产者-消费者，动作分块预测 a_{t:t+H−1}，重叠块经可自定义聚合函数 f 合并）。
- 代码量宣称：<100 LOC 训练、<40 LOC 部署（Section 3.3）。

## 关键发现
- 社区规模：16K+ 数据集、2.2K+ 贡献者（2025-09）；SO-10X 贡献 50%+ 数据集，但下载量与数据量领先的是 Panda/xArm（研究型集中采集平台）。
- 数据质量现象：未标注机器人平台的 "unknown" 数据集占 Other 类第一（2370 个，Table 4a）；论文未提出清洗/质检机制，仅作现象陈述。
- 流式：稳态下 Streaming 与本地加载时序相当（Figure 10）。
- 异步推理（SmolVLA + SO-100，3 个真实任务，Table 5）：成功率相当（平均 Sync 78.3 vs Async 73.3），单集平均耗时 13.75s→9.70s，60s 内搬方块 1.8→3.8 个。
- ACT 流行原因：小模型、快推理，约 50 条真实轨迹即可训练可用策略（Section 3.3）。

## 证据等级
- 等级: direct
- 理由: 调研对象（数据集格式与数据层工具）的一手来源，直接描述 schema、存储、流式与归一化统计接口；但过滤/清洗/去重等操作多为一笔带过或未说明（见"不确定项"）。

## 引用
- 数据集格式与元数据：Section 3.2
- 存储组织与流式实现：Appendix C.1
- delta_timestamps 多帧读取：Appendix C.2、D.1
- 归一化统计（dataset_stats）：Appendix D.1、D.2 代码示例
- 社区数据规模与下载量：Section 3.2、Table 1b、Table 4、Figure 5
- middleware 与遥操/控制流程：Section 3.1、Appendix B
- 推理解耦与聚合函数 f：Section 3.4、Figure 8、Appendix E
- 模型列表与 ACT 流行度：Section 3.3、Figure 7
- 异步推理评测：Appendix E（Table 5）

## 不确定项
- 调研问题中的"v2 格式"：正文未出现版本号 "v2"，仅称 LeRobotDataset，两者对应关系文中未说明。
- 过滤、分片、重采样、去重等数据集操作：正文仅显式提及"过滤"一次；其余操作未在本文描述（应查代码库/文档，文中未说明）。
- norm stats 的计算方式（mean/std 口径、覆盖哪些字段、图像与状态是否分别归一化）未展开。
- 数据清洗/质量控制机制：未描述；仅在 Table 4 讨论 "unknown" 平台标签的劣质数据现象。
- 数据记录脚本与上传流程细节未在正文展开（仅 middleware 遥操代码示例，Appendix B）。
