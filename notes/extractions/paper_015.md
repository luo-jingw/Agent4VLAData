# AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems

## 基本信息
- paper_id: paper_015
- 年份: 2025（arXiv:2503.06669v4, 2025-08-04）
- 作者: Team AgiBot-World（香港大学、AgiBot Inc.、上海创新研究院、上海 AI Lab 联合；正文无个人署名，贡献名单见附录 CONTRIBUTIONS）
- 来源: arXiv 预印本（v4）；按 CHANGELOG 为 IROS camera-ready 更新

## 研究问题
如何通过扩大真实世界机器人数据规模来解决通用操作问题；核心诉求之一是"用标准化采集管线 + 人机回环验证保证数据高质量与多样性"（Abstract）。

## 方法
- 平台：4000+ 平米采集设施，5 大域（domestic/retail/industrial/restaurant/office），100+ 台 AgiBot G1 双 7-DoF 臂人形机器人；8 相机（RGB-D + 鱼眼），30 Hz 记录。
- 采集管线三阶段（Section III.B, Fig. 2）：(1) 预采集验证任务可行性并制定采集标准；(2) 遥操作员按标准正式采集，本地做有效性校验（如检查缺帧），确认完整后上传云端；(3) 后处理阶段标注员核对每段是否符合阶段 1 标准并加语言标注。
- 质量相关处理步骤（明确写出的）：本地缺帧等有效性校验 → 云端人工核对 → 丢弃不合格数据（Fig. 2 有 Discard 分支，无数量）；失败恢复轨迹保留并人工标注失败原因与时间戳；人机回环：小批量采集→训策略→部署评估→按表现修订采集协议；引入后处理步骤"消除空闲帧"（idle frames）。
- 标注体系：任务级语言指令 + 子步骤 key-frame/instruction 标注 + item/scene/skill（子任务切分）标注（Table I 注）。

## 关键发现
- 规模：最新版 1,001,552 条轨迹，总时长 2976.4 小时，217 个任务，87 项技能，106 个场景（Section III 开头）；3000+ 物体；多数轨迹 30–60 秒，最长超 2 分钟。
- 失败恢复数据约占数据集 1%（"approximately one percent"），保留而非丢弃。
- 质量消融（Section V.E）：RDT 在 Wipe Table 上用 verified 528 条 vs unverified 482 条微调，更小的人工验证集带来 0.18 completion score 提升（Fig. 7b）。
- 缩放规律：10% alpha → 100% alpha → beta，轨迹数 9.2k–1M，Pearson r = 0.97 幂律缩放（Section V.D）。
- 数据价值：AgiBot World alpha（约 236h）相比 OXE（约 2000h）成功率更高；预训练于本数据集较 OXE 平均提升 in-distribution 0.30、OOD 0.29（Section V.B）。
- alpha 为早期子集，约占 beta 轨迹数的 14%；2025-01 发布的 alpha 含 92,214 条轨迹（约 10%）。
- GO-1（ViLLA）：LAM（k=4 个离散隐动作 token，VQ-VAE）→ latent planner（InternVL2.5-2B，24 层）→ action expert（扩散，H=30 动作块）。

## 候选相关性（总体）
- 总体相关性: high
- 理由: 论文直接描述了数据采集到发布的质量控制管线（含消融实验量化质量筛选收益 0.18），直接回答调研问题中的清洗/过滤与质量评分处理问题；数字为论文自报。

## 引用
- 管线与质量控制：Section III.B（Data Collection: Protocol and Quality）、Fig. 2；失败恢复数据：Section III.B "Failure recovery" 段；人机回环与空闲帧后处理：Section III.B "Human-in-the-loop" 段。
- 规模统计：Section III 开头（1,001,552 条、2976.4h、217/87/106）；Section III.C、Fig. 3；Table I。
- 质量消融：Section V.E、Fig. 7(b)；缩放：Section V.D、Fig. 7(a)。
- 标注说明：Table I 下方注（"Extensive human annotations ... item, scene, skill (sub-task segmented), and task-level"）。

## 不确定项
- 未给出"verified/不合格"的具体判定标准或检查清单（只举例缺帧检查、是否符合阶段 1 标准）；Discard 比例未报告。
- 未定义任何数值化质量评分；次优演示无评分机制，唯一显式处理是保留失败恢复数据（约 1%）并标注失败原因。
- "unverified"（482 条）与"verified"（528 条）两组的划分流程未描述，也未说明二者是否有重叠数据。
- 全文无个人作者署名，只有团队与机构；个人贡献见附录。
- 采集标准（阶段 1）的具体内容未公布；标注员数量、标注通过率未报告。
- 未提及发布前是否有额外的离线过滤（如异常值、非稳态数据清洗）步骤。
