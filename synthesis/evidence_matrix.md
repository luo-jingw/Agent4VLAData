# Evidence Matrix

## RQ1 主问题：机器人数据预处理是否存在公认范式

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_004](https://arxiv.org/abs/2602.22818) | direct | 支持 | LeRobot 成为社区事实标准：统一 schema + parquet/mp4 存储 + delta_timestamps + norm stats 接口 + streaming；16K 数据集 2.2K 贡献者 | Sec 3.2, Fig 5 | 高 |
| [paper_002](https://arxiv.org/abs/2310.08864) | direct(部分) | 支持 | 跨数据集标准化范式：RLDS 格式 + canonical view + 7 维动作归一化离散化；但无质量过滤描述 | Sec IV-A | 高 |
| [paper_001](https://arxiv.org/abs/2403.12945) | direct | 支持 | DROID 完整管线：成功标记+训练剔除、众包标注、场景去重、GPT-4V 分类、自动标定质量过滤 | Sec V-A, App G | 高 |
| [paper_015](https://arxiv.org/abs/2503.06669) | direct | 支持 | AgiBot 三阶段管线：预采集验证→采集+本地校验→云端人工核对+语言标注；verified 数据带来 +0.18 分 | Sec III.B, V.E | 高 |

**RQ1 结论形态**：存在共识性"管线步骤"（记录→校验→标注→过滤→标准化→重采样/增广），
但不存在单一权威范式（在所调研文献范围内）；各步骤的判据多为项目自定，无标准。

## RQ1.1 机器人学习数据管线标准做法

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_004](https://arxiv.org/abs/2602.22818) | direct | 支持 | 数据层操作显式提及的只有过滤（按任务文本）+ norm stats；分片/重采样/去重未描述 | Sec 3.2, App D | 高 |
| [paper_003](https://arxiv.org/abs/2304.13705) | direct | 支持 | 遥操→action chunking→temporal ensembling 标签构造管线；清洗/归一化未说明 | Sec III-IV | 高 |
| [paper_001](https://arxiv.org/abs/2403.12945) | direct | 支持 | 采集协议（GUI 随机任务采样、周期性场景增强、逐条成功标记） | Sec III-B | 高 |
| [paper_016](https://arxiv.org/abs/2505.15558) | direct | 支持 | 存储/传输/加载管线工程（EBML 容器、压缩、缓存）；不含质量判断 | Sec III-IV | 高 |
| [paper_002](https://arxiv.org/abs/2310.08864) | indirect | 支持 | RLDS 标准化细节被指向项目网站 | Sec IV-A | 中 |

## RQ1.2 LfD 轨迹分割/关键帧/时间对齐

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_012](https://arxiv.org/abs/1907.03146) | indirect | 支持 | LfD 综述：演示=状态/状态-动作轨迹；轨迹分割两类（技能相似性/显著事件）；关键帧仅作为演示获取形式提及（Akgun 2012）；无预处理专章、无 DTW 专节 | Sec 6.2, 6.4, 8.2 | 高 |
| [paper_011](https://arxiv.org/abs/2603.01465) | direct | 支持 | 现代 keyframe：语义阶段完成检测（KSM）选取关键帧，链式注入 VLA；固定步长采样有 horizon-resolution 权衡 | Sec 4.2, 5.1 | 高 |
| [paper_003](https://arxiv.org/abs/2304.13705) | direct | 支持 | action chunking + temporal ensembling 是对"何时重新采样标签"的现代回答 | Sec IV-A | 高 |

## RQ1.3 数据质量过滤与次优演示去除

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_005](https://arxiv.org/abs/2606.10229) | direct | 反对(部分) | 7 种几何/统计 curation 指标的检测 AUROC 与下游 BC 提升脱钩（ρ=−0.14）；gripper timing AUROC 最高(0.804)却最差(13.3%)；时长混杂解释近 1.0 的原始 AUROC | Sec V.B, VI.A | 高 |
| [paper_014](https://arxiv.org/abs/2606.15064) | direct | 反对 | phase-localized 策展负面结果：3 任务无最优、2 任务最差；信号稀释机制（LIFT 信号权重稀释至约 1/4） | Sec IV, Tab I | 高 |
| [paper_013](https://arxiv.org/abs/2503.03707) | direct | 支持 | Demo-SCORE：用在线 rollout 训练成功/失败分类器过滤演示；仿真 +15–35%、真实 ALOHA +15–40%；过滤 16–67% 演示；<100 条 rollout 即可 | Sec IV-V | 高 |
| [paper_033](https://arxiv.org/abs/2505.22626) | direct | 支持 | SCIZOR：自监督策展——"两帧间真实流逝时间"作进度代理训练时间距离分类器（DINO-V2 + 5 bin），无 LLM/无奖励/无标注 | 全文 | 高 |
| [paper_034](https://arxiv.org/abs/2606.16208) | direct | 支持 | ATHENA：闭环影响力函数（flow 代理 + rollout 回报加权）策展，加速 313.4×，仿真 50 任务 + 真机 6 任务下游验证 | 全文 | 高 |
| [paper_001](https://arxiv.org/abs/2403.12945) | direct | 支持 | 仅靠人工成功标记剔除失败（约 16k 条不训练） | Sec V-A | 高 |
| [paper_015](https://arxiv.org/abs/2503.06669) | direct | 支持(弱) | 人工核对+丢弃不合格；失败恢复数据（~1%）保留并标注原因 | Sec III.B | 高 |

## RQ1.4 语义级/语言级数据处理先例

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_010](https://arxiv.org/abs/2509.17321) | direct | 支持 | GVL：VLM in-context 预测逐帧任务进度（0-100%）；VOC 秩相关作质量度量；在 LeRobot hub 上演示三类问题检出；开源模型约为闭源 60–70%；策展为定性案例 | Sec 2-4 | 高 |
| [paper_001](https://arxiv.org/abs/2403.12945) | direct | 支持 | GPT-4V 自动场景分类 + 人工复核；GPT-4 动词去重 | App C, Sec IV | 高 |
| [paper_021](https://arxiv.org/abs/2509.20070) | direct | 支持 | GPT-4o 从 1 条示范生成关键位姿标注与增强数据；生成成功率超人工标注基线 2-3 倍 | Sec III-B, IV | 高 |
| [paper_035](https://arxiv.org/abs/2309.00743) | direct | 支持(弱) | 语言条件切点检测识别子任务边界（旋钮 1 组件先例）；+1.78±0.82% 提升；无下游策略验证 | 全文 | 中 |

## RQ2.1 LLM agent 基础（定义/组成/运行）

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_006](https://arxiv.org/abs/2309.07864) | direct | 支持 | Xi 框架：brain/perception/action；四属性（autonomy/reactivity/pro-activeness/social）；技术脉络五阶段 | Sec 2.2-3 | 高 |
| [paper_017](https://arxiv.org/abs/2308.11432) | direct | 支持 | Wang 框架：profile/memory/planning/action 四模块及子组件；能力获取三途径 | Sec 2.1-2.2 | 高 |
| [paper_007](https://arxiv.org/abs/2210.03629) | direct | 支持 | ReAct 机制与四基准对比；幻觉从 56%→0%（失败轨迹），推理错误率升高；ALFWorld +34% 绝对 | Sec 2-4 | 高 |
| [paper_018](https://arxiv.org/abs/2302.04761) | direct | 支持 | Toolformer 自监督工具调用（sampling→executing→filtering→finetune）；775M 参数涌现 | Sec 2, 4.4 | 高 |

## RQ2.2 agentic 数据策展/质量评分方法

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_009](https://arxiv.org/abs/2407.03502) | indirect | 支持 | AgentInstruct 三 flow 生成 25.8M 指令；质量仅经下游基准间接验证（Orca-3 平均 +34%） | Sec 2-4 | 高 |
| [paper_008](https://arxiv.org/abs/2306.05685) | direct | 支持(带限定) | LLM-as-judge 与人类 85%/87% 一致；position bias（交换后一致率 23.8–65%）、verbosity bias（GPT-4 失败率 8.7%）、数学误判 70%→15% | Sec 3.3, 4.2 | 高 |
| [paper_010](https://arxiv.org/abs/2509.17321) | direct | 支持 | VOC 作为自动质量度量（必要非充分信号） | Sec 2 | 高 |
| [paper_022](https://arxiv.org/abs/2310.03714) | direct | 支持 | DSPy：声明式管线 + 指标驱动编译优化；GPT-3.5 GSM8K 33%→82% | Sec 3-6 | 高 |
| [paper_023](https://arxiv.org/abs/2406.07496) | direct | 支持 | TextGrad：文本梯度自动优化复合系统；LeetCode +16%、GPQA 51→55 | Sec 2-3 | 高 |

## RQ2.3 机器人领域 agent/语义数据管线

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_019](https://arxiv.org/abs/2401.12963) | direct | 支持 | AutoRT：VLM+LLM 编排批量采集 77k episode；Robot Constitution 安全约束；多样性打分；下游 RT-1 提升 | Sec 4-5 | 高 |
| [paper_010](https://arxiv.org/abs/2509.17321) | direct | 支持 | 见 RQ1.4 | Sec 2-4 | 高 |
| [paper_020](https://arxiv.org/abs/2309.14320) | direct | 支持 | MUTEX：LLM 生成文本规范+人工过滤、TTS 语音规范；六模态任务规范标注管线 | Sec 3.3 | 高 |
| [paper_021](https://arxiv.org/abs/2509.20070) | direct | 支持 | 见 RQ1.4 | Sec III | 高 |

## RQ2.4 离线 agent 管线工程模式

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_022](https://arxiv.org/abs/2310.03714) | direct | 支持 | 指标与管线分离 + 编译器三阶段 + teleprompter 可组合 | Sec 3-4 | 高 |
| [paper_023](https://arxiv.org/abs/2406.07496) | direct | 支持 | 文本微分跨组件（LLM+代码解释器+模拟器）反向传播 | Sec 2 | 高 |
| [paper_019](https://arxiv.org/abs/2401.12963) | direct | 支持 | 批量闭环：生成→执行→打多样性分→下游训练验证；人机监督比 1:3–8 | Sec 4-5 | 高 |
| [paper_005](https://arxiv.org/abs/2606.10229), [paper_014](https://arxiv.org/abs/2606.15064) | direct | 支持(反面) | 离线指标与下游收益脱钩——验证回路必须测下游 | Sec V-VI | 高 |

## RQ3.1 训练/验证划分与泄漏

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_031](https://arxiv.org/abs/2606.16826) | direct | 支持 | ATOM-Bench：原子技能（motor/instruction）定义；held-out 组合划分（微调 15 原子任务、12 组合任务零演示） | 全文 | 高 |
| [paper_030](https://arxiv.org/abs/2412.13877) | direct | 支持(负面) | RoboMIND 107k 轨迹但**无显式 train/val 划分协议**；评测任务与训练数据重叠关系未声明 | Sec V | 高 |
| [paper_005](https://arxiv.org/abs/2606.10229) | direct | 相关 | episode 长度是评估的混杂因素（截断前 AUROC 近 1.0 全是时长）——划分/评测中同样要控制 | Sec III.D | 高 |

**RQ3.1 结论**：组合泛化的 held-out 协议有先例（[paper_031](https://arxiv.org/abs/2606.16826)）；但"训练/验证按
episode 划分防自相关泄漏"这条规则在数据集论文中几乎从不被声明（[paper_030](https://arxiv.org/abs/2412.13877)
反例）。属于"实践共识、文献沉默"的区域（gap_list GAP-09）。

## RQ3.2 多传感器时间同步与时间戳对齐

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_016](https://arxiv.org/abs/2505.15558) | direct | 支持 | Robo-DM：各流带相对时间戳对齐，不依赖时间戳假设 | Sec III | 高 |
| [paper_032](https://arxiv.org/abs/2201.09170) | weak | 支持 | VINS 在线自标定：时间偏移作为滤波状态变量估计（Eq.29-30），退化条件=恒定角速度+恒定线速度 | Sec 5-6 | 中 |
| [paper_030](https://arxiv.org/abs/2412.13877) | direct | 支持(负面) | 多视角相机时间同步机制未涉及（107k 轨迹的基准论文） | 不确定项 | 高 |

**RQ3.2 结论**：遥操数据集的跨相机/状态时间对齐实践几乎没有文献记录；
可迁移的方法形态来自 SLAM 域（时间偏移作为优化变量估计，[paper_032](https://arxiv.org/abs/2201.09170)，弱证据）。
（gap_list GAP-10）

## RQ3.3 数据增强的安全性

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_024](https://arxiv.org/abs/2310.17596) | direct | 支持 | MimicGen：对象中心段变换→真实执行→成功才保留；10 演示→1000；但线性插值段无碰撞保证、naive 成功过滤造成覆盖偏置（43.5–66.4%） | Sec 4-7, App R | 高 |
| [paper_026](https://arxiv.org/abs/2412.03252) | direct | 反对(限定) | 变速度回放增强：真实环境反应 vs 时间缩放复制 88% vs 31%（插值区间）——**速度变化在接触/力控任务中不是安全增强**；"稳态速度无关"只在无接触动力学段成立 | Sec 4-5, Tab 1-2 | 高 |
| [paper_025](https://arxiv.org/abs/2410.04370) | direct | 支持 | 下采样增强评测：对称对齐（DABI）最优、纯降采样最差——增强的时序对齐方式影响结果 | 全文 | 高 |
| [paper_027](https://arxiv.org/abs/2411.16959) | direct | 支持 | RoCoDA：不变性/等变性/因果性统一的反事实增强框架 | 全文 | 高 |
| [paper_028](https://arxiv.org/abs/2503.18738) | direct | 支持 | RoboEngine：语义分割+背景生成的视觉增强（无需绿幕/标定） | 全文 | 高 |

**RQ3.3 结论**：增强的安全性 = 变换必须保任务语义（[paper_027](https://arxiv.org/abs/2411.16959) 的因果框架）；
速度/时序类增强有接触动力学边界（[paper_026](https://arxiv.org/abs/2412.03252)，与用户"稳态场景"约束的边界条件
一致：无接触段速度无关、接触段不成立）；增强轨迹必须验证（[paper_024](https://arxiv.org/abs/2310.17596) 执行+成功
过滤，但注意其覆盖偏置）。

## RQ3.4 观测可分性（state aliasing）标注

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_029](https://arxiv.org/abs/2512.24638) | direct | 支持(模型侧) | 状态歧义五类：双向重叠轨迹/固定时长静止/顺序子任务依赖/重复时间循环/动态交互响应；PAM 用 300 帧工作记忆解决；**不做数据标注、无逐帧歧义度量** | Sec I, III | 高 |
| [paper_011](https://arxiv.org/abs/2603.01465) | direct | 相关 | keyframe 检测（KSM）本质是"哪些帧是信息充分的阶段标记" | Sec 4.2 | 高 |

**RQ3.4 结论**：歧义五类分类法是旋钮 4 标注的现成设计素材（[paper_029](https://arxiv.org/abs/2512.24638)）；
但"逐帧观测可分性标注"无先例，模型侧方案（历史注入）是当前主流
（gap_list GAP-08 维持 open，补充了分类法）。

## RQ3.5 覆盖分析与补采建议

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_030](https://arxiv.org/abs/2412.13877) | direct | 支持 | RoboMIND 失败案例分析指出数据短板（物体放置非随机、夹爪闭合帧数不足）并建议补采——"可定位缺口→补采"的形态 | Sec V-F | 高 |
| [paper_019](https://arxiv.org/abs/2401.12963) | direct | 支持 | AutoRT 语言/视觉多样性打分驱动采集策略 | Sec 5.1 | 高 |
| [paper_001](https://arxiv.org/abs/2403.12945) | direct | 支持 | DROID 场景多样性消融（同轨迹数、高场景多样性子集 OOD 更好）——覆盖影响泛化 | Sec V | 高 |

## 负证据汇总

1. **离线几何/统计过滤指标与下游策略提升脱钩**（[paper_005](https://arxiv.org/abs/2606.10229)：AUROC 最高者下游最差；
   [paper_014](https://arxiv.org/abs/2606.15064)：phase 局部化策展全面失败）。含义：任何"先评分后过滤"的管线都必须
   以下游策略性能为最终判据，不能只看检出率。
2. **开源 VLM 的语义进度预测显著弱于闭源**（[paper_010](https://arxiv.org/abs/2509.17321)：约 60–70%；小模型接近随机）。
3. **LLM-as-judge 有系统性偏差**（[paper_008](https://arxiv.org/abs/2306.05685)：position/verbosity/self-enhancement）。
4. **AgentInstruct 无直接数据质量度量**（[paper_009](https://arxiv.org/abs/2407.03502)：质量好坏只靠下游基准间接推断）。
5. **ReAct 式交错推理提高推理错误率**（[paper_007](https://arxiv.org/abs/2210.03629)：失败轨迹 reasoning error 47% vs 16%）。
6. **变速度回放在接触/力控任务中不是安全增强**（[paper_026](https://arxiv.org/abs/2412.03252)：真实反应 vs 时间
   缩放复制 88% vs 31%）——"稳态场景速度无关"的边界条件：无接触段成立、
   接触段不成立。
7. **naive 成功过滤造成数据覆盖偏置**（[paper_024](https://arxiv.org/abs/2310.17596)：Square D2 覆盖仅 66.4%、
   Three Piece Assembly D1 43.5%）。
8. **主流基准不声明 train/val 划分协议**（[paper_030](https://arxiv.org/abs/2412.13877)：107k 轨迹无划分说明）。

## 证据缺口

见 `synthesis/gap_list.md`。


## RQ4.1/4.2/4.3/4.5 动作模态对齐（首轮，2026-09-08）

| paper_id | 证据类型 | 支持/反对 | 证据摘要 | 来源位置 | 可信度 |
|---|---|---|---|---|---|
| [paper_036](https://arxiv.org/abs/2606.14752) | direct | 支持 | X-Tokenizer：SRQ 非对称量化 + 三头语义对齐，把 action 序列编码进 vision/text 共同空间；M=16/64 隐式低秩 | 全文 | 高 |
| [paper_037](https://arxiv.org/abs/2503.00200) | direct | 支持 | UVAM：联合视频-动作潜表示 + 解耦扩散（无离散化）；7 公开基准 + 真机 OOD | 全文 | 高 |
| [paper_038](https://arxiv.org/abs/2606.17046) | direct | 支持 | GAM：动作 token 与几何 token 在同一个 GFM 潜序列中联合解码 | 全文 | 高 |
| [paper_039](https://arxiv.org/abs/2605.15725) | direct | 支持 | DiLA：预测瓶颈驱动的 content-structure 分解（连续 dz=256；VQ 流形扭曲对比） | 全文 | 高 |
| [paper_040](https://arxiv.org/abs/2607.27549) | direct | 支持 | Behavior-Aligned：对齐行为表征（EE 轨迹最有效，BC+表征联合损失），跨本体可迁移 | 全文 | 高 |
| [paper_041](https://arxiv.org/abs/2602.13764) | direct | 支持 | MOTIF：动作母题 = 进度感知 InfoNCE + 本体对抗 GRL 的 VQ 聚类；去运动学规范化 -10.33% | 全文 | 高 |
| [paper_042](https://arxiv.org/abs/2501.10105) | direct | 支持 | UAT：通用动作 256x128 离散码本 28 本体；新本体微调仅 0.8% 参数；256 动作目检 + JS 散度验证 | 全文 | 高 |
| [paper_043](https://arxiv.org/abs/2505.04999) | direct | 支持 | CLAM：连续潜在动作 + 联合训练解码器；连续+联合 74% vs 离散+非联合 16% | 全文 | 高 |

**首轮结论**：动作模态对齐有两条实证路线——显式编码（tokenizer/码本，036/042）与
潜表示对齐（040/041/043）；"连续性"是关键分歧（CLAM 74% vs 16%）。跨本体以
行为/母题/通用码本对齐（040/041/042），与 sim-实 对齐迁移同构（衔接 OPT-002）。
空缺：对齐度量本身（RQ4.4）与时序保真（RQ4.6）无直接命中——第二轮检索方向。
