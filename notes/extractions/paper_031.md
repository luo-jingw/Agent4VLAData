# 论文标题
ATOM-Bench: A Real-World Benchmark for Atomic Skills and Compositional Generalization in Manipulation Policies

## 基本信息
- paper_id: paper_031
- 年份: 2026（arXiv v1：2026-06-15）
- 作者: Zenan Wu, Bingqing Wei, Lu Liu, Zheqi He（通讯）, Xi Wang, Jiakang Liu, Zehui Li, Guocai Yao, Jing-Shu Zheng, Xi Yang, Yongtao Wang（北京智源人工智能研究院、北京大学）
- 来源: arXiv:2606.16826v1 [cs.RO]

## 研究问题
诊断通用操作策略的真实世界泛化：策略能否习得可靠的原子技能（motor + instruction）；习得的原子技能能否重组解决微调期间从未演示过的 held-out 组合任务；失败源于原子技能不足还是组合复用不足。

## 方法
- 任务分解：桌面操作分解为 motor atoms（pick-and-place、reorientation、pushing、stacking、pouring、articulated-object access）与 instruction atoms（color、shape、size、source_relation、goal_destination、count、exclusion）。选型标准（Section 3.1）：常见于操作任务、暴露不同物理/语义失败模式、可在真实机器人上可靠实例化、可重组为 held-out 组合任务。
- 双平台配对任务：Franka Panda（单臂）与 Agilex Cobot Magic（双臂）；每平台 7 个 instruction 任务 + 8 个 motor 任务 + 12 个 held-out 组合任务；双臂任务为单臂任务的对应版（额外要求手角色分配、双臂协调、有序动作序列）。
- 数据：每原子任务 100 条专家遥操演示，共 3,000 条；30 Hz 同步记录观测与动作；Franka 约 9 小时、Cobot Magic 约 12.5 小时；演示数据与评测 rollout 数据均公开。
- 划分协议（held-out）：微调只用 15 个原子任务（atomic-transfer：联合微调 15 任务、1,500 条演示；single-task：每任务独立 checkpoint、100 条演示，仅 Pi0.5）；Composition Set 12 个任务在微调期间无任何演示；每任务 10 个固定物理测试种子（mask-guided placement 复现初始摆放），结果跨任务 macro-average。
- 指标（Section 3.3）：SR；PSR（过程得分率）；AS(X) = 参与原子 a∈A(X) 的均值 PSR（原子基线天花板）；CFS(X) = max(0, AS−PSR)/(1−PSR)（不可由弱原子解释的组合失败份额）；TG（单任务与联合微调的 SR 差）。

## 关键发现
- 原子技能习得（Table 1）：instruction 原子比 motor 原子容易；Pi0.5 Franka instruction SR 94.3%、motor 46.2%；motor 原子中 pick-and-place 跨模型平均 SR 74% 最高，access 9%、reorientation 11%、pushing 16%、pouring 20% 最低；count 与 exclusion 等集合级约束难；SmolVLA 全面落后。
- 组合泛化（Table 2）：Pi0.5 AS 达 83.3%（Franka）/79.5%（Cobot），组合 SR 仅 15.8%/16.7%；CFS 73.7%/56.8%，其失败主要来自组合本身而非未习得原子；SmolVLA 低 AS 低 CFS（失败仍受限于原子）。
- 失败模式：motor failure、reference failure、timeout 合计超 90%（Section 4.4）；扰动测试（Appendix C）：原子任务对背景/措辞扰动稳健（SR 保持 90–100%），组合任务明显变脆（dx1 60%→40%，dx12 50%→40%/20%）。
- 与数据缺口的关系：原子微调数据每任务固定 100 条；held-out 组合任务按构造无演示（即被控制的数据缺口）；Limitations 指出未覆盖可变形物体、工具使用、移动操作、长时程等长尾技能。

## 全文相关性（fulltext_relevance）
- 全文相关性: high
- 理由: 原子技能定义、held-out 划分协议与组合泛化指标是论文核心内容，且以 3,000 条演示、2,700 次实机 rollout 的观测结果直接支撑 RQ3.1 的"技能覆盖—评估协议—数据缺口"问题。

## 引用
- 原子定义与选型标准：Section 3.1、Appendix E
- 划分协议与微调设置：Section 4.1、Appendix D（1,500 演示）
- 指标公式：Section 3.3
- 数字结果：Table 1、Table 2、Section 4.2/4.3；种子与摆放复现：Appendix B

## 不确定项
- "held-out"指组合任务在微调中无演示，但组合任务的物理场景/物体是否在原子演示中出现未完全说明。
- 每任务 100 条演示的量级依据未给出（无数据量消融）。
- 组合任务 10 个物理测试种子与原子任务种子是否共享未说明。
- 与补采的关系未直接讨论：未给出"组合失败→应补采什么数据"的实证建议（仅提供诊断框架）。
