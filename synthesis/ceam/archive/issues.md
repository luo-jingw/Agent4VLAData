# ISSUE-006

Status: open

Area: 闭环 MVP / 静止锁死

## Observation

LIBERO-Goal init 0、K = 32、候选效果为示范真实效果（Phase 5）：任务 0、6、7 上检索到的 32 个候选全部来自其他任务的示范，在本任务查询方向下 cos 均不大于 0，MATCH 选静止；静止不改变状态，检索结果不变，于是连续静止到 300 步（任务 7：30/30 次静止；任务 0：29/30；任务 6：28/30）。成功的任务（2、4、5、9）中，起点的候选部分来自本任务，之后几乎全部来自本任务。

## Impact

在共享起点附近，ψ 最近邻不按任务区分，候选集中可能没有与本任务一致的动作；当前设计中没有从静止中恢复的机制。

## Evidence

`outputs/closed_loop/mvp_phase5_init0/task*_match/results.jsonl`。Phase 6（500 个 episode，`docs/closed_loop_mvp.md`）：match 的 362 个失败 episode 中 269 个有 ≥ 10 次静止决策、224 个 ≥ 20 次；成功的 episode 平均 0.14 次静止。

## Hypotheses

- 原设计中"选中静止且离目标远 → 判定卡住 → 重采样或重规划"的机制在 MVP 中没有实现；对检索候选而言，对应做法是卡住时扩大检索范围。

## Next Experiment

Phase 10–11 已完成：见 `docs/closed_loop_mvp.md`。卡住时本任务示范状态在 ψ 距离中的排名中位数为 2571（从未进入前 512），在冻结特征相似度中为 177（74% 在前 512 内）；卡住时执行最高分候选（E2）不提升成功率，缩短执行步数（E1、E3）降低成功率。检索空间改为冻结特征（用户批准，第二计划 Phase 1–2）后，真实效果 + 规则的卡住 episode 从 188 降到 35，但成功率不变（0.326 对 0.312，p = 0.63），失败转为目标物体动了但没到目标；预测效果 + 规则下仍有 133 个卡住，其中 87 个的最终候选中有本任务状态，但 Δ 对它们的打分 ≤ 0（`docs/closed_loop_mvp.md`）。

## Resolution

# ISSUE-007

Status: open

Area: 仿真渲染 / node-71 CUDA 7

## Observation

node-71 的 CUDA 设备 7 上，新进程的渲染 4 次中仅 1 次正常：出现读回滞后一帧、条纹、构建即黑帧（最多 4 次中 3 次）；被拒绝一次后接受的环境仍然损坏（均值通过 ≥ 40 检查，但与数据集起点的 DINOv2 余弦降到 0.17）；两次冒烟运行以 SIGABRT 退出。设备 5（4/4）与设备 1（3/3）正常。之后一次在设备 1 上的开发运行（任务 2，预测效果）在两次 75–78 s 的执行后以 SIGABRT 退出，设备 5 重跑正常。

## Impact

黑帧检查不能发现滞后或条纹帧；闭环结果可能被损坏的画面污染。

## Evidence

Phase 5 诊断日志与图像（worktree 已删除）；结论记录在本条。

## Hypotheses

- 与该卡上其他用户的负载或 EGL 上下文状态有关。

## Next Experiment

每个闭环进程开始时做起点观测的余弦检查（`--check-start-observation`），不达标的卡不用。

## Resolution

# ISSUE-010

Status: open

Area: 仿真 / 控制器目标在 set_init_state 之后的首步

## Observation

Phase 12 发现，`reset()` 加 `set_init_state` 之后，robosuite OSC 控制器的第一次 `set_goal` 以 reset 时缓存的末端位姿为基准，除非设置 `controller.new_update = True`。在恢复示范帧时，不设置它会让第一步的本体状态偏差约 2e-2 rad（`docs/libero_sim.md`，Restoring a dataset frame）。

## Impact

`LiberoTaskEnv.reset` 在 `set_init_state` 之后执行 10 步静止等待，第一步可能也以 reset 位姿为目标。闭环起点与 benchmark 的约定是否一致没有核查。

## Evidence

Phase 12 开发检查（`tmp/runlogs/p12/dev/`），报告在 `docs/libero_sim.md`。闭环起点观测与数据集起点的余弦为：腕部 0.850–0.889，第三视角 0.929–0.963（v2）。

## Hypotheses

- 静止等待的后 9 步把机械臂拉回 set_init_state 的位姿，影响可忽略。

## Next Experiment

对若干 init state，比较 `LiberoTaskEnv.reset` 与"设置 new_update 后再静止等待"两种做法的起始本体状态差。

## Resolution

# ISSUE-012

Status: 暂缓（用户暂停以成功示范为中心的训练路线，2026-09-28；原为框架改动提案，只写不实施）

Area: 阶段一 / 后见语言重标注

## Observation

Phase 8（`docs/rollout_progress.md`）的结果：

- 失败 episode 中，无进展或倒退的动作块，真实效果的 match m(z) = cos(ψ(o′) − ψ(o), q_l) 为正的比例是 63%。
- 在"m(z) > 0"的动作块里，有进展的只占 8%；成功 episode 为 35%，示范为 39%。

指令对比项在不同指令之间做对比，回答的是"这段位移属于哪条指令"，没有见过失败状态上的位移，因此在失败状态上给出的正分不可靠。

第二计划 Phase 26（`docs/branch_success_test.md`）用仿真成败作标签，在 π 均值块闭环失败的 episode 上，从每个决策点分支执行 π 的 32 个小偏移候选：

- match 最高分候选的后续成功率比随机候选只高 +0.008–0.032；
- 同尺度候选内按 match 排序的 AUC 为 0.47–0.54，只有定版 ψ 的 q_l 高于随机（0.542，区间 [0.513, 0.572]）；
- 旧 v、新 v 在三个 ψ 上都接近随机。

## Impact

仅扩充 Δ 的数据无法消除这部分误判：即使 Δ 完全准确，也会给这些动作块打正分。

## Evidence

- `docs/rollout_progress.md`；
- `outputs/analysis/rollout_progress_r0/report/baseline/transitions.csv`（每个转移的进展标签）。

## Hypotheses

- 提案：把失败 rollout 片段重新标注为它实际完成的事情，从"做了什么"反推指令。这与逆向视角一致：效果表征先存在，语言去查询它。
  - 片段的实际效果成为新指令的正样本。
  - 对原指令无进展或倒退的片段成为原指令的负样本。
  - 两者都进入现有的指令对比项：只扩充对比集合，不新增损失形式。
- 标注来源，由近到远：
  1. **谓词重标注**：检查片段末尾的场景是否满足其他 LIBERO-Goal 任务的 BDDL 目标（例如开错了抽屉）。可以直接复用现有 10 条指令，但覆盖少。
  2. **模板化描述**：由场景量生成描述，例如 "push the plate to the left"、"lift the bowl"、"close the drawer"。SigLIP 可以编码任意文本；q_l 由此见到新指令。
  3. **视觉语言模型描述**：需要引入外部模型，最后考虑。
- 数据：rollouts_r0 的 23,202 条转移及其场景记录，Phase 8 的进展标签可直接复用。之后各轮 rollout 同样适用。

## Next Experiment

用户确认后：

1. 先用谓词重标注与模板化描述在 rollouts_r0 上生成标注，统计覆盖率与标注分布。
2. 阶段一在示范 + 重标注片段上重训，其余配置不变。
3. 下游重建 ψ 表与键表，重训 Δ（数据 A）。
4. 比较：
   - Phase 8 指标（失败状态上 P(m > 0 | 无进展或倒退)）；
   - 阶段一健全检查；
   - 闭环成功率，与当前最优配置配对。

## Resolution

# ISSUE-013

Status: 暂缓（用户暂停以成功示范为中心的训练路线，2026-09-28；原为框架改动提案，只写不实施）

Area: 阶段一 / 时间距离表征与对比式目标条件打分

## Observation

与 ISSUE-012 同一现象：cos(z, q_l) 在失败状态上不是进展的可靠信号。

## Impact

同 ISSUE-012。

## Evidence

- `docs/rollout_progress.md`；
- 第二计划 Phase 12（`docs/closed_loop_mvp.md`）：把方向换成随状态变化的 ĝ_l − ψ(o) 后，失败状态上"正分无进展"的比例没有下降——真实效果下 q_l 为 0.63，ĝ − ψ 为 0.69，U 投影版为 0.67。完整空间版的闭环成功率还降到 0.074–0.178。

## Hypotheses

- **HILP 式时间距离训练**（Park 等，2024）：
  - 训练 ψ，使 ‖ψ(s) − ψ(g)‖ 近似从 s 到 g 所需的最少步数。
  - 打分为 ⟨ψ(o′) − ψ(o), (ψ(g) − ψ(o))/‖ψ(g) − ψ(o)‖⟩，与现在的 match 几乎同构，但沿该方向移动在定义上就是在接近目标。
  - 目标 ψ(g) 由指令给出（ĝ）。
- **对比式目标条件打分**（Contrastive RL，Eysenbach 等，2022）：
  - 打分为 φ(o, a)ᵀψ(g)，用 InfoNCE 训练：正样本是同一轨迹的未来状态，失败 rollout 自然成为负样本。
  - 学到的内积对应到达目标的概率。
- 用户的判断：两者都会使 match 趋向价值函数，偏离"简单 match"的设计，故排在最后。

## Next Experiment

用户确认后再定；应先看 ISSUE-012 的结果。

## Resolution

# ISSUE-014

Status: open（需要用户决定；Phase 11 因此暂停）

Area: 闭环 / match 选择与提议分支

## Observation

第二计划 Phase 10 第 0 轮（`docs/proposal_branch.md`，10 任务 × 50 初始状态，Δ +A，冻结检索，规则 32/128/512）：

| 候选与选择方式 | 成功率 |
|---|---|
| 检索候选 + MATCH | 0.326 |
| π 候选 + MATCH | 0.120（复跑 0.142） |
| 检索 + π 候选 + MATCH | 0.126 |
| 每次决策直接执行 π 的均值动作块，不经 MATCH | 0.826 |

- π 均值动作块 vs π 候选 + MATCH：362 / 9，p = 1.4e-94。
- MATCH 几乎从不选 π 的均值块：11,696 次中只选了 19 次；偏好偏移最大的样本：尺度 1.0 的样本占选中动作块的 64–66%，而它只占候选的三分之一。

## Impact

在 π 给出的好候选附近，按 cos(Δ̂, q_l) 选 argmax 会系统性地偏离好动作。这与之前的结果一致：

- Phase 8：失败状态上 63% 的无进展动作块真实 match 为正。
- 第一计划 Phase 12：扰动块的真实 match 常高于示范块，只有 3.1% 的状态上示范块高于全部扰动块。
- Phase 9 仿真核查：MATCH 选出的动作块真实 match 高于示范块（0.410 对 0.369），但进展率不高于示范块（0.326 对 0.349）。

match 奖励的是"沿任务方向移动得多"，不是"完成任务"。按计划重训 Δ 只能改进 Δ̂ 对真实效果的近似，不能改变这一点（Phase 8）。

## Evidence

- `outputs/closed_loop/v5_{proposal,mixed,proposal_replicate,proposal_mean_only}_dynA/`；
- 分析脚本与日志：`tmp/runlogs/s3p10/`。

## Hypotheses

- 在允许的超参数范围内，可以用两个诊断定位 match 的作用（已启动，`tmp/runlogs/s3p10/launch_v5b.sh`）：
  - 小偏移候选（尺度 0.05/0.1/0.2）：离均值很近时，MATCH 选择是否接近均值动作块的表现；
  - 只让均值动作块与静止比较（K = 1，不扩大检索）：match 作为是否执行的门槛，是有益还是有害。
- 框架层面的选项需要用户决定，例如：
  - match 只作门槛或排序中的一项，不作 argmax；
  - 先改变 match 的训练目标（ISSUE-012、ISSUE-013、OPT-008）。

两个诊断的结果（`docs/proposal_branch.md`）：

- 小偏移候选 + MATCH：0.584，对均值动作块 40 / 161，p = 2e-18。
- 均值动作块 vs 静止：0.738，对均值动作块 7 / 51，p = 2e-9；其中 103 个 episode 因均值动作块预测 match ≤ 0 而一直静止。

结论：让 match 参与决定的每种方式，无论作为 argmax 还是作为门槛，成功率都低于直接执行 π 的均值动作块。

## Next Experiment

由用户决定 Phase 11（rollout → 重训 Δ → 再 rollout）是否继续，以及 match 在闭环中的角色。

## Resolution

# ISSUE-015

Status: open

Area: 可复现性 / 渲染噪声经连续动作块放大

## Observation

Phase 10 发现，在完全相同的 qpos、qvel 下，LIBERO 渲染的腕部画面偶尔有 1–2 个像素相差一个亮度级。

- 检索候选是离散选择，这样的差异通常不改变选择，所以检索与混合来源逐位复现。
- π 输出连续动作块，差异直接进入物理，下一步的 qpos 偏差约 3–6e-6，随后放大。结果：相同命令的两次提议来源运行，5 个 episode 中有 4 个在第 15–28 次决策后分叉。
- 提议噪声的随机流按进程内的决策序号计数。一个 episode 的长度一旦变化，同一进程中之后所有 episode 的噪声都会随之改变。在复跑中，这使 64 个 episode 从第一次决策起就不同。

## Impact

提议来源的闭环结果在运行之间有波动：成功率 0.120 对 0.142，配对 13 / 2，p = 0.0074。用 π 采集 rollout 时，数据不能逐位复现。

## Evidence

- `docs/proposal_branch.md`（phase 10 一节）；
- `tmp/runlogs/s3p10/logs/`。

## Hypotheses

- 把噪声流改为按 episode 键控，例如 (seed, task, start, init, decision)，可把分叉限制在受影响的 episode 内。渲染噪声本身无法消除。

## Next Experiment

- 噪声流已改为按 (seed, task, start, init, decision) 键控（第二计划 Phase 13，提交 `b4487f6`）。长度变化不再波及同一进程中之后的 episode。
- 渲染噪声本身仍在：同一 GPU 上的两个进程之间也会出现，例如任务 7 同一命令两次运行，10 个 episode 中 8 个在第 7–15 次决策分叉。
- 该问题只影响连续动作块的逐位复现，不影响统计结论：新默认配置 v6 与此前的 v5 均值版为 0.828 对 0.826，p = 1。

## Resolution

# ISSUE-016

Status: open（D_ψ 停止梯度后塌缩已消除；待确认后关闭）

Area: 阶段一 / 完整空间逐帧插值目标的退化解

## Observation

第二计划 Phase 17（`docs/stage_one_flow.md`）。完整 256 维 ψ 上的插值目标 ‖ψ_t − ((1 − t/T)·sg[ψ₀] + (t/T)·sg[ψ_T])‖² / D_ψ，D_ψ = ‖ψ_T − ψ₀‖² + ε（带梯度），训练 5000 步后没有实现目标：

- 中间帧停在沿线位置 p ≈ 0.63 的平台上（HELDOUT 在 t/T = 0.1/0.25/0.5/0.75 处为 0.628/0.632/0.638/0.643）；路径项停在约 0.09，与平台值 ∫(0.63 − f)² df = 0.100 相符。
- ψ 的有效秩从 16.6 降到 1.1，首个主方向占 ψ 方差的 98%。
- 起止帧被推离中间帧：到中间帧均值的距离为 714 / 457，定版为 15.5 / 30.4；‖ψ_T − ψ₀‖ 为 1156，定版为 35。
- 任务 ID 探针从 1.0 降到 0.88–0.90。

投影坐标 z = Uᵀψ 上的同一目标（对照）实现了目标，ψ 的秩与探针不变。

插值方向没有写反，t/T 的计算也已核对（`docs/stage_one_flow.md`）。

## Impact

完整空间版作为用户指定的主版本不可用，它的离线评估反映的是退化的表征。

## Evidence

- `outputs/runs/goal_instruction_contrast_flow_full/`（训练日志与诊断）；
- `docs/stage_one_flow.md`。

## Hypotheses

以下未经验证：

- 分母 D_ψ 带梯度。增大起止帧之间的距离，比让中间帧沿线分布更容易降低该项。
- SIGReg 只作用于均匀帧，t ∈ {0, …, T − 2}，末帧从不进入，起始帧也很少进入，所以起止帧远离分布不受约束。原始设计 C3 要求"所有帧"的聚合分布接近高斯，这里实现与原设计不一致。
- 修正都属于设计改动，未实施：
  - 让 SIGReg 覆盖起止帧；
  - 或对 D_ψ 停止梯度。

## Next Experiment

用户决定（2026-09-27）：让 SIGReg 覆盖所有帧（含起止帧），恢复原设计 C3；不对 D_ψ 停止梯度。按此重训完整空间版（第二计划 Phase 21，`docs/stage_one_flow.md`）。

该修正**未能消除塌缩**：

- ψ 有效秩为 1.02，首个主方向占方差的 0.997，‖ψ_T − ψ₀‖ = 1338。
- 终点帧被拉回主体（到中间帧均值的距离为 1.09 个中间帧离散度）；起始帧成为孤立的远簇（10.2 个离散度）。
- 中间帧的沿线位置 p ≈ 0.99，路径项停在约 0.25。

实测：ALL 形状的样本中，起始帧只占 5.3%。起始帧簇远离时，SIGReg 统计量在距离 ≥ 100 后稳定在约 0.007，对这些帧的梯度 < 2e-6。也就是说，SIGReg 对一小簇远离点只收取有界的、不随距离增长的代价。

假设（未验证）：带梯度的 D_ψ 持续奖励增大 ‖ψ_T − ψ₀‖，SIGReg 的有界代价抵不住这一推动。剩下的手段是对 D_ψ 停止梯度，或加入无界的尺度项。二者都是设计改动，需要用户决定。

用户决定（2026-09-27）：对 D_ψ 停止梯度，它只用于使损失对整体缩放不变，不应成为可优化的量。修正后重训，见第二计划 Phase 24。

**结果（第二计划 Phase 24，`docs/stage_one_flow.md`）：塌缩消除。**

- ψ 有效秩 8.56，首个主方向占方差 0.198，‖ψ_T − ψ₀‖ = 27.5。
- 插值目标实现：p 在 t/T = 0.1/0.25/0.5/0.75 处为 0.029/0.230/0.430/0.668，平均 |p − t/T| = 0.061，定版为 0.102。
- 没有整体收缩，没有中间帧压缩。

仍有三处较弱的现象，只作观察：

- 各 episode 的起始帧几乎重合（离散度 0.90，定版 10.2），早期帧几乎不动；
- 有效秩只有定版的一半；
- HELDOUT 中 9/43 个 episode（任务 3–5）在 t/T ≈ 0.5 之前沿起止方向倒退。

第二计划 Phase 28（`docs/proposal_branch.md`）：sg[D_ψ] 版 ψ 上训练的 π（三个种子）与定版、投影插值版没有可检出的差别，F(2, 6) p = 0.88。上述三处弱现象按用户指令只记录、不修。

## Resolution

# ISSUE-017

Status: open

Area: 打分 / 完整空间的 ĝ − ψ 没有定义

## Observation

第二计划 Phase 18（`docs/rollout_progress.md`）的结果：

- 目标损失只比较 Uᵀĝ 与 Uᵀsg[ψ_T]，没有别的项用到 ĝ。
- 三个检查点中，ĝ 在 U 坐标下都落在示范终点处，偏差为起止距离的 0.6%–0.8%；而 ĝ − ψ_T 有 99% 以上在 span(U) 之外。
- 投影插值版上，完整空间的 cos(z, ĝ − ψ) 排序反转：Phase 9 的 AUC 为 0.12，失败状态为 0.41；同一检查点在 U 坐标下为 0.69 / 0.80。
- 打分代码、符号与帧顺序都已核对。定版的重算结果与独立记录一致，误差 ≤ 1.4e-5。

## Impact

任何在完整 256 维空间里使用 ĝ − ψ 的打分都取决于 ĝ 未受约束的分量，结论不可解释，包括第二计划 Phase 12 与 Phase 14 中完整空间 ĝ − ψ 的结果。

## Evidence

- `outputs/analysis/labeled_chunk_scores/three_scores/`（report.md 中的几何表与符号核对）。

## Hypotheses

- 若使用 ĝ − ψ，只应在 U 坐标下计算。

## Next Experiment

无；在后续使用 ĝ 的实验中统一采用 U 坐标版本。

## Resolution

# ISSUE-018

Status: open

Area: 评估 / 闭环比较被 π 训练种子的方差主导

## Observation

第二计划 Phase 28 与 29（`docs/proposal_branch.md`、`docs/no_psi_baseline.md`）：同一 ψ 上，π 的三个训练种子在 10 × 50 均值块闭环中差异很大。

- 投影插值 ψ：0.904 / 0.796 / 0.884；
- sg[D_ψ] 版 ψ：0.906 / 0.806 / 0.906。

同一 ψ 内种子的合并标准差为 0.049，三个 ψ 均值的标准差为 0.010。

用当前代码重跑种子 0，与原结果只差 0–2 个 episode。所以这不是运行噪声，而是不同种子训练出的 π 本身不同。

方差集中在任务 3、4（放碗）：这两个任务上种子标准差为 0.23，其余 8 个任务为 0.012。

## Impact

- 第二计划 Phase 21–27 中的单种子闭环比较不可靠，尤其是 Phase 27 的"0.828 → 0.905 来自 ψ"。
- 用 3 个种子，也只能检出约 0.11 以上的版本差异。

## Evidence

- `tmp/runlogs/s3p28/seed_analysis.txt`；
- `tmp/runlogs/s3p29/nopsi_seed_analysis.txt`；
- `outputs/closed_loop/v8_mean_*`、`v9_mean_nopsi_*`。

## Hypotheses

以下未验证：

- 放碗任务存在两种近似等价的抓放模式，π 的均值回归在两者之间摇摆，种子决定落在哪一边。
- HELDOUT MSE 与闭环成功率相关（12 个 π：Pearson −0.89），可以作为筛选种子的廉价信号。

## Next Experiment

- 今后的闭环结论至少用 3 个 π 种子，按 Phase 28 的方法检验版本间与种子间方差（需用户确认，见 `docs/summary_2026-09-28_overnight.md` 第 11 节）。

## Resolution

# ISSUE-019

Status: open

Area: 框架 / ψ 与 q_l 作为 π 的输入没有带来收益

## Observation

第二计划 Phase 29（`docs/no_psi_baseline.md`）：不输入 ψ 与 q_l、改用冻结 SigLIP 指令 token 的 π，训练配方相同，三个种子的均值块闭环为 0.940 / 0.922 / 0.918，即 0.927 ± 0.012。

三个 ψ 版本的 π 为：

- 定版 0.853 ± 0.022；
- 投影插值 0.861 ± 0.057；
- sg[D_ψ] 0.873 ± 0.058。

与 9 个 ψ run 合并比较：

- 置换检验单侧 p = 0.0045，双侧 p = 0.027；
- 按种子自助的差值为 +0.064，区间 [+0.038, +0.094]。

差距主要在任务 3、4；其余 8 个任务上无 ψ 也更高（0.913 对 0.883）。无 ψ π 的 HELDOUT MSE 也最低：0.2608，ψ π 为 0.2670–0.2705。

## Impact

在 LIBERO-Goal 与当前配方下，"ψ 作为 π 的条件"这一用法没有被证明有用。当前默认闭环所用的 π 不是最好的 π。

## Evidence

- `outputs/closed_loop/v9_mean_nopsi_s{0,1,2}`；
- `tmp/runlogs/s3p29/`。

## Hypotheses

以下未验证：

- ψ 是 256 维的任务无关状态，q_l 是仅由文本得到的方向。两者对 π 而言都是冻结视觉与指令信息的有损重编码，只增加输入噪声。
- ~~在低数据量下情况可能不同~~：已被第二计划 Phase 31（`docs/data_scaling.md`）否定。数据越少，投影插值 ψ π 落后越多（无 ψ − 投影插值）：

  | 数据量 | 无 ψ π | 投影插值 ψ π | 差值 [区间] |
  |---|---|---|---|
  | 25% | 0.869 | 0.527 | +0.343 [+0.290, +0.397] |
  | 50% | 0.899 | 0.795 | +0.103 [+0.065, +0.134] |
  | 100% | 0.927 | 0.861 | +0.065 [+0.022, +0.129] |

  - 版本 × 数据量的交互：F(2, 12) = 26.35，p < 1e-4。
  - 少数据时，阶段一的诊断也变差：25% 时共享起点区分度为 0.498，100% 时为 0.804。阶段一与 π 在这里同时变化，所以这是否就是原因只是假设。

## Next Experiment

- 无。按用户决定，以成功示范为中心的训练路线已暂停（2026-09-28）；表征改由交互数据联合训练，见第三计划（`plan.md`）。

## Resolution

# ISSUE-020

Status: open

Area: 仿真 / 快照恢复不复现后续轨迹

## Observation

第二计划 Phase 26（`docs/branch_success_test.md`）：`LiberoTaskEnv` 的快照恢复能复现决策点的观测。本体状态在 98–99.6% 的状态上逐位一致，最大差 1e-4。

但它不复现之后的整条轨迹：

- 在 validate 对照中，最终 MuJoCo 状态 9 个 episode 中 9 个与重置运行不同，观测在 5 个中分叉；
- 从恢复点继续执行 π 均值块，只在 46–48% 的状态上逐位重演原失败；
- 这些分支有约 1% 成功（31 / 18 / 12 个分支）。

## Impact

分支检验的噪声底约 1%。本次检验中 match 与 π 均值块之间 +0.11–0.17 的差远大于噪声底，结论不受影响。但若要比较更细的差异，需要考虑这一点。

## Evidence

- `outputs/analysis/branch_test/*/validate`、`report`。

## Hypotheses

- 求解器的热启动状态等不在快照中，未验证。

## Next Experiment

- 在快照中加入 MuJoCo 的 warm-start 等求解器状态后重测重演率（仿真工具改动，需用户确认）。

## Resolution

# ISSUE-021

Status: open

Area: 基础设施 / NFS 回环挂载挂起与 GPU 故障

## Observation

- **node-71 的 NFS 挂起**：node-71 上 `$SHARED_HOME` 是本机的 NFS 回环挂载。2026-09-27 约 23:37 起，经这个路径启动的项目 Python 进入不可中断等待（`nfs_wait_bit_killable`），其他会话的 `tail`、`ugrep` 也受影响。`.venv` 的解释器路径指向 `$SHARED_HOME`，所以在 node-71 上运行项目 Python 都受影响。
- **node-64 GPU 3 故障**：约 23:33 进入"GPU requires reset"。之后 torch 在 node-64 上无法初始化 CUDA（NVML 看到 8 个设备，CUDA runtime 只看到 7 个）。在不设置 CUDA_VISIBLE_DEVICES 的约束下，整个节点不可用。

## Impact

- 在 node-71 上运行分析脚本会挂起。
- node-64 在管理员重置 GPU 3 之前不可用（用户已知悉，2026-09-28：GPU 3 放在那里不要使用）。

## Evidence

- Phase 26、28、29、30 的代理报告；
- `tmp/runlogs/s3p28/`、`tmp/runlogs/s3p30/`、`tmp/p26/`（Phase 26 工作树内）。

## Hypotheses

- 无。

## Next Experiment

- 项目 Python 改在计算节点上通过 ssh 运行。
- node-64 GPU 3：用户已知悉，放置不用（2026-09-28）。

## Resolution

# ISSUE-022

Status: open

Area: 工具 / 续训与无阶段一的 π

## Observation

- `scripts/train_stage_one.py` 不能从检查点续训：检查点只含主干与查询权重、配置与步数，没有优化器、学习率调度与采样器状态；训练器也拒绝已存在的运行目录。
- `scripts/evaluate_proposals.py` 拒绝不带阶段一的 π；无 ψ π 的 HELDOUT MSE 改用 `tmp/runlogs/s3p29/heldout_mse.py` 计算。
- 无 ψ π 的闭环仍然加载阶段一，只用于日志中的 ψ、检索表、起始观测检查与场景探针。完全去掉它需要改动冻结的 `ObservationEncoder` 与 `ClosedLoopRunner`。
- 在远程节点上，worktree 里的 `git rev-parse` 会失败，因为 worktree 的 `.git` 指向 node-71 的 `$LOCAL_HOME` 路径。启动脚本需要从文件读取提交号。
- `scripts/build_task_relations.py` 需要 `CEAM_HOST_HOME`，而 `ceam_env.sh` 已不再设置它；它的默认任务文件路径指向共享 conda 环境。Phase 32 改为显式传入 `.venv` 内的路径。

## Impact

- 暂停的阶段一训练只能从第 0 步重跑。
- 其余几项都有临时绕过办法。

## Evidence

- `docs/stage_one_ablation.md`、`docs/data_scaling.md`、`docs/libero_object.md`（各自的暂停分支上）；
- `docs/no_psi_baseline.md`。

## Hypotheses

- 无。

## Next Experiment

- 按需修复，需用户确认。

## Resolution

# ISSUE-023

Status: open（只记录，不据此改动策略；用户决定，2026-09-28）

Area: 策略 / π 均值块在部分状态上确定性失败

## Observation

第二计划 Phase 26（`docs/branch_success_test.md`）：在 π 均值块闭环失败的 episode 中，从任一决策点改执行 π 的任意一个小偏移候选，再继续执行 π 均值块，后续成功率明显高于在该点继续执行均值块：

- 随机候选：定版 0.099，投影插值 0.153，sg[D_ψ] 0.165；
- 全候选扫描中的全部候选平均：0.141、0.202、0.193；
- 继续执行 π 均值块：0.009–0.013，其中约 1% 为快照恢复的噪声底（ISSUE-020）。

也就是说，在这些状态上，任意一个小偏移都能挽回约 10–17%，而 π 均值块本身几乎确定地失败。

挽回集中在 episode 前期：决策 1–5 为 0.37–0.59，决策 16–30 不超过 0.05。

## Impact

说明 π 均值块在部分状态上存在确定性失败。按用户决定，只记录，不据此改动策略。

## Evidence

- `outputs/analysis/branch_test/summary/summary.md`。

## Hypotheses

- 无（不据此展开）。

## Next Experiment

- 无。

## Resolution

# ISSUE-027

Status: open（需要用户决定下一步）

Area: MVP / 在 π 的候选之间，match 选择不如均值块

## Observation

MVP（`docs/query_match.md`）的设定：

- **r：** 历史实现的定版 ψ + 数据 A 效果头，冻结；
- **查询与 match：** 同一观测内对比，任务编码做 SIGReg；
- **采样器：** 无 ψ π 的 32 个候选，尺度 0.05/0.1/0.2。

闭环 10 × 50，三个种子：

| 选择方式 | 种子 0 | 种子 1 | 种子 2 | 均值 ± 标准差 |
|---|---|---|---|---|
| 选 match 最高者 | 0.846 | 0.842 | 0.840 | 0.843 ± 0.003 |
| 同一批候选中均匀随机选 | 0.884 | 0.866 | 0.878 | 0.876 ± 0.009 |
| π 均值块 | 0.940 | 0.922 | 0.918 | 0.927 ± 0.012 |

- **合并配对 McNemar：**
  - 选 match 最高者对均值块：82 / 208，p = 8.7e-14；
  - 选 match 最高者对随机：142 / 192，p = 0.0073。
- **选中的候选：** 79% 为尺度 0.2 的候选；均值块在 21,623 次决策中只被选中 1 次。
- **新增失败：** 多为"推开但未抬起"，集中在任务 2 与任务 7。
- **离线：** 1,084 个扫描状态上：
  - 同尺度 AUC 为 0.513–0.520；
  - 最高分候选 − 随机为 +0.031 至 +0.035；
  - 与历史的 q_l 用同一 r 时的结果无差异（p ≥ 0.40）。

## Impact

在当前 r 与这一采样器下，match 选择使闭环变差；只要离开 π 均值块，本协议下就要损失约 5 个百分点。

## Evidence

**r 在同一观测下对 π 候选的可区分程度很低**（框架 2，动作表征侧）：

- 扫描状态上，32 个 π 候选的 r 的状态内方差占比为 0.020；Phase 5 反事实集上为 0.017。
- 状态内有效秩约为 2；候选之间两两余弦为 0.997。

**q 的情况：**

- 示范块胜过全部 π 带噪候选的比例：TRAIN 44–46%，HELDOUT 29–31%。
- HELDOUT 的 top-1 从第 500 步起停在约 0.25。
- 把文本换成其他任务后，HELDOUT top-1 从约 0.24 降到约 0.12，所以查询依赖任务。
- 任务编码最近任务归类准确率 1.0。

**诊断性线性读出**（只用于诊断，不作训练）：在扫描状态上，按成败对候选排序的状态内 AUC：

| 读出对象 | AUC |
|---|---|
| r | 0.552 |
| 动作块本身 | 0.541 |
| 只看偏移尺度 | 0.599 |
| 训练得到的 q | 0.502 |

**实现层面：**

- 温度上限 s ≤ 100 时，InfoNCE 排不开余弦约 0.999 的候选；小集合过拟合只到 13/16。
- 训练时 68% 的 π 带噪候选因距离过近被排除，所以训练与推理时的候选分布不一致。

## Hypotheses

以下未验证：

- 首先受限于 r：π 的候选在 r 中几乎重合。
- 其次是 q：它连 r 中已有的弱信号都没有用上，于是最高分偏向最大的偏移。

## Next Experiment

待用户决定，见当日汇报中的选项。

## Resolution

# ISSUE-033

Status: open

Area: 聚焦 q/m / 同起点成功对比在闭环中形成有害偏好

## Observation

- 三个训练 q 都把 812 个 TRAIN 混合组拟合到 top-positive 1.0 和候选 oracle 0.6819。
- 在 360 个 HELDOUT 失败起点上，训练后的候选成功率为 0.4000/0.4111/0.4361，低于各自 round 0 的
  0.4417/0.4667/0.4528。
- 在 1,500 个闭环 episode 上，训练 q 的均值为 0.8047，低于 round 0 的 0.9020、uniform 的 0.8760 和
  π mean 的 0.9267。相对 round 0 的三个种子差值为 −0.096、−0.094 和 −0.102。
- 训练 q 对 uniform 的合并独赢/独输为 126/233，McNemar p = `1.75e-8`。因此 q 学到的偏好会主动降低
  成功率，而不是仅仅没有利用已有信号。
- 十个任务的平均成功率都没有超过 round 0；降幅最大的是任务 1、7 和 2，分别为 −0.267、−0.187 和
  −0.153。

## Impact

当前 q/m 训练不能作为框架中任务相关局部对齐的实现。R6_all、候选 sampler 和聚焦标签保持固定时，继续
增加同一聚焦数据的训练步数没有闭环依据。LIBERO-90 基线可以继续执行，但下一代表征或 q/m 实验不能把
TRAIN 可分性或 top-positive 当作主要选择标准。

## Evidence

- `outputs/focused_query/reports/offline_metrics.json`
- `outputs/focused_query/reports/closed_loop_metrics.json`
- `docs/focused_experiment.md`
- `outputs/closed_loop/focused_argmax_s{0,1,2}/`

## Hypotheses

- 聚焦数据只来自 π 均值策略失败 episode 的前 15 个决策，q 可以利用该条件分布中的候选来源、尺度、状态
  或任务捷径，却没有学到闭环访问状态上的动作效果等价关系。
- 一次候选分支后由 π 均值策略续跑得到的最终成败是长时域、高方差标签；它可能把后续 π 的恢复能力归因
  给当前候选，不能稳定定义局部动作效果。
- R6_all 的冻结差分坐标让训练 q 通过记忆训练状态完成分离，但没有提供跨场景与跨状态一致的预测结构。

## Next Experiment

下一代表征学习计划把候选来源与尺度可预测性、跨 episode 和跨场景迁移、短时域可预测后果及最终成功分别列为
独立观测。下一代表征先用示教超集学习任务无关的动作效果结构，固定后再训练 q/m；保持 sampler、配对
初始状态、uniform、round 0 和 π mean 对照不变。

## Resolution

# ISSUE-029

Status: open

Area: 集群网关 / bash 等待进程无法经 kill_project_pid.sh 结束

## Observation

- Phase 21 的第 5 轮等待脚本 `same_data_waiter.sh`（node-62 上 pid 2754227）是一个每 60 秒检查一次的 bash 循环。
- 它的 argv0 是 `bash`，所以 `kill_project_pid.sh` 拒绝结束它，退出码 3。
- 它不占 GPU，按 6 小时超时约在美东 10:48 自行退出。

## Next Experiment

以后的等待进程改用项目路径下的 Python 进程，或者改为检查一个停止文件。

## Resolution

# ISSUE-030

Status: open

Area: LIBERO-90 原始演示重生成验证

## Observation

- `s6-l1` 的 `LiberoTaskEnv` 可选基准任务名接口已通过 Goal 对照：默认路径和按任务名路径在 96 个数组上均为零差异。
- 空 fixture 序列化已修复。六任务 LIBERO-90 抽样全部写出并可加载；两个 living-room 任务的 fixture position 和 quaternion 形状分别为 `[50, 0, 3]` 和 `[50, 0, 4]`。
- Goal 发布数据有 433/500 个演示；两次独立重生成均保留 432/500。缺失项为 `push_the_plate_to_the_front_of_the_stove/demo_25`：当前 replay 在 action 135 后首次成功，在最后一个 action 后失败。
- 若改用 `any(done)`，重生成会保留 448/500，而不是发布数据的 433/500；相对 final-done 规则新增 16 个演示。因此发布 kept set 不能通过简单改成 any-success 恢复。
- `push_the_plate_to_the_front_of_the_stove/demo_25` 的 raw 长度、no-op-filtered 长度和匹配发布 episode 长度均为 153；first-done 为零起始 action index 135，final-done 为 false。
- 433 个发布 episode 均按长度和 float32 action 唯一匹配，动作最大绝对误差为零。joint 最大误差为 `9.302508668e-3`，gripper 最大误差为 `7.54e-4`；17 个 episode 的 joint 误差超过 `1.5e-4`。
- 两次 Goal 重生成在排除 provenance 与运行耗时后，每个任务的 22 个 payload 数组都逐位相同。最大误差 episode 与数值也和 `docs/demo_renders.md` 已记录的 replay divergence 相同。

## Impact

空 fixture 序列化不再阻塞转换。当前 replay 的 kept set 和发布数据不同，但所有发布 episode 的动作都能精确映射；该差异作为数据质量与跨场景训练覆盖信息继续报告。

## Evidence

- 实现分支：`tmp/worktrees/phase-1-l1-validation`；基础实现提交 `1d6dfe0` 与 `3a9f145`。
- 原始数据：`outputs/raw/LIBERO-datasets/libero_90/`，90 个 hdf5，共 66.7 GB。
- fresh Goal 产物：`outputs/analysis/libero90_prep/phase1_goal_regeneration/`，92 MiB。
- fresh 六任务产物：`outputs/analysis/libero90_prep/phase1_libero90_sample/`，57 MiB。
- node-65 CPU-only 运行从美东 02:37:57 到 02:43:46，共 349 秒 wall time、1,757.97 CPU-seconds；没有申请 GPU。每进程峰值 RSS 为 1.83–2.70 GiB。

## Hypotheses

- 已验证的当前 replay 是确定性的；重复执行不会消除相对发布数据的漂移。
- 发布数据生成时产生不同轨迹的具体运行条件或实现差异尚未确定。当前仓库没有足够证据把差异归因于某一个环境参数。

## Next Experiment


## Resolution

Goal 对照以 433 个发布 episode 的唯一动作映射和动作逐位一致作为转换验收；replay kept set、proprio 误差和两次重生成一致性作为直接观测。该规则不掩盖 replay 差异，也不要求确定性 replay 复现未知的发布运行条件。六任务抽样全部写出并可加载，Phase 1 完成。

# ISSUE-035

Status: open

Area: LIBERO-90 / cluster execution interruption

## Observation

The node-66 Phase-8 resource log stops at 2026-09-30 15:02:47 EDT, without a completion marker. At the 19:17 EDT
check, neither its evaluation workers nor continuation process existed; its GPUs were occupied by other jobs.
Seed 2 had 4,185 of 4,500 episode rows and eight incomplete task runs. Twenty rollout and π-noise tasks had not run.

## Impact

The missing work has been recovered on node-61 and node-63, one GPU per node. All 13,500 benchmark rows and both
90-task store manifests are complete. The interruption's cause and exact termination time remain unknown.

## Evidence

`tmp/runlogs/phase8/full_single_node3/`, `outputs/analysis/libero90_recovery_20260930/interruption_snapshot.json`,
`tmp/runlogs/phase8/recovery_mean_seed2/` and `tmp/runlogs/phase8/recovery_stores.status`.
The two partial task directories are retained in the recovery analysis directory. All eight incomplete tasks were
rerun with identical checkpoint and benchmark initialization settings; aggregation uses complete reruns only.

## Hypotheses

No available evidence identifies whether node maintenance, process termination or another external event stopped
the workers. Resource accounting for the interrupted allocation is a logged lower bound.

## Next Experiment

Compare any project-local worker termination evidence with the final resource timestamp if such evidence becomes
available. Recovery does not require access to other users' jobs or system logs.

## Resolution

# ISSUE-037

Status: open

Area: Demonstration refinement / allocation-dependent sampling throughput

## Observation

The same in-memory Goal proposal recipe sampled in about 0.021–0.024 seconds on node-61 and 0.687–0.849 seconds on
node-63, yielding about 9.4–9.6 versus 1.1–1.3 updates/s. Both node-63 processes held about 11 GiB RSS. A short read-only
memory check reported about 276 GiB available and no swap-in/out during the observation; this does not identify the
cause of the sampling discrepancy.

## Impact

The two node-63 training processes were stopped by exact project PID. Partial run directories are preserved under
`outputs/demo_refinement/pilot_v1/aborted_training/`. Their complete 3,000-update cells will restart with unchanged
data, seed, architecture and optimizer on free GPUs of node-65. Completed node-61 checkpoints are reused.

## Evidence

`tmp/runlogs/demo_refinement/grid_lane{0,1,2,3}/train_*.log`, their resource CSVs, and the project-gateway invocation
of `tmp/demo_refinement/status_memory.sh`.

## Hypotheses

Node-dependent memory-access conditions may contribute, but the read-only observation cannot attribute the cause.

## Next Experiment

The restarted cells on node-65 reached about 9.26–9.28 updates/s and 0.0246–0.0249 seconds/sample by updates 400–450.
Subsequent cells also varied within allocations: node-65 moved between about 3.3 and 9.2 updates/s; the final
node-61 guided-seed-2 cell completed at about 3.46 updates/s with sample_seconds near 0.206. This timer includes
batch construction and device transfer. The IID sampler, data and training recipe remained unchanged. All nine
full checkpoints and evaluations completed; the throughput discrepancy's cause remains unknown.
A future resource investigation needs separate host-gather and transfer timings on fixed sampled indices.

## Resolution

# ISSUE-038

Status: open

Area: Demonstration refinement / source cost and downstream policy quality

## Observation

Random and guided reduce selected-source steps from 3,531 to 3,379 and 3,397. Their pooled policy SR is 87.13%
and 90.27%, versus 88.93% for original. Guided − original differs by −2.80 / +0.20 / +6.60 pp across π seeds;
the benchmark-paired interval is [−0.73, +3.40] pp. Source step/command cost ordering does not match policy SR ordering.

## Impact

The fixed pilot supports guided versus random shortlisting, but does not establish reliable BC benefit over original
data across training seeds. Source trajectory cost alone is insufficient evidence for a better demonstration target.

## Evidence

`docs/demo_refinement_results.md`; `outputs/demo_refinement/pilot_v1/reports/policy_comparison.json`;
`outputs/demo_refinement/pilot_v1/reports/selection_summary.json`.

## Hypotheses

Training variation, selected source coverage and the relation between trajectory cost and BC learnability may
contribute. This grid does not isolate their contributions.

## Next Experiment

Replicate fixed-bank comparisons across refinement/source seeds and additional π seeds before increasing edit scope.
Keep an original control and separate true trajectory cost from downstream policy benefit.

## Resolution

# ISSUE-039

Status: open

Area: LIBERO-90 R6 / transfer normalization and scene-label interfaces

## Observation

`SceneLabelTable.read` rejects the complete 89-shard LIBERO-90 held-out counterfactual store because object/fixture
names differ. A CPU load reproduces the error on a KITCHEN_SCENE1 cabinet task. The representation CLI also checks
that every model names the evaluation config's feature/action caches; Goal R6_all names Goal caches, while the
LIBERO-90 baseline requires its own statistics.

## Impact

The existing Goal diagnostic contract cannot implement the full LIBERO-90 transfer/scene report. The deferred
work is recorded in `plans/libero90_pending_work.md`; the dedicated native-normalization and topology interfaces
require implementation and validation before PCA/head execution. Expanded representation work remains pending.

## Evidence

`outputs/analysis/libero90_evaluation_contracts/scene_loader.json`;
`ceam/diagnostics/scene_label_table.py`; `scripts/evaluate_representation.py` common-cache validation;
`ceam/diagnostics/representation_model.py` native input contract.

## Hypotheses


## Next Experiment

`plans/libero90_r6_evaluation.md` defines model-native input statistics and topology-local scene labels. Validate
normalization equivalence, global frame coverage and task attribution before running the baseline pipeline.

## Resolution

# ISSUE-040

Status: open

Area: Retrieved-action ranking / task compatibility and successful-branch quality

## Observation

The completed retrieval pilot contains 180 banks and 1,618 same-start branches. Physical retrieval supplies a
successful faster donor at 71/90 starts. Original-q direct match selection succeeds at 85.56%, versus 79.69% for
uniform all-bank expectation and 100% for original. Focused query seeds achieve 78.89 / 76.67 / 80.00%.

Original-q four-donor shortlisting with physics acceptance reduces mean steps from 113.267 to 110.256, but its
difference from exact random shortlisting is −0.087 steps with interval [−0.294, 0.135]. All eight retrieval/query
shortlist step comparisons span zero. Physical-bank successful minimum-step oracle raises EEF jerk RMS from
6.933 to 7.852 m/s³ while reducing mean steps to 109.456.

## Impact

Current frozen match can damage a valid demonstration when used as the sole edit selector. This grid provides
no stable shortlist efficiency advantage over random and no physical smoothness advantage over original.
Successful faster alternatives are available in the bank, enabling ranking improvement to be evaluated without
first expanding the sampler. A speed-only accepted label is insufficient evidence for all three quality goals.

## Evidence

`docs/retrieval_refinement_results.md`;
`outputs/retrieval_refinement/pilot_v1/reports/comparison.json`;
`configs/refinement/retrieval_pilot.json`.

## Hypotheses

Candidate-distribution shift, insufficient frozen action distinctions, query training targets and the absence of
successful-branch quality supervision may contribute. The frozen comparison does not isolate these mechanisms.

## Next Experiment

OPT-011 describes a separate TRAIN-branch collection and fixed-r ranking comparison. Keep successful/failed
compatibility evidence separate from preferences among successful alternatives. Treat the existing HELDOUT
branches as evaluation/development evidence rather than training labels; use fresh held-out source episodes for
confirmation after this result has informed design. Re-render changed trajectories before any policy training.

## Resolution

# ISSUE-041

Status: open

Area: Physical world pilot / generalized-velocity replay repeatability

## Observation

An independently repeated original ten-step chunk for task 1, development episode 69, anchor 170 differs
from its recorded endpoint by 0.0730366 in one generalized-velocity component. Worst EEF/object xyz distance
is 4.56e-7 m, maximum qpos difference 7.09e-5 and object rotation-matrix element difference 1.10e-5.
The same task's other calibration/development original endpoints agree in xyz. TRAIN original endpoints
match overlapping demo windows to within 1.20e-7 m per object coordinate.

## Impact

Whole-state maximum error combines quantities with different units and does not represent a translation error.
The observed pose discrepancy is below the pilot's millimetre reporting thresholds. Velocity targets have
measured numerical variation; strict complete-state reproducibility is unestablished. No source was excluded.

## Evidence

`outputs/physics_world/pilot_v1/tasks/task1_reference_diagnostic.json`;
`outputs/physics_world/pilot_v1/tasks/replay_diagnostic.json`; task manifests and schemas.

## Hypotheses

The cause of the endpoint velocity variation has not been isolated.

## Next Experiment

Before using velocity predictions for contact-sensitive control, repeat the affected source with per-joint
velocity and per-step contact observations, distinguishing restore variation from subsequent solver variation.

## Resolution

# ISSUE-042

Status: open

Area: Physical world pilot / prediction precision, grasp changes and confidence utility

## Observation

On 13 fresh HELDOUT sources across nine tasks, mixed latent-128 has mean worst EEF/object xyz error 22.26 mm,
p95 44.51 mm and same-start centered effect R² .658. Its grasp accuracy is 96.36%, but only 6.85% of the
73 actual changed grasp signatures are predicted. Demo ridge predicts 35.62% of grasp changes. Positive
endpoint grasp prevalence is 7.55%; changed-signature prevalence is 3.91%.

All seven predictors reject every development/fresh candidate under the fixed calibrated 5/10/20 mm bounds.
Mixed latent-128's smallest fresh bound is 32.45 mm and median 91.37 mm. Actual task failure among accepted
candidates is undefined because no candidate is accepted.

## Impact

The predictor supplies coarse action distinctions but cannot yet justify skipping physical verification. Overall
grasp accuracy conceals missed transitions. Coordinate endpoints alone do not establish collision-free paths,
physical consistency or task success. This result does not evaluate learned task matching or policy SR.

## Evidence

`docs/physical_world_results.md`; `outputs/physics_world/pilot_v1/reports/comparison.json`;
`outputs/physics_world/pilot_v1/reports/observations.json`; fixed model/config/source provenance.

## Hypotheses

Insufficient contact-event coverage, state/target design, loss scaling and shared-model errors may contribute.
The current grid does not isolate these causes. Episode-max pooled calibration may also limit forecast coverage.

## Next Experiment

OPT-012 defines an independent short-horizon, object-relative/contact-aware physical predictor comparison.
Retain the current baseline and candidate banks. Preserve independent calibration and fresh confirmation.
Do not loosen an error threshold solely to make a predictor accept candidates.

## Resolution

# ISSUE-043

Status: open

Area: Preference feedback / benchmark success and observed joint-limit feasibility

## Observation

The fixed feasibility screen allows arm joint range excess up to 1e-3 rad. Some successful original TRAIN
demonstrations exceed this: task 1 episode 54 reaches 0.009066 rad excess, episode 57 reaches 0.004829;
task 8 episodes 353–356 and task 9 episode 396 also exceed the screen. Their original commands are valid
and the sampled arm/environment contact count is zero. All such banks are retained. The complete collection
rejects 21/9/6/3 original banks in TRAIN/calibration/development/remaining HELDOUT. Two development banks and
one remaining-HELDOUT bank have no candidate satisfying success and observed feasibility.

## Impact

Benchmark task success does not imply acceptance under this experiment's declared physical screen. Some banks
have no feasible-successful candidate, so preserving the original cannot guarantee this combined criterion.
Compatibility training and reporting keep task success, sampled feasibility and their conjunction separate.
The screen observes control-step endpoints and does not certify continuous collisions or hardware safety.

## Evidence

`outputs/preference/pilot_v1/banks/`; `outputs/preference/pilot_v1/tasks/preliminary_coverage.json`;
`ceam/preference/branch_validity.py`; `docs/preference_feedback_results.md`; `plans/preference_feedback.md`.

## Hypotheses

The reason for the observed joint-range overshoot has not been isolated.

## Next Experiment

For control or dataset publication, inspect per-joint excess timing and physics substeps, and define a physical
feasibility protocol appropriate to the platform. Do not relax the threshold merely to accept more examples.

## Resolution

# ISSUE-044

Status: open

Area: Preference feedback / compatibility generalization and direct selection

## Observation

On remaining preference-training-held-out sources, fixed-r preference learning improves successful-feasible
Pareto ranking from 52.79% to 61.49%; task-independent r adaptation raises it to 69.98%, averaged over three seeds.
TRAIN preference accuracies are 82.31% and 87.65%. Compatibility pair ranking remains 56.08% and 55.71%; Brier
scores are .199 and .255. Direct predicted-compatibility-gated preference selection succeeds on 94.02% and
85.47% of branches, versus 100% original task success. Combined success/observed-feasibility rates are 84.62%
and 82.91%, versus 92.31% original. Actual-verified shortlisting improves time/path but remains verifier-dependent.

The added preference target does not constrain jerk. Original remaining-HELDOUT successful-feasible RMS jerk is
5.63, versus 7.03 and 6.96 for fixed/adapted preference shortlists on their respective successful-feasible subsets.

The input/label audit verifies that the validity observer starts with the common prefix's accumulated result,
but bank files retain only whole-path aggregates. Prefix, candidate and continuation violations cannot be
separated losslessly from those aggregates. Outcomes use the original source suffix and source horizon;
neither an explicit continuation descriptor nor remaining-budget field enters the matcher. A paired diagnostic
now measures 46/239 success-label changes and 191/448 strict preference reversals under source versus frozen-policy
continuation. The exact-intervention subset retains 176/380 reversals. All 20 original candidates succeed under
both continuations. Evidence: docs/continuation_sensitivity_results.md.

## Impact

Preference supervision is useful but cannot justify direct label edits or control. Representation adaptation
does not consistently improve compatibility or smoothness. The result does not establish that dataset quantity,
embedding capacity or any single loss term is the bottleneck. Local fixed-suffix ranking does not resolve
redundant task operations or alternate strategies.

## Evidence

`docs/preference_feedback_results.md`; `outputs/preference/pilot_v1/reports/comparison.json`;
`reports/validation.json`, `reports/all_role_ranking.json`; twelve fixed-grid models and recorded source provenance.
Input/label ownership and literature assumptions: `docs/long_horizon_matching_review.md`.

## Hypotheses

Limited source/event coverage, unobserved controller/velocity/interaction state and shared-r target interference
may contribute. Feasibility labels include common-prefix violations, which are not necessarily observable from
the model's current images/proprio; this experiment does not isolate that information boundary.

## Next Experiment

Phase-local records and continuation sensitivity are measured in plans/continuation_sensitivity.md.
OPT-013 retains the subsequent fixed-continuation learning and matched architecture controls. Retain all twelve current
cells as controls. Assess source coverage and smoothness before regenerating observations, publishing action
labels or training policies. A model's own high match is not a better-action label.

## Resolution


# ISSUE-045

Status: Open

Area: Continuation diagnostic restore sensitivity

## Observation

Paired source/policy executions of identical candidate commands at episode 292 have a maximum candidate-stage
simulator-state difference of 0.045091. Maximum EEF-position difference is 7.393e-8 m and proprio difference is
7.264e-7. Differences begin at the restored boundary; their cause has not been isolated. Episode 414 has a much
smaller 1.525e-12 state difference. The other eighteen starts have bit-identical paired state traces.

## Evidence

outputs/preference/continuation_v1/reports/integrity.json; docs/continuation_sensitivity_results.md.
All twenty independent original-branch replays reproduce success sequences and discrete feasibility fields.
Task 8's floating joint-excess difference is 1.794e-11 rad and does not change feasibility or contact counts.

The residual-editing follow-up reproduces the sensitivity at episode292: one selected candidate has .03994
maximum paired candidate-stage state discrepancy, and an original-policy repeat differs by 2.84025 over its full
state trace while retaining identical steps and success sequence. Six episode414 candidate pairs differ by at
most 1.355e-11. All forty repeated-original controls preserve success sequences and step counts. The eighteen
starts excluding292/414 retain policy success losses and loss of source quality improvements. Evidence:
docs/residual_editing_results.md and outputs/residual_editing/pilot_v1/reports/continuation.json.

## Impact

Full-grid comparisons include a control discrepancy. The bit-identical-start sensitivity subset still shows
44/216 success-label changes and 176/380 strict preference reversals, so the measured continuation dependence
is not confined to affected starts. Exact replay of every simulator field is not established.

## Hypotheses

Contact/controller state not fully captured by the existing restore recipe may contribute. Its cause is not verified.

## Next Experiment

If future training or acceptance uses these starts, isolate residual simulator state and assess repeated same-mode
rollouts before making per-candidate causal or safety claims. Retain the full dataset and the exact-state sensitivity
subset. The completed diagnostic does not require additional collection.

## Resolution


# ISSUE-046

Status: Open

Area: Cross-task physical-effect coverage and diagnostic interpretation

## Observation

In 5,135 physical-pilot demonstration windows, all 390 plate displacements above 5 mm occur in task 2;
all 137 initial bilateral cream-cheese contacts occur in task 6. Contact fields record both finger pads touching
an object simultaneously; they do not represent all contact modes. A fixed 256-window cross-task ridge grid
retains EEF discrimination under other-task training but has object-centered R² −.4622 even with relative/contact
interaction features. The overall effect R² is .0012. Own-task interaction training achieves .4817 overall.

## Evidence

`docs/cross_episode_representation_research.md`;
`outputs/physics_world/cross_task_probe_v1/{manifest,results,summary,coverage,validation}.json`;
`ceam/world/physical_state.py`, `ceam/world/cross_task_effect_probe.py`.
All 360 fits and ten persistence controls are retained. The 5 mm threshold is a post-fit descriptive statistic,
not a training or evaluation filter. Source episodes are disjoint; the evaluation banks were used in prior research.

## Impact

Holding out task identity also removes some interaction coverage. The current flat model gives named object slots
separate coefficients and has no explicit shared object-dynamics mechanism. Cross-task error therefore cannot
isolate whether reusable cross-context representations are possible. Privileged inputs and a shared physical
schema also do not establish transfer across visually unlike scenes. The target excludes fixture joint motion,
rotation and feasibility. Stationary objects dominate many absolute-error measurements.

## Hypotheses

Shared relative-interaction parameters, better contact-mode evidence and broader action coverage may improve
transfer. Preserving spatial visual information may improve contact observability. None is isolated by this grid.

## Next Experiment

OPT-014 specifies a coverage-aware cross-context representation study. Separate seen interaction modes in unseen
contexts from unseen interaction modes; report object-specific and candidate-centered effects. Compare shared and
unshared predictors under matched inputs and source windows before enlarging representation pretraining.

## Resolution

# ISSUE-047

Status: Open

Area: Teacher-based editing quality and continuation dependence

## Observation

The 600-cell frozen-teacher pilot gives 16/144 Pareto improvements for direct Gaussian edits, 24/144 for teacher
noise and 19/144 after TRAIN variance calibration. Illegal-command rejections are 19, 7 and 5 respectively;
conditional on legal execution, successes are 123/125, 129/137 and 131/139. Calibration's two additional total
successes coincide with two fewer command rejections. It has no consistent added quality yield in this grid.

Source-suffix winners do not preserve their efficiency ordering under a common teacher continuation. At one
selected drawer location, a calibrated edit changes source-suffix completion from 174 to 129 steps, but under
shared continuation seed 1 it takes 191 steps against the original's 136. It rescues another failed original
continuation, so success and efficiency changes must remain separate. Across two selected locations and three
continuation seeds, calibration rescues two original failures but has no paired Pareto improvement when both
branches succeed. Reconstruction alone preserves all source-suffix successes yet changes three completion times
by one step, produces one Pareto improvement and changes one EEF path by 7.70 mm.

## Evidence

`docs/teacher_editing_results.md`; `outputs/teacher_editing/pilot_v1/report/all_cells.csv`;
`outputs/teacher_editing/pilot_v1/continuation/{all_cells.csv,summary.json}`;
`configs/teacher_editing/{pilot,continuation}.json`. All 59 numerical cells, 600 source-suffix cells and
36 selected continuation cells are retained. No candidate quality labels were used in TRAIN scale estimation.

## Impact

Coordinate calibration and accurate local reconstruction do not establish trajectory-quality ranking or invariance
to the continuation. Six query episodes and two selected follow-up locations do not establish general editor
superiority. The current random direct baseline also leaves simpler bounded/temporal alternatives untested.
No downstream policy trained on edited labels has been evaluated.

## Hypotheses

Dataset summaries that retain temporal/contact context or measured preferences may help beyond variance scaling.
Small command changes may alter contacts and subsequent visual-policy decisions, amplifying reconstruction
residuals. These mechanisms are not isolated by the pilot. The observations do not reject broader teacher/dataset
compression or direct adaptation.

## Next Experiment

OPT-015 retains trajectory-context and quality-aware calibration as future candidates. Specify the eventual
continuation or deployment policy, compare against bounded and temporally structured direct edits, and fit any
selector on separate TRAIN counterfactuals. Assess held-out candidate ranking before increasing teacher or student
training cost. Downstream data utility requires matched BC/adaptation tests from the same teacher initialization.
`docs/action_label_optimization_review.md` records phase-aware relabeling and quality-supervision precedents.
Define whether replacement labels are BC targets or measured trajectories; only the latter implies regenerated
observations. The completed pilot's fixed gripper timing and ten-step edits do not assess full-phase replacements.

## Resolution

# ISSUE-048

Status: Open

Area: Structured action edits, selection transfer and continuation-dependent utility

## Observation

The residual-editing grid executes 1,473 candidates at twenty starts. Under success, sampled feasibility,
nonincreasing time/path and at most five-percent jerk increase, donor rank-6/rank-12 edits improve 11/10 starts,
donor IID improves two, and original-centered rank-12 improves sixteen. Without the jerk condition all five
donor/direct/residual families find time/path improvements at seventeen starts. Low-frequency structure supplies
a useful smoothness prior, but the tested donor-residual pools do not exceed direct transfer's matched-budget
mean saved steps.

With identical information and approximately 0.75 million parameters, a new direct scorer has 69.59% query pair
accuracy and a new dual matcher 59.52%; historical adapted preference matchers have 59.76%. Actual-verified
four-candidate shortlists have no established gain over random: direct saves .349 steps with descriptive
episode-bootstrap interval [-.752,.026]; dual changes steps by +.051 with interval [-.470,.722].

Source-selected original-centered residual edits retain only three of sixteen quality improvements under the
frozen policy and lose two successes. Direct donor edits retain three of ten and lose two successes. The exact
candidate-state eighteen-start subset retains these two-success losses; quality transfer remains weak.

A target-alignment follow-up retains 2,269/3,334 TRAIN preference pairs after imposing a five-percent jerk
bound. Three-seed direct and dual verified shortlist means both become 109.783 steps, versus random109.849;
neither change establishes an advantage. Direct joint-quality pair accuracy changes from66.71% to64.86%,
dual from57.40% to58.23%. Filtering old preference pairs alone does not resolve this selection gap.

## Evidence

docs/residual_editing_results.md; outputs/residual_editing/pilot_v1/reports/;
twelve newly trained and six historical evaluated checkpoints; 182 continuation/control branches.

## Impact

The current source-suffix preference does not establish continuation-independent action quality. Pairwise
ranking improvement does not establish useful small-budget selection or edited-label student SR. The completed
study retains all successful and failed branches but publishes no replacement dataset. The fixed policy's
inability to exploit some edits does not disprove their utility when learning from coherent regenerated data.

## Hypotheses

Effect/endpoint compatibility, short motion history and declared continuation context may improve selection.
Phase-level edits may expose larger gains than ten-step perturbations. These mechanisms have not been isolated.
The direct scorer result leaves dual factorization's benefit unestablished under these inputs and targets.

## Next Experiment

Define a segment's required effects and its continuation before collecting new preferences. Keep source and
unedited donor references available while testing adaptations. Separate actual candidate availability, learned
shortlisting and downstream policy utility. Compare against the direct scorer with matched information; assess
jitter explicitly. Restore sensitivity at episode292 needs isolation before accepting that start's counterfactuals
as precise causal training evidence. Broader phase editing and learned generators remain future work in OPT-015.

## Resolution
