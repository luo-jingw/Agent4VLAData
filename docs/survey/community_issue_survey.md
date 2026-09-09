# 社区共性问题综述

> 数据源：GitHub issues/PR（检索日期 2026-09-07），覆盖
> openpi、lerobot、SmolVLA、GR00T、OpenVLA、Diffusion Policy、ACT、Robo-DM
> 等 repo（检索过程见 `logs/search_log.md`）。
> 结论分级沿用 `design/problems_and_requirements.md` §4 体系：
> [跨仓库反复] = 多个独立仓库反复出现；[单点] = 单仓库出现（可能是 feature
> request 而非 bug）；[社区自建] = 社区为了弥补工具缺失自写的工具。
> 注意：issues 数量≠流行度，很多条目是 feature request 或环境问题。

## 1 主题聚类与结论

### 1.1 数据质量诊断工具缺失，社区在自建（[跨仓库反复+社区自建]）

社区最常见的动作：**没有官方质量工具，自己写**。

- [Dataset quality diagnostic tool -- catches silent training failures](https://github.com/huggingface/lerobot/issues/3280)（lerobot，诊断静默训练失败）
- [feat: Add lerobot-analyze-dataset CLI for dataset quality analysis](https://github.com/huggingface/lerobot/pull/2936)（5 维评分+行动建议）
- [Feature: add a read-only normalization compatibility report](https://github.com/Physical-Intelligence/openpi/pull/1026)（norm stats vs 训练配置比对）
- [Feat/norm stats compatibility report](https://github.com/Physical-Intelligence/openpi/pull/1026)（同上，验证文档）

**含义**：质量环节"工具没成熟"（我们的调研结论）在社区被实证——现象发生在每家大 repo，且都以"造轮子"的方式解决。**这直接支持我们的 ③ 标注模块 + 训练前门控（`H(ℓ|v)` 探针、几何代理掩码）设计方向：社区缺的正是"可复用的诊断接口"。**

### 1.2 元数据与 episode 边界损坏是反复出现的 bug 类（[跨仓库反复]）

- [compute_episode_data_index silently returns wrong boundaries when episode lengths are corrupted](https://github.com/huggingface/lerobot/issues/4143)（静默错边界）
- [fix(rl): crop_dataset_roi writes incomplete episode metadata and drifts episode boundaries](https://github.com/huggingface/lerobot/pull/4561)
- [fix(datasets): preserve feature fps metadata in merge/split/delete operations](https://github.com/huggingface/lerobot/pull/2969)
- [v3 datasets disagree on task association: episodes-index tasks vs per-frame task_index - which is canonical?](https://github.com/huggingface/lerobot/issues/4519)
- [StreamingLeRobotDataset yields zero frames when video decode fails, with no error surfaced](https://github.com/huggingface/lerobot/issues/4066)（静默零帧）
- [pi05_base pretrained model has action/state dimension mismatch with lerobot/libero dataset, causing official Quick Start example to fail](https://github.com/huggingface/lerobot/issues/2963)（维度不匹配）
- [X-VLA LIBERO: Inconsistent config.json](https://github.com/huggingface/lerobot/issues/2954)（配置不一致）

**含义**：静默失效（silent failure）是社区最高频的抱怨模式：行数对不齐、边界漂移、解码失败不报错、配置不一致——**这正是我们管线 ② 完整性检查 + ⑩ 验收模块的理由**（`verify_dataset.py` 6 项 + `probe_arc_batch.py` 端到端探针）。我们的 9 项验收在社区没有等价物。

### 1.3 相机与帧对齐问题是跨 repo 的主题（[跨仓库反复]）

- [X-VLA Camera count mismatch and rollout arguments parsing issues with 2-camera setups](https://github.com/huggingface/lerobot/issues/4245)（相机数量不匹配）
- [feat(scripts): add lerobot-check-cameras to pin and restore camera mount poses](https://github.com/huggingface/lerobot/pull/4495)（相机安装位姿校验）
- [feat(record): guard against dropped image writes during recording](https://github.com/huggingface/lerobot/issues/4468)（录制丢图像）
- [Unitree G1 gripper control + multi-camera streaming](https://github.com/huggingface/lerobot/pull/3984)（多相机）
- [fix(async_inference): do not resize camera frames to the dataset resolution](https://github.com/huggingface/lerobot/issues/4468)（推理侧 resize 不一致）

**含义**：相机环节的问题不只在采集歧义（"无 VR"那种信息通道差），还有**工程层**的：录制丢帧、相机装位漂移、推理 resize 不一致。对应我们 ISSUE-002 与 ② 对齐模块 + ⑤ 观测处理，且证明社区没有统一解法。

### 1.4 标签与数据集编辑操作的语义不完整（[社区自建]）

- [feat: Add feature to drop last n keyframes for delta_timestamps](https://github.com/huggingface/lerobot/pull/129)（delta_timestamps 尾帧处理 → 与我们的"集尾钳制/21.7% padding"同厂）
- [feat(trim): adding optional trimming option in reencode_video](https://github.com/huggingface/lerobot/pull/3779)（视频裁剪）
- [feat(teleoperators): add accessible_teleop for operators without a leader arm](https://github.com/huggingface/lerobot/pull/4299)（无 leader 臂的遥操）
- [feat(datasets): rename_feature / merge / split / delete 工具系列](https://github.com/huggingface/lerobot/pull/4518)（数据集编辑工具族）

**含义**：社区在数据集编辑能力上有明确需求（裁剪、重采样、重命名、合并），但这些都停留在"单功能 PR"，没有系统性方法——**这支持我们的 ⑥ 标签构建模块做系统性重采样 + 契约声明的差异化**。

### 1.5 训练后部署平滑性 / 输出抖动（[跨仓库反复]）

- [Why does lerobot-pi05 have strong inference jitter, while openpi-pi05 has smooth inference?](https://github.com/huggingface/lerobot/issues/2559)（跨实现推理抖动差异）
- [Post-hoc EMA smoothing and velocity clipping for rollout actions](https://github.com/huggingface/lerobot/issues/3794)（事后平滑+速度限幅）
- [feat(multi_task_dit): add Real-Time Chunking (RTC) inference support](https://github.com/huggingface/lerobot/pull/4394)（RTC 推理）

**含义**：数据侧的非等时标签 vs 部署插值的"消费契约"问题在社区以"抖动"的面貌出现——社区做法（EMA/裁剪）是**部署侧打补丁**，正对应我们的"运控速查表/消费契约"设计（我们主张数据侧声明契约，不是事后打补丁）。**这是支持我们 §4 的又一个社区证据：契约缺失的后果在社区实测可见。**

### 1.6 抓取/末端失败表征为"动作质量"问题（[单点，openpi]）

- [Low action loss but failed real-world pick on Franka Panda with Pi05 LoRA: possible causes?](https://github.com/Physical-Intelligence/openpi/issues/906)（loss 低但真实失败）
- [pi0.5 Robot reaches object but fails to grasp, keeps pushing object with small actions](https://github.com/Physical-Intelligence/openpi/issues/912)（到位不抓、推搡）
- [BEHAVIOR-1K / π0.5 Gripper never closes although navigation and arm reaching work correctly](https://github.com/Physical-Intelligence/openpi/issues/1012)（导航好但夹爪不闭合）
- [State/Action representation in LIBERO dataset (EEF vs Joint)](https://github.com/Physical-Intelligence/openpi/issues/637)（state/action 表示）

**含义**：开环 loss 与真实成功率的脱钩现象在社区也被反复问（"low loss but failed"）——**这正是我们证据库里 paper_005"离线指标与下游脱钩"的社区实证**。我们是"测量后再改"，社区是"问为什么"。

## 2 对照与结论

| 我们的模块/问题 | 社区对应证据 | 差值 |
|---|---|---|
| ② 完整性检查 + ⑩ 验收（6 项验收+端到端探针） | 1.2 静默失效族（4+ 独立 issue） | **社区缺等价物** |
| ③ 标注 + 门控（探针/歧义度/几何掩码） | 1.1 自建诊断工具族 | 社区自建无标准；我们的接口可复用 |
| ⑥ 标签构建 + 消费契约（非等时/PCHIP） | 1.4 尾部处理需求 + 1.5 抖动 | 社区在部署侧打补丁，我们主张契约先行 |
| ISSUE-002 相机/观测通道 | 1.3 相机族 | 社区只到"工具层面"（check-cameras），无信息论诊断 |
| 次优解/关键帧交互 | 无先例（1.2/1.6 只是"问号"不是系统方法） | GAP-03 维持 |

**总体结论**：社区共性问题与我们框架的对应关系良好——尤其**静默失效（1.2）、诊断工具缺失（1.1）、部署抖动（1.5）**三点，是对我们的 ②⑩ 验收模块、③ 门控、⑥ 契约设计最有力的外部支持。社区没有任何"数据预处理系统方法论"层面的讨论，全部停在工具/补丁层——GAP-11 的社区维度确认（即使放宽到 community，也无人系统化处理数据预处理管线）。

## 3 检索记录

- 关键词：data quality / dataset cleaning / missing frames / camera timestamp / episode filtering / resample keyframe / train-test split / augmentation crop / gripper suction / validation split。
- 检索方式：GitHub issues search API（匿名，限 10 次/分）；部分结果被 issue 正文关键词命中，已人工筛除无关项（如纯硬件问题）。
- 局限：只覆盖 8 个主流 repo 的中英 issues；更多小 repo/论坛（Discord/HF 论坛）未覆盖；`drift` 类修复大多作为 issue 存在，无法统计实际修复率。