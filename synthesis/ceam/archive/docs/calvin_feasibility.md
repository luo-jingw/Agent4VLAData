# CALVIN 可行性评估（Phase 16）

只调研，没有下载任何数据集、没有安装任何东西、没有改动项目环境、代码或配置。调研时间：2026-09-28 美东 21:30–21:50。依据：`docs/framework.md`。

- 查询方式：
  - 对数据 zip 发 HTTP HEAD；
  - 用 HTTP range 只读取各 zip 末尾的中央目录（文件清单与每个文件的大小，约 0.6 GB 元数据；不读取任何帧，只把汇总写到磁盘）；
  - GitHub 源码，论文 PDF，排行榜页面。
- 草稿与汇总在 `tmp/p16/`（不入库）：`cd_task_*.txt` 为各 zip 的目录汇总，`zip_eocd.py` 与 `cd_summary.py` 为查询脚本（只用 `/usr/bin/python3` 标准库，未用项目 Python）。
- 标注：外部数字以 [Sx] 注明来源（URL 见文末）；"（估计）"是推算，"（未核实）"是没有一手核对的内容。

## 结论

- **数据**：ABC→D 训练集 1,795,045 帧，其中 17,870 个带语言标注的窗口；D→D 训练集 512,077 帧，5,124 个窗口。数据量是 LIBERO-Goal 的 10–34 倍，适合"动作表征用全部无标注数据、查询只用标注片段"这一设定。
- **规模**：zip 为 D 177 GB、ABC 555 GB、ABCD 704 GB，debug 1.3 GB。20 GB 以内能拿到的只有 debug 集，以及按 range 单独取的标注文件。
- **环境**：`calvin_env`（PyBullet，EGL 插件）本身不依赖 torch，可以装进一个独立的项目内 uv 环境。主要风险：
  - 代码停更（最后提交在 2022-12），且面向 Python 3.8；
  - `pyhash` 没有 Linux wheel；
  - pip 版 pybullet 可能没开 NumPy；
  - 数据帧与本机 EGL 重渲染是否一致未知。
- **评估**：1000 条、每条 5 个连续指令，每个子任务最多 360 步，指标为平均完成长度。单次完整评估约 0.4–0.6M 环境步（估计），在两张 L40S 上多进程约 0.5–1.5 小时墙钟（估计，需实测）。
- **框架对应**：框架 1–4 都能对应。缺口在于"同一观测下的其他可达动作"：玩耍数据每个观测只有一个动作，需要从记录的状态复位后在仿真中分支，或用近邻观测的块。复位保真度须先测。
- **建议第一步（需用户确认）**：
  1. 下载 debug 集（1.3 GB）；
  2. 按 range 取各拆分的语言标注文件（< 100 MB）；
  3. 建独立的项目内 `.venv-calvin`，在一个 L40S 节点上做环境冒烟。

  判据见第 7 节。之后再决定下载 D（177 GB）还是 ABC（555 GB）。

## 1. CALVIN 概况

| 项 | 内容 | 来源 |
|---|---|---|
| 环境 | 4 个结构相同的桌面环境 A、B、C、D。纹理不同，滑门、抽屉、按钮、开关的位置不同；桌子、机器人、静态相机位置相同 | [S1] §III-A |
| 可操作物体 | 滑门、抽屉、按钮（控制绿色 LED）、开关（控制黄色灯泡），以及红、蓝、粉三个方块 | [S1] |
| 机器人 | 7 自由度 Franka Emika Panda，平行夹爪；物理引擎 PyBullet | [S1] |
| 相机 | 静态 RGB 200×200×3 与深度 200×200；夹爪 RGB 84×84×3 与深度 84×84；另有触觉 | [S1] Fig. 2，[S2]，[S3] |
| 触觉形状 | 论文写 120×160×2；数据 README 写 `rgb_tactile` (160,120,6)、`depth_tactile` (160,120,2)。我们不用触觉 | [S1]，[S3] |
| 本体 `robot_obs`（15 维） | 末端位置 3、欧拉角 3、夹爪开度 1、关节角 7、夹爪指令 1 | [S3] |
| 场景状态 `scene_obs`（24 维） | 滑门、抽屉、按钮、开关的关节 4 维，两盏灯 2 维，三个方块位姿各 6 维。基准规定不作为策略输入 | [S3]，[S1] 脚注 1 |
| 动作空间 | 数据含两种 7 维动作：`actions` 为绝对末端位姿（世界系位置 3、欧拉角 3、夹爪 ±1）；`rel_actions` 为相对末端位移，裁剪到 [−1, 1]，位置乘 0.02 m、姿态乘 0.05 rad（数据 README 写作缩放因子 50 与 20）。环境另支持关节空间动作。夹爪 −1 为闭合，1 为张开 | [S3]；`robot.py` `max_rel_pos=0.02`、`max_rel_orn=0.05` [S13] |
| 相对动作的参考系 | 论文 Fig. 2 写"相对夹爪系"。数据 README 与代码（`relative_to_absolute`：`target_pos += rel_pos`）是世界系，且累加在目标位姿上（`use_target_pose: true`）。以代码为准 | [S1]，[S3]，[S13] |
| 控制频率 | 30 Hz；PyBullet 240 Hz，每个动作 8 个物理子步 | [S1]，`play_table_env.yaml` [S13] |
| 玩耍数据 | 3 名未受训的操作者用 VR 遥操作，每个环境约 6 小时，共约 24 小时、约 2.4M 步 | [S1] §III-B |
| 语言标注比例 | 论文称只标注 1%。实现上，自动标注器按 `eps: 0.01` × 数据集窗口数 / 任务数，为每个任务抽取样本，并用仿真状态判定任务是否完成（`lang_ann.yaml`、`automatic_lang_annotator_mp.py`）。所以"1%"是标注窗口数约占窗口（≈ 帧）数的 1%：ABCD 为 22,966 / 2,307,126 = 1.0%。每个窗口长 32–64 帧（按代码：窗口 64，扣掉 16–32 步前置），窗口可能重叠，被某个窗口覆盖的帧比例未核实 | [S1]，[S11]，[S5] |
| 任务 | 34 个子任务，每个由场景状态的变化判定完成。排行榜 MTLC 一栏写 "32 tasks"，与论文不一致（未核实原因） | [S1] Fig. 6，[S8] `task_categories`，[S6] |
| 指令 | 训练用 389 条众包指令（34 个任务，每个 7–20 条）；测试用另外 34 条（每任务 1 条），训练中未出现。论文称"超过 400 条" | [S10]（计数自 yaml），[S1] |
| 预计算语言嵌入 | 默认 MiniLM 384 维；另附 9 种（mpnet、CLIP 等）。我们用自己的冻结 SigLIP 文本编码，不需要它们 | [S3] |

## 2. 数据

### 各拆分

| 拆分 | zip 大小（HEAD） | 解压后 | 训练帧 | 验证帧 | 训练 / 验证标注窗口 | 下载时间（单连接实测约 11 MB/s） |
|---|---|---|---|---|---|---|
| debug（`calvin_debug_dataset`） | 1,299,150,917 B = 1.30 GB（1.21 GiB） | 1.31 GB | 2,771（1 段连续录制） | 1,675（1 段） | 很少（文件 15 KB；未计数） | 约 2 分钟 |
| D→D（`task_D_D`） | 177,379,436,142 B = 177.4 GB（165.2 GiB） | 178.3 GB（166.0 GiB） | 512,077（31 段） | 99,022（4 段，环境 D） | 5,124 / 1,011 | 约 4.5 小时 |
| ABC→D（`task_ABC_D`） | 555,309,812,705 B = 555.3 GB（517.2 GiB） | 558.1 GB（519.8 GiB） | 1,795,045（147 段） | 99,022（环境 D） | 17,870 / 1,087 | 约 14 小时 |
| ABCD→D（`task_ABCD_D`） | 704,022,347,117 B = 704.0 GB（655.7 GiB） | 707.6 GB（659.0 GiB） | 2,307,126（178 段） | 99,022（环境 D） | 22,966 / 1,087 | 约 18 小时 |

来源与推算方法 [S5]：

- **大小**：大小由 HEAD 的 Content-Length 得到。README 写的 166 / 517 / 656 GB [S3] 实际是 GiB。
- **帧数**：帧数是中央目录里 `training/`、`validation/` 下 `episode_*.npz` 的个数。
- **段数**：段数由 `ep_start_end_ids.npy` 的字节数推出（(大小 − 128) / 16）。
- **标注窗口数**：由同一目录下两种嵌入文件的大小差推出。mpnet 768 维与 MiniLM 384 维 float32 每条相差 1,536 字节，四个结果都是整数，但没有打开文件核对。
- **各拆分是否相同**：D 与 ABC/ABCD 的验证集帧数相同，但标注窗口数不同（1,011 vs 1,087）；原因未核实，可能与 2022-09 的标注修正有关 [S2]。
- **单帧大小**：单帧 npz 约 0.29 MB，含所有模态（RGB、深度、触觉、本体、场景状态、两种动作）。

**数据文件格式**：

- 每帧一个 `episode_XXXXXXX.npz`；
- `ep_start_end_ids.npy` 给出连续录制段；
- `lang_annotations/auto_lang_ann.npy` 为 pickle 字典：文本、任务 id、嵌入、窗口 (start, end)；
- `.hydra/merged_config.yaml` 是构建环境所需的配置 [S3]。

**修正补丁**：changelog 记有 2022-09 的 ABC/ABCD 语言标注与 `scene_info.npy` 修正、2023-02 的 D `scene_info.npy` 修正 [S2]。zip 的 Last-Modified 分别是 2022-09-15（ABC/ABCD）与 2023-02-23（D），应已包含修正（未核实）。下载后用 `sha256sum.txt` [S5] 校验。

### 20 GB 以内现在可取的

| 内容 | 大小 | 用途 |
|---|---|---|
| debug 集（完整 zip） | 1.30 GB | 读数据格式；用其 `.hydra/merged_config.yaml` 构建环境（场景可指定为 `calvin_scene_D`）；重渲染对照；复位保真度；计时 |
| 各拆分的 `auto_lang_ann.npy`，用 range 单独取 | 压缩后 D 训练 6.6 MB、ABC 23 MB、ABCD 30 MB，加验证集各约 1.5 MB | 窗口长度、重叠、任务分布、同一帧被多个任务窗口覆盖的比例（第 5 节"区分项"要用） |
| 环境 D 的部分验证帧，用 range 按文件取 | 按需（全部 99,022 帧为 26.6 GiB，超过 20 GB） | 在真实 D 帧上复查重渲染与编码器 |

服务器支持 range 请求：读取中央目录时返回 206。按文件取子集技术上可行，但需要自己解析 zip 本地头（未实现）。

### 对共享盘的影响

当前 `df`：总 27.72 TiB，已用 23.31 TiB，可用 3.15 TiB，显示 89%。

下表中，特征缓存按我们现在的格式计算：每帧两视角 × 64 token × 768 × fp16 = 196,608 B，与 `demo_renders` 相同（`docs/demo_renders.md`：52,895 帧 9.77 GiB）。

| 方案 | 新增 | 之后可用 | df 使用率（约） |
|---|---|---|---|
| debug | 0.001 TiB | 3.15 TiB | 89% |
| D：流式解压（不留 zip）+ 8×8 特征 | 0.162 + 0.109 = 0.27 TiB | 2.88 TiB | 90% |
| D：zip 与解压并存 + 8×8 特征 | 0.43 TiB | 2.72 TiB | 90% |
| ABC：流式解压 + 8×8 特征 | 0.508 + 0.339 = 0.85 TiB | 2.30 TiB | 92% |
| ABC：zip 与解压并存 + 8×8 特征 | 1.35 TiB | 1.80 TiB | 94% |
| ABC：流式解压 + 4×4 特征（每帧 48 KB） | 0.508 + 0.085 = 0.59 TiB | 2.56 TiB | 91% |
| ABCD：流式解压 + 8×8 特征 | 0.644 + 0.430 = 1.07 TiB | 2.08 TiB | 93% |

- 缩小占用的做法：
  - 解压时只保留 `rgb_static`、`rgb_gripper`、`robot_obs`、`scene_obs`（只作诊断标签）、`rel_actions`、`actions`，其原始字节约 0.14 MB/帧；
  - 或编码完成后删除原始帧；
  - 或特征用 4×4 池化，或按时间隔帧。
- 约 240 万个小文件放在同一个共享盘上，对 NFS 的元数据压力大（未测）。应一次性打包成我们的分片格式，训练时不直接读 npz。

## 3. 环境安装

### 依赖

| 组件 | 版本约束 | 与我们环境（CPython 3.10.20，torch 2.7.1，numpy 2.2.6）的关系 | 来源 |
|---|---|---|---|
| `calvin_env` | 纯 Python，面向 py3.8（black `target-version py38`）。`calvin` 仓库固定的子模块提交为 `1431a46`，main 最新为 `797142c`（2022-12-23），其后多 7 个提交，包括"按钮在 rollout 中的 bug""step 后动作被修改的 bug"两项修复 | 需决定用哪个提交：固定提交与排行榜可比，最新提交修了 bug | [S13]，GitHub compare API |
| `pybullet` | 3.2.7，有 cp310 manylinux wheel（103 MB） | 可装；pip 版可能未开 NumPy（`isNumpyEnabled() == 0`，bullet3 #4523），`getCameraImage` 会返回元组，转换较慢 | [S14]，[S16] |
| `numpy-quaternion` | 2024.0.13，cp310 wheel，numpy ≥ 1.25、< 3 | 兼容 numpy 2 | [S14] |
| `gym`（旧版，非 gymnasium） | 未固定 | 我们的锁文件只有 gymnasium 1.3.0。gym 0.26 与 numpy 2 的兼容性未核实；`calvin_env` 只用 `gym.Env` 基类与 `gym.utils.seeding` | [S13] |
| `hydra-core`、`omegaconf`、`gitpython`、`opencv-python`、`scipy`、`numba`、`pandas`、`matplotlib`、`cloudpickle` | `calvin_env` 未固定版本（`calvin_models` 固定 hydra-core 1.1.1） | 我们的 `.venv` 没有 hydra/omegaconf/gitpython/pandas。`play_table_env.py` 构造时调用 `git.Repo(search_parent_directories=True)`，需要位于某个 git 仓库内 | [S12]，[S13] |
| `tacto`、`pyrender`、PyOpenGL 分支（git 安装） | 仅触觉相机需要 | 构建环境时去掉触觉相机（`get_env(..., obs_space=...)` 支持）就不会 import，不必安装 | [S13]，[S15] |
| `calvin_models`（官方训练与评估代码） | torch==1.13.1、pytorch-lightning==1.8.6、hydra-core==1.1.1、setuptools==57.5.0、`pyhash`、`MulticoreTSNE` | 与 torch 2.7.1 冲突，不安装。评估只需要其中三个文件的逻辑（`evaluate_policy.py`、`multistep_sequences.py`、`utils.py`），可按原样移植 | [S12] |
| `pyhash` 0.9.3 | 只有 macOS wheel 与 sdist，构建要 setuptools < 58（README 也提到） | 评估用它的 `fnv1_32` 为 1000 条序列的初始状态做确定性打乱。FNV-1 32 位可以用十几行纯 Python 实现，须与原实现逐位核对 | [S9]，[S14]，[S2] |

### EGL 与无头渲染

- **插件加载**：`use_egl: true` 时，`calvin_env` 以 `p.connect(p.DIRECT)` 加 `loadPlugin("eglRendererPlugin")` 在 GPU 上渲染 [S13]。
- **每进程一个环境**：它用的是全局 `pybullet` 客户端，所以一个进程只能有一个 EGL 环境；并行评估须多进程，与我们的 LIBERO 做法相同。
- **设备选择**：由 `EGL_VISIBLE_DEVICES` 决定，它是 EGL 设备列表中的序号 [S2] FAQ。
  - 官方 `get_egl_device_id` 会在包目录里调用 `build.sh`，编译一个 C++ 程序 `EGL_options.o` [S13]。
  - 我们不走这条路。我们已有 EGL 序号与 CUDA 编号的对应，node-71 与 node-61 的两张表见 `docs/libero_sim.md`（`ceam/sim/robosuite_runtime.py`），可以直接设 `EGL_VISIBLE_DEVICES`。
  - pybullet 插件是否与 MuJoCo 看到同一个设备列表，未核实，冒烟时用 `nvidia-smi` 确认进程落在目标卡上。
- **纹理差异**：FAQ 说 CPU 渲染与 GPU 渲染的纹理"略有不同"[S2]。数据帧由 `datarenderer.py` 从录制状态渲染而来，环境配置默认用 EGL，但数据帧与我们在 L40S 上的 EGL 重渲染是否逐像素接近，未核实，冒烟时要测（第 7 节）。
- **节点**：只用 L40S 节点（node-60–69，不用 node-64 的 GPU 3），每节点 2 张卡；不用 node-70、node-71（`PROJECT.md`）。

### 独立的项目内 uv 环境

- **为什么独立**：`calvin_env` 的依赖（旧 gym、hydra、gitpython 等）不应进入现有 `.venv`，否则可能改变 LIBERO 的环境（按约束不改现有环境）。
- **做法**：
  - 在项目内另建 uv 项目，例如 `envs/calvin/pyproject.toml` 与它自己的 `uv.lock`，同步到 `.venv-calvin`；
  - 解释器用同一个 uv 管理的 CPython 3.10.20；
  - 其中也装 torch 2.7.1、transformers 4.53.3 等我们闭环所需的包。uv 缓存在项目内 `.uv/cache`，同盘可硬链接，额外占用应较小（未测；现有 `.venv` 为 7.1 GB）。
- **不装的**：`calvin_models`，以及 `tacto`、`pyhash`。
- **路径与缓存**：`calvin_env` 的源码放在项目内并固定提交，满足它的 git 检查。hydra 的输出目录、`/tmp` 与 `~/.cache` 的写入要在冒烟时逐项确认，全部重定向到项目 `tmp/`。

### 风险

1. **停更**：`calvin_env` 自 2022-12 起不再维护，且面向 py3.8。与 numpy 2、新版 hydra 的组合可能要打小补丁；补丁应在项目内以补丁文件记录。
2. **评估逻辑的复现**：评估代码依赖 `pyhash` 与 `calvin_models`，我们要移植评估逻辑。须核对 1000 条序列及其初始状态与官方完全相同，否则结果与文献不可比。
3. **渲染速度**：pip 版 pybullet 若未开 NumPy，渲染转换会变慢。
4. **帧域**：数据帧与我们重渲染帧之间可能有纹理差异（项目规则要求用仿真渲染帧训练）。
5. **复位保真度**：`env.reset(robot_obs, scene_obs)` 只恢复关节与物体位姿，不恢复速度与接触状态。从记录状态复位后执行数据动作，与记录的偏离有多大未知（LIBERO 上同类问题见 `docs/libero_sim.md`）。

## 4. 评估协议

| 项 | 内容 | 来源 |
|---|---|---|
| 长程评估（LH-MTLC） | 1000 条指令链，每条 5 个连续子任务。子任务由仿真状态判定完成后，才切换到下一条指令；某个子任务失败，整条链结束 | [S1] §III-C，[S7] |
| 序列生成 | 从合法初始条件（灯、滑门、抽屉、三个方块的位置组合）出发，枚举可行的 5 任务序列，过滤循环、冗余与相似，得到 1000 条；numpy 种子固定为 0，与进程数无关 | [S1]，[S8] |
| 初始状态 | 每条链开始时机器人复位到中性位姿，物体按初始条件放置；子任务之间不复位 | [S1]，[S9] `get_env_state_for_initial_condition` |
| 每个子任务的步数上限 | 360 步（`EP_LEN`，12 秒）；`NUM_SEQUENCES = 1000` | [S7] |
| 指令 | 每个子任务用验证 yaml 中的那 1 条指令（训练中未出现） | [S7]，[S10] |
| 指标 | 连续完成 1–5 个子任务的比例，以及平均完成长度 = 1000 条链完成子任务数的平均（0–5） | [S7]，[S9] |
| 跨环境 | ABC→D：只在 A、B、C 训练，在未见过的 D 上评估，D 的纹理与滑门、抽屉、按钮、开关的位置都不同。D→D：同环境。ABCD→D：多环境 | [S1] |
| 单任务评估（MTLC） | 每个任务从合法状态出发做 10 次；官方代码只对其基线实现 | [S1]，[S2] |
| 参考数值 | ABC→D 平均长度：官方 MCIL 基线（静态 + 夹爪 RGB）0.31，HULC 0.67，RoboFlamingo 2.47，GR-1 3.06，3D Diffuser Actor 3.27，MoDE 4.01，Seer-Large 4.28，FLOWER 4.53。D→D：MCIL 0.41–0.64，HULC 2.64，MDT 3.72，FLOWER 4.35 | [S6] |

**我们的接入方式**：

- **接口**：官方接口是 `CustomModel.step(obs, goal)`，每步返回一个动作 [S7]。我们在内部缓冲 10 步块，每执行完一块重新观测，与 LIBERO 相同。
- **配对**：1000 条链是确定的，不同选择方式按链的编号配对；检验沿用 `docs/no_psi_baseline.md`，至少 3 个种子。

**仿真时间（估计，需在冒烟中实测）**：

- **步数上界**：1000 × 5 × 360 = 1.8M 步。
- **实际步数**：失败的子任务耗满 360 步，之后整条链结束。假设成功的子任务平均约 100 步（未核实），平均长度在 0.5–4.5 之间时，一次完整评估约 0.4–0.6M 步。
- **单步开销**：
  - 仿真（8 个子步、IK、200² 与 84² 两次渲染）估计 5–15 ms；
  - 我们的决策约每 10 步一次，每次 30–60 ms（LIBERO 上带查询网络的决策为 60–69 ms、10 进程共卡，`docs/query_match.md`），折合每步 3–6 ms。
- **单次评估**：约 1–3.5 进程·小时。两张 L40S 上跑 20 个进程（每进程 2 线程）约 0.5–1.5 小时墙钟，可能受 CPU 限制。
- **对照组合**：3 个种子 × {选 match 最高者，同批随机选，π 均值块} = 9 次评估，约 10–30 进程·小时。

## 5. 与框架的对应（`docs/framework.md`）

| 框架条目 | CALVIN 上的实现 | 说明 |
|---|---|---|
| 1 动作表征 r(o, a)，与任务无关 | 用全部无标注玩耍数据的转移 (o_t, a_{t:t+10}, o_{t+10}) 训练：ABC 约 1.8M 帧，D 约 0.5M 帧，不读语言与成败 | 可直接用现有候选实现：冻结 DINOv2 特征的固定 PCA 加训练的效果头（R6），或 ψ 与效果头联合训练。"以下一观测表征之差为回归目标"是历史实现的工程选择，不是框架要求。连续录制段按时间切出 HELDOUT 块 |
| 2 动作表征侧不坍缩（同一观测下不同动作可区分） | 玩耍数据中每个观测只有一个动作。检验与监督都需要同一观测下的多个动作，来源有两个：(a) 从记录帧复位（`reset(robot_obs, scene_obs)`）后在仿真中执行多个候选块，得到真实结果，与 LIBERO 数据 A 的做法相同；(b) 冻结特征近邻观测上的其他块，只是近似 | (a) 依赖复位保真度（第 3 节风险 5）。G1 式判据（状态内方差占比、有效秩、两两余弦）可在 (a) 上原样计算 |
| 1 查询 q(o, l) | 只用带标注的窗口：窗口内满 10 步块的帧 o，配指令 l，窗口中执行的块为接受项 a⁺。窗口数 ABC 17,870、D 5,124；按窗口 32–64 帧推算，(o, l, a⁺) 约 0.5–1.0M 个（估计） | 比 LIBERO-Goal 的 43,701 个 TRAIN 观测多一个数量级 |
| 2 查询侧不坍缩（作用在任务编码上） | 每步对 34 个任务各取一条训练指令，对 e(l) 做 SIGReg | LIBERO 只有 10 个任务 × 5 种说法；CALVIN 有 34 个任务、389 条说法，任务间共享物体与动作（同一方块的抬起、推、旋转），更接近框架假定的任务间共享结构。测试指令未在训练中出现，e(l) 须能泛化到新说法 |
| 3 match 在同一观测内比较：接受与任务一致的，区分其他可达的 | 接受项：该观测上标注窗口的块。区分项（都在同一 o 上）：(i) 采样器候选（π 带噪块、光滑随机块、下面的检索块）；(ii) 冻结特征近邻观测上其他任务标注窗口的块（沿用 MVP 做法）；(iii) 近邻观测上的无标注玩耍块 | (iii) 有假负样本风险：标注只是抽样，无标注的块可能也完成了同一任务。只放进采样器，不作区分项，或只在用户确认后使用。同一帧可能同时落在两个不同任务的窗口里（如先抬起再放入），这时该块对两个任务都是接受项，不得作为其中任何一个的区分项。须先用标注文件统计这种重叠 |
| 4 采样器（可替换）+ 选 match 最高者 + 重新观测 | 三种候选来源：(a) 无 ψ π 的 CALVIN 版，在标注窗口上做行为克隆（冻结 DINOv2 token、本体、SigLIP 指令），均值块 + 带噪块，对应现有基线；(b) 玩耍检索采样器：取冻结特征最近的 K 个玩耍帧，用它们接下来执行的块做候选，不用语言，全部语言信息由 match 提供，可复用 `--retrieval-space frozen` 的检索代码；(c) 光滑随机块（`SmoothActionSampler`，按 `rel_actions` 的 [−1, 1] 范围） | (b) 直接对应"玩耍数据中的其他可达动作"，是 CALVIN 上最能单独检验 match 的采样器。(c) 在长程链上的基础成功率很可能接近 0，会遇到 G2 的功效前提问题 |
| 最终标准 | 长程评估的平均完成长度与第 1 个子任务的成功率，与 π 均值块按链配对，至少 3 个种子；分支检验：从记录状态复位，逐个执行候选块，再用任务判定器得到仿真成败标签，含随机候选对照 | 任务判定器 `Tasks.get_task_info` 由场景状态判定，可直接提供分支检验的成败标签 |

**框架外**（不用；如需使用，须用户确认）：

- 用任务判定器给全部无标注玩耍窗口自动打语言标签。这会改变"查询只用少量标注片段"的前提，判定器只用于评估与诊断。
- 事后目标图像重标注（MCIL、HULC 等基线使用），属于终点或目标类的监督。
- 把 `scene_obs` 作为模型输入。基准本身也不允许，只作诊断标签，与 LIBERO 的 `SceneState` 相同。
- 停留选项、升级或卡住规则。
- 深度与触觉：不在框架内，也不需要。
- flow matching 可以作为采样器 (b) 的可选工程实现，不是框架要求。

## 6. 与当前 LIBERO 流程的差异

| 方面 | LIBERO-Goal（现状） | CALVIN | 复用 / 新写 |
|---|---|---|---|
| 数据格式 | LeRobot v2.1：parquet 加两路 512² mp4，20 fps；433 个短示范，每个都带任务 | 每帧一个 npz，约 240 万个文件；连续录制段；标注是 (start, end, 任务, 文本) 窗口 | 新写：读取器与打包成 `TransitionStore` 分片的程序，另存 `robot_obs`、`scene_obs` 旁车文件 |
| 训练帧 | 数据视频帧与仿真不一致，所以在仿真中重放全部示范并渲染（`docs/demo_renders.md`） | 数据帧本身就是 `calvin_env` 的仿真渲染。若冒烟证明与本机 EGL 重渲染一致，就不必重放；否则需要重渲染约 0.5–1.8M 帧 | 视冒烟结果而定 |
| 渲染 | MuJoCo 3.3.2 与 robosuite，512²，EGL；图像需旋转 180° | PyBullet EGL 插件，静态 200²、夹爪 84²；方向未核实 | 新写 `CalvinTaskEnv`：复位、快照、按块执行、EGL 设备映射；复用 EGL 与 CUDA 编号对应的逻辑 |
| 冻结编码器 | DINOv2-base 以 224² 输入，8×8 池化 token；SigLIP 文本 | 静态视角 200→224，接近原生分辨率。夹爪视角 84→224，放大 2.7 倍，每个 patch 约对应原图 5 像素，信息少，DINOv2 在其上的表现未核实。SigLIP 编码 389 + 34 条指令，开销可以忽略 | 复用 `DinoV2VisionEncoder`、`SiglipTextEncoder` |
| 本体 | 9 维：关节 7 + 两指位置 2 | 15 维 `robot_obs`。可取关节 7 + 夹爪开度 1 + 夹爪指令 1，也是 9 维，但语义不同 | 代码中的本体维度要参数化，并核查写死 9 的地方 |
| 动作块与控制 | 20 Hz；10 步块 = 0.5 s；7 维末端增量；夹爪 {0, 1} | 30 Hz；10 步块 = 0.33 s，15 步块 = 0.5 s；`rel_actions` ∈ [−1, 1]，累加在目标位姿上；夹爪 ±1 | 块长需决定；`ActionCache` 的标准化、夹爪映射、`SmoothActionSampler` 的范围参数化 |
| 评估 | 10 任务 × 50 个 benchmark 初始状态；成功率 | 1000 条 5 指令链；平均完成长度；按链配对 | 新写：移植官方序列生成、初始状态与任务判定，按链分片到多进程；复用配对检验与种子检验脚本 |
| 基线 | 无 ψ π（均值块），0.927 ± 0.012 | 同构的无 ψ π，在标注窗口上训练。玩耍数据只有约 1% 带标注，预计远低于排行榜（官方 MCIL 基线 ABC→D 为 0.31），比较只在我们自己的配对设定内进行 | 复用 `ProposalHead` 及其训练器（新的 data 配置） |
| r、q、match | R6 与效果头；`TaskEncoder`、`QueryNetwork`、`Match`、同一观测候选、训练器 | 同上 | 复用模型与损失。新写：查询表格（标注窗口、窗口重叠的排除规则、近邻观测）与候选来源 (b) |

## 7. 工作量与算力

### 阶段（开发时间为估计）

| 阶段 | 内容 | 开发 | 算力 |
|---|---|---|---|
| C0 冒烟（需确认） | 下载 debug 集；按 range 取标注文件；建 `.venv-calvin`；做下面的冒烟检查 | 0.5–1 天 | < 1 GPU·h |
| C1 数据 | 下载 D（约 4.5 h）或 ABC（约 14 h）；流式解压，只留所需字段；打包；编码特征 | 1 天 | D 0.6–1.1 GPU·h，ABC 1.8–3.5 GPU·h（估计：两视角每卡约 150–300 帧/秒，据 `docs/encoders.md` 的批量延迟推算）；另需 NFS 读取原始帧 D 约 0.16 TiB、ABC 约 0.5 TiB |
| C2 评估与基线 | 评估框架（1000 链，分片，逐位复现官方序列与初始状态）；CALVIN 版无 ψ π，3 个种子；π 均值块评估 | 1–1.5 天 | 训练 3–6 GPU·h；评估 3 次 × 1–3.5 进程·h |
| C3 r | 在全部玩耍转移上训练 R6 式效果头；在复位分支上采集少量反事实，例如数千个状态 × 16–32 个块，做 G1 式可区分程度检查 | 1 天 | 训练 2–4 GPU·h；采集以 CPU 为主，约数十进程·h（估计） |
| C4 q 与 match | 查询表格；训练 3 个种子；闭环：{π、检索} 采样器 × {选 match 最高者，同批随机选}；分支检验 | 1–1.5 天 | 训练 5–10 GPU·h；评估 12 次 × 1–3.5 进程·h |
| 合计 | | 约 4.5–6 天 | 约 25–50 GPU·h（L40S），外加 50–100 CPU 进程·h 的仿真。每节点 2 卡；可在 2–3 个 L40S 节点上并行 |

### 主要风险

1. **磁盘**：ABC 加 8×8 特征约占 0.85 TiB，可用空间从 3.15 TiB 降到约 2.3 TiB；若 zip 与解压并存，只剩约 1.8 TiB。共享盘上其他用户也在增长。建议流式解压、只留必要字段，或用 4×4 特征。
2. **I/O**：数据在 node-71 本地盘上，计算节点经 `/shared` NFS 读取约 240 万个小文件，吞吐与元数据开销未测。应一次性打包。
3. **环境代码老旧**：见第 3 节的风险 1–5。其中评估逻辑能否逐位复现官方序列，决定了结果能否与文献对照。
4. **帧域与复位保真度**：决定能否直接用数据帧训练，以及复位分支（框架 2 的检验、分支检验）是否可信。
5. **框架层面**：同一观测下的其他可达动作只能靠复位分支或近邻近似；无标注块可能是假负样本；不同任务的窗口会重叠；测试指令是新说法。
6. **统计功效**：若 π 在 ABC→D 上的第 1 个子任务成功率很低，配对比较的功效不足，情形与 G2 的功效前提相同。因此先做 D→D，或在 C2 用小规模试运行估计基础成功率，再定评估设定。
7. **与排行榜不可比**：我们的 π 只用标注窗口、冻结编码器，结论只在配对设定内成立，不与排行榜数值比较。

### 建议第一步（需用户确认）

C0 冒烟，数据总量 < 2 GB。在一个 L40S 节点（例如 node-62）上用 1 张卡运行，全部写入项目目录。

1. 下载 debug 集（1.30 GB），校验 sha256；按 range 取 D 与 ABC 的 `auto_lang_ann.npy`（< 100 MB）。
2. 新建 `envs/calvin/` 独立 uv 项目，生成 `.venv-calvin`：
   - CPython 3.10.20；
   - `calvin_env` 固定在 `1431a46` 或 `797142c`（请用户决定：前者与排行榜一致，后者包含 bug 修复）；
   - pybullet 3.2.7；
   - 不装 `tacto`、`pyhash`、`calvin_models`；
   - 不改动现有 `.venv`、`pyproject.toml`、`uv.lock`。
3. 冒烟判据（事先固定，逐项报告）：
   - EGL 插件在目标卡上加载成功，`nvidia-smi` 确认进程所在的卡；`pybullet.isNumpyEnabled()` 的值；`/tmp`、`~/.cache` 与 hydra 输出目录没有项目外写入。
   - 用 debug 的配置构建环境，场景指定为 D，并去掉触觉相机；逐位复现官方 1000 条序列与初始状态。纯 Python `fnv1_32` 用标准测试向量核对。
   - 重渲染与数据帧对照：在 debug 帧上用 `reset(robot_obs, scene_obs)` 复位后渲染，与数据中的 `rgb_static`、`rgb_gripper` 比较，报告 MSE、图像方向与 DINOv2 token 余弦（参照 `docs/libero_sim.md`）。
   - 复位保真度：复位后执行数据中的 10 步与 64 步 `rel_actions`，报告 `robot_obs` 与 `scene_obs` 相对记录的偏离。
   - 任务判定器在 debug 的标注窗口上给出的任务与标注一致的比例。
   - 计时：单步（仿真加渲染）的中位与 p90；用随机动作跑 20 条链，外推一次完整评估的进程·小时。
   - 标注文件统计：窗口长度分布，窗口重叠的比例，同一帧属于多个任务的比例，每个任务的窗口数。
4. 用户看过冒烟报告后，决定：
   - 下载 D（177 GB）还是 ABC（555 GB）；
   - 特征粒度，8×8 还是 4×4；
   - 块长，10 步还是 15 步。

按 `plan.md`"下一步安排"，CALVIN 是第三步，排在 LIBERO-Goal 实现问题（Phase 13–15）与 LIBERO-90 迁移之后。C0 只占 1 张卡、不到 1 GPU·h，可以与第一步并行。

## 来源

- [S1] Mees et al., "CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks", RA-L 2022. https://arxiv.org/abs/2112.03227（PDF：https://arxiv.org/pdf/2112.03227）
- [S2] CALVIN README（提交 `fa03f01`，2025-09-08）：https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/README.md
- [S3] 数据 README：https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/dataset/README.md ；下载脚本：https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/dataset/download_data.sh
- [S5] 数据服务器，2026-09-28 美东 21:33–21:43 的 HEAD 与中央目录 range 读取：http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip 、http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip 、http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip 、http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip ；校验和：http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt 。汇总在 `tmp/p16/cd_task_*.txt`
- [S6] 排行榜（2026-09-28 读取）：http://calvin.cs.uni-freiburg.de/
- [S7] `evaluate_policy.py`：https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/calvin_agent/evaluation/evaluate_policy.py
- [S8] `multistep_sequences.py`：https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/calvin_agent/evaluation/multistep_sequences.py
- [S9] 评估 `utils.py`（`pyhash`、`count_success`、初始状态）：https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/calvin_agent/evaluation/utils.py
- [S10] 指令：https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/conf/annotations/new_playtable.yaml 、https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/conf/annotations/new_playtable_validation.yaml
- [S11] 自动标注：https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/conf/lang_ann.yaml 、https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/calvin_agent/utils/automatic_lang_annotator_mp.py
- [S12] 官方训练依赖：https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/requirements.txt 、https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/install.sh
- [S13] `calvin_env`（main `797142c`；`calvin` 固定的子模块 `1431a46`）：https://github.com/mees/calvin_env 。其中 `requirements.txt`、`conf/env/play_table_env.yaml`、`conf/cameras/cameras/{static,gripper}.yaml`、`conf/robot/panda.yaml`、`calvin_env/envs/play_table_env.py`、`calvin_env/robot/robot.py`、`calvin_env/utils/utils.py`；两提交比较：https://github.com/mees/calvin_env/compare/1431a46bd36bde5903fb6345e68b5ccc30def666...797142c588c21e76717268b7b430958dbd13bf48
- [S14] PyPI：https://pypi.org/project/pybullet/3.2.7/ 、https://pypi.org/project/pyhash/0.9.3/ 、https://pypi.org/project/numpy-quaternion/2024.0.13/
- [S15] tacto 依赖：https://github.com/lukashermann/tacto/blob/master/requirements/requirements.txt
- [S16] pip 版 pybullet 未开 NumPy：https://github.com/bulletphysics/bullet3/issues/4523
