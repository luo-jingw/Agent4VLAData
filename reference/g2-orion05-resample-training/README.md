# G2 Orion 0.5 · 曲率重采样训练包

给已经会训 pi0.5、但缺这套方法的 policy / config / 数据流水线的人。
2026-08-25。AgiBot G2 双吸盘,取件 + 放件两个子任务。

- **方法说明**(重采样 / 事件窗等时 / 关键帧加权,含算法与伪代码):
  [`docs/ACTION_CHUNK_RESAMPLING.md`](docs/ACTION_CHUNK_RESAMPLING.md)
- **实验与实现细节**:[`docs/RESAMPLING_REPORT.md`](docs/RESAMPLING_REPORT.md)

本文件只讲**怎么跑**。

---

## 0. 一句话说明这套方法

标准做法是按**时间**等距取动作标签(LeRobot 的 `delta_timestamps`)。这套改成按**曲率**取:
弯的地方点密、直的地方点疏,吸盘瞬态单独按时间等距处理。好处是同样 50 个点覆盖更长的
轨迹、更少浪费在停顿上;代价是**标签不再等时**,推理侧必须知道这件事(见 §6)。

---

## 1. 环境

需要一个能跑 openpi 的环境(和你训 pi0.5 时用的同一个即可)。本包**不修改 openpi 源码树**,
配置在运行时注入。

额外依赖:

| 用途 | 包 |
|---|---|
| 建数据集 | `numpy scipy pandas pyarrow tyro` + 系统 `ffmpeg` / `ffprobe` |
| 分析脚本 | 上面这些 + `matplotlib av` |
| `ruckig_demo.py` | `ruckig`(可选,只有这一个脚本用) |
| `chunk_playback.py` | **只要 numpy**,可直接抄进控制栈 |

---

## 2. 目录

```
progress_resample.py       方法核心:曲率进度坐标 + PCHIP 采样 + 事件窗等时 + 关键帧权重。纯函数无 IO
build_new_dataset.py       清洗 + 重采样 + 建 LeRobot 数据集(写 action_chunk 与 action_loss_weight)
merge_datasets.py          pick + place 合并成双子任务数据集
compute_arc_norm_stats.py  norm stats(绕开 openpi 那条要解码全部视频的慢路径)
provenance.py              溯源:裁掉前导帧后,验收脚本要回到原始录制
verify_dataset.py          数据集验收(6 项)
probe_arc_batch.py         端到端探针:标签 → dataloader → 模型输入
chunk_playback.py          推理侧参考实现(见 §6)
chunk_playback_demo.py     四种播放策略的对比测量
export_hf.py upload_hf.py  导出与上传
joint_limits.py            G2 关节限位表(来自 GDK 知识库)

<分析脚本,同样平铺在包根,见 §7>

openpi_patch/
  config_arc.py            TrainConfig
  agibot_g2_arc_policy.py  输入输出 transform
  register.py              运行时注入 openpi 注册表 + 启动训练
  convert_wrapper.py       JAX → PyTorch 转换

assets_urdf/
  G2_omnipicker_fixed_dual.urdf   正运动学用,分析脚本默认从这里读

docs/
  ACTION_CHUNK_RESAMPLING.md      方法:问题、原理、算法、核心代码
  RESAMPLING_REPORT.md            实验报告
```

> **布局说明(相对 2026-08-20 那版的变化)**:所有 `.py` 现在**一律平铺在包根**,不再有
> `analysis/` 子目录。这些脚本内部用 `ROOT = Path(__file__).parent` 定位数据与输出,还有
> `parent / "openpi_patch"` 的 `sys.path` 插入——放进子目录后这些相对路径全部差一级,旧包里
> `analysis/` 下的脚本因此实际跑不起来。平铺就是它们本来的运行方式。

---

## 3. ⚠️ 路径假设(先读这节)

**本包所有脚本都假设"数据在自己旁边"。** 它们是在一个扁平工作目录里长出来的,没有做过
安装/打包处理。具体来说:

1. **脚本靠同目录 import。** `build_new_dataset.py` 里是 `import progress_resample as ar`,
   靠的是两者同级。所以:

   ```bash
   cd <本包根目录>          # 必须,别从别处调用
   python build_new_dataset.py ...
   ```

2. **分析脚本同样从包根调用**,不要移进子目录——它们用 `Path(__file__).parent` 定位
   `assets_urdf/`、`openpi_patch/` 和输出目录。

   ```bash
   cd <本包根目录> && python eef_error.py ...
   ```

3. **默认参数里的数据路径是我们机器上的布局**,例如
   `--src data/new/suction-bin-picking`、`--dataset-dir data/pick-arc-v4`。
   **这些默认值对你几乎肯定是错的,每个都要显式传。** 下面 §4 的命令里我都写全了。
   (`--urdf` 是例外,默认已指向包内的 `assets_urdf/`,不必传。)

4. **`openpi_patch/` 下的脚本假设它上一级是包根**(`ROOT = 父目录`),用来放
   `checkpoints/`、`assets/`、`wandb/`。保持这个相对结构就行。

5. **`eef_error.py` 及依赖它的脚本需要 G2 URDF。** 用的是 AgiBot 公开的
   `G2_omnipicker_fixed_dual.urdf`,手臂链与吸盘版一致,只有末端工具变换不同。
   **本包已随附**(`assets_urdf/`),各脚本的 `--urdf` 默认值直接指向它,不必再传。

---

## 4. 端到端复现

### 4.1 准备原始录制

LeRobot v2.1 格式,每个子任务一个目录:

```
<你的路径>/suction-bin-picking/     meta/{info,episodes,episodes_stats,tasks}.jsonl + data/ + videos/
<你的路径>/workpiece-placement/
```

`state` 28 维、`actions` 24 维(前 22 是关节,22/23 是左右吸盘二值命令),30 fps,三路相机
(`image` / `left_wrist_image` / `right_wrist_image`)。

### 4.2 建两个单任务数据集

```bash
cd <本包根目录>

python build_new_dataset.py \
    --src  <你的路径>/suction-bin-picking \
    --dst  data/pick-arc-v4 \
    --task-kind pick

python build_new_dataset.py \
    --src  <你的路径>/workpiece-placement \
    --dst  data/place-arc-v4 \
    --task-kind place
```

`--task-kind` 决定清洗规则,两个集合的"失败"定义是相反的:
**pick** 是"抓住且保持到最后一帧",没抓住的丢掉;**place** 是"放开",从没放开的丢掉。
之后再按时长上尾分位数(默认 p95)在保留集合内裁掉异常慢的。

**视频会被重编码成 `-g 2`(每两帧一个关键帧)。** 别用 `--video-mode link` 图快:LeRobot 随机读帧,
继承源数据约 190 帧的 GOP 会让每次读多解近百帧,实测训练从 0.97 s/it 掉到 2.08 s/it。

### 4.3 合并成双子任务

```bash
python merge_datasets.py \
    --sources data/pick-arc-v4 data/place-arc-v4 \
    --dst     data/pickplace-arc-v4 \
    --repo-id world-studio/agibot-g2-pickplace-arc-v4
```

### 4.4 norm stats + 验收

```bash
python compute_arc_norm_stats.py --dataset-dir data/pickplace-arc-v4
python verify_dataset.py         --dataset-dir data/pickplace-arc-v4
```

六项应全 PASS。**不要跳过验收**——数据集"能跑起来"和"是对的"是两回事,
其中一项专门检查 parquet 行数与视频帧数一致,不一致是静默错标而不是崩溃。

### 4.5 训练

```bash
# LeRobot 按 repo_id 找数据集,建个软链
mkdir -p ~/.cache/huggingface/lerobot/world-studio
ln -sfn $(pwd)/data/pickplace-arc-v4 \
        ~/.cache/huggingface/lerobot/world-studio/agibot-g2-pickplace-arc-v4

CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python -u openpi_patch/register.py \
    --dataset-dir data/pickplace-arc-v4 \
    --exp-name run1 --overwrite
```

`register.py` 会自己建那条软链(`--dataset-dir` 指哪它链哪),上面手动建那步其实可以省;
写出来是为了说清 LeRobot 是怎么找到数据集的。

`--dry-run` 只走一遍 dataloader,用来先确认数据接得上。

checkpoint 落在 `checkpoints/g2_pickplace_v1/run1/`。默认 `save_interval=10000`,
只留 10k/20k/30k;每个约 13 GB。

### 4.6 转 PyTorch

```bash
JAX_PLATFORMS=cpu python openpi_patch/convert_wrapper.py \
    --checkpoint-dir checkpoints/g2_pickplace_v1/run1/30000 \
    --output-path   ours_30000_torch
```

转换器要按**训练时那个 config** 重建结构,而它不在 openpi 注册表里,所以走这个 wrapper。

---

## 5. 关键配置

数据(`build_new_dataset.py` 默认值,我们就是用这套训的):

| | | |
|---|---|---|
| `span_frames` | 200 | 一个 chunk 目标覆盖的源帧数 |
| `action_horizon` | 50 | chunk 点数 |
| `metric` | `curvature` | 进度度量,`raw = \|d²q\|^(1/3)` |
| `event_points` | 32 | 一个吸盘事件窗最多给几个点(且不超过窗长) |
| `key_weight` | 3.0 | 事件窗内点的 loss 权重 |
| `duration_quantile` | 0.95 | 时长上尾裁剪 |
| `trim_lead_frames` | 15 | 冻结开头保留几帧,见下 |
| `deadzone` | 0.002 | 实际不起作用(立方根把静止帧抬到阈值六倍以上),保留是为了兼容 |

我们的结果:**271 episodes / 123,898 frames**(原始 287 个,清洗掉 16 个)。

训练(`openpi_patch/config_arc.py`):

```
pi05=True, paligemma_variant="gemma_2b_lora"
action_horizon 50, action_dim 24, state_dim 28
batch_size 32, fsdp_devices 2, num_workers 24
num_train_steps 30_000, ema_decay=None
action_sequence_keys=()      ← 关键:标签走 action_chunk 列,不走 delta_timestamps
prompt_dropout 0.0
```

两条 prompt(逐字):

```
Use the right-hand suction cup to pick up a workpiece from the material bin.
Place the workpiece held by the right-hand suction cup onto the conveyor belt.
```

### 关于 `trim_lead_frames`

每段录制开头有约 3.4 秒机械臂完全不动(录像先开、人后动),占全片 20.7%。
这些帧的 state 冻结、三路相机只有传感器噪声差异,**是同一个训练样本重复一百次**。
重采样本来就不给它们动作点(0.20×),但它们仍以行的形式留在表里,于是每个 batch 有五分之一
是同一个观测。保留最后 15 帧(0.5 秒)足够教会"从 home 起步"。

实现上有一处必须注意:**进度曲线在完整轨迹上计算,裁行发生在之后**。先截断再算,
同样的进度会分给更少的自由帧,`step` 会偏移——那是把轨迹重新计时,而不是去冗余。
已验证裁剪版与未裁版的标签**逐位相同**。

---

## 6. ⚠️ 推理侧:标签不等时

**这是最容易踩的坑,单独说。**

点按曲率排布,所以**相邻动作点在时间上不等距**。实测(8 个 episode):

```
相邻点关节距离   p05 0.0108   p50 0.0685   p95 0.1641 rad   —— 差 15 倍
相邻点真实时长   p05 1.44     p50 2.92     p95 4.95   帧    —— 差 3.4 倍
```

**如果控制器按固定速率一拍一个点,命令速度会在 15 倍范围内摆动**,把速度限幅器顶到饱和 →
落后 → 追赶,表现为强烈顿挫。这不是插值能解决的,缺的是**时间**不是路径。

两件事都要做:

**① 给点分配时间**

```
dt_i = max(|q_{i+1} - q_i| / v_nominal, dt_min)
```

`dt_min` 不是保险系数:吸盘瞬态里点在空间上故意很密,但代表**真实等待时间**(抽真空需要
那些毫秒),纯按距离会一冲而过。用 10 条录制标定过:

| `v_nominal` | `dt_min` | 相对录制时长 |
|---:|---:|---:|
| 0.60 | 0.04 | 1.075× |
| **0.75** | **0.06** | **0.903×** |
| 0.90 | 0.06 | 0.782× |

**② 点之间用 C1 插值**

直线连接会让速度在每个节点上阶跃。用单调三次 Hermite(PCHIP,和训练标签同一族,保形不过冲),
**逐关节位置和速度都连续**。加速度仍不连续,那一层要靠 jerk 限幅器。

`chunk_playback.py` 是这两件事的参考实现,纯 numpy,含滞后帧(连续量,不取整)和
smoothstep 融合。`chunk_playback_demo.py` 复现下面这组对比:

| 策略 | 时长 | 速度 p50 | 峰值 | \|Δv\| p99 | 速度跨度 |
|---|---:|---:|---:|---:|---:|
| 一拍一点 | 5.67 s | 1.888 | 7.518 | 2.64 | 1075× |
| 调度 + 线性插值 | 15.93 s | 0.750 | 5.679 | **4.61** | 750000× |
| 调度 + Hermite | 15.77 s | 0.750 | 1.680 | **0.50** | 6× |
| + 融合 + 限幅 | 15.77 s | 0.756 | 1.404 | 0.49 | 4× |
| (录制本身) | 18.83 s | 0.434 | 4.006 | — | — |

注意第二行:**只加时间调度、不换插值反而更糟**。这条路要走到底,半途会更抖。

> **如果你用的是 `portal-agibot-g2-agent` 那套 runtime**,上面这些已经在分支
> `fix/rtc-c1-reference-interpolation` 里落地了:`sample_reference` 已换成 PCHIP,
> 融合权重已换 smoothstep,并新增可选 `--pace even` 做 chunk 重定时。
> 那种情况下 `chunk_playback.py` 只作参考,不用接。

---

## 7. 分析脚本

支撑 report 里各项结论的实验代码。**不是复现训练必需的**,是给想验证或改方法的人看的。
和主流水线一样平铺在包根,从包根调用。

| | |
|---|---|
| `eef_error.py` | URDF 正运动学,把关节误差换算成吸盘位置的毫米 |
| `eef_geometry.py` | 轨迹几何:停顿 / 无效游走 / 有效位移的三分 |
| `eef_space_budget.py` `eef_speed.py` | 末端空间占用与速度剖面 |
| `keypoint_budget.py` `linear_interp_budget.py` `window_causal_budget.py` | 稀疏点重建实验:多少个点够用 |
| `sliding_window.py` `noise_floor.py` | 滑窗对比、噪声底 |
| `path_error_check.py` `closed_loop_check.py` `suction_check.py` | 路径误差、开环拟合、吸盘区域检查 |
| `uniform_metric_viz.py` `resample_video.py` | 可视化、按采样重定时的视频 |
| `dataset_review.py` | 逐 episode × 逐维的方差归因,查"某一维是不是被单个 episode 带偏" |
| `ruckig_demo.py` | 用 Ruckig 做时间参数化的对照(需 `pip install ruckig`) |

跑法:

```bash
cd <本包根目录> && python eef_error.py ...      # --urdf 有默认值,指向包内 assets_urdf/
```

---

## 8. 权重

已训好的 30k 模型不在本包内。

- 合作者当前部署用的是 `g2_pickplace-30k-gguf-fp16`,asset id `world-studio/agibot-g2-pickplace-arc-v4`
- HF 位置:**（待填）**

---

## 9. 已知限制

- **`deadzone` 是死参数。** 立方根把静止帧的度量值抬到阈值六倍以上,全集合 271 个 episode 里
  它一帧都没归零过。静止段的抑制完全来自曲率度量本身(0.070×)。保留只为兼容旧数据集。
- **默认路径全是我们机器上的布局**,见 §3。
- **部分分析脚本较早写成**,参数默认值可能指向已不存在的中间数据集
  (`task_2-arc-v2` 之类)。显式传参即可。
- **本包只在我们的两个集合上跑过。** 清洗规则(`--task-kind`)是按这两个集合的失败模式写的,
  换任务要重新想。
