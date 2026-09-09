# Geometric Action Model for Robot Learning

## 基本信息
- paper_id: paper_038
- 年份: 2026（arXiv v2：2026-06-22）
- 作者: Jisang Han、Seonghu Jeon（共同一作）、Jaewoo Jung、René Zurbrügg、Honggyu An、Tifanny Portela、Marco Hutter、Marc Pollefeys、Seungryong Kim、Sunghwan Hong（通讯）；KAIST AI、ETH Zurich、ETH AI Center
- 来源: arXiv:2606.17046v2 [cs.RO]

## 研究问题
VLA 与视频世界-动作模型（WAM）本质是 2D（深度/尺度/遮挡隐含），在初始状态与相机视角变化下泛化受限。如何把显式 3D 几何先验并入策略？GAM 把预训练几何基础模型（GFM，如 VGGT/DA3）直接复用为感知、时间预测与动作解码的共享 substrate：在 GFM 分层之间插入因果未来预测器，让动作与几何在同一 token 序列/同一次前向中共同产生——回答"动作表示如何与几何特征共享空间、几何结构如何支撑动作建模"。

## 方法
- 基础：DA3-Giant（在 Track4World 微调，40 层 transformer，frame-wise/global 交替注意力，DPT head 从多层隐状态解码每像素几何）。输入每视角 = 1 camera token + P patch token，V 视角拼接。观测为多视角 RGB + proprio 状态 s（ds=7）+ 语言指令。
- 阶段 1 观测编码：GFM 在 Ls=12 处拆分为编码器 E≤Ls（0–12 层）与解码器 D>Ls（13–39 层）；Ls 需深于特征充分性又浅于 DPT 最早取层 m1（Ls<m1），使预测的未来状态可被 DPT 解码为几何。每个上下文时间步独立编码 → 每步几何潜在 {Z(Ls)_{t−H+1},…,Z(Ls)_t}。
- 阶段 2 因果未来预测器（插入在 Ls）：12 层，width dg=1024。每时间步构造 block U_t = [ψs(s_t); ψa(a_{t−1}); Z(Ls)_t]（proprio token + 上一动作 token + GFM 几何 token），语言用冻结 T5 编码为 L_ℓ；block-causal self-attention（含帧内/跨视角掩码，防止未来泄漏，并扩到 GFM 后续 global attention 层）。几何 slot 输出预测下一帧几何 token Ẑ(Ls)_{t'+1}；previous-action slot 投影为下一动作 token ã_{t'}（d 维）——动作与几何在同层联合预测。
- 阶段 3 特征传播与解码：动作 token 复制 V 份（每视角各 1 个）拼进对应几何序列，经 GFM 深层块传播；解码头：轻量动作头 hact 聚合上下文窗口动作 token → 回归 C=8 步、da=7 维 delta 动作 chunk；GFM 原始深度头（DPT）解码动作对齐的未来深度图（scale-invariant + gradient-matching 惩罚，来自[13,15]）。GFM 深层块被重新用作"世界模型预测未来的解码器"。
- 训练：Ltotal = λact·Lact + λfeat·Lfeat + λdepth·Ldepth；Lact 为动作 chunk 的 ℓ1 回归；Lfeat 把预测未来 token 与冻结 GFM 提取的真实下一帧 token 对齐（ℓ1）；Ldepth 监督解码未来深度。λact=3、λfeat=1、λdepth=3。预训练 784K 单臂轨迹（OXE 72%/MimicGen 18%/RoboCasa365 10%，OXE 用 teacher 伪深度、仿真用重渲染深度），64×GH200 ≈96 小时；后训练每基准 8×GH200、batch 160、≈48 小时；H=4（预训）→ H=1（后训），冻结 Ls 前层与 DPT head，AdamW 常数 LR（骨干 5.16e-5、预测器/动作头 5.16e-4）。
- 推理：KV 缓存在线维护历史，每步单次前向预测 C 步动作。

## 关键发现
- LIBERO 与 LIBERO-Plus（Table 1，四套件平均）：GAM Orig 97.6、Plus 85.5（↓12.1）；相机扰动 Cam 83.1（比最强基线 π0.5 72.0、Cosmos-Policy 73.4 高 ↑9.7%p）；Robot 70.0、Lang 84.8、Light 97.2、BG 94.3、Noise 95.3、Layout 79.1。对比：Cosmos-Policy 98.5/82.4（↓16.1）；π0.5 96.9/84.6（↓12.3）Cam 72.0；OpenVLA-OFT 97.1/69.6（↓27.5）；π0.5+Spatial Forcing Cam 0.1；π0.5+ROCKET 95.3/47.5（↓46.6）Cam 30.9；Fast-WAM 97.6/50.0。
- 真机 4 任务（每任务 20 trial = 10 ID + 10 OOD；OOD 为外置相机平移 85cm+旋转 45°）：GAM 显著优于全部基线（π0.5、Spatial Forcing），OOD 下仍保持鲁棒（Fig. 4；各任务精确 bar 值未以表格给出）。
- 推理速度（单 GH200，同一输入、bf16、Torch Compile、无 CUDA Graphs）：GAM 17.5ms vs π0.5 29.2ms、OpenVLA-OFT 70.1ms（正文 Table 4 表记为 77.8ms）、Cosmos-Policy 382.4ms；GAM 部署配置（+CUDA Graphs，单遍静态推理）6.9ms ≈145Hz，相对 Cosmos-Policy 最大 55× 加速。
- 规模：总参数 ≈1.4B（骨干 ViT-Giant 1136.5M、其中 13–39 块可训 ≈765M；DPT 50.1M 冻结；Causal Future Predictor 210.2M；action head 8.0M），可训 ≈983.2M；远小于 π0.5 3.3B、OpenVLA-OFT 7B、Cosmos-Policy 2B。
- 消融（Table 2，Object 套件）：预训练是关键——无预训练 Orig 93.6（vs 99.6）、Plus 50.0（vs 89.7）；有预训练时去掉 Ldepth 或 Lfeat 影响很小（Plus 89.0/89.5 vs 89.7）；无预训练时 future-prediction losses 提供强几何监督（仅 Ldepth 80.0、仅 Lfeat 66.5、全开 73.4）；H=1 优于 H=2（84.4）/H=4（85.1）。
- 分裂层消融（Table 3）：Ls=12 最佳（Orig 99.6/Plus 70.1，无 Ldepth）；Ls=0 崩溃（5.4/1.8）、Ls=19 次优（95.6/63.4）、Ls≥27 完全失败（1.2/0.0）——预测 token 需足够深层交互以整合进 3D 先验；默认 Ls=12 恰在 frame-wise/global 注意力切换处。
- RoboCasa-Kitchen（24 任务，Table 10，动作 chunk 8→16）：GAM 69.4% vs Cosmos Policy 67.1%、FLARE 66.4%、Video Policy 66.0%、π0 62.5%、GROOT-N1 49.6%。
- 相机扰动难度分解（Fig. 5，L1–L5）：GAM 每级均高于所有基线，最强扰动下优势仍存在。

## 证据等级
- 等级: A（一手实验）
- 理由: 直接目标论文；覆盖四套 LIBERO + LIBERO-Plus 全扰动维度与难度级分解、RoboCasa-Kitchen 24 任务、真机 4 任务 ID/OOD、推理速度与参数量对照（π0.5、Cosmos-Policy、OpenVLA-OFT、Spatial Forcing、ROCKET 等重评或引用公开结果）；局限：真机结果无表格逐任务数字（正文仅定性+图），部分基线取自他人论文（Fei et al.、Zheng et al.），仿真评估每套件单独微调而非单一跨套件模型。

## 引用
- 问题与相关工作：Sec.1–2（p.2–4）；GFM 预备与 GAM 方法：Sec.3–4（p.4–6）；训练与推理：Sec.4.4（p.6）；实现/数据细节：Sec.5.1、App. A.1–A.2（p.6–7、p.10–12）；LIBERO/LIBERO-Plus 主结果：Sec.5.3、Table 1（p.7）；消融：Sec.5.4、Table 2/3（p.8）；速度与规模：Sec.5.5、Table 4/8/9（p.8、p.13–14）；真机：App. A.3（p.12）；RoboCasa-Kitchen：App. B.1（p.14–15）。

## 不确定项
- "低秩/流形"：论文未用这些词；动作表示为单个 d 维动作 token（复制 V 份/视角）在 GFM 潜空间中的几何结构上预测，未报告动作潜空间的维度分析或流形性质。
- 真机图 4 的逐任务成功率（12 根柱）无表格化精确值；文本流中无法恢复各任务 ID/OOD 精确百分比，仅可引用正文定性描述；完整表格未见。
- Table 1 主行聚合值（Orig 97.6 等）为四套件平均、每套件独立微调模型；Table 12 显示套件间差异大（Long 套件 Orig 94.6/Plus 78.0）。不同套件的几何监督来源一致（仿真 GT depth）。
- Table 2 中部分配置行的数值排列在文本流中存在歧义（某行与全量配置同为 99.6/89.7），本抽取按原文行序转录，未做推测对齐。
- 附录 A.1 提到对 OXE 用"common control interface"映射与排除，具体映射规则及列表未提供；DCT/流形无关的背景（本文不涉及）。
- 与 Geocentric/其他 GFM（VGGT）的替换对比未提供（仅 DA3-Giant）；结论限制自述：语言推理能力受冻结 T5 编码器限制。
