# ProGAL-VLA: Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models

## 基本信息
- 论文：ProGAL-VLA: Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models（paper_045）
- 作者：Nastaran Darabi，Amit Ranjan Trivedi
- 单位：University of Illinois Chicago
- 出处：arXiv:2604.09824v1（cs.RO），2026-04-10，v1 预印本，未标注发表状态
- 项目页：https://nstrndrbi.github.io/ProGAL

## 研究问题
- VLA 两大失败：(1) 语言漠视（依赖视觉捷径，对指令变化不敏感）；(2) 机器人不稳定性（语义推理扰动视觉运动协调，语义推理强的模型机器人类别弱）。
- 根源断言：多数 VLA 用浅层拼接融合模态，语言嵌入调节视觉但不保证符号目标对应场景中可执行的实体，控制层运行在未验证表示上。
- 目标：动作执行前强制"验证瓶颈"，使符号子目标绑定到具体 3D 实体；同时让歧义可量化（SACA 熵）。

## 方法
- 四组件框架：
  1. π_slow：Qwen-2.5-VL-Instruct-7B，异步、每 episode 调用一次（或 grounding 失败时），输出符号模板（如 "grasp green mug"）s_t=π_slow(L, O_t, M_{t−1})；只规范词法、不做低层推理或多步前瞻。
  2. GSM（grounded state module）：YOLO-World（L 版）开放式检测 + Metric3D(v2, ViT-L) 深度 → 抬升至 3D 实体图 G_t={e_1,…,e_n}；实体记忆 M_t（Nmax=16 WAS fifo 更新，实体嵌入 d=1024），E_t=Concat(G_t, Retrieve(M_{t−1}, O_t))；实体编码外观/位姿/语义；用于遮挡与短暂时序平滑，不做长程推理。
  3. SACA：q=Embed_sym(s_t)，K,V=Embed_gnd(E_t)，g_t=Softmax(QKᵀ/√d)V；注意力熵 H_t=−Σα_{t,i} log α_{t,i} 作为固有歧义信号（低熵=唯一绑定，高熵=多实体可匹配）；仅验证后的 g_t 传给控制器。
  4. π_fast：OpenVLA-7B（与所有基线一致），输入 g_t（投影为 4096 维）+O_t+q_t，a_t~π_fast(g_t,O_t,q_t)，L_action=‖a_t−a_t*‖²；不含原始语言 ⇒ 结构上强制 a_t⊥L|(g_t,O_t,q_t)（验证瓶颈，架构性约束而非统计性）。
- 跨模态对齐（GAC，符号文本↔3D 实体）：q=Embed_sym(s_t), k⁺=Embed_gnd(e⁺)；L_GAC=−log exp(sim(q,k⁺)/τ)/Σ_i exp(sim(q,k_i)/τ)（infoNCE，n 个负实体）；L_total=L_action+λ·L_GAC，λ=0.1，τ=0.07，AdamW lr=2e−5、batch 128、4×A6000。
- 对齐对离线生成（弱监督）：(i) 教师 VLM 对演示符号分段得到 s_t；(ii) GSM 建 3D tracklet {T_obj_i}；(iii) 子目标结束时刻最近邻匹配 e⁺=argmin_i‖pos(T_obj_i,t_end)−pos(gripper,t_end)‖；实体身份与子目标标签从不手工标注。
- 理论（第 4 节，含证明）：命题 1 语言影响分解 I(L;a_t|O_t,q_t)=I(L;g_t|O_t,q_t)−I(L;g_t|a_t,O_t,q_t)；定理 1 实体级 InfoNCE 下界 I(S;E)≥log N−E[L_GAC]；命题 2 动作鲁棒性 TV(π_fast(g_t)‖π_fast(g'_t))≤L_π·L_Ψ·d_G(T,id)（Lipschitz 摄动界）；熵阈值化 → Risk-Coverage 曲线（选择性预测）。
- 度量：语言漠视指数 Λ^π=1−E[I_t^π]/log|L|（由条件互信息估计，越低越好）；实体检索用 SACA logits 的 Recall@1/5（候选集合 N=8/16/32）。
- 数据集：LIBERO-Plus [9]（7 维扰动：Camera/Robot/Language/Light/Background/Noise/Layout）；自建 CAB——场景 32/8/8（train/val/test），每场景物体 3/4.8/6（min/mean/max），类别 blocks/mugs/bottles/fruit，颜色 4（≈25% 各）、尺寸 2（≈50% 各）；指令 2400（1200 无歧义/1200 歧义），歧义规则=删除属性直至 ≥2 实体匹配；成功=无歧义指令正确执行 /歧义指令输出 "[CLARIFY]"（熵门控），歧义下指定动作算失败。

## 关键发现
- LIBERO-Plus 鲁棒性（成功率%）：ProGAL-VLA 总 85.5 vs OpenVLA 17.3、OpenVLA-OFT+ 79.6；8 列中 6 列最优；机器人 3.5→71.5（摘要口径"30.3→71.5"为相对最强先验基线 OFT+ 30.3）、布局 77.6→86.7、相机 0.8→93.2、语言 23.0→93.6。
- 语言漠视：条件于 L 复现漠视失败；条件于 s_t 略降但受限（未实体绑定）；条件于 g_t 成功率最高且漠视最低——0.36/0.49/0.57 → 0.08/0.14/0.19（simple/spatial/relational，越低越好），摘要称降 3x–4x。
- 实体检索（SACA logits，Recall@1）：N=8 0.41→0.71（R@5 0.72→0.93）；N=16 0.28→0.58；N=32 0.15→0.41。
- CAB：AUROC 0.52→0.81；AUPR 0.49→0.79；ECE 14.3→4.6；Cov@95 0.31→0.78；FPR@95 0.72→0.18；Clar@Ambig 0.09→0.81；Unambig SR 0.74→0.89；Total 0.47→0.79。
- 失效分析：grounding 失败率 OpenVLA 41.2% → w/o LGAC 22.4% → ProGAL-VLA 6.3%（0.83 GAC 后 6.3）；残余错误主为抓取失败（7.1%）；表 7 分解：Joint Drift 2.1→68.2、Joint Jitter 4.9→74.8、Viewpoint Shift 1.2→85.1、FOV 0.4→91.7。
- 延迟（每步 ms，均值±σ，表 5）：Detector 43.0±26.4、GSM 15.8±9.7、SACA 10.7±6.6、π_fast 26.9±16.5、总计 96.4±59.2（表 5 标吞吐 10.31 FPS；正文另称 ≈107.5±66.0 ms、9.31 FPS）。
- 消融：无 π_slow（Q=Embed_sym(L)）与无 GSM（patch token 作 K,V）与无 L_GAC 均在语言漠视/检索/歧义上下降；π_slow 主要贡献为词法正则（非 LLM 模板抽取器替代后大部分增益保持）。

## 证据等级
- 实证：基于仿真 LIBERO-Plus + 自建 CAB（8 条基线：OpenVLA/OpenVLA-OFT 三变体/π0/π0-Fast/NORA/WorldVLA/UniVLA/RIPT-VLA，3 组消融）；提供命题 1/定理 1/命题 2 的证明；自报 arXiv v1 预印本，无正式发表、无第三方复现、无实体机器人实验（限制中明确）。

## 引用
- 基线：OpenVLA [15]，OpenVLA-OFT/w/m/+ [16]，π0 [2]，π0-Fast [25]，NORA [13]，WorldVLA [6]，UniVLA [5]，RIPT-VLA [32]，LIBERO-Plus [9]，CLIP [28]，R3M [24]，Say-Can [4]，Code as Policies [19]。

## 不确定项
- 延迟数值 96.4±59.2 ms（表 5，10.31 FPS）与正文 ≈107.5±66.0 ms（9.31 FPS）不一致，文中未说明；表 5 分项合计亦不精确等于总计值 107.5。
- 摘要"30.3%"对应 OpenVLA-OFT+（表 2 OpenVLA 为 3.5），与正文"robot perturbations increases from 30.3 to 71.5 percent"对比口径（先验最强基线）文中未明确说明；布局 77.6→86.7 同理。
- GAC 训练时负实体数 N 的具体取值、批次内负采样策略、Softmax sim 的具体形式（sim 未定义，仅示公式）、SACA 维度 √d 的 d 值——文中未说明。
- 表 3/表 4（CAB 与检索）的重复次数、seed 数、量程与置信区间——文中未说明；CAB 只覆盖属性级歧义（自述局限）。
- 基线对比的公平性口吻：π_fast 每步推理仍为 OpenVLA-7B，但 π_slow（7B）+detector+GSM+SACA 为额外系统复杂度，仅以延迟表格衡量——文中未说明公平性口径。
