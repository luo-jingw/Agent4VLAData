# Learning to Discern: Imitating Heterogeneous Human Demonstrations with Preference and Representation Learning

## 基本信息
- 作者：Sachit Kuhar*, Shuo Cheng, Shivang Chopra, Matthew Bronars, Danfei Xu（Georgia Institute of Technology）
- 发表：CoRL 2023（7th Conference on Robot Learning）；arXiv:2310.14196v1 [cs.RO]，2023-10-22
- 类型：方法论文（L2D，离线模仿学习框架），目录分类：C6 多样化数据

## 研究问题
- 离线 IL 中演示质量参差且行为异质（不同演示者风格差异、同等质量的不同完成方式）；主流 IL 假设演示均为理想
- 无环境交互、无奖励信号、仅少量粗粒度质量标签时，如何泛化地估计未见演示（含全新演示者、未见过的质量模式）的质量并选出高质量演示；两个子问题：状态层面特征不足以表征行为（需时序）；专家/无监督方法（ILEED、ELICIT）无法处理风格多样化与等质不同模式

## 方法
- L2D 三段式：(1) 时序对比学习——编码器 E 将轨迹片段映射到 d 维潜空间，triplet margin loss；两种采样策略：S1（anchor/positive 来自特定质量集 A，negative 来自另一质量集 B）、S2（利用领域知识，从"不影响质量"的区域采样 negative，如 Square 任务所有演示结尾相同；附录中细分为 S2.Initial/S2.Final，均取相同区域相似性）(2) 位置编码——增强观测 o'_t = [o_t, p_t]，p_t = t/T 归一化时间步，应对长程非循环任务中相同状态-动作对可对应不同质量 (3) 质量批判 Q：片段嵌入→标量质量分，成对排序损失训练；再用 GMM G（模式数 5）建模不同质量的片段分分布
- 推理：新演示按片段求分→GMM 最大概率归属→统计归入"good"集合的片段占比→排序选 top-k 训练 BC-RNN
- 数据与场景：Robomimic Square 300 条（6 操作者×50），分 good/ok/bad 各 100；场景 A 熟悉演示者（均匀划分），场景 B 不可见演示者（D_known/D_unknown 各 50，来自互不相同演示者）；真实任务 Lift（各侧 60）、Stack/Square（各侧 100）
- 评价指标：样本间 Wasserstein 距离（潜评分分离度，衡量表示质量）；过滤后 BC-RNN 的成功率 SR
- 基线：Naive、ILEED、ELICIT、Preference（TREX 改编+成对排序）、Contrastive（triplet margin+余弦相似度过滤）、Oracle
- 关键超参数（Robomimic 消融）：判别器片段数 20000、片段长 48、批次 128、训练步 500000、初始片段长 12、最终片段长 6、嵌入 12、Q 标签噪声 0.1、判别器学习率 0.0001

## 关键发现
- 消融（Wasserstein 总距离，不可见演示者，Square）：Preference 0.420 → +S1 0.964 → +位置编码 1.773 → +S2 1.869（Good vs Bad 从 0.143 升至 0.848）；附录扩展消融：baseline 0.72 → +S2.Initial 0.87 → +time warp 增强 0.92 → +S2.Final 1.24
- 熟悉演示者（Table 2）：SR Naive 0.44 / ILEED 0.54 / L2D 0.66 / Oracle 0.66（达到 oracle 级）；成功识别池中 top50 高质量演示中的 43 条；选择集构成（100 条）good/ok/bad：Naive 68/16/16、L2D 93/4/3、Oracle 100/0/0
- 不可见演示者（Table 3）：SR Naive 0.20 / Contrastive 0.36 / Preference 0.38 / ELICIT 0.38 / L2D 0.44 / Oracle 0.46；top-50 中 good 数：Naive 18、Contrastive 23、Preference 27、ELICIT 23、L2D 39、Oracle 50
- 现实场景（Table 4）：模拟 Square：Naive 选 37%/SR 10% → L2D 95%/32%；真实 Stack：33%/50% → 97%/80%；真实 Lift：33%/30% → 45%/53%
- 真实 Stack 10 次测试成功率 80% vs 50%（附录 B）；L2D 能对含抖动/摆动片段的演示给出低质量分（Fig.5 标注）
- 局限：整条演示过滤（低质量演示中可能含高质量片段）；good/bad 界限常模糊

## 证据等级
- 实验性：仿真+真实机器人、多样本基线对比、逐组件消融；但样本量小（每场景多数 100–150 条、单任务为主），统计显著性未全部报告；多数性能数字基于单次或少量运行

## 引用
- 对比/前作：Robomimic（Mandlekar et al. CoRL 2021）、ILEED（Beliaev et al. ICML 2022）、ELICIT（Gandhi et al. CoRL 2023）、TREX（Brown et al. ICML 2019）、Behavior Retrieval（Du et al. arXiv:2304.08742）、PEBBLE（Lee et al. 2021）
- 支撑文献：偏好学习（Christiano et al. 2017, Ibarz et al. 2018）、Diffusion Policy（Chi et al. 2023）、BeT（Shafiullah et al. NeurIPS 2022）、CAIL（Zhang et al. NeurIPS 2021）

## 不确定项
- Table 2 与 3 的"选择集构成"与"top 50 演示"的组合关系及"43 out of top 50"的统计口径在正文中未完全说明
- 真实机器人任务的每质量档位演示人数与排序细节（附录 D 提及但未给逐档分布）
- 文中未说明：对"质量等级间界限模糊"场景的量化处理；L2D 与图像输入配合的多模态细节（仅提及真实任务含三相机 RGB）
