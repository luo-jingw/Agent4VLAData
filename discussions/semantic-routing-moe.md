# 条件计算与语义路由（外部讨论总结）

> 来源：ChatGPT 讨论归档（原始 html 仅存本地）。本文为内容总结，非逐字记录。

## 核心命题

1. **encoder 不必然损失信息**；必然发生的是"有限容量/压缩性/不变性要求"下必须选择性丢信息。
2. **没有绝对的"好表征"**——representation is sufficient **relative to a query**。
3. 更强命题：**"理解"不等于拥有正确的表征，而等于拥有根据目标动态构造表征/计算的能力**（类比：有限大小的 Python 解释器执行无限多程序）。
4. 语义本质是一种"程序"：有限模型通过共享原语 + 语义组合表示巨大的人类语义函数集。
5. 因此"哪些东西需要可分"不是训练时一次性决定的——可在计算过程中动态构造当前所需内部状态。

## 实现形态：MoE 语义路由

- **语义条件 = compile-time information**；MoE experts = instruction set/primitives；routing sequence = program。
- 假设的强版本：语义条件 → 一次性生成跨所有层的完整 routing program → encoder/decoder 共享同一 program → 各层仅廉价 residual routing（encoder_i/decoder_j 对，i+j=层数，z 可放任意层，路由可脱离当前数据流）。

## 文献现状（对话检索结论）

| 组件 | 现状 |
|---|---|
| task-level 粗粒度路由 | 已有（2021 task-MoE：token/sentence/task 三粒度对比，task 级可推理前加载 experts、吞吐占优） |
| 两级路由（语义 router + 动态 router） | 已有（THOR-MoE, ACL 2025：先预测 domain/language 预选，再 token 级细路由） |
| instruction 直接做路由 | 已有雏形（MoIRA 2026，机器人领域：按文本指令选专家） |
| 路由签名涌现 | 有实证（同一 task 的 prompt 产生相似 routing signature，仅靠 signature 四分类交叉验证 92.5%） |
| 残差路由局限 | 有反证（MSR 2026 counterfactual：标准 top-k router 在脆弱推理 token 上未必选到更好的等算力路径） |
| **语义条件 → 全局 routing program → encoder/decoder 共享 + 残差路由** | **无标准范式——组合缺口** |

## 结论

- 纯静态路由不可行（token 级信息在脆弱点仍有价值）；可辩护位置是"全局 program + 残差细路由"的混合。
- 工程价值直接：粗粒度 task routing 减少逐 token expert dispatch 的动态性，利于编译、预取、expert placement 与推理调度。
- 与表征问题的统一：metric、representation、encoder/decoder、semantic routing 合为一体——表征等价性由路由程序定义，而非人为规定欧氏距离。
