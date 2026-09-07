# 术语表

## 机器人数据

- **遥操（teleoperation）**：人操作机器人录制示范数据；数据质量受操作者水平影响。
- **episode**：一次任务从开始到结束的完整录制（含多路相机与关节状态）。
- **VA/VLA 模型**：Vision-Action / Vision-Language-Action 策略模型，输入图像与
  指令，输出动作。
- **action chunk**：策略一次推理输出的固定长度动作序列（长度 H）。
- **动作重采样（action resampling）**：不改观测帧、只改标签内部动作序列的
  采样方式（本项目现有方法）。
- **次优解（suboptimal demonstration）**：能完成任务但路径/策略非最优的
  演示（绕路、犹豫、多余动作）。本调研的核心剔除对象。
- **事件窗（event window）**：吸盘触发命令沿 + 真空反馈过渡的帧区间。
- **损失加权（loss weighting）**：对不同帧/步的监督乘不同权重。

## 经典示教学习（LfD）

- **LfD / Imitation Learning**：从人类/专家示范学习策略。
- **轨迹分割（trajectory / motion segmentation）**：把连续轨迹切成有语义的段。
- **关键帧提取（keyframe extraction）**：从轨迹挑少量代表帧/点。
- **时间对齐（time alignment / DTW）**：对齐不同演示的时序。
- **运动基元（motion primitive / ProMP）**：可复用的参数化运动单元。

## Agent 基础

- **LLM / VLM**：大语言模型 / 视觉语言模型。
- **agent**：以 LLM/VLM 为决策核心、可调用工具、按任务自主规划与执行的系统。
- **ReAct**：Reasoning + Acting 交替（思考→行动→观测→再思考）的 agent 范式。
- **tool use / function calling**：模型生成结构化调用外部函数的请求。
- **planning**：任务分解与步骤生成。
- **memory**：跨步/跨会话保存的信息。
- **multi-agent**：多个 agent 分工协作/互审。
- **LLM-as-judge**：用 LLM 对输出/样本做质量评分。
- **agentic data curation**：用 agent 生成、筛选、标注、优化数据。
- **DSPy / TextGrad**：用程序化/梯度式方法自动优化 LLM 管线的框架。

## 证据等级

| 等级 | 定义 |
|---|---|
| direct | 直接回答调研问题 |
| indirect | 研究相邻机制 |
| weak | 只提供背景、指标或方法启发 |
| irrelevant | 不能支撑当前调研问题 |
