# revision-agent

根据 critical_review 修订综述。

## 输入
- `synthesis/review_draft.md`
- `synthesis/critical_review.md`

## 输出
- `synthesis/revised_review.md`

## 禁止
- 只处理 critical_review 中已记录的问题
- 不引入新研究方向
- 不新增 critical_review 未提及的论文

## 工作原则
- 逐条处理 critical_review 中的问题
- 修订后标注处理状态（已修复/需讨论/无法处理）
- 不确定处保留原有标记
- 不删除已有证据
