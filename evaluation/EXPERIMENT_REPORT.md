# Agent Tracer Evaluation 实验报告

## 实验目标

对 Athena Coding Agent 的任务理解、代码定位、Evidence-to-Action、工具效率、错误恢复、验证完整性和 patch 质量做轨迹级评测。本实验只评估 Agent 产品效果，不用于 SFT/RL 数据生产。

## 设置

- 被测 Agent：Athena + DeepSeek flash 执行模型。
- Judge：GLM-5.2，temperature=0。
- 任务集：8 个本地 Python 诊断任务，涵盖边界条件、解析、多文件数据流、缓存状态和集合语义。
- 重复次数：每题 3 次，共 24 条独立轨迹。
- 隔离：每条轨迹在独立 Git 工作区运行，先验证 baseline 失败，再运行 Agent，最后执行同一测试并快照 patch。
- Rubric：7 维、100 权重；`not_applicable` 维度从分母剔除后重新归一。

## 结果

| 指标 | 结果 |
|---|---:|
| 有效轨迹 | 24 / 24 |
| Instance Success Rate | 100.00% |
| 平均 Check Success Rate | 99.31% |
| 平均 Effective Action Ratio | 88.85% |
| 平均冗余工具调用率 | 2.03% |
| 工具失败恢复率 | 100.00% |
| GLM Judge 覆盖 | 24 / 24 |
| 平均 Judge 加权分 | 99.58 / 100 |
| Judge 分数分布 | 100 分×20；97.5 分×4 |
| Failure Onset | 0 |
| 平均输入 / 输出 token | 3571.7 / 1571.5 |

维度分析显示，任务理解、定位、Evidence-to-Action、恢复、验证和 patch 质量的平均分均为 100；工具效率为 95.83。尽管 24 条轨迹全部修复成功，单轨迹 EAR 介于 66.67%–100%，说明 pass/fail 无法显示的工具路径波动可被 Tracer 捕获。

## 真实故障与恢复

- 首轮 8 条 Judge 中有 1 条返回缺失 `checks`；增量续跑复用 7 条缓存，只补 1 次请求。
- 24 条批量评测中，长轨迹曾因输出预算发生 JSON 截断或字段缺失。系统增加二阶段回退：分别评判 Rubric checks 和 step labels/onset，再合并做统一证据校验，最终覆盖率达到 24/24。
- Agent 轨迹中出现命令 guardrail 拦截、`pytest` 不可用和自建 sanity case 错误等情况；Judge 标注出 17 个 `RECOVERY`、12 个 `RECOVERABLE_ERROR` 和 5 个 `NEUTRAL` 步骤，未出现未恢复的 Failure Onset。

## 结论边界

这是本地定向诊断集，不是 Multi-SWE-bench 或 SWE-bench Verified；它能支撑“流水线可用、执行路径可诊断、Judge 回退有效”，不能支撑“模型在公开编码 Benchmark 上领先”。
