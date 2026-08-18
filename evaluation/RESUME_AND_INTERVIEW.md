# 简历与面试材料

## 简历项目经历

**Agent Tracer Evaluation｜Trajectory Diagnosis + Rubric Eval and Wash Pipeline**

• **项目背景/目标：** Coding Agent 的任务通过率无法解释规划、工具效率、错误恢复与 patch 质量差异；目标是构建轨迹级自动评测流水线，定位 Agent 失败或低效步骤，为 Harness 迭代提供可追溯证据。

• **我的职责：** 项目 owner，从 0 到 1 设计并落地 `run → evaluate → wash → judge → report` 链路，完成 Agent 内核埋点、独立任务运行器、规则评估、LLM-as-Judge 和增量报告。

• **Tracer 与轨迹建模：** 为 model request/response、tool call/result、guardrail、context compression 和 turn lifecycle 设计稳定 ID 与 append-only JSONL 事件协议；大文本分离存 artifact，写入前脱敏，Tracer 异常 fail-open 不影响 Agent 主链路。

• **Rubric Eval + Failure Onset：** 建立 7 维 100 权重 Rubric，用确定性 check 计算 CSR、ISR、EAR、冗余调用率和恢复率；GLM-5.2 逐步标注 `CORRECT/RECOVERABLE_ERROR/RECOVERY/FAILURE_ONSET/CASCADE_FAILURE`，强制引用真实 event_id 并校验首次未恢复错误。

• **Judge 稳定性与 Wash：** 实现并发数 + RPM 双限流、429/5xx 指数退避、逐轨迹原子落盘与增量续跑；对长输出 JSON 截断/缺字段自动回退为 checks 与 step diagnosis 二阶段补判，Wash 仅生成派生清单、不删除原轨迹。

• **实验结果：** 在 8 类 Python 诊断任务×3 次重复的 24 条独立轨迹上，实现 24/24 有效轨迹与 Judge 覆盖，ISR 100%、平均 CSR 99.31%、EAR 88.85%、工具失败恢复率 100%、Judge 均分 99.58/100；全部任务通过时，仍识别出单轨 EAR 66.67%–100% 的执行效率波动。

## 30 秒面试讲法

这个项目不是训练模型，而是评估 Agent 整体系统。我在 Athena 的 model、tool、guardrail 和 compression 调用链上做了轨迹埋点，再用规则指标和 GLM-5.2 Rubric 分析每一步是有效行动、可恢复错误还是失败起点。流水线支持任务隔离、增量续跑、双限流和长 JSON 二阶段补判。24 条实验全部修复成功，但 EAR 从 66.67% 到 100%，说明它能找出 pass rate 看不到的 Agent 执行效率问题。

## 算法岗面试重点

- 不只看 pass@1：相同最终结果可能来自不同的工具成本、恢复能力和 patch 风险。
- 规则 + LLM：完整性、调用数、重复调用和测试结果由确定性程序判断；语义理解、Evidence-to-Action 和 patch 质量才交给 Judge。
- Failure Onset：首个产生后续影响且没有被恢复的错误决策；单次工具失败不等于 Onset。
- 证据约束：Judge 结论必须引用存在的 event_id，并校验维度集合、step 集合和 onset-label 一致性。
- 局限：当前任务较基础，Failure Onset 为 0，不能宣称 Onset 检测在公开 Benchmark 上的准确率；后续需要带人工金标的失败轨迹。

## 开发岗面试重点

- 事件协议：`trace_id/session_id/turn_id/step_id/api_request_id/tool_call_id` 区分轨迹、轮次、模型请求和工具调用。
- 可靠性：append-only JSONL、单轨迹原子结果文件、幂等缓存、不覆盖 raw trace、子进程组超时终止。
- 并发控制：`ThreadPoolExecutor` 限制在途数，滑动一分钟窗口限制 RPM，对 429/5xx 做带 jitter 的指数退避。
- 故障案例：长轨迹 JSON 被截断；增加输出预算后仍发现输出膨胀，最终用 checks/steps 分阶段补判限制单请求尺寸。
- 测试：覆盖轨迹完整性、脱敏、大字段 artifact、规则指标、Judge 证据校验、429 重试、增量缓存、任务隔离和二阶段回退。
