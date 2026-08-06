# hello-agent

一个基于 Anthropic 工具调用的轻量 coding agent。项目重点实现了可预测的工具失败处理：
工具可以并发执行，但失败分类与共享状态始终按模型发出的顺序归并。

## 快速启动

1. 在 `.env` 配置 `ANTHROPIC_API_KEY`（或 `API_KEY`）和 `MODEL_ID`。
2. 按 `requirements.txt` 创建环境并安装依赖。
3. 运行 `python -m agent.conversation_loop`。

运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python tests/test_tool_guardrails.py
```

## 工具失败流水线

```text
tool_use → preflight → handler（可并发）→ 主线程顺序归并
                                            ├─ ToolOutcome 分类
                                            ├─ 重复失败 guardrail
                                            ├─ 文件修改失败核验
                                            └─ trace / tool_result
```

`ToolOutcome` 明确区分：`succeeded`、`failed`、`blocked`、`unknown`、
`internal_error` 与 `cancelled`。其中策略或用户权限拒绝属于 `blocked`，不会被误计为
工具执行失败；未知工具和 handler 异常则会计入重复失败保护。

文件修改会在同一轮内追踪：某路径的 `write_file` 或 `patch` 失败后，只有后续成功写入
同一路径才会清除失败状态。最终回答会披露仍未恢复的失败，避免模型在部分修改失败时声称
任务已全部完成。

环境变量：

- `TOOL_GUARDRAIL_WARNINGS`：是否追加重复失败提示，默认 `true`。
- `TOOL_GUARDRAIL_HARD_STOP`：是否在达到阈值时熔断，默认 `true`。

## 目录结构

```text
agent/      对话循环、工具执行归并、失败分类、guardrail 与 trace
tools/      可被模型调用的工具及权限检查
tests/      guardrail 与工具调用链回归测试
notebook/   实验与学习笔记
logs/       本地运行轨迹（不提交）
rag.py      独立的 RAG 实验入口
```
