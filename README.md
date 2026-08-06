# hello-agent

一个基于 Anthropic 工具调用的轻量 coding agent。项目重点实现了可预测的工具失败处理：
工具可以并发执行，但失败分类与共享状态始终按模型发出的顺序归并。

## 快速启动

1. 在 `.env` 配置 `ANTHROPIC_API_KEY`（或 `API_KEY`）和 `MODEL_ID`。
2. 按 `requirements.txt` 创建环境并安装依赖。
3. 运行 `python -m agent.conversation_loop`。

模型回复会流式显示。任务运行期间第一次按 `Ctrl+C` 会主动关闭当前模型流、
终止正在运行的 bash 进程组，并保留模型已经输出的文本；2 秒内再次按下会退出
程序。REPL 空闲时，输入框有内容则 `Ctrl+C` 清空输入，输入框为空才退出。

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
工具执行失败。工具执行失败和 handler 异常会进入重复失败保护；未知工具由对话循环按
连续模型轮次独立限制。

文件修改会在同一轮内追踪：某路径的 `write_file` 或 `patch` 失败后，只有后续成功写入
同一路径才会清除失败状态。最终回答会披露仍未恢复的失败，避免模型在部分修改失败时声称
任务已全部完成。

工具循环阈值在项目根目录 `config.yaml` 的 `tool_loop_guardrails` 中配置。
配置结构与 Hermes 一致，包含 `warnings_enabled`、`hard_stop_enabled`、
`warn_after` 和 `hard_stop_after`。护栏不读取环境变量；配置在进程启动时读取一次，
修改后需要重启 Agent。配置文件或字段缺失时使用代码中的默认值。

## 目录结构

```text
agent/      对话循环、工具执行归并、失败分类、guardrail 与 trace
tools/      可被模型调用的工具及权限检查
tests/      guardrail 与工具调用链回归测试
notebook/   实验与学习笔记
logs/       本地运行轨迹（不提交）
rag.py      独立的 RAG 实验入口
```
