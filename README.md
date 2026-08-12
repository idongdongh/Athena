# Athena

一个基于 Anthropic 工具调用的轻量 coding agent。项目重点实现了可预测的工具失败处理
和上下文状态管理：工具可以并发执行，但失败分类与共享状态始终按模型发出的顺序归并。

## 快速启动

1. 在 `.env` 配置 `ANTHROPIC_API_KEY`（或 `API_KEY`）和 `MODEL_ID`。
2. 按 `requirements.txt` 创建环境并安装依赖。
3. 运行 `python cli.py`。

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
                                            └─ tool_result
```

`ToolOutcome` 明确区分：`succeeded`、`failed`、`blocked`、`unknown`、
`internal_error` 与 `cancelled`。其中策略或用户权限拒绝属于 `blocked`，不会被误计为
工具执行失败。工具执行失败和 handler 异常会进入重复失败保护；未知工具由对话循环按
连续模型轮次独立限制。

文件修改会在同一轮内追踪：某路径的 `write_file` 或 `patch` 失败后，只有后续成功写入
同一路径才会清除失败状态。最终回答会披露仍未恢复的失败，避免模型在部分修改失败时声称
任务已全部完成。

工具循环阈值在项目根目录 `config.yaml` 的 `tool_loop_guardrails` 中配置。
配置包含 `warnings_enabled`、`hard_stop_enabled`、`warn_after` 和
`hard_stop_after`。护栏不读取环境变量；配置在进程启动时读取一次，
修改后需要重启 Agent。配置文件或字段缺失时使用代码中的默认值。

## 上下文状态

每次模型响应都会先归一化停止原因和 token usage。`end_turn`、`tool_use`、
`max_tokens`、`refusal` 等状态由主循环分别处理，不再把所有非工具响应都视为正常完成。
Provider 未返回 usage 时，Agent 会根据 system、消息历史和工具 schema 做本地粗估。

纯文本达到 `max_tokens` 时会从断点有界续写；包含工具调用的截断响应不会执行，Agent
会扩大输出预算后有限重试。输出上限、上下文窗口、压缩阈值和续写次数在 `config.yaml`
的 `model`、`context`、`compression` 段配置。

上下文压缩由 `ContextEngine` 定义生命周期，`ContextCompressor` 执行旧工具结果裁剪、
首部保护、token 预算尾部保护、中段结构化摘要、
迭代摘要和工具调用配对修复，`conversation_compression` 负责成功后原子替换历史。
主循环在 API 调用前使用粗估 token 检查，并在工具结果返回后使用真实 usage 再检查；
摘要失败时原消息保持不变，连续无效压缩会停止触发。

## 会话持久化与搜索

会话状态默认保存在 `.athena/state.db`。`SessionDB` 使用 SQLite WAL 模式，`sessions`
保存会话元数据和 token 统计，`messages` 增量保存用户、模型及工具消息。程序重启默认
创建新会话；历史对话需要通过 `/resume` 显式恢复。

上下文压缩支持两种持久化模式。默认在压缩后结束父会话并建立
`parent_session_id` continuation；`compression.in_place: true` 则保持同一会话 ID，
把旧消息标记为 `active=0, compacted=1`。两种模式都不会物理删除压缩前历史。

REPL 支持：

- `/new`：创建新会话。
- `/sessions`：列出历史会话。
- `/resume <id>`：恢复会话；父 ID 会自动定位到最新压缩 continuation。
- `/search <关键词>`：使用 FTS5 搜索消息，中文子串使用 trigram 索引。
- `/archive [id]`：归档指定会话，省略 ID 时归档当前会话。

会话数据库路径和开关由 `config.yaml` 的 `session` 段控制；数据库及 WAL/SHM 文件已排除
在 Git 之外。普通启动始终创建新会话，只有显式 `/resume` 才恢复历史会话。

旧版本创建的 `.hello-agent/state.db` 会被自动识别并继续复用，避免改名后丢失历史会话。

## 目录结构

```text
cli.py        经典交互式 CLI 入口与 AthenaCLI 编排器
run_agent.py  AIAgent 会话状态和 Agent 内核适配入口
session_db.py SQLite 会话存储、压缩链和全文搜索
athena_cli/   CLI 配置、斜杠命令、会话列表和 Agent 初始化
agent/        对话循环、上下文压缩、工具执行与 guardrail
tools/        可被模型调用的工具及权限检查
tests/        本地回归测试（不发布到远端）
notebook/     本地实验与学习笔记（不发布到远端）
```
