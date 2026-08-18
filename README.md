# Athena Eval

Athena Eval 是一个可运行的 Coding Agent 与轨迹级评测工程。仓库将 Agent
执行内核、结构化 Tracing、确定性 Checks、LLM-as-Judge、Failure Onset
定位和 Wash/Report 流水线放在同一个可复现环境中。

评测系统不只检查任务是否通过，还会分析工具效率、错误恢复、验证完整性、
patch 质量以及首个未恢复的错误决策。

## 功能

- 使用 `bash`、`read_file`、`write_file`、`patch`、`search_files`、`web_search` 和
  `web_extract` 完成编码任务。
- 对工具结果进行统一失败分类，并阻止未知工具、重复失败和无进展调用无限循环。
- 流式输出模型回复；任务运行期间支持 `Ctrl+C` 优雅暂停。
- 达到上下文阈值时压缩早期消息，同时保留近期任务和工具调用配对。
- 使用 SQLite 持久化会话，并通过 FTS5 搜索历史消息。
- 使用项目内 Markdown 文件保存长期记忆，并在后台定期复盘对话。

## 环境要求

- Python 3.12 或更高版本
- [uv](https://docs.astral.sh/uv/)
- Anthropic API key，或兼容 Anthropic Messages API 的服务

## 快速启动

克隆项目并进入目录：

```bash
git clone https://github.com/idongdongh/Athena-eval.git
cd Athena-eval
```

创建虚拟环境并安装依赖：

```bash
uv venv
uv pip install -r requirements.txt
```

在项目根目录创建 `.env`：

```dotenv
ANTHROPIC_API_KEY=your_api_key
MODEL_ID=your_model_id

# 使用兼容服务时按需设置
# BASE_URL=https://example.com
```

启动 Athena：

```bash
uv run python cli.py
```

目前仓库尚未发布 `athena` 全局可执行命令，因此仍需从项目目录启动。

## 配置

非敏感配置统一放在项目根目录的 `config.yaml`；API key 等秘密只放在 `.env`。
以下是完整的常用配置示例：

```yaml
model:
  max_output_tokens: 4096
  context_window: 128000

context:
  max_length_continuations: 2

compression:
  enabled: true
  threshold: 0.75
  target_ratio: 0.20
  protect_first_n: 3
  protect_last_n: 6
  abort_on_summary_failure: true
  max_per_turn: 2
  in_place: false

session:
  enabled: true
  database: .athena/state.db

memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  nudge_interval: 10
  directory: memories

tool_loop_guardrails:
  warnings_enabled: true
  hard_stop_enabled: true
  warn_after:
    exact_failure: 2
    same_tool_failure: 3
    idempotent_no_progress: 2
  hard_stop_after:
    exact_failure: 5
    same_tool_failure: 8
    idempotent_no_progress: 5
```

配置文件或字段缺失时使用代码默认值。修改配置后需要重启 Athena。

## 交互命令

进入 REPL 后可以使用：

- `/new`：结束当前会话并创建新会话。
- `/sessions`：列出近期会话；`/session` 是它的别名。
- `/resume [session_id]`：恢复历史会话；不传 ID 时先列出可恢复会话。
- `/search <关键词>`：搜索历史消息。
- `/archive [session_id]`：归档指定会话；省略 ID 时归档当前会话。

程序重启时默认创建新会话，不会自动恢复历史对话。

## 中断行为

模型回复会流式显示。任务运行期间第一次按 `Ctrl+C` 会请求暂停：关闭当前模型流、终止
正在运行的 bash 进程组，并保留模型已经输出的文本；2 秒内再次按下会退出程序。

REPL 空闲时：

- 输入框有内容：`Ctrl+C` 清空输入。
- 输入框为空：`Ctrl+C` 退出 Athena。
- `Ctrl+D`：结束输入并退出。

## 工具失败处理

```text
tool_use → preflight → handler（可并发）→ 主线程按模型顺序归并
                                            ├─ 失败分类
                                            ├─ 重复失败 guardrail
                                            ├─ 文件修改失败核验
                                            └─ tool_result
```

工具结果分为 `succeeded`、`failed`、`blocked`、`unknown`、`internal_error` 和
`cancelled`。策略或用户权限拒绝属于 `blocked`，不会被误计为工具执行失败；未知工具和
重复失败都有独立的有界恢复路径。

文件修改会在同一轮内追踪。某路径的 `write_file` 或 `patch` 失败后，只有后续成功写入
同一路径才会清除失败状态；最终回答会披露仍未恢复的修改失败。

## 上下文管理

每次模型响应都会归一化停止原因和 token usage。纯文本达到输出上限时会从断点有限续写；
包含工具调用的截断响应不会执行，而是扩大输出预算后有限重试。

上下文接近阈值时，Athena 会裁剪较旧的工具结果、保护首部和近期消息、生成结构化摘要，
并修复 `tool_use` / `tool_result` 配对。摘要或持久化失败时不会替换原消息历史。

默认情况下，压缩会结束父会话并创建带 `parent_session_id` 的 continuation；设置
`compression.in_place: true` 后则在同一会话中把旧消息标记为已压缩。两种模式都不会
物理删除压缩前的历史。

## 会话持久化与搜索

会话默认保存在 `.athena/state.db`。SQLite 数据库使用 WAL 模式，增量保存用户消息、模型
回复、工具消息、token 统计和会话元数据。

除 `/search` 外，模型也可以调用只读 `session_search` 工具：搜索关键词、浏览近期会话、
读取指定会话，或围绕命中消息查看上下文。历史搜索结果只代表过去记录；当前文件、URL
和外部系统状态仍以原始来源为准。

旧版本创建的 `.hello-agent/state.db` 会被自动识别并继续复用。

## 长期记忆

启用 `memory` 配置后，Athena 使用模型可调用的 `memory` 工具维护：

- `memories/MEMORY.md`：项目环境、约定和可复用经验。
- `memories/USER.md`：用户身份、偏好和沟通习惯。

记忆在会话开始时读取为冻结快照并注入 system prompt。会话中新增或修改的内容会立即
持久化，但只会在下次会话或上下文压缩边界进入新的 prompt 快照。

每完成 `memory.nudge_interval` 个用户 turn，Athena 会在正常回答交付后启动一次隔离的
后台记忆复盘。后台复盘只能调用 `memory`，不会修改前台消息状态；将间隔设为 `0` 可关闭。

记忆写入和 prompt 注入都经过威胁模式扫描，疑似提示注入内容不会进入 system prompt。
`memories/` 包含本地信息，默认不会提交到 Git。

## 测试

```bash
uv run python -m unittest discover -s tests
uv run python -m compileall -q agent athena_cli tools cli.py run_agent.py session_db.py
```

## 目录结构

```text
cli.py        交互式 CLI 入口与 AthenaCLI 编排器
run_agent.py  AIAgent 会话状态和 Agent 内核适配入口
session_db.py SQLite 会话存储、压缩链和全文搜索
athena_cli/   CLI 配置、斜杠命令、会话列表和 Agent 初始化
agent/        对话循环、上下文压缩、工具执行与 guardrail
tools/        模型可调用工具及其权限和安全检查
tests/        单元测试
```

## Trajectory Evaluation

仓库包含独立的 `evaluation/` 评测管线，可在隔离工作区中运行编码任务，
记录模型请求、工具调用、Guardrail、上下文压缩和任务结果，并生成确定性
Checks、过程指标、LLM Rubric 与 Failure Onset 诊断。

详细命令、输出格式、指标口径和可复现实验见
[`evaluation/README.md`](evaluation/README.md)。
