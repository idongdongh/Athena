# 工具执行引擎对照：hello-agent vs hermes

> 文件位置：见下方「## hermes 实现方式」「## 我的项目实现方式」两节列出的绝对路径。
>
> 结论先行：**hermes 把执行拆成「单轮批量编排器 execute_tool_calls_* + 无状态单调用分派器 handle_function_call」两层，分属 agent/ 与根 model_tools.py；**
> 你的版本把两层核心合进单一 `tool_executor.py`（根目录，不在 tools/ 内），因只有一个调用面（loop），无需照搬双文件拆分。语义一致，厚壳（并发/预算/hook/桥接）全砍。

## hermes 实现方式

两层分离：

- **批量编排层** `agent/tool_executor.py` → `execute_tool_calls_sequential()` / `execute_tool_calls_concurrent()`：一个 turn 拿到 `assistant_message.tool_calls` 后，逐条或并发执行，管顺序/并行度、中间件、pre_tool_call hook、guardrail、单轮预算、显示 spinner、中断信号（`_interrupt_requested` 标志，sequential 每个工具前检查，命中跳过剩余并 `close_interrupted_tool_sequence` 收尾）、错误隔离，最后把结果回写对话。
- **单调用分派层** 仓库根 `model_tools.py` → `handle_function_call()`（约 327 行）：只处理「一个」工具调用，做类型 coerce（`coerce_tool_args`）、tool_search 桥接 + scope gate、插件 pre/post block hook、ACP/Zed 编辑审批、真正 `registry.dispatch(...)`，返回 JSON 字符串。

两文件分离的根因：**多调用面**。CLI / ~20 网关 / cron / 子代理都要无状态的 `get_tool_definitions` + `handle_function_call` 原语，但不想为这一点把整个重的 `agent/` 运行时（loop、memory、skills、providers）import 进来，所以 `model_tools.py` 刻意放仓库根做轻量界面。

- `/Users/idongdong/Documents/Projects/hermes-agent/agent/tool_executor.py` — 批量编排层（共 1538 行；sequential 约 682 行、concurrent 约 573 行）
- `/Users/idongdong/Documents/Projects/hermes-agent/model_tools.py` — `handle_function_call`（约 327 行，根目录）+ `get_tool_definitions` / `coerce_tool_args` / `registry.dispatch`
- `/Users/idongdong/Documents/Projects/hermes-agent/agent/conversation_loop.py` — 主循环，调用 `execute_tool_calls_*`；中断处理 `_interrupt_requested` / `close_interrupted_tool_sequence`
- `/Users/idongdong/Documents/Projects/hermes-agent/tools/tool_result_storage.py` — `enforce_turn_budget`（单轮字符预算，超限把最大结果持久化到沙箱+替换为短引用）

## 我的项目实现方式

**触发**：`agent_loop` 每轮 API 返回后，对 `res.content`（含 text / tool_use 块的列表）调用 `run_tool_calls(res.content, messages)`。

**调用链**：
1. `conversation_loop.py` 模块顶层 `from tool_executor import run_tool_calls`；顶层常量 `CONCURRENT_TOOLS=True`、`TOOL_MAX_WORKERS=4`；
2. `agent_loop` 每轮 `client.messages.create(...)` 拿到 `res`，若 `res.stop_reason == "tool_use"`，调 `run_tool_calls(res.content, messages, concurrent=CONCURRENT_TOOLS, max_workers=TOOL_MAX_WORKERS)`；
3. `run_tool_calls` 遍历 content，挑 `tool_use` 块，主线程按原顺序打印 `use_tool:<name>`；若 `concurrent and 本轮>1 调用` 用 `ThreadPoolExecutor` 并行派发（`as_completed` 回填、保序），否则顺序逐个 `dispatch_tool_call(name, input)`；
4. `dispatch_tool_call` 用 `registry.get_entry(name)` 查条目：无则 `"Unknown tool: ..."`；有则 `try: entry.handler(**args) except Exception as e: "Error: ..."`，把结果 `str()` 返回（失败隔离）；
5. `run_tool_calls` 收集 `tool_result`，调 `_enforce_turn_budget(results)` 对超 `TURN_BUDGET_CHARS` 的结果截断标注，再把非空结果作为一条 `role=user` 消息 `append` 回 `messages`，进入下一轮循环。

砍掉 hermes 厚壳：无 pre/post hook、无显示 spinner、无 tool_search 桥接、无 ACP 编辑审批、无类型 coerce、预算不按上下文窗口动态缩放（用固定值）。并发执行 + 单轮预算**已实现**（见差异点表）。

- `/Users/idongdong/Documents/Projects/hello-agent/tool_executor.py` — `dispatch_tool_call()` + `run_tool_calls()`（根目录，不在 `tools/` 内，避免 `discover()` 误扫）
- `/Users/idongdong/Documents/Projects/hello-agent/conversation_loop.py` — 顶层 `from tool_executor import run_tool_calls`；`agent_loop` 内 `run_tool_calls(res.content, messages, concurrent=CONCURRENT_TOOLS, max_workers=TOOL_MAX_WORKERS)` 替代原内联执行块；并加 `except KeyboardInterrupt` 兜底修复历史
- `/Users/idongdong/Documents/Projects/hello-agent/tools/registry.py` — `get_entry()`（dispatch 用它查 handler），**未改动**

---

## 相同点（与 hermes 一致）

下列维度你的实现与 hermes 语义一致。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 失败隔离 | handler 异常包成错误文本返回，不让 loop 崩 | `dispatch_tool_call` 内 `try/except` → `"Error: {e}"` | 一致：模型传错参数名不会导致 agent 崩溃 |
| 未知工具 | scope gate / dispatch 拒绝 | `get_entry` 返回 None → `"Unknown tool: ..."` | 一致：模型编造工具名时优雅降级 |
| 结果回写形态 | 每条 `tool_use` 对应一条 `tool_result`（带 `tool_use_id`） | 同（`type/tool_use_id/content`） | 一致：Anthropic 多轮工具协议格式相同 |
| 按名分发 | `registry.dispatch(name, ...)` | `registry.get_entry(name).handler(**args)` | 一致：以模型 `tool_use.name` 查表并执行 |
| 执行与 loop 分离 | executor 不在 loop 内联 | `tool_executor.py` 独立模块 | 一致：loop 只管对话节奏，执行路径结构化 |

---

## 差异点（与 hermes 不同）

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 文件布局 | 双文件：`agent/tool_executor.py` + 根 `model_tools.py` | 单文件根 `tool_executor.py` | 你仅一个调用面，无需双文件；若未来长出多调用面（网关/子代理），可拆 `dispatch` 到根、留 `run` 在 `agent/` |
| 并发执行 | `execute_tool_calls_concurrent` 线程池并行 | **已实现** `run_tool_calls(concurrent=True)`：ThreadPoolExecutor 并行派发，仅当本轮 >1 个 tool_use 才启线程，单调用走顺序路径；`conversation_loop` 默认 `CONCURRENT_TOOLS=True`；Ctrl+C 时 `cancel_futures` 优雅取消 | 已实现核心；hermes 还带显示 spinner/多模态/与顺序共享中介壳，你砍了（按别复杂） |
| 单轮预算 | `enforce_turn_budget` 上下文预算上限（按上下文窗口动态缩放；超限把**最大**结果持久化到沙箱+替换为短引用，数据不丢） | **已实现** `TURN_BUDGET_CHARS=300_000` 固定上限 + `_enforce_turn_budget()` 对溢出结果**按顺序截断+丢弃**并标注 | 核心对齐（都是字符预算）；差异：hermes 持久化保数据且优先截最大，你截断丢弃按顺序截、用固定值非动态缩放 |
| 类型 coerce | `coerce_tool_args` 把 `"42"→42` | 无，直接 `**args` | 你的 handler 多接受 str；若模型传错类型会落到 `Error` 隔离，不崩 |
| 钩子管线 | pre/post tool_call hook + guardrail + 插件 block | 无 | 你无任何钩子接缝（权限闸门未建，见后续） |
| 显示/中断 | spinner + `_set_interrupt` 多模态；sequential 每个工具前查 `_interrupt_requested`，命中跳过剩余并 `close_interrupted_tool_sequence` 收尾 | 两级 Ctrl+C：`_on_interrupt` 首按置标志+复位 SIG_DFL，loop 顶检查跳出；次按 SIG_DFL 抛 KeyboardInterrupt，`conversation_loop` 的 `except KeyboardInterrupt` 修复历史（丢未完成 tool_use + 补 assistant 占位 `[任务被用户中断]`）不崩 REPL；并发分支 Ctrl+C 时 `cancel_futures` 优雅取消 | 核心对齐（标志+跳过+收尾/兜底）；hermes 在顺序层逐工具查中断并显式关序列，你并发靠线程池 cancel、loop 顶层兜底修复历史 |
| tool_search 桥接 + scope gate | `handle_function_call` 内校验 toolset 可见性 | 无（无 toolset 概念） | 你无「按会话授权工具集」；执行期不做 scope 校验（属会话/权限层，暂挂起） |

---

## 结论

**执行语义同源，组织粒度不同。**

- 对工具「一次调用怎么派发、一轮调用怎么收尾」这条链路：两者都是
  **查表 → 调 handler → 失败隔离 → 拼 tool_result 回写对话**。你用 `tool_executor.py` 把原来塌缩在 `conversation_loop` 里的内联块抽成了独立模块，结构上对齐了 hermes 的「执行与 loop 分离」。
- 差异集中在 hermes 生产级外壳：并发、预算、钩子、桥接、显示。你按「别复杂」原则只保留核心分派，
  且因单一调用面把双文件合并为单文件，避免照搬 `model_tools.py`(根) + `tool_executor.py`(agent/) 的切分。
- 关键约束（记牢）：`tool_executor.py` **不能放进 `tools/`**，否则会被 `discover()` 扫描，
  可能引发 `conversation_loop → tool_executor → registry` 的循环 import，并模糊「什么是工具」。

## 取舍建议（按场景判断是否要补）

| 若你遇到 | 建议动作 | 对应 hermes 能力 |
|---|---|---|
| 单轮要并行跑多个工具 | ✅ 已实现：`run_tool_calls(concurrent=True)` + `ThreadPoolExecutor`（见差异点） | execute_tool_calls_concurrent |
| 长会话需控单轮 token 预算 | ✅ 已实现：`TURN_BUDGET_CHARS` + `_enforce_turn_budget`（见差异点） | 单轮预算 |
| 用户双击 Ctrl+C 中途退出 | ✅ 已实现：loop 两级 Ctrl+C + `except KeyboardInterrupt` 修复历史（见差异点「显示/中断」） | `_interrupt_requested` + `close_interrupted_tool_sequence` |
| 要做权限闸门（审批/钩子） | 在 `dispatch_tool_call` 前加 pre-hook 接缝（见权限笔记） | pre/post tool_call hook + guardrail |
| 长出网关/子代理等多调用面 | 拆 `dispatch_tool_call` 到根 `model_tools` 等价模块，留编排在 `agent/` | model_tools.py + tool_executor.py 双文件 |

> 注：本文件记录「工具执行引擎」对照。注册/发现见 `tool_registry.md`，最大步数限制见 `max_iterations.md`，权限闸门（待建）见后续笔记。
