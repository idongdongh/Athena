# 最大步数限制对照：hello-agent vs hermes

> 文件位置：见下方「## hermes 实现方式」「## 我的项目实现方式」两节列出的绝对路径。
>
> 结论先行：**核心机制一致——默认 90 步上限 + 触顶后去掉 tools 做一次总结调用优雅收尾；**
> 差异在 hermes 多了线程安全预算对象、subagent 独立预算、execute_code refund、grace call、边界防御与任务调度集成。
> 你砍掉的都是 hermes 的厚壳，核心防失控逻辑一致。

## hermes 实现方式

默认 `max_iterations = 90`。主循环双闸门：`api_call_count < max_iterations AND iteration_budget.remaining > 0`（外加 `_budget_grace_call` 逃生口）。触顶调 `_handle_max_iterations` 做「去掉 tools 的总结调用」优雅收尾（注入 user 消息引导总结 + 失败 fallback 文案）。预算用 `IterationBudget` 线程安全类（`consume`/`refund`/`remaining`）；subagent 独立预算 `delegation.max_iterations` 默认 50；`execute_code` 轮次 `refund()` 不计预算；临近 `max_iterations-1` 出错记 `error_near_max_iterations` 特殊处理；kanban worker 预算耗尽记 `timed_out` + 熔断计数。

- `/Users/idongdong/Documents/Projects/hermes-agent/agent/agent_init.py` — `max_iterations` 默认 90（agent 构造参数，`config.yaml` 可调）
- `/Users/idongdong/Documents/Projects/hermes-agent/agent/conversation_loop.py` — 主循环 while 条件（双闸门 + `_budget_grace_call`），`api_call_count += 1` 在循环顶部
- `/Users/idongdong/Documents/Projects/hermes-agent/agent/turn_finalizer.py` — 触顶处理：调 `_handle_max_iterations` + kanban worker 信令（`timed_out`）
- `/Users/idongdong/Documents/Projects/hermes-agent/agent/iteration_budget.py` — `IterationBudget` 线程安全类（`consume`/`refund`/`remaining`，带 `threading.Lock`）
- `/Users/idongdong/Documents/Projects/hermes-agent/agent/chat_completion_helpers.py` — `_handle_max_iterations` 实现（注入 user 消息 + 失败 fallback 文案）

## 我的项目实现方式

**触发**：用户在 `__main__` 交互循环输入 query → `history.append({"role":"user",...})` → 调 `agent_loop(history)`，步数限制从此刻起作用。

**调用链**：
1. `agent_loop` 初始化 `api_call_count = 0`；
2. `while api_call_count < MAX_ITERATIONS:`：每轮顶部 `api_call_count += 1`，调 `client.messages.create(tools=registry.definitions())`；
3. **正常出口**：`res.stop_reason != "tool_use"` → `return`（模型给文字回复即结束，不触顶）；
4. 否则执行工具、append `tool_result`，回到 while 顶部继续；
5. **触顶**：`api_call_count` 到 90，while 条件不满足退出 → 打印 `⚠️ Iteration budget exhausted (90/90) — requesting summary...` → 做一次**不带 `tools=`** 的 `messages.create(max_tokens=MAX_TOKENS_SUMMARY)` 让模型总结（默认 4096，对齐 hermes `or 4096` 兜底） → append assistant；
6. `agent_loop` 返回，`__main__` 打印最后一条 assistant 的 text。

`MAX_ITERATIONS = 90` 模块常量；单闸门裸计数；触顶总结不注入 user 消息（保持 role 严格交替）；砍掉 `IterationBudget` / subagent 预算 / `execute_code` refund / grace call / 边界防御 / kanban 集成等厚壳。

- `/Users/idongdong/Documents/Projects/hello-agent/conversation_loop.py` — `MAX_ITERATIONS = 90` 常量 + `agent_loop(messages)`（计数 while + 触顶 summary 块）+ `__main__` 交互入口

---

## 相同点（与 hermes 一致）

下列维度你的实现与 hermes 逐字节一致或等价，防无限循环的核心行为无差。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 默认上限 | `max_iterations = 90`（`agent_init.py:177`） | `MAX_ITERATIONS = 90`（模块常量） | 一致：90 步触发收尾 |
| 计数口径 | 一次迭代 = 一次 API 调用，`api_call_count += 1` 在循环顶部 | 同（`api_call_count += 1` 在 while 顶部） | 一致：统计的是 API 调用轮次 |
| 正常出口 | `stop_reason != "tool_use"` 时退出循环 | 同（`return`） | 一致：模型给文字回复即结束 |
| 触顶不硬崩 | 调 `_handle_max_iterations`，做一次**去掉 tools** 的总结调用（`turn_finalizer.py:53-70`） | 同（while 退出后做一次不带 `tools=` 的 `messages.create`） | 一致：优雅收尾而非死循环中途崩 |
| 触顶目的 | 让模型总结「做到了哪一步」，给用户一个交代 | 同 | 一致：避免 agent 无声烧 token |

---

## 差异点（与 hermes 不同）

下列维度你的实现与 hermes 存在实质差异（多为「厚壳取舍」）。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 上限闸门 | 双闸门：`api_call_count < max_iterations` **AND** `iteration_budget.remaining > 0`（`conversation_loop.py:612`） | 单闸门：`api_call_count < MAX_ITERATIONS` | 你少一层预算对象；单进程下等价 |
| 预算对象 | `IterationBudget` 线程安全类（`consume`/`refund`/`remaining`，带 `threading.Lock`，`iteration_budget.py`） | 裸 `int` 计数器 | 你非线程安全；单线程 agent 无影响 |
| 可配置性 | `max_iterations` 是 agent 构造参数 + `config.yaml` 可调 | 硬编码模块常量，改值改源码 | 你不可运行时配置；个人项目够用 |
| subagent 预算 | 每个 subagent 独立预算，`delegation.max_iterations` 默认 50 | 无 subagent 体系 | 不适用 |
| execute_code refund | 程序化工具调用轮次 `refund()` 不计入预算（`iteration_budget.py:28`） | 无此机制 | 你所有调用都计步；无程序化工具调用场景 |
| 逃生口 | `_budget_grace_call` 允许触顶后多走一次 | 无 | 你触顶即收尾；hermes 多一次容错 |
| 触顶总结实现 | `_handle_max_iterations` **注入 user 消息**引导总结 + 失败 fallback 文案（`chat_completion_helpers.py:1410/1635`） | 直接去 tools 调用，不注入消息、不包 try | 你更简；网络抖动时总结调用会抛异常（hermes 有 fallback） |
| 触顶 max_tokens | `agent.max_tokens or 4096`（用户可配 + 兜底，`chat_completion_helpers.py:625`） | `MAX_TOKENS_SUMMARY = int(os.getenv("MAX_TOKENS_SUMMARY", "4096"))`（模块常量 + env 覆盖，默认 4096） | 一致：兜底 4096 给你够用的总结输出空间；都允许用户覆盖 |
| 边界防御 | 临近 `max_iterations-1` 出错记 `error_near_max_iterations` 特殊处理（`conversation_loop.py:4869`） | 无 | 你边界偶发错误会正常计步；影响小 |
| 任务调度集成 | kanban worker 预算耗尽记 `timed_out` 失败 + 熔断计数（`turn_finalizer.py:85-122`） | 无 | 你无任务调度体系；不适用 |
| 单轮工具结果总大小（另一维度） | `enforce_turn_budget` 限制单轮 tool_result **总字符数**，超限把最大的结果持久化到沙箱并替换为短引用（`tool_result_storage.py:181` / 调用处 `tool_executor.py:846`） | **已实现** `_enforce_turn_budget`：`TURN_BUDGET_CHARS=300_000` 字符上限，超限按顺序截断+标注（见 `tool_executor.md`） | 核心对齐（都是字符预算）；hermes 持久化+引用（数据不丢）且优先截最大，你截断丢弃、按顺序截 |

> 注意：「单轮工具结果总大小」(`enforce_turn_budget`) 与「总迭代步数上限」(`max_iterations`) 是两个不同维度。本笔记讨论的是后者；前者见 `tool_executor.md`。

---

## 结论

**防失控的核心一致，预算治理的厚壳不同。**

- 对「防止 agent 陷入无限工具调用循环」这一核心目标：两者都是 **90 步硬上限 + 触顶去 tools 做总结收尾**，
  行为等价。模型再怎么反复调工具，最多 90 轮就会被强制收尾，不再无声烧 token。
- 对你（实现者）：差异集中在 hermes 在核心之外补的预算治理层——**线程安全预算对象、subagent 独立预算、
  execute_code refund、grace call、边界防御、kanban 调度集成**（见「差异点」）。这些服务多线程/多 agent/
  可配置/任务调度场景，你单进程单模型用不上，按「别复杂」原则已砍掉。

## 取舍建议（按场景判断是否要补）

| 若你遇到 | 建议动作 | 对应 hermes 能力 |
|---|---|---|
| 触顶总结调用偶发网络抖动报错 | 给总结调用包 `try/except`，失败时回退一句文案 | `_handle_max_iterations` fallback |
| 想运行时调步数（如简单任务少给步数） | `MAX_ITERATIONS` 改成 `agent_loop(messages, max_iterations=90)` 参数 | `max_iterations` 构造参数 |
| 引入 subagent / 委托执行 | 抽 `IterationBudget` 类，subagent 独立预算(默认 50) | `delegation.max_iterations` + `IterationBudget` |
| 接入后台任务调度（kanban 类） | 预算耗尽记 `timed_out` + 熔断计数 | kanban worker 信令 |
| 单轮工具结果过多撑爆 token | 加单轮 tool_result 数上限 | `enforce_turn_budget` |

> 注：本文件记录「最大步数限制」对照。工具注册/发现见 `tool_registry.md`，搜索工具见 `web_search_tool.md`，抽取工具见 `web_extract_tool.md`。
