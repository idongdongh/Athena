# 工具清单对照：hello-agent vs hermes

> 文件位置：见下方「## hermes 实现方式」「## 我的项目实现方式」两节列出的绝对路径。
>
> 结论先行：**hermes 有 67 个核心工具、28 个 toolset，是庞大生态；你的项目已补齐 coding 地基——7 个工具（file 四件套 + bash + web search/extract），无 toolset 分组。**
> 两者共享同一套注册/发现机制（见 `tool_registry.md`），差异在「工具数量」与「治理厚壳」（toolset 分组 / check_fn 可用性探测 / 多源聚合）。

## hermes 实现方式

工具由 `tools/*.py` 各文件顶层 `registry.register(name=..., toolset=..., check_fn=...)` 自注册，`model_tools.py::get_tool_definitions()` 聚合并按 toolset 过滤、跑 `check_fn` 可用性探测（无凭证工具不发给模型）。共 **67 个核心工具，28 个 toolset**。按对 coding agent 的相关度分组如下：

**🟢 coding 地基 (9)** — `file`: read_file, write_file, patch, search_files ｜ `terminal`: terminal, process, read_terminal, close_terminal ｜ `code_execution`: execute_code

**🟡 浏览器 (12)** — `browser` (10): navigate, snapshot, click, type, scroll, back, press, get_images, vision, console ｜ `browser-cdp` (2): browser_cdp, browser_dialog

**🟡 记忆/上下文 (2)** — `memory`: memory（长期记忆）｜ `session_search`: session_search（搜历史会话）

**🟡 协作/任务 (16)** — `kanban` (9): show, list, complete, block, heartbeat, comment, create, unblock, link ｜ `todo`: todo ｜ `project` (3): list, create, switch ｜ `delegation`: delegate_task（子 agent）｜ `cronjob`: cronjob ｜ `clarify`: clarify（向用户追问）

**🟡 技能系统 (3)** — `skills`: skill_manage, skills_list, skill_view

**🔵 多模态 (5)** — `image_gen`: image_generate ｜ `video_gen`: video_generate ｜ `video`: video_analyze ｜ `vision`: vision_analyze ｜ `tts`: text_to_speech

**⚪ 平台集成 (18)** — `discord`/`discord_admin` ｜ `feishu_doc`: feishu_doc_read ｜ `feishu_drive` (4): 评论相关 ｜ `homeassistant` (4): ha_list_entities/get_state/list_services/call_service ｜ `computer_use`: computer_use ｜ `x_search`: x_search ｜ `yb_*` (5): 元宝 query_group_info/members, send_dm, search/send_sticker

**✅ web (2)** — `web`: web_search, web_extract

- `/Users/idongdong/Documents/Projects/hermes-agent/tools/registry.py` — `ToolRegistry` 单例 + `register(toolset=, check_fn=)` + `discover_builtin_tools()`
- `/Users/idongdong/Documents/Projects/hermes-agent/model_tools.py` — `get_tool_definitions()` 聚合出口（toolset 过滤 + check_fn 可用性探测 + 30s TTL 缓存）
- `/Users/idongdong/Documents/Projects/hermes-agent/tools/` — 67 个工具分散在各 `*.py`（如 `file_tools.py`、`browser_tool.py`、`code_execution_tool.py`、`memory_tool.py`、`kanban_tools.py`）

## 我的项目实现方式

**触发**：`python conversation_loop.py` 启动（或 `import conversation_loop`）。模块顶层 `discover()` 一次性扫描注册，之后运行时只读。

**调用链**：
1. 启动时 `conversation_loop.py` 模块顶层执行 `discover()`（定义在 `tools/registry.py`）；
2. `discover()` 扫描 `tools/*.py`，AST 检测顶层（含 `if` 块内）的 `registry.register(...)`，命中的模块才 `importlib.import_module`；
3. import 副作用触发各工具文件顶层 `registry.register(name, schema, handler)`，写入单例 `registry`；
4. `agent_loop` 每轮用 `registry.definitions()` 喂 `tools=`；模型回 `tool_use` 时 `registry.get_entry(name).handler(**block.input)` 分发。

当前 **7 个工具，无 toolset 分组**：file 四件套（`read_file`/`write_file`/`patch`/`search_files`）+ `bash` + `web_search`/`web_extract`。砍掉 hermes 厚壳：无 toolset 分组 / check_fn / 多源聚合 / 执行引擎。

- `/Users/idongdong/Documents/Projects/hello-agent/tools/registry.py` — `ToolRegistry` 单例 + `register(name, schema, handler, toolset)` + `discover()` + `_module_registers_tools()`（AST 检测）
- `/Users/idongdong/Documents/Projects/hello-agent/conversation_loop.py` — 模块顶层 `discover()` 触发注册；`agent_loop` 内 `registry.definitions()` 喂 `tools=` + `get_entry().handler()` 分发
- `/Users/idongdong/Documents/Projects/hello-agent/tools/read_file_tool.py` — `read_file`（行号+分页，纯 Python pathlib）
- `/Users/idongdong/Documents/Projects/hello-agent/tools/write_file_tool.py` — `write_file`（覆盖写+自动建父目录）
- `/Users/idongdong/Documents/Projects/hello-agent/tools/patch_tool.py` — `patch`（定点 find-and-replace + unified diff）
- `/Users/idongdong/Documents/Projects/hello-agent/tools/search_files_tool.py` — `search_files`（正则搜内容 / glob 找文件）
- `/Users/idongdong/Documents/Projects/hello-agent/tools/bash_tool.py` — `bash`（subprocess.run 一次性无状态）
- `/Users/idongdong/Documents/Projects/hello-agent/tools/web_search_tool.py` — `web_search`（Tavily 单后端，只回元数据）
- `/Users/idongdong/Documents/Projects/hello-agent/tools/web_extract_tool.py` — `web_extract`（Tavily extract + 分级 LLM 摘要）

---

## 相同点（与 hermes 一致）

下列维度你的实现与 hermes 逐字节一致或等价，工具「从哪来、怎么被模型看到」的链路无差。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 注册机制 | 注册表单例 + 文件自注册 + AST 目录发现 | 同（见 `tool_registry.md`） | 一致：加工具只改文件、不改中心 |
| web 工具形态 | `web_search` / `web_extract` 两工具 | 同两个，且形态已对齐 | 一致：模型调用体验无差 |
| file 四件套名称 | `read_file` / `write_file` / `patch` / `search_files` | 同四个，名称完全对齐 | 一致：模型调用同名工具 |
| schema 格式 | Anthropic 原生 `{name, description, input_schema}` | 同 | 一致：传给模型的 `tools=` 形态相同 |
| 按名查执行 | 执行层按 `name` 分发 | `registry.get_entry(name).handler(**input)` | 一致 |

---

## 差异点（与 hermes 不同）

下列维度你的实现与 hermes 存在实质差异（多为「厚壳取舍」）。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 工具数量 | **67 个核心工具** | **7 个**（file 四件套 + bash + web search/extract） | 你仍缺 code_execution/browser/memory/任务协作/多模态/平台集成 |
| toolset 分组 | 28 个 toolset，按场景启停一组工具 | 无（所有注册工具始终可见） | 你无法「按场景裁剪工具集」；工具少无影响 |
| 可用性探测 `check_fn` | `register(check_fn=...)`，无凭证工具不发给模型 | 无；工具始终可见 | 你缺「无 key 时自动隐藏工具」 |
| 工具来源 | 多源聚合：核心工具 + memory + skills + plugins | 单一 registry（仅 `tools/*.py`） | 你无插件/skill/memory 体系 |
| file 工具 | `read_file`/`write_file`/`patch`/`search_files`，走 **shell(sed/wc/cat)** 后端 | 已实现同名四件套，走**纯 Python pathlib** 后端 | 行为等价（分页/行号/diff/搜索）；hermes 壳砍：9 策略模糊匹配、V4A 多文件 patch、CRLF/BOM、lint/LSP |
| shell 工具 | `terminal`(持久 PTY) + `process`/`read_terminal`/`close_terminal`（管后台进程） | 已实现 `bash`（subprocess.run 一次性无状态） | 你无持久会话/`cd` 跨步保留/交互式/后台进程；coding agent 90% 命令够用，详见 `tool_inventory.md` 决策 |
| 代码执行 | `execute_code`（程序化工具调用，refund 不计步） | 无 | 你无沙箱代码执行（可复用 bash） |
| 浏览器 | 12 个 browser 工具（含 vision/console/cdp） | 无 | 你无浏览器自动化 |
| 记忆 | `memory`（长期）+ `session_search`（历史会话） | 无 | 你无跨会话记忆（见方向 B 讨论） |
| 任务/协作 | kanban(9) + todo + project + delegation + cronjob + clarify | 无 | 你无任务调度/委派/澄清 |
| 多模态 | image_gen / video_gen / video / vision / tts | 无 | 你无多模态 |
| 平台集成 | discord / feishu / homeassistant / computer_use / x_search / yb | 无 | 个人 agent 大概率不需要 |

---

## 结论

**工具生态规模仍悬殊，但 coding 地基已对齐，注册机制同源。**

- 对「工具怎么被收集、怎么出现在模型视野」这条链路：两者完全同源（注册表 + 自注册 + AST 发现，见 `tool_registry.md`）。差异不在机制，在**库存**。
- 对你（实现者）：hermes 的 67 个工具是长期演进的产物，覆盖 coding / 浏览器 / 记忆 / 任务 / 多模态 / 平台集成全谱。你按「别复杂」补齐了 coding 地基（file 四件套 + bash，核心行为对齐 hermes，后端用纯 Python 等价实现、壳砍掉），其余能力按需补。

## 取舍建议（按 coding agent 优先级补）

| 若你遇到 | 建议动作 | 对应 hermes 工具 |
|---|---|---|
| ~~agent 要能读写/改文件~~ | ✅ 已实现 read_file/write_file/patch/search_files | file (4) |
| ~~agent 要能执行命令~~ | ✅ 已实现 bash（选 bash 而非 terminal，砍 PTY 厚壳） | terminal (4) |
| 当前安全缺口：bash 能跑 rm -rf、文件能改绝对路径 | **接权限闸门（hermes 三层：guardrail + 审批 allowlist）** | guardrail + shell_hooks allowlist |
| agent 要能跑代码沙箱 | 加 `execute_code`（可复用 bash，或独立沙箱） | code_execution |
| 需要浏览器交互/登录页 | 加 `browser` 工具集（最重，最后做） | browser (10) + browser-cdp (2) |
| 活动量够了，要跨会话记忆 | 加 `memory` + `session_search` | memory + session_search |
| 要委派子任务 / 追问澄清 | 加 `delegate_task` / `clarify` | delegation / clarify |

> 推荐补全顺序：~~file + bash（地基）~~ ✅ → **权限闸门（hermes 三层，当前最高优先）** → code_execution → browser → memory → 其余按需。
>
> 注：本文件记录「工具清单」对照。工具注册/发现机制见 `tool_registry.md`，web_search 见 `web_search_tool.md`，web_extract 见 `web_extract_tool.md`，最大步数限制见 `max_iterations.md`。
