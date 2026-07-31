# 工具注册 / 发现机制对照：hello-agent vs hermes

> 文件位置：见下方「## hermes 实现方式」「## 我的项目实现方式」两节列出的绝对路径。
>
> 结论先行：**两者都是「注册表单例 + 工具文件自注册 + AST 目录发现」三件套，机制同源；**
> 差异在 hermes 在三件套之外多了 toolset 分组、`check_fn` 可用性探测、多 provider adapter 与独立的执行/权限引擎。
> 你砍掉的都是 hermes 的厚壳，核心发现逻辑一致。

## hermes 实现方式

三件套：注册表单例 + 工具文件自注册 + AST 目录发现。中心代码不感知具体工具，加工具 = 新建文件 + 顶层 `registry.register(...)`。三件套之外还有 toolset 分组（按场景启停一组工具）、`check_fn` 可用性探测（无凭证工具不发给模型）、多 provider adapter（内部 OpenAI 格式 ↔ Anthropic/Gemini 翻译）、独立执行/权限引擎。

- `/Users/idongdong/Documents/Projects/hermes-agent/tools/registry.py` — `ToolRegistry` 单例 + `register()` + `discover_builtin_tools()` + `_module_registers_tools()`（AST 静态检测顶层 `register` 调用）
- `/Users/idongdong/Documents/Projects/hermes-agent/model_tools.py` — `get_tool_definitions()` 聚合出口（含 `check_fn` 可用性探测、toolset 过滤、30s TTL 缓存）
- `/Users/idongdong/Documents/Projects/hermes-agent/agent/conversation_loop.py` — 主循环，`tools=agent.tools` 喂给模型
- `/Users/idongdong/Documents/Projects/hermes-agent/tools/computer_use_tool.py` — 自注册 shim 示例（顶层 `registry.register(...)`）

## 我的项目实现方式

**触发**：`python conversation_loop.py` 启动（或任何 `import conversation_loop`）。模块顶层在 `discover()` 调用处一次性触发整个注册流程，之后运行时只读不写。

**调用链**：
1. 启动时 `conversation_loop.py` 模块顶层执行 `discover()`（定义在 `tools/registry.py`）；
2. `discover()` 扫描 `tools/*.py`，用 `_module_registers_tools()` 做 AST 静态检测，只对顶层（含顶层 `if` 块内）含 `registry.register(...)` 的模块 `importlib.import_module`。**下划线开头的辅助模块（如 `_path` / `_binary`）不含 `register` 调用，自动被 AST 闸门跳过——`{"__init__","registry"}` 黑名单只挡结构性身份冲突，辅助模块无需进黑名单**；
3. import 的副作用触发各工具文件顶层的 `registry.register(name, schema, handler, ...)`，把 schema + handler 写入模块级单例 `registry`；
4. `agent_loop` 每轮 API 调用用 `registry.definitions()` 取 schema 列表喂给 `tools=`；模型回 `tool_use` 时用 `registry.get_entry(block.name).handler(**block.input)` 分发执行。

砍掉 hermes 厚壳：无 toolset 分组 / check_fn 可用性探测 / 多 provider adapter / 执行引擎；所有注册工具始终可见。

- `/Users/idongdong/Documents/Projects/hello-agent/tools/registry.py` — `ToolRegistry` 单例 + `register()` + `discover()` + `_module_registers_tools()`（AST 检测，兼容顶层 `if` 块；`discover()` 黑名单注释说明辅助模块靠 AST 隔离）
- `/Users/idongdong/Documents/Projects/hello-agent/conversation_loop.py` — 模块顶层 `discover()` 触发注册；`agent_loop` 内 `registry.definitions()` 喂 `tools=` + `registry.get_entry(name).handler(**block.input)` 分发
- `/Users/idongdong/Documents/Projects/hello-agent/tools/_path.py` — 共享辅助模块：`resolve(path) -> Path`（供 read_file/write_file/patch/search_files 复用，**本身不注册**）
- `/Users/idongdong/Documents/Projects/hello-agent/tools/_binary.py` — 共享辅助模块：`BINARY_EXTS` 二进制扩展名集合（供 read_file/search_files 复用，**本身不注册**）
- `/Users/idongdong/Documents/Projects/hello-agent/tools/web_search_tool.py` — 自注册示例（末尾 `registry.register(name="web_search", ...)`）
- `/Users/idongdong/Documents/Projects/hello-agent/tools/web_extract_tool.py` — 自注册示例（末尾 `registry.register(name="web_extract", ...)`)

---

## 相同点（与 hermes 一致）

下列维度你的实现与 hermes 逐字节一致或等价，工具「从哪来、怎么被模型看到」的链路无差。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 注册表单例 | `tools/registry.py` 模块级 `registry = ToolRegistry()` 单例 | 同 | 一致：去中心化收集的基础，所有工具文件共享同一实例 |
| 自注册 | 每个工具文件顶层 `registry.register(...)` | 同（末尾 `registry.register(name=..., schema=..., handler=...)`） | 一致：工具自己声明自己，中心代码零感知 |
| AST 目录发现 | `discover_builtin_tools()` + `_module_registers_tools()` 静态 AST 判断顶层有无 `registry.register` | 同（`discover()` + `_module_registers_tools`，额外兼容顶层 `if` 块内） | 一致：裸测试段/非工具模块不会被误 import |
| 跳过非工具模块 | 跳过 `__init__` / `registry` / `mcp_tool` | 跳过 `__init__` / `registry`（黑名单只挡结构性身份冲突；下划线辅助模块 `_path`/`_binary` 靠 AST 闸门自动跳过，无需进黑名单） | 一致：非工具模块不被误导入（`common.py` 已删，`WORKDIR` 内联进 `conversation_loop.py`） |
| 导入即注册 | `importlib.import_module(...)` 触发顶层 `register` 副作用 | 同 | 一致：注册动作不需要任何人显式调用 |
| 默认扫描目录 | `registry.py` 同级目录（基于 `__file__`，不依赖 cwd） | 同（`Path(__file__).resolve().parent`） | 一致：从任意目录启动都能正确找到 `tools/` |
| schema 格式（Anthropic 端） | 原生 `{name, description, input_schema}` | 同 | 一致：传给模型的 `tools=` 形态完全相同 |
| 聚合出口 | `get_tool_definitions()` 从 registry 取 | `registry.definitions()` 从 registry 取 | 一致：加工具只改文件、不改中心 |
| 按名查执行 | `get_tool_definitions` 后由执行层按 `name` 分发 | `registry.get_entry(name).handler(**block.input)` | 一致：模型 `tool_use` 的 name 同时用于查表与分发 |

---

## 差异点（与 hermes 不同）

下列维度你的实现与 hermes 存在实质差异（多为「厚壳取舍」）。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 工具来源 | 多源聚合：核心工具 + `memory_manager.get_all_tool_schemas()` + context engine + skills/plugins | 单一 registry（仅 `tools/*.py` 自注册） | 你无插件/skill/memory 体系；小项目无需 |
| toolset 分组 | 有 toolset 概念，按场景启停一组工具 | 无（所有注册工具始终可见） | 你无法「按场景裁剪工具集」；工具少无影响 |
| 可用性探测 `check_fn` | `register(check_fn=...)`，运行时跑 `check_fn`（30s TTL 缓存 + 抖动抑制），无凭证则工具**不发给模型** | 无 `check_fn`；工具始终注册可见 | 你缺「无 key 时自动隐藏工具」；当前也无运行时权限闸门（计划建 hermes 三层机制） |
| 权限模型 | 独立 executor 层：guardrail（自动防循环）+ bash 持久化 allowlist（`shell-hooks-allowlist.json`，首次问后写盘免问）+ 路径安全集中 `check_tool_call` | **当前无**（旧 `deny`/`gate` 已删，工具裸奔；计划重建为 hermes 三层机制：guardrail + 审批 allowlist） | 你暂无任何闸门；hermes 权限重且与插件耦合 |
| 多模型适配 | 内部 OpenAI Chat Completions 格式 → adapter 翻成 Anthropic/Gemini/OpenAI | 仅 Anthropic 原生，单 SDK | 你不可切换后端；单模型场景足够 |
| 执行引擎 | 并行调用多个 tool_use、`_executing_tools` 状态机、重试、失败隔离 | 顺序 `entry.handler(**block.input)` + `try/except` 包成 `Error` | 你无并行；个人 agent 够用 |
| 提示缓存保护 | 工具 schema 变动刻意不破坏 prompt-cache 前缀 | 无该考量 | 你工具稳定时无差 |
| 入口文件结构 | `run_agent.py`（CLI 入口，"handles the conversation loop"）+ `agent/conversation_loop.py`（循环函数 `run_conversation`） | 单文件 `conversation_loop.py`（入口兼循环） | 你未拆 CLI 与循环；小项目清晰够用 |
| `discover` 触发方式 | 每次调用前 `_ensure_web_plugins_loaded()` 触发插件注册表填充 | 启动时一次 `discover()`，之后静态 | 因你无插件体系，无需每次重扫 |

---

## 结论

**发现机制同源，抽象层级不同。**

- 对工具「怎么被收集、怎么出现在模型视野」这条链路：两者都是
  **单例 registry + 文件自注册 + AST 把关的目录扫描**，加新工具都是「新建 `tools/xxx_tool.py` →
  末尾 `registry.register(...)` 一句 → 中心代码零改动」。这正是你之前对齐 hermes 的核心价值。
- 对你（实现者）：差异集中在 hermes 在三件套之外补的四层壳——**toolset 分组、check_fn 可用性探测、
  多 provider adapter、独立执行/权限引擎**（见「差异点」）。你按「别复杂」原则只保留了注册表内核：
  toolset / check_fn / adapter 对当前规模属过度工程；权限引擎则已清空旧的 `deny`/`gate`，
  计划重建为 hermes 三层机制（guardrail + 审批 allowlist），见后续。

## 取舍建议（按场景判断是否要补）

| 若你遇到 | 建议动作 | 对应 hermes 能力 |
|---|---|---|
| 接入插件 / skill / 长期记忆，需要工具来源不止 `tools/` | 仿 `get_tool_definitions` 做多源聚合 | 多源聚合 |
| 想「无 API key 时自动让工具隐身」而非运行时报错 | 给 `ToolEntry` 加 `check_fn`，`definitions()` 过滤 | check_fn 可用性探测 |
| 工具数变多，想按场景（如「只读模式」）启停一组 | 加 toolset 字段 + `discover(toolset=...)` 过滤 | toolset 分组 |
| 需要同一 agent 跑多家模型 | 引入 OpenAI 风格内部格式 + adapter 翻译 | 多 provider adapter |
| 单次要并行调用多个工具 | 在 `agent_loop` 加并行分发（如 `asyncio.gather`） | 并行执行引擎 |

> 注：本文件记录「工具注册/发现」对照。搜索工具见 `web_search_tool.md`，抽取工具见 `web_extract_tool.md`，最大步数限制见 `max_iterations.md`。
