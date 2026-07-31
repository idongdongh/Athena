# web_search 工具对照：hello-agent vs hermes

> 文件位置：见下方「## hermes 实现方式」「## 我的项目实现方式」两节列出的绝对路径。
>
> 结论先行：**形态完全对齐 hermes，内核是简化子集。**
> `web_search` 是「薄分发层」——本函数只解析后端、取 provider、调 `provider.search()`、序列化 JSON 字符串。

## hermes 实现方式

`web_search` 是薄分发层：解析后端 → 取 provider → 调 `provider.search()` → `json.dumps` 返回字符串。7 后端可插拔（firecrawl / parallel / tavily / exa / searxng / brave-free / ddgs），无配置默认偏好 **Firecrawl**（优先级最高且有凭证时）。选择逻辑走注册表 + `_resolve()`：按配置 → 唯一可用 → 遗留优先级（付费优先）。

- `/Users/idongdong/Documents/Projects/hermes-agent/tools/web_tools.py` — `web_search_tool` 薄分发层 + JSON 序列化（legacy 契约 `{"success":true,"data":{"web":[...]}}`）
- `/Users/idongdong/Documents/Projects/hermes-agent/agent/web_search_provider.py` — `WebSearchProvider` 抽象基类
- `/Users/idongdong/Documents/Projects/hermes-agent/agent/web_search_registry.py` — provider 注册表 + `_resolve()` 按凭证/能力解析
- `/Users/idongdong/Documents/Projects/hermes-agent/plugins/web/tavily/provider.py` — Tavily 后端
- `/Users/idongdong/Documents/Projects/hermes-agent/plugins/web/firecrawl/provider.py` — Firecrawl 后端（默认偏好）

## 我的项目实现方式

**触发**：模型在 `agent_loop` 中决定调用 `web_search`，返回 `tool_use` block（`name="web_search"`，`input={query, limit?}`）。

**调用链**：
1. `agent_loop` 收到 `tool_use`，`registry.get_entry("web_search").handler(**block.input)` 分发到 `web_search(query, limit)`；
2. `web_search` 内 `TavilyClient(api_key=TAVILY_API_KEY).search(query, max_results=limit)` 发真实搜索请求；
3. 遍历 `results`，每条映射为 `{title, url, description(=content 摘要字段), position(=index+1)}`，不设 `include_raw_content`（不拉正文）；
4. `json.dumps(..., ensure_ascii=False, indent=2)` 返回字符串；
5. 字符串作为 `tool_result` 回传模型。

固定 **Tavily 单后端**，无注册表/provider 抽象；search 永远只回元数据，不拉正文。

- `/Users/idongdong/Documents/Projects/hello-agent/tools/web_search_tool.py` — `web_search(query, limit)` + `WEB_SEARCH_TOOL` schema；`TavilyClient(api_key=...).search(query, max_results=limit)`，取 `r.get("content","")` 映射为 `description`
- `/Users/idongdong/Documents/Projects/hello-agent/conversation_loop.py` — `agent_loop` 内 `registry.get_entry(name).handler(**block.input)` 分发 + `tool_result` 回传

---

## 相同点（与 hermes 一致）

下列维度你的实现与 hermes **逐字节一致或等价**，模型调用体验无差。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 工具拆分 | `web_search` + `web_extract` 两个独立工具 | 同两个 | 一致：search 只回元数据，正文由 extract 按需拉 |
| 工具名 | `web_search` | `web_search` | 一致 |
| 主入参 | `query: str`，`limit: int`（可选） | 同 | 一致 |
| 返回结构 | `{"success": true, "data": {"web": [...]}}`（legacy 契约，逐字节一致） | 同 | 一致：模型拿到的 JSON 完全同形态 |
| 单条字段 | `title / url / description / position` | 同 | 一致 |
| 字段映射 | Tavily `content` → `description`；`position = index+1` | 同（还多一层 `or description` 兜底） | 一致：你额外兜底了空 content 的情况 |
| 只回元数据 | search 永远只回 `title/url/description`（不设 `include_raw_content`） | 同（不设 `include_raw_content`，取 `content` 摘要） | 一致：不会撑爆 token |
| 返回形态 | `json.dumps(..., ensure_ascii=False, indent=2)` 字符串 | 同 | 一致：面向模型的契约是字符串而非 dict |
| 同步性 | 同步（`web_search_tool` 及所有 provider.search 均 sync） | 同步 | 一致 |
| 失败不抛异常 | search 失败返回 `{"success":False,"error":...}`，不 raise | 同 | 一致：模型拿到结构化错误自行决定下一步 |
| search 与 extract 的关系 | 两个平级独立工具，search 不带正文 | 同 | 一致：长度问题隔离在 extract 阶段，search 天然短 |
| `limit` 非法值处理 | `int(limit)` 失败回退 5 | 同 | 一致 |
| 工具 schema 中 limit 描述 | 默认 5，范围说明依后端 | 默认 5，范围 1-20 | 一致 |

---

## 差异点（与 hermes 不同）

下列维度你的实现与 hermes 存在实质或轻微差异。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 后端数量 | **7 后端可插拔**（firecrawl / parallel / tavily / exa / searxng / brave-free / ddgs） | 固定 **Tavily 单后端** | 你无法换引擎；但少一套注册表/选择逻辑 |
| 选择逻辑 | 注册表 + `_resolve()`：配置 → 唯一可用 → 遗留优先级（付费优先） | 直接 `TavilyClient(...).search()` | 你无「按凭证自动选后端」能力，但实现更直白 |
| 无配置默认 | 偏好 **Firecrawl**（优先级最高且有凭证时） | 始终是 Tavily | 你抽不到「公开但靠 JS 渲染的 SPA」（Tavily 只拿 HTML 骨架），Firecrawl 能渲染 |
| 扩展性 | 加后端 = 新子类 + `register_provider()`，核心零改动 | 加后端需改 `web_search` 函数体 | 你牺牲了可插拔性换取简洁（符合「工具本身专注、不加注册表抽象」的要求） |
| DDGS 免费后端 | 支持（免 key，`include_raw_content` 不可用） | 仅注释示例（`pip install ddgs` 后改两行） | 你未默认启用；无 key 场景需手动切 |
| `limit` 上限 clamp | 工具层夹到 **[1, 100]**（后端实际再各自 cap，如 Tavily `min(limit,20)`） | 直接夹到 **[1, 20]** | 轻微：你的上限正好等于 Tavily 真实上限，实际无功能差 |
| 中断检查 | 调用前 `tools.interrupt.is_interrupted()`，命中返回 `{"success":False,"error":"Interrupted"}` | 无 | 你无法在长搜索中响应「取消」信号；单轮工具影响小 |
| 错误包装 | 异常包成 `"Error searching web: {e}"` + `logger.debug` | 直接返回 `str(exc)` | 你错误信息更裸，无前缀与日志 |
| 调试日志 | `DebugSession`（env `WEB_TOOLS_DEBUG=true`）记录入参/响应大小/报错到 `./logs/` | 无 | 你排查需自己加 print（如文件末尾的测试段） |
| 插件发现 | 每次调用前 `_ensure_web_plugins_loaded()` 触发注册表填充 | 无（固定 Tavily，无需发现） | 因你单后端，这一步天然不需要 |
| 密钥缺失反馈 | 后端 `is_available()` 在工具注册门控时即决定工具可见性，运行时给精确「X_API_KEY 未设置」 | 运行时 `TavilyClient` 抛异常被 `except` 接住 | 一致目标：不让 agent 崩，只是提示时机更靠后 |

---

## 结论

**形态完全对齐 hermes，内核是简化子集。**

- 对 LLM：两个 `web_search` 的「长相」基本一致——同名、同入参、同返回 JSON 结构、同样
  「只回元数据」。模型调用体验无差（见「相同点」）。
- 对你（实现者）：差异集中在 **后端可插拔性** 与 **工程护栏** 两块（见「差异点」）：
  - 你牺牲了「7 后端可插拔 + 自动选择」换来单文件直白实现，符合你提出的
    「工具本身专注、不要注册表/provider 抽象」的要求。
  - 你缺少「中断检查 / 调试日志」等护栏——对个人单轮工具影响很小，但对外暴露或长任务时值得补。

## 取舍建议（按场景判断是否要补）

| 若你遇到 | 建议动作 | 对应 hermes 能力 |
|---|---|---|
| 公开但靠 JS 渲染的页面 Tavily 搜不到好结果 | 加 **Firecrawl 双后端兜底**（Tavily 失败回落 Firecrawl） | 多后端可插拔 |
| 需要无 API key 的搜索 | 切到 DDGS（文件尾部注释已给改法） | ddgs 后端 |
| 长搜索任务需要可取消 | 加 `is_interrupted()` 检查 | 中断检查 |
| 线上/对外暴露，需可观测 | 加最小调试日志（记录入参、结果数、报错） | DebugSession |

> 注：本文件仅记录 `web_search` 的对照。抽取工具见 `web_extract_tool.md`，工具注册/发现见 `tool_registry.md`，最大步数限制见 `max_iterations.md`。
