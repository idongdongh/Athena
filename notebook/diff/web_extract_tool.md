# web_extract 工具对照：hello-agent vs hermes

> 文件：`hello-agent/tools/web_extract_tool.py` vs `hermes-agent/tools/web_tools.py` + `plugins/web/*/provider.py`
>
> 结论先行：**形态与摘要手段均已对齐 hermes；后端与安全层仍是简化子集**。
> 本轮迭代把"长度控制 = 截断"改为"分级 LLM 摘要 + 失败降级"，与 hermes `process_content_with_llm` 完全一致。

---

## 相同点（与 hermes 一致）

下列维度你的实现与 hermes 逐字节一致或等价，模型调用体验无差。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 工具拆分 | `web_search` + `web_extract` 两工具独立 | 同 | 一致 |
| 工具名 | `web_extract` | `web_extract` | 一致 |
| 入参 | `urls` 数组（`maxItems=5`），可选 `max_chars` | 同 | 一致 |
| 返回形态 | JSON 字符串给模型 | 同 | 一致 |
| 失败处理 | `failed_results` / `failed_urls` 合并到同一 `results` 列表，失败项带 `error` 字段 | 同 | 一致 |
| 摘要手段 | LLM 摘要压到 `max_chars`（保留关键信息，避免截断丢尾部） | 同 | 一致（本轮对齐） |
| 分级阈值 | `<5000` 原样 / `5000–500k` 单遍摘要 / `500k–2M` 分块（100k/块）摘要后合成 / `>2M` 拒绝 | 同 | 一致 |
| 失败降级 | LLM 摘要超时/报错 → 回退「截断 + 提示」，不抛异常 | 同（`_trim` 兜底） | 一致 |
| 摘要模型 | 独立 auxiliary model（独立配置，可走更便宜的小模型） | `SUMMARY_MODEL_ID` 覆盖主模型；未设则复用 `MODEL_ID` | 你默认用主模型做摘要（成本/延迟稍高），建议在 `.env` 设 haiku 档作 `SUMMARY_MODEL_ID` |
| 去 base64 图片 | `clean_base64_images()`（摘要前调用） | `_strip_base64_images()`（同位置） | 一致 |
| 超大页面拒绝 | `>2MB` 拒绝 | 同（`REFUSE_CHARS = 2_000_000`） | 一致 |

> 本轮对齐要点：摘要手段从「截断到 `max_chars`」改为「分级 LLM 摘要压到 `max_chars`」，与 hermes `process_content_with_llm` 完全对齐；阈值 `5000 / 500k / 2M` 三档一致，超长走分块（100k/块）后合成再收口；摘要异常自动回退 `_trim`，绝不向模型抛错。

---

## 差异点（与 hermes 不同）

下列维度你的实现与 hermes 存在实质差异。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 后端 | 7 后端可插拔，无配置默认偏好 **Firecrawl** | 固定 **Tavily** 单后端 | 你抽不到「公开但靠 JS 渲染的 SPA」（Tavily 只拿 HTML 骨架）；Firecrawl 能渲染 |
| extract 内核 | 默认 Firecrawl `scrape`（headless 浏览器级，JS 渲染 + 代理抗反爬） | Tavily `extract`（服务端 fetch） | 强反爬/JS 渲染页面你弱；登录墙页面两者都拿不到 |
| 异步模型 | `web_extract` 是 `async`；同步 provider 走 `asyncio.to_thread`，不阻塞事件循环；单 URL 60s 超时 | 同步 + `timeout=30` | 你阻塞事件循环（个人 agent 小项目可忽略） |
| 网站准入策略 | `check_website_access(url)` 抓取前拦截黑名单/恶意站点；抓取后对已重定向最终 URL 再查一次 | 无 | 你不拦截恶意/黑名单站点 |
| SSRF 防护 | `async_is_safe_url()` 过滤私有/内网地址 | 无 | 你少一层内网探测防护 |
| URL 嵌入密钥拦截 | `_PREFIX_RE` 扫描原始 URL 及其百分号解码形态，命中即拒绝 | 无 | 你少一层防数据外泄防护 |
| 超时兜底提示 | 提示「改用 `browser_navigate`」 | 提示「换更聚焦的 URL / 先用 web_search」 | 你无浏览器兜底层（对个人 agent 通常够用） |

---

## 未对齐项与取舍

| 维度 | 取舍 |
|---|---|
| 后端固定 vs 可插拔 | 专注工具本身，不引入注册表/provider 抽象；若常遇到 JS 渲染的公开页面再考虑加 Firecrawl 双后端 |
| 异步模型 | 小项目同步足够；接入异步主循环时再改 |
| 安全层（SSRF / 策略 / 密钥拦截） | 个人本地 agent 默认信任请求源；如对外暴露（MCP / 多用户）再补这层 |
| 超时兜底提示 | 无浏览器工具可兜底，所以改成「换 URL / 先用 web_search」提示，符合你的工具集 |
