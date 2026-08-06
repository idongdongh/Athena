"""web_extract_tool.py — 借鉴 hermes web_extract 思路的网页抽取工具。

与 web_search 分离（hermes 也是两个独立工具）：
- web_search 只回元数据（标题/链接/摘要），永不返回正文，所以不会撑爆 token；
- web_extract 才真正抓取页面正文，并按长度分级控制（与 hermes 一致）：
    * < 5000 字符：在调用方要求的 max_chars 内原样返回
    * 5000 – 500k：单遍 LLM 摘要压到 max_chars（最大 5000）
    * 500k – 2M：切成 100k/块，逐块摘要后合成，再压到 max_chars
    * > 2M：拒绝，提示换更聚焦的 URL
- 摘要失败 → 自动回退到截断（永不报错给 LLM），与 hermes 失败降级一致；
- 清掉内嵌 base64 图片（最烧 token 的噪声），与 hermes 一致。

后端用你项目已有的 Tavily extract，无需新依赖；摘要复用 conversation_loop.py 的
Anthropic 客户端风格（BASE_URL + API_KEY + MODEL_ID），并允许 SUMMARY_MODEL_ID
覆盖（贴近 hermes auxiliary model 思路，用更便宜的小模型做摘要）。
"""

import json
import os
import re
import sys

try:
    from tools.registry import registry
except ImportError:
    registry = None

from tavily import TavilyClient
from dotenv import load_dotenv
from agent.interrupt_controller import (
    ToolExecutionCancelled,
    interrupt_controller,
    run_interruptible,
)
# 显式定位项目根 .env，避免从不同 cwd 运行时找不到
_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_here, "..", ".env"), override=True)


# 分级阈值（与 hermes process_content_with_llm 完全一致，单位：字符）
MIN_SUMMARY_CHARS = 5_000      # 小于此值不调用摘要模型，仍受 max_chars 限制
CHUNK_THRESHOLD = 500_000      # 大于此值走分块摘要
CHUNK_SIZE = 100_000           # 分块大小
MAX_OUTPUT_CHARS = 5_000       # 单页输出硬上限
MAX_SUMMARY_TOKENS = 8_192     # 摘要模型单次输出 token 上限
REFUSE_CHARS = 2_000_000       # 超过约 2MB 直接拒绝
MAX_URLS = 5                   # 单次最多抽取 URL 数（对应 hermes 的 maxItems=5）

_BASE64_IMG_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^;]+;base64,[^)]+\)")


def _strip_base64_images(text: str) -> str:
    """去掉内嵌 base64 图片，借鉴 hermes clean_base64_images。"""
    return _BASE64_IMG_RE.sub("", text)


def _trim(text: str, max_chars: int) -> str:
    """把正文和截断提示一起限制在 max_chars 内。"""
    if len(text) <= max_chars:
        return text
    suffix = (
        f"\n\n[... 内容已截断：仅显示前 {max_chars:,} / 共 {len(text):,} 字符。"
        "如需更多，请换更聚焦的 URL 或用 web_search 获取摘要 ...]"
    )
    if len(suffix) >= max_chars:
        return text[:max_chars]
    return text[:max_chars - len(suffix)] + suffix


_summary_client = None  # lazy singleton：长文分块时复用同一客户端，避免每块新建连接


def _get_summary_client():
    """复用 conversation_loop 的 Anthropic 客户端（lazy import + 模块级缓存）。

    用户可在 .env 设 SUMMARY_MODEL_ID 指定更便宜的辅助模型，否则默认复用主模型 MODEL_ID。
    这就是 hermes 的 auxiliary model 思路。
    """
    global _summary_client
    if _summary_client is not None:
        return _summary_client
    from anthropic import Anthropic
    # 兼容项目里两套命名：API_KEY / BASE_URL（conversation_loop）与
    # ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL（其他脚本）
    api_key = os.getenv("API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY / ANTHROPIC_API_KEY 未设置，无法调用摘要模型")
    base_url = os.getenv("BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
    _summary_client = Anthropic(
        base_url=base_url,
        api_key=api_key,
    )
    return _summary_client


def _summarize_chunk(text: str, max_chars: int) -> str:
    """对一段正文做单次 LLM 摘要（hermes 核心做法）。失败抛异常让上层走降级。"""
    client = _get_summary_client()
    model = os.getenv("SUMMARY_MODEL_ID") or os.getenv("MODEL_ID", "")
    if not model:
        raise RuntimeError("MODEL_ID 未设置，无法调用摘要模型")
    def cancel_summary_request() -> None:
        global _summary_client
        try:
            client.close()
        finally:
            if _summary_client is client:
                _summary_client = None

    resp = run_interruptible(
        lambda: client.messages.create(
            model=model,
            # max_chars 是字符预算，不可直接当 token 数。Hermes 的摘要调用使用固定上限；
            # 这里按请求大小缩放，但始终限制在模型可接受的单次输出范围内。
            max_tokens=min(MAX_SUMMARY_TOKENS, max(256, max_chars)),
            messages=[{
                "role": "user",
                "content": (
                    f"请将以下网页正文总结为简洁的 markdown 要点列表，保留关键事实、数字、"
                    f"人名、日期和主要结论。输出必须不超过 {max_chars} 字符。\n\n"
                    f"---BEGIN CONTENT---\n{text}\n---END CONTENT---"
                ),
            }],
        ),
        on_cancel=cancel_summary_request,
        thread_name="web-summary-request",
    )
    summary = "".join(getattr(b, "text", "") for b in resp.content)
    return summary.strip()[:max_chars]


def _summarize_long_text(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """按 hermes 的分级策略做摘要：单遍 / 分块后合成。失败抛异常。"""
    if len(text) <= CHUNK_THRESHOLD:
        return _summarize_chunk(text, max_chars)
    # 长文分块（每块 100k，hermes 行为）
    parts = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    chunk_summaries = []
    for part in parts:
        interrupt_controller.raise_if_requested("web_extract interrupted by user")
        chunk_summaries.append(_summarize_chunk(part, max_chars))
    combined = "\n\n".join(chunk_summaries)
    # 合成后通常仍超 max_chars，再做一次收口摘要
    if len(combined) > max_chars:
        combined = _summarize_chunk(combined, max_chars)
    return combined[:max_chars]


def web_extract(urls, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """抽取指定 URL 的页面正文，返回结构化 JSON 字符串。

    Args:
        urls: 一个 URL 字符串，或 URL 列表（最多 5 个）
        max_chars: 每个页面返回的正文上限，范围 1-5000
    """
    if isinstance(urls, str):
        urls = [urls]
    urls = urls[:MAX_URLS]
    if not urls:
        return json.dumps({"success": False, "error": "urls 不能为空"}, ensure_ascii=False)

    try:
        max_chars = int(max_chars)
    except (TypeError, ValueError):
        max_chars = MAX_OUTPUT_CHARS
    max_chars = min(max(max_chars, 1), MAX_OUTPUT_CHARS)

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    close_client = getattr(client, "close", lambda: None)
    try:
        resp = run_interruptible(
            lambda: client.extract(urls=urls, timeout=30),
            on_cancel=close_client,
            thread_name="web-extract-request",
        )
    except ToolExecutionCancelled:
        raise
    except Exception as exc:
        msg = str(exc)
        if "timed out" in msg.lower():
            msg = "抽取超时(>30s)：页面可能过大或无响应，建议换更聚焦的 URL，或先用 web_search 取摘要。"
        return json.dumps({"success": False, "error": msg}, ensure_ascii=False)
    finally:
        close_client()

    results = []
    for item in resp.get("results", []):
        interrupt_controller.raise_if_requested("web_extract interrupted by user")
        url = item.get("url", "")
        content = item.get("raw_content", "") or item.get("content", "")

        if len(content) > REFUSE_CHARS:
            refused = f"[页面过大（>{REFUSE_CHARS // 1_000_000}MB），已拒绝；请换更聚焦的来源]"
            results.append({
                "url": url,
                "content": _trim(refused, max_chars),
                "truncated": True,
            })
            continue

        content = _strip_base64_images(content)

        # 分级控制（与 hermes process_content_with_llm 完全一致）：
        #   < 5000       原样返回
        #   5000 – 500k  单遍 LLM 摘要
        #   500k – 2M    分块摘要后合成
        # 摘要失败 → 降级到 _trim，绝不把异常抛给模型。
        if len(content) <= MIN_SUMMARY_CHARS:
            final = _trim(content, max_chars)
            was_summarized = False
            is_truncated = len(content) > max_chars
        else:
            try:
                final = _summarize_long_text(content, max_chars)
                was_summarized = True
                is_truncated = True
            except ToolExecutionCancelled:
                raise
            except Exception as e:
                print(f"[web_extract] 摘要失败，已回退截断：{e}", file=sys.stderr)
                final = _trim(content, max_chars)
                was_summarized = False
                is_truncated = True

        result = {"url": url, "content": final, "truncated": is_truncated}
        if was_summarized:
            result["summarized"] = True
        results.append(result)

    # 失败项合并到同一 results 列表（与 hermes _normalize_tavily_documents 一致）：
    # LLM 在一个数组里就能看到「哪个 URL 成功、哪个失败、为什么失败」。
    for fail in resp.get("failed_results", []):
        results.append({
            "url": fail.get("url", ""),
            "content": "",
            "error": fail.get("error", "extraction failed"),
        })
    # failed_urls：Tavily 只给 url（无 error 详情）时的兜底
    for fail_url in resp.get("failed_urls", []):
        if isinstance(fail_url, str):
            url_str = fail_url
        elif isinstance(fail_url, dict):
            url_str = fail_url.get("url", "")
        else:
            url_str = str(fail_url)
        results.append({
            "url": url_str,
            "content": "",
            "error": "extraction failed",
        })

    return json.dumps(
        {"success": True, "results": results},
        ensure_ascii=False, indent=2,
    )


# 工具 schema：供注册表自注册；discover() 自动发现后交给 LLM，无需手动加进 tools 列表
WEB_EXTRACT_TOOL = {
    "name": "web_extract",
    "description": (
        "抽取指定网页 URL 的正文内容（markdown 格式）。适合在 web_search 之后，"
        "需要深入阅读某个页面全文时使用。每个 URL 返回正文上限约 5000 字符，"
        "超长会自动用 LLM 摘要压缩以保留关键信息（>500k 字符会分块摘要），"
        "摘要失败回退到截断；过大的页面（>2MB）会被拒绝。单次最多传入 5 个 URL。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要抽取内容的 URL 列表，最多 5 个",
            },
            "max_chars": {
                "type": "integer",
                "description": "每个页面返回的正文上限字符数，默认及最大 5000（可选）",
                "minimum": 1,
                "maximum": 5000,
            },
        },
        "required": ["urls"],
    },
}

if registry is not None:
    registry.register(name="web_extract", schema=WEB_EXTRACT_TOOL, handler=web_extract)


if __name__ == "__main__":
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    raw = client.extract(urls="https://chat.deepseek.com/")
    # .get("results", [])
    print(json.dumps(raw, indent=2, ensure_ascii=False))
