"""Tavily 网络搜索工具。

hermes 思路只保留一条：把搜索结果归一化成干净的
{title, url, description, position} 结构，再返回 JSON 字符串给模型，
而不是把 vendor 原始 dict 直接丢出去。

当前项目不保留 Hermes 的多 provider 抽象，工具通过本地 registry 自注册。
"""

import os
import json

from tavily import TavilyClient
from agent.interrupt_controller import ToolExecutionCancelled, run_interruptible

# 如果当脚本被直接执行，sys.pathp[0]=xx/tools，然后在这个目录下面找tools.registry，会找不到
try:
    from tools.registry import registry
except ImportError:
    registry = None


def web_search(query: str, limit: int = 5) -> str:
    """
    使用可用的搜索 API 后端搜索网络信息。

    当前后端固定为 Tavily。

    注意：此函数仅返回搜索结果的元数据（URL、标题、描述）。
    如需获取特定 URL 的完整内容，请使用 web_extract_tool。

    Args:
        query (str): 搜索查询关键词
        limit (int): 返回结果的最大数量（默认值：5）

    Returns:
        str: 包含搜索结果的 JSON 字符串，结构如下：
             {
                 "success": bool,
                 "data": {
                     "web": [
                         {
                             "title": str,
                             "url": str,
                             "description": str,
                             "position": int
                         },
                         ...
                     ]
                 }
             }

    失败时返回 ``{"success": false, "error": ...}``，不向调用方抛异常。
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = min(max(limit, 1), 20)

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    close_client = getattr(client, "close", lambda: None)
    try:
        response = run_interruptible(
            lambda: client.search(query, max_results=limit, timeout=30),
            on_cancel=close_client,
            thread_name="web-search-request",
        )
        raw = response["results"]
    except ToolExecutionCancelled:
        raise
    # 有异常则执行，Exception 是 python 内置异常基类包括很多异常
    except Exception as exc:
        # ensure_ascii=False：所有非 ascii 字符不转义
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)
    finally:
        close_client()

    web = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "description": r.get("content", "") or r.get("description", ""),
            "position": i,
        }
        for i, r in enumerate(raw, start=1)
    ]
    return json.dumps(
        {"success": True, "data": {"web": web}},
        ensure_ascii=False,
        indent=2,
    )


# 工具 schema：供注册表自注册；discover() 自动发现后交给 LLM，无需手动加进 tools 列表
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "联网搜索网页。当你需要回答关于时事、最新事实、"
        "或本地知识库中找不到的信息时使用。返回包含标题、链接、摘要的结果列表。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "limit": {
                "type": "integer",
                "description": "返回结果数量，默认 5，范围 1-20（可选）",
            },
        },
        "required": ["query"],
    },
}

if registry is not None:
    registry.register(name="web_search", schema=WEB_SEARCH_TOOL, handler=web_search)

if __name__ == "__main__":
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    raw = client.search("马斯克多有钱", max_results=1, include_raw_content=True)
    print(type(raw))
    res = json.dumps(raw, indent=2, ensure_ascii=False)
    print(res)
