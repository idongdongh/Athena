"""正常 turn 结束后异步运行的受限记忆复盘 Agent。"""

from __future__ import annotations

import json
import threading
from typing import Any


MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work "
    "style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. Make all related changes "
    "in one atomic operations call. If nothing is worth saving, just say "
    "'Nothing to save.' and stop. You may only call the memory tool."
)


def spawn_background_review_thread(parent: Any, messages_snapshot: list[dict]) -> threading.Thread:
    """创建单次 daemon 线程；线程内的 Agent 与父 turn 状态及会话存储隔离。"""
    return threading.Thread(
        target=_run_review,
        args=(parent, list(messages_snapshot)),
        daemon=True,
        name="athena-memory-review",
    )


def _run_review(parent: Any, messages_snapshot: list[dict]) -> None:
    from run_agent import AIAgent

    review_agent = None
    try:
        review_agent = AIAgent(
            model=parent.model,
            system_prompt="",
            context_settings=parent.context_settings,
            client=parent.client,
            memory_settings=parent.memory_settings,
            memory_root=parent._memory_store.memory_dir,
            working_directory=parent.working_directory,
            tool_allowlist=frozenset({"memory"}),
            is_background_review=True,
        )
        # 复盘与前台写入必须落到同一个 live store；文件锁负责并发顺序化。
        review_agent._memory_store = parent._memory_store
        review_agent._rebuild_system_prompt()
        review_messages = list(messages_snapshot)
        review_messages.append({"role": "user", "content": MEMORY_REVIEW_PROMPT})
        review_agent.run_conversation(review_messages, stream_output=False)
        actions = _successful_memory_actions(review_messages, len(messages_snapshot))
        if actions:
            print(f"\n  💾 定期记忆复盘：{'；'.join(actions)}")
    except Exception as exc:
        print(f"\n\033[33m⚠️  后台记忆复盘失败：{exc}\033[0m")


def _successful_memory_actions(messages: list[dict], start: int) -> list[str]:
    """只汇报复盘新增的成功 memory tool_result，不回显后台推理过程。"""
    actions: list[str] = []
    for message in messages[start:]:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if block_type != "tool_result":
                continue
            raw = block.get("content") if isinstance(block, dict) else getattr(block, "content", "")
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("success") is True and payload.get("target") in {"memory", "user"}:
                label = "项目记忆" if payload["target"] == "memory" else "用户画像"
                actions.append(f"已更新{label}")
    return list(dict.fromkeys(actions))
