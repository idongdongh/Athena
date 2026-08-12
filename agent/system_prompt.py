"""按 Hermes 的 stable/context/volatile 分层构建会话 system prompt。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    MEMORY_GUIDANCE,
    PARALLEL_TOOL_CALL_GUIDANCE,
    SESSION_SEARCH_GUIDANCE,
    TASK_COMPLETION_GUIDANCE,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
)


def build_system_prompt_parts(agent: Any) -> dict[str, str]:
    """只为 Athena 当前真实可用的能力生成提示，不描述已裁剪功能。"""
    tool_names = {item.get("name") for item in agent.tool_definitions()}
    stable = [DEFAULT_AGENT_IDENTITY]
    if tool_names:
        stable.extend([
            TASK_COMPLETION_GUIDANCE,
            PARALLEL_TOOL_CALL_GUIDANCE,
            TOOL_USE_ENFORCEMENT_GUIDANCE,
        ])
    if "memory" in tool_names:
        stable.append(MEMORY_GUIDANCE)
    if "session_search" in tool_names:
        stable.append(SESSION_SEARCH_GUIDANCE)

    context = [f"Working directory: {Path(agent.working_directory).resolve()}"]
    if agent.caller_system_prompt:
        context.append(agent.caller_system_prompt)

    volatile = []
    store = agent._memory_store
    settings = agent.memory_settings
    if store is not None and settings is not None:
        if settings.memory_enabled:
            block = store.format_for_system_prompt("memory")
            if block:
                volatile.append(block)
        if settings.user_profile_enabled:
            block = store.format_for_system_prompt("user")
            if block:
                volatile.append(block)
    volatile.append(f"Conversation started: {datetime.now().strftime('%A, %B %d, %Y')}")
    return {
        "stable": "\n\n".join(stable),
        "context": "\n\n".join(context),
        "volatile": "\n\n".join(volatile),
    }


def build_system_prompt(agent: Any) -> str:
    parts = build_system_prompt_parts(agent)
    return "\n\n".join(parts[name] for name in ("stable", "context", "volatile") if parts[name])
