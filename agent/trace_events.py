"""Agent 运行时的只读轨迹事件出口。

运行时只依赖 ``TraceSink`` 协议，不感知评估数据的存储格式。
任何记录失败都必须 fail-open，不得改变 Agent 行为。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TraceSink(Protocol):
    """运行时轨迹事件接收器的最小协议。"""

    def emit(self, event_type: str, payload: Mapping[str, Any] | None = None) -> None:
        """记录一条结构化事件。"""


def emit_trace(
    sink: TraceSink | None,
    event_type: str,
    **payload: Any,
) -> None:
    """以 fail-open 方式向可选 sink 发送事件。"""
    if sink is None:
        return
    try:
        sink.emit(event_type, payload)
    except Exception:
        logger.warning("Trace sink failed while emitting %s", event_type, exc_info=True)
