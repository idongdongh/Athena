"""运行轨迹追踪（最小核心版）。

在 agent 运行期间把关键事件以 JSONL 落盘到 logs/，便于回放、审计、调试。
对齐 hermes 的 agent trace 概念，但砍掉厚壳（不持久化完整 message、不做 spans 树）。

事件类型：
- step_start : 每一轮 API 调用的开始（时间、事件类型、步号、消息数、是否带 tools）
- step_done  : 每轮模型返回（stop_reason、本轮调用的工具名列表）
- tool_call  : 单个工具被派发（名称 + 入参，超限截断）
- tool_result: 单个工具返回（名称 + 结果，超限截断）
- finish     : 整个 agent_loop 结束（原因 + 总步数）

模块级单例：conversation_loop 在每个 agent_loop 开头 reset_tracer() 新建一份轨迹，
tool_executor 通过 get_tracer() 取同一份。并发写由 _lock 保护。
"""

import time
import json
import os
import threading
import uuid
from datetime import datetime

TRACE_DIR = "logs"
_lock = threading.Lock()


def _truncate(s: str, n: int = 500) -> str:
    """超长字符串截断并打标记，避免单条 trace 把行撑爆。"""
    return s if len(s) <= n else s[:n] + f"...[截断 {len(s) - n} 字符]"


class Tracer:
    def __init__(self, path=None):
        os.makedirs(TRACE_DIR, exist_ok=True)
        trace_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}"
        self.path = path or os.path.join(TRACE_DIR, f"trace_{trace_id}.jsonl")
        self.t0 = time.time()
        self.steps = 0

    # 写入实例初始化到首次调用模型的时间 + 不同时间的记录项
    def _emit(self, kind, **data):
        ev = {"t": round(time.time() - self.t0, 3), "kind": kind, **data}
        line = json.dumps(ev, ensure_ascii=False)
        with _lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    # 记录步号，消息数，是否使用工具，以及本轮用户 query
    def step_start(self, n, n_messages, with_tools, query=None):
        self.steps = n
        self._emit("step_start", step=n, messages=n_messages, with_tools=with_tools,
                   query=_truncate(query) if query else None)

    def step_done(self, stop_reason, tool_calls, assistant_text=None):
        self._emit("step_done", step=self.steps, stop_reason=stop_reason, tool_calls=tool_calls,
                   assistant_text=_truncate(assistant_text) if assistant_text else None)

    def tool_call(self, name, args):
        self._emit("tool_call", name=name, args=_truncate(json.dumps(args, ensure_ascii=False)))

    def tool_result(self, name, output):
        self._emit("tool_result", name=name, output=_truncate(output))

    def finish(self, reason, total_steps):
        self._emit("finish", reason=reason, total_steps=total_steps)


_tracer = None


def get_tracer():
    """取当前会话的 tracer 单例（首次调用时惰性创建）。"""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def reset_tracer():
    """每个 agent_loop 开头调用，新建一份独立轨迹文件。"""
    global _tracer
    _tracer = Tracer()
    return _tracer
