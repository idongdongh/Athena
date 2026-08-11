"""线程安全的单进程中断状态。

当前项目同一时间只运行一个 agent turn，因此一个进程级控制器即可让模型请求、
工具调度和可中断工具共享同一取消信号。若未来支持多个并行 agent，应改为每个
agent 实例持有独立控制器。
"""

import threading
from typing import Callable, TypeVar


T = TypeVar("T")


class ToolExecutionCancelled(Exception):
    """工具响应用户中断并完成自身资源清理。"""


class InterruptController:
    """基于 ``threading.Event`` 的协作式中断控制器。"""

    def __init__(self) -> None:
        """初始化一个线程事件对象，一个供线程共享的、线程安全的布尔信号开关，用来告诉其他线程某个事件发生了"""
        self._event = threading.Event()

    def request(self) -> None:
        """发出信号，表示用户已中断。"""
        self._event.set()

    def clear(self) -> None:
        """清除信号，清除上一轮中断，供下一轮安全复用。"""
        self._event.clear()

    def is_requested(self) -> bool:
        """检查信号，检查是否用户是否中断。"""
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """等待信号，等待 timeout 秒，如果收到信号返回 True 反之返回 False。"""
        return self._event.wait(timeout)

    def raise_if_requested(self, reason: str = "operation interrupted by user") -> None:
        """收到中断请求，抛出 ToolExecutionCancelled 异常。"""
        if self.is_requested():
            raise ToolExecutionCancelled(reason)


interrupt_controller = InterruptController()


def run_interruptible(
    operation: Callable[[], T],
    *,
    on_cancel: Callable[[], None] | None = None,
    poll_interval: float = 0.05,
    thread_name: str = "interruptible-operation",
) -> T:
    """在 daemon worker 中执行阻塞调用，让当前线程可响应用户中断。

    Python 不能强制终止线程，因此 ``on_cancel`` 应尽量关闭底层 HTTP client；即使
    SDK 没有立即解除阻塞，daemon worker 也不会拖住当前 turn 或进程退出。
    """
    interrupt_controller.raise_if_requested()
    finished = threading.Event()
    state: dict[str, object] = {"result": None, "error": None}

    def worker() -> None:
        try:
            state["result"] = operation()
        except Exception as exc:
            state["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=worker, name=thread_name, daemon=True).start()

    while not finished.wait(poll_interval):
        if interrupt_controller.is_requested():
            break

    if interrupt_controller.is_requested():
        # 清理回调只执行一次。关闭 HTTP client 等操作不应依赖调用方幂等，
        # 同时保留“worker 刚结束时才收到中断”这一竞态下的取消语义。
        if on_cancel is not None:
            try:
                on_cancel()
            except Exception:
                pass
        raise ToolExecutionCancelled("operation interrupted by user")
    if state["error"] is not None:
        raise state["error"]  # type: ignore[misc]
    return state["result"]  # type: ignore[return-value]
