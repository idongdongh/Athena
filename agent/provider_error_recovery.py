"""模型 Provider 请求错误分类与有界重试。"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Callable, TypeVar

# 定义一个泛型类型，类型标识符：T
T = TypeVar("T")
# 可重试状态码
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class ClassifiedProviderError:
    """Provider 异常的稳定分类。"""

    kind: str
    retryable: bool # 这种类型错误是否需要重试
    status_code: int | None = None # 状态码
    retry_after: float | None = None # 多久后重试


class ProviderRequestFailed(RuntimeError):
    """Provider 请求在不可重试或重试耗尽后失败。"""

    def __init__(self, error: ClassifiedProviderError, cause: Exception):
        super().__init__(str(cause))
        self.error = error
        self.cause = cause


class ProviderRequestInterrupted(RuntimeError):
    """用户在 Provider 重试退避期间中断了请求。"""


def _status_code(exc: Exception) -> int | None:
    """提取异常对象中的 http 状态码

    Args:
        exc (Exception): 异常对象

    Returns:
        int | None: 要么返回状态码，要么返回 None
    """
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _retry_after(exc: Exception) -> float | None:
    """从异常对象中读取响应报文中的 retry-after（可能是日期格式也能是秒数）（建议等待多少秒后重试）

    Args:
        exc (Exception): 异常对象

    Returns:
        float | None:
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return min(max(float(value), 0.0), 600.0)
    except (TypeError, ValueError):
        try:
            # 解析时间格式；服务器返回过去时间（负值）按"无有效 retry-after"处理
            # —— 立即重试会绕过 429/503 的退避语义。
            seconds = parsedate_to_datetime(str(value)).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
        if seconds < 0:
            return None
        return min(max(seconds, 0.0), 600.0)


def classify_provider_error(exc: Exception) -> ClassifiedProviderError:
    """按状态码和 SDK 异常名称判断 Provider 错误是否可重试。"""
    status = _status_code(exc)
    # 打印小写的类型名称
    name = type(exc).__name__.lower()

    if status == 401 or "authentication" in name:
        kind = "authentication"
    elif status == 403 or "permission" in name:
        kind = "permission"
    elif status == 404 or "notfound" in name:
        kind = "model_not_found"
    elif status in {400, 413, 422} or "badrequest" in name:
        kind = "invalid_request"
    elif status == 429 or "ratelimit" in name:
        kind = "rate_limit"
    elif status == 529 or "overload" in name:
        kind = "overloaded"
    elif status == 408:
        kind = "timeout"
    elif status == 409:
        kind = "transient_conflict"
    elif status is not None and 500 <= status < 600:
        kind = "server_error"
    elif "timeout" in name:
        kind = "timeout"
    elif "connection" in name:
        kind = "connection"
    else:
        kind = "unknown"

    retryable = status in RETRYABLE_STATUS_CODES or kind in {
        "rate_limit", "overloaded", "server_error", "timeout", "connection",
        "transient_conflict", "unknown",
    }
    return ClassifiedProviderError(kind, retryable, status, _retry_after(exc))


def jittered_backoff(retry_number: int, *, base_delay: float = 1.0) -> float:
    """带抖动的退避：计算“第几次重试前需要等待多久”，使用的是指数退避 + 随机抖动。计算公式：等待时间 = min(基础时间 × 2^(重试次数-1), 60) × 0.8～1.2 的随机值"""
    delay = min(base_delay * (2 ** max(retry_number - 1, 0)), 60.0)
    return delay * random.uniform(0.8, 1.2)


def _interruptible_wait(
    seconds: float,
    *,
    is_interrupted: Callable[[], bool],
    sleep: Callable[[float], None],
) -> None:
    if is_interrupted():
        raise ProviderRequestInterrupted("Provider retry interrupted")
    remaining = max(seconds, 0.0)
    # 每个 0.2 秒检查一次用户是否中断
    while remaining > 0:
        interval = min(remaining, 0.2)
        sleep(interval)
        remaining -= interval
        if is_interrupted():
            raise ProviderRequestInterrupted("Provider retry interrupted")


def request_with_retries(
    request: Callable[[], T],
    *,
    max_retries: int = 3,
    is_interrupted: Callable[[], bool] = lambda: False,
    on_retry: Callable[[ClassifiedProviderError, int, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """执行 Provider 请求；可恢复错误按退避策略进行有限重试。"""
    retries = 0
    while True:
        try:
            return request()
        except Exception as exc:
            classified = classify_provider_error(exc)
            allowed_retries = 1 if classified.kind == "unknown" else max_retries
            if not classified.retryable or retries >= allowed_retries:
                # from 表示新异常是由哪个异常引起的
                raise ProviderRequestFailed(classified, exc) from exc

            retries += 1
            wait_seconds = classified.retry_after
            if wait_seconds is None:
                wait_seconds = jittered_backoff(retries)
            if on_retry is not None:
                on_retry(classified, retries, wait_seconds)
            _interruptible_wait(
                wait_seconds,
                is_interrupted=is_interrupted,
                sleep=sleep,
            )
