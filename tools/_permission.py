"""权限闸门总调度：按工具名分发到对应检查函数（table-driven）。

被 ``tool_executor.dispatch_tool_call`` 在执行 handler 前调用。
对应 hermes「每个工具自管权限」的统一替代 —— 本项目 7 工具单进程，
用一张表 + 一个接缝比 hermes 把 check_fn 散到各工具里更干净。

加新工具的权限检查：在 ``_CHECKERS`` 加一行 ``"工具名": lambda args: check_xxx(...)``
即可，``dispatch_tool_call`` 不用改。
"""

from typing import Callable, Dict, Optional

from tools.approval import check_command
from tools.path_security import check_write_path

# 工具名 → 检查函数。
# 约定：检查函数签名为 ``(args: dict) -> Optional[str]``，
#        返回 None 放行、返回 str 为拒绝消息（写给模型）。
_CHECKERS: Dict[str, Callable[[dict], Optional[str]]] = {
    "bash": lambda args: check_command(args.get("command", "")),
    "write_file": lambda args: check_write_path(args.get("path", "")),
    "patch": lambda args: check_write_path(args.get("path", "")),
    # read_file / search_files：V1 放行（只读无破坏性）
    # web_search / web_extract：无破坏性，放行
}


def check_tool_permission(name: str, args: dict) -> Optional[str]:
    """权限闸门总入口。

    Args:
        name: 工具名（如 "bash" / "write_file"）。
        args: 模型传入的参数 dict。

    Returns:
        ``None`` = 放行，执行 handler；非空字符串 = 拒绝消息（直接返回给模型，
        不执行 handler）。
    """
    # 这里是获取到 check_command(args.get("command", ""))
    checker = _CHECKERS.get(name)
    if checker is None:
        return None  # 无配置检查 = 放行（白名单策略）
    # 这里才是调用 check_command
    return checker(args)
