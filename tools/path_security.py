"""write_file / patch 的敏感路径拒绝规则。

对齐 Hermes ``agent/file_safety.py:is_write_denied`` 的两个关键行为：先解析
realpath（覆盖相对路径和符号链接），再按精确路径、目录前缀和 basename 判断。

这是一层面向模型的 defense-in-depth，不是 OS 沙箱：bash 与当前进程拥有相同的
文件权限。若需要真正隔离，应把工具放进受限执行环境，而不是继续扩充路径黑名单。
"""

import os
from pathlib import Path
from typing import Optional

from tools._path import resolve


_SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/", "/private/var/",
)
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

_SENSITIVE_USER_WRITES = (
    "~/.ssh/",
    "~/.env",
    "~/.bashrc", "~/.zshrc", "~/.profile", "~/.bash_profile", "~/.zprofile",
    "~/.netrc", "~/.pgpass", "~/.npmrc", "~/.pypirc",
)

_SENSITIVE_PROJECT_BASENAMES = {".env", ".env.local", "config.yaml"}


def check_write_path(path: str) -> Optional[str]:
    """检查原生文件工具的写入目标；返回 ``None`` 放行，否则返回拒绝原因。"""
    try:
        resolved_str = os.path.realpath(resolve(path))
    except (OSError, TypeError, ValueError):
        resolved_str = os.path.abspath(os.path.expanduser(str(path)))

    for prefix in _SENSITIVE_PATH_PREFIXES:
        if resolved_str.startswith(prefix):
            return (
                f"拒绝写入系统敏感路径: {path}\n"
                "系统文件（/etc, /boot 等）不可通过 agent 修改。"
                "如确需修改，请手动用 sudo 在终端执行。"
            )
    if resolved_str in _SENSITIVE_EXACT_PATHS:
        return (
            f"拒绝写入系统关键文件: {path}\n"
            "（如 docker.sock 等服务控制接口）"
        )

    # 用户敏感路径也使用 realpath；否则工作区符号链接可绕到 ~/.ssh。
    for target in _SENSITIVE_USER_WRITES:
        target_real = os.path.realpath(os.path.expanduser(target))
        if resolved_str == target_real or resolved_str.startswith(target_real + os.sep):
            return (
                f"拒绝写入敏感文件: {path}\n"
                "（SSH 密钥、shell 配置、凭据文件或 .env）—— "
                "写入这些文件可被用于植入后门或窃取凭据。如确需修改请手动操作。"
            )

    # basename 判断同时覆盖仓库根目录和子目录；.env.example 等模板不会命中。
    if Path(resolved_str).name in _SENSITIVE_PROJECT_BASENAMES:
        return (
            f"拒绝写入项目配置文件: {path}\n"
            "（.env / config.yaml 含敏感配置）—— 如确需修改请手动操作。"
        )

    return None
