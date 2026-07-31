"""路径解析辅助（工具共享）。

抽出来避免 read_file / write_file / patch / search_files 各维护一份
``_resolve`` 副本——曾经 4 份函数体逐字节相同，却要改 4 处才能同步。
"""

import os
from pathlib import Path


def resolve(path: str) -> Path:
    """解析路径：展开 ~，相对路径相对进程 cwd。允许绝对路径。"""
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    return Path(p)
