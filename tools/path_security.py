"""file 工具的路径安全（写操作闸门）。

对齐 hermes 两处源的子集：
- ``tools/path_security.py`` 的 ``validate_within_dir`` / ``has_traversal_component``
  （workspace 边界 + ``..`` 快速检测，逐行搬）。
- ``tools/file_tools.py:416-470`` 的敏感路径黑名单（``_SENSITIVE_PATH_PREFIXES`` 等），
  + ``tools/approval.py:206-219`` 的敏感写入目标（ssh/env/config/credentials）。

砍掉的 hermes 厚壳：
- ``_resolve_path_for_task``（task_id 多工作目录）→ 复用本项目已有的 ``tools/_path.resolve``
  （单 cwd 场景）。
- symlink hop 遍历（20 跳）→ V1 单次 ``Path.resolve()`` 够用。
- V4A patch header ``..`` 检测 → 本项目 patch 工具无 V4A 模式。
- ``_get_hermes_config_resolved`` → 用等价的 ``~/.env`` 保护替代。

本模块只负责「写操作」（write_file / patch）。读操作（read_file / search_files）V1 放行，
hermes 那边的设备文件拦截（/dev/zero、/proc/*/environ）属厚壳，按「别复杂」暂不做。
"""

from pathlib import Path
from typing import Optional

from tools._path import resolve


def validate_within_dir(path: Path, root: Path) -> Optional[str]:
    """确保 *path* 解析后落在 *root* 之内。

    返回错误消息字符串（校验失败）或 ``None``（安全）。
    用 ``Path.resolve()`` 跟符号链接 + 归一化 ``..`` 分量。

    用法::

        error = validate_within_dir(user_path, allowed_root)
        if error:
            return json.dumps({"error": error})
    """
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
    except (ValueError, OSError) as exc:
        return f"Path escapes allowed directory: {exc}"
    return None


def has_traversal_component(path_str: str) -> bool:
    """若 *path_str* 含 ``..`` 路径分量则返回 True。

    在做完整 resolve 之前的快速词法预检（无 IO）。
    """
    parts = Path(path_str).parts
    return ".." in parts


# ── 系统敏感路径（来源：hermes file_tools.py:416-420）─────────────────────
# 写入这些路径前缀/精确路径一律拒绝。realpath 后匹配，防符号链接绕过。
_SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/", "/private/var/",   # macOS 符号链接镜像
)
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

# ── 用户/项目敏感写入目标（来源：hermes approval.py:206-219 简化）─────────
# ssh 密钥、shell rc、凭据文件、项目 .env/config —— 写入这些可植入后门/窃取凭据。
# expanduser + normpath 后做前缀/精确匹配。
_SENSITIVE_USER_WRITES = (
    "~/.ssh/",            # authorized_keys 植入、私钥窃取
    "~/.env",             # 本项目根 .env（API key 明文）
    "~/.bashrc", "~/.zshrc", "~/.profile", "~/.bash_profile", "~/.zprofile",  # 登录时执行命令注入
    "~/.netrc", "~/.pgpass", "~/.npmrc", "~/.pypirc",  # 各类凭据
)
# 项目级 .env / config 的精确后缀判断在 check_write_path 内做（含 .example 模板白名单）


def check_write_path(path: str) -> Optional[str]:
    """写操作路径安全检查。返回 ``None`` 放行，返回 ``str`` 为拒绝消息。

    对应 hermes ``_check_sensitive_path``（file_tools.py:443）+ shell 侧敏感写入目标的合并。
    被 ``tools/_permission.py`` 在 write_file / patch 执行前调用。

    Args:
        path: 工具传入的目标路径（可为绝对/相对/~ 开头）。

    Returns:
        ``None`` = 放行；非空字符串 = 拒绝原因（写给模型看）。
    """
    # 1. resolve：跟符号链接 + 归一化 .. —— 防符号链接绕过
    try:
        resolved = resolve(path)
        resolved_str = str(resolved.resolve())  # resolve() 已 join cwd，再 .resolve() 跟符号链接
    except (OSError, ValueError):
        resolved_str = path  # resolve 失败时退化到原始字符串比对（fail-open，与 hermes 一致）

    # 2. 系统敏感路径前缀/精确匹配（realpath 后）
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

    # 3. 用户/项目敏感写入目标（~ 展开 + normpath 后匹配）
    #    用 os.path.normpath 归一化 .././ 等，与 hermes _check_sensitive_path 的双路径比对一致
    import os
    normalized = os.path.normpath(os.path.expanduser(path))
    for target in _SENSITIVE_USER_WRITES:
        target_norm = os.path.normpath(os.path.expanduser(target))
        # 精确匹配（如 ~/.bashrc）或前缀匹配（如 ~/.ssh/ 下任意文件）
        if normalized == target_norm or normalized.startswith(target_norm + "/"):
            return (
                f"拒绝写入敏感文件: {path}\n"
                "（SSH 密钥、shell 配置、凭据文件或 .env）—— "
                "写入这些文件可被用于植入后门或窃取凭据。如确需修改请手动操作。"
            )
    # 项目级 .env / config.yaml：任意目录下的（宽松 endsuffix，覆盖子目录）
    # .env / .env.local / config.yaml 算敏感（拦）；.env.example / .env.sample 是模板（放行）
    _SENSITIVE_PROJECT_SUFFIXES = ("/.env", "/config.yaml")
    _TEMPLATE_SUFFIXES = (".example", ".sample")
    # 先排除模板文件（.env.example / config.yaml.sample 等）
    if not normalized.endswith(_TEMPLATE_SUFFIXES):
        for suffix in _SENSITIVE_PROJECT_SUFFIXES:
            if normalized.endswith(suffix) or normalized.endswith(suffix + ".local"):
                return (
                    f"拒绝写入项目配置文件: {path}\n"
                    "（.env / config.yaml 含敏感配置）—— 如确需修改请手动操作。"
                )

    return None
