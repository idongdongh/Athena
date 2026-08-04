"""bash 命令权限闸门（HARDLINE 地板拒绝 + DANGEROUS 需审批）。

对齐 hermes ``tools/approval.py``（2171 行）的核心子集，砍掉厚壳：
- gateway 异步审批（``_await_gateway_decision`` / ``submit_pending``）
- permanent allowlist 持久化（cross-session）
- yolo / cron / gateway 多模式
- tirith 安全扫描集成、``_smart_approve``（LLM 风险评估）
- approval hooks / observability context
- pattern_key aliases（legacy 兼容）
- ``check_execute_code_guard``（本项目无 execute_code）

保留的核心（防失控 + 防绕过）：
1. **HARDLINE_PATTERNS**（12 条）：不可逆损害，无条件拒绝，不可绕过。
2. **DANGEROUS_PATTERNS**（高频子集 ~20 条）：可逆但有风险，命中暂停问用户。
3. **``_normalize_command_for_detection``**（6 道清洗）：防 ANSI/null/全角/反斜杠/空串绕过。
4. **``_check_sudo_stdin_guard``**：防 sudo 密码爆破。
5. **会话级审批记忆**：用户选 ``s`` 后，本会话不再问同类命令。

闸门挂载点：``tool_executor.dispatch_tool_call`` 在执行 handler 前调 ``check_command``。
"""

import os
import re
import threading
import unicodedata
from typing import Optional, Tuple


# ════════════════════════════════════════════════════════════════════════
# 1. ANSI strip（搬 hermes tools/ansi_strip.py，纯函数）
# ════════════════════════════════════════════════════════════════════════
# 命令里的 ANSI 转义可用来切碎关键字绕过正则（如 r\x1b[m → rm）。
# 全 ECMA-48 覆盖：CSI/OSC/DCS/SOS/PM/APC/nF/Fp/Fe/Fs + 8-bit C1。

_ANSI_ESCAPE_RE = re.compile(
    r"\x1b"
    r"(?:"
        r"\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"     # CSI sequence
        r"|\][\s\S]*?(?:\x07|\x1b\\)"                  # OSC (BEL or ST terminator)
        r"|[PX^_][\s\S]*?(?:\x1b\\)"                   # DCS/SOS/PM/APC strings
        r"|[\x20-\x2f]+[\x30-\x7e]"                    # nF escape sequences
        r"|[\x30-\x7e]"                                 # Fp/Fe/Fs single-byte
    r")"
    r"|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"       # 8-bit CSI
    r"|\x9d[\s\S]*?(?:\x07|\x9c)"                       # 8-bit OSC
    r"|[\x80-\x9f]",                                    # Other 8-bit C1 controls
    re.DOTALL,
)
_HAS_ESCAPE = re.compile(r"[\x1b\x80-\x9f]")


def _strip_ansi(text: str) -> str:
    """去 ANSI 转义序列。无 ESC/C1 字节时走快速路径直接返回。"""
    if not text or not _HAS_ESCAPE.search(text):
        return text
    return _ANSI_ESCAPE_RE.sub("", text)


# ════════════════════════════════════════════════════════════════════════
# 2. HARDLINE_PATTERNS（地板拒绝，不可绕过）
# ════════════════════════════════════════════════════════════════════════
# 来源：hermes approval.py:263-285。不可逆损害：rm -rf /、mkfs、dd 写块设备、
# fork bomb、kill -1（杀全部进程）、shutdown/reboot。连用户 --yolo 都不能绕过。

# 命令起始位置（行首/分隔符后/子shell），兼容 sudo/env/exec 等 wrapper 前缀
_CMDPOS = (
    r"(?:^|[;&|\n`]|\$\()"         # start position
    r"\s*"
    r"(?:sudo\s+(?:-[^\s]+\s+)*)?"  # optional sudo with flags
    r"(?:env\s+(?:\w+=\S*\s+)*)?"   # optional env with VAR=VAL pairs
    r"(?:(?:exec|nohup|setsid|time)\s+)*"  # optional wrapper commands
    r"\s*"
)

HARDLINE_PATTERNS = [
    # rm recursive targeting the root filesystem or protected roots
    (r'\brm\s+(-[^\s]*\s+)*(/|/\*|/ \*)(\s|$)', "递归删除根文件系统"),
    (r'\brm\s+(-[^\s]*\s+)*(/home|/home/\*|/root|/root/\*|/etc|/etc/\*|/usr|/usr/\*|/var|/var/\*|/bin|/bin/\*|/sbin|/sbin/\*|/boot|/boot/\*|/lib|/lib/\*)(\s|$)', "递归删除系统目录"),
    (r'\brm\s+(-[^\s]*\s+)*(~|\$HOME)(/?|/\*)?(\s|$)', "递归删除 home 目录"),
    # Filesystem format
    (r'\bmkfs(\.[a-z0-9]+)?\b', "格式化文件系统 (mkfs)"),
    # Raw block device overwrites (dd + redirection)
    (r'\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*', "dd 写裸块设备"),
    (r'>\s*/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*\b', "重定向写裸块设备"),
    # Fork bomb (classic shell form)
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Kill every process on the system
    (r'\bkill\s+(-[^\s]+\s+)*-1\b', "杀掉系统所有进程"),
    # System shutdown / reboot — 锚定命令位置（行首/分隔符后/wrapper 之后），
    # 避免 "echo reboot" 或 "grep 'shutdown' logs" 误报
    (_CMDPOS + r'(shutdown|reboot|halt|poweroff)\b', "系统关机/重启"),
    (_CMDPOS + r'init\s+[06]\b', "init 0/6 (关机/重启)"),
    (_CMDPOS + r'systemctl\s+(poweroff|reboot|halt|kexec)\b', "systemctl poweroff/reboot"),
    (_CMDPOS + r'telinit\s+[06]\b', "telinit 0/6 (关机/重启)"),
]

_RE_FLAGS = re.IGNORECASE | re.DOTALL
HARDLINE_PATTERNS_COMPILED = [
    (re.compile(p, _RE_FLAGS), desc) for p, desc in HARDLINE_PATTERNS
]


# ════════════════════════════════════════════════════════════════════════
# 3. DANGEROUS_PATTERNS（高频子集 ~20 条，需审批）
# ════════════════════════════════════════════════════════════════════════
# 来源：hermes approval.py:381-530 选高频子集。命中后暂停问用户（y/s/n）。
# 覆盖你日常会遇到的风险：递归删除、权限滥用、git 破坏、脚本绕过、远程执行、提权、SQL 破坏。

DANGEROUS_PATTERNS = [
    # ── 删除 ──
    (r'\brm\s+(-[^\s]*\s+)*/', "根路径下删除"),
    (r'\brm\s+-[^\s]*r', "递归删除"),
    (r'\brm\s+--recursive\b', "递归删除 (长选项)"),
    # ── 权限滥用 ──
    (r'\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b', "设置 world/other 可写权限"),
    (r'\bchown\s+(-[^\s]*)?R\s+root', "递归 chown 给 root"),
    # ── git 破坏（丢未提交工作/重写远端历史）──
    (r'\bgit\s+reset\s+--hard\b', "git reset --hard (丢弃未提交改动)"),
    (r'\bgit\s+push\b.*--force\b', "git force push (重写远端历史)"),
    (r'\bgit\s+push\b.*-f\b', "git force push 短选项 (重写远端历史)"),
    (r'\bgit\s+clean\s+-[^\s]*f', "git clean 强制 (删除未跟踪文件)"),
    (r'\bgit\s+branch\s+-D\b', "git branch 强制删除"),
    # ── 脚本绕过（解决 notebook/当前问题.md 记的绕过缺口）──
    # bash -c / sh -c：shell 注入绕过命令名匹配
    (r'\b(bash|sh|zsh|ksh)\s+-[^\s]*c(\s+|$)', "通过 -c/-lc 标志执行 shell 命令"),
    # python3 -c / perl -e：脚本执行绕过——模型用 python3 -c "import os; os.remove(...)" 绕 rm 黑名单
    (r'\b(python[23]?|perl|ruby|node)\s+-[ec]\s+', "通过 -e/-c 标志执行脚本"),
    # ── 远程执行 ──
    (r'\b(curl|wget)\b.*\|\s*(?:[/\w]*/)?(?:ba)?sh(?:\s|$|-c)', "管道远程内容到 shell"),
    # ── 提权 ──
    (r'\bsudo\b[^;|&\n]*?\s+(?:-S\b|--stdin\b|-A\b|--askpass\b)',
     "sudo 带特权标志 (stdin/askpass)"),
    # ── SQL 破坏 ──
    (r'\bDROP\s+(TABLE|DATABASE)\b', "SQL DROP"),
    # 用 [^\n]* 而非 .* ，防 DOTALL 让下一行的 WHERE 满足负向预查，误放行无 WHERE 的 DELETE
    (r'\bDELETE\s+FROM\b(?![^\n]*\bWHERE\b)', "SQL DELETE 无 WHERE"),
    (r'\bTRUNCATE\s+(TABLE)?\s*\w', "SQL TRUNCATE"),
    # ── 格式化（DANGEROUS 版，HARDLINE 已有 mkfs，这里补 dd 磁盘拷贝）──
    (r'\bdd\s+.*if=', "磁盘拷贝 (dd)"),
    (r'>\s*/dev/sd', "写块设备"),
]

DANGEROUS_PATTERNS_COMPILED = [
    (re.compile(p, _RE_FLAGS), desc) for p, desc in DANGEROUS_PATTERNS
]


# ════════════════════════════════════════════════════════════════════════
# 4. 命令归一化（防绕过核心：6 道清洗）
# ════════════════════════════════════════════════════════════════════════
# 来源：hermes approval.py:567-603。在匹配正则前对命令做 6 道清洗，
# 让 ANSI 转义/null 字节/全角字符/绝对 home 路径/反斜杠转义/空字符串字面量
# 都无法绕过模式检测。

def _fold_home_prefix(command: str) -> str:
    """把绝对 home 路径前缀折叠成 ``~/``。

    简化自 hermes ``_rewrite_resolved_user_home``（砍 Windows 分隔符 + HERMES_HOME）。
    这样静态模式 ``~/.ssh`` 也能匹配 ``/home/alice/.ssh`` 和 realpath 后的形式。
    """
    try:
        home = os.path.expanduser("~")
        candidates = [home, os.path.realpath(home)]
    except Exception:
        return command
    # 长路径优先（避免短前缀吃掉长前缀需要的部分）
    for c in sorted({p for p in candidates if p and p != "/"}, key=len, reverse=True):
        # 用正则替换绝对 home 前缀（含后续路径段），换成 ~ + tail
        pat = re.compile(re.escape(c) + r"(/.*)?")
        command = pat.sub(lambda m: "~" + (m.group(1) or ""), command)
    return command


def _normalize_command_for_detection(command: str) -> str:
    """命令归一化：6 道清洗，防止混淆技术绕过模式检测。

    顺序敏感（见各步骤注释）。
    """
    # 1. 去所有 ANSI 转义序列（CSI/OSC/DCS/8-bit C1 等）
    command = _strip_ansi(command)
    # 2. 去 null 字节
    command = command.replace('\x00', '')
    # 3. Unicode NFKC 归一化（防全角 Latin 混淆，如 ｒｍ → rm）
    command = unicodedata.normalize('NFKC', command)
    # 4. 折叠绝对 home 路径 → ~/（必须在反斜杠剥离之前：Windows home 含反斜杠）
    command = _fold_home_prefix(command)
    # 5. 去反斜杠转义：r\\m → rm（防 \\-注入绕过）
    command = re.sub(r'\\([^\n])', r'\1', command)
    # 6. 去空字符串字面量：r''m → rm, r""m → rm（防引号切碎关键字）
    command = re.sub(r"''|\"\"", '', command)
    return command


# ════════════════════════════════════════════════════════════════════════
# 5. sudo stdin guard（防 sudo 密码爆破）
# ════════════════════════════════════════════════════════════════════════
# Hermes 只有在 terminal 层真正完成 ``sudo -S`` 命令改写和 stdin 密码注入后，
# 才会因 SUDO_PASSWORD 已配置而放行。本项目 bash 工具没有这条注入链，因此任何
# 显式 ``sudo -S`` 都拒绝，避免“配置存在但实际未使用”的半实现。
_SUDO_STDIN_RE = re.compile(
    r'(?:^|[;&|`\n]|&&|\|\||\$\()\s*sudo\s+-S\b',
    re.IGNORECASE)


def _check_sudo_stdin_guard(command: str) -> Tuple[bool, Optional[str]]:
    """检测显式 ``sudo -S``；当前项目不支持安全的密码 stdin 注入。

    Returns:
        (is_blocked, description) —— 被拦时 description 非空。
    """
    normalized = _normalize_command_for_detection(command).lower()
    if _SUDO_STDIN_RE.search(normalized):
        return (True, "尝试通过 stdin 获取 sudo 密码 (sudo -S)")
    return (False, None)


# ════════════════════════════════════════════════════════════════════════
# 6. 检测函数（HARDLINE / DANGEROUS 入口）
# ════════════════════════════════════════════════════════════════════════

def detect_hardline_command(command: str) -> Tuple[bool, Optional[str]]:
    """检查命令是否匹配无条件硬地板黑名单。

    Returns:
        (is_hardline, description) 或 (False, None)
    """
    normalized = _normalize_command_for_detection(command).lower()
    for pattern_re, description in HARDLINE_PATTERNS_COMPILED:
        if pattern_re.search(normalized):
            return (True, description)
    return (False, None)


def detect_dangerous_command(command: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """检查命令是否匹配需审批的危险模式。

    Returns:
        (is_dangerous, pattern_key, description) 或 (False, None, None)。
        pattern_key 用 description（人读），对齐 hermes（不用 regex 当 key）。
    """
    normalized = _normalize_command_for_detection(command).lower()
    for pattern_re, description in DANGEROUS_PATTERNS_COMPILED:
        if pattern_re.search(normalized):
            return (True, description, description)
    return (False, None, None)


# ════════════════════════════════════════════════════════════════════════
# 7. 会话级审批记忆
# ════════════════════════════════════════════════════════════════════════
# 来源：hermes approval.py:734, 827, 878。用户选 ``s`` 后，本会话不再问同类。
# 本项目单 CLI 进程，固定 session_key = "cli"。砍掉 permanent allowlist 持久化。

_session_approved: dict[str, set[str]] = {}  # session_key → {pattern_key, ...}
_session_lock = threading.Lock()
_SESSION_KEY = "cli"


def _is_approved(pattern_key: str) -> bool:
    """本会话是否已批准过该 pattern_key。"""
    with _session_lock:
        return pattern_key in _session_approved.get(_SESSION_KEY, set())


def _approve_session(pattern_key: str) -> None:
    """把 pattern_key 记进本会话的已批准集合。"""
    with _session_lock:
        _session_approved.setdefault(_SESSION_KEY, set()).add(pattern_key)

def clear_session() -> None:
    """清空本会话的审批记忆（测试/重置用）。"""
    with _session_lock:
        _session_approved.pop(_SESSION_KEY, None)


# ════════════════════════════════════════════════════════════════════════
# 8. 同步审批 prompt
# ════════════════════════════════════════════════════════════════════════

def _prompt_approval(command: str, pattern_key: str, description: str) -> Optional[str]:
    """同步问用户是否允许危险命令。返回 ``None`` 放行，``str`` 拒绝消息。

    选项：y=本次放行 / s=本会话记忆（不再问同类） / n=拒绝（默认）。
    ``EOFError``（Ctrl+D）当拒绝处理，不崩栈。
    ``KeyboardInterrupt`` 不在此 catch —— 让它冒泡到 ``agent_loop`` 的 force_quit
    （审批时中断 = 用户想中断整个任务，可接受语义）。
    """
    print(f"\n\033[33m⚠️  危险命令需审批: {description}\033[0m")
    print(f"    \033[33m命令: {command}\033[0m")
    try:
        choice = input("    允许执行？[y=本次/s=本会话/n=拒绝] ").strip().lower()
    except EOFError:
        # Ctrl+D：当拒绝处理
        choice = "n"
    if choice in ("y", "yes"):
        return None  # 本次放行，不记忆
    if choice in ("s", "session"):
        _approve_session(pattern_key)
        return None  # 放行 + 本会话记忆
    # n / 其他 / 空 → 拒绝
    return (
        f"用户拒绝了危险命令: {command}\n"
        f"（风险: {description}）请换一个更安全的方案，不要绕过此限制。"
    )


# ════════════════════════════════════════════════════════════════════════
# 9. 总入口 check_command
# ════════════════════════════════════════════════════════════════════════

def check_command(command: str) -> Optional[str]:
    """bash 命令权限闸门总入口。

    被 ``tools/_permission.py`` 在 bash 工具执行前调用。

    流程（对齐 hermes ``check_dangerous_command`` 的核心顺序）：
    1. sudo stdin guard（密码爆破）→ 拒绝
    2. HARDLINE（不可逆损害）→ 无条件拒绝，不可绕过
    3. DANGEROUS：
       a. 本会话已批准过该 pattern_key → 放行
       b. 否则调 ``_prompt_approval`` 问用户

    Args:
        command: bash 工具传入的 shell 命令字符串。

    Returns:
        ``None`` = 放行；非空字符串 = 拒绝消息（写给模型看）。
    """
    if not command or not command.strip():
        return None  # 空命令不拦（bash 工具自己会处理）

    # 1. sudo stdin guard
    blocked, sudo_desc = _check_sudo_stdin_guard(command)
    if blocked:
        return (
            f"拒绝: {sudo_desc}。\n"
            "不要向 'sudo -S' 管道传输密码 —— 这是暴力破解向量。"
            "当前 bash 工具不注入 sudo 密码；如确需提权，请在 agent 外手动执行。"
        )

    # 2. HARDLINE 地板
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        return (
            f"拒绝 (硬地板): {hardline_desc}。\n"
            "此命令在无条件黑名单上，不可通过 agent 执行。"
            "如确需运行，请在终端手动执行（agent 之外）。"
        )

    # 3. DANGEROUS 审批
    is_dangerous, pattern_key, dangerous_desc = detect_dangerous_command(command)
    if not is_dangerous:
        return None  # 无命中，放行

    assert pattern_key is not None and dangerous_desc is not None
    
    # 3a. 本会话已批准
    if _is_approved(pattern_key):
        return None

    # 3b. 问用户
    return _prompt_approval(command, pattern_key, dangerous_desc)
