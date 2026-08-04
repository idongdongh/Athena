"""工具调用重复检测控制器（per-turn 状态机）。

对齐 hermes ``agent/tool_guardrails.py`` 的核心状态机，省略当前项目不需要的配置加载和观测字段：
- ``ToolCallGuardrailConfig.from_mapping``（从 config.yaml 加载）——全部用默认值，YAGNI
- Hermes 的 ``terminal`` 对应本项目的 ``bash``，保留 ``exit_code`` 失败分类
- ``file_mutation_result_landed`` 白名单（file 写入成功不算失败的特判）——本项目无
- ``_tool_failure_recovery_hint`` per-tool 提示文案——本项目无
- metadata / observability 字段——本项目无 observability

保留的核心（4 类重复检测）：
- **精确失败重复**：同一 (name, args hash) 失败 N 次 → warn → block
- **同工具失败重复**：同一 name（无论参数）失败 N 次 → warn → halt turn
- **只读无进展**：只读工具 (name, args hash) 返回相同结果 N 次 → warn → block
- **halt latch**：首次 halt 决策锁存，loop 顶部检查后整 turn 终止

四级 action：``allow | warn | block | halt``
- ``allow``：放行
- ``warn``：放行 + 结果追加引导文本
- ``block``：阻止，拒绝单次调用，返回 synthetic result（不执行 handler）
- ``halt``：停止，标记 `_halt_decision`，loop 顶部见到后整 turn 退出
block / halt 都会由控制器锁存到 ``halt_decision``，供主循环结束当前 turn。
"""

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Optional


# ════════════════════════════════════════════════════════════════════════
# 工具分类（IDEMPOTENT 只读 vs MUTATING 写）
# ════════════════════════════════════════════════════════════════════════
# 两类工具的"重复"含义不同：
# - 只读工具重复 = "数据不变你已读过了"（看结果 hash，不看失败次数）
# - 写工具重复   = "持续失败整工具坏了"（看失败次数，不看结果 hash）
# 这是 hermes 的核心设计：分类决定检测策略。

IDEMPOTENT_TOOL_NAMES = frozenset({
    "read_file",
    "search_files",
    "web_search",
    "web_extract",
})

# ════════════════════════════════════════════════════════════════════════
# 配置（全部默认值，砍 config.yaml 加载）
# ════════════════════════════════════════════════════════════════════════
# 阈值试验调出来的经验值。对齐 hermes ToolCallGuardrailConfig 默认。
# warn 永远开（不阻断，只追加提示）；hard_stop 默认也开（CLI 阶段够用）。

@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """重复检测的 7 个阈值。全部默认值，不可配置（砍 from_mapping）。"""
    warnings_enabled: bool = True
    hard_stop_enabled: bool = True          # 开启强制阻断模式，也就是默认开 block/halt
    exact_failure_warn_after: int = 2       # 在 2 次精确工具调用失败后触发 → warn
    exact_failure_block_after: int = 5      # 同 (name, args) 失败 5 次 → block
    same_tool_failure_warn_after: int = 3   # 同 name 失败 3 次 → warn
    same_tool_failure_halt_after: int = 8   # 同 name 失败 8 次 → halt
    no_progress_warn_after: int = 2         # 只读无进展 2 次 → warn
    no_progress_block_after: int = 5        # 只读无进展 5 次 → block


# ════════════════════════════════════════════════════════════════════════
# 调用指纹 + 决策（dataclass）
# ════════════════════════════════════════════════════════════════════════

def _sha256(s: str) -> str:
    """稳定哈希（用于 args 规范化后压缩成指纹）。"""
    # 使用 utg-8 先编码为字节流，然后使用 sha256 哈希算法编码，在转成 16 进制，最后取哈希值的前 16 位
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _canonical_args(args: Mapping[str, Any]) -> str:
    """args 系列化：转化为 json 字符串，（防 dict 顺序变化导致指纹不同）。"""
    return json.dumps(
        dict(args or {}),
        ensure_ascii=False,  # 允许输出包含非 ASCII 字符（如中文），而不是转义成 \uXXXX 形式
        sort_keys=True,  # 按字典的键字母顺序排序后再序列化
        separators=(",", ":"),  # 去掉 JSON 中多余的空白字符（空格），含义就是以在字典每项之间的分割符就是","，没有空格，另一个同理
        default=str,    # 非 JSON 原生类型（如 Path）转 str
    )


@dataclass(frozen=True)
class ToolCallSignature:
    """一次调用的稳定指纹：tool_name + args hash。

    dataclass(frozen=True) → 可哈希 → 可当 dict 的 key。
    """
    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        """初始化一个 ToolCallSignature 实例，并初始化好参数哈希值和工具名称"""
        return cls(tool_name=tool_name, args_hash=_sha256(_canonical_args(args or {})))


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """控制器返回的决策。"""
    action: str = "allow"   # allow | warn | block | halt
    code: str = "allow"     # 机器可读的决策码（如 repeated_exact_failure_block）
    message: str = ""       # 给模型/用户的人读消息
    tool_name: str = ""
    count: int = 0
    signature: Optional[ToolCallSignature] = None

    @property
    def allows_execution(self) -> bool:
        """True = 可以执行 handler（allow 或 warn）。"""
        return self.action in {"allow", "warn"}

# ════════════════════════════════════════════════════════════════════════
# 失败分类（通用版，砍工具特化）
# ════════════════════════════════════════════════════════════════════════
# 与 Hermes ``classify_tool_failure`` 保持同一契约：shell 工具看 exit_code，
# 其他工具再使用结构化 error 和字符串启发式兜底。

def is_tool_failure(tool_name: str, result: str | None) -> bool:
    """判断是否工具调用失败：通过判断工具调用结果中是否包含 error 和 failed 等来判断工具调用结果是否为失败。

    Args:
        tool_name: 工具名。
        result: 工具返回字符串。

    Returns:
        True = 失败；False = 成功（含空结果——空不算失败）。

    Note:
        空结果（如 ``{"results": [], "count": 0}``）**不算失败**——工具执行成功了，
        只是没数据。空结果归 ``_no_progress`` 检测管（只读工具无进展），不归这里。
    """
    if result is None:
        return False
    # 尝试解析 JSON，看是否有 "error" / "message" 字段
    try:
        data = json.loads(result) if isinstance(result, str) else None
    except (json.JSONDecodeError, ValueError):
        data = None

    if isinstance(data, dict):
        if tool_name == "bash":
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True
        err = data.get("error") or data.get("message")
        if err and (data.get("success") is False or "error" in data):
            return True

    # 通用启发式：非 JSON 字符串以 "Error" 开头，或前 500 字含 '"error"' / '"failed"'
    if isinstance(result, str):
        lower = result[:500].lower()
        if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
            return True

    return False


# ════════════════════════════════════════════════════════════════════════
# 控制器（per-turn 状态机）
# ════════════════════════════════════════════════════════════════════════

# 工具调用护栏控制器
class ToolCallGuardrailController:
    """Per-turn 工具调用重复检测控制器。

    每个 turn（用户一条 query）开始时 ``reset_for_turn()``。
    每次工具调用前后分别调 ``before_call`` / ``after_call``。
    loop 在每批工具执行后检查 ``halt_decision``，非 None 时结束当前 turn。
    """

    def __init__(self, config: Optional[ToolCallGuardrailConfig] = None):
        self.config = config or ToolCallGuardrailConfig()
        self._halt_lock = threading.Lock()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        """每个 turn 开头调用：清空 per-turn 计数器 + halt 锁存。"""
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._halt_decision: Optional[ToolGuardrailDecision] = None

    @property  # 把方法变成属性
    def halt_decision(self) -> Optional[ToolGuardrailDecision]:
        """已锁存的 halt 决策（loop 顶部检查用）。"""
        with self._halt_lock:
            return self._halt_decision

    @staticmethod  # 静态方法
    def _is_idempotent(tool_name: str) -> bool:
        """判断是否是只读方法"""
        return tool_name in IDEMPOTENT_TOOL_NAMES

    def _set_halt(self, decision: ToolGuardrailDecision) -> None:
        """锁存首个 halt 决策（多次 halt 只记一次）。"""
        with self._halt_lock:
            if self._halt_decision is None:
                self._halt_decision = decision

    # ── before_call：执行前检查 ───────────────────────────────────────────
    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        """工具执行前调用：判是否阻断（精确失败 / 只读无进展 达阈值）。

        Returns:
            ToolGuardrailDecision。allows_execution=False 时调用方应跳过 handler，
            返回 synthetic result；block/halt 会由控制器自动锁存。
        """
        signature = ToolCallSignature.from_call(tool_name, args)
        cfg = self.config

        # 如果没有配置就使用默认配置，现在只有默认配置，配置加载系统还没实现
        if not cfg.hard_stop_enabled:
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        # ① 精确失败达 block_after：同一 (name, args) 失败太多次 → block
        exact_count = self._exact_failure_counts.get(signature, 0)
        if exact_count >= cfg.exact_failure_block_after:
            decision = ToolGuardrailDecision(
                action="block",
                code="repeated_exact_failure_block",
                message=(
                    f"已阻断 {tool_name}：同一调用连续失败 {exact_count} 次。"
                    "停止重试同一参数；换策略或换工具。"
                ),
                tool_name=tool_name,
                count=exact_count,
                signature=signature,
            )
            self._set_halt(decision)
            return decision

        # ② 只读工具无进展达 block_after：同一 (name, args) 返回相同结果太多次 → block
        if self._is_idempotent(tool_name):
            record = self._no_progress.get(signature)
            if record is not None:
                _result_hash, repeat_count = record
                if repeat_count >= cfg.no_progress_block_after:
                    decision = ToolGuardrailDecision(
                        action="block",
                        code="idempotent_no_progress_block",
                        message=(
                            f"已阻断 {tool_name}：同一只读调用返回相同结果 {repeat_count} 次。"
                            "停止重复读同一信息；用已有结果或换查询。"
                        ),
                        tool_name=tool_name,
                        count=repeat_count,
                        signature=signature,
                    )
                    self._set_halt(decision)
                    return decision

        return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

    # ── after_call：执行后计数 ────────────────────────────────────────────
    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | None,
        *,
        failed: Optional[bool] = None,
    ) -> ToolGuardrailDecision:
        """工具执行后调用：根据失败/成功更新计数器。

        Args:
            failed: 显式失败标志（推荐传入，避免重复分类）；None 时用 classify_tool_failure。

        Returns:
            ToolGuardrailDecision。warn/block/halt 意味着计数已达阈值。
        """
        signature = ToolCallSignature.from_call(tool_name, args)
        cfg = self.config

        if failed is None:
            failed = is_tool_failure(tool_name, result)

        if failed:
            # 精确失败 +1；清无进展记录（失败时不应该计无进展）
            self._exact_failure_counts[signature] = self._exact_failure_counts.get(signature, 0) + 1
            self._no_progress.pop(signature, None)

            # 同工具失败（按 name，不限 args）+1
            same_count = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._same_tool_failure_counts[tool_name] = same_count

            # ① 同工具失败达 halt_after → halt（整 turn 终止）
            if cfg.hard_stop_enabled and same_count >= cfg.same_tool_failure_halt_after:
                decision = ToolGuardrailDecision(
                    action="halt",
                    code="same_tool_failure_halt",
                    message=(
                        f"已终止 {tool_name}：本 turn 内失败 {same_count} 次。"
                        "停止重试这一工具的任意路径，换不同方式。"
                    ),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )
                self._set_halt(decision)
                return decision

            # ② 同一 (name, args) 失败达 warn_after → warn
            exact_count = self._exact_failure_counts[signature]
            if cfg.warnings_enabled and exact_count >= cfg.exact_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="repeated_exact_failure_warning",
                    message=(
                        f"{tool_name} 已用相同参数失败 {exact_count} 次。"
                        "看起来像卡在重复——分析错误，换策略。"
                    ),
                    tool_name=tool_name,
                    count=exact_count,
                    signature=signature,
                )

            # ③ 同 name 失败（不限参数）达 warn_after → warn
            if cfg.warnings_enabled and same_count >= cfg.same_tool_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="same_tool_failure_warning",
                    message=(
                        f"{tool_name} 本 turn 内失败 {same_count} 次。"
                        "换路径或换工具。"
                    ),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )

            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        # Hermes 语义：成功会终止对应调用和工具的连续失败计数。
        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        # 只读成功路径：记录结果 hash，供下次 before_call 判"只读无进展"
        result_hash = _sha256(result or "")
        previous = self._no_progress.get(signature)
        if previous is not None and previous[0] == result_hash:
            repeat_count = previous[1] + 1
        else:
            repeat_count = 1
        self._no_progress[signature] = (result_hash, repeat_count)

        if cfg.warnings_enabled and self._is_idempotent(tool_name) and \
                repeat_count >= cfg.no_progress_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="idempotent_no_progress_warning",
                message=(
                    f"{tool_name} 返回相同结果 {repeat_count} 次。"
                    "你已读过此信息——用已有结果或换查询。"
                ),
                tool_name=tool_name,
                count=repeat_count,
                signature=signature,
            )

        return ToolGuardrailDecision(tool_name=tool_name, signature=signature)


# ════════════════════════════════════════════════════════════════════════
# 工具函数：合成 blocked result + 追加 warning 后缀
# ════════════════════════════════════════════════════════════════════════
# 对应 hermes ``_guardrail_block_result`` + ``append_toolguard_guidance``。

def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """合成 blocked/halt 调用的假 result 字符串（替代真执行）。

    调用方在 allows_execution=False 时用这个替代真 handler 返回值。
    """
    return json.dumps({
        "error": decision.message,
        "guardrail": decision.code,
        "blocked": True,
    }, ensure_ascii=False)


def append_guardrail_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """warn/halt 决策时，把引导文本追加到 result 末尾给模型看。

    action=warn：追加后缀，放行执行（result 已是真结果）
    action=halt：同样是追加后缀（result 是真结果，但 turn 即将终止）
    """
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    suffix = f"\n\n[{label}: {decision.code}; count={decision.count}; {decision.message}]"
    if result is None:
        return suffix.lstrip()
    return result + suffix
