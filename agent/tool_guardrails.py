"""工具调用重复检测控制器（per-turn 状态机）。

核心状态机和 ``tool_loop_guardrails`` YAML 结构参考 Hermes
``agent/tool_guardrails.py``；护栏配置统一从 ``config.yaml`` 加载。

重复检测分为三类：
- **精确失败重复**：同一工具、相同参数连续失败，先警告，达到上限后 block。在工具调用结果提示正确情况下，用来检测模型是否重复输出相同调用。
- **同工具失败重复**：同一工具即使不断更换参数仍持续失败，达到上限后 halt。这个可以用来检测某个工具是否有问题。
- **只读无进展**：相同只读调用反复返回相同结果，先警告，达到上限后 block。

决策共有四种：
- ``allow``：允许执行 handler。
- ``warn``：允许执行 handler，并在工具结果中加入换策略提示。
- ``block``：在执行前拦截当前调用，不执行 handler，改为返回 synthetic result。
- ``halt``：在执行后发现同一工具累计失败过多；当前工具批次处理完成后，
  主循环不再进入下一轮模型工具调用。

``block`` 和 ``halt`` 的触发时机不同，但都会写入 ``halt_decision``：
- ``block`` 发生在 ``before_call()``，拦住尚未执行的重复调用；
- ``halt`` 发生在 ``after_call()``，此时当前调用已经执行完毕。

主循环在每批工具调用处理完成后检查 ``halt_decision``。因此，同一批里已经
通过预检的其他调用仍会按 Hermes 的批次语义执行；只要批次结束时已有停止
决策，就结束当前 turn。控制器只保留本 turn 内出现的第一个停止决策，并在
下一条用户请求开始时重置。

与 Hermes 完整实现相比，本项目未包含插件回调和独立观测存储；Hermes 的
``terminal`` 工具在本项目中对应 ``bash``。
"""

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from agent.tool_result_classification import classify_tool_result


# ════════════════════════════════════════════════════════════════════════
# 只读工具白名单
# ════════════════════════════════════════════════════════════════════════
# 名单内的工具会额外检测“相同参数反复返回相同结果”，用于识别无进展读取。
# 名单外的工具不做结果 hash 比较，但工具调用失败仍会进入精确失败和同工具失败计数。

IDEMPOTENT_TOOL_NAMES = frozenset({
    "read_file",
    "search_files",
    "web_search",
    "web_extract",
})

# ════════════════════════════════════════════════════════════════════════
# 配置（Hermes YAML 结构）
# ════════════════════════════════════════════════════════════════════════
# 阈值试验调出来的经验值。对齐 hermes ToolCallGuardrailConfig 默认。
# warnings 默认开启，hard stop 默认关闭；都可在 config.yaml 中显式配置。

@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """重复检测阈值。"""
    warnings_enabled: bool = True
    hard_stop_enabled: bool = False          # 是否开启强制 block/halt
    exact_failure_warn_after: int = 2       # 在 2 次精确工具调用失败后触发 → warn
    exact_failure_block_after: int = 5      # 同 (name, args) 失败 5 次 → block
    same_tool_failure_warn_after: int = 3   # 同 name 失败 3 次 → warn
    same_tool_failure_halt_after: int = 8   # 同 name 失败 8 次 → halt
    no_progress_warn_after: int = 2         # 只读无进展 2 次 → warn
    no_progress_block_after: int = 5        # 只读无进展 5 次 → block

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ToolCallGuardrailConfig":
        """从 YAML 配置中 tool_loop_guardrails 对应的那一层字典加载配置。"""
        if not isinstance(data, Mapping):
            return cls()

        warn_after = data.get("warn_after")
        if not isinstance(warn_after, Mapping):
            warn_after = {}
        hard_stop_after = data.get("hard_stop_after")
        if not isinstance(hard_stop_after, Mapping):
            hard_stop_after = {}
        defaults = cls()
        return cls(
            warnings_enabled=_as_bool(
                data.get("warnings_enabled"), defaults.warnings_enabled
            ),
            hard_stop_enabled=_as_bool(
                data.get("hard_stop_enabled"), defaults.hard_stop_enabled
            ),
            exact_failure_warn_after=_positive_int(
                warn_after.get("exact_failure", data.get("exact_failure_warn_after")),
                defaults.exact_failure_warn_after,
            ),
            same_tool_failure_warn_after=_positive_int(
                warn_after.get(
                    "same_tool_failure", data.get("same_tool_failure_warn_after")
                ),
                defaults.same_tool_failure_warn_after,
            ),
            no_progress_warn_after=_positive_int(
                warn_after.get(
                    "idempotent_no_progress", data.get("no_progress_warn_after")
                ),
                defaults.no_progress_warn_after,
            ),
            exact_failure_block_after=_positive_int(
                hard_stop_after.get(
                    "exact_failure", data.get("exact_failure_block_after")
                ),
                defaults.exact_failure_block_after,
            ),
            same_tool_failure_halt_after=_positive_int(
                hard_stop_after.get(
                    "same_tool_failure", data.get("same_tool_failure_halt_after")
                ),
                defaults.same_tool_failure_halt_after,
            ),
            no_progress_block_after=_positive_int(
                hard_stop_after.get(
                    "idempotent_no_progress", data.get("no_progress_block_after")
                ),
                defaults.no_progress_block_after,
            ),
        )

# ════════════════════════════════════════════════════════════════════════
# 调用指纹 + 决策（dataclass）
# ════════════════════════════════════════════════════════════════════════

def _sha256(s: str) -> str:
    """稳定哈希（用于 args 规范化后压缩成指纹）。"""
    # 使用 utf-8 先编码为字节流，然后使用 sha256 哈希算法编码，在转成 16 进制，最后取哈希值的前 16 位
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """args 序列化：转化为 json 字符串，（防 dict 顺序变化导致指纹不同）。"""
    return json.dumps(
        dict(args or {}),
        ensure_ascii=False,  # 允许输出包含非 ASCII 字符（如中文），而不是转义成 \uXXXX 形式
        sort_keys=True,  # 按字典的键字母顺序排序后再序列化
        separators=(",", ":"),  # 去掉 JSON 中多余的空白字符（空格），含义就是以在字典每项之间的分割符就是","，没有空格，另一个同理
        default=str,    # 非 JSON 原生类型（如 Path）转 str
    )


@dataclass(frozen=True)
class ToolCallSignature:
    """一次工具调用的唯一标识。

    由工具名称和参数哈希组成，用于识别是否重复调用了同一个工具。
    """
    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        """初始化一个 ToolCallSignature 类对象，并计算好参数哈希值和工具名称"""
        return cls(tool_name=tool_name, args_hash=_sha256(canonical_tool_args(args or {})))


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """护栏做出的处理决定。"""
    action: str = "allow"   # allow | warn | block | halt
    code: str = "allow"     # 机器可读的决策码（如 repeated_exact_failure_block），代码可以据此判断
    message: str = ""       # 人和模型可读的具体说明，解释发生了什么以及下一步怎么做。
    tool_name: str = ""
    count: int = 0
    signature: Optional[ToolCallSignature] = None

    @property
    def allows_execution(self) -> bool:
        """判断 action 的值是否为 allow 或 warn。"""
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        """判断 action 的值是否为 block 或 halt。"""
        return self.action in {"block", "halt"}

    def to_metadata(self) -> dict[str, Any]:
        """把内部的 ToolGuardrailDecision 对象转换成普通字典，方便记录、传输和序列化

        Returns:
            dict[str, Any]: 转化后的字典对象
        """
        data: dict[str, Any] = {
            "action": self.action,
            "code": self.code,
            "message": self.message,
            "tool_name": self.tool_name,
            "count": self.count,
        }
        if self.signature is not None:
            data["signature"] = {
                "tool_name": self.signature.tool_name,
                "args_hash": self.signature.args_hash,
            }
        return data

# ════════════════════════════════════════════════════════════════════════
# 失败分类（兼容入口）
# ════════════════════════════════════════════════════════════════════════
# 真正分类位于 ``tool_result_classification``；本函数保留为 Hermes 同签名兜底入口。
def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """独立调用方的失败分类兜底。"""
    outcome = classify_tool_result(tool_name, result)
    if not outcome.counts_as_failure:
        return False, ""
    if outcome.exit_code is not None:
        return True, f" [exit {outcome.exit_code}]"
    return True, " [error]"


def _result_hash(result: str | None) -> str:
    """对 JSON 结果按语义哈希，避免字段顺序或空白变化绕过无进展检测。"""
    raw = result or ""
    try:
        parsed = json.loads(raw)
        canonical = json.dumps(
            parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        canonical = raw
    return _sha256(canonical)


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
        """加载工具调用护栏配置，申请一个线程锁对象，重置不同方式工具调用失败次数（精准调用失败、相同工具名调用失败、只读工具无进展、终止工具调用的决策）

        Args:
            config (Optional[ToolCallGuardrailConfig], optional): ToolCallGuardrailConfig 对象，默认为 None
        """
        self.config = config or ToolCallGuardrailConfig()
        self._halt_lock = threading.Lock()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        """重置不同方式工具调用失败次数。"""
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._halt_decision: Optional[ToolGuardrailDecision] = None

    @property  # 把方法变成属性
    def halt_decision(self) -> Optional[ToolGuardrailDecision]:
        """已锁存（写入后无法修改）的 halt 决策（loop 顶部检查用）。"""
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
        """工具执行前调用：判断精确失败 / 只读无进展是否达到阈值。

        Returns:
            ToolGuardrailDecision 类对象。
        """
        signature = ToolCallSignature.from_call(tool_name, args)
        # 工具护栏的配置
        cfg = self.config

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
            ToolGuardrailDecision。
        """
        signature = ToolCallSignature.from_call(tool_name, args)
        cfg = self.config

        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)

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
                message=_tool_failure_recovery_hint(tool_name, same_count),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )

            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        # 工具调用成功会终止对应工具的连续失败计数。
        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        # 只读成功路径：记录结果 hash，供下次 before_call 判"只读无进展"
        result_hash = _result_hash(result)
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

def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """ 预检时调用 before_call 后，decision 的 action 字段为 block 时，合成假的工具调用 result 字符串（替代真执行）。

    合成规则：
    1. "error": decision.message,
    2. "guardrail": decision.to_metadata(),
    3. "blocked": True,

    Args:
        decision (ToolGuardrailDecision): 工具护栏决策

    Returns:
        str: json 字符串
    """
    return json.dumps({
        "error": decision.message,
        "guardrail": decision.to_metadata(),
        "blocked": True,
    }, ensure_ascii=False)


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """warn/halt 决策时，把引导文本追加到 result 末尾给模型看。

    action=warn：追加后缀，放行执行（result 已是真结果）
    action=halt：同样是追加后缀（result 是真结果，但 turn 即将终止）
    """
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    suffix = f"\n\n[{label}: {decision.code}; count={decision.count}; {decision.message}]"
    if result is None:
        # .lstrip：删除空白符号、换行符号、制表符
        return suffix.lstrip()
    return result + suffix


def _tool_failure_recovery_hint(tool_name: str, count: int) -> str:
    """为重复失败提供可执行的下一步，而不只是要求模型“重试”。"""
    common = (
        f"{tool_name} 本 turn 内失败 {count} 次。先检查最新错误和假设，"
        "不要原样重试。"
    )
    if tool_name == "bash":
        return common + "可先用 pwd && ls -la 诊断，再尝试绝对路径、更小的命令或文件工具。"
    if tool_name in {"read_file", "search_files", "write_file", "patch"}:
        return common + "请重新读取目标，确认路径和当前文本，缩小查询或修改范围。"
    if tool_name in {"web_search", "web_extract"}:
        return common + "请缩小关键词、更换查询或使用已获得的结果。"
    return common + "请改用不同参数、更窄的查询或其他工具。"


def _as_bool(value: Any, default: bool) -> bool:
    """安全解析 YAML 布尔值。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _positive_int(value: Any, default: int) -> int:
    """安全解析正整数阈值，非法值回退默认值。"""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default
