"""guardrail 单元测试（朴素脚本版，无 pytest 依赖）。

只 import ToolCallGuardrailController（不调 API、不跑 agent_loop、不连真实工具）。
在 hello-agent 根目录运行：``.venv/bin/python tests/test_tool_guardrails.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tool_guardrails import ToolCallGuardrailController


# ── 工具小函数：跑一个测试，红/绿打印 ────────────────────────────────────
def run_test(name, fn):
    try:
        fn()
        print(f"  \033[32m✓ {name}\033[0m")
    except AssertionError as e:
        print(f"  \033[31m✗ {name}: {e}\033[0m")
        raise SystemExit(1)


# ── 测试 1：精确失败循环 ────────────────────────────────────────────────
# 同一 (bash, {"command": "rm x"}) 失败 5 次 → after_call 应在 count=5 时不 block，
# 但下一次 before_call 应该 block（exact_failure_block_after=5）
def test_exact_failure_blocks():
    c = ToolCallGuardrailController()
    c.reset_for_turn()
    args = {"command": "rm test.txt"}
    for _ in range(5):
        c.after_call("bash", args, '{"error": "Permission denied"}', failed=True)
    # 第 6 次 before_call 应被 block
    before = c.before_call("bash", args)
    assert before.action == "block", f"期望 block，实际 {before.action}"
    assert before.code == "repeated_exact_failure_block", f"code 错: {before.code}"
    assert before.count == 5, f"count 错: {before.count}"


# ── 测试 2：同工具失败循环（按 name，不限 args）─────────────────────────
# 同 name 不同 args 失败 8 次 → after_call 第 8 次返回 halt
# （same_tool_failure_halt_after=8）
def test_same_tool_failure_halt():
    c = ToolCallGuardrailController()
    c.reset_for_turn()
    for i in range(7):
        args = {"command": f"cmd_{i}"}  # 每次不同 args，避免命中精确计数
        c.after_call("bash", args, '{"error": "x"}', failed=True)
    # 第 8 次应触发 halt
    args = {"command": "cmd_8"}
    decision = c.after_call("bash", args, '{"error": "x"}', failed=True)
    assert decision.action == "halt", f"期望 halt，实际 {decision.action}"
    assert decision.code == "same_tool_failure_halt", f"code 错: {decision.code}"
    assert c.halt_decision is not None, "halt 应锁存到 halt_decision"
    assert c.halt_decision is decision or c.halt_decision.code == "same_tool_failure_halt", \
        "halt_decision 应是首次 halt 决策"


# ── 测试 3：只读无进展（read_file 同 args 同 result 多次）────────────────
# read_file 是 IDEMPOTENT。返回相同成功结果 5 次 → 前 4 次 after_call 记无进展，
# 第 5 次 after_call 后仍是 warning；但第 5 次 before_call（再调时）应被 block
def test_idempotent_no_progress_blocks():
    c = ToolCallGuardrailController()
    c.reset_for_turn()
    args = {"path": "config.yaml"}
    same_result = '{"content": "same old"}'
    # 走 5 次 after_call 成功路径，建立无进展计数
    for i in range(5):
        c.after_call("read_file", args, same_result, failed=False)
    # 第 6 次 before_call — 因 repeat_count 已达 5，应 block
    before = c.before_call("read_file", args)
    assert before.action == "block", f"期望 block，实际 {before.action}"
    assert before.code == "idempotent_no_progress_block", f"code 错: {before.code}"


# ── 测试 4：reset_for_turn 清空状态 ─────────────────────────────────────
# 防止跨 turn 计数残留
def test_reset_clears_state():
    c = ToolCallGuardrailController()
    c.reset_for_turn()
    # 制造一些失败
    for i in range(5):
        c.after_call("bash", {"command": f"x{i}"}, '{"error": "x"}', failed=True)
    assert c.halt_decision is None  # 5 次还没 halt（halt 是 8 次）
    assert len(c._exact_failure_counts) > 0, "应有计数"
    # reset
    c.reset_for_turn()
    assert c.halt_decision is None
    assert len(c._exact_failure_counts) == 0, "reset 后计数应清空"
    assert len(c._same_tool_failure_counts) == 0
    assert len(c._no_progress) == 0


# ── 测试 5：warn 阈值（同 args 失败 2 次 → warn，未达 block）─────────────
def test_warn_at_threshold():
    c = ToolCallGuardrailController()
    c.reset_for_turn()
    args = {"command": "rm x"}
    # 第 1 次：count=1，allow（<2 warn 阈值）
    d1 = c.after_call("bash", args, '{"error": "x"}', failed=True)
    assert d1.action == "allow", f"第 1 次应 allow，实际 {d1.action}"
    # 第 2 次：count=2，warn（exact_failure_warn_after=2）
    d2 = c.after_call("bash", args, '{"error": "x"}', failed=True)
    assert d2.action == "warn", f"第 2 次应 warn，实际 {d2.action}"
    assert d2.code == "repeated_exact_failure_warning", f"code 错: {d2.code}"
    # 此时 before_call 还不应该 block（count=2，<5 阈值）
    before = c.before_call("bash", args)
    assert before.allows_execution, f"count=2 时应允许，实际 {before.action}"


# ── 测试 6：成功路径不计失败（确保不影响后续）────────────────────────────
def test_success_does_not_count_as_failure():
    c = ToolCallGuardrailController()
    c.reset_for_turn()
    args = {"path": "foo.py"}
    # 成功调用 5 次
    for _ in range(5):
        c.after_call("read_file", args, '{"content": "hi"}', failed=False)
    # 没有失败计数
    assert len(c._exact_failure_counts) == 0, "成功不应进失败计数"
    # 无进展计数有（read_file 成功也记，因 result 相同）
    before = c.before_call("read_file", args)
    # 第 5 次后无进展 repeat_count=5，before_call 应 block
    assert before.action == "block", f"无进展应 block，实际 {before.action}"


def test_success_resets_failure_streak():
    c = ToolCallGuardrailController()
    args = {"command": "flaky-command"}
    c.after_call("bash", args, '{"error": "temporary"}', failed=True)
    c.after_call("bash", args, '{"exit_code": 0}', failed=False)
    assert c.before_call("bash", args).action == "allow"
    assert len(c._exact_failure_counts) == 0
    assert len(c._same_tool_failure_counts) == 0


# ── 跑 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== guardrail 单元测试 ===\n")
    run_test("精确失败循环 → block",                   test_exact_failure_blocks)
    run_test("同工具失败循环 → halt",                  test_same_tool_failure_halt)
    run_test("只读无进展 → block",                     test_idempotent_no_progress_blocks)
    run_test("reset_for_turn 清空状态",                test_reset_clears_state)
    run_test("warn 阈值（count=2 时 warn 不 block）",   test_warn_at_threshold)
    run_test("成功路径不计失败",                       test_success_does_not_count_as_failure)
    run_test("成功会重置连续失败计数",                 test_success_resets_failure_streak)
    print("\n\033[32m🎉 所有测试通过\033[0m\n")
