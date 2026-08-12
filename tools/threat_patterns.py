"""共享的上下文威胁模式库；记忆写入使用最严格扫描范围。"""

from __future__ import annotations

import re


# (regex, pattern_id, scope)。all ⊂ context ⊂ strict。
_PATTERNS = [
    (r"ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+(?:\w+\s+)*instructions", "prompt_injection", "all"),
    (r"system\s+prompt\s+override", "sys_prompt_override", "all"),
    (r"disregard\s+(?:\w+\s+)*(your|all|any)\s+(?:\w+\s+)*(instructions|rules|guidelines)", "disregard_rules", "all"),
    (r"act\s+as\s+(if|though)\s+(?:\w+\s+)*you\s+(?:\w+\s+)*(have\s+no|don't\s+have)\s+(?:\w+\s+)*(restrictions|limits|rules)", "bypass_restrictions", "all"),
    (r"<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->", "html_comment_injection", "all"),
    (r"<\s*div\s+style\s*=\s*['\"][\s\S]*?display\s*:\s*none", "hidden_div", "all"),
    (r"translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)", "translate_execute", "all"),
    (r"do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user", "deception_hide", "all"),
    (r"you\s+are\s+(?:\w+\s+)*now\s+(?:a|an|the)\s+", "role_hijack", "context"),
    (r"pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+", "role_pretend", "context"),
    (r"output\s+(?:\w+\s+)*(system|initial)\s+prompt", "leak_system_prompt", "context"),
    (r"(respond|answer|reply)\s+without\s+(?:\w+\s+)*(restrictions|limitations|filters|safety)", "remove_filters", "context"),
    (r"you\s+have\s+been\s+(?:\w+\s+)*(updated|upgraded|patched)\s+to", "fake_update", "context"),
    (r"\bname\s+yourself\s+\w+", "identity_override", "context"),
    (r"register\s+(as\s+)?a?\s*node", "c2_node_registration", "context"),
    (r"(heartbeat|beacon|check[\s\-]?in)\s+(to|with)\s+", "c2_heartbeat", "context"),
    (r"pull\s+(down\s+)?(?:new\s+)?task(?:ing|s)?\b", "c2_task_pull", "context"),
    (r"connect\s+to\s+the\s+network\b", "c2_network_connect", "context"),
    (r"you\s+must\s+(?:\w+\s+){0,3}(register|connect|report|beacon)\b", "forced_action", "context"),
    (r"only\s+use\s+one[\s\-]?liners?\b", "anti_forensic_oneliner", "context"),
    (r"never\s+(?:\w+\s+)*(?:create|write)\s+(?:\w+\s+)*(?:script|file)\s+(?:\w+\s+)*disk", "anti_forensic_disk", "context"),
    (r"unset\s+\w*(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC)\w*", "env_var_unset_agent", "context"),
    (r"\b(?:cobalt\s*strike|sliver|havoc|mythic|metasploit|brainworm)\b", "known_c2_framework", "context"),
    (r"\bc2\s+(?:server|channel|infrastructure|beacon)\b", "c2_explicit", "context"),
    (r"\bcommand\s+and\s+control\b", "c2_explicit_long", "context"),
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_curl", "all"),
    (r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_wget", "all"),
    (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)", "read_secrets", "all"),
    (r"(send|post|upload|transmit)\s+.*\s+(to|at)\s+https?://", "send_to_url", "strict"),
    (r"(include|output|print|share)\s+(?:\w+\s+)*(conversation|chat\s+history|previous\s+messages|full\s+context|entire\s+context)", "context_exfil", "strict"),
    (r"authorized_keys", "ssh_backdoor", "strict"),
    (r"\$HOME/\.ssh|~/\.ssh", "ssh_access", "strict"),
    (r"\$HOME/\.athena/\.env|~/\.athena/\.env", "athena_env", "strict"),
    (r"(update|modify|edit|write|change|append|add\s+to)\s+.*(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)", "agent_config_mod", "strict"),
    (r"(update|modify|edit|write|change|append|add\s+to)\s+.*\.athena/(config\.yaml|SOUL\.md)", "athena_config_mod", "strict"),
    (r"(?:api[_-]?key|token|secret|password)\s*[=:]\s*['\"][A-Za-z0-9+/=_-]{20,}", "hardcoded_secret", "strict"),
]

INVISIBLE_CHARS = frozenset({
    "\u200b", "\u200c", "\u200d", "\u2060", "\u2062", "\u2063", "\u2064",
    "\ufeff", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u2066",
    "\u2067", "\u2068", "\u2069",
})

_COMPILED: dict[str, list[tuple[re.Pattern, str]]] = {}


def _compile() -> None:
    if _COMPILED:
        return
    groups = {"all": [], "context": [], "strict": []}
    for pattern, pattern_id, scope in _PATTERNS:
        entry = (re.compile(pattern, re.IGNORECASE), pattern_id)
        if scope == "all":
            targets = ("all", "context", "strict")
        elif scope == "context":
            targets = ("context", "strict")
        elif scope == "strict":
            targets = ("strict",)
        else:
            raise ValueError(f"unknown threat-pattern scope: {scope}")
        for target in targets:
            groups[target].append(entry)
    _COMPILED.update(groups)


_compile()


def scan_for_threats(content: str, scope: str = "context") -> list[str]:
    """返回命中的稳定 pattern id；同时检测隐形与双向 Unicode。"""
    if not content:
        return []
    patterns = _COMPILED.get(scope)
    if patterns is None:
        raise ValueError(f"unknown threat-pattern scope: {scope}")
    findings = [
        f"invisible_unicode_U+{ord(char):04X}"
        for char in set(content) & INVISIBLE_CHARS
    ]
    findings.extend(pattern_id for pattern, pattern_id in patterns if pattern.search(content))
    return findings


def first_threat_message(content: str, scope: str = "strict") -> str | None:
    findings = scan_for_threats(content, scope)
    if not findings:
        return None
    pattern_id = findings[0]
    if pattern_id.startswith("invisible_unicode_"):
        return f"Blocked: content contains invisible unicode character {pattern_id.removeprefix('invisible_unicode_')} (possible injection)."
    return (
        f"Blocked: content matches threat pattern '{pattern_id}'. Content is injected "
        "into the system prompt and must not contain injection or exfiltration payloads."
    )


__all__ = ["INVISIBLE_CHARS", "scan_for_threats", "first_threat_message"]
