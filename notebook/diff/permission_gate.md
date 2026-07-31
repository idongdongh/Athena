# 工具权限闸门对照：hello-agent vs hermes

> 文件位置：见下方「## hermes 实现方式」「## 我的项目实现方式」两节列出的绝对路径。
>
> 结论先行：**核心防御机制一致——HARDLINE 地板拒绝 + DANGEROUS 需审批 + 命令归一化防绕过 + 路径敏感文件保护；**
> 差异在 hermes 把闸门散到每个工具自管（40+ 工具、多后端、多攻击面），你用「统一 pre-tool-call 接缝 + table-driven 调度」集中处理（7 工具单进程更干净）；
> 你砍掉的是 hermes 的厚壳——gateway 异步审批、permanent allowlist 持久化、yolo/cron 多模式、tirith 集成、LLM 风险评估、approval hooks、observability、execute_code guard（本项目无该工具）。

## hermes 实现方式

三层防御 + 分散挂载。每个工具**自己** import 并调用对应的检查（terminal 调 `check_all_command_guards`、file 工具用 `check_fn=_check_file_reqs` 注册时挂、execute_code 调 `check_execute_code_guard`），因为 hermes 有 40+ 工具、多后端、多攻击面，集中接缝不够灵活。

- `/Users/idongdong/Documents/Projects/hermes-agent/tools/approval.py` — 2171 行，bash 命令闸门总仓库：`HARDLINE_PATTERNS`（12 条，:263）+ `DANGEROUS_PATTERNS`（47 条，:381）+ `_normalize_command_for_detection`（6 道清洗，:567）+ `_check_sudo_stdin_guard`（:310）+ `check_dangerous_command`（总入口，:1300）+ gateway 异步审批 + permanent allowlist + yolo/cron 多模式
- `/Users/idongdong/Documents/Projects/hermes-agent/tools/path_security.py` — 44 行，路径原语：`validate_within_dir` + `has_traversal_component`（给 skill/cron/credential 工具用，**file 工具不用它**）
- `/Users/idongdong/Documents/Projects/hermes-agent/tools/file_tools.py` — file 工具的实际路径闸门：`_check_sensitive_path`（:443，敏感系统路径前缀/精确匹配 + Hermes config 保护）+ `_SENSITIVE_PATH_PREFIXES`（:416）+ `_is_blocked_device`（:374，设备文件拦截，给 read 用）+ 注册时挂 `check_fn=_check_file_reqs`（:1890）
- `/Users/idongdong/Documents/Projects/hermes-agent/tools/ansi_strip.py` — 45 行，`strip_ansi`（全 ECMA-48 ANSI 转义剥离，归一化的第 1 道清洗用它）
- `/Users/idongdong/Documents/Projects/hermes-agent/tools/code_execution_tool.py:1117` — execute_code 专用的 `check_execute_code_guard`（静态扫 Python 脚本危险模式）

## 我的项目实现方式

**触发**：模型回 `tool_use` → `run_tool_calls` → `dispatch_tool_call(name, args)` → 在执行 `entry.handler(**args)` **之前**调 `check_tool_permission(name, args)`。被拦（返回非 None）则不执行 handler，拒绝消息直接作为 `tool_result` 回写给模型。

**调用链**：
1. `dispatch_tool_call(name, args)`（`tool_executor.py:28`）取 tracer + registry entry；
2. 调 `check_tool_permission(name, args)`（`tools/_permission.py`）—— table-driven 调度，按工具名查 `_CHECKERS` 字典；
3. `bash` → `check_command(args["command"])`（`tools/approval.py`）；
4. `write_file` / `patch` → `check_write_path(args["path"])`（`tools/path_security.py`）；
5. 其他工具（read/search/web）→ `_CHECKERS` 无配置 = 返回 None = 放行；
6. 返回 None 放行 → 继续 `entry.handler(**args)`；返回 str 拦截 → 写 trace + 直接 return 给模型。

砍掉 hermes 厚壳：无 gateway 异步审批（`_await_gateway_decision`/`submit_pending`）、无 permanent allowlist 持久化（会话记忆够用）、无 yolo/cron/gateway 多模式、无 tirith 安全扫描、无 `_smart_approve`（LLM 风险评估）、无 approval hooks/observability、无 pattern_key aliases（legacy 兼容）、无 execute_code guard（本项目无该工具）。

- `/Users/idongdong/Documents/Projects/hello-agent/tools/_permission.py` — 调度层：`_CHECKERS` 字典（table-driven）+ `check_tool_permission(name, args)` 总入口。加新工具的检查只改字典一行
- `/Users/idongdong/Documents/Projects/hello-agent/tools/approval.py` — bash 命令闸门：`HARDLINE_PATTERNS`（12 条）+ `DANGEROUS_PATTERNS`（高频子集 ~20 条）+ `_normalize_command_for_detection`（6 道清洗）+ `_check_sudo_stdin_guard` + `detect_hardline_command`/`detect_dangerous_command` + `_session_approved` 会话记忆 + `check_command` 总入口
- `/Users/idongdong/Documents/Projects/hello-agent/tools/path_security.py` — file 工具路径安全：`validate_within_dir` + `has_traversal_component`（搬 hermes）+ `_SENSITIVE_PATH_PREFIXES`/`_SENSITIVE_EXACT_PATHS`（搬 file_tools）+ `_SENSITIVE_USER_WRITES`（ssh/env/credentials）+ `check_write_path`
- `/Users/idongdong/Documents/Projects/hello-agent/tool_executor.py` — `dispatch_tool_call` 加 3 行 hook 接缝（:48-51）：调 `check_tool_permission`，被拦则记 trace + return

---

## 相同点（与 hermes 一致）

下列维度你的实现与 hermes 逐字节一致或等价，核心防御行为无差。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| HARDLINE 地板拒绝 | 12 条正则（`approval.py:263`），无条件拒绝，连 yolo 都不能绕 | 同（12 条逐条搬） | 一致：rm -rf /、mkfs、dd 写块设备、fork bomb、shutdown 等不可逆损害一律拒 |
| 命令归一化防绕过 | 6 道清洗（`approval.py:567`）：strip_ansi/null/NFKC/home 折叠/反斜杠/空串 | 同（6 道全搬，home 折叠简化版砍 Windows） | 一致：ANSI 转义/null 字节/全角字符/绝对 home/反斜杠/空字符串都无法绕过模式检测 |
| sudo stdin guard | `_check_sudo_stdin_guard`（`approval.py:310`），未配 SUDO_PASSWORD 时拦 `sudo -S` | 同 | 一致：防 sudo 密码爆破 |
| pattern_key 用 description | `detect_dangerous_command` 返回人读 description 当 key（`approval.py:714`） | 同 | 一致：会话记忆按"人读风险类别"记住，不按易变的 regex |
| 会话级审批记忆 | `_session_approved: dict[str,set]` + lock（`approval.py:734,827`） | 同（`_session_approved` + `_session_lock`） | 一致：用户批准过一次的类别，本会话不再问 |
| 敏感系统路径保护 | `_SENSITIVE_PATH_PREFIXES`（/etc//boot/...）+ `_SENSITIVE_EXACT_PATHS`（`file_tools.py:416`） | 同（逐条搬） | 一致：写 /etc/passwd、docker.sock 等被拒 |
| ANSI 剥离实现 | `tools/ansi_strip.py` 全 ECMA-48（CSI/OSC/DCS/8-bit C1） | 同（`_strip_ansi` 逐行搬） | 一致：模型无法用 ANSI 转义切碎关键字（r\x1b[m → rm）绕正则 |

---

## 差异点（与 hermes 不同）

下列维度你的实现与 hermes 存在实质差异（多为「厚壳取舍」或「架构选择」）。

| 维度 | hermes | 你的版本 | 差异影响 |
|---|---|---|---|
| 闸门挂载架构 | 分散：每个工具自管（terminal import guards、file 用 check_fn 注册挂、execute_code 自调 guard） | 集中：`dispatch_tool_call` 一个 pre-tool-call 接缝 + `_permission._CHECKERS` table-driven | 你 7 工具单进程更干净；hermes 40+ 工具多攻击面必须分散。加新工具你只改字典一行 |
| DANGEROUS 模式数量 | 47 条（`approval.py:381-530`），覆盖极广（含 launchctl/docker compose/launchd 等平台特定） | ~20 条高频子集 | 你覆盖日常风险（rm -r/git 破坏/-c 绕过/SQL/远程执行/提权）；平台特定的（launchctl 等）暂不要 |
| 审批选项 | 4 选：o=once/s=session/a=always（持久化）/deny（`prompt_dangerous_approval:979`） | 3 选：y=本次/s=本会话/n=拒绝 | 你砍 `always`（permanent allowlist 持久化 ~200 行）；个人 CLI 会话记忆够用 |
| 审批传输 | 双模：CLI 同步 input() + gateway 异步队列（`_await_gateway_decision`/`submit_pending`） | 单模：同步 `input()` | 你无 gateway，无需异步审批；审批时 Ctrl+C 冒泡到 force_quit（可接受语义） |
| yolo 模式 | `--yolo`/`/yolo` 绕过所有 DANGEROUS 审批（HARDLINE 仍拦，`:1333`） | 无 | 你必须每次审批；个人 agent 更保守，可后续加 |
| LLM 风险评估 | `_smart_approve`（`:1208`）调辅助 LLM 评估命令风险，`smart` 模式下在 prompt 前 | 无 | 你纯正则 + 人工；砍掉 LLM 调用开销与复杂度 |
| read 工具设备拦截 | `_is_blocked_device`（`file_tools.py:374`）拦 /dev/zero、/dev/stdin、/proc/*/environ 等 | 无（V1 read 放行） | 你 read 不拦；如需防 hang/泄密，V2 可补 |
| workspace 硬边界 | 无硬边界（只敏感路径保护）；`_path_resolution_warning` 是软警告 | 同（V1 不做硬边界） | 一致：两者都允许绝对路径，只保护敏感路径 |
| permanent allowlist | 跨会话持久化到 config（`load_permanent_allowlist:946`），按命令文本 fnmatch | 无 | 你无跨会话记忆；如需，可加本地 JSON 持久化 |
| pattern_key aliases | legacy regex-key 向后兼容（`_PATTERN_KEY_ALIASES`） | 无 | 你全新实现无需兼容旧 key |
| execute_code guard | `check_execute_code_guard`（`approval.py:1898`，190 行）静态扫 Python | 无（本项目无 execute_code 工具） | 不适用；待 execute_code 工具落地后再搬 |

---

## 结论

**核心防御一致，挂载架构与厚壳不同。**

- 对「防止 agent 造成不可逆损害 + 防止命令绕过」这一核心目标：两者都是
  **HARDLINE 地板拒绝 + DANGEROUS 需审批 + 6 道命令归一化 + 敏感路径保护**，
  行为等价。`rm -rf /`、`mkfs`、`python3 -c "import os; os.remove(...)"`（解决 notebook 记的绕过缺口）、
  `git push --force`、写 `~/.ssh/authorized_keys` 等都会被拦下。
- 对你（实现者）：差异集中在两点——
  (1) **挂载架构**：hermes 分散到每个工具自管（40+ 工具必须），你用统一接缝 + table-driven（7 工具更干净）；
  (2) **厚壳取舍**：砍掉 gateway 异步审批、permanent 持久化、yolo、LLM 风险评估、tirith、execute_code guard。
  这些服务多平台/多会话/可配置/自动化场景，你单进程 CLI 用不上，按「别复杂」原则已砍掉。

## 取舍建议（按场景判断是否要补）

| 若你遇到 | 建议动作 | 对应 hermes 能力 |
|---|---|---|
| read_file 读 /dev/zero 卡死或 /proc/*/environ 泄密 | 给 `check_read_path` 加设备文件拦截 | `_is_blocked_device`（file_tools.py:374） |
| 嫌每次审批同类命令烦，想跨会话记忆 | 加 permanent allowlist（本地 JSON 持久化，按命令文本 fnmatch） | `load_permanent_allowlist` / `save_permanent_allowlist` |
| 想完全跳过 DANGEROUS 审批（HARDLINE 仍拦） | 加 `--yolo` 开关 + `_session_yolo` 集合 | `is_current_session_yolo_enabled` + yolo 短路 |
| DANGEROUS 模式漏了某个新型风险 | 在 `DANGEROUS_PATTERNS` 加一条 `(regex, description)` 元组 | 同结构（hermes 47 条） |
| 落地 execute_code 工具后需扫 Python 脚本 | 搬 `check_execute_code_guard` 子集 | `approval.py:1898` |
| 审批时 Ctrl+C 想区分"拒本次"和"中断任务" | 在 `_prompt_approval` catch KeyboardInterrupt 当"拒本次" | hermes 走 daemon thread + timeout，路径不同 |

> 注：本文件记录「工具权限闸门」对照。工具注册/发现见 `tool_registry.md`，执行引擎见 `tool_executor.md`，最大步数限制见 `max_iterations.md`。
