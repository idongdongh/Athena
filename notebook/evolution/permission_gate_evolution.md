# 演进日志：权限闸门（permission gate）

> 记录权限闸门这个 feature 从「发现问题」到「解决」到「又出现新问题」的循环。
> 不是静态对照表，是**思路推进的过程**——记录我在每个节点看到的问题、当时的判断、选的解法，以及解法后来又暴露了什么新问题。
> 阅读顺序：从上往下，每一节是一个「问题→解→新问题」的完整循环。
>
> 本日志只记录权限闸门一个 feature 的演进。其他 feature 各有自己的演进日志（见 `evolution/` 目录）。
> 权限闸门的静态异同点对照见 `diff/permission_gate.md`。

---

## 循环 1：工具裸奔 → 权限闸门 → 绕过攻防

### 问题 1.1：7 个工具完全裸奔

**时间**：项目初期（交接文档 §六标 P0）

**现状**：`bash` 能跑任意命令（含 `rm -rf`），`write_file` 能写任意路径（含 `~/.ssh/authorized_keys`），无任何闸门。

**为什么是问题**：agent 能造成**不可逆损害**。一个 prompt injection 或模型误判，就能 `rm -rf ~` 删掉整个 home 目录、覆写 `.bashrc` 植入后门、读 `/proc/*/environ` 偷环境变量里的 API key。这不是 edge case，是**每天可能发生**的事。

**当时想的解法（错的方向）**：靠 system prompt 说明"别做危险操作"。
**为什么错**：system prompt 是软约束——模型会忘（长上下文稀释）、会"创造性绕过"（它觉得 `python3 -c` 不算 `rm`）、会被 prompt injection 覆盖。软约束不能挡安全风险。

---

### 解 1.1：三层防御闸门（对齐 hermes `approval.py` 核心子集）

**解法**：硬拦截，不靠模型自觉。搬 hermes 三层防御 + file 路径安全：

1. **HARDLINE 地板拒绝**（12 条）：`rm -rf /`、`mkfs`、`dd 写块设备`、fork bomb、shutdown——无条件拒绝，不可绕过。
2. **DANGEROUS 需审批**（~20 条高频）：`rm -r`、`chmod 777`、`git push --force`、`bash -c`、`python3 -c`、`curl|sh`、`DROP TABLE`——命中暂停问用户（y/s/n）。
3. **命令归一化**（6 道清洗）：剥 ANSI、去 null、NFKC、折叠 home、去反斜杠、去空串——防伪装绕过。
4. **file 路径安全**：`write_file`/`patch` 写 `/etc/`、`~/.ssh/`、`~/.env` 等敏感路径被拒。

**挂载方式**：`dispatch_tool_call` 一个 pre-tool-call 接缝 + `_permission._CHECKERS` table-driven。加新工具检查只改字典一行。

**代码**：
- `tools/approval.py`（~340 行，bash 闸门）
- `tools/path_security.py`（~140 行，file 路径安全）
- `tools/_permission.py`（~35 行，调度）
- `tool_executor.py:48-51`（hook 接缝 3 行）

---

### 问题 1.2：`python3 -c` 绕过命令名黑名单

**时间**：2026-07-22（`notebook/当前问题.md` 记的）

**现象**：我想拦 `rm`，模型却返回 `python3 -c "import os; os.remove('test.txt')"`。效果等同于 `rm`，但不匹配 `rm` 的正则，绕过了审批。

**当时的认知误区**：我在笔记里写"这个命令不需要审查"。**这是错的**——`python3 -c` **本身就是危险模式**，因为它能执行任意代码绕过任何命令名黑名单。正确判断是：`-c`/`-e` flag 本身就是红旗，内容不用管。

**当时想的解法（不够）**：在 system prompt 说明"不要绕过"。
**为什么不够**：同问题 1.1——软约束挡不住。

---

### 解 1.2：拦截 `-c`/`-e` flag 本身，不拦具体内容

**解法**：在 `DANGEROUS_PATTERNS` 加一条——

```python
(r'\b(python[23]?|perl|ruby|node)\s+-[ec]\s+', "通过 -e/-c 标志执行脚本"),
(r'\b(bash|sh|zsh|ksh)\s+-[^\s]*c(\s+|$)', "通过 -c/-lc 标志执行 shell 命令"),
```

**核心思路**：不试图理解脚本内容（不可能——任意代码任意行为），而是拦**"执行任意代码的入口"**这个 flag。`python3 -c`、`bash -c`、`perl -e` 都会被命中 DANGEROUS，触发审批。

**为什么 flag 级拦截比内容级拦截好**：
- 内部级（查 `os.remove`/`subprocess`）：漏（`os.system`、`__import__`、eval），且模型会用 `base64` 解码 + exec 绕过。
- flag 级：一抓一个准，模型无法绕——要跑 Python 必须用 `-c` 或写文件（写文件又被 `write_file` 闸门管）。

---

### 问题 1.3：模型用 ANSI 码 / 全角字符 / 反斜杠伪装命令

**时间**：解 1.2 落地后浮现的

**现象**：正则拦得住明文的 `python3 -c`，但模型（尤其被 prompt injection 诱导时）可以用**字符伪装**让正则变瞎：

| 伪装手法 | 命令长什么样 | 正则为什么瞎 |
|---------|------------|------------|
| ANSI 转义码 | `r\x1b[31mm -rf /` | `r` 和 `m` 之间塞颜色码，正则找 "rm" 找不到 |
| null 字节 | `rm\x00 -rf` | null 切断词边界 |
| 全角字符 | `ｒｍ -rf`（全角 ｒｍ） | 字节序列完全不同 |
| 反斜杠转义 | `r\m -rf` | `\` 分隔 |
| 空字符串字面量 | `r''m -rf` | `''` 分隔 |

**关键理解**：正则匹配是**字面字符比对**，任何能改变字节序列的手段都能绕过。

---

### 解 1.3：`_normalize_command_for_detection` 6 道清洗（正则跑之前）

**解法**：在所有正则匹配**之前**，先把命令洗成"标准形"，让正则看到纯净文本。

```python
def _normalize_command_for_detection(command):
    command = _strip_ansi(command)                      # 1. 去 ANSI 转义
    command = command.replace('\x00', '')               # 2. 去 null
    command = unicodedata.normalize('NFKC', command)    # 3. 全角→半角
    command = _fold_home_prefix(command)                # 4. /home/alice → ~
    command = re.sub(r'\\([^\n])', r'\1', command)      # 5. r\m → rm
    command = re.sub(r"''|\"\"", '', command)           # 6. r''m → rm
    return command
```

**顺序敏感**：
- 4 必须在 5 之前（home 折叠要识别 `/` 分隔符，5 会吃掉反斜杠，Windows 路径会乱）。
- 6 在最后（去空串后可能产生新的可剥离字符）。

**归一化不拦东西，它是"防瞎"——让后面的正则能看见真命令。**

---

### 问题 1.4（当前未解）：归一化的边界 + 会话级记忆的误放行

解 1.3 落地后浮现的新问题，**当前版本未解决**：

**a. 归一化的过清洗风险**：`_strip_ansi` 可能误伤合法含 ANSI 的命令（如 `echo -e "\033[31mred\033[0m"` 被清洗成 `echo -e "red"`，可能影响行为）。当前风险可接受（模型很少用 ANSI echo），但要知道这个权衡。

**b. 会话记忆的"类别过宽"风险**：用户对一条 `git push --force origin main` 选了 `s`（会话记忆），pattern_key 是 description（"git force push (重写远端历史)""）。之后模型发 `git push -f origin dev`，**也会被放行**（同 description）。当前行为其实合理（同类风险一次授权），但用户可能以为"只记了那条具体命令"——预期差。

**c. read 工具裸奔**：V1 只拦写，`read_file("/etc/passwd")`、`read_file("/proc/1/environ")`（偷环境变量）放行。hermes 有 `_is_blocked_device` 拦这类。当前判断：个人本机风险低，YAGNI；若部署到服务器或多人环境必须补。

**d. `_CHECKERS` 漏注册风险**：集中式接缝的代价——未来加 `execute_code` 工具时，若忘了在 `_permission._CHECKERS` 加它的 check，它会裸奔。hermes 分散式（每工具自带 check_fn）无此风险。**缓解**：在 `_permission.py` 顶部注释明确写了"加新工具必须在此注册检查"。

---

### 小结：权限闸门的演进链

```
裸奔(问题1.1) → 三层防御(解1.1) → python3-c 绕过(问题1.2) → flag 级拦截(解1.2)
→ ANSI/全角伪装(问题1.3) → 6 道归一化(解1.3) → 归一化边界/会话记忆/read裸奔(问题1.4, 未解)
```

**核心学习**：安全不是"加一层防御就完事"，是**攻防螺旋**——每个解法都会暴露新的攻击面。做 AI agent 安全开发，要永远问"这个解法会被怎么绕？"，然后针对那个绕法再加一层。永远没有"做完"。

---

## 循环 2：集中式 preflight → 非法参数崩溃 → 错误边界补齐

### 问题 2.1：权限检查默认相信模型参数符合 JSON Schema

**时间**：2026-08-06

**现象**：模型传入以下参数时，权限检查直接抛出异常，整个 agent turn 崩溃：

```python
_preflight_call("bash", {"command": {"unexpected": "object"}})
# AttributeError: 'dict' object has no attribute 'strip'
```

`bash.command` 的 schema 虽然声明为 string，但模型输出不是可信输入。schema 是给模型的
约束提示，不是运行时安全边界。此前只有 handler 外层有异常捕获，参数转换和 permission
preflight 位于这个边界之外。

### 解 2.1：参数转换、preflight、handler 三段都有稳定结果

**解法**：

1. `coerce_tool_args` 先验证 tool input 必须是 JSON object，再校验 required、基础类型和 enum。
2. 单工具兼容入口和批量调度都捕获参数错误，返回 `invalid_arguments`，不执行 handler。
3. `_preflight_call` 自己建立异常边界：参数形状错误返回 `invalid_arguments`，权限/护栏内部异常返回 `preflight_exception`。
4. 错误仍生成对应 `tool_result`，保证下一轮消息协议完整；重复非法调用仍进入失败护栏，防止模型无限重试。

**新的不变量**：模型可以生成错误参数，但不能靠错误参数让对话循环崩溃，也不能绕过
permission preflight 直接进入 handler。

---

## 问题 2.2（待决）：bash 可以绕过 file 工具的敏感路径保护

**时间**：2026-08-06

以下命令当前都会通过 `bash` 权限检查：

```bash
printf x > config.yaml
printf x > ~/.ssh/authorized_keys
python - <<'PY'
from pathlib import Path
Path("config.yaml").write_text("x")
PY
```

原因不是少了一条正则，而是当前有两套不同强度的安全语义：

- `write_file` / `patch`：能看到明确的 `path` 参数，可以在执行前做 realpath 和敏感路径检查。
- `bash`：接收任意程序，文件目标可能藏在重定向、解释器、脚本文件、变量、子 shell 或程序内部，无法可靠提取。

### 为什么不继续补“写入目标正则”

拦住 `>` 仍可用 `tee`，拦住 `tee` 仍可用 Python，拦住 `python -c` 仍可用 heredoc、已有
脚本或其他解释器。继续扩充命令黑名单只能减少常见误操作，不能形成“敏感路径一定写不了”
的安全证明，反而容易给人虚假的安全感。

### 当前结论：安全模型决策延期，不做伪修复

当前版本应明确理解为：

- HARDLINE / DANGEROUS 是常见危险命令的防失控与审批层；
- `check_write_path` 只保护原生 file 写工具；
- bash 与 agent 进程拥有相同文件权限，**不是文件系统沙箱**。

后续需要在两条路线中单独决策：

1. **所有 bash 或高能力 bash 强制审批**：实现简单，降低自治体验，但仍依赖用户判断。
2. **受限执行环境**：通过容器/OS sandbox 把敏感路径设为不可写，安全边界可靠，但会增加平台适配、挂载和开发工具链成本。

在路线确定前，不用不完备的 shell 目标解析冒充统一路径策略。

---

## 更新后的权限闸门演进链

```text
工具裸奔
→ 命令黑名单 + 路径检查
→ 任意解释器绕过
→ flag 级审批 + 命令归一化
→ 非法参数击穿 preflight
→ 三段错误边界
→ bash 绕过 file 路径策略
→ 待决：全量审批 or OS 级受限执行环境
```
