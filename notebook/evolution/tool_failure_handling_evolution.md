# 演进日志：工具调用失败处理

> 记录 hello-agent 如何从“工具返回一段字符串”演进到“可分类、可恢复、可熔断、可观测”的失败处理系统。
>
> 这里的顺序是**逻辑演进顺序**，用于解释每一层为什么出现，不代表 Git commit 的严格时间顺序。
> 核心状态机参考 Hermes `agent/tool_guardrails.py`、`agent/tool_executor.py` 和
> `agent/tool_result_classification.py`，但只保留当前项目需要的部分。

---

## 最初的问题：工具失败只是“一段返回文本”

最早的工具调用可以概括为：

```python
result = handler(**args)
messages.append(tool_result(result))
```

这段逻辑能跑，但主循环不知道下面这些结果有什么区别：

- `{"exit_code": 1, "stderr": "not found"}`：工具执行了，但失败了。
- `{"error": "permission denied"}`：可能是 handler 失败，也可能根本没有执行。
- `Unknown tool: xxx`：模型调用了不存在的工具。
- 空字符串：可能是正常的无输出命令，也可能是写工具没有真正落盘。
- handler 抛异常：如果没有边界，会直接让整个 Agent 崩溃。
- 用户按下 Ctrl+C：这是取消，不应该算工具质量差或模型重试失败。

**根本问题**：工具结果只有内容，没有明确状态。只要失败语义依赖字符串猜测，后面的
重试、护栏、文件核验和最终回答就会各自做一套判断，并逐渐互相矛盾。

---

## 循环 1：字符串猜失败 → 统一 `ToolOutcome`

### 问题 1.1：不同消费者对“失败”的理解不一致

例如 bash 最权威的失败信号是非零退出码；文件工具通常返回 `error` 字段；普通文本工具
可能只能使用错误前缀兜底。如果 guardrail 和文件修改检查分别解析一次，很容易出现：

- 文件核验认为成功，guardrail 却计作失败；
- `success=true, error=null` 因为包含单词 `error` 被误判；
- 网页结果列表中某一项包含 `error` 字段，整个工具调用被误判失败；
- 成功但没有 stdout 的 bash 被误认为失败。

### 解 1.1：建立统一结果对象

在 `agent/tool_result_classification.py` 引入 `ToolOutcome`：

```text
succeeded      handler 已执行并成功
failed         handler 已执行，但业务结果失败
blocked        权限或护栏在执行前拒绝
unknown        工具不存在，未执行
internal_error 参数、preflight、调度器或 handler 内部异常
cancelled      用户或系统取消
```

所有下游只消费一次分类后的 `ToolOutcome`：

```text
handler 原始结果
    ↓
classify_tool_result
    ↓
ToolOutcome
    ├─ guardrail 是否累计失败
    ├─ file mutation 是否真正落盘
    └─ tool_result 返回给模型
```

### 解 1.2：结构化信号优先，文本扫描只做兜底

当前分类顺序是：

1. 调度器显式传入的 `blocked / unknown / internal_error / cancelled` 最优先。
2. bash 的 `exit_code` 是权威信号：`0` 成功，非 `0` 失败。
3. 写工具必须返回可验证的成功字段。
4. JSON object 使用 `success`、`error`、`failed` 等顶层字段判断。
5. JSON 数组和标量只要能正常解析，就视为结构化成功结果。
6. 只有无法解析的普通文本，才扫描有限长度的错误关键词。

### 新的不变量

> 一次工具调用只分类一次；guardrail、文件核验共享同一个结论。

---

## 循环 2：异常击穿对话循环 → 分阶段错误边界

### 问题 2.1：只捕获 handler 异常仍然不够

模型不保证永远遵守 JSON Schema。例如：

```python
_preflight_call("bash", {"command": {"unexpected": "object"}})
```

旧逻辑会在权限检查里调用 `.strip()`，抛出 `AttributeError`。这时 handler 尚未运行，
但异常已经越过了对话循环。

### 解 2.1：三个阶段分别建立边界

```text
参数准备                preflight                 handler
coerce + schema 校验 → 护栏/注册/权限检查 → 真正执行工具
       │                    │                    │
invalid_arguments     preflight_exception   handler_exception
       └────────────────────┴────────────────────┘
                    ToolOutcome
```

具体规则：

- tool input 必须是 JSON object；校验 required、基础类型和 enum。
- 参数错误返回 `internal_error / invalid_arguments`，不执行 handler。
- preflight 自身异常返回 `internal_error / preflight_exception`。
- handler 抛出的普通异常返回 `internal_error / handler_exception`。
- 每个 `tool_use` 始终产生一个对应的 `tool_result`，不能破坏消息协议。

### 为什么非法参数计入失败

非法参数虽然不是工具业务失败，但它代表模型没有生成可执行调用。如果完全不计数，模型可以
无限重复同一组坏参数。因此它进入重复失败护栏；权限拒绝和用户取消则不进入。

---

## 循环 3：失败能返回给模型 → 模型仍会机械重试

### 问题 3.1：一次失败可恢复，重复失败会形成死循环

只把错误文本交还模型，并不能保证模型真的换策略。常见循环包括：

- 同一工具、同一参数反复失败；
- 同一工具不断换参数，但工具路径始终失败；
- 只读工具反复读取同一内容，没有产生任何新信息。

单一“失败次数”无法区分这三类问题。

### 解 3.1：按不同 key 建立三类状态

#### A. 精确失败重复：检测模型卡在同一个调用

```text
key = (tool_name, args_hash)
```

同一工具和同一参数连续失败：

- 达到 `exact_failure_warn_after`：返回 warning，引导模型分析错误、换参数或换策略。
- 达到 `exact_failure_block_after`：下一次 `before_call` 直接 block，不再执行 handler。

它主要回答：**模型是不是在重复输出同一个错误调用？**

#### B. 同工具失败重复：检测整条工具路径不可用

```text
key = tool_name
```

即使参数不同，只要同一个工具在本 turn 内持续失败：

- 达到 `same_tool_failure_warn_after`：提示模型换路径。
- 达到 `same_tool_failure_halt_after`：`after_call` 产生 halt，结束当前 turn。

它主要回答：**是不是这个工具或依赖本身已经不可用？**

#### C. 只读无进展：检测成功但无信息增量

只对 `IDEMPOTENT_TOOL_NAMES` 白名单生效：

```text
read_file, search_files, web_search, web_extract
```

```text
key   = (tool_name, args_hash)
value = (result_hash, repeat_count)
```

工具调用虽然成功，但相同调用反复返回相同结果时：

- 达到 `no_progress_warn_after`：提示使用已有结果或换查询。
- 达到 `no_progress_block_after`：下一次执行前 block。

它主要回答：**调用成功了，但 Agent 是否仍然没有取得进展？**

### 解 3.2：结果做语义哈希

下面两个 JSON 语义相同：

```json
{"count": 0, "results": []}
{"results": [], "count": 0}
```

因此 `_result_hash()` 会先解析 JSON，再按 key 排序、去除无意义空白后哈希。否则模型或工具
仅改变字段顺序，就能让无进展计数被错误重置。

### 解 3.3：成功重置连续失败

工具一旦成功：

- 清除对应精确调用的失败次数；
- 清除该工具的连续失败次数；
- 如果是只读工具，转入结果 hash 的无进展统计。

因此统计的是**连续失败**，而不是一个 turn 内永不清零的历史总失败。

---

## 循环 4：只有“继续/停止” → `allow / warn / block / halt`

### 问题 4.1：block 和 halt 看起来相似，但发生时机不同

如果只保留一个 stop 状态，就无法同时表达：

- “这个调用已经确定不应该再执行”；
- “当前调用已经执行完，但整个工具路径应该停止”。

### 解 4.1：四级决策

| action | 发生时机 | handler 是否执行 | 作用 |
|---|---|---:|---|
| `allow` | 执行前 | 是 | 正常放行 |
| `warn` | 执行后 | 已执行 | 在真实结果后追加换策略提示 |
| `block` | 执行前 | 否 | 返回 synthetic result，阻止重复调用 |
| `halt` | 执行后 | 已执行 | 当前批次归并后结束整个 turn |

关键区别：

- 精确失败和只读无进展的阈值来自过去已经观察到的结果，所以下一次调用可在执行前 block。
- 同工具失败达到阈值，是当前调用执行完才知道的，因此只能在 `after_call` 产生 halt。

`block` 和 `halt` 都锁存到 `halt_decision`。主循环在当前批次结果归并完成后检查它，
不再额外请求模型“总结一下为什么停了”，避免停止之后又开启新的工具循环。

---

## 循环 5：并发提高速度 → 状态更新出现竞态

### 问题 5.1：worker 线程不能直接修改 per-turn 护栏状态

如果多个 worker 同时执行 `after_call()`：

- 计数顺序可能不稳定；
- 第 N 次调用不一定看得到前 N-1 次的结果；
- 返回给模型的结果顺序可能与 tool_use 顺序不一致；
- 写失败跟踪也会出现竞态。

### 解 5.1：worker 无状态，主线程按模型顺序归并

```text
worker：只运行 handler → RawToolExecution
主线程：分类 → after_call → mutation tracker → tool_result
```

并发规则进一步收紧为：

- 写工具顺序执行；
- bash 顺序执行；
- 同名工具顺序执行；
- 只有名称不同的只读工具可以并发。

这样既保留安全的并发收益，也保证同名重复调用能看到前一次更新后的护栏状态。

---

## 循环 6：未知工具也返回错误 → 仍可能无限调用

### 问题 6.1：未知工具不是 handler 失败

模型可能调用一个 registry 中不存在的工具。它没有真正执行，因此：

- 不能进入 handler；
- 不能伪装成普通执行失败；
- 同一批其他工具也不能在协议不完整的情况下盲目执行；
- 但必须返回对应的 `tool_result`，告诉模型有哪些工具可用。

### 解 6.1：在 conversation loop 按“模型轮次”恢复

- 当前批次含未知工具时，为批次中的每个 tool_use 生成协议完整结果。
- 未知工具结果包含可用工具列表；同批合法工具返回 skipped，等待模型重新生成整批调用。
- 一批出现多个未知工具，只消耗一次重试次数，而不是按工具数量计数。
- 连续三轮调用未知工具后受控停止。
- 一轮合法工具调用会重置未知工具重试次数。

未知工具拥有独立的恢复预算，不污染重复失败护栏。

---

## 循环 7：写工具返回“看似成功” → 最终回答过度声称完成

### 问题 7.1：空结果对读工具和写工具含义不同

`bash("true")` 没有输出是正常成功；但 `write_file` 或 `patch` 返回空内容，无法证明文件已经
写入。如果统一把空结果视为成功，模型可能告诉用户“已经修改完成”，实际文件没有变化。

### 解 7.1：写操作要求可验证的落盘证据

- `write_file` 必须返回 `bytes_written`。
- `patch` 必须返回 `success=true`。
- 写工具空结果分类为 `empty_mutation_result` 失败。

`FileMutationTracker` 以 realpath 为 key，记录本 turn 内仍未恢复的写失败：

- 同一路径后来写成功，会清掉之前的失败；
- 软链接和真实路径会归一为同一个文件；
- turn 结束仍未恢复时，把提示追加到最终 assistant 消息，避免模型过度声称完成。

---

## 循环 8：取消被当作失败 → 护栏误判工具坏了

### 问题 8.1：用户暂停不是工具失败

用户按 Ctrl+C 时，可能存在三类工具：

- 尚未开始；
- 正在运行本地进程；
- 正阻塞在网络 SDK。

如果都记作 failed，会错误增加精确失败和同工具失败次数；下次用户继续工作时可能直接触发
warning、block 或 halt。

### 解 8.1：取消成为独立状态

- 未开始的调用返回 `cancelled`，不运行 handler。
- bash 终止整个进程组，避免遗留子进程。
- read/search 在扫描期间协作检查中断信号。
- web_search、web_extract 和摘要请求在 daemon worker 中运行；取消时关闭 HTTP client，主线程及时返回。
- `cancelled` 不进入 `after_call`，不累计护栏失败，也不污染文件失败记录。

这一步把“操作失败”和“用户改变意图”彻底分开。

---

## 循环 9：工具已经返回 → 模型却没有完成回答

### 问题 9.1：工具执行成功不等于 Agent turn 成功

模型拿到 tool_result 后，仍可能返回空消息。如果直接结束，用户只看到工具过程，看不到最终
结论；如果无限重试，又会形成新的循环。

### 解 9.1：工具后空回复只补救一次

- 第一次空回复：写入协议完整的空响应占位，并明确要求模型处理工具结果、回答用户。
- 第二次仍为空：受控停止，返回本地生成的停止说明。

它不属于具体工具失败，也不进入工具护栏；它属于**工具调用之后的 Agent 恢复失败**。

---

## 当前完整链路

```text
模型生成 tool_use
    ↓
参数转换与 schema 校验
    ├─ 非法 → internal_error / invalid_arguments
    ↓
before_call
    ├─ 重复调用达到阈值 → blocked
    ├─ 未注册 → unknown
    ├─ 权限拒绝 → blocked
    └─ 允许执行
    ↓
handler
    ├─ 正常返回 → 统一分类
    ├─ 抛异常 → internal_error / handler_exception
    └─ 用户中断 → cancelled
    ↓
主线程按模型顺序 finalize
    ├─ after_call 更新重复失败/无进展状态
    ├─ warn/halt guidance 写入结果
    ├─ FileMutationTracker 更新未恢复写失败
    └─ 生成一一对应的 tool_result
    ↓
conversation loop
    ├─ halt_decision → 结束 turn
    ├─ 工具后空回复 → 有界补救一次
    └─ 正常生成最终回答
```

---

## 当前状态语义速查

| 状态 | 执行过 handler | 计入重复失败 | 典型来源 |
|---|---:|---:|---|
| `succeeded` | 是 | 否 | exit 0、结构化成功、普通有效结果 |
| `failed` | 是 | 是 | exit 非 0、`error`、`success=false` |
| `blocked` | 否 | 否 | 权限拒绝、重复调用执行前拦截 |
| `unknown` | 否 | 否 | 模型调用未注册工具 |
| `internal_error` | 视阶段而定 | 是 | 非法参数、preflight 或 handler 异常 |
| `cancelled` | 可能已开始 | 否 | 用户暂停、尚未执行的调用被取消 |

这里最重要的不是状态数量，而是三条边界：

1. **没有执行**不等于**执行失败**。
2. **用户取消**不等于**工具有问题**。
3. **调用成功**不等于**Agent 有进展**。

---

## 配置与当前默认策略

阈值统一从 `config.yaml -> tool_loop_guardrails` 加载，结构与 Hermes 对齐：

```yaml
tool_loop_guardrails:
  warnings_enabled: true
  hard_stop_enabled: true
  warn_after:
    exact_failure: 2
    same_tool_failure: 3
    idempotent_no_progress: 2
  hard_stop_after:
    exact_failure: 5
    same_tool_failure: 8
    idempotent_no_progress: 5
```

代码默认值用于配置项缺失时兜底；环境变量不再覆盖这些开关和阈值。

---

## 尚未解决或需要后续阶段处理

### 1. `max_tokens` 仍可能被误认为正常完成

这是模型响应终止原因分类问题，不属于某个 handler 的工具失败。按当前规划留到上下文管理阶段，
与 token 预算、自动压缩和续写策略一起处理。

### 2. bash 不是文件系统沙箱

bash 可以通过重定向、解释器或脚本绕过 file 工具的敏感路径检查。它属于权限执行环境问题，
不是失败恢复问题；路线记录在 `permission_gate_evolution.md`，后续在“全量审批”和“OS 级沙箱”
之间单独决策。

### 3. 网络 SDK 的真正强制取消依赖底层实现

当前主线程能及时返回，HTTP client 也会被关闭；但 Python 无法强杀已经运行的线程。如果第三方
SDK 不响应 close，daemon worker 可能继续存活到请求自身超时。未来可考虑原生异步 client、
可取消 transport，或把高风险长任务放进独立进程。

### 4. 成功语义仍需要按工具扩展

当前 file 工具有落盘证据，bash 有 exit code。未来新增数据库、消息发送、部署等有副作用工具时，
必须定义各自的权威成功证据，不能仅靠“handler 没抛异常”。

---

## 一句话演进路线

```text
原始字符串
→ 统一 ToolOutcome
→ 参数/preflight/handler 三段错误边界
→ 精确失败、同工具失败、只读无进展三类计数
→ allow/warn/block/halt 四级决策
→ worker 无状态、主线程顺序归并
→ 未知工具独立恢复预算
→ 写操作落盘核验
→ cancelled 与 failed 分离
→ 工具后空回复有界补救
```

**核心学习**：工具失败处理不是简单的 `try/except`。完整系统必须同时回答：工具有没有执行、
为什么没成功、是否值得重试、重复的是参数还是工具路径、调用有没有产生新信息、是否应该停止
当前调用或整个 turn，以及最终能否向用户证明副作用真的发生了。
