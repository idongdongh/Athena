# Athena Agent Tracer Evaluation

该模块对 Athena 在编码任务中的端到端行为进行可复现评测。它记录模型请求、
工具调用、Guardrail、上下文压缩和任务结果，再组合确定性 Checks 与
LLM Rubric 生成轨迹级诊断。

实验设置、结果与适用边界见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

## 评估原则

- `completed` 表示 Agent turn 正常结束，不代表任务成功。
- `resolved` 只由隔离工作区中的外部测试与非空 Patch 确定。
- 规则可确定的 Check 不调用 LLM。
- LLM Judge 的每个结论必须引用实际 `event_id`。
- Failure Onset 是首个未被恢复的关键错误决策，不等于首次工具失败。
- Wash 只生成派生清单，不删除原始轨迹。

## 命令

执行隔离任务：

```bash
uv run python -m evaluation run eval_tasks/smoke.yaml \
  --output eval_runs/smoke --repetitions 1
```

执行确定性评价、Wash 和报告：

```bash
uv run python -m evaluation evaluate eval_runs/smoke
uv run python -m evaluation wash eval_runs/smoke
uv run python -m evaluation report eval_runs/smoke
```

Judge 默认只输出请求计划，不发起付费请求：

```bash
uv run python -m evaluation judge eval_runs/smoke
```

显式加上 `--execute` 才会调用 Judge 模型：

```bash
uv run python -m evaluation judge eval_runs/smoke \
  --model "$JUDGE_MODEL_ID" --max-concurrency 2 --rpm 20 --execute
```

执行 Agent 与 Judge 可使用独立 Provider：

```dotenv
EVAL_EXECUTOR_API_KEY=...
EVAL_EXECUTOR_BASE_URL=https://api.deepseek.com/anthropic
EVAL_EXECUTOR_MODEL=deepseek-v4-flash
JUDGE_API_KEY=...
JUDGE_BASE_URL=https://open.bigmodel.cn/api/paas/v4
JUDGE_MODEL_ID=glm-5.2
JUDGE_API_FORMAT=openai
```

## 输出结构

```text
eval_runs/<run>/
├── <task>__r<n>/
│   ├── workspace/
│   ├── traces/<trace_id>/
│   │   ├── events.jsonl
│   │   ├── artifacts/
│   │   └── task_result.json
│   └── runner_result.json
├── evaluated/evaluations.jsonl
├── evaluated/wash_manifest.json
├── evaluated/report.md
└── judged/
```

## 指标口径

- ISR / Instance Success Rate：有效任务中 `resolved=true` 的比例。
- CSR / Check Success Rate：适用的确定性 Checks 中通过的比例。
- EAR / Effective Action Ratio：去重后的成功工具行动数除以全部工具调用数。
- TFRR / Tool Failure Recovery Rate：被后续同工具成功恢复的失败数除以全部可观测工具失败数。
- Redundant Tool Call Rate：同名同参数重复调用数除以全部工具调用数。

EAR 当前是可复现的规则代理指标，不等价于人工语义标注。正式报告需同时
报告规则 EAR 和人工/Judge 校准结果。
