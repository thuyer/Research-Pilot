# ResearchPilot

面向 LLM 实验诊断与受控调参的 Stateful Tool-Using Agent。

ResearchPilot 将实验任务拆分为“感知 → 决策 → 执行 → 观察 → 再决策”闭环：Agent 先读取目标脚本，选择并调用受限实验工具；实验结果经过结构化提取后写入任务状态，模型再根据上一轮反馈决定是否调整参数并复跑。

## 项目亮点

- **多步 Tool Calling Loop**：支持模型连续调用代码检查、历史查询、任务上下文和实验执行工具，并将工具结果重新注入消息历史。
- **反馈驱动决策**：下一次实验必须基于上一轮返回的状态、指标或失败原因，由模型自主决定是否继续以及如何调整参数。
- **受控实验执行**：使用 `subprocess.run` 的参数列表模式执行脚本，限制在 `experiment_repo/` 目录和 Python 文件范围内，不暴露通用 shell 执行能力。
- **结构化实验记忆**：用 `manifest.jsonl` 保存完整运行流水，用 `task state` 保存当前任务的尝试、失败历史和最佳结果，两者职责分离。
- **可验证的工程评测**：通过声明式 `expected_runs` 检查运行次数、参数关系、执行状态和 state 一致性。

当前版本聚焦“实验执行 + 结构化指标提取 + stateful 调参循环”，不试图覆盖完整的研究工作流。

> **V1 范围说明**：仓库中的实验脚本是确定性合成案例，目的是稳定复现工具调用和状态管理行为；它们不代表真实训练任务，也不能单独证明 Agent 在真实数据集上的泛化能力。

## 1. 项目结构

```text
learn-claude-code/
├── researchpilot/
│   ├── core.py                  # Agent 主循环与工具注册
│   ├── state_manager.py         # 任务 state 的初始化、保存、更新
│   ├── hooks.py                 # 工具调用生命周期事件与耗时记录
│   ├── extractor/
│   │   └── experiment_extractor.py  # 从 stdout/stderr 提取指标
│   ├── experiment_repo/         # 用于验证 Agent 行为的实验脚本
│   ├── runs/manifest.jsonl      # 所有实验运行记录（运行时生成）
│   ├── states/<task_id>.json    # 任务 state（运行时生成）
│   └── README.md
└── researchpilot_eval_private/  # 与 researchpilot 同级的评测目录
    ├── case_v1.json             # 6 个评测案例
    ├── evaluate.py              # 评测入口脚本
    └── results.jsonl            # 历史评测记录（运行时追加）
```

## 2. 架构说明

### 2.1 Agent 主循环

`core.py` 中的 `loop(text, state, max_steps=12)` 是核心控制逻辑：

- 构造系统提示词与用户任务
- 调用 OpenAI Chat Completions 工具调用接口
- 解析模型返回的工具调用
- 根据工具名转发到 `TOOL_HANDLERS`
- 把工具输出写回消息上下文
- 在接收到非工具响应时停止，并返回：
  - `final_answer`
  - `tool_trace`
  - `hook_events`
  - `state`
  - `steps`

这个循环依赖模型生成的工具调用和传入的 task state 来驱动实验。一次 `loop()` 调用内部会保留当前任务的消息历史；新的用户交互是否复用旧 state，由调用方决定。

### 2.2 执行安全边界

`core.py` 里有一个安全根目录 `SAFE_ROOT`，并且 `run_experiment()` 在执行前强制要求：

- 目标脚本必须位于 `experiment_repo/` 下
- 必须是 `.py` 文件
- 不能越出 `experiment_repo` 目录
- 不接受非 Python 文件

因此它是一个基于路径约束的“受限实验执行器”，而不是通用 shell 执行器或生产级安全沙盒。

### 2.3 指标提取器

`extractor/experiment_extractor.py` 使用独立的 LLM client，负责从脚本 stdout/stderr 中抽取结构化指标，当前约束如下：

- 只抽取日志中明确打印的数值
- 不能虚构指标
- 若日志中的核心指标出现 `NaN`、`Inf`、`-Inf`，则返回：
  - `metrics: {}`
  - `outcome: "failed"`
  - `failure_type: "nan_or_inf"`
- 若没有任何有效数值指标，则返回 `metrics: {}`
- 如果日志里存在多个同类过程值，只取最后一次打印的值

当前执行链路不会把 `target_metric` 直接传给提取器，因此 state 的指标比较要求提取结果中存在与 `state["target_metric"]` 相同的键。

也就是说，提取器是“日志解析器”，不是训练框架本身。

### 2.4 评测入口

`researchpilot_eval_private/evaluate.py` 仅负责：

- 读取 `case_v1.json`
- 为每个 case 创建独立任务 state
- 调用 `loop(...)`
- 提取工具调用中的 `run_experiment` 记录
- 检查 `state` 是否和实际运行一致
- 验证每次实验是否满足该 case 的 `expected_runs` 规则

它不是用来训练模型，也不是衡量模型通用能力的统计 benchmark，而是用于工程级行为检查。

### 2.5 Hook 可观测性

`hooks.py` 提供工具调用生命周期 Hook，用于记录 Agent 运行过程中的可观测信息，不参与模型决策，也不替代任务 state：

- `before_tool_call`：记录工具名称、参数和开始时间。
- `after_tool_call`：记录结果摘要和工具执行耗时。
- `on_tool_error`：记录工具分发层的错误类型、错误信息和耗时。

Hook 事件通过 `loop()` 返回的 `hook_events` 字段暴露给调用方；完整工具输入输出仍由 `tool_trace` 保存，实验任务状态仍由 `state` 保存。

## Demo：根据实验反馈自动恢复

以 `experiment_repo/train.py` 为例，用户只给出初始任务和 `lr=0.1`。Agent 的关键行为序列如下：

```text
set_task_context(script=train.py, target_metric=loss, direction=min)
→ run_experiment(lr=0.1)
  → stdout 出现 NaN，提取结果 outcome=failed
→ run_experiment(lr=0.001)
  → loss: 5.0 → 2.5 → 1.67 → 1.25 → 1.0，outcome=success
→ 停止并返回诊断总结
```

这里的重点不是预先写死“第二次必须使用 `0.001`”，而是 Agent 读取第一次实验的反馈后选择更小的学习率。评测脚本只验证约束和结果，不把这条决策路径硬编码到 Agent loop 中。

## 3. Tool 职责

`core.py` 中注册的工具如下：

### list_files

职责：列出某目录下的文件和子目录，返回结构化列表。

返回例子：

```python
[{"name": "train.py", "type": "file"}, {"name": "experiment_repo", "type": "directory"}]
```

用途：让 LLM 先确认 workspace 结构，而不必依赖 shell 输出。

### read_file

职责：按行读取文本文件，支持 `offset` 和 `limit`。

限制：

- 只允许读取根目录内部文件
- 读取文件最大约 100MB
- 超过 8000 字符时截断输出

用途：读取 `train.py`、`runtime_error.py` 等实验脚本。

### search_code

职责：在目录中按关键字搜索文本。

用途：在不直接读取整个仓库的情况下，寻找与问题相关的符号/日志关键词。

### set_task_context

职责：设置当前任务上下文，写入任务 state：

- `script_path`
- `target_metric`
- `metric_direction`
- `title`

在 Agent loop 中必须在 `run_experiment` 之前调用；否则该次实验调用会被拒绝。

### load_experiment_records

职责：读取 `runs/manifest.jsonl` 中指定脚本的历史记录，并按时间倒序返回。

用途：让 Agent 在下一轮实验前参考先前结果，避免重复错误。

### run_experiment

职责：实际执行实验脚本，生成结构化运行结果，并将完整结果追加到 `runs/manifest.jsonl`。任务 state 的更新由 Agent loop 在 `run_experiment` 返回后交给 `state_manager.update_state()` 完成。

执行流程：

1. 规范化 `script_path`
2. 校验脚本位于 `experiment_repo` 且为 `.py`
3. 组装 `python script.py --arg value ...`
4. `subprocess.run(..., timeout=timeout_seconds)`
5. 收集 stdout/stderr
6. 调用 `extract_experiment_summary()` 解析指标
7. 生成运行结果字典
8. 追加到 `runs/manifest.jsonl`
9. 返回结果；如果该调用来自 `loop()`，loop 随后更新当前任务 state

返回字段包括：

- `run_id`
- `script_path`
- `params`
- `status`
- `exit_code`
- `metrics`
- `outcome`
- `failure_type`
- `failure_reason`
- `stdout_tail`
- `created_time`
- `elapsed_time`

## 4. State 字段

`state_manager.py` 中的 `init_state()` 返回的字段如下：

```python
{
  "task_id": ...,              # 任务唯一 ID
  "user_prompt": ...,          # 原始用户任务描述
  "title": None,               # 任务标题
  "script_path": None,         # 当前目标脚本
  "target_metric": None,       # 目标指标名，例如 accuracy / loss
  "metric_direction": None,    # "max" 或 "min"
  "runs": [],                  # 运行历史记录
  "failure_history": [],       # 失败记录
  "best_run_id": None,         # 当前最优运行 ID
  "best_metric_value": None,   # 当前最优指标值
  "attempt_count": 0,          # 实验尝试次数
  "updated_at": "..."          # 最后更新时间
}
```

其中，`task_id` 表示一次完整的研究/调参任务，`run_id` 表示该任务中的一次实验尝试；同一个脚本的不同参数可以拥有不同 `run_id`，并通过同一个 `task_id` 进行比较。

每个 `runs` 元素字段约定为：

```python
{
  "run_id": ...,
  "params": {...},
  "metrics": {...}
}
```

`update_state(state, result_data)` 会：

- 把新 run 追加进 `state["runs"]`
- 对失败 run 追加到 `failure_history`
- 如该 run 是成功且指标能比较，则更新：
  - `best_run_id`
  - `best_metric_value`
- 递增 `attempt_count`
- 调用 `save_state()` 持久化到 `states/<task_id>.json`

## 5. 运行方法

### 5.1 安装依赖

```bash
python -m pip install openai python-dotenv
```

并在项目根目录设置 `.env`：

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_ID=gpt-4o
```

### 5.2 交互模式

```bash
cd researchpilot
python core.py
```

启动后，程序会要求输入用户任务，例如：

```text
User: Please diagnose and run experiment_repo/train.py.
```

### 5.3 评测模式

```bash
cd <learn-claude-code>
python researchpilot_eval_private/evaluate.py
```

例如在当前容器布局中：

```bash
cd /app
python researchpilot_eval_private/evaluate.py
```

说明：`evaluate.py` 会在运行时切换到同级的 `researchpilot/` 作为项目根目录，并直接导入 `core.py`。Windows 下将 `/app` 换成包含 `researchpilot` 和 `researchpilot_eval_private` 的目录即可。

## 6. 六个评测案例

私有测试集 `case_v1.json` 共包含 6 个案例，分别是：

1. `case_001_lr_recovery`
   - 任务：先执行 `experiment_repo/train.py`，使用 `lr=0.1`，观察是否发散；如果失败则调整学习率并再次验证，直到成功停止。
   - 预期：第一次失败，第二次使用更小学习率，最终成功。

2. `case_002_lr_normal`
   - 任务：直接运行 `experiment_repo/train.py`，参数为 `lr=0.001`，正常完成后停止。
   - 预期：只执行一次，且状态为 `completed`。

3. `case_003_invalid_path`
   - 任务：调用 `run_experiment` 执行 `experiment_repo/non_existing_train.py`，验证路径校验。
   - 预期：脚本不存在时返回 `rejected`，并且不继续尝试其他脚本。

4. `case_004_runtime_error`
   - 任务：执行 `experiment_repo/runtime_error.py`，在默认参数下观察是否触发运行时错误。
   - 预期：返回 `failed`，并保留 `runtime_error` 诊断。

5. `case_005_timeout`
   - 任务：执行 `experiment_repo/timeout_experiment.py`，并以 `timeout_seconds=1` 进行超时测试。
   - 预期：返回 `timeout`，说明脚本需要更长时间完成。

6. `case_006_accuracy_maximize`
   - 任务：调试 `experiment_repo/accuracy_experiment.py`，目标是提高 `accuracy`。
   - 预期：先用 `threshold=0.9`，发现结果不理想；随后调小阈值并达到更高 accuracy，最终成功停止。

## 7. 已知限制

这是一个最小可运行版本，当前存在明确限制：

1. 只允许执行 `experiment_repo/` 下的 Python 脚本，不能执行任意脚本或 shell 命令。
2. 没有真正的研究平台、数据库、任务计划器、向量检索或代码修复引擎。
3. Agent 依赖 OpenAI Chat Completions 工具调用，若 API 配置不正确，循环无法正常运行。
4. `extract_experiment_summary()` 依赖模型对日志进行理解；它不是可靠的语法级解析器，不能保证所有日志都能稳定抽取。
5. `run_experiment()` 只捕获 `subprocess.run` 与超时错误，不提供真实训练框架的完整生命周期管理。
6. 评测中使用的是“合成实验脚本”，不是真实的大型训练任务，因此它更像是工具调用与状态管理 benchmark，而不是科研执行环境。

## 8. 实际运行状态

最近一次私有评测共 6 个案例，结果为：

```text
case_001_lr_recovery: PASS
case_002_lr_normal: PASS
case_003_invalid_path: PASS
case_004_runtime_error: PASS
case_005_timeout: PASS
case_006_accuracy_maximize: PASS
Summary: 6/6 passed
```

评测结果会以 JSON Lines 形式追加保存到 `researchpilot_eval_private/results.jsonl`，因此该文件可能同时包含多次运行的历史记录。

如果缺少 `openai` 依赖，则会触发：

```text
ModuleNotFoundError: No module named 'openai'
```

因此，运行前需要先安装依赖，并确认 `.env` 中的 OpenAI 配置正确。
