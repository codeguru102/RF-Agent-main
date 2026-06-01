# RF-Agent

Offline, folder-backed RF-Agent for RL training workflows in both Python and MATLAB.

This project retains the core RF-Agent reward generation and search logic, but replaces IsaacGym live training with a cross-language, asynchronous file-based interface. You can run your RL agent training logic in **either Python or MATLAB** by following a simple contract:

```text
RF-Agent reads a single task folder
RF-Agent generates candidate reward function files
Your Python or MATLAB trainer trains pending candidates offline
Your trainer writes logs and optionally summary files
RF-Agent reloads the task folder, exports the best reward, and opens a dashboard window
```

## Task Folder Structure

Each RL task resides in a dedicated folder as follows:

```text
tasks/<task_name>/
  task.json
  description.md
  observations.md
  environment.md
  original_reward.py or original_reward.m
  logs/
  candidates/
  visualization/
  best_reward_fcn.py or best_reward_fcn.m
  best_reward_summary.json
```

Replace the starter files in [tasks/python_task](tasks/python_task) (or create your own) with your real task description, observation schema, original reward, environment notes, and any reference logs. Use either Python (`.py`) or MATLAB (`.m`) files depending on your RL codebase.

## Candidate Folder

Each generated reward candidate is stored under its task folder:

```text
tasks/<task_name>/candidates/candidate_000001/
  reward_fcn.py or reward_fcn.m
  description.txt
  metadata.json
  status.json
  prompt_messages.json
  logs/
    train.csv
    eval.csv
  summary.json
```

- For Python projects, `reward_fcn.py` will be generated.
- For MATLAB projects, `reward_fcn.m` will be generated.

`metadata.json` stores RF-Agent tree information such as:

```json
{
  "candidate_id": "candidate_000001",
  "parent_id": null,
  "action_type": "initialize",
  "action_index": 0,
  "generation": 0,
  "reward_file": "reward_fcn.py",
  "design_thought": "...",
  "status": "pending"
}
```

`summary.json` is written by your trainer and should look like:

```json
{
  "status": "trained",
  "max_task_score": 0.82,
  "final_task_score": 0.76,
  "mean_return": 120.5,
  "success_rate": 0.71,
  "constraint_violation": 0.02,
  "notes": ""
}
```

The field used for candidate ranking/selection is determined by your `task.json`. If your trainer (Python or MATLAB) only outputs CSV logs, simply run `sync` after copying logs into each candidate's `logs/` folder. RF-Agent will build `summary.json` from numeric CSV columns and mark those candidates as trained.

## RF-Agent Logic Preserved

The offline agent retains all RF-Agent search actions:

- `initialize`
- `mutation_mechanism`
- `mutation_param`
- `crossover_elite`
- `tree_reasoning`
- `different_thought`

It rebuilds parent/child relationships from candidate metadata, computes RF-Agent-style Q values and UCT scores, includes self-verification in selection, chooses trained parents for expansion, generates new pending rewards with feedback/validation, persists the RF-Agent elite set used by crossover, and exports the best trained reward.

## Commands

Inspect current task state:

```bash
python src/main.py --mode inspect --task-dir tasks/python_task
```
or (for a MATLAB task named `matlab_task`):
```bash
python src/main.py --mode inspect --task-dir tasks/matlab_task
```

Generate candidate rewards:

```bash
python src/main.py --mode generate --task-dir tasks/python_task --num-candidates 8
# or for MATLAB:
python src/main.py --mode generate --task-dir tasks/matlab_task --num-candidates 8
```

Sync CSV-only offline logs into summary:

```bash
python src/main.py --mode sync --task-dir tasks/python_task
python src/main.py --mode sync --task-dir tasks/matlab_task
```

No-API smoke test:

```bash
python src/main.py --mode generate --task-dir tasks/python_task --num-candidates 2 --dry-run
# or for MATLAB:
python src/main.py --mode generate --task-dir tasks/matlab_task --num-candidates 2 --dry-run
```

Visualization is always handled automatically: `inspect`, `generate`, and the supplied worker templates open a dashboard window before exiting, and write:

```text
tasks/<task_name>/visualization/tree.json
tasks/<task_name>/visualization/tree_dashboard.png
```

The dashboard presents the entire search tree, parent/child relationships, candidate status, reward score, Q value, visits, UCT score, selected update parent, selected candidates for training, the persisted elite set, and the emphasized best reward function. Tune `elite_max_length` for the candidate pool, and `dashboard_elite_max` for displayed elite nodes.

When at least one trained candidate exists, the best reward is also exported as:

```text
tasks/<task_name>/best_reward_fcn.py or best_reward_fcn.m
tasks/<task_name>/best_reward_summary.json
```

## RL Trainer Worker (Python or MATLAB)

Your RL trainer (Python or MATLAB) should:

1. Scan `tasks/<task_name>/candidates/candidate_*`.
2. Find candidates whose `status.json` has `"status": "pending"`.
3. Train/evaluate using the generated reward function (`reward_fcn.py` or `reward_fcn.m`).
4. Write `logs/train.csv` and/or `logs/eval.csv`.
5. Write `summary.json`, or run `python src/main.py --mode sync` to infer summaries from CSV logs.
6. Update `status.json` to `"trained"` or `"failed"`; `sync` can also mark candidates with logs as trained.
7. (Optional) Use the supplied worker templates (see below) to automate running. You can also run `python src/main.py --mode inspect --task-dir tasks/<task_name>` to export the best reward and refresh the dashboard.

See [python/run_pending_candidates_template.py](python/run_pending_candidates_template.py) for a Python worker and [matlab/run_pending_candidates_template.m](matlab/run_pending_candidates_template.m) for a MATLAB worker.

## OpenAI API Setup

Set an API key before running non-dry-run candidate generation:

```bash
set OPENAI_API_KEY=your_key_here
```

On PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_key_here"
```

You can now use RF-Agent for folder-backed RL reward search in your Python **or** MATLAB projects.