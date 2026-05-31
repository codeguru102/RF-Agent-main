# RF-Agent-Python

Offline, folder-backed RF-Agent for Python RL training workflows.

This project keeps the RF-Agent reward generation/search logic, but replaces IsaacGym live training with an asynchronous Python file contract:

```text
RF-Agent-Python reads one task folder
RF-Agent-Python generates candidate reward_fcn.py files
your Python trainer trains pending candidates offline
your Python trainer writes logs and optionally summary files
RF-Agent-Python reloads the task folder, exports the best reward, and opens a Python dashboard window
```

## Task Folder

Each RL task lives in one folder:

```text
tasks/<task_name>/
  task.json
  description.md
  observations.md
  environment.md
  original_reward.py
  logs/
  candidates/
  visualization/
  best_reward_fcn.py
  best_reward_summary.json
```

Replace the starter files in [tasks/python_task](tasks/python_task) with your real task description, observation schema, original reward, environment notes, and any reference logs.
Configure which task files are loaded into the LLM context in `task.json`:

```json
"task_text_files": {
  "description": {"path": "description.md", "title": "Purpose / Description"},
  "observations": {"path": "observations.md", "title": "Observations"},
  "environment": {"path": "environment.md", "title": "Environment"},
  "original_reward": {"path": "original_reward.py", "title": "Original Reward"}
}
```

Each task can use a different set of keys and filenames.

## Candidate Folder

Each generated reward candidate is stored under the task folder:

```text
tasks/<task_name>/candidates/candidate_000001/
  reward_fcn.py
  description.txt
  metadata.json
  status.json
  prompt_messages.json
  logs/
    train.csv
    eval.csv
  summary.json
```

`metadata.json` stores RF-Agent tree information:

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

`summary.json` is written by your Python trainer:

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

The score field used for selection is configured in `task.json`.

If your offline trainer can only return CSV logs, run `sync` after copying logs into each candidate's `logs/` folder. RF-Agent-Python will build `summary.json` from numeric CSV columns and mark those candidates as trained.

## RF-Agent Logic Preserved

The offline agent keeps the RF-Agent search actions:

- `initialize`
- `mutation`
- `crossover_elite`
- `tree_reasoning`
- `different_thought`

After root initialization, each selected trained leaf expands into four pending candidates sampled with action probabilities: `mutation` 30%, `crossover_elite` 30%, `tree_reasoning` 20%, and `different_thought` 20%.

It rebuilds parent/child relationships from candidate metadata, computes RF-Agent-style Q values and UCT scores, includes the self-verification score in selection, selects trained parents for expansion, generates new pending rewards with validation/retry feedback, persists the RF-Agent elite set used by crossover, and exports the best trained reward.

## Commands

Inspect current task state:

```bash
python src/main.py --mode inspect --task-dir tasks/python_task
```

Generate candidate rewards:

```bash
python src/main.py --mode generate --task-dir tasks/python_task --num-candidates 8
```

Sync CSV-only offline logs into scoreable summaries:

```bash
python src/main.py --mode sync --task-dir tasks/python_task
```

No-API smoke test:

```bash
python src/main.py --mode generate --task-dir tasks/python_task --num-candidates 2 --dry-run
```

There is no separate visualize option. `inspect`, `generate`, and the worker template open a Python dashboard window before ending. They also write:

```text
tasks/<task_name>/visualization/tree.json
tasks/<task_name>/visualization/tree_dashboard.png
```

The dashboard shows the whole search tree, parent/child relationships, node status, reward score, Q value, visits, UCT score, selected update parent, selected training candidates, the persisted RF-Agent elite set, and the final best reward function emphasized. `elite_max_length` caps the persisted crossover pool; `dashboard_elite_max` only caps how many persisted elite nodes are displayed.

When at least one trained candidate exists, the best reward is also exported to:

```text
tasks/<task_name>/best_reward_fcn.py
tasks/<task_name>/best_reward_summary.json
```

## Python Trainer Worker

Your Python trainer should:

1. Scan `tasks/<task_name>/candidates/candidate_*`.
2. Find candidates whose `status.json` has `"status": "pending"`.
3. Train/evaluate using `reward_fcn.py`.
4. Write `logs/train.csv` and/or `logs/eval.csv`.
5. Write `summary.json`, or run `python src/main.py --mode sync` to infer it from CSV logs.
6. Update `status.json` to `"trained"` or `"failed"`; `sync` can also mark candidates with logs as trained.
7. The worker template opens the dashboard before ending. You can also run `python src/main.py --mode inspect --task-dir tasks/<task_name>` to export the best reward and redraw the dashboard.

See [python/run_pending_candidates_template.py](python/run_pending_candidates_template.py) for a simple worker template.

## OpenAI Setup

Set an API key before non-dry-run generation. The project automatically reads `RF-Agent-Python/.env`:

```text
OPENAI_API_KEY=your_key_here
```

You can also set it manually:

```bash
set OPENAI_API_KEY=your_key_here
```

On PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_key_here"
```
