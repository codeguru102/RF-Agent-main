# RF-Agent-MATLAB

Offline, folder-backed RF-Agent for MATLAB RL training workflows.

This project mirrors the current `RF-Agent-Python` implementation while generating MATLAB reward functions:

```text
RF-Agent-MATLAB reads one task folder
RF-Agent-MATLAB generates candidate reward_fcn.m files
your MATLAB trainer trains pending candidates offline
your MATLAB trainer writes logs and optionally summary files
RF-Agent-MATLAB reloads the task folder, exports the best reward, and opens a Python dashboard window
```

## Task Folder

Each RL task lives in one folder:

```text
tasks/<task_name>/
  task.json
  description.md
  observations.md
  environment.md
  original_reward.m
  logs/
  candidates/
  visualization/
  best_reward_fcn.m
  best_reward_summary.json
```

Replace the starter files in [tasks/matlab_task](tasks/matlab_task) with your real task description, observation schema, original MATLAB reward, environment notes, and any reference logs.

Configure which task files are loaded into the LLM context in `task.json`:

```json
"task_text_files": {
  "description": {"path": "description.md", "title": "Purpose / Description"},
  "observations": {"path": "observations.md", "title": "Observations"},
  "environment": {"path": "environment.md", "title": "Environment"},
  "original_reward": {"path": "original_reward.m", "title": "Original Reward"}
}
```

## Candidate Folder

Each generated reward candidate is stored under the task folder:

```text
tasks/<task_name>/candidates/candidate_001/
  reward_fcn.m
  description.txt
  metadata.json
  status.json
  prompt_messages.json
  logs/
    train.csv
    eval.csv
  summary.json
```

`summary.json` can be written directly by MATLAB:

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

If MATLAB only returns CSV logs, run `sync` after copying logs into each candidate's `logs/` folder. RF-Agent-MATLAB builds `summary.json` from numeric CSV columns and marks those candidates as trained.

## RF-Agent Logic

This project keeps the same RF-Agent logic as `RF-Agent-Python`:

- Full initial expansion from the root.
- UCT-based selection of the best trained leaf before every next expansion.
- Four-candidate action-bundle generation from the selected node, sampled with action probabilities: `mutation` 30%, `crossover_elite` 30%, `tree_reasoning` 20%, and `different_thought` 20%.
- Actions: `initialize`, `mutation`, `crossover_elite`, `tree_reasoning`, `different_thought`.
- Parent/child tree rebuild from candidate metadata on every run.
- Q value, UCT, self-verification score, elite-set persistence, validation retries, CSV sync, best reward export, and dashboard rendering.

## Commands

Inspect current task state:

```bash
python src/main.py --mode inspect --task-dir tasks/matlab_task
```

Generate candidate rewards:

```bash
python src/main.py --mode generate --task-dir tasks/matlab_task
```

Sync CSV-only offline logs into scoreable summaries:

```bash
python src/main.py --mode sync --task-dir tasks/matlab_task
```

No-API smoke test:

```bash
python src/main.py --mode generate --task-dir tasks/matlab_task --dry-run
```

There is no separate visualize option. `inspect`, `generate`, `sync`, and the MATLAB worker template redraw the dashboard and open a Python window before ending. They also write:

```text
tasks/<task_name>/visualization/tree.json
tasks/<task_name>/visualization/tree_dashboard.png
```

The dashboard shows the whole search tree, parent/child relationships, node status, reward score, Q value, visits, UCT score, selected update parent, selected training candidates, persisted RF-Agent elite set, and final best reward emphasis.

## MATLAB Worker

MATLAB should:

1. Scan `tasks/<task_name>/candidates/candidate_*`.
2. Find candidates whose `status.json` has `"status": "pending"`.
3. Train/evaluate using `reward_fcn.m`.
4. Write `logs/train.csv` and/or `logs/eval.csv`.
5. Write `summary.json`, or run `python src/main.py --mode sync` to infer it from CSV logs.
6. Update `status.json` to `"trained"` or `"failed"`.
7. Run the dashboard step before ending.

See [matlab/run_pending_candidates_template.m](matlab/run_pending_candidates_template.m) for a worker template.

## OpenAI Setup

Set an API key before non-dry-run generation. The project automatically reads `RF-Agent-MATLAB/.env`:

```text
OPENAI_API_KEY=your_key_here
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_key_here"
```
