# RF-Agent-MATLAB

Offline, folder-backed RF-Agent for MATLAB training workflows.

This project is intentionally separate from the original `RF-Agent` repository. It keeps the RF-Agent reward generation and selection ideas, but replaces IsaacGym live training with an asynchronous file contract:

```text
RF-Agent-MATLAB generates reward candidates
MATLAB trains pending candidates offline
MATLAB writes logs and summary files
RF-Agent-MATLAB reloads folders and continues the search
```

## Main Concepts

- `candidate_*` folders are the source of truth.
- Every candidate stores reward code, metadata, status, logs, and a training summary.
- Parent-child metadata lets the project rebuild the RF-Agent tree on every run.
- The agent preserves RF-Agent-style actions:
  - `initialize`
  - `mutation_mechanism`
  - `mutation_param`
  - `crossover_elite`
  - `tree_reasoning`
  - `different_thought`
- MATLAB is treated as an offline worker. Python does not need real-time training access.

## Folder Contract

Each generated candidate looks like:

```text
experiments/<task_name>/candidate_000001/
  reward.m
  description.txt
  metadata.json
  status.json
  logs/
    train.csv
    eval.csv
  summary.json
```

`metadata.json` is written by RF-Agent-MATLAB:

```json
{
  "candidate_id": "candidate_000001",
  "parent_id": null,
  "action_type": "initialize",
  "action_index": 0,
  "generation": 0,
  "reward_language": "matlab",
  "reward_file": "reward.m",
  "design_thought": "...",
  "status": "pending"
}
```

`summary.json` is written by MATLAB:

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

## Generate Candidates

From `RF-Agent-MATLAB`:

```bash
python src/main.py --mode generate --num-candidates 8
```

For a no-API smoke test:

```bash
python src/main.py --mode generate --num-candidates 2 --dry-run
```

Inspect current state:

```bash
python src/main.py --mode inspect
```

## MATLAB Worker

MATLAB should:

1. Scan `experiments/<task_name>/candidate_*`.
2. Find candidates whose `status.json` has `"status": "pending"`.
3. Train/evaluate using `reward.m`.
4. Write `logs/train.csv` and/or `logs/eval.csv`.
5. Write `summary.json`.
6. Update `status.json` to `"trained"` or `"failed"`.

See `matlab/run_pending_candidates_template.m` for the expected worker shape.

## OpenAI Setup

Set an API key before non-dry-run generation:

```bash
set OPENAI_API_KEY=your_key_here
```

On PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_key_here"
```

