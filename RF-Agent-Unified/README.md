# RF-Agent-Unified

Offline, folder-backed RF-Agent for Python and MATLAB reward generation.

The reward language is configured per task in `task.json`:

```json
{
  "reward_language": "matlab",
  "reward_file": "reward_fcn.m",
  "reward_signature": "function [reward, isDone, test] = reward_fcn(Roll_dot, Roll_deg)"
}
```

or:

```json
{
  "reward_language": "python",
  "reward_file": "reward_fcn.py",
  "reward_signature": "def reward_fcn(state, action, next_obs, info):"
}
```

## Layout

```text
configs/agent.json
prompts/
  matlab/
  python/
src/
tasks/<task_name>/
  task.json
  candidates/
  logs/
```

`src/main.py` reads the task language, chooses `prompts/<reward_language>/`, validates generated code for that language, and writes the candidate reward file with the matching extension.

## Run

From this folder:

```powershell
python src/main.py --task-dir tasks/Motor_New
```

Generate without calling the LLM:

```powershell
python src/main.py --task-dir tasks/Motor_New --dry-run --internal-action generate
```

Sync candidate CSV logs into `summary.json`:

```powershell
python src/main.py --task-dir tasks/Motor_New --internal-action sync
```

## Scoring

Candidate tree selection uses the summary score plus the optional eval/Q score path:

```text
logs/eval.txt -> q_value_settings.calc_formula -> q_leaf_value / UCT
```

Prompt feedback also reports the eval/Q value when `eval.txt` is available, with the summary score as fallback.

