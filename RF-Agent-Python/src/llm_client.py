from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List


class LLMClient:
    def __init__(self, model: str, temperature: float = 1.0, dry_run: bool = False):
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run

    def complete(self, messages: List[dict]) -> str:
        if self.dry_run:
            return self._dry_run_response(messages)

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in RF-Agent-Python/.env or use --dry-run.")

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            for attempt in range(20):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content
                except Exception:
                    if attempt == 19:
                        raise
                    time.sleep(1)
        except ImportError:
            import openai

            openai.api_key = api_key
            for attempt in range(20):
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        n=1,
                    )
                    return response["choices"][0]["message"]["content"]
                except Exception:
                    if attempt == 19:
                        raise
                    time.sleep(1)

    def _dry_run_response(self, messages: List[dict]) -> str:
        return """{
  "design_thought": "Dry-run reward: penalize tracking errors, control effort, and constraint violations while adding a success bonus.",
  "reward_code": "def reward_fcn(state, u_action, prev_u_action=None, flag=None):\\n    \\"\\"\\"Dry-run placeholder reward generated without an LLM call.\\"\\"\\"\\n    position_term = -abs(next_obs['position_error'])\\n    velocity_term = -0.1 * abs(next_obs['velocity_error'])\\n    torque = action['motor_torque']\\n    control_term = -0.001 * sum(t * t for t in torque)\\n    constraint_term = -10.0 * info['constraint_violation']\\n    success_bonus = 5.0 * float(info['success'])\\n    return position_term + velocity_term + control_term + constraint_term + success_bonus"
}"""


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or find_dotenv()
    if env_path is None or not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()

        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_env_value(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value


def find_dotenv() -> Path | None:
    candidates = []
    cwd = Path.cwd().resolve()
    candidates.extend(parent / ".env" for parent in [cwd, *cwd.parents])

    project_root = Path(__file__).resolve().parents[1]
    candidates.append(project_root / ".env")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def strip_env_value(value: str) -> str:
    if "#" in value and not value.startswith(("'", '"')):
        value = value.split("#", 1)[0].rstrip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
