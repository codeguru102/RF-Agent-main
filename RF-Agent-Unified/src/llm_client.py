from __future__ import annotations

import os
import json
import re
import time
from pathlib import Path
from typing import List


class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        dry_run: bool = False,
        dry_run_reward_signature: str = "function reward = reward_fcn(state, u_action, prev_u_action, flag)",
    ):
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run
        self.dry_run_reward_signature = dry_run_reward_signature

    def complete(self, messages: List[dict]) -> str:
        if self.dry_run:
            return self._dry_run_response(messages)

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in RF-Agent-Unified/.env or use --dry-run.")
        proxy_url = configure_proxy_env()

        try:
            from openai import DefaultHttpxClient, OpenAI

            client_kwargs = {"api_key": api_key}
            if proxy_url:
                client_kwargs["http_client"] = DefaultHttpxClient(proxy=proxy_url)
            client = OpenAI(**client_kwargs)
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
            if proxy_url:
                openai.proxy = proxy_url
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
        signature = self.dry_run_reward_signature.strip().rstrip(";")
        if signature.lstrip().startswith("def "):
            code = _python_dry_run_code(signature)
        else:
            if not signature.lower().startswith("function"):
                signature = "function reward = reward_fcn(state, u_action, prev_u_action, flag)"
            code = _matlab_dry_run_code(signature)
        return json.dumps(
            {
                "design_thought": "Dry-run reward: penalize tracking errors, control effort, and constraint violations while adding a success bonus.",
                "reward_code": code,
            },
            indent=2,
        )


def _python_dry_run_code(signature: str) -> str:
    signature = signature.rstrip().rstrip(":")
    args = _signature_args(signature)
    action = args[1] if len(args) > 1 else "action"
    next_obs = args[2] if len(args) > 2 else "next_obs"
    info = args[3] if len(args) > 3 else "info"
    return "\n".join(
        [
            f"{signature}:",
            '    """Dry-run placeholder reward generated without an LLM call."""',
            f"    position_term = -abs({next_obs}.get('position_error', 0.0)) if hasattr({next_obs}, 'get') else 0.0",
            f"    velocity_term = -0.1 * abs({next_obs}.get('velocity_error', 0.0)) if hasattr({next_obs}, 'get') else 0.0",
            f"    torque = {action}.get('motor_torque', []) if hasattr({action}, 'get') else []",
            "    control_term = -0.001 * sum(t * t for t in torque)",
            f"    constraint_term = -10.0 * float({info}.get('constraint_violation', 0.0)) if hasattr({info}, 'get') else 0.0",
            f"    success_bonus = 5.0 * float({info}.get('success', 0.0)) if hasattr({info}, 'get') else 0.0",
            "    return position_term + velocity_term + control_term + constraint_term + success_bonus",
        ]
    )


def _matlab_dry_run_code(signature: str) -> str:
    args = _signature_args(signature)
    state = args[0] if len(args) > 0 else "state"
    action = args[1] if len(args) > 1 else "u_action"
    return "\n".join(
        [
            signature,
            "% Dry-run placeholder reward generated without an LLM call.",
            f"positionTerm = -abs(getfield_default({state}, 'position_error', 0.0));",
            f"velocityTerm = -0.1 * abs(getfield_default({state}, 'velocity_error', 0.0));",
            f"controlTerm = -0.001 * sum({action}(:).^2);",
            f"successBonus = 5.0 * getfield_default({state}, 'success', 0.0);",
            f"constraintTerm = -10.0 * getfield_default({state}, 'constraint_violation', 0.0);",
            "reward = positionTerm + velocityTerm + controlTerm + successBonus + constraintTerm;",
            "end",
            "",
            "function value = getfield_default(data, name, defaultValue)",
            "if isstruct(data) && isfield(data, name)",
            "    value = data.(name);",
            "else",
            "    value = defaultValue;",
            "end",
            "end",
        ]
    )


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


def configure_proxy_env() -> str:
    proxy_url = first_env_value(
        "OPENAI_PROXY",
        "PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    )
    if not proxy_url:
        return ""

    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        os.environ.setdefault(key, proxy_url)
    return proxy_url


def first_env_value(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return ""


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


def _signature_args(signature: str) -> List[str]:
    match = re.search(r"\((.*?)\)", signature)
    if not match:
        return []
    args = []
    for raw_arg in match.group(1).split(","):
        name = raw_arg.strip().split("=", 1)[0].strip()
        if name:
            args.append(name)
    return args
