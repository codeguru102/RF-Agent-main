from __future__ import annotations

import os
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional


DEFAULT_MAX_TOKENS = 4096


class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        provider: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        dry_run: bool = False,
        dry_run_reward_signature: str = "function reward = reward_fcn(state, u_action, prev_u_action, flag)",
    ):
        self.model = model
        self.temperature = temperature
        self.provider = normalize_explicit_provider(provider)
        self.max_tokens = int(max_tokens or DEFAULT_MAX_TOKENS)
        self.dry_run = dry_run
        self.dry_run_reward_signature = dry_run_reward_signature

    def complete(self, messages: List[dict]) -> str:
        if self.dry_run:
            return self._dry_run_response(messages)

        load_dotenv()
        provider = self.current_provider()
        if provider == "anthropic":
            return self._complete_anthropic(messages)
        return self._complete_openai(messages)

    def current_provider(self) -> str:
        return resolve_provider(self.provider, self.model)

    def display_name(self) -> str:
        model = (self.model or "unknown").strip()
        if self.dry_run:
            return f"dry-run {model}"

        provider = self.current_provider()
        model_lower = model.lower()
        if provider == "anthropic":
            return model if model_lower.startswith("claude") else f"Claude {model}"
        if provider == "openai":
            return model if model_lower.startswith(("gpt", "o1", "o3", "o4")) else f"OpenAI {model}"
        return model

    def _complete_openai(self, messages: List[dict]) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or use --dry-run.")
        proxy_url = configure_proxy_env("openai")

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

    def _complete_anthropic(self, messages: List[dict]) -> str:
        api_key = first_env_value("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Put it in .env or use --dry-run.")
        configure_proxy_env("anthropic")

        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
            request = anthropic_request_payload(messages, self.model, self.temperature, self.max_tokens)
            for attempt in range(20):
                try:
                    response = client.messages.create(**request)
                    return anthropic_response_text(response)
                except Exception:
                    if attempt == 19:
                        raise
                    time.sleep(1)
        except ImportError:
            request = anthropic_request_payload(messages, self.model, self.temperature, self.max_tokens)
            return complete_anthropic_http(api_key, request)

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


def normalize_explicit_provider(provider: Optional[str]) -> str:
    value = (provider or "").strip().lower()
    if value in {"anthropic", "claude"}:
        return "anthropic"
    if value in {"openai", "gpt"}:
        return "openai"
    return ""


def resolve_provider(provider: Optional[str], model: str) -> str:
    explicit = normalize_explicit_provider(provider)
    if explicit:
        return explicit
    model_name = (model or "").strip().lower()
    if model_name.startswith("claude"):
        return "anthropic"
    return "openai"


def anthropic_request_payload(messages: List[dict], model: str, temperature: float, max_tokens: int) -> dict:
    system_parts = []
    anthropic_messages = []
    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        if role == "system":
            system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        anthropic_messages.append({"role": role, "content": content})

    request = {
        "model": model,
        "messages": merge_same_role_messages(anthropic_messages),
        "max_tokens": int(max_tokens or DEFAULT_MAX_TOKENS),
        "temperature": temperature,
    }
    if system_parts:
        request["system"] = "\n\n".join(part for part in system_parts if part.strip())
    return request


def merge_same_role_messages(messages: List[dict]) -> List[dict]:
    merged = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] += "\n\n" + message["content"]
        else:
            merged.append(dict(message))
    return merged or [{"role": "user", "content": ""}]


def anthropic_response_text(response) -> str:
    chunks = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            chunks.append(text)
    return "".join(chunks)


def complete_anthropic_http(api_key: str, request_payload: dict) -> str:
    body = json.dumps(request_payload).encode("utf-8")
    for attempt in range(20):
        try:
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=body,
                headers={
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "x-api-key": api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                return "".join(
                    block.get("text", "")
                    for block in data.get("content", [])
                    if block.get("type") == "text"
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if attempt == 19:
                raise RuntimeError(f"Anthropic API error {exc.code}: {detail}") from exc
            time.sleep(1)
        except Exception:
            if attempt == 19:
                raise
            time.sleep(1)


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


def configure_proxy_env(provider: str = "openai") -> str:
    provider_proxy = "ANTHROPIC_PROXY" if provider == "anthropic" else "OPENAI_PROXY"
    proxy_url = first_env_value(
        provider_proxy,
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
