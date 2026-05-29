from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class RewardValidationResult:
    valid: bool
    errors: List[str]

    @property
    def message(self) -> str:
        return "\n".join(self.errors)


def validate_reward_code(code: str, task_config: dict) -> RewardValidationResult:
    errors: List[str] = []
    stripped = code.strip()
    expected_name = task_config.get("reward_function_name", "reward_fcn")

    if not stripped:
        return RewardValidationResult(False, ["Reward code is empty."])
    if "```" in stripped:
        errors.append("Reward code must not include Markdown code fences.")
    if re.search(r"^\s*def\s+\w+\s*\(", stripped, flags=re.MULTILINE):
        errors.append("Reward code looks like Python. Generate a MATLAB function instead.")
    if not re.search(r"^\s*function\b", stripped, flags=re.IGNORECASE | re.MULTILINE):
        errors.append("MATLAB reward code must define a function.")

    function_match = re.search(
        r"^\s*function\s+(?:(?:\[[^\]]+\]|\w+)\s*=\s*)?(?P<name>\w+)\s*\((?P<args>[^)]*)\)",
        stripped,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if function_match is None:
        errors.append(
            f"Expected MATLAB function signature for '{expected_name}', "
            "for example: function reward = reward_fcn(state, u_action, prev_u_action, flag)"
        )
    else:
        actual_name = function_match.group("name")
        if actual_name != expected_name:
            errors.append(f"Expected function '{expected_name}', found '{actual_name}'.")

        expected_args = _expected_args(task_config.get("reward_signature", ""))
        if expected_args:
            actual_args = [
                arg.strip().split("=", 1)[0].strip()
                for arg in function_match.group("args").split(",")
                if arg.strip()
            ]
            if actual_args != expected_args:
                errors.append(
                    f"Expected signature arguments {expected_args}, got {actual_args}."
                )

    if "reward" not in stripped:
        errors.append("Reward function should assign and return a reward value.")

    return RewardValidationResult(not errors, errors)


def _expected_args(signature: str) -> List[str]:
    match = re.search(r"\((.*?)\)", signature or "")
    if not match:
        return []
    args = []
    for raw_arg in match.group(1).split(","):
        name = raw_arg.strip().split("=", 1)[0].strip()
        if name:
            args.append(name)
    return args
