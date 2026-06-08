from __future__ import annotations

import ast
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
    reward_language = str(task_config.get("reward_language", "python")).lower()
    if reward_language in {"matlab", "m"}:
        return validate_matlab_reward_code(code, task_config)
    return validate_python_reward_code(code, task_config)


def validate_python_reward_code(code: str, task_config: dict) -> RewardValidationResult:
    errors: List[str] = []
    stripped = code.strip()
    expected_name = task_config.get("reward_function_name", "reward_fcn")

    if not stripped:
        return RewardValidationResult(False, ["Reward code is empty."])
    if "```" in stripped:
        errors.append("Reward code must not include Markdown code fences.")

    try:
        tree = ast.parse(code)
        compile(tree, "<reward_fcn.py>", "exec")
    except SyntaxError as exc:
        return RewardValidationResult(False, [f"SyntaxError: {exc}"])
    except Exception as exc:
        return RewardValidationResult(False, [f"Compile error: {exc}"])

    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    reward_def = next((node for node in function_defs if node.name == expected_name), None)
    if reward_def is None:
        found = ", ".join(node.name for node in function_defs) or "none"
        errors.append(f"Expected function '{expected_name}', found: {found}.")
    else:
        expected_args = _expected_python_args(task_config.get("reward_signature", ""))
        if expected_args:
            actual_args = [arg.arg for arg in reward_def.args.args]
            if actual_args != expected_args:
                errors.append(f"Expected signature arguments {expected_args}, got {actual_args}.")

    forbidden_imports = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if forbidden_imports and not task_config.get("allow_imports", False):
        errors.append("Reward code should be self-contained and not import modules unless task.json sets allow_imports=true.")

    return RewardValidationResult(not errors, errors)


def validate_matlab_reward_code(code: str, task_config: dict) -> RewardValidationResult:
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

        expected_args = _expected_matlab_args(task_config.get("reward_signature", ""))
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


def _expected_python_args(signature: str) -> List[str]:
    if not signature:
        return []
    try:
        parsed = ast.parse(signature.rstrip(":") + ":\n    pass")
    except SyntaxError:
        return []
    function_def = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function_def is None:
        return []
    return [arg.arg for arg in function_def.args.args]


def _expected_matlab_args(signature: str) -> List[str]:
    match = re.search(r"\((.*?)\)", signature or "")
    if not match:
        return []
    args = []
    for raw_arg in match.group(1).split(","):
        name = raw_arg.strip().split("=", 1)[0].strip()
        if name:
            args.append(name)
    return args
