from __future__ import annotations

import ast
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
    try:
        tree = ast.parse(code)
        compile(tree, "<reward_fcn.py>", "exec")
    except SyntaxError as exc:
        return RewardValidationResult(False, [f"SyntaxError: {exc}"])
    except Exception as exc:
        return RewardValidationResult(False, [f"Compile error: {exc}"])

    expected_name = task_config.get("reward_function_name", "compute_reward")
    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    reward_def = next((node for node in function_defs if node.name == expected_name), None)
    if reward_def is None:
        found = ", ".join(node.name for node in function_defs) or "none"
        errors.append(f"Expected function '{expected_name}', found: {found}.")
    else:
        expected_args = _expected_args(task_config.get("reward_signature", ""))
        if expected_args:
            actual_args = [arg.arg for arg in reward_def.args.args]
            if actual_args != expected_args:
                errors.append(
                    f"Expected signature arguments {expected_args}, got {actual_args}."
                )

    forbidden_imports = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if forbidden_imports and not task_config.get("allow_imports", False):
        errors.append("Reward code should be self-contained and not import modules unless task.json sets allow_imports=true.")

    return RewardValidationResult(not errors, errors)


def _expected_args(signature: str) -> List[str]:
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
