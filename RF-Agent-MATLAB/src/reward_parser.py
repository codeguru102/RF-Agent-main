from __future__ import annotations

import json
import re
from typing import Tuple


def parse_reward_response(content: str) -> Tuple[str, str]:
    data = _try_parse_json(content)
    if data:
        thought = str(data.get("design_thought", "")).strip()
        code = str(data.get("reward_code", "")).strip()
        if code:
            return thought, code

    code = _extract_code_block(content)
    thought = content.replace(code, "").strip() if code else content.strip()
    return thought, code or content.strip()


def _try_parse_json(content: str):
    content = content.strip()
    candidates = [content]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL)
    candidates.extend(block.strip() for block in fenced)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_code_block(content: str) -> str:
    patterns = [
        r"```matlab\s*(.*?)```",
        r"```m\s*(.*?)```",
        r"```python\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    function_match = re.search(r"(function\s+.*)", content, flags=re.DOTALL | re.IGNORECASE)
    if function_match:
        return function_match.group(1).strip()
    return ""

