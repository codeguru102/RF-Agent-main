from __future__ import annotations

import os
import time
from typing import List


class LLMClient:
    def __init__(self, model: str, temperature: float = 1.0, dry_run: bool = False):
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run

    def complete(self, messages: List[dict]) -> str:
        if self.dry_run:
            return self._dry_run_response(messages)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use --dry-run for a local smoke test.")

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
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
  "reward_code": "def compute_reward(obs, action, next_obs, info):\\n    \\"\\"\\"Dry-run placeholder reward generated without an LLM call.\\"\\"\\"\\n    position_term = -abs(next_obs['position_error'])\\n    velocity_term = -0.1 * abs(next_obs['velocity_error'])\\n    torque = action['motor_torque']\\n    control_term = -0.001 * sum(t * t for t in torque)\\n    constraint_term = -10.0 * info['constraint_violation']\\n    success_bonus = 5.0 * float(info['success'])\\n    return position_term + velocity_term + control_term + constraint_term + success_bonus"
}"""
