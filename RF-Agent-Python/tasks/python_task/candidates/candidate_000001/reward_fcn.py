def compute_reward(obs, action, next_obs, info):
    """Dry-run placeholder reward generated without an LLM call."""
    position_term = -abs(next_obs['position_error'])
    velocity_term = -0.1 * abs(next_obs['velocity_error'])
    torque = action['motor_torque']
    control_term = -0.001 * sum(t * t for t in torque)
    constraint_term = -10.0 * info['constraint_violation']
    success_bonus = 5.0 * float(info['success'])
    return position_term + velocity_term + control_term + constraint_term + success_bonus
