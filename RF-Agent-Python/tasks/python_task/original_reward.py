def compute_reward(obs, action, next_obs, info):
    """Baseline reward placeholder.

    Replace this file with your original reward implementation. RF-Agent-Python
    includes this code in the LLM prompt as a reference, but generated candidates
    are saved separately as candidate_*/reward_fcn.py.
    """
    position_term = -abs(next_obs["position_error"])
    velocity_term = -0.1 * abs(next_obs["velocity_error"])
    torque = action["motor_torque"]
    control_term = -0.001 * sum(t * t for t in torque)
    constraint_term = -10.0 * info["constraint_violation"]
    success_bonus = 5.0 * float(info["success"])
    return position_term + velocity_term + control_term + constraint_term + success_bonus

