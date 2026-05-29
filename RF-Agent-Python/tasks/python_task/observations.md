# Observations

Replace this file with the observation dictionary/schema used by your trainer.

Example:

- `obs['position_error']`: current position tracking error.
- `obs['velocity_error']`: current velocity tracking error.
- `next_obs['position_error']`: next-step position tracking error.
- `next_obs['velocity_error']`: next-step velocity tracking error.
- `action['motor_torque']`: list or array of actuator torques.
- `info['constraint_violation']`: nonnegative violation amount.
- `info['success']`: boolean or numeric success indicator.

