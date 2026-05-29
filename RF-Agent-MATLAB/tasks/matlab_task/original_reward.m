function reward = reward_fcn(state, u_action, prev_u_action, flag)
% Replace this starter reward with your current MATLAB baseline reward.
positionTerm = -abs(getfield_default(state, 'position_error', 0.0));
velocityTerm = -0.1 * abs(getfield_default(state, 'velocity_error', 0.0));
controlTerm = -0.001 * sum(u_action(:).^2);
successBonus = 5.0 * getfield_default(state, 'success', 0.0);
constraintTerm = -10.0 * getfield_default(state, 'constraint_violation', 0.0);
reward = positionTerm + velocityTerm + controlTerm + successBonus + constraintTerm;
end

function value = getfield_default(data, name, defaultValue)
if isstruct(data) && isfield(data, name)
    value = data.(name);
else
    value = defaultValue;
end
end
