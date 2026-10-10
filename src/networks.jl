# ============================================================================
# networks.jl
#
# Actor and critic network builders for MADDPG, using Flux.
#
# Per the paper (centralised training, decentralised execution):
#   - Actor_i:  own observation  ->  own action.
#               Output layer is tanh, so actions are bounded to [-1, 1]
#               (matching the environment's action_limit = 1).
#   - Critic_i: ALL observations and ALL actions  ->  a single Q-value.
#               The critic is centralised: it sees the whole joint
#               observation-action, which makes the environment look
#               stationary from its perspective.
#
# Float64 note: the environment produces Float64 states, so networks are
# forced to Float64 with `f64`. (Flux defaults to Float32; mixing the two
# silently degrades precision and is a classic source of subtle bugs.)
# ============================================================================

using Flux

"""
    build_actor(obs_dim, act_dim; hidden=64) -> Flux.Chain

MLP mapping a single agent's observation to its (bounded) action.
"""
function build_actor(obs_dim::Int, act_dim::Int; hidden::Int = 64)
    return f64(Chain(
        Dense(obs_dim => hidden, relu),
        Dense(hidden  => hidden, relu),
        Dense(hidden  => act_dim, tanh),   # bounded action in [-1, 1]
    ))
end

"""
    build_critic(total_obs_dim, total_act_dim; hidden=64) -> Flux.Chain

MLP mapping the concatenated [all observations; all actions] to a scalar
Q-value. Input width = total_obs_dim + total_act_dim.
"""
function build_critic(total_obs_dim::Int, total_act_dim::Int; hidden::Int = 64)
    return f64(Chain(
        Dense(total_obs_dim + total_act_dim => hidden, relu),
        Dense(hidden => hidden, relu),
        Dense(hidden => 1),                # scalar value (no activation)
    ))
end
