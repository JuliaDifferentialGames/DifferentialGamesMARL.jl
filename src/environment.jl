# ============================================================================
# environment.jl
#
# MADDPGEnv: wraps a DifferentialGamesBase.GameProblem so it behaves as a
# CommonRLInterface.AbstractEnv.
#
# Mapping (see architecture doc):
#   game.dynamics       -> state update inside act!   (forward-Euler with dt)
#   game.objectives     -> rewards returned by act!   (reward = -cost)
#   game.initial_state  -> what reset! restores
#   game.time_horizon   -> when terminated returns true
#   game.constraints    -> handled inside act! via a SWAPPABLE strategy
#                          (see `constraint_mode`). Default = :barrier.
#
# Constraint-handling strategies (set via constraint_mode):
#   :barrier  (DEFAULT, per Bennet's recommendation)
#       A smooth, always-defined barrier cost is subtracted from reward:
#           cost = barrier_weight * exp(barrier_sharpness * g)
#       For an inequality g(x,u) <= 0 (satisfied when g < 0): the cost is ~0
#       deep inside the feasible region, equals barrier_weight at the boundary
#       (g = 0), and grows sharply once violated (g > 0). Unlike a classical
#       log barrier it is finite everywhere, so it tolerates the infeasible
#       states that random / early-training policies inevitably produce.
#   :penalty
#       The original strategy: subtract penalty_weight * max(0, g) for an
#       inequality, penalty_weight*|g| for an equality.
#   :none
#       Constraints ignored (useful when debugging the learner in isolation).
#
# Multi-agent conventions:
#   - Joint action u = [u_1; u_2; ...; u_n], sliced via control_offsets.
#   - act! returns a Vector of rewards (index = player_id).
#   - observe returns the full joint state.
#   - local_observations(env) returns the PER-AGENT observations the MADDPG
#     actors consume. By default each agent observes the full joint state; an
#     environment builder may inject a custom obs_fn (cooperative navigation
#     does this so agents can perceive the landmarks).
# ============================================================================

"""
    MADDPGEnv(game; constraint_mode=:barrier, kwargs...)

CommonRLInterface environment wrapping a `GameProblem`.

# Keyword arguments
- `constraint_mode`   : `:barrier` (default), `:penalty`, or `:none`
- `barrier_weight`    : scale of the barrier cost (default 1.0)
- `barrier_sharpness` : how sharply the barrier grows near the boundary
                        (default 8.0; larger = steeper wall)
- `barrier_cap`       : upper limit on the barrier cost of a single constraint
                        (default Inf = uncapped). Useful when deep violations
                        are reachable (e.g. an agent far outside a soft arena
                        boundary): the exponential barrier would otherwise
                        produce rewards orders of magnitude larger than the
                        stage costs, destabilising early training.
- `penalty_weight`    : weight used when `constraint_mode == :penalty`
- `action_limit`      : controls clamped to [-action_limit, action_limit]
- `obs_fn`            : optional `state -> Vector{Vector{Float64}}` giving the
                        per-agent observations; if `nothing`, every agent
                        observes the full joint state
- `state_projection`  : optional in-place function `state -> state` applied
                        after every integration step, projecting the state
                        back into the feasible set (e.g. a hard arena wall
                        that clamps positions and zeroes outward velocity).
                        `nothing` (default) disables projection. Use this for
                        constraints that must NEVER be violated; barrier /
                        penalty costs then only shape behaviour near the
                        boundary.
- `reset_fn`          : optional zero-argument function returning a FRESH
                        initial state for each episode. `nothing` (default)
                        reproduces the historical behaviour exactly: every
                        episode starts from `game.initial_state`. Supply one
                        (e.g. the `sample_start` returned by
                        `cooperative_navigation` / `predator_prey`) to train
                        and evaluate over a DISTRIBUTION of initial
                        conditions rather than a single trajectory. A Nash
                        equilibrium of a differential game is defined over
                        the state space; a fixed start measures one path
                        through it.
"""
mutable struct MADDPGEnv{G<:GameProblem,F,P,R} <: AbstractEnv
    game::G
    state::Vector{Float64}
    t::Int
    constraint_mode::Symbol
    barrier_weight::Float64
    barrier_sharpness::Float64
    barrier_cap::Float64
    penalty_weight::Float64
    action_limit::Float64
    obs_fn::F
    state_projection::P
    reset_fn::R
    # ── Conventions, both defaulting to the historical behaviour ────────────
    # `discrete_dynamics = false` forward-Eulers the continuous right-hand
    # side, which is what every task here has always done. `true` treats the
    # dynamics as a discrete map x_{k+1} = f(x_k, u_k, k), which is what
    # `LinearDynamics` actually is: forward-Eulering a map that is already a
    # map is simply wrong, and it is why the CWH formation game could not be
    # wrapped as an environment at all.
    #
    # `cost_state = :joint` passes the FULL state to a player's stage cost,
    # which is what cooperative_navigation and predator_prey are written
    # against (their coverage and pursuit costs are functions of every
    # agent's position). `:player` passes that player's own state slice,
    # which is the convention DifferentialGamesBase itself uses everywhere
    # else, including inside `expand`. The two cannot both be right, and a
    # game built for one is silently wrong under the other, so this is
    # explicit rather than inferred.
    discrete_dynamics::Bool
    cost_state::Symbol
end

function MADDPGEnv(game::GameProblem;
                   constraint_mode::Symbol = :barrier,
                   barrier_weight::Real     = 1.0,
                   barrier_sharpness::Real  = 8.0,
                   barrier_cap::Real        = Inf,
                   penalty_weight::Real     = 10.0,
                   action_limit::Real       = 1.0,
                   obs_fn = nothing,
                   state_projection = nothing,
                   reset_fn = nothing,
                   discrete_dynamics::Bool = false,
                   cost_state::Symbol = :joint)
    constraint_mode in (:barrier, :penalty, :none) ||
        throw(ArgumentError("constraint_mode must be :barrier, :penalty, or :none"))
    cost_state in (:joint, :player) ||
        throw(ArgumentError("cost_state must be :joint or :player"))
    # `:player` only means something when the game actually declares a state
    # block per player. A game with one shared state block (the CWH formation
    # game, for instance) has no per-player slice to pass, and silently
    # passing the whole state under a flag that says otherwise is exactly the
    # kind of convention drift this flag exists to stop.
    if cost_state === :player &&
       length(game.metadata.state_dims) != game.n_players
        throw(ArgumentError(
            "cost_state = :player needs one state block per player, but the " *
            "game declares $(length(game.metadata.state_dims)) block(s) for " *
            "$(game.n_players) players. Use :joint."))
    end
    return MADDPGEnv(game, copy(game.initial_state), 0,
                     constraint_mode,
                     Float64(barrier_weight), Float64(barrier_sharpness),
                     Float64(barrier_cap),
                     Float64(penalty_weight), Float64(action_limit),
                     obs_fn, state_projection, reset_fn,
                     discrete_dynamics, cost_state)
end

"""
    player_state(env, i) -> view

Player i's own slice of the joint state, using the game's own offsets.
"""
function player_state(env::MADDPGEnv, x::AbstractVector, i::Int)
    off = env.game.metadata.state_offsets[i]
    dim = env.game.metadata.state_dims[i]
    return view(x, off+1:off+dim)
end

# ─── Small helpers ───────────────────────────────────────────────────────────

"Number of agents/players in the wrapped game."
num_agents(env::MADDPGEnv) = env.game.n_players

"Total dimension of the stacked joint control vector."
joint_control_dim(env::MADDPGEnv) = sum(env.game.metadata.control_dims)

"View of player i's slice of the joint control vector `u`."
function player_control(env::MADDPGEnv, u::AbstractVector, i::Int)
    off = env.game.metadata.control_offsets[i]
    dim = env.game.metadata.control_dims[i]
    return view(u, off+1:off+dim)
end

"""
    local_observations(env) -> Vector{Vector{Float64}}

Per-agent observations consumed by the MADDPG actors. Uses `env.obs_fn` if one
was supplied at construction; otherwise every agent observes the full joint
state.
"""
function local_observations(env::MADDPGEnv)
    if env.obs_fn === nothing
        return [copy(env.state) for _ in 1:num_agents(env)]
    else
        return env.obs_fn(env.state)
    end
end

# ─── Required CommonRLInterface functions ────────────────────────────────────

function CommonRLInterface.reset!(env::MADDPGEnv)
    if env.reset_fn === nothing
        env.state .= env.game.initial_state
    else
        s = env.reset_fn()
        length(s) == length(env.state) ||
            throw(DimensionMismatch("reset_fn returned a state of length " *
                  "$(length(s)); expected $(length(env.state))"))
        env.state .= s
    end
    env.t = 0
    return nothing
end

"""
    reset_to!(env, state)

Reset the environment to an EXPLICIT state, bypassing both `reset_fn` and
`game.initial_state`. This is what makes a held-out evaluation set possible:
the same start states can be replayed at every evaluation point, so successive
evaluations differ only because the policy changed.
"""
function reset_to!(env::MADDPGEnv, state::AbstractVector)
    length(state) == length(env.state) ||
        throw(DimensionMismatch("state has length $(length(state)); " *
              "environment expects $(length(env.state))"))
    env.state .= state
    env.t = 0
    return nothing
end

CommonRLInterface.observe(env::MADDPGEnv) = copy(env.state)

CommonRLInterface.terminated(env::MADDPGEnv) = env.t >= env.game.time_horizon.N

"""
    act!(env, u) -> Vector{Float64}

Apply joint control `u`, advance one forward-Euler step of length `dt`, and
return per-player rewards (negated stage cost minus the constraint cost
dictated by `constraint_mode`).
"""
function CommonRLInterface.act!(env::MADDPGEnv, u::AbstractVector)
    game = env.game
    dt   = game.time_horizon.dt

    uc = clamp.(u, -env.action_limit, env.action_limit)

    # 1. Dynamics.
    if env.discrete_dynamics
        # Already a discrete map: evaluate it, do not integrate it. The time
        # argument is the INTEGER step index, which is what LinearDynamics
        # dispatches on; passing a Float64 here is the MethodError that made
        # the CWH game unusable as an environment.
        xn = evaluate_dynamics(game.dynamics, env.state, uc, nothing, env.t + 1)
        env.state .= xn
    else
        t_real = env.t * dt
        xdot = evaluate_dynamics(game.dynamics, env.state, uc, nothing, t_real)
        env.state .+= dt .* xdot
    end
    env.t += 1

    # 1b. Optional projection onto the feasible set (e.g. hard arena wall).
    # Applied before rewards, so costs are evaluated at the projected state.
    if env.state_projection !== nothing
        env.state_projection(env.state)
    end

    # 2. Objectives -> rewards (negated costs).
    rewards = zeros(game.n_players)
    for obj in game.objectives
        ui   = player_control(env, uc, obj.player_id)
        xc   = env.cost_state === :joint ? env.state :
                                          player_state(env, env.state, obj.player_id)
        cost = evaluate_stage_cost(obj.stage_cost, xc, ui, nothing, env.t)
        rewards[obj.player_id] = -obj.scaling * cost
    end

    # 3. Constraints -> reward cost, via the selected strategy.
    if env.constraint_mode != :none
        for c in game.private_constraints
            cost = _constraint_cost(env, c, uc)
            rewards[get_player(c)] -= cost
        end
        for c in game.shared_constraints
            cost = _constraint_cost(env, c, uc)
            for p in get_players(c)
                rewards[p] -= cost
            end
        end
    end

    return rewards
end

"""
    _constraint_cost(env, c, u) -> Float64

Reward cost from one constraint `c` evaluated at the current state and control,
according to env.constraint_mode. Convention: inequality satisfied when
g(x,u) <= 0; equality target is g(x,u) == 0.
"""
function _constraint_cost(env::MADDPGEnv, c, u)
    g = evaluate_constraint(c, env.state, u, nothing, env.t)
    if env.constraint_mode === :barrier
        if is_inequality(c)
            # exp barrier: small when g<<0, == weight at g=0, sharp for g>0.
            # Capped at barrier_cap so deep violations stay commensurate with
            # the stage-cost scale (cap = Inf reproduces the pure barrier).
            return min(env.barrier_cap,
                       env.barrier_weight *
                       sum(gi -> exp(env.barrier_sharpness * gi), g))
        else
            # equality: symmetric exponential well around g == 0.
            return min(env.barrier_cap,
                       env.barrier_weight *
                       sum(gi -> exp(env.barrier_sharpness * abs(gi)) - 1.0, g))
        end
    else  # :penalty
        return env.penalty_weight *
               (is_inequality(c) ? sum(gi -> max(gi, 0.0), g) : sum(abs, g))
    end
end

"""
    actions(env)

Continuous action space: one (lower, upper) tuple per component of the joint
control vector. Sufficient for sampling random actions.
"""
function CommonRLInterface.actions(env::MADDPGEnv)
    return fill((-env.action_limit, env.action_limit), joint_control_dim(env))
end
