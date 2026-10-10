# ============================================================================
# gps_guidance.jl
#
# The bridge between the MADDPG solver in src/ and the local LQ-game solver in
# sgnep/GuidedPolicySearch.jl: it turns a policy into the guidance controller
# that `update_agents!` pulls the actors towards.
#
# WHAT THIS IS FOR
#
# Multi-agent guided policy search (Li et al. 2026) is MADDPG with one extra
# term in the actor loss,
#
#     ĉ^i_ρ(x,u) = c^i(x,u) + ρ ||u^i - ǔ^i(x)||²,
#
# where ǔ^i is the control a local LQ-game solution would apply. The paper's
# trust region is a quadratic penalty, not a KL constraint, which is why this
# needs no KL machinery — and why `solve_local` is called with `kl_bound = Inf`
# and the unimplemented trust region is never touched.
#
# So the whole of GPS, relative to MADDPG, is: build ǔ, and pass ρ. Setting
# ρ = 0 turns this file off and recovers MADDPG exactly. That is deliberate.
# A comparison between two algorithms is worth nothing if the two are separate
# implementations, because every difference then has two possible causes.
#
# HOW ǔ IS BUILT
#
#   1. Roll the CURRENT greedy policy out from the evaluation start, giving a
#      nominal (X, U).
#   2. Expand the game about that nominal and assemble the LTV LQ game
#      (`local_model`).
#   3. Solve it for the feedback Nash solution (`solve_local`), giving
#      u^i_t = -K^i_t x_t - α^i_t in the LOCAL coordinates.
#   4. Convert back: at nominal step t the guidance control is
#         ǔ^i(x) = Ū^i_t - α^i_t - K^i_t (x - X̄_t).
#      At x = X̄_t this is Ū^i_t - α^i_t, which is the form recorded in
#      SESSION_REPORT.md section 13c: the deviation is zero at the nominal, so
#      the absolute/deviation coordinate mismatch documented there does not
#      arise.
#
# WHICH STEP t
#
# The local solution is time-varying but the replay batch mixes timesteps and
# `Transition` does not record one. Each batch state is therefore matched to
# the NEAREST nominal state and given that step's law. This is an
# approximation, and the honest name for it is nearest-neighbour scheduling;
# the alternative is to record t in the buffer, which is a larger change.
# `guidance_diagnostics` reports the matching distance so it can be checked
# rather than assumed: if the batch states sit far from every nominal state,
# the guidance is being extrapolated and the numbers should not be trusted.
#
# RESOLVED LIMITATION (previously blocked discrete-time LQ games as MADDPG
# environments; fixed by the `discrete_dynamics`/`cost_state` fields added to
# `MADDPGEnv` in environment.jl)
#
# `MADDPGEnv` used to always forward-Euler a CONTINUOUS-time right-hand side,
# while a game carrying `LinearDynamics` (e.g. the CWH formation game, or the
# Li et al. Appendix A LQ game) is already a discrete-time map taking an
# integer step index — stepping it with forward Euler double-discretizes it
# and is a MethodError waiting to happen on the time argument. `MADDPGEnv` now
# takes `discrete_dynamics::Bool` (evaluate the map directly, via
# `evaluate_dynamics(dyn, x, u, p, env.t+1)`, when `true`) and `cost_state::Symbol`
# (`:joint` passes the full state to each player's stage cost — what
# cooperative_navigation/predator_prey are written against; `:player` passes
# only that player's own state slice — the convention `expand`/`quadraticize_costs`
# use everywhere else in DifferentialGamesBase.jl). `build_guidance` below
# reads `env.discrete_dynamics` to decide whether `expand` needs the Euler
# discretization built here or can linearise `LinearDynamics` exactly
# (`da = nothing`), so GPS guidance and MADDPG training now agree on which
# discrete map they are both approximating/exactly representing.
# ============================================================================

using LinearAlgebra: norm, I
using CommonRLInterface: reset!, act!, terminated
using DifferentialGamesBase:
    DiscreteApproximation, AbstractDiscretizationMethod,
    evaluate_dynamics, dynamics_jacobian,
    total_state_dim, total_control_dim

"""
    EulerDiscretization

The discretisation `MADDPGEnv` actually uses: one forward-Euler step of the
continuous right-hand side. It is not among the three methods
DifferentialGamesBase ships, all of which are more accurate, and that is the
point. `expand` must linearise THE MAP THE AGENT IS TRAINED ON, not a better
approximation of the same continuous system: a guidance law derived from an
RK4 map is a control law for a different discrete system, and the error shows
up as guidance that is subtly wrong everywhere with nothing to point at. The
first version of the bridge did exactly that and the test caught it — one env
step and one `rollout` step disagreed by 4.4e-03.
"""
struct EulerDiscretization <: AbstractDiscretizationMethod end

"""
    euler_da(game) -> DiscreteApproximation

Forward Euler at the game's own `dt`, with the analytic discrete Jacobian
`(I + dt A, dt B)` built from the continuous-RHS Jacobian, so the map and its
linearisation are consistent by construction.

CAVEAT, on record: `MADDPGEnv.act!` also clamps the control to
`env.action_limit` and may apply a state projection, and neither is modelled
here. The guidance is therefore exact for the unconstrained interior and an
approximation wherever the environment's own clipping bites.
"""
function euler_da(game)
    dyn = game.dynamics
    dt  = game.time_horizon.dt
    n   = total_state_dim(dyn)
    m   = total_control_dim(dyn)
    fwd = (x, u, p, t) -> x .+ dt .* evaluate_dynamics(dyn, x, u, p, t)
    jac = function (x, u, p, t)
        A, B = dynamics_jacobian(dyn, x, u, p, t)
        return (I + dt .* A, dt .* B)
    end
    return DiscreteApproximation{typeof(dt), typeof(fwd), typeof(jac)}(
        fwd, jac, n, m, dt, EulerDiscretization())
end

"""
    greedy_nominal(agents, env; start = nothing) -> (X, U)

Roll the current greedy (noise-free) policy out from `start`, or from the
environment's own reset if `start` is `nothing`. Returns `X` as `(n, N+1)` and
`U` as `(m, N)`, the shapes `expand` wants.
"""
function greedy_nominal(agents, env; start = nothing)
    if start === nothing
        reset!(env)
    else
        reset_to!(env, start)
    end
    xs = Vector{Vector{Float64}}()
    us = Vector{Vector{Float64}}()
    push!(xs, copy(env.state))
    while !terminated(env)
        obs = local_observations(env)
        a   = select_actions(agents, obs; noise_std = 0.0)
        u   = reduce(vcat, a)
        act!(env, u)
        push!(us, copy(u))
        push!(xs, copy(env.state))
    end
    isempty(us) && error("greedy_nominal: the environment terminated immediately")
    return reduce(hcat, xs), reduce(hcat, us)
end

"""
    GuidanceLaw

The affine guidance controller, in ABSOLUTE coordinates, sampled at each
nominal step. `Ustar[i]` is `(act_dim, N)`: the guidance control at the
nominal itself. `K[i][t]` multiplies the deviation from `Xbar[:, t]`.
"""
struct GuidanceLaw
    Xbar    :: Matrix{Float64}                 # (n, N+1)
    Ustar   :: Vector{Matrix{Float64}}         # per player, (act_dim, N)
    K       :: Vector{Vector{Matrix{Float64}}} # per player, per step
    offsets :: Vector{UnitRange{Int}}          # control rows per player
    clamp_to:: Tuple{Float64,Float64}
end

"""
    build_guidance(agents, env, solver_type; start, clamp_to) -> GuidanceLaw

Steps 1-4 above. `solver_type` is a `LocalGameSolver`; `DeterministicLQNash()`
is the one that matches the deterministic MADDPG environments, and
`StochasticLQNash(noise)` the noisy ones.
"""
function build_guidance(agents, env, solver_type = DeterministicLQNash();
                        start = nothing, clamp_to = (-1.0, 1.0),
                        regularize::Bool = false, da = nothing)
    X, U = greedy_nominal(agents, env; start = start)
    # The expansion must linearise THE MAP THE ENVIRONMENT STEPS. A
    # continuous-time game is stepped by forward Euler, so it gets the
    # matching Euler approximation. A game whose dynamics are already a
    # discrete map (`discrete_dynamics = true`, LinearDynamics) must NOT: Euler
    # would linearise I + dt*A instead of A. `expand` handles LinearDynamics
    # exactly when given `da = nothing`.
    dapp = da !== nothing ? da :
           (env.discrete_dynamics ? nothing : euler_da(env.game))
    lm   = local_model(env.game, X, U; regularize = regularize, da = dapp)
    pol  = solve_local(solver_type, lm; kl_bound = Inf)

    np   = length(pol.K)
    dims = env.game.metadata.control_dims
    offs = UnitRange{Int}[]
    o    = 1
    for d in dims
        push!(offs, o:(o + d - 1))
        o += d
    end

    N = size(U, 2)
    Ustar = Vector{Matrix{Float64}}(undef, np)
    for i in 1:np
        M = Matrix{Float64}(undef, length(offs[i]), N)
        for t in 1:N
            # Ū^i_t - α^i_t : the guidance control AT the nominal.
            M[:, t] = U[offs[i], t] .- pol.alpha[i][t]
        end
        Ustar[i] = M
    end
    return GuidanceLaw(X, Ustar, pol.K, offs, clamp_to)
end

"""
    nearest_step(g, x) -> t

Index of the nominal state closest to `x`, restricted to `1:N` (the last
nominal state has no control after it).
"""
function nearest_step(g::GuidanceLaw, x::AbstractVector)
    N = size(g.Ustar[1], 2)
    best, bd = 1, Inf
    for t in 1:N
        d = 0.0
        @inbounds for r in eachindex(x)
            δ = x[r] - g.Xbar[r, t]
            d += δ * δ
        end
        if d < bd
            bd, best = d, t
        end
    end
    return best, sqrt(bd)
end

"""
    (g::GuidanceLaw)(states, i) -> (act_dim, B) matrix

The signature `update_agents!` expects.
"""
function (g::GuidanceLaw)(states, i::Int)
    B  = length(states)
    ad = length(g.offsets[i])
    M  = Matrix{Float64}(undef, ad, B)
    lo, hi = g.clamp_to
    for b in 1:B
        x = states[b]
        t, _ = nearest_step(g, x)
        δ = x .- g.Xbar[:, t]
        M[:, b] = clamp.(g.Ustar[i][:, t] .- g.K[i][t][:, :] * δ, lo, hi)
    end
    return M
end

"""
    guidance_diagnostics(g, states) -> NamedTuple

How far the batch sits from the nominal, and how often the guidance control
saturates against the actor's own output range. Both are ways for the guidance
to be silently meaningless: a batch far from the nominal is being extrapolated
from, and a guidance control pinned at ±1 carries no gradient information
beyond "go to the edge".
"""
function guidance_diagnostics(g::GuidanceLaw, states)
    ds = [nearest_step(g, x)[2] for x in states]
    np = length(g.Ustar)
    sat = Float64[]
    for i in 1:np
        M = g(states, i)
        lo, hi = g.clamp_to
        push!(sat, count(u -> u <= lo + 1e-9 || u >= hi - 1e-9, M) / length(M))
    end
    return (mean_dist = sum(ds)/length(ds), max_dist = maximum(ds),
            saturated_fraction = sat)
end

"""
    gps_builder(solver_type = DeterministicLQNash(); start, clamp_to, quiet)

Returns the `guidance_builder` callable that `MADDPGSolver` expects. Failures
(a non-PSD control Hessian, a local model that will not assemble) return
`nothing`, which the solver logs and treats as "continue unguided" rather than
aborting a long run.
"""
function gps_builder(solver_type = DeterministicLQNash();
                     start = nothing, clamp_to = (-1.0, 1.0),
                     regularize::Bool = false, da = nothing, quiet::Bool = false)
    return function (agents, env)
        try
            return build_guidance(agents, env, solver_type;
                                  start = start, clamp_to = clamp_to,
                                  regularize = regularize, da = da)
        catch err
            quiet || @warn "gps_builder: local solve failed, continuing unguided" exception=err
            return nothing
        end
    end
end
