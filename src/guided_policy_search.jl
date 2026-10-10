# ============================================================================
# guided_policy_search.jl
#
# Ported from Alessandro-Work/sgnep/GuidedPolicySearch.jl, trimmed to the
# deterministic local step (SinglePlayerLQR / DeterministicLQNash), which is
# what gps_guidance.jl's `build_guidance`/`gps_builder` actually exercise for
# MADDPG training.
#
# SCOPE REDUCTION FROM THE ORIGINAL: `StochasticLQNash(noise)` and
# `local_exploitability` (lq_exploitability.jl) relied on Alessandro's
# original `NoiseModel` (additive + state-/control-multiplicative noise).
# DifferentialGamesBase.jl's reworked `StochasticGameProblem` only models
# additive noise today (via `ItoSDEDynamics`/`ConstantDiffusion` — see Base's
# docs/src/api/stochastic.md). Re-deriving the full multiplicative-noise
# exploitability/best-response machinery against the new vocabulary is left
# as explicit future work rather than half-ported; both are stubbed below
# with a clear error.
#
# The one idea, unchanged: guided policy search solves a local LQ problem
# every iteration, so making THAT pluggable turns the same outer loop into
# classic GPS or GPS for differential games with no other change.
# ============================================================================

using LinearAlgebra
using DifferentialGamesBase:
    expand, assemble_lq_game, rollout, n_steps, num_players, is_lq_game,
    total_control_dim, get_A, get_B

using DifferentialGamesBaseSolvers: FNELQ

import DifferentialGamesBase: solve

"""
    LocalGameSolver

Strategy for the local step: given an LTV LQ approximation of the game
around the current nominal trajectory, produce per-player affine feedback
laws. Choosing the concrete type selects the character of the method.
"""
abstract type LocalGameSolver end

"""
    SinglePlayerLQR()

Degenerate one-player case (classic GPS): dispatches to `FNELQ` on a
single-player game, asserting the player count.
"""
struct SinglePlayerLQR <: LocalGameSolver end

"""
    DeterministicLQNash()

Coupled Riccati recursion with no noise terms — plain `FNELQ` on the
assembled local LQ game. The default solver for `build_guidance`.
"""
struct DeterministicLQNash <: LocalGameSolver end

"""
    StochasticLQNash(noise)

Not ported this round — DifferentialGamesBase.jl's reworked
`StochasticGameProblem` only models additive noise, and this type
historically also covered state-/control-multiplicative noise. Construct an
additive-noise `StochasticGameProblem`/`SFNELQ` directly
(`DifferentialGamesBaseSolvers.SFNELQ`) if additive-noise local guidance is
needed; see this file's header for the full scope note.
"""
struct StochasticLQNash{N} <: LocalGameSolver
    noise::N
end

# ─── Data carried between steps ──────────────────────────────────────────────

"""
    LocalModel{T}

One condition's LTV LQ approximation: the original game, the expansion
about the nominal `(X, U)`, and the assembled LQ game the local solver
consumes. `X` is `(n × N+1)` and `U` is `(m × N)`, matching `expand`.
"""
struct LocalModel{T}
    game      :: Any
    expansion :: Any
    lq        :: Any
    X         :: Matrix{T}
    U         :: Matrix{T}
end

"""
    LocalPolicies{T}

Per-player affine feedback `u^i_t = -K[i][t] x_t - alpha[i][t]`, to be
sampled with covariance `Sigma[i][t]`.
"""
struct LocalPolicies{T}
    K              :: Vector{Vector{Matrix{T}}}
    alpha          :: Vector{Vector{Vector{T}}}
    Sigma          :: Vector{Vector{Matrix{T}}}
    expected_costs :: Vector{T}
end

Base.@kwdef struct GPSConfig
    n_iterations :: Int     = 20
    n_samples    :: Int     = 5
    kl_bound     :: Float64 = 2.0
    verbose      :: Bool    = true
end

# ─── Step 2 and 3: the local model ───────────────────────────────────────────

"""
    zero_control_nominal(game) -> (X, U)

The zero-control rollout, as a starting nominal.
"""
function zero_control_nominal(game)
    N     = n_steps(game)
    m     = total_control_dim(game.dynamics)
    times = collect(range(0.0, game.time_horizon.tf, length = N + 1))
    U     = zeros(Float64, m, N)
    X     = rollout(game.dynamics, game.initial_state, U, nothing, times)
    return Matrix{Float64}(X), U
end

"""
    local_model(game, X, U; regularize=false, da=nothing) -> LocalModel

Linearise and quadraticise `game` about `(X, U)` and assemble the LTV LQ
game. `regularize = false` by default (see Alessandro-Work's original
docstring: turning it on for an already-LQ game would make exactness checks
vacuous; turn it on for genuinely nonlinear games with an indefinite `Huu`).
"""
function local_model(game, X, U; regularize::Bool = false, da = nothing)
    ex = expand(game, X, U, da; regularize = regularize)
    lq = assemble_lq_game(ex, game)
    is_lq_game(lq) || error("local_model: assemble_lq_game did not return an LQ game")
    return LocalModel{Float64}(game, ex, lq, Matrix{Float64}(X), Matrix{Float64}(U))
end

affine_term_magnitude(lm::LocalModel) =
    isempty(lm.expansion.dynamics.c) ? 0.0 :
    maximum(norm(ck) for ck in lm.expansion.dynamics.c)

# ─── The contract ────────────────────────────────────────────────────────────

"""
    solve_local(solver, lm; prev=nothing, kl_bound=Inf) -> LocalPolicies

Solve the local LQ game and return per-player affine feedback with sampling
covariances. `kl_bound` must be `Inf` for now (the trust region is not
implemented — see Alessandro-Work/sgnep/GPS_DESIGN.md §4.2).
"""
function solve_local(solver::LocalGameSolver, lm::LocalModel; prev = nothing, kl_bound = Inf)
    isinf(kl_bound) || error(
        "solve_local: kl_bound = $kl_bound, but the trust region is not implemented.")
    solver isa StochasticLQNash && error(
        "StochasticLQNash local solves are not ported this round — see this file's header.")
    if solver isa SinglePlayerLQR
        np = num_players(lm.lq)
        np == 1 || error("SinglePlayerLQR: game has $np players. Use DeterministicLQNash.")
    end

    sol = solve(lm.lq, FNELQ())
    np  = num_players(lm.lq)
    N   = n_steps(lm.lq)
    dyn = lm.lq.dynamics
    Z   = sol.solver_info[:cost_to_go_matrices]

    K     = sol.strategy.gains
    alpha = sol.strategy.feedforward
    Sigma = [Vector{Matrix{Float64}}(undef, N) for _ in 1:np]
    for k in 1:N, i in 1:np
        Bi = get_B(dyn, i, k)
        Ri = get_objective_R(lm.lq, i, k)
        block = Ri + Bi' * Z[i][k+1] * Bi
        F = cholesky(Symmetric(block); check = false)
        issuccess(F) || error(
            "solve_local: control Hessian block for player $i at step $k is not " *
            "positive definite. Try regularize=true in local_model.")
        Sigma[i][k] = Matrix(Symmetric(inv(F)))
    end

    costs = [t.cost for t in sol.trajectories]
    return LocalPolicies{Float64}(K, alpha, Sigma, costs)
end

function get_objective_R(game, i::Int, k::Int)
    obj = only(filter(o -> o.player_id == i, game.objectives))
    DifferentialGamesBase.get_R(obj.stage_cost, k)
end

"""
    local_exploitability(solver, lm, pol)

Not ported this round — depends on lq_exploitability.jl's full
multiplicative-noise best-response machinery. See this file's header.
"""
local_exploitability(solver::LocalGameSolver, lm::LocalModel, pol::LocalPolicies) =
    error("local_exploitability is not ported this round — see guided_policy_search.jl header.")

# ─── Not implemented (ported verbatim from Alessandro-Work: still stubs there too) ─

fit_dynamics(samples) = error(
    "GuidedPolicySearch: fit_dynamics not implemented. The local step linearises the known model.")
fit_global_policy!(policy, states, actions, weights) = error(
    "GuidedPolicySearch: fit_global_policy! not implemented")
run_gps(game, solver::LocalGameSolver, policy, conditions, cfg::GPSConfig) = error(
    "GuidedPolicySearch: run_gps not implemented.")
