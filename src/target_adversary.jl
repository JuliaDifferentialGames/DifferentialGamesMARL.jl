# ============================================================================
# target_adversary.jl
#
# The two "target and adversary" benchmark tasks of the MADDPG paper
# (Lowe et al., arXiv 1706.02275, Sec. 5.1), defined as GameProblems:
#
#   keep_away          cooperating agents know which landmark is the target and
#                      are rewarded for reaching it; adversaries must keep them
#                      off it but do not know which landmark is the target.
#   physical_deception cooperating agents are rewarded when ANY of them reaches
#                      the target and when the adversary is far from it; the
#                      adversary is rewarded for reaching the target but does
#                      not know which landmark it is, so the agents must spread
#                      out over the landmarks to deceive it.
#
# Modelling choices (all deviations from the paper's particle environment are
# stated here rather than hidden):
#   - Same damped double-integrator point mass as cooperative navigation, so
#     both tasks are SeparableDynamics PD-GNEPs and run with any solver that
#     supports them.
#   - Rewards are the multi-agent particle environment's: costs are the
#     negative rewards,
#         agent cost (keep-away)          ||p_i - g|| - min_k ||q_k - g||
#         agent cost (physical deception) min_j ||p_j - g|| - ||q - g||
#         adversary cost                  ||q - g||
#     plus a small control penalty; p are agent and q adversary positions and
#     g the target. (In keep-away each agent is paid its OWN distance; in
#     physical deception the team shares the distance of its best member.)
#   - The paper's physical pushing is NOT reproduced: the particle environment
#     resolves overlaps with soft contact forces, which couple the agents'
#     dynamics and would break the separable structure. Keep-away instead uses
#     pairwise collision-avoidance constraints between each agent and each
#     adversary (an adversary can BLOCK an agent by occupying space but cannot
#     shove it). Physical deception uses the agent-agent constraints only.
#   - Information asymmetry lives in the observation function: the adversaries'
#     observation has the target slot zeroed. Observations have the same length
#     for every player, which the MADDPG learner relies on.
# ============================================================================

function _target_adversary_task(kind::Symbol;
    n_agents::Int, n_adversaries::Int, n_landmarks::Int, target::Int,
    dt::Float64, tf::Float64, accel::Float64, accel_adversary::Float64,
    damping::Float64, agent_radius::Float64, control_penalty::Float64,
    arena::Float64)

    n_x, n_u = 4, 2
    n_total  = n_agents + n_adversaries
    @assert 1 <= target <= n_landmarks "target must index one of the landmarks"
    @assert kind in (:keep_away, :physical_deception)

    # ── Layout ──────────────────────────────────────────────────────────────
    # Agents start on an outer ring, adversaries near the centre, landmarks on
    # an intermediate ring. Fixed so results are reproducible; `sample_start`
    # randomises the start of every episode.
    ring(r, k, K, φ = 0.0) = r .* [cos(2π * k / K + φ), sin(2π * k / K + φ)]
    agent_pos = [ring(0.9, k, n_agents) for k in 0:n_agents-1]
    adv_pos   = [ring(0.25, k, n_adversaries, π / 4) for k in 0:n_adversaries-1]
    landmarks = [ring(0.55, k, n_landmarks, π / n_landmarks) for k in 0:n_landmarks-1]
    starts    = [vcat(p, [0.0, 0.0]) for p in vcat(agent_pos, adv_pos)]
    goal      = landmarks[target]

    # ── Dynamics ────────────────────────────────────────────────────────────
    make_point_mass = a -> (xi, ui, p, t) -> [xi[3], xi[4],
                                              a * ui[1] - damping * xi[3],
                                              a * ui[2] - damping * xi[4]]
    accels = [fill(accel, n_agents); fill(accel_adversary, n_adversaries)]
    dyn = SeparableDynamics([make_point_mass(a) for a in accels],
                            fill(n_x, n_total), fill(n_u, n_total))

    pos_of(x, a) = (x[n_x*(a-1)+1], x[n_x*(a-1)+2])
    dist_to_goal(x, a) = (p = pos_of(x, a); sqrt((p[1] - goal[1])^2 + (p[2] - goal[2])^2))
    adversaries = n_agents+1:n_total
    min_adv_dist(x) = minimum(a -> dist_to_goal(x, a), adversaries)
    min_agent_dist(x) = minimum(a -> dist_to_goal(x, a), 1:n_agents)

    # Cost of player i as a function of the joint state. The control is
    # selected from the joint vector when a solver passes one (ALGAMES), so the
    # control penalty is always the player's own.
    cost_of(i) = i <= n_agents ?
        (kind === :keep_away ? (x -> dist_to_goal(x, i) - min_adv_dist(x)) :
                               (x -> min_agent_dist(x) - min_adv_dist(x))) :
        (x -> dist_to_goal(x, i))
    own_u(i, u) = length(u) == n_u * n_total ? u[n_u*(i-1)+1:n_u*i] : u

    players = [
        PlayerSpec(i, n_x, n_u, starts[i], make_point_mass(accels[i]),
                   PlayerObjective(i,
                       NonlinearStageCost((x, u, p, t) ->
                           cost_of(i)(x) + control_penalty * sum(abs2, own_u(i, u))),
                       NonlinearTerminalCost((x, p) -> cost_of(i)(x))),
                   Any[state_bounds(i; state_offset = n_x*(i-1), state_dim = n_x,
                                    lower = [-arena, -arena, -Inf, -Inf],
                                    upper = [ arena,  arena,  Inf,  Inf])])
        for i in 1:n_total
    ]

    # ── Collision constraints ───────────────────────────────────────────────
    pairs = kind === :keep_away ?
        [(i, j) for i in 1:n_agents for j in adversaries] :
        [(i, j) for i in 1:n_agents for j in i+1:n_agents]
    shared = [collision_avoidance([i, j]; i_offset = n_x*(i-1), j_offset = n_x*(j-1),
                                  pos_dim = 2, d_min = 2 * agent_radius)
              for (i, j) in pairs]
    game = DifferentialGame(dyn, players, tf, dt; shared_constraints = shared)

    # ── Observations ────────────────────────────────────────────────────────
    # own state | target position (zeros for an adversary) | all landmarks |
    # the other players' positions.
    lm_flat = reduce(vcat, landmarks)
    obs_fn = function (state)
        pos = [state[n_x*(a-1)+1:n_x*(a-1)+2] for a in 1:n_total]
        map(1:n_total) do i
            own    = state[n_x*(i-1)+1:n_x*i]
            tgt    = i <= n_agents ? goal : zeros(2)
            others = reduce(vcat, [pos[j] for j in 1:n_total if j != i])
            vcat(own, tgt, lm_flat, others)
        end
    end

    # ── Randomised episode starts ───────────────────────────────────────────
    # Everybody uniform in [-0.9, 0.9]^2 at rest, rejecting pairs that violate
    # a collision constraint so no episode begins infeasible. Landmarks stay
    # fixed (the cost and observations close over them).
    sample_start = function (; rng::AbstractRNG = Random.default_rng(), max_tries::Int = 10_000)
        span = 0.9 * min(arena, 1.0)
        for _ in 1:max_tries
            pos = [(2span) .* rand(rng, 2) .- span for _ in 1:n_total]
            ok = all(pairs) do (i, j)
                hypot(pos[i][1] - pos[j][1], pos[i][2] - pos[j][2]) >= 2 * agent_radius
            end
            ok && return reduce(vcat, [vcat(p, [0.0, 0.0]) for p in pos])
        end
        error("$kind: no feasible start found in $max_tries draws")
    end

    return (game = game, landmarks = landmarks, target = target, goal = goal,
            n_agents = n_agents, n_adversaries = n_adversaries,
            agent_radius = agent_radius, obs_fn = obs_fn, sample_start = sample_start)
end

"""
    keep_away(; kwargs...) -> (game, landmarks, target, goal, n_agents, n_adversaries,
                               agent_radius, obs_fn, sample_start)

The *keep-away* task of [Lowe et al.](https://arxiv.org/abs/1706.02275): `n_agents`
cooperating agents know which landmark is the target and are rewarded for reaching
it; `n_adversaries` adversaries are rewarded for reaching the target too, but do not
know which landmark it is and must infer it from the agents' motion, and so keep the
agents away. Players `1:n_agents` are the agents, the rest are adversaries.

# Keyword arguments
- `n_agents = 2`, `n_adversaries = 1`, `n_landmarks = 2`, `target = 1`
- `dt = 0.1`, `tf = 2.5`
- `accel = 3.0`, `accel_adversary = 3.0`, `damping = 0.5`
- `agent_radius = 0.15`, `control_penalty = 0.01`, `arena = 1.2`

Each agent pays its own distance to the target minus the distance of the closest
adversary; an adversary pays its distance to the target. Agent–adversary pairs have
a collision-avoidance constraint (adversaries can *block* but, unlike the particle
environment, cannot *push*: contact forces would couple the dynamics). The
adversaries' observation has the target slot zeroed.
"""
keep_away(; n_agents::Int = 2, n_adversaries::Int = 1, n_landmarks::Int = 2, target::Int = 1,
          dt::Float64 = 0.1, tf::Float64 = 2.5, accel::Float64 = 3.0,
          accel_adversary::Float64 = 3.0, damping::Float64 = 0.5,
          agent_radius::Float64 = 0.15, control_penalty::Float64 = 0.01,
          arena::Float64 = 1.2) =
    _target_adversary_task(:keep_away; n_agents, n_adversaries, n_landmarks, target, dt, tf,
                           accel, accel_adversary, damping, agent_radius, control_penalty, arena)

"""
    physical_deception(; kwargs...) -> (game, landmarks, target, goal, n_agents, n_adversaries,
                                        agent_radius, obs_fn, sample_start)

The *physical deception* task of [Lowe et al.](https://arxiv.org/abs/1706.02275):
`n_agents` cooperating agents and one adversary share `n_landmarks` landmarks, one of
which is the target. The agents are rewarded when the closest of them reaches the
target and when the adversary is far from it; the adversary is rewarded for reaching
the target but does not know which landmark it is. The agents must therefore
spread over the landmarks to deceive it.

Keyword arguments are those of [`keep_away`](@ref) with defaults
`n_agents = 2`, `n_adversaries = 1`, `n_landmarks = 2` (use `n_agents = 4` and
`n_landmarks = 4` for the paper's larger setting). Agents pay
`min_j ‖p_j − g‖ − ‖q − g‖` (a shared team cost) and the adversary `‖q − g‖`;
agent–agent pairs have a collision-avoidance constraint.
"""
physical_deception(; n_agents::Int = 2, n_adversaries::Int = 1, n_landmarks::Int = 2, target::Int = 1,
                   dt::Float64 = 0.1, tf::Float64 = 2.5, accel::Float64 = 3.0,
                   accel_adversary::Float64 = 3.0, damping::Float64 = 0.5,
                   agent_radius::Float64 = 0.15, control_penalty::Float64 = 0.01,
                   arena::Float64 = 1.2) =
    _target_adversary_task(:physical_deception; n_agents, n_adversaries, n_landmarks, target, dt, tf,
                           accel, accel_adversary, damping, agent_radius, control_penalty, arena)
