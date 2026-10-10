# ============================================================================
# cooperative_navigation.jl
#
# The cooperative navigation benchmark from the MADDPG paper (Lowe et al.,
# arXiv 1706.02275), defined as a DifferentialGamesBase.GameProblem.
#
# Task: n agents must spread out to cover n landmarks without colliding.
#
# Modeling choices:
#   - Each agent is a 2D point mass with damped double-integrator dynamics:
#         state  x_i = [px, py, vx, vy]
#         control u_i = [fx, fy]            (forces, in [-1, 1])
#         dynamics:  p' = v,   v' = accel*u - damping*v
#     -> SeparableDynamics (each agent's state evolves independently).
#   - Every agent shares the same coverage stage cost (cooperative game):
#         sum over landmarks of (distance to the NEAREST agent)
#     plus a small control-effort regularizer. This couples all agents
#     through the cost, exactly like the paper's shared reward.
#   - Collisions are pairwise shared ProximityConstraints (d_min = 2r).
#     The MADDPGEnv wrapper converts violations into reward penalties.
# ============================================================================

"""
    cooperative_navigation(; kwargs...) -> (game, landmarks, agent_radius)

Build the cooperative navigation `GameProblem`.

# Keyword arguments
- `n_agents = 3`          : number of agents (= number of landmarks)
- `dt = 0.1`, `tf = 2.5`  : time step and horizon (25 steps, as in the paper)
- `accel = 3.0`           : force-to-acceleration gain
- `damping = 0.5`         : velocity damping coefficient
- `agent_radius = 0.15`   : physical radius (collision when centers < 2r apart)
- `control_penalty = 0.01`: weight on the control-effort cost term
- `randomize = false`     : random agent/landmark positions in [-1,1]^2
- `rng = Random.default_rng()` : RNG used when `randomize = true`

Returns a NamedTuple `(game, landmarks, agent_radius, obs_fn, sample_start)`.
The landmarks are needed for plotting and for building per-agent observations.
`sample_start(; rng)` draws a fresh, feasible initial state; pass
`reset_fn = () -> nav.sample_start(rng = rng)` to `MADDPGEnv` to randomise the
start of every episode.
"""
function cooperative_navigation(;
    n_agents::Int        = 3,
    dt::Float64          = 0.1,
    tf::Float64          = 2.5,
    accel::Float64       = 3.0,
    damping::Float64     = 0.5,
    agent_radius::Float64 = 0.15,
    control_penalty::Float64 = 0.01,
    randomize::Bool      = false,
    rng::AbstractRNG     = Random.default_rng(),
)
    n_x, n_u = 4, 2   # per-agent state and control dimensions

    # ── Initial states and landmarks ─────────────────────────────────────────
    # Each agent's state is 4D: [px, py, vx, vy]. Agents start at rest, so the
    # velocity components are zero. Landmarks are just 2D target points (they
    # are consumed by the cost function, never used as agent state).
    if randomize
        positions = [2 .* rand(rng, 2) .- 1 for _ in 1:n_agents]
        landmarks = [2 .* rand(rng, 2) .- 1 for _ in 1:n_agents]
    else
        # Agents on an outer circle, landmarks on an inner circle, offset so
        # no agent starts on "its" landmark.
        positions = [0.9 .* [cos(2π*i/n_agents), sin(2π*i/n_agents)]
                     for i in 0:n_agents-1]
        landmarks = [0.45 .* [cos(2π*i/n_agents + π/n_agents),
                              sin(2π*i/n_agents + π/n_agents)]
                     for i in 0:n_agents-1]
    end
    # Append zero initial velocity to each position -> 4D initial state.
    starts = [vcat(p, [0.0, 0.0]) for p in positions]

    # ── Dynamics (identical point mass for every agent) ──────────────────────
    # Continuous-time RHS, signature (x_i, u_i, p, t) per SeparableDynamics.
    point_mass = (xi, ui, p, t) -> [
        xi[3],
        xi[4],
        accel * ui[1] - damping * xi[3],
        accel * ui[2] - damping * xi[4],
    ]
    dyn = SeparableDynamics(fill(point_mass, n_agents),
                            fill(n_x, n_agents), fill(n_u, n_agents))

    # ── Shared coverage cost ─────────────────────────────────────────────────
    # Stage cost receives the JOINT state x and the player's OWN control u.
    coverage = function (x, u, p, t)
        c = 0.0
        for L in landmarks
            dmin = Inf
            for a in 1:n_agents
                px = x[n_x*(a-1) + 1]
                py = x[n_x*(a-1) + 2]
                d  = sqrt((px - L[1])^2 + (py - L[2])^2)
                dmin = min(dmin, d)
            end
            c += dmin
        end
        return c + control_penalty * sum(abs2, u)
    end
    terminal = (x, p) -> begin
        c = 0.0
        for L in landmarks
            dmin = Inf
            for a in 1:n_agents
                px = x[n_x*(a-1) + 1]
                py = x[n_x*(a-1) + 2]
                dmin = min(dmin, sqrt((px - L[1])^2 + (py - L[2])^2))
            end
            c += dmin
        end
        return c
    end

    # ── Players ──────────────────────────────────────────────────────────────
    players = [
        PlayerSpec(i, n_x, n_u, starts[i],
                   point_mass,
                   PlayerObjective(i,
                                   NonlinearStageCost(coverage),
                                   NonlinearTerminalCost(terminal)))
        for i in 1:n_agents
    ]

    # ── Pairwise collision-avoidance constraints ─────────────────────────────
    # Satisfied when agents i and j stay at least d_min = 2r apart.
    shared = [
        collision_avoidance([i, j];
                            i_offset = n_x*(i-1),
                            j_offset = n_x*(j-1),
                            pos_dim  = 2,
                            d_min    = 2 * agent_radius)
        for i in 1:n_agents for j in i+1:n_agents
    ]

    game = DifferentialGame(dyn, players, tf, dt; shared_constraints = shared)

    # ── Per-agent observation function ───────────────────────────────────────
    # Agent i observes: its own 4D state, then every landmark position, then
    # every OTHER agent's position. Positions are absolute (simpler and less
    # error-prone than relative coordinates for a first implementation).
    #   obs_dim = 4 + 2*n_agents (landmarks) + 2*(n_agents-1) (other agents)
    # All agents are homogeneous, so every observation has the same length —
    # the MADDPG learner relies on this.
    obs_fn = function (state)
        positions = [state[n_x*(a-1)+1 : n_x*(a-1)+2] for a in 1:n_agents]
        lm_flat   = reduce(vcat, landmarks)                    # all landmarks
        obs = Vector{Vector{Float64}}(undef, n_agents)
        for i in 1:n_agents
            own    = state[n_x*(i-1)+1 : n_x*(i-1)+n_x]        # full 4D state
            others = reduce(vcat, [positions[j] for j in 1:n_agents if j != i])
            obs[i] = vcat(own, lm_flat, others)
        end
        return obs
    end

    # ── Randomised episode starts ────────────────────────────────────────────
    # `randomize = true` above draws ONCE, inside this constructor, and that
    # single draw then becomes `game.initial_state` for every episode. This
    # sampler is different: it is called per episode by `MADDPGEnv`'s
    # `reset_fn`, so training sees a distribution of starts.
    #
    # Agents are drawn uniformly in [-1,1]^2 at rest and rejected if any pair
    # is closer than 2*agent_radius (the collision-avoidance limit), so no
    # episode begins already in violation.
    #
    # LANDMARKS ARE NOT RANDOMISED. Both the stage cost and `obs_fn` close
    # over `landmarks` at construction time, so moving them per episode would
    # require rebuilding the game. Randomising the agents alone already
    # removes the fixed-trajectory defect; randomising the landmarks as well
    # would be a larger change to the task.
    sample_start = function (; rng::AbstractRNG = Random.default_rng(),
                               max_tries::Int = 10_000)
        d_min = 2 * agent_radius
        for _ in 1:max_tries
            pos = [2 .* rand(rng, 2) .- 1 for _ in 1:n_agents]
            ok = true
            for i in 1:n_agents, j in i+1:n_agents
                if hypot(pos[i][1] - pos[j][1], pos[i][2] - pos[j][2]) < d_min
                    ok = false
                    break
                end
            end
            ok && return reduce(vcat, [vcat(p, [0.0, 0.0]) for p in pos])
        end
        error("cooperative_navigation: no feasible start found in " *
              "$max_tries draws (agent_radius = $agent_radius)")
    end

    return (game = game, landmarks = landmarks,
            agent_radius = agent_radius, obs_fn = obs_fn,
            sample_start = sample_start)
end
