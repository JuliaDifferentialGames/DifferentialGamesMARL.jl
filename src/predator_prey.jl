# ============================================================================
# predator_prey.jl
#
# Predator-prey pursuit-evasion: the non-cooperative counterpart to
# cooperative navigation, modelled on the MADDPG paper's competitive
# "simple tag" scenario (Lowe et al., arXiv 1706.02275).
#
# Task: n_predators slower predators (players 1..n_predators) chase one
# faster prey (player n_predators+1) inside a soft-bounded square arena.
#
# Modeling choices:
#   - Same damped double-integrator point mass as cooperative navigation,
#     but with a PER-AGENT acceleration gain: the prey is faster than the
#     predators (as in the paper), which is what makes cooperation among
#     the predators necessary.
#   - Rewards are adversarial and distance-based (smooth shaping rather
#     than the paper's sparse tag bonus, which can be added later):
#         predator i cost:  ||p_i - p_prey||          (chase the prey)
#         prey cost:       -min_i ||p_i - p_prey||    (flee the NEAREST
#                                                      predator)
#     plus a small control-effort regularizer for everyone. The game is
#     zero-sum in the capture distance for the nearest predator.
#   - Arena boundary: a per-player `state_bounds` PRIVATE constraint keeps
#     positions in [-arena, arena]^2 (velocities unbounded). Without this
#     the prey's optimal strategy is to flee to infinity. The MADDPGEnv
#     wrapper converts violations into barrier costs charged only to the
#     offending player.
#   - Predator-predator collisions: pairwise shared ProximityConstraints,
#     exactly as in cooperative navigation. There is deliberately NO
#     proximity constraint between predators and the prey - closing that
#     distance is the whole point.
# ============================================================================

"""
    predator_prey(; kwargs...) ->
        (game, n_predators, arena, agent_radius, obs_fn, state_projection)

Build the predator-prey pursuit-evasion `GameProblem`.

Players `1:n_predators` are predators; player `n_predators + 1` is the prey.

# Keyword arguments
- `n_predators = 3`       : number of predators (prey is one extra player)
- `dt = 0.1`, `tf = 2.5`  : time step and horizon (25 steps)
- `accel_predator = 3.0`  : predator force-to-acceleration gain
- `accel_prey = 4.0`      : prey gain (faster, as in the paper)
- `damping = 0.5`         : velocity damping coefficient
- `arena = 1.2`           : soft boundary; positions constrained to [-arena, arena]^2
- `agent_radius = 0.15`   : physical radius (predator-predator collisions)
- `control_penalty = 0.01`: weight on the control-effort cost term
- `randomize = false`     : random start positions in [-0.9, 0.9]^2
- `rng = Random.default_rng()` : RNG used when `randomize = true`
- `obstacles = ConvexObstacle[]` : convex polygon obstacles (see
  `polygon_obstacle`, `rectangle_obstacle`, `regular_polygon_obstacle`).
  Each obstacle contributes a per-agent barrier cost AND is made impassable
  through the returned `state_projection`. Obstacles are inflated by
  `agent_radius` so that point-mass agents keep their body clear.
- `obstacle_barrier_scale = 0.0` : strength of the obstacle barrier. Obstacles
  are impassable regardless, so any value > 0 is pure reward shaping — a hint
  that "near an obstacle is bad", which the agent could instead discover from
  being blocked. Defaults to OFF for that reason. At 1.0 it is strong enough
  to rival the pursuit objective itself (see `obstacle_constraints`).
- `arena_barrier = true` : smooth stay-off-the-wall cost on top of the hard
  wall. Strictly speaking this is shaping, since the wall is impassable
  anyway, but it is retained by default: it was present for all earlier runs,
  so keeping it means the new results stay comparable with those already in
  the write-up. Pass `false` for the least-guided configuration. Measured on a
  trained policy it costs the PREY about 2.7 per episode against a capture cost
  of 4.3, while the prey spends only 0.54 of 25 steps actually at the wall, so
  it is a broad gradient over the outer third of the arena rather than a
  contact penalty.
- `predator_collisions = true` : pairwise predator-predator proximity
  constraints at `d_min = 2*agent_radius`. Pass `false` to remove them; see the
  note at their construction.
- `reward_mode = :distance` : `:distance` (dense shaping — closing distance is
  rewarded directly), `:sparse` (the paper's tag objective: reward only on
  actual contact), or `:sparse_team` (contact reward shared by all predators).
  `:sparse` and `:sparse_team` embed far less human knowledge but are much
  harder to learn.
- `capture_radius = 2*agent_radius` : contact distance for the sparse modes.
- `capture_bonus = 1.0` : reward per step of contact in the sparse modes.

Returns a NamedTuple `(game, n_predators, arena, agent_radius, obs_fn,
state_projection, obstacles, obstacles_inflated, capture_radius,
sample_start)`. Pass `state_projection` to `MADDPGEnv` to make the arena
boundary a hard wall and the obstacles impassable; omit it for the soft
(barrier-only) versions. `sample_start(; rng)` draws a fresh feasible initial
state; pass `reset_fn = () -> pp.sample_start(rng = rng)` to `MADDPGEnv` to
randomise the start of every episode.
"""
function predator_prey(;
    n_predators::Int     = 3,
    dt::Float64          = 0.1,
    tf::Float64          = 2.5,
    accel_predator::Float64 = 3.0,
    accel_prey::Float64  = 4.0,
    damping::Float64     = 0.5,
    arena::Float64       = 1.2,
    agent_radius::Float64 = 0.15,
    control_penalty::Float64 = 0.01,
    randomize::Bool      = false,
    rng::AbstractRNG     = Random.default_rng(),
    obstacles::Vector{ConvexObstacle} = ConvexObstacle[],
    obstacle_barrier_scale::Float64 = 0.0,
    arena_barrier::Bool  = true,
    predator_collisions::Bool = true,
    reward_mode::Symbol  = :distance,
    capture_radius::Float64 = 0.0,
    capture_bonus::Float64  = 1.0,
)
    n_agents = n_predators + 1        # prey is the last player
    prey     = n_agents
    n_x, n_u = 4, 2                   # per-agent dims (same as coop nav)
    # Default capture radius: bodies touching.
    capture_radius = capture_radius > 0 ? capture_radius : 2 * agent_radius

    # ── Initial states ───────────────────────────────────────────────────────
    # Predators start on an outer circle; the prey starts near the centre,
    # slightly offset so the configuration is not perfectly symmetric (exact
    # symmetry makes the predators' task degenerate).
    if randomize
        positions = [1.8 .* rand(rng, 2) .- 0.9 for _ in 1:n_agents]
    else
        positions = [0.9 .* [cos(2π*i/n_predators), sin(2π*i/n_predators)]
                     for i in 0:n_predators-1]
        push!(positions, [0.15, 0.0])             # prey
    end
    starts = [vcat(p, [0.0, 0.0]) for p in positions]

    # ── Dynamics (point mass; prey has a larger acceleration gain) ───────────
    make_point_mass = accel -> (xi, ui, p, t) -> [
        xi[3],
        xi[4],
        accel * ui[1] - damping * xi[3],
        accel * ui[2] - damping * xi[4],
    ]
    accels = [fill(accel_predator, n_predators); accel_prey]
    dyn = SeparableDynamics([make_point_mass(a) for a in accels],
                            fill(n_x, n_agents), fill(n_u, n_agents))

    # ── Adversarial distance costs ───────────────────────────────────────────
    # Helper: distance from predator a's position to the prey's position,
    # read out of the JOINT state x.
    pred_prey_dist = function (x, a)
        px = x[n_x*(a-1) + 1] - x[n_x*(prey-1) + 1]
        py = x[n_x*(a-1) + 2] - x[n_x*(prey-1) + 2]
        return sqrt(px^2 + py^2)
    end

    # ── Objectives ───────────────────────────────────────────────────────────
    # Three reward modes, differing ONLY in how much human knowledge is built
    # into the objective. Costs are returned; the wrapper negates them.
    #
    #   :distance    Dense shaping. Predator i pays its own distance to the
    #                prey; the prey is paid its distance to the nearest
    #                predator. Easy to learn, but it TELLS the agents that
    #                closing distance is good -- knowledge they could have
    #                discovered. This is not the paper's objective.
    #   :sparse      The paper's "tag" objective. A predator is rewarded only
    #                in the steps where it is actually within `capture_radius`
    #                of the prey; the prey pays the same amount whenever any
    #                predator is that close. Nothing tells an agent that
    #                *approaching* helps -- it must discover that itself.
    #   :sparse_team All predators share the capture reward, so no predator is
    #                individually rewarded for being the one to make contact.
    #                Removes the individual credit that :sparse still supplies,
    #                and is the natural fix for one predator not participating.
    #
    # Sparse rewards are strictly less guided and strictly harder to learn: if
    # early exploration never produces a capture there is no gradient at all.
    # Whether it succeeds here is an empirical question worth reporting.
    @assert(reward_mode in (:distance, :sparse, :sparse_team),
            "reward_mode must be :distance, :sparse or :sparse_team")

    captured_by = (x, a) -> pred_prey_dist(x, a) < capture_radius
    any_capture = x -> any(a -> captured_by(x, a), 1:n_predators)

    if reward_mode === :distance
        make_predator_stage = i -> (x, u, p, t) ->
            pred_prey_dist(x, i) + control_penalty * sum(abs2, u)
        make_predator_terminal = i -> (x, p) -> pred_prey_dist(x, i)
        prey_stage = (x, u, p, t) ->
            -minimum(a -> pred_prey_dist(x, a), 1:n_predators) +
            control_penalty * sum(abs2, u)
        prey_terminal = (x, p) -> -minimum(a -> pred_prey_dist(x, a), 1:n_predators)
    else
        # Cost = -reward, so a capture is a NEGATIVE cost for the predator.
        pred_hit = reward_mode === :sparse_team ?
            ((x, i) -> any_capture(x)) : ((x, i) -> captured_by(x, i))
        make_predator_stage = i -> (x, u, p, t) ->
            (pred_hit(x, i) ? -capture_bonus : 0.0) + control_penalty * sum(abs2, u)
        make_predator_terminal = i -> (x, p) ->
            (pred_hit(x, i) ? -capture_bonus : 0.0)
        prey_stage = (x, u, p, t) ->
            (any_capture(x) ? capture_bonus : 0.0) + control_penalty * sum(abs2, u)
        prey_terminal = (x, p) -> (any_capture(x) ? capture_bonus : 0.0)
    end

    # ── Arena boundary (private constraint, one per player) ──────────────────
    # Positions bounded to [-arena, arena]^2; velocities left unbounded (Inf).
    boundary = i -> state_bounds(i;
        state_offset = n_x*(i-1),
        state_dim    = n_x,
        lower = [-arena, -arena, -Inf, -Inf],
        upper = [ arena,  arena,  Inf,  Inf])

    # ── Obstacle barrier costs (private constraints, per agent) ──────────────
    # Obstacles are inflated by the agent radius so a point-mass agent keeps
    # its body clear of the surface.
    inflated = ConvexObstacle[
        polygon_obstacle(o.vertices; margin = agent_radius) for o in obstacles
    ]
    # Obstacles are impassable already, so their barrier only needs to hint.
    # At full strength it rivals the pursuit signal: with a barrier weight of
    # 1.0 the prey's mean per-step reward fell from +0.64 to +0.003, i.e. the
    # obstacle cost consumed essentially the whole evasion signal. A quarter
    # weight keeps the avoidance gradient while leaving the chase dominant.
    obs_cons = obstacle_barrier_scale > 0 ?
        obstacle_constraints(inflated, n_agents; n_x = n_x,
                             barrier_scale = obstacle_barrier_scale) : Any[]
    n_obs_cons = obstacle_barrier_scale > 0 ? length(inflated) : 0

    # ── Tunnelling guard ─────────────────────────────────────────────────────
    # Blocking is a position test, not a swept test, so an obstacle narrower
    # than one step of travel could be crossed without ever being entered.
    # Rather than leave that as a comment, check it: integrate the fastest
    # agent's speed over a full episode and compare against each obstacle's
    # minimum width.
    if !isempty(inflated)
        n_steps_ep = Int(round(tf / dt))
        max_disp = 0.0
        for a in accels
            v = 0.0
            for _ in 1:n_steps_ep
                v += dt * (a * 1.0 - damping * v)
                max_disp = max(max_disp, v * dt)
            end
        end
        for (k, o) in enumerate(inflated)
            w = min_width(o)
            if w <= max_disp
                @warn("predator_prey: obstacle $k may be tunnelled through. Its " *
                      "narrowest inflated width is $(round(w, digits=3)), but the " *
                      "fastest agent travels $(round(max_disp, digits=3)) per step. " *
                      "Widen the obstacle, reduce dt, or lower the acceleration.",
                      min_width = w, max_step_displacement = max_disp)
            end
        end
    end

    """
    Private constraints for player i.

    Both of these are OPTIONAL and default to OFF, because the hard wall and
    the obstacle projection already make violation impossible. A barrier on
    top of an impassable boundary is pure reward shaping: it tells the agent
    "near here is bad" instead of letting it discover that being blocked
    wastes a step and loses ground. Turning them off removes that hint at no
    cost to feasibility.
    """
    player_cons = i -> begin
        c = Any[]
        arena_barrier && push!(c, boundary(i))
        for k in 1:n_obs_cons
            push!(c, obs_cons[(i - 1) * n_obs_cons + k])
        end
        c
    end

    # ── Players ──────────────────────────────────────────────────────────────
    players = [
        PlayerSpec(i, n_x, n_u, starts[i],
                   make_point_mass(accels[i]),
                   PlayerObjective(i,
                       NonlinearStageCost(i == prey ? prey_stage :
                                          make_predator_stage(i)),
                       NonlinearTerminalCost(i == prey ? prey_terminal :
                                             make_predator_terminal(i))),
                   player_cons(i))
        for i in 1:n_agents
    ]

    # ── Predator-predator collision avoidance (shared constraints) ───────────
    # OPTIONAL, because it is not free. `d_min` here is 2*agent_radius = 0.3,
    # which is also the capture radius, so three predators closing on one prey
    # are pushed into the range where this barrier is active, and it is charged
    # to BOTH members of every pair. Measured on a trained policy it costs each
    # predator about 1.2 per episode against a contact reward of 4.3. The
    # :sparse_team reward exists so that cutting off an escape route pays; this
    # term charges for it. Set `predator_collisions = false` to remove the tax
    # and let the predators overlap.
    shared = predator_collisions ? [
        collision_avoidance([i, j];
                            i_offset = n_x*(i-1),
                            j_offset = n_x*(j-1),
                            pos_dim  = 2,
                            d_min    = 2 * agent_radius)
        for i in 1:n_predators for j in i+1:n_predators
    ] : Any[]

    game = DifferentialGame(dyn, players, tf, dt; shared_constraints = shared)

    # ── Per-agent observation function ───────────────────────────────────────
    # Agent i observes: its own 4D state, then every OTHER agent's position,
    # then every other agent's velocity (absolute, in player-index order).
    # Velocities matter here: intercepting a moving prey (or dodging a moving
    # predator) requires knowing where everyone is headed.
    #   obs_dim = 4 + 2*(n_agents-1) + 2*(n_agents-1)   (= 16 for 3 vs 1)
    # Same length for every agent - the MADDPG learner relies on this.
    obs_fn = function (state)
        pos = [state[n_x*(a-1)+1 : n_x*(a-1)+2] for a in 1:n_agents]
        vel = [state[n_x*(a-1)+3 : n_x*(a-1)+4] for a in 1:n_agents]
        obs = Vector{Vector{Float64}}(undef, n_agents)
        for i in 1:n_agents
            own       = state[n_x*(i-1)+1 : n_x*(i-1)+n_x]
            other_pos = reduce(vcat, [pos[j] for j in 1:n_agents if j != i])
            other_vel = reduce(vcat, [vel[j] for j in 1:n_agents if j != i])
            obs[i] = vcat(own, other_pos, other_vel)
        end
        return obs
    end

    # ── Hard arena wall (state projection) ───────────────────────────────────
    # In-place projection applied by MADDPGEnv after every integration step:
    # positions are clamped to [-arena, arena]^2 and the outward velocity
    # component is zeroed on contact (an inelastic wall - otherwise an agent
    # pinned against the wall keeps its outward momentum and cannot peel
    # away). This makes leaving the arena IMPOSSIBLE ("block" the boundary,
    # per Bennet); the state_bounds constraints above are then never violated
    # and act only as a smooth stay-off-the-wall cost near the boundary.
    wall_projection = function (state)
        for a in 1:n_agents, d in 1:2
            pidx = n_x*(a-1) + d          # position component
            vidx = n_x*(a-1) + 2 + d      # matching velocity component
            if state[pidx] > arena
                state[pidx] = arena
                state[vidx] = min(state[vidx], 0.0)
            elseif state[pidx] < -arena
                state[pidx] = -arena
                state[vidx] = max(state[vidx], 0.0)
            end
        end
        return state
    end

    # Obstacles are blocked the same way: pushed out onto the nearest face
    # with the inward velocity component zeroed. Obstacles are resolved first
    # so the wall has the final say at a corner where the two could conflict.
    state_projection = compose_projections(
        obstacle_projection(inflated, n_agents; n_x = n_x),
        wall_projection)

    # ── Randomised episode starts ────────────────────────────────────────────
    # As in cooperative_navigation, `randomize = true` above draws once inside
    # the constructor; this sampler is called per episode through the
    # environment's `reset_fn`.
    #
    # Rejection conditions, each of which the fixed start also satisfied:
    #   * every agent inside 0.9*arena (never starting on the wall, where the
    #     projection would immediately alter the state);
    #   * predators mutually clear by 2*agent_radius (the collision limit);
    #   * no predator within `capture_radius` of the prey, which would pay out
    #     a capture reward at t = 0 for nothing;
    #   * nobody inside an inflated obstacle.
    #
    # Note what this changes qualitatively: the predators no longer begin on a
    # ring around a central prey. Some draws hand them a cornered prey, others
    # put all three on the same side of it. Per-episode outcomes are therefore
    # much more variable than under the fixed start, which is why evaluation
    # should average several held-out starts rather than one.
    start_span = 0.9 * arena
    sample_start = function (; rng::AbstractRNG = Random.default_rng(),
                               max_tries::Int = 10_000)
        for _ in 1:max_tries
            pos = [(2 * start_span) .* rand(rng, 2) .- start_span
                   for _ in 1:n_agents]
            ok = true
            for i in 1:n_predators, j in i+1:n_predators
                if hypot(pos[i][1] - pos[j][1], pos[i][2] - pos[j][2]) <
                   2 * agent_radius
                    ok = false
                    break
                end
            end
            if ok
                for i in 1:n_predators
                    if hypot(pos[i][1] - pos[prey][1],
                             pos[i][2] - pos[prey][2]) < capture_radius
                        ok = false
                        break
                    end
                end
            end
            if ok && !isempty(inflated)
                for p in pos, o in inflated
                    if is_inside(o, p)
                        ok = false
                        break
                    end
                end
            end
            ok && return reduce(vcat, [vcat(p, [0.0, 0.0]) for p in pos])
        end
        error("predator_prey: no feasible start found in $max_tries draws " *
              "(arena = $arena, agent_radius = $agent_radius, " *
              "capture_radius = $capture_radius, " *
              "$(length(inflated)) obstacles)")
    end

    return (game = game, n_predators = n_predators, arena = arena,
            agent_radius = agent_radius, obs_fn = obs_fn,
            state_projection = state_projection,
            obstacles = obstacles, obstacles_inflated = inflated,
            capture_radius = capture_radius,
            sample_start = sample_start)
end

"""
    default_obstacles(; arena = 1.2, agent_radius = 0.15) -> Vector{ConvexObstacle}

A reference obstacle layout for predator-prey: two blocks and a triangle,
placed off-centre and well clear of the agents' starting ring so that no agent
begins inside an obstacle. Deliberately asymmetric — a symmetric layout lets
the prey circle indefinitely, while broken symmetry forces the predators to
coordinate around cover.

```julia
pp  = predator_prey(; obstacles = default_obstacles())
env = MADDPGEnv(pp.game; obs_fn = pp.obs_fn,
                state_projection = pp.state_projection)
```
"""
function default_obstacles(; arena::Float64 = 1.2, agent_radius::Float64 = 0.15)
    # Dimensions checked numerically against three requirements:
    #   * every obstacle's narrowest INFLATED width (0.64-0.87) exceeds the
    #     fastest agent's per-step travel (0.578 for the prey), so none can be
    #     tunnelled through -- an earlier, thinner layout failed exactly this;
    #   * every start position keeps a clearance of at least 0.17, so no agent
    #     begins inside an obstacle or paying a large barrier cost at step 1;
    #   * the three obstacles are mutually disjoint and together block ~26%
    #     of the arena.
    return ConvexObstacle[
        rectangle_obstacle([-0.62, 0.05], 0.36, 0.50),
        rectangle_obstacle([ 0.45, 0.60], 0.44, 0.34),
        polygon_obstacle([[0.10, -0.82], [0.66, -0.74], [0.34, -0.34]]),
    ]
end
