using Test
using DifferentialGamesMARL
using CommonRLInterface: reset!, act!, observe, terminated, actions

@testset "DifferentialGamesMARL.jl" begin

    @testset "cooperative_navigation builds" begin
        nav = cooperative_navigation()
        @test nav.game.n_players == 3
        @test length(nav.landmarks) == 3
        @test length(nav.game.initial_state) == 12          # 3 agents x 4 states
        @test length(nav.game.shared_constraints) == 3      # 3 agent pairs
        @test nav.obs_fn !== nothing
    end

    @testset "environment wrapper" begin
        nav = cooperative_navigation()
        env = MADDPGEnv(nav.game; obs_fn = nav.obs_fn)

        @test num_agents(env) == 3
        @test joint_control_dim(env) == 6
        @test length(actions(env)) == 6

        reset!(env)
        @test observe(env) == nav.game.initial_state
        @test !terminated(env)

        r = act!(env, zeros(6))
        @test r isa Vector{Float64}
        @test length(r) == 3
        @test all(r .≈ r[1])          # shared cost, zero control => identical

        # Episode terminates after exactly N steps.
        reset!(env)
        steps = 0
        while !terminated(env)
            act!(env, 2 .* rand(6) .- 1)
            steps += 1
            @test steps <= 1000
        end
        @test steps == nav.game.time_horizon.N
    end

    @testset "local observations" begin
        nav = cooperative_navigation()
        env = MADDPGEnv(nav.game; obs_fn = nav.obs_fn)
        reset!(env)
        obs = local_observations(env)
        @test length(obs) == 3
        # obs_dim = 4 (own) + 2*3 (landmarks) + 2*2 (other agents) = 14
        @test all(length(o) == 14 for o in obs)
    end

    @testset "constraint modes" begin
        nav = cooperative_navigation()

        # Put two agents on top of each other -> collision constraint active.
        function collide_reward(mode)
            env = MADDPGEnv(nav.game; obs_fn = nav.obs_fn, constraint_mode = mode)
            reset!(env)
            env.state[5:6] .= env.state[1:2]   # agent 2 onto agent 1
            return act!(env, zeros(6))
        end

        r_barrier = collide_reward(:barrier)
        r_penalty = collide_reward(:penalty)
        r_none    = collide_reward(:none)

        # With :none the collision is unpunished; barrier and penalty both
        # make the involved agents worse off than :none.
        @test r_barrier[1] < r_none[1]
        @test r_penalty[1] < r_none[1]
    end

    @testset "predator_prey builds" begin
        pp = predator_prey()
        @test pp.game.n_players == 4                       # 3 predators + prey
        @test length(pp.game.initial_state) == 16          # 4 agents x 4 states
        @test length(pp.game.shared_constraints) == 3      # predator pairs
        @test length(pp.game.private_constraints) == 4     # one boundary each

        env = MADDPGEnv(pp.game; obs_fn = pp.obs_fn)
        @test num_agents(env) == 4
        @test joint_control_dim(env) == 8
        reset!(env)
        obs = local_observations(env)
        # obs_dim = 4 (own) + 2*3 (other positions) + 2*3 (other velocities)
        @test all(length(o) == 16 for o in obs)

        # Episode terminates after exactly N steps.
        steps = 0
        while !terminated(env)
            act!(env, 2 .* rand(8) .- 1)
            steps += 1
            @test steps <= 1000
        end
        @test steps == pp.game.time_horizon.N
    end

    @testset "predator_prey adversarial rewards" begin
        pp = predator_prey()
        prey = pp.game.n_players

        # Rewards with the prey at its default start vs teleported right next
        # to predator 1. Zero control, so the reward differences come from
        # the distance costs alone.
        function rewards_with_prey_at(pos)
            env = MADDPGEnv(pp.game; obs_fn = pp.obs_fn)
            reset!(env)
            env.state[4*(prey-1)+1 : 4*(prey-1)+2] .= pos
            return act!(env, zeros(8))
        end

        pred1_pos = pp.game.initial_state[1:2]
        r_far  = rewards_with_prey_at([0.15, 0.0])         # default start
        r_near = rewards_with_prey_at(pred1_pos .+ 0.05)   # beside predator 1

        # Adversarial structure: closing the gap helps predator 1, hurts prey.
        @test r_near[1] > r_far[1]
        @test r_near[prey] < r_far[prey]

        # Arena boundary: with the prey outside the arena, the barrier makes
        # the prey much worse off than with constraints ignored, while the
        # predators (well inside) are barely affected.
        function rewards_outside(mode)
            env = MADDPGEnv(pp.game; obs_fn = pp.obs_fn, constraint_mode = mode)
            reset!(env)
            env.state[4*(prey-1)+1] = pp.arena + 0.8       # beyond the wall
            return act!(env, zeros(8))
        end
        r_none    = rewards_outside(:none)
        r_barrier = rewards_outside(:barrier)
        @test r_none[prey] - r_barrier[prey] > 5.0         # prey pays dearly
        @test abs(r_none[1] - r_barrier[1]) < 1.0          # predator ~unaffected

        # barrier_cap bounds the cost of deep violations.
        function rewards_outside_capped(cap)
            env = MADDPGEnv(pp.game; obs_fn = pp.obs_fn, barrier_cap = cap)
            reset!(env)
            env.state[4*(prey-1)+1] = pp.arena + 0.8
            return act!(env, zeros(8))
        end
        r_capped = rewards_outside_capped(25.0)
        @test r_none[prey] - r_capped[prey] <= 25.0 + 1e-9  # cost capped
        @test r_capped[prey] < r_far[prey]                  # but still punished
    end

    @testset "predator_prey hard wall" begin
        pp = predator_prey()
        env = MADDPGEnv(pp.game; obs_fn = pp.obs_fn,
                        state_projection = pp.state_projection)
        prey = pp.game.n_players

        # Drive the prey straight at the +x wall with outward velocity and
        # full outward force: it must stop AT the wall, with its outward
        # velocity component zeroed on contact.
        reset!(env)
        env.state[4*(prey-1)+1] = pp.arena - 0.01   # just inside the wall
        env.state[4*(prey-1)+3] = 5.0               # large outward velocity
        act!(env, vcat(zeros(6), [1.0, 0.0]))
        @test env.state[4*(prey-1)+1] <= pp.arena + 1e-12
        @test env.state[4*(prey-1)+3] <= 0.0        # outward momentum killed

        # A whole episode of maximal outward force: nobody ever leaves.
        reset!(env)
        while !terminated(env)
            act!(env, ones(8))
            worst = maximum(abs(env.state[4*(a-1)+d]) for a in 1:4, d in 1:2)
            @test worst <= pp.arena + 1e-12
        end
    end

    @testset "polygon obstacle geometry" begin
        sq = polygon_obstacle([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
        @test clearance(sq, [0.5, 0.5]) ≈ -0.5        # inside
        @test clearance(sq, [2.0, 0.5]) ≈  1.0        # outside, distance-like
        @test abs(clearance(sq, [1.0, 0.5])) < 1e-12  # on the surface
        @test is_inside(sq, [0.5, 0.5])
        @test !is_inside(sq, [2.0, 0.5])

        # Winding order must not matter: clockwise input is normalised.
        cw = polygon_obstacle([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
        for p in ([0.5,0.5], [2.0,0.5], [-1.0,-1.0])
            @test clearance(sq, p) ≈ clearance(cw, p)
        end

        # margin inflates outward
        infl = polygon_obstacle([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]; margin = 0.15)
        @test clearance(infl, [1.1, 0.5]) ≈ -0.05
        @test is_inside(infl, [1.1, 0.5])
        @test !is_inside(sq,  [1.1, 0.5])

        # convenience constructors
        r = rectangle_obstacle([0.0, 0.0], 2.0, 4.0)
        @test is_inside(r, [0.0, 1.9]) && !is_inside(r, [0.0, 2.1])
        hexa = regular_polygon_obstacle([0.0, 0.0], 1.0, 6)
        @test is_inside(hexa, [0.0, 0.0])
        @test !is_inside(hexa, [2.0, 0.0])
        @test_throws AssertionError polygon_obstacle([[0.0,0.0],[1.0,0.0]])

        # min_width: for an axis-aligned rectangle it is the shorter side,
        # and inflation widens it by 2*margin on each axis.
        @test min_width(rectangle_obstacle([0.0,0.0], 1.0, 0.4)) ≈ 0.4
        @test min_width(rectangle_obstacle([0.0,0.0], 1.0, 0.4; margin = 0.1)) ≈ 0.6
    end

    @testset "default obstacles cannot be tunnelled" begin
        # An agent crossing an obstacle within one integration step would never
        # be seen inside it, so every obstacle must be wider than the fastest
        # agent's per-step travel. This caught a real defect: an earlier
        # layout had a 0.52-wide obstacle against a prey step of 0.578.
        pp = predator_prey(; obstacles = default_obstacles())
        dt, damping, tf = 0.1, 0.5, 2.5
        max_disp = 0.0
        for a in (3.0, 4.0)                     # predator and prey gains
            v = 0.0
            for _ in 1:Int(round(tf / dt))
                v += dt * (a - damping * v)
                max_disp = max(max_disp, v * dt)
            end
        end
        for o in pp.obstacles_inflated
            @test min_width(o) > max_disp
        end
    end

    @testset "obstacle projection blocks entry" begin
        obs = ConvexObstacle[rectangle_obstacle([0.0, 0.0], 0.4, 0.4)]
        proj = obstacle_projection(obs, 1; n_x = 4)

        # The point (0.05, 0.02) is nearest the +x face, so that is the exit
        # normal. A velocity of -x is therefore INWARD and must be zeroed,
        # while the tangential (y) component survives untouched.
        s = [0.05, 0.02, -1.0, 0.5]
        proj(s)
        @test clearance(obs[1], s[1:2]) > 0.0   # strictly outside, not merely on
        @test isapprox(s[3], 0.0; atol = 1e-12) # inward component removed
        @test s[4] ≈ 0.5                        # tangential motion preserved

        # Same position, but already travelling outward: velocity untouched.
        s2 = [0.05, 0.02, 1.0, 0.0]
        proj(s2)
        @test clearance(obs[1], s2[1:2]) > 0.0
        @test s2[3] ≈ 1.0

        # An agent already outside is untouched.
        s3 = [1.0, 1.0, -1.0, -1.0]
        before = copy(s3)
        proj(s3)
        @test s3 == before

        @test obstacle_projection(ConvexObstacle[], 1) === nothing
        @test compose_projections(nothing, nothing) === nothing
    end

    @testset "predator_prey with obstacles" begin
        obstacles = default_obstacles()
        pp = predator_prey(; obstacles = obstacles)
        n  = pp.game.n_players

        # By default: the arena barrier is kept (one per agent, for continuity
        # with earlier runs), while the obstacle barrier is off, since the
        # obstacles are impassable and a barrier on top would be pure shaping.
        @test length(pp.game.private_constraints) == n

        # Both can be switched on: one boundary plus one per obstacle.
        pp_b = predator_prey(; obstacles = obstacles,
                              arena_barrier = true, obstacle_barrier_scale = 0.25)
        @test length(pp_b.game.private_constraints) == n * (1 + length(obstacles))

        # ...and both can be switched off, for the least-guided setup.
        pp_n = predator_prey(; obstacles = obstacles, arena_barrier = false)
        @test isempty(pp_n.game.private_constraints)

        env = MADDPGEnv(pp.game; obs_fn = pp.obs_fn,
                        state_projection = pp.state_projection)
        reset!(env)

        # No agent starts inside an obstacle.
        for a in 1:n, o in pp.obstacles_inflated
            @test !is_inside(o, env.state[4(a-1)+1 : 4(a-1)+2])
        end

        # Obstacles must be impassable: drive everyone at full force in every
        # diagonal direction for whole episodes and assert nobody is ever
        # inside one. This is the test that exercises the projection on real
        # trajectories, rather than on hand-picked points.
        for u in ([1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0])
            reset!(env)
            while !terminated(env)
                act!(env, repeat(u, n))
                for a in 1:n, o in pp.obstacles_inflated
                    @test !is_inside(o, env.state[4(a-1)+1 : 4(a-1)+2])
                end
            end
        end

        # When the barrier IS enabled, being near an obstacle must be worse.
        function reward_at(pos, mode)
            e = MADDPGEnv(pp_b.game; obs_fn = pp_b.obs_fn, constraint_mode = mode)
            reset!(e)
            e.state[1:2] .= pos
            return act!(e, zeros(2n))
        end
        near = [-0.62, 0.05]     # centre of the first obstacle
        far  = [ 1.10, 1.10]     # empty corner
        @test reward_at(near, :barrier)[1] < reward_at(far, :barrier)[1]
        # ...and that penalty must come from the obstacle, not the geometry.
        @test reward_at(near, :none)[1] > reward_at(near, :barrier)[1]
    end

    @testset "reward modes carry different amounts of guidance" begin
        n = 4
        # Place predator 1 on top of the prey, and far from it, with zero
        # control, and compare what each reward mode reports.
        # constraint_mode = :none is essential here: this test measures the
        # OBJECTIVE, and the arena barrier is a position-dependent cost that
        # would otherwise be folded into every reward. At [1.1, 1.1], for
        # instance, the wall barrier alone contributes 2*exp(8*-0.1) = 0.90,
        # which would swamp the capture term being tested.
        function rewards(mode, pred1_pos)
            pp = predator_prey(; reward_mode = mode)
            e  = MADDPGEnv(pp.game; obs_fn = pp.obs_fn, constraint_mode = :none)
            reset!(e)
            e.state[1:2] .= pred1_pos
            return act!(e, zeros(2n))
        end
        prey_pos = [0.15, 0.0]
        touching = prey_pos .+ [0.05, 0.0]     # inside 2*agent_radius = 0.3
        distant  = [1.1, 1.1]

        # :distance rewards ANY reduction in distance -- a gradient everywhere.
        d_near = rewards(:distance, touching)[1]
        d_mid  = rewards(:distance, [0.6, 0.0])[1]
        d_far  = rewards(:distance, distant)[1]
        @test d_near > d_mid > d_far           # strictly informative everywhere

        # :sparse pays only on contact, so mid and far are indistinguishable:
        # that is precisely the guidance being withheld.
        s_near = rewards(:sparse, touching)[1]
        s_mid  = rewards(:sparse, [0.6, 0.0])[1]
        s_far  = rewards(:sparse, distant)[1]
        @test s_near > s_mid
        @test isapprox(s_mid, s_far; atol = 1e-12)

        # :sparse_team pays every predator when any one makes contact.
        t = rewards(:sparse_team, touching)
        @test t[1] ≈ t[2] ≈ t[3]               # shared, no individual credit
        @test t[1] > rewards(:sparse_team, distant)[1]

        @test_throws AssertionError predator_prey(; reward_mode = :nonsense)
    end

    @testset "replay buffer" begin
        b = ReplayBuffer(3)
        @test length(b) == 0
        mk(v) = Transition([[v]], [[v]], [v], [[v]], false)
        store!(b, mk(1.0)); store!(b, mk(2.0))
        @test length(b) == 2
        store!(b, mk(3.0)); store!(b, mk(4.0))   # overwrites oldest
        @test length(b) == 3                     # capped at capacity
        batch = DifferentialGamesMARL.sample(b, 5)
        @test length(batch) == 5
    end

    @testset "networks + one update step" begin
        nav = cooperative_navigation()
        env = MADDPGEnv(nav.game; obs_fn = nav.obs_fn)
        n = num_agents(env)
        reset!(env)
        obs_dim = length(local_observations(env)[1])
        act_dim = 2

        agents = [Agent(obs_dim, act_dim, n; hidden = 16) for _ in 1:n]

        # actor maps obs_dim -> act_dim, bounded to [-1,1]
        a_out = agents[1].actor(local_observations(env)[1])
        @test length(a_out) == act_dim
        @test all(-1 .<= a_out .<= 1)

        # Build a tiny batch and run one update; it should not error and
        # should change parameters.
        buf = ReplayBuffer(100)
        for _ in 1:10
            obs = local_observations(env)
            acts = [2 .* rand(act_dim) .- 1 for _ in 1:n]
            rews = act!(env, reduce(vcat, acts))
            nobs = local_observations(env)
            store!(buf, Transition(obs, acts, rews, nobs, terminated(env)))
        end
        before = copy(DifferentialGamesMARL.Flux.trainables(agents[1].actor)[1])
        batch = DifferentialGamesMARL.sample(buf, 8)
        update_agents!(agents, batch, n, act_dim)
        after = DifferentialGamesMARL.Flux.trainables(agents[1].actor)[1]
        @test before != after          # learning actually moved the weights
    end

end
