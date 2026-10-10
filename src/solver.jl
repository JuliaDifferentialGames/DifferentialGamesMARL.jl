# ============================================================================
# solver.jl
#
# MADDPGSolver: the hyperparameter configuration, plus the top-level training
# loop that ties together the environment, the agents, and the replay buffer.
# ============================================================================

using CommonRLInterface: reset!, act!, terminated
using Random
using Serialization: serialize

"""
    MADDPGSolver(; kwargs...)

Hyperparameters for MADDPG training. Defaults are reasonable starting points
for cooperative navigation, not tuned optima.

NOTE: `episodes` defaults to 25,000 — the standard run length for this
project. Pass a small value explicitly (e.g. `episodes = 30`) when all you
want is an end-to-end pipeline check; at 25,000 a run takes hours.

EXPLORATION NOTE (Sep 2026): the original schedule
(`noise_decay = 0.9995`, floor 0.01) drove exploration noise to a quarter of
its initial value by episode ~2,800 and to the floor by episode ~6,000. For a
25,000-episode run that means the actors are effectively deterministic for
three quarters of training: they can refine what they already do but cannot
discover anything new. This is the most likely explanation for the fact that
cooperative navigation, predator-prey with distance shaping, and
predator-prey with a sparse team reward ALL stopped improving in the same
2,000-5,000 episode window despite being different problems -- genuine
convergence would not agree across tasks like that, but a shared schedule
would. `noise_decay = 0.9999` with a floor of 0.02 keeps meaningful
exploration alive across the whole run (noise ~0.12 at episode 5,000 and
~0.02 at 25,000, against ~0.016 and 0.01 before). Sustained exploration
matters most in exactly these settings: adversarial play needs it to keep
finding counters to a moving opponent, and sparse rewards need it to stumble
on coordinated behaviour that only pays off many steps later.

TUNING NOTE (Sep 2026): to address early plateaus and noisy learning curves,
the defaults were revised: `γ` 0.95 → 0.90 (shorter lookahead: with dense
per-step rewards over a 25-step episode, a long credit horizon amplifies
critic error and curve variance), `lr_actor` 1e-3 → 5e-4 and `update_every`
1 → 2 (fewer, gentler updates at the plateau — most of the sharp downward
spikes in earlier curves were single bad updates, not exploration). Greedy
evaluation episodes (`eval_every`) were added so that plots can show policy
quality WITHOUT exploration noise; the training curve conflates the two.

# Fields
- `episodes`     : number of training episodes
- `batch_size`   : minibatch size for each gradient update
- `buffer_size`  : replay buffer capacity
- `γ`            : discount factor
- `τ`            : soft target-update rate
- `lr_actor`     : actor learning rate
- `lr_critic`    : critic learning rate
- `hidden`       : hidden-layer width
- `noise_std`    : initial exploration noise standard deviation
- `noise_decay`  : multiplicative decay applied to noise_std each episode
- `noise_floor`  : exploration noise never falls below this. Set it well
                   above zero for long runs: an actor with no noise cannot
                   discover new behaviour, only polish existing behaviour
- `update_every` : environment steps between gradient updates
- `warmup`       : steps of pure random action before learning starts
- `eval_every`   : every `eval_every` episodes, run one GREEDY episode (no
                   exploration noise) and record its per-agent rewards. These
                   evaluations measure policy quality cleanly and are what
                   learning-curve plots should show; they do not affect
                   training (nothing is stored in the replay buffer).
- `grad_clip`    : maximum norm of any single gradient update (0 disables).
                   Long runs occasionally suffer a catastrophic single update
                   that undoes a converged policy; clipping bounds how much
                   damage one minibatch can do.
- `n_eval_episodes` : number of held-out start states averaged over at each
                   greedy evaluation. The start set is drawn ONCE, before
                   training, and replayed at every evaluation point, so two
                   evaluations differ only because the policy changed. With
                   `env.reset_fn === nothing` there is only one possible
                   start and this reduces exactly to the historical single
                   fixed-start evaluation whatever the value.
- `checkpoint_every` : serialise the agents to `checkpoint_dir` every this
                   many episodes (0 disables). Without this no trained policy
                   survives the session that produced it, and no
                   retrospective analysis is possible without a full retrain.
- `checkpoint_dir` : directory for the `.jls` checkpoints; created if needed.
- `track_best`   : keep a copy of the best-scoring policy by greedy
                   evaluation and RETURN THAT rather than whatever the last
                   episode happened to leave behind. Set `false` for
                   adversarial games, where a higher summed score is not
                   "better" -- the equilibrium is the object of interest, and
                   one side scoring well just means the other is losing.
"""
Base.@kwdef mutable struct MADDPGSolver
    episodes::Int      = 25_000
    batch_size::Int    = 256
    buffer_size::Int   = 100_000
    γ::Float64         = 0.90      # lookahead; see tuning note above
    τ::Float64         = 0.01
    lr_actor::Float64  = 5e-4      # see tuning note above
    lr_critic::Float64 = 1e-3
    hidden::Int        = 64
    noise_std::Float64 = 0.2
    noise_decay::Float64 = 0.9999    # see exploration note above
    noise_floor::Float64 = 0.02      # never stop exploring entirely
    update_every::Int  = 2         # see tuning note above
    warmup::Int        = 1000
    eval_every::Int    = 50        # greedy evaluation episode cadence
    n_eval_episodes::Int = 1       # held-out starts averaged per evaluation
    checkpoint_every::Int = 0      # serialise agents every N episodes (0 = off)
    checkpoint_dir::String = ""    # where the .jls checkpoints go
    grad_clip::Float64 = 1.0       # bound on any single update; 0 disables
    track_best::Bool   = true      # return the best-evaluated policy
    # ── Guided policy search ────────────────────────────────────────────────
    # ρ = 0 (the default) is plain MADDPG and costs nothing. ρ > 0 adds the
    # guidance penalty of Li et al. (2026) to the actor loss, pulling each
    # actor towards the control that a local LQ-game solution would apply.
    # `guidance_builder(agents, env)` -- or `(agents, env, buffer)`, if it
    # accepts the replay buffer -- is called every `guide_every` episodes
    # and must return a callable `(states, i) -> (act_dim, batch)` matrix; it
    # lives outside src/ so that the package core carries no dependency on the
    # differential-game solver.
    ρ::Float64 = 0.0
    guidance_builder = nothing
    guide_every::Int = 50
end

"""
    train!(solver, env; rng, verbose) ->
        (agents, history, agent_history, evals)

Train MADDPG agents on `env`. Returns the trained agents, a vector of
per-episode total TRAINING rewards (summed over agents and timesteps), an
`(episodes, n_agents)` matrix of per-agent training episode rewards, and
`evals = (episodes = Vector{Int}, rewards = Matrix, ...)` holding the
per-agent rewards of the periodic GREEDY evaluations (see `eval_every`). Each
evaluation is the MEAN over the held-out start set, so with
`n_eval_episodes = 1` (or no `reset_fn`) the numbers mean exactly what they
did before. `evals.eval_starts` returns the start set itself, so a later
tournament can be scored on the same conditions.

Training rewards include exploration noise and mid-update jitter; the
evaluation rewards do not, and are what learning-curve plots should show.
Destructuring fewer return values (`agents, history = train!(...)`) still
works, so older scripts are unaffected.
"""
function train!(solver::MADDPGSolver, env::MADDPGEnv;
                rng::AbstractRNG = Random.default_rng(),
                verbose::Bool = true,
                init_agents = nothing,
                frozen = Int[])
    n = num_agents(env)
    # ── Best-response support (both default OFF) ────────────────────────────
    # `init_agents` starts training from existing agents (a checkpoint) rather
    # than fresh networks. `frozen` lists agents that never learn and always
    # act greedily: a FIXED opponent. Together they train a best response to a
    # frozen policy, which is the only way to measure exploitability when no
    # exact solver exists. With the defaults nothing below changes, verified by
    # tools/regression_fingerprint.jl giving byte-identical output.
    frozen = collect(Int, frozen)
    all(i -> 1 <= i <= n, frozen) ||
        throw(ArgumentError("frozen indices $(frozen) out of range 1:$n"))
    if !isempty(frozen) && init_agents === nothing
        throw(ArgumentError("frozen agents need init_agents: freezing a freshly " *
                            "initialised network freezes a random policy"))
    end

    # Infer observation/action dimensions from one reset.
    reset!(env)
    obs0    = local_observations(env)
    obs_dim = length(obs0[1])
    act_dim = env.game.metadata.control_dims[1]   # homogeneous agents

    # Build agents and the shared replay buffer.
    agents = if init_agents === nothing
        [Agent(obs_dim, act_dim, n;
               lr_actor = solver.lr_actor,
               lr_critic = solver.lr_critic,
               hidden = solver.hidden,
               grad_clip = solver.grad_clip) for _ in 1:n]
    else
        length(init_agents) == n || throw(ArgumentError(
            "init_agents has $(length(init_agents)) agents; env has $n"))
        deepcopy(init_agents)
    end
    # A frozen agent's TARGET actor must equal its actor: the other agents'
    # TD targets use it to predict what the opponent does next, and a frozen
    # opponent does exactly what its actor says. A checkpoint's target actor
    # is a lagged copy from training, which would be the wrong opponent.
    for i in frozen
        agents[i].target_actor = deepcopy(agents[i].actor)
    end
    buffer = ReplayBuffer(solver.buffer_size)

    history = Float64[]
    agent_history = zeros(solver.episodes, n)
    eval_eps  = Int[]
    eval_rews = Vector{Float64}[]
    eval_q    = Float64[]              # critic magnitude, for diagnosis
    eval_noise = Float64[]             # exploration level, for diagnosis
    best_agents = nothing
    best_score  = -Inf
    best_ep     = 0
    noise   = solver.noise_std
    total_steps = 0
    # Wall-clock, so a learning curve can be plotted against training TIME and
    # not only against episode count. Two algorithms that cost different
    # amounts per update are not comparable on an episode axis.
    t_start   = time()
    eval_wall = Float64[]
    # Guidance state for GPS. `guidance` is rebuilt periodically from the
    # current policy; `guide_status` records what happened at the last rebuild
    # so a run that silently fell back to unguided updates is visible.
    guidance     = nothing
    guide_status = solver.ρ > 0 ? "not yet built" : "unguided"
    guide_log    = Tuple{Int,String}[]

    # ── Held-out evaluation start states ────────────────────────────────────
    # Drawn ONCE, here, and replayed at every evaluation point. Two things
    # follow. First, an evaluation measures the POLICY over a set of
    # conditions rather than one trajectory. Second, because the set is fixed,
    # a change between two evaluation points is attributable to the policy and
    # not to a lucky draw.
    #
    # With no reset_fn there is exactly one reachable start, so this collapses
    # to the historical behaviour regardless of n_eval_episodes.
    #
    # CAVEAT, on record: drawing these consumes RNG before training begins, so
    # a run is not bit-identical to one made without them. There is no
    # systematic effect, just a different sample path.
    eval_starts = if env.reset_fn === nothing
        [copy(env.game.initial_state)]
    else
        starts = Vector{Vector{Float64}}(undef, max(1, solver.n_eval_episodes))
        for k in eachindex(starts)
            reset!(env)
            starts[k] = copy(env.state)
        end
        starts
    end
    if solver.checkpoint_every > 0 && !isempty(solver.checkpoint_dir)
        isdir(solver.checkpoint_dir) || mkpath(solver.checkpoint_dir)
        serialize(joinpath(solver.checkpoint_dir, "eval_starts.jls"), eval_starts)
    end

    for ep in 1:solver.episodes
        # ── Rebuild the guidance controller ──────────────────────────────────
        # A failure here is logged and the run continues UNGUIDED rather than
        # dying, but it is recorded: a GPS run that spent most of its episodes
        # without guidance is a MADDPG run wearing a different label.
        if solver.ρ > 0 && solver.guidance_builder !== nothing &&
           (ep == 1 || ep % solver.guide_every == 1) && total_steps > solver.warmup
            try
                # A builder may also take the replay buffer, so that the
                # local model can be FITTED from the data the agents already
                # hold (guided policy search's dynamics-fitting step) rather
                # than read from the true model or from extra rollouts.
                g = applicable(solver.guidance_builder, agents, env, buffer) ?
                    solver.guidance_builder(agents, env, buffer) :
                    solver.guidance_builder(agents, env)
                if g === nothing
                    guide_status = "builder declined"
                else
                    guidance = g
                    guide_status = "ok"
                end
            catch err
                guide_status = "failed: " * string(typeof(err))
                verbose && @warn "guidance rebuild failed; continuing unguided" episode=ep exception=err
            end
            push!(guide_log, (ep, guide_status))
        end

        reset!(env)
        ep_reward = 0.0
        ep_rews   = zeros(n)

        while !terminated(env)
            obs = local_observations(env)
            # The joint state BEFORE the step, recorded only when guidance
            # needs it, so a plain MADDPG run stores exactly what it always
            # did. (It was first captured after act!, i.e. one step late; the
            # Appendix A gate caught it, because a dynamics fit on (x_{t+1},
            # u_t) -> x_{t+1} cannot recover the model.)
            st = solver.ρ > 0 ? copy(env.state) : Float64[]

            # Random actions during warmup, then noisy actor actions.
            acts = if total_steps < solver.warmup
                [2 .* rand(rng, act_dim) .- 1 for _ in 1:n]
            else
                select_actions(agents, obs; noise_std = noise, rng = rng)
            end
            # Frozen agents play their fixed policy, greedily, always, even
            # during warmup. Overwriting AFTER the draw above keeps the rng
            # stream identical to an unfrozen run.
            for i in frozen
                acts[i] = clamp.(agents[i].actor(obs[i]), -1.0, 1.0)
            end

            joint = reduce(vcat, acts)          # stacked joint control
            rews  = act!(env, joint)
            nobs  = local_observations(env)
            done  = terminated(env)

            store!(buffer, Transition(obs, acts, rews, nobs, done, st))
            ep_reward += sum(rews)
            ep_rews  .+= rews
            total_steps += 1

            # Gradient update once enough data has accumulated.
            if length(buffer) >= solver.batch_size &&
               total_steps > solver.warmup &&
               total_steps % solver.update_every == 0
                batch = sample(buffer, solver.batch_size; rng = rng)
                update_agents!(agents, batch, n, act_dim;
                               γ = solver.γ, τ = solver.τ,
                               ρ = solver.ρ, guidance = guidance,
                               frozen = frozen)
            end
        end

        push!(history, ep_reward)
        agent_history[ep, :] .= ep_rews
        noise = max(noise * solver.noise_decay, solver.noise_floor)

        # Periodic greedy evaluation: one noise-free episode, recorded but
        # NOT stored in the replay buffer, so it cannot affect training.
        if ep == 1 || ep % solver.eval_every == 0
            er = zeros(n)
            for s0 in eval_starts
                reset_to!(env, s0)
                while !terminated(env)
                    eobs  = local_observations(env)
                    eacts = select_actions(agents, eobs; noise_std = 0.0, rng = rng)
                    er  .+= act!(env, reduce(vcat, eacts))
                end
            end
            er ./= length(eval_starts)     # MEAN per-agent reward per episode
            push!(eval_eps, ep)
            push!(eval_rews, er)

            # Critic magnitude. Runaway Q values are the usual cause of a
            # converged policy suddenly collapsing, so logging |Q| lets the
            # diagnosis be checked against the curve rather than assumed.
            qmag = NaN
            if length(buffer) >= solver.batch_size
                b = sample(buffer, solver.batch_size; rng = rng)
                qmag = mean(abs, agents[1].critic(
                    vcat(joint_obs_matrix(b, n), joint_act_matrix(b, n))))
            end
            push!(eval_q, qmag)
            # Exploration level at this point. If a learning curve flattens at
            # the same episode this decays away, the cause is the schedule,
            # not convergence.
            push!(eval_noise, noise)
            push!(eval_wall, time() - t_start)

            # Keep the best policy seen, so the reported result is not
            # hostage to whatever the final episode happened to produce.
            if solver.track_best
                score = sum(er)
                if score > best_score
                    best_score  = score
                    best_agents = deepcopy(agents)
                    best_ep     = ep
                end
            end
        end

        # ── Checkpoint ───────────────────────────────────────────────────────
        # Serialising the agents is what makes a cross-play tournament (and
        # any other retrospective analysis) possible without retraining.
        if solver.checkpoint_every > 0 && !isempty(solver.checkpoint_dir) &&
           (ep % solver.checkpoint_every == 0 || ep == solver.episodes)
            isdir(solver.checkpoint_dir) || mkpath(solver.checkpoint_dir)
            serialize(joinpath(solver.checkpoint_dir,
                               "agents_ep" * lpad(ep, 6, '0') * ".jls"),
                      deepcopy(agents))
        end

        if verbose && (ep % 50 == 0 || ep == 1)
            recent = history[max(1, end-49):end]
            @info "episode $ep" mean_reward=round(sum(recent)/length(recent), digits=3) noise=round(noise, digits=3)
        end
    end

    eval_matrix = isempty(eval_rews) ? zeros(0, n) :
                  permutedims(reduce(hcat, eval_rews))

    returned = (solver.track_best && best_agents !== nothing) ? best_agents : agents
    if solver.track_best && best_agents !== nothing && verbose
        @info "Returning best-evaluated policy" best_episode=best_ep best_score=best_score
    end

    return returned, history, agent_history,
           (episodes = eval_eps, rewards = eval_matrix, q_magnitude = eval_q,
            noise = eval_noise, wall = eval_wall, best_episode = best_ep,
            best_score = best_score, final_agents = agents,
            eval_starts = eval_starts, guide_log = guide_log,
            wall_total = time() - t_start)
end
