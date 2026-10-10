# ============================================================================
# agent.jl
#
# One MADDPG agent and the learning update.
#
# Each agent owns four networks:
#   actor, critic              — the learning networks
#   target_actor, target_critic — slow-moving copies used to form stable
#                                 TD targets (soft-updated each step)
# plus the Flux optimiser state for the actor and critic.
#
# ── The update, for agent i ──────────────────────────────────────────────
# Given a minibatch of transitions (o, a, r, o', done), where o/a/o' are
# per-agent:
#
#   Critic (TD regression):
#     a'_j   = target_actor_j(o'_j)                  for every agent j
#     y      = r_i + γ (1-done) target_critic_i([o'_all ; a'_all])
#     loss   = mean( (critic_i([o_all ; a_all]) - y)^2 )
#
#   Actor (deterministic policy gradient):
#     â_i    = actor_i(o_i)            (other agents' actions stay = batch a)
#     loss   = -mean( critic_i([o_all ; a_all with i replaced by â_i]) )
#              + ρ * mean( ||â_i - ǔ_i(x)||² )
#
# The second term is the GUIDANCE PENALTY of multi-agent guided policy search
# (Li et al. 2026): ǔ_i is a control supplied by a local LQ-game solution
# about the current nominal, and ρ weights how hard the actor is pulled
# towards it. ρ = 0 removes the term exactly, so ρ = 0 IS plain MADDPG — the
# comparison between the two algorithms runs through one code path, and any
# difference measured between them cannot be an implementation difference.
#
#   Soft target update:  θ_target ← τ θ + (1-τ) θ_target
#
# HOMOGENEITY ASSUMPTION: all agents share the same obs_dim and act_dim
# (true for cooperative navigation). This keeps the concatenation offsets
# simple. If heterogeneous agents are needed later, the offset arithmetic
# below is the thing to generalise.
# ============================================================================

using Flux
using Optimisers
using Statistics: mean
using Random

mutable struct Agent
    actor
    critic
    target_actor
    target_critic
    actor_opt
    critic_opt
end

"""
    Agent(obs_dim, act_dim, n_agents; lr_actor, lr_critic, hidden)

Construct an agent with freshly initialised actor/critic and matching target
networks. `n_agents` sets the centralised critic's input width.
"""
function Agent(obs_dim::Int, act_dim::Int, n_agents::Int;
               lr_actor::Float64 = 1e-3,
               lr_critic::Float64 = 1e-3,
               hidden::Int = 64,
               grad_clip::Float64 = 1.0)
    actor  = build_actor(obs_dim, act_dim; hidden = hidden)
    critic = build_critic(n_agents * obs_dim, n_agents * act_dim; hidden = hidden)
    # GRADIENT CLIPPING. A single unlucky minibatch can otherwise produce a
    # huge actor update that destroys a converged policy in one step -- the
    # cause of the sudden deep drops seen late in long runs. Clipping bounds
    # the damage any one update can do without changing the direction of a
    # normal update. `grad_clip <= 0` disables it.
    mkopt(lr) = grad_clip > 0 ?
        Optimisers.OptimiserChain(Optimisers.ClipGrad(grad_clip), Adam(lr)) :
        Adam(lr)
    return Agent(actor, critic,
                 deepcopy(actor), deepcopy(critic),
                 Flux.setup(mkopt(lr_actor), actor),
                 Flux.setup(mkopt(lr_critic), critic))
end

# ─── Batch assembly helpers ──────────────────────────────────────────────────
# Flux Dense layers act column-wise: a (features, batch) matrix is a batch of
# `batch` samples. We build such matrices from a vector of Transitions.

"Stack agent i's observation across the batch -> (obs_dim, B) matrix."
obs_matrix(batch, i)  = reduce(hcat, (t.obs[i]  for t in batch))
nobs_matrix(batch, i) = reduce(hcat, (t.nobs[i] for t in batch))
act_matrix(batch, i)  = reduce(hcat, (t.acts[i] for t in batch))

"Concatenate every agent's observation -> (n*obs_dim, B) matrix."
joint_obs_matrix(batch, n)  = reduce(vcat, (obs_matrix(batch, i)  for i in 1:n))
joint_nobs_matrix(batch, n) = reduce(vcat, (nobs_matrix(batch, i) for i in 1:n))
joint_act_matrix(batch, n)  = reduce(vcat, (act_matrix(batch, i)  for i in 1:n))

# ─── The MADDPG learning step ────────────────────────────────────────────────

"""
    update_agents!(agents, batch, n_agents, act_dim; γ, τ)

Perform one MADDPG gradient update for every agent, using a shared minibatch.
Mutates the agents' networks and optimiser state in place.

`frozen` lists agents whose networks are never updated (a fixed opponent, for
a best-response run). Empty by default, which is plain MADDPG.

`ρ > 0` together with a `guidance` callable turns this into the multi-agent
guided policy search update. `guidance(states, i)` must return an
`(act_dim, batch)` matrix of guidance controls for agent `i`. With `ρ = 0`
(the default) the guidance term is not evaluated at all.
"""
function update_agents!(agents::Vector{Agent}, batch, n_agents::Int, act_dim::Int;
                        γ::Float64 = 0.95, τ::Float64 = 0.01,
                        ρ::Float64 = 0.0, guidance = nothing,
                        frozen = ())
    B = length(batch)

    # ---- Guidance controls, if this is a guided (GPS) update ----
    # `guidance(states, i)` returns an (act_dim, B) matrix: the control the
    # local LQ-game solution would apply for agent i at each batch state.
    # Built once per update and shared across the per-agent loop.
    guided = ρ > 0 && guidance !== nothing
    U_guide = if guided
        states = [t.state for t in batch]
        any(isempty, states) && error(
            "update_agents!: ρ = $ρ with guidance, but some transitions carry " *
            "no joint state. The buffer must be filled by a solver that records " *
            "it (see Transition.state); a buffer from a plain MADDPG run cannot " *
            "be reused for a guided update.")
        [guidance(states, i) for i in 1:n_agents]
    else
        nothing
    end

    # Joint matrices shared across all agents' updates.
    O_all  = joint_obs_matrix(batch, n_agents)   # (n*obs_dim, B)
    O2_all = joint_nobs_matrix(batch, n_agents)  # (n*obs_dim, B)
    A_all  = joint_act_matrix(batch, n_agents)   # (n*act_dim, B)

    # Target actions a'_j = target_actor_j(o'_j), concatenated -> (n*act_dim, B)
    A2_all = reduce(vcat,
        (agents[j].target_actor(nobs_matrix(batch, j)) for j in 1:n_agents))

    for i in 1:n_agents
        # A FROZEN agent is a fixed opponent: no critic step, no actor step.
        # Its actions still enter every other agent's critic through the batch
        # and through its target actor, which is what a best response needs.
        i in frozen && continue
        ag = agents[i]

        # rewards_i and done as 1×B rows
        r_i  = reshape([t.rews[i]          for t in batch], 1, B)
        done = reshape([t.done ? 1.0 : 0.0 for t in batch], 1, B)

        # ---- Critic update (TD regression) ----
        q_target = ag.target_critic(vcat(O2_all, A2_all))      # (1, B)
        y = r_i .+ γ .* (1.0 .- done) .* q_target              # (1, B), fixed
        critic_in = vcat(O_all, A_all)                         # (n*(o+a), B)

        gs_c = gradient(ag.critic) do c
            mean((c(critic_in) .- y) .^ 2)
        end
        Flux.update!(ag.critic_opt, ag.critic, gs_c[1])

        # ---- Actor update (deterministic policy gradient) ----
        # Replace agent i's action block with â_i = actor_i(o_i); keep the
        # other agents' actions fixed at their batch values.
        O_i   = obs_matrix(batch, i)                           # (obs_dim, B)
        lo    = (i - 1) * act_dim                              # rows before i
        preA  = A_all[1:lo, :]                                 # fixed
        postA = A_all[lo + act_dim + 1 : end, :]               # fixed

        Ǔ_i = guided ? U_guide[i] : nothing                    # (act_dim, B)

        gs_a = gradient(ag.actor) do a
            âi    = a(O_i)                                     # (act_dim, B)
            A_hat = vcat(preA, âi, postA)                      # (n*act_dim, B)
            q = -mean(ag.critic(vcat(O_all, A_hat)))           # maximise Q
            # ρ = 0 short-circuits before any guidance arithmetic, so the
            # unguided gradient is bit-identical to the pre-GPS code.
            guided ? q + ρ * mean(sum((âi .- Ǔ_i) .^ 2, dims = 1)) : q
        end
        Flux.update!(ag.actor_opt, ag.actor, gs_a[1])
    end

    # ---- Soft target updates (after all gradient steps) ----
    for (i, ag) in enumerate(agents)
        i in frozen && continue
        soft_update!(ag.target_actor,  ag.actor,  τ)
        soft_update!(ag.target_critic, ag.critic, τ)
    end
    return nothing
end

"""
    soft_update!(target, source, τ)

In-place Polyak averaging: target ← τ·source + (1-τ)·target, over every
trainable parameter array. Uses `Flux.trainables` (not `destructure`, which
silently downcasts Float64 → Float32).
"""
function soft_update!(target, source, τ::Float64)
    for (pt, ps) in zip(Flux.trainables(target), Flux.trainables(source))
        pt .= τ .* ps .+ (1 - τ) .* pt
    end
    return nothing
end

"""
    select_actions(agents, obs_list; noise_std, rng) -> Vector{Vector{Float64}}

Greedy actor actions plus exploration noise, clamped to [-1, 1]. `obs_list` is
the per-agent observation vector from `local_observations(env)`.
"""
function select_actions(agents::Vector{Agent}, obs_list;
                        noise_std::Float64 = 0.1,
                        rng::AbstractRNG = Random.default_rng())
    acts = Vector{Vector{Float64}}(undef, length(agents))
    for i in eachindex(agents)
        a = agents[i].actor(obs_list[i])
        # Draw noise only when actually exploring: greedy calls (noise_std=0)
        # must consume NOTHING from the rng, so that adding evaluation
        # episodes does not perturb the training stream and old seeded runs
        # stay reproducible.
        if noise_std > 0
            a = a .+ noise_std .* randn(rng, length(a))
        end
        acts[i] = clamp.(a, -1.0, 1.0)
    end
    return acts
end
