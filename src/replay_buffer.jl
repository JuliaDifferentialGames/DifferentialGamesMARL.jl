# ============================================================================
# replay_buffer.jl
#
# A single replay buffer shared by all agents (see architecture doc). Each
# stored transition is one environment step:
#
#   obs   :: Vector{Vector{Float64}}   per-agent observation BEFORE the step
#   acts  :: Vector{Vector{Float64}}   per-agent action taken
#   rews  :: Vector{Float64}           per-agent reward received
#   nobs  :: Vector{Vector{Float64}}   per-agent observation AFTER the step
#   done  :: Bool                      whether the episode terminated
#   state :: Vector{Float64}           JOINT state before the step
#
# `state` exists for guided policy search, which needs the joint state (not
# the per-agent observations) to build a local model of the dynamics around
# the visited point. It defaults to an empty vector, so the five-argument
# constructor used everywhere else keeps working and plain MADDPG never pays
# for it beyond one empty vector per transition.
#
# Storage is a fixed-capacity circular buffer: once full, the oldest
# transition is overwritten. `sample` draws a uniform random minibatch.
# ============================================================================

using Random

struct Transition
    obs::Vector{Vector{Float64}}
    acts::Vector{Vector{Float64}}
    rews::Vector{Float64}
    nobs::Vector{Vector{Float64}}
    done::Bool
    state::Vector{Float64}
end

"Five-argument form: no joint state recorded (plain MADDPG)."
Transition(obs, acts, rews, nobs, done) =
    Transition(obs, acts, rews, nobs, done, Float64[])

mutable struct ReplayBuffer
    capacity::Int
    data::Vector{Transition}
    pos::Int          # next write index (1-based, circular)
    full::Bool        # has the buffer wrapped around at least once?
end

ReplayBuffer(capacity::Int) =
    ReplayBuffer(capacity, Vector{Transition}(undef, capacity), 1, false)

"Current number of stored transitions."
Base.length(b::ReplayBuffer) = b.full ? b.capacity : (b.pos - 1)

"Store one transition, overwriting the oldest if the buffer is full."
function store!(b::ReplayBuffer, t::Transition)
    b.data[b.pos] = t
    b.pos += 1
    if b.pos > b.capacity
        b.pos = 1
        b.full = true
    end
    return nothing
end

"""
    sample(buffer, batch_size; rng) -> Vector{Transition}

Uniformly sample `batch_size` transitions (with replacement). Caller is
responsible for ensuring `length(buffer) >= batch_size`.
"""
function sample(b::ReplayBuffer, batch_size::Int;
                rng::AbstractRNG = Random.default_rng())
    n = length(b)
    idx = rand(rng, 1:n, batch_size)
    return b.data[idx]
end
