# DifferentialGamesMARL

Multi-agent reinforcement learning for the [DifferentialGames.jl](https://github.com/JuliaDifferentialGames) ecosystem.

Provides a MADDPG solver and `CommonRLInterface` environments wrapping
`DifferentialGamesBase.jl` `GameProblem`s, plus a guided-policy-search bridge
to `DifferentialGamesBaseSolvers.jl`'s local LQ solvers (FNELQ/SFNELQ).
Problem definitions (including stochastic dynamics, chance constraints, and
the POMDPs.jl adapter) live upstream in `DifferentialGamesBase.jl`; concrete
solvers live in `DifferentialGamesBaseSolvers.jl`; this package is where
multi-agent RL algorithms, training loops, and RL-environment wrappers live.

> ⚠️ **Work in progress** — migrated from exploratory work in a sibling
> `Alessandro-Work` repository; API may change before v1.0.0.

## Contents

- `MADDPGEnv` — wraps a `GameProblem` as a `CommonRLInterface.AbstractEnv`,
  with swappable constraint handling (`:barrier`/`:penalty`/`:none`),
  optional state projection, and explicit `discrete_dynamics`/`cost_state`
  flags so both continuous-time and already-discrete (`LinearDynamics`)
  games can be used as training environments without ambiguity.
- `cooperative_navigation`/`predator_prey` — benchmark multi-agent tasks.
- `Agent`/`ReplayBuffer`/`MADDPGSolver`/`train!` — the MADDPG learner.
- `LocalGameSolver`/`solve_local`/`build_guidance`/`gps_builder` — guided
  policy search: a local LQ solve (via `DifferentialGamesBaseSolvers.FNELQ`)
  used to pull MADDPG actor training towards a model-based guidance
  controller. Scoped to the deterministic case this round — see
  `src/guided_policy_search.jl`'s header for what's not ported yet
  (multiplicative-noise local solves, trust-region/outer-loop GPS).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaDifferentialGames/DifferentialGamesMARL.jl")
```

## Disclosure of Generative AI Usage

Generative AI was used in the creation of this library, migrating and
cleaning up prior exploratory work into this package. All code has been
reviewed by the author(s) for accuracy.
