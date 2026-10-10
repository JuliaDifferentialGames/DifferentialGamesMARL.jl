# ============================================================================
# DifferentialGamesMARL.jl — main module
#
# Multi-agent reinforcement learning for the DifferentialGames.jl ecosystem:
# a MADDPG solver with CommonRLInterface environments wrapping
# DifferentialGamesBase.jl GameProblems, plus a guided-policy-search bridge
# to DifferentialGamesBaseSolvers.jl's local LQ solvers (FNELQ/SFNELQ).
#
# Migrated and cleaned up from Alessandro-Work's MADDPG.jl package and
# sgnep/{GuidedPolicySearch.jl,gps_guidance.jl}. The stochastic-LQ problem
# type and solver (formerly sgnep/SGNEP.jl + SFNELQ.jl) now live upstream in
# DifferentialGamesBase.jl / DifferentialGamesBaseSolvers.jl — see those
# packages' docs/src/api/stochastic.md.
#
# Files (flat layout, one responsibility per file):
#   environment.jl             GameProblem -> CommonRLInterface wrapper
#   obstacles.jl                polygon obstacle geometry
#   cooperative_navigation.jl  benchmark task as a GameProblem
#   predator_prey.jl           benchmark task as a GameProblem
#   networks.jl                actor and critic network builders (Flux)
#   replay_buffer.jl           shared experience replay
#   agent.jl                   per-agent networks + the MADDPG update step
#   solver.jl                  MADDPGSolver config + training loop
#   guided_policy_search.jl    local LQ step (SinglePlayerLQR/DeterministicLQNash)
#   gps_guidance.jl            bridges the local step into MADDPGSolver guidance
# ============================================================================

module DifferentialGamesMARL

using LinearAlgebra
using Random
using Statistics

import CommonRLInterface
using CommonRLInterface: AbstractEnv

# Import only what we use from DifferentialGamesBase. Do NOT blanket-`using`
# it — it exports `observe`, which collides with CommonRLInterface.observe.
using DifferentialGamesBase:
    GameProblem,
    DifferentialGame,
    PlayerSpec,
    PlayerObjective,
    NonlinearStageCost,
    NonlinearTerminalCost,
    SeparableDynamics,
    evaluate_dynamics,
    evaluate_stage_cost,
    evaluate_constraint,
    is_inequality,
    get_player,
    get_players,
    collision_avoidance,
    state_bounds,
    PrivateInequality
import DifferentialGamesBase

# Environment + tasks
export MADDPGEnv, cooperative_navigation, predator_prey, reset_to!

# Polygon obstacles
export ConvexObstacle, polygon_obstacle, rectangle_obstacle,
       regular_polygon_obstacle, clearance, is_inside, min_width,
       obstacle_projection, obstacle_constraints, compose_projections,
       default_obstacles
export num_agents, joint_control_dim, local_observations

# Learner
export Agent, ReplayBuffer, Transition, MADDPGSolver, store!
export build_actor, build_critic, train!, select_actions, update_agents!

# Guided policy search
export LocalGameSolver, SinglePlayerLQR, DeterministicLQNash, StochasticLQNash,
       LocalModel, LocalPolicies, GPSConfig,
       local_model, solve_local, local_exploitability,
       zero_control_nominal, affine_term_magnitude
export EulerDiscretization, euler_da, greedy_nominal, GuidanceLaw,
       build_guidance, gps_builder, nearest_step, guidance_diagnostics

include("environment.jl")
include("obstacles.jl")
include("cooperative_navigation.jl")
include("predator_prey.jl")
include("networks.jl")
include("replay_buffer.jl")
include("agent.jl")
include("solver.jl")
include("guided_policy_search.jl")
include("gps_guidance.jl")

end # module
