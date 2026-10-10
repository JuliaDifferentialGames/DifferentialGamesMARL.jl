# ============================================================================
# obstacles.jl
#
# Convex polygon obstacles, per the advisor's request to "add polygon
# obstacles to predator prey".
#
# ── Representation ──────────────────────────────────────────────────────────
# A convex polygon is stored as an intersection of half-planes,
#
#     interior = { p : A p <= b },
#
# with the rows of `A` being UNIT OUTWARD normals. Writing
#
#     h(p) = max_i (A_i . p - b_i),
#
# gives a signed clearance: h < 0 strictly inside, h = 0 on the surface,
# h > 0 outside, and for a point outside h is (to first order) the distance to
# the polygon. Everything below is expressed through h.
#
# ── Two mechanisms, matching the arena wall precedent ───────────────────────
#   * BLOCKING (hard): after every integration step, any agent found inside an
#     obstacle is projected out onto the nearest face and its inward velocity
#     component is zeroed. Entering an obstacle is impossible by construction.
#   * BARRIER (soft): a smooth cost `w * exp(-k * h)` that is negligible far
#     away, equals `w` at the surface, and grows sharply on approach. Without
#     it the learner would only discover obstacles by colliding with them; the
#     barrier gives it a gradient to avoid them in advance.
#
# The barrier is expressed as a standard `PrivateInequality` constraint
# (g = -h <= 0), so it flows through the existing constraint machinery in
# environment.jl and is charged only to the offending agent. No changes to
# the reward or learning code were required.
#
# NOTE: obstacles are STATIC and are not added to the observation vector.
# Agents learn their location implicitly from their own position, which keeps
# obs_dim unchanged and preserves the learner's homogeneity assumption. If
# obstacles are later randomised per episode, they must be observed.
# ============================================================================

"""
    ConvexObstacle

A convex polygon obstacle in the plane.

# Fields
- `vertices` : polygon vertices in counter-clockwise order (kept for plotting)
- `A`, `b`   : half-plane form, `A p <= b`, rows of `A` unit outward normals,
               already inflated by `margin`
- `margin`   : radius by which the polygon was inflated, so that an agent
               treated as a point keeps its physical radius clear of the wall

Construct with [`polygon_obstacle`](@ref), [`rectangle_obstacle`](@ref) or
[`regular_polygon_obstacle`](@ref).
"""
struct ConvexObstacle
    vertices::Vector{Vector{Float64}}
    A::Matrix{Float64}
    b::Vector{Float64}
    margin::Float64
end

"""
    polygon_obstacle(vertices; margin = 0.0) -> ConvexObstacle

Build an obstacle from `vertices` (a vector of 2-element points). Vertices are
accepted in either winding order and normalised to counter-clockwise
internally. `margin` inflates the polygon outward — pass the agent radius so a
point-mass agent keeps its body clear.

```julia
obs = polygon_obstacle([[0.0,0.0],[0.4,0.0],[0.2,0.4]]; margin = 0.15)
```
"""
function polygon_obstacle(vertices::AbstractVector; margin::Real = 0.0)
    V = [Float64.(collect(v)) for v in vertices]
    @assert length(V) >= 3 "a polygon needs at least 3 vertices"
    @assert all(length(v) == 2 for v in V) "vertices must be 2-dimensional"

    # Signed area: negative means clockwise, so reverse to make it CCW.
    area2 = 0.0
    for i in eachindex(V)
        p, q = V[i], V[mod1(i + 1, length(V))]
        area2 += p[1] * q[2] - q[1] * p[2]
    end
    @assert abs(area2) > 1e-12 "polygon is degenerate (zero area)"
    area2 < 0 && (V = reverse(V))

    m = length(V)
    A = zeros(m, 2)
    b = zeros(m)
    for i in 1:m
        v1 = V[i]
        v2 = V[mod1(i + 1, m)]
        e  = v2 .- v1
        n  = [e[2], -e[1]]              # outward normal for CCW winding
        len = sqrt(n[1]^2 + n[2]^2)
        @assert len > 1e-12 "polygon has a repeated vertex"
        n ./= len                       # unit normal => h has distance units
        A[i, :] = n
        b[i]    = n[1] * v1[1] + n[2] * v1[2] + margin
    end
    return ConvexObstacle(V, A, b, Float64(margin))
end

"""
    rectangle_obstacle(centre, width, height; margin = 0.0) -> ConvexObstacle
"""
function rectangle_obstacle(centre::AbstractVector, width::Real, height::Real;
                            margin::Real = 0.0)
    cx, cy = Float64(centre[1]), Float64(centre[2])
    hw, hh = Float64(width) / 2, Float64(height) / 2
    return polygon_obstacle([[cx - hw, cy - hh], [cx + hw, cy - hh],
                             [cx + hw, cy + hh], [cx - hw, cy + hh]];
                            margin = margin)
end

"""
    regular_polygon_obstacle(centre, radius, n_sides; rotation = 0.0, margin = 0.0)

Regular `n_sides`-gon with circumradius `radius`, rotated by `rotation` radians.
"""
function regular_polygon_obstacle(centre::AbstractVector, radius::Real,
                                  n_sides::Int; rotation::Real = 0.0,
                                  margin::Real = 0.0)
    @assert n_sides >= 3 "need at least 3 sides"
    cx, cy = Float64(centre[1]), Float64(centre[2])
    r = Float64(radius)
    verts = [[cx + r * cos(2π * k / n_sides + rotation),
              cy + r * sin(2π * k / n_sides + rotation)] for k in 0:n_sides-1]
    return polygon_obstacle(verts; margin = margin)
end

# ─── Geometry queries ────────────────────────────────────────────────────────

"""
    clearance(obs, p) -> Float64

Signed clearance `h(p) = max_i (A_i·p - b_i)`: negative inside the obstacle,
zero on its surface, positive (and approximately the distance) outside.

`p` may be any 2-element indexable — a `Vector`, a view, or a `Tuple`. The
hot paths (projection, constraints) pass tuples deliberately, to avoid
allocating a vector per agent per obstacle per step.
"""
function clearance(obs::ConvexObstacle, p)
    h = -Inf
    for i in 1:size(obs.A, 1)
        hi = obs.A[i, 1] * p[1] + obs.A[i, 2] * p[2] - obs.b[i]
        hi > h && (h = hi)
    end
    return h
end

"True when `p` lies strictly inside the (inflated) obstacle."
is_inside(obs::ConvexObstacle, p) = clearance(obs, p) < 0.0

"""
    min_width(obs) -> Float64

Narrowest width of the obstacle over all directions. For a convex polygon the
minimum width is always attained perpendicular to one of its edges, so it is
enough to measure the extent along each face normal.

This is the quantity that governs tunnelling: an agent whose per-step
displacement exceeds `min_width` can cross the obstacle entirely between two
integration steps and never be observed inside it.
"""
function min_width(obs::ConvexObstacle)
    m = size(obs.A, 1)
    # Vertices of the INFLATED polygon: intersections of adjacent half-planes.
    verts = Vector{Vector{Float64}}(undef, m)
    for i in 1:m
        j = mod1(i - 1, m)
        M = [obs.A[i, 1] obs.A[i, 2]; obs.A[j, 1] obs.A[j, 2]]
        verts[i] = M \ [obs.b[i], obs.b[j]]
    end
    w = Inf
    for i in 1:m
        lo = Inf
        for v in verts
            proj = obs.A[i, 1] * v[1] + obs.A[i, 2] * v[2]
            proj < lo && (lo = proj)
        end
        w = min(w, obs.b[i] - lo)
    end
    return w
end

"""
    nearest_face(obs, p) -> Int

Index of the face an interior point is closest to — the face whose half-plane
is least violated, i.e. the `argmax` of `A_i·p - b_i`. Pushing out along that
face's normal is the shortest exit. `p` may be any 2-element indexable.
"""
function nearest_face(obs::ConvexObstacle, p)
    best, hbest = 1, -Inf
    for i in 1:size(obs.A, 1)
        hi = obs.A[i, 1] * p[1] + obs.A[i, 2] * p[2] - obs.b[i]
        hi > hbest && (hbest = hi; best = i)
    end
    return best
end

# ─── Blocking: state projection ──────────────────────────────────────────────

"""
    _PUSH_EPS

Skin depth: agents are pushed a hair BEYOND the surface rather than exactly
onto it. Landing exactly on the surface is not safe in floating point --- the
resulting clearance rounds to a tiny negative number and the agent still reads
as inside. Physics engines call this a skin or slop; the same trick is needed
here.
"""
const _PUSH_EPS = 1e-9

"""
    obstacle_projection(obstacles, n_agents; n_x = 4, pos_idx = 1:2, vel_idx = 3:4)

Build an in-place `state -> state` projection that pushes any agent found
inside an obstacle out onto the nearest face (plus `_PUSH_EPS`) and zeroes its
inward velocity component, leaving any tangential or outward motion untouched
(an inelastic obstacle, matching the arena wall: an agent pressed against it
can still slide along or peel away).

Pass the result to `MADDPGEnv(...; state_projection = ...)`, or compose it
with another projection via [`compose_projections`](@ref).

!!! note "Tunnelling"
    Projection is a position test, not a swept test: an agent that crosses an
    obstacle entirely within one integration step is never observed inside it.
    The requirement is therefore `min_width(obstacle) > max_speed * dt`. For
    the shipped layout the narrowest inflated obstacle is 0.64 against a
    fastest per-step travel of 0.578 (the prey), so it holds — but only after
    resizing: an earlier layout had a 0.52-wide obstacle the prey could pass
    straight through. `predator_prey` now checks this at construction and
    warns, rather than relying on this note being read.
"""
function obstacle_projection(obstacles::Vector{ConvexObstacle}, n_agents::Int;
                             n_x::Int = 4, pos_idx = 1:2, vel_idx = 3:4)
    isempty(obstacles) && return nothing
    return function (state)
        for a in 1:n_agents
            off = n_x * (a - 1)
            px = off + first(pos_idx); py = off + last(pos_idx)
            vx = off + first(vel_idx); vy = off + last(vel_idx)
            # Two sweeps: pushing out of one obstacle can, in a tight gap,
            # place the agent inside a neighbour. Two passes resolve the
            # configurations that arise here; obstacles are kept well separated.
            for _ in 1:2
                for obs in obstacles
                    p = (state[px], state[py])
                    h = clearance(obs, p)
                    h >= 0 && continue                      # outside: nothing to do
                    j = nearest_face(obs, p)
                    nx_, ny_ = obs.A[j, 1], obs.A[j, 2]
                    # h < 0, so this pushes out; the extra eps clears the
                    # surface so the result is STRICTLY outside in floating
                    # point (see _PUSH_EPS).
                    state[px] -= (h - _PUSH_EPS) * nx_
                    state[py] -= (h - _PUSH_EPS) * ny_
                    vn = state[vx] * nx_ + state[vy] * ny_
                    if vn < 0                               # still moving inward
                        state[vx] -= vn * nx_
                        state[vy] -= vn * ny_
                    end
                end
            end
        end
        return state
    end
end

"""
    compose_projections(fs...) -> Function

Chain several in-place state projections (e.g. the arena wall and the
obstacles) into one. `nothing` entries are skipped, so
`compose_projections(wall, obstacle_projection(...))` is safe when there are
no obstacles.
"""
function compose_projections(fs...)
    active = [f for f in fs if f !== nothing]
    isempty(active) && return nothing
    length(active) == 1 && return active[1]
    return function (state)
        for f in active
            f(state)
        end
        return state
    end
end

# ─── Barrier: constraints ────────────────────────────────────────────────────

"""
    obstacle_constraints(obstacles, n_agents; n_x = 4, pos_idx = 1:2,
                         barrier_scale = 1.0, barrier_sharpness = 8.0)

One `PrivateInequality` per (agent, obstacle) encoding "stay outside", as
`g = -h(p) <= 0`. Because these are PRIVATE constraints, the environment
charges the resulting barrier cost only to the agent that approaches the
obstacle, exactly as it does for the arena `state_bounds`.

# Scaling the obstacle barrier independently
`MADDPGEnv` applies a single global `barrier_weight` to every constraint, so
obstacle barriers would otherwise be as expensive as the arena wall and the
collision constraints. That is usually too strong: obstacles are already
IMPASSABLE via the state projection, so their barrier only has to hint, not
to enforce. Left at full strength it can rival the pursuit signal itself and
the learner ends up optimising obstacle avoidance rather than the chase.

`barrier_scale` multiplies the obstacle barrier cost by that factor. It is
implemented by shifting `g` by `log(barrier_scale)/barrier_sharpness`, since
`w·exp(k(g + log(s)/k)) = s·w·exp(k·g)` exactly. Pass the environment's
`barrier_sharpness` so the two agree; the default 8.0 matches `MADDPGEnv`.

Note the shift also moves the nominal `g > 0` threshold inward by
`|log(barrier_scale)|/k`, so `is_active` on these constraints reads slightly
inside the surface. Only the barrier cost is used here, so this is harmless,
but it is why the shift is documented rather than silent.
"""
function obstacle_constraints(obstacles::Vector{ConvexObstacle}, n_agents::Int;
                              n_x::Int = 4, pos_idx = 1:2,
                              barrier_scale::Real = 1.0,
                              barrier_sharpness::Real = 8.0)
    @assert barrier_scale > 0 "barrier_scale must be positive"
    shift = log(barrier_scale) / barrier_sharpness
    cons = Any[]
    for a in 1:n_agents
        off = n_x * (a - 1)
        ix, iy = off + first(pos_idx), off + last(pos_idx)
        for obs in obstacles
            # Capture obs/ix/iy per iteration; g <= 0 means "outside".
            f = (x, u, p, t) -> [-clearance(obs, (x[ix], x[iy])) + shift]
            push!(cons, PrivateInequality(a; func = f, dim = 1))
        end
    end
    return cons
end
