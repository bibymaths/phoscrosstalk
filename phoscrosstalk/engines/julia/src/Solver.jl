module Solver

using DifferentialEquations
using SciMLSensitivity
using LinearAlgebra
using Random
using Metaheuristics

import ..Dynamics: SimContext, build_problem, mech_code
import ..Loss: msle_loss

export simulate, nsga3_optimize, scalarize_pick

function simulate(ctx::SimContext,
                  t::Vector{Float64},
                  P_scaled::Matrix{Float64},
                  A0_full::Matrix{Float64},
                  θ::Vector{Float64};
                  mechanism::String="dist",
                  stiff_solver::Symbol=:Rodas5P,
                  reltol::Float64=1e-7,
                  abstol::Float64=1e-9,
                  saveat::Union{Nothing,Vector{Float64}}=nothing)

    K, M, N = ctx.K, ctx.M, ctx.N
    T = length(t)
    @assert size(P_scaled,1) == N
    @assert size(A0_full,1) == K

    # initial conditions: S0 from 0, A0 from col 1, Kdyn0 from 0, p0 from P_scaled[:,1]
    u0 = zeros(Float64, 2K + M + N)
    @views u0[K+1:2K] .= A0_full[:,1]
    @views u0[2K+M+1:end] .= P_scaled[:,1]

    tspan = (t[1], t[end])
    mech = mech_code(mechanism)
    prob, t_eval = build_problem(ctx, θ, u0, tspan, t, mech)

    alg = stiff_solver == :QNDF ? QNDF() : Rodas5P()

    sol = solve(prob, alg; reltol=reltol, abstol=abstol,
                saveat=(saveat === nothing ? t_eval : saveat),
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(false)))

    U = Array(sol)  # (state, time)

    S_sim    = @view U[1:K, :]
    A_sim    = @view U[K+1:2K, :]
    Kdyn_sim = @view U[2K+1:2K+M, :]
    P_sim    = @view U[2K+M+1:2K+M+N, :]

    return Array(P_sim), Array(A_sim), Array(S_sim), Array(Kdyn_sim)
end

# Simple scalarization to pick “best” point from Pareto set
function scalarize_pick(F::Matrix{Float64})
    # normalize each objective to [0,1] then sum
    mins = mapslices(minimum, F; dims=1)
    ptp  = mapslices(x->maximum(x)-minimum(x)+1e-12, F; dims=1)
    Fn = (F .- mins) ./ ptp
    J = vec(sum(Fn; dims=2))
    best = argmin(J)
    return best, J
end

function nsga3_optimize(ctx::SimContext,
                        t::Vector{Float64},
                        P_scaled::Matrix{Float64},
                        A0_full::Matrix{Float64},
                        xl::Vector{Float64},
                        xu::Vector{Float64};
                        mechanism::String="dist",
                        pop::Int=128,
                        gen::Int=200,
                        multistart::Int=4,
                        seed::Int=1)

    rng = MersenneTwister(seed)

    dim = length(xl)
    bounds = hcat(xl, xu)

    # objectives: f1 site MSLE, f2 protein MSLE (optional placeholder), f3 complexity (L2 on θ)
    function obj(θvec)
        θ = Vector{Float64}(θvec)
        P_sim, A_sim, _, _ = simulate(ctx, t, P_scaled, A0_full, θ; mechanism=mechanism)

        f1 = msle_loss(P_scaled, P_sim)
        # If you have protein abundance targets, pass them from Python; for now 0 if absent:
        f2 = 0.0
        f3 = sum(abs2, θ) / length(θ)

        return [f1, f2, f3]
    end

    bestF = nothing
    bestX = nothing

    for s in 1:multistart
        # NSGA3 in Metaheuristics.jl
        res = optimize(obj, bounds, NSGA3(populationSize=pop), ngen=gen, rng=MersenneTwister(rand(rng, UInt)))
        F = hcat(res.f...)' |> Matrix{Float64}
        X = hcat(res.x...)' |> Matrix{Float64}

        if bestF === nothing
            bestF, bestX = F, X
        else
            bestF = vcat(bestF, F)
            bestX = vcat(bestX, X)
        end
    end

    return bestF, bestX
end

end # module
