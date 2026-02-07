module PhosBackend

export SimContext, simulate, msle_loss, frechet_distance, nsga3_optimize, scalarize_pick

include(joinpath(@__DIR__, "Dynamics.jl"))
include(joinpath(@__DIR__, "Loss.jl"))
include(joinpath(@__DIR__, "Solver.jl"))

using .Dynamics: SimContext
using .Solver: simulate, nsga3_optimize, scalarize_pick
using .Loss: msle_loss, frechet_distance

end # module
