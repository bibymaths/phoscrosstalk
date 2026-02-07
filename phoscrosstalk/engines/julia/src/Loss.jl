module Loss

using LinearAlgebra

export msle_loss, frechet_distance

@inline function msle_loss(ytrue::AbstractMatrix, ypred::AbstractMatrix; eps::Float64=1e-8)
    # mean squared log error over all entries
    @assert size(ytrue) == size(ypred)
    s = 0.0
    n = length(ytrue)
    @inbounds for i in eachindex(ytrue)
        a = log(ypred[i] + eps)
        b = log(ytrue[i] + eps)
        d = a - b
        s += d*d
    end
    return s / n
end

function frechet_distance(true_coords::AbstractMatrix{<:Real}, pred_coords::AbstractMatrix{<:Real})
    # Discrete Fréchet distance, O(n*m) DP, matches your Python semantics
    n = size(true_coords, 1)
    m = size(pred_coords, 1)

    cost = fill(Inf, n, m)

    dist(i,j) = sqrt(sum((true_coords[i,k] - pred_coords[j,k])^2 for k in 1:size(true_coords,2)))

    cost[1,1] = dist(1,1)

    for i in 2:n
        cost[i,1] = max(cost[i-1,1], dist(i,1))
    end
    for j in 2:m
        cost[1,j] = max(cost[1,j-1], dist(1,j))
    end
    for i in 2:n, j in 2:m
        cost[i,j] = max(min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1]), dist(i,j))
    end
    return cost[n,m]
end

end # module
