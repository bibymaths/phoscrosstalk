module Dynamics

using LinearAlgebra
using DifferentialEquations
using SciMLSensitivity
using ForwardDiff

export SimContext, decode_theta!, rhs!, build_problem, mech_code

# -------------------------
# Mechanism encoding
# -------------------------
@inline function mech_code(mech::AbstractString)
    mech == "dist" && return 0
    mech == "seq"  && return 1
    mech == "rand" && return 2
    error("Unknown mechanism: $mech")
end

# -------------------------
# Context (all hot-path arrays are typed, no Any)
# -------------------------
struct SimContext{TF<:AbstractFloat, TI<:Integer,
                  TM1<:AbstractMatrix{TF}, TM2<:AbstractMatrix{TF}, TM3<:AbstractMatrix{TF}, TM4<:AbstractMatrix{TF}, TM5<:AbstractMatrix{TF}, TM6<:AbstractMatrix{TF},
                  TVi<:AbstractVector{TI}, TVb<:AbstractVector{TI}}
    K::Int
    M::Int
    N::Int

    # topology / data
    Cg::TM1
    Cl::TM2
    K_site_kin::TM3
    R::TM4
    L_alpha::TM5

    site_prot_idx::TVi
    kin_to_prot_idx::TVi
    receptor_mask_prot::TVb
    receptor_mask_kin::TVb

    # decoded parameters (allocated once, filled each eval)
    k_act::Vector{TF}
    k_deact::Vector{TF}
    s_prod::Vector{TF}
    d_deg::Vector{TF}

    alpha::Vector{TF}
    kK_act::Vector{TF}
    kK_deact::Vector{TF}
    k_off::Vector{TF}

    beta_g::TF
    beta_l::TF

    gamma_S_p::TF
    gamma_A_S::TF
    gamma_A_p::TF
    gamma_K_net::TF

    # work buffers (allocated once)
    p_buf::Vector{TF}
    coup_buf::Vector{TF}
    num_p::Vector{TF}
    num_c::Vector{TF}
    den::Vector{TF}

    u_sub::Vector{TF}
    u_net::Vector{TF}
    k_on_eff::Vector{TF}
    last_occ::Vector{TF}
    has_prev::Vector{TF}
end

function SimContext(Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                    kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin)
    TF = Float64
    TI = Int64
    K = length(receptor_mask_prot)
    M = length(receptor_mask_kin)
    N = length(site_prot_idx)

    return SimContext{TF,TI, typeof(Cg), typeof(Cl), typeof(K_site_kin), typeof(R), typeof(L_alpha), typeof(L_alpha),
                      typeof(site_prot_idx), typeof(receptor_mask_prot)}(
        K, M, N,
        Cg, Cl, K_site_kin, R, L_alpha,
        Vector{TI}(site_prot_idx), Vector{TI}(kin_to_prot_idx),
        Vector{TI}(receptor_mask_prot), Vector{TI}(receptor_mask_kin),
        ones(TF,K), ones(TF,K), ones(TF,K), ones(TF,K),
        ones(TF,M), ones(TF,M), ones(TF,M), ones(TF,N),
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        similar(zeros(TF,N)), similar(zeros(TF,N)),
        similar(zeros(TF,K)), similar(zeros(TF,K)), similar(zeros(TF,K)),
        similar(zeros(TF,M)), similar(zeros(TF,M)),
        similar(zeros(TF,N)), similar(zeros(TF,K)), similar(zeros(TF,K))
    )
end

@inline clip_scalar(x, lo, hi) = x < lo ? lo : (x > hi ? hi : x)

# -------------------------
# Decode theta (log params -> exp, gammas -> tanh)
# theta layout matches your Python decode_theta
# -------------------------
function decode_theta!(ctx::SimContext, θ::AbstractVector{<:Real})
    K, M, N = ctx.K, ctx.M, ctx.N
    idx = 1

    @views log_k_act    = θ[idx:idx+K-1]; idx += K
    @views log_k_deact  = θ[idx:idx+K-1]; idx += K
    @views log_s_prod   = θ[idx:idx+K-1]; idx += K
    @views log_d_deg    = θ[idx:idx+K-1]; idx += K

    log_beta_g = θ[idx]; idx += 1
    log_beta_l = θ[idx]; idx += 1

    @views log_alpha    = θ[idx:idx+M-1]; idx += M
    @views log_kK_act   = θ[idx:idx+M-1]; idx += M
    @views log_kK_deact = θ[idx:idx+M-1]; idx += M

    @views log_k_off    = θ[idx:idx+N-1]; idx += N
    @views raw_gamma = θ[idx:idx+3]; idx += 4

    @inbounds for i in 1:K
        ctx.k_act[i]   = exp(clamp(log_k_act[i],   -20.0, 10.0))
        ctx.k_deact[i] = exp(clamp(log_k_deact[i], -20.0, 10.0))
        ctx.s_prod[i]  = exp(clamp(log_s_prod[i],  -20.0, 10.0))
        ctx.d_deg[i]   = exp(clamp(log_d_deg[i],   -20.0, 10.0))
    end
    @inbounds for m in 1:M
        ctx.alpha[m]   = exp(clamp(log_alpha[m],   -20.0, 10.0))
        ctx.kK_act[m]  = exp(clamp(log_kK_act[m],  -20.0, 10.0))
        ctx.kK_deact[m]= exp(clamp(log_kK_deact[m],-20.0, 10.0))
    end
    @inbounds for n in 1:N
        ctx.k_off[n]   = exp(clamp(log_k_off[n],   -20.0, 10.0))
    end

    ctx.beta_g = exp(clip_scalar(log_beta_g, -20.0, 10.0))
    ctx.beta_l = exp(clip_scalar(log_beta_l, -20.0, 10.0))

    ctx.gamma_S_p   = 2.0 * tanh(raw_gamma[1])
    ctx.gamma_A_S   = 2.0 * tanh(raw_gamma[2])
    ctx.gamma_A_p   = 2.0 * tanh(raw_gamma[3])
    ctx.gamma_K_net = 2.0 * tanh(raw_gamma[4])

    return nothing
end

# -------------------------
# RHS: in-place, zero-allocation (uses ctx buffers)
# State layout: [S(1:K), A(K+1:2K), Kdyn(2K+1:2K+M), p(2K+M+1:end)]
# -------------------------
function rhs!(du, u, p, t)
    ctx, mech = p.ctx, p.mech
    θ = p.theta
    decode_theta!(ctx, θ)

    K, M, N = ctx.K, ctx.M, ctx.N

    S    = @view u[1:K]
    A    = @view u[K+1:2K]
    Kdyn0= @view u[2K+1:2K+M]
    p0   = @view u[2K+M+1:2K+M+N]

    dS   = @view du[1:K]
    dA   = @view du[K+1:2K]
    dK   = @view du[2K+1:2K+M]
    dp   = @view du[2K+M+1:2K+M+N]

    # clip Kdyn -> u_net, clip p -> p_buf
    @inbounds for m in 1:M
        v = Kdyn0[m]
        v = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v)
        ctx.u_net[m] = v
    end
    Kdyn = ctx.u_net

    @inbounds for i in 1:N
        v = p0[i]
        v = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v)
        ctx.p_buf[i] = v
    end
    pvec = ctx.p_buf

    # stimulus
    uext = 1.0 / (1.0 + exp(-(t) / 0.1))

    # coupling: coup = tanh(beta_g*(Cg*p) + beta_l*(Cl*p))
    mul!(ctx.coup_buf, ctx.Cg, pvec)                 # coup_buf = Cg*p
    @inbounds for i in 1:N
        ctx.coup_buf[i] *= ctx.beta_g
    end
    mul!(ctx.k_on_eff, ctx.Cl, pvec)                 # reuse k_on_eff as temp = Cl*p
    @inbounds for i in 1:N
        ctx.coup_buf[i] += ctx.beta_l * ctx.k_on_eff[i]
        ctx.coup_buf[i]  = tanh(ctx.coup_buf[i])
    end
    coup = ctx.coup_buf

    # per-protein aggregates
    fill!(ctx.num_p, 0.0); fill!(ctx.num_c, 0.0); fill!(ctx.den, 0.0)
    @inbounds for i in 1:N
        prot = ctx.site_prot_idx[i] + 1 # Python indices are 0-based; snapshot uses ints. Enforce 0-based -> +1 here.
        ctx.num_p[prot] += pvec[i]
        ctx.num_c[prot] += coup[i]
        ctx.den[prot]   += 1.0
    end

    # S, A dynamics
    @inbounds for k in 1:K
        mp = ctx.den[k] > 0 ? ctx.num_p[k] / ctx.den[k] : 0.0
        mc = ctx.den[k] > 0 ? ctx.num_c[k] / ctx.den[k] : 0.0

        D_S = 1.0 + ctx.gamma_S_p * mp + mc
        if ctx.receptor_mask_prot[k] == 1
            D_S += uext
        end
        D_S = D_S < 0.0 ? 0.0 : D_S

        dS[k] = ctx.k_act[k] * D_S * (1.0 - S[k]) - ctx.k_deact[k] * S[k]

        s_eff = ctx.s_prod[k] * (1.0 + ctx.gamma_A_S * S[k])
        s_eff = s_eff < 0.0 ? 0.0 : s_eff
        dA[k] = s_eff - ctx.d_deg[k] * A[k]
    end

    # Kdyn drive
    mul!(ctx.u_sub, ctx.R, pvec)                     # u_sub = R*p

    if size(ctx.L_alpha,1) > 0 && ctx.gamma_K_net != 0.0
        mul!(ctx.u_net, ctx.L_alpha, Kdyn)          # u_net = L_alpha*Kdyn
        @inbounds for m in 1:M
            ctx.u_net[m] = -ctx.u_net[m]
        end
    else
        fill!(ctx.u_net, 0.0)
    end

    @inbounds for m in 1:M
        U = ctx.u_sub[m] + ctx.gamma_K_net * ctx.u_net[m]

        prot0 = ctx.kin_to_prot_idx[m]
        if prot0 >= 0
            prot = prot0 + 1
            U += ctx.gamma_A_S * S[prot] + ctx.gamma_A_p * A[prot]
        end
        if ctx.receptor_mask_kin[m] == 1
            U += uext
        end
        act_term = tanh(U)
        dK[m] = ctx.kK_act[m] * act_term * (1.0 - Kdyn[m]) - ctx.kK_deact[m] * Kdyn[m]
    end

    # phosphorylation: k_on_eff = K_site_kin * (alpha .* Kdyn)
    @inbounds for m in 1:M
        ctx.u_sub[m] = ctx.alpha[m] * Kdyn[m]
    end
    mul!(ctx.k_on_eff, ctx.K_site_kin, ctx.u_sub)

    if mech == 0
        @inbounds for i in 1:N
            cp = coup[i]; cp = cp < 0 ? 0.0 : cp
            vraw = ctx.k_on_eff[i] * (1.0 + cp) * (1.0 - pvec[i])
            von  = vraw / (1.0 + abs(vraw))
            voff = (ctx.k_off[i] * pvec[i]); voff = voff / (1.0 + voff)
            dp[i] = von - voff
        end
    elseif mech == 1
        fill!(ctx.last_occ, 0.0); fill!(ctx.has_prev, 0.0)
        @inbounds for i in 1:N
            prot = ctx.site_prot_idx[i] + 1
            gate = ctx.has_prev[prot] == 0.0 ? 1.0 : ctx.last_occ[prot]
            cp = coup[i]; cp = cp < 0 ? 0.0 : cp
            vraw = ctx.k_on_eff[i] * (1.0 + cp) * gate * (1.0 - pvec[i])
            von  = vraw / (1.0 + abs(vraw))
            voff = (ctx.k_off[i] * pvec[i]); voff = voff / (1.0 + voff)
            dp[i] = von - voff
            ctx.has_prev[prot] = 1.0
            ctx.last_occ[prot] = pvec[i]
        end
    else
        @inbounds for i in 1:N
            prot = ctx.site_prot_idx[i] + 1
            coop = ctx.den[prot] > 0 ? (1.0 + ctx.num_p[prot] / ctx.den[prot]) : 1.0
            cp = coup[i]; cp = cp < 0 ? 0.0 : cp
            vraw = ctx.k_on_eff[i] * (1.0 + cp) * coop * (1.0 - pvec[i])
            von  = vraw / (1.0 + abs(vraw))
            voff = (ctx.k_off[i] * pvec[i]); voff = voff / (1.0 + voff)
            dp[i] = von - voff
        end
    end

    return nothing
end

# Problem builder (stiff solver + AD Jacobian)
function build_problem(ctx::SimContext, θ::AbstractVector, u0::Vector{Float64}, tspan::Tuple{Float64,Float64}, t_eval::Vector{Float64}, mech::Int)
    p = (; ctx=ctx, theta=θ, mech=mech)

    f = ODEFunction(rhs!;
        jac = (J,u,p,t) -> ForwardDiff.jacobian!(J, uu -> (du = similar(uu); rhs!(du, uu, p, t); du), u)
    )

    prob = ODEProblem(f, u0, tspan, p)
    return prob, t_eval
end

end # module
