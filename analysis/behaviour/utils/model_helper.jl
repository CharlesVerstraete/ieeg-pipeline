# julia 1.10
# -*- coding: utf-8 -*-

"""
model_helper.jl
================

Description:
    This file contains the model helper functions
Author :
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2025-05
"""

function custom_softmax!(P, V, β, ϵ = 0.0)
    M = maximum(V)
    S = 0.0
    for i = eachindex(P)
        S += P[i] = exp(β * (V[i] - M))
    end
    P ./= S
    P .*= (1 - ϵ)
    P .+= ϵ / length(P)
    return P
end


function init_internals(ρ, ω₀, na, ns)
    return  (Γ = Vector{typeof(ω₀)}(fill(1/(NSETS+1), NSETS+1)),
            tmp_gamma = Vector{typeof(ω₀)}(fill(1/(NSETS+1), NSETS+1)),
            Π = Vector{typeof(ω₀)}(fill(1/(NSETS+1), NSETS+1)),
            Φ = Vector{typeof(ρ)}(fill(1/(NSETS+1), NSETS+1)),
            counter = [1.0],
            ϕ₀ = 1/(NSETS+1),
            q = Matrix{typeof(ρ)}(fill(1/na, na, ns)),
            P = zeros(typeof(ω₀), na))
end

function update_memory_trace!(counter, Γ, tmp_gamma, Φ, ω₀, ϕ₀)
    counter[1] += 1
    α = 1 / counter[1]
    Γ .*= 1 - α
    tmp_gamma .= Φ
    tmp_gamma .*= 1-ω₀
    tmp_gamma .+= ω₀ * ϕ₀
    tmp_gamma .*= α
    Γ .+= tmp_gamma
end

function update_prior!(Π, Φ, ω, ωᵣ)
    Π .= Φ
    Π[1:end-1] .*= ω[end]
    Π[end] *= 1 - ωᵣ
end

function update_reliabilities!(Π, Φ, Γ, ω, ωᵣ, a, s, r, ρ, na)
    for i = 1:NSETS
        for st = 1:3 # Switch type 
            for j in TYPE_IDX[i,st]
                Z = 0.0
                for k in TYPE_IDX[j,st]
                    Z += Γ[k]
                end
                Π[i] += ω[st] * Φ[j] * Γ[i] / Z
            end
        end
        Π[i] += ωᵣ * Φ[end] * Γ[i] / (1 - Γ[end]) # From random set 
        Π[end] += ω[end-1] * Φ[i] * Γ[end] / (1 - Γ[i]) # From set i to random
        Π[i] *= ALL_SETS[i][a,s] == r ? ρ : 1-ρ  
    end
    Π[end] *= r == 1 ? 1/na : 1-1/na
    Φ .= Π
    Z = sum(Φ)
    Φ ./= Z
end

function compute_q!(w, q, na)
    q .= 0.0
    for i in 1:NSETS
        q .+= w[i] .* ALL_SETS[i]
    end
    q .+= w[end]/na
    return q
end

function epsilon_greedy!(P, ϵ)
    P .*= (1 - ϵ)
    P .+= ϵ / length(P)
    return P
end

function build_trans_mat(Θ, τ)
    p = logistic.(Θ)
    p[2:5] .*=  (0.5 - 0.5*p[1]) / sum(p[2:5])
    p[1] = 0.5 * (1 + p[1])
    for i = 2:5
        p[i] /= HN[i]
    end
    l = log.(p)
    Ltrans = getindex.([l], HTRANS .+ 1)
    Ltrans[end,:] .= log((1 - ForwardDiff.value(τ))/NSETS) # Probability of leaving exploration
    Ltrans[end,end] = log(ForwardDiff.value(τ)) # Probability of staying in exploration
    
    return Ltrans
end

function get_llh(ρ, hs, s, a)
    return hs[s] ≠ a ? log(ForwardDiff.value(ρ)/2) : log(1 - ForwardDiff.value(ρ))
end

