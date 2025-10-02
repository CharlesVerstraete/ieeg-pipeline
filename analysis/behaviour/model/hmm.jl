# julia 1.10
# -*- coding: utf-8 -*-

"""
Description:
    This file contains the model definition for the hidden Markov model to detect behavioral switches

Author :
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2025-05
"""
@model hmm_switch(states, actions) = begin
    Θ ~ MvNormal(5, 1.0)
    τ ~ Beta()
    ρ ~ truncated(Beta(), 0.0, 0.5)

    nt = length(states)

    Lprior = Vector{Float64}(fill(log(1/NSETS+1), NSETS+1))

    LLH = zeros(Float64,NSETS+1)

    Lpost = Vector{Float64}(fill(log(1/NSETS+1), NSETS+1))

    #bestPrior = Array{Int, 2}(undef, nt, nh+1)
    bestPrior = zeros(Int, nt, NSETS+1)
    Ltrans = build_trans_mat(Θ, τ)

    tmp_prior = zeros(Float64, size(Ltrans))
    tmp_post = zeros(Float64, size(Ltrans))
    for t in 1:nt
        s = states[t]
        a = actions[t]
        tmp_prior .= Ltrans 
        tmp_prior .+= Lprior
        tmp_post .= Ltrans
        tmp_post .+= Lpost
        # Forward pass
        for hi in eachindex(HIDDEN_STATES)
           Lprior[hi] = logsumexp(@views(tmp_prior[:,hi]))
            LLH[hi] = get_llh(ρ, HIDDEN_STATES[hi], s, a)
            bv, bestPrior[t, hi] = findmax(@views(tmp_post[:,hi]))
            Lpost[hi] = bv + LLH[hi]
        end
        # Exploration state
        Lprior[end] = logsumexp(@views(tmp_prior[:,end]))
        LLH[end] = log(1/3)
        bv, bestPrior[t,end] = findmax(@views(tmp_post[:,end]))
        Lpost[end] = bv + LLH[end]

        # Update prior
        Lprior .+= LLH
        margL = logsumexp(Lprior)
        Lprior .-= margL
        Turing.@addlogprob! margL
    end

    # Backtracking 
    H = Vector{Int}(undef, nt)
    H[end] = argmax(Lpost) # The last state is the one with the full trajectory's MAP
    for t = nt-1:-1:1 # For each previous trial
        H[t] = bestPrior[t+1, H[t+1]]
    end
    return H
end
