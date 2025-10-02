# julia 1.10
# -*- coding: utf-8 -*-

"""
Description:
    This file contains the model definition for the strategic inference model with transition probabilities

Author :
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2025-05
"""

@model stratinf_transition(states, actions, feedbacks) = begin
    ρ ~ truncated(Beta(), 0.5, 1)
    ω₀ ~ Beta() # Long-term memory
    ω ~ MvNormal(5, 1.0) # Volatility for each type of switches + random + stability
    ωᵣ ~ Beta() # Proba switching out of random
    ϵ ~ Beta()
    β ~ truncated(LogNormal(), 0, 100)

    ω_ = softmax(ω)
    nt = length(states)
    ns, na = maximum(states), maximum(actions)
    Γ, tmp_gamma, Π, Φ, counter, ϕ₀, q, P = init_internals(ρ, ω₀, na, ns)
    la = zeros(typeof(ω₀), nt)
    for t = 1:nt
        s = states[t]
        a = actions[t]
        r = feedbacks[t]
        
        q = compute_q(Φ, q, na)
        P .= custom_softmax!(P, @views(q[:,s]), β, ϵ)

        actions[t] ~ Categorical(P)
        la[t] = log(P[a])
        
        update_memory_trace!(counter, Γ, tmp_gamma, Φ, ω₀, ϕ₀)
        update_prior!(Π, Φ, ω_, ωᵣ)
        update_reliabilities!(Π, Φ, Γ, ω_, ωᵣ, a, s, r, ρ, na)
        
    end
return sum(la)
end
