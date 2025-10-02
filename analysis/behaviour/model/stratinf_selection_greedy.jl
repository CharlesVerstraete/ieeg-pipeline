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

@model stratinf_selection_greedy(states, actions, feedbacks) = begin
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
        
        selected_strategy = rand(Categorical(softmax(β*@views(Φ))))
        if selected_strategy == 28
            P .= 1.0 / na
        else
            q .= 0.0
            q .+= ALL_SETS[selected_strategy]
            P = q[:, s]
            norm = softmax(β*@views(Φ))[selected_strategy]
            P .*= norm
            P .+= (1-norm) / length(P)
            P .*= (1 - ϵ)
            P .+= ϵ / length(P)
        end

        actions[t] ~ Categorical(P)
        la[t] = log(P[a])
        
        update_memory_trace!(counter, Γ, tmp_gamma, Φ, ω₀, ϕ₀)
        update_prior!(Π, Φ, ω_, ωᵣ)
        update_reliabilities!(Π, Φ, Γ, ω_, ωᵣ, a, s, r, ρ, na)
        
        
    end
return sum(la)
end
