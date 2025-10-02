# julia 1.10
# -*- coding: utf-8 -*-

"""
Description:
    Inference script to run the MCMC

Author :
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2025-05
"""

include("librairy.jl")

subject = parse(Int, ENV["SUBJECT"])
nsample = parse(Int, ENV["N_SAMPLES"])
nchains = parse(Int, ENV["N_CHAINS"])

f_sub = @sprintf("%03d", subject)
file_path = string(DATA_PATH, "/beh", "/sub-$(f_sub)_task-stratinf_beh.tsv")
data = CSV.read(file_path, DataFrame, delim = ',')
println("Data loaded from $(file_path)")

cleaned_data = filter(x -> (x.trial_succeed .== 1), data)

states = cleaned_data.stim
actions = cleaned_data.choice
feedbacks = cleaned_data.fb

########################################################################################################################
############################## HMM #####################################################################################
########################################################################################################################

# sampler = Gibbs(NUTS(100, 0.65, init_ϵ = 0.05, :τ, :ρ), ESS(:Θ))

# mdl = hmm_switch(states, actions)
# chain_data = sample(mdl, sampler, MCMCThreads(), nsample, nchains)
# chains_params = MCMCChains.get_sections(chain_data, :parameters)
# latent = generated_quantities(mdl, chains_params)
# pllh = collect(values(pointwise_loglikelihoods(mdl, chains_params)))

# chain_file = string(DATA_PATH, "/chains", "/HMM", "/sub-$(f_sub)_chains.jls")
# latent_file = string(DATA_PATH, "/chains", "/HMM", "/sub-$(f_sub)_latent.jld2")
# pllh_file = string(DATA_PATH, "/chains", "/HMM", "/sub-$(f_sub)_pllh.jld2")

# serialize(chain_file, chain_data)
# JLD2.save(pllh_file, "pllh", pllh)
# JLD2.save(latent_file, "latent", latent)

# println("HMM completed")



# ########################################################################################################################
# ############################## SI Selection ############################################################################
# ########################################################################################################################

sampler = Gibbs(NUTS(100, 0.65, init_ϵ = 0.05, :ρ, :ω₀, :ωᵣ, :ϵ, :β), ESS(:ω))
mdl = stratinf_selection(states, actions, feedbacks)
chain_data = sample(mdl, sampler, MCMCThreads(), nsample, nchains)
chains_params = MCMCChains.get_sections(chain_data, :parameters)
latent = generated_quantities(mdl, chains_params)
pllh = collect(values(pointwise_loglikelihoods(mdl, chains_params)))

chain_file = string(DATA_PATH, "/chains", "/SI", "/selection", "/sub-$(f_sub)_chains.jls")
latent_file = string(DATA_PATH, "/chains", "/SI", "/selection", "/sub-$(f_sub)_latent.jld2")
pllh_file = string(DATA_PATH, "/chains", "/SI", "/selection", "/sub-$(f_sub)_pllh.jld2")

serialize(chain_file, chain_data)
JLD2.save(pllh_file, "pllh", pllh)
JLD2.save(latent_file, "latent", latent)

println("SI selection completed")


# ########################################################################################################################
# ############################## SI Entropy ##############################################################################
# ########################################################################################################################


# sampler = Gibbs(NUTS(100, 0.65, init_ϵ = 0.05, :ρ, :ω₀, :ωᵣ, :ϵ, :β, :α), ESS(:ω))
# mdl = stratinf_entropy(states, actions, feedbacks)
# chain_data = sample(mdl, sampler, MCMCThreads(), nsample, nchains)
# chains_params = MCMCChains.get_sections(chain_data, :parameters)
# latent = generated_quantities(mdl, chains_params)
# pllh = collect(values(pointwise_loglikelihoods(mdl, chains_params)))

# chain_file = string(DATA_PATH, "/chains", "/SI", "/entropy", "/sub-$(f_sub)_chains.jls")
# latent_file = string(DATA_PATH, "/chains", "/SI", "/entropy", "/sub-$(f_sub)_latent.jld2")
# pllh_file = string(DATA_PATH, "/chains", "/SI", "/entropy", "/sub-$(f_sub)_pllh.jld2")

# serialize(chain_file, chain_data)
# JLD2.save(pllh_file, "pllh", pllh)
# JLD2.save(latent_file, "latent", latent)

# println("SI entropy completed")




########################################################################################################################
############################## SI greedy ##############################################################################
########################################################################################################################


# sampler = Gibbs(NUTS(100, 0.65, init_ϵ = 0.05, :ρ, :ω₀, :ωᵣ, :ϵ, :β), ESS(:ω))
# mdl = stratinf_selection_greedy(states, actions, feedbacks)
# chain_data = sample(mdl, sampler, MCMCThreads(), nsample, nchains)
# plot(chain_data)
# chains_params = MCMCChains.get_sections(chain_data, :parameters)
# latent = generated_quantities(mdl, chains_params)
# pllh = collect(values(pointwise_loglikelihoods(mdl, chains_params)))

# chain_file = string(DATA_PATH, "/chains", "/SI", "/greedy", "/sub-$(f_sub)_chains.jls")
# latent_file = string(DATA_PATH, "/chains", "/SI", "/greedy", "/sub-$(f_sub)_latent.jld2")
# pllh_file = string(DATA_PATH, "/chains", "/SI", "/greedy", "/sub-$(f_sub)_pllh.jld2")

# serialize(chain_file, chain_data)
# JLD2.save(pllh_file, "pllh", pllh)
# JLD2.save(latent_file, "latent", latent)

# println("SI greedy completed")


