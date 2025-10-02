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


function get_chain(subject, model; type = "")
    f_sub = @sprintf("%03d", subject)
    chain_file = string(DATA_PATH, "/chains", "/", model, "/", type, "/sub-$(f_sub)_chains.jls")
    chain = deserialize(chain_file)
    return chain
end

function get_pllh!(subject, chain_df, model; type = "")
    f_sub = @sprintf("%03d", subject)
    pllh_file = string(DATA_PATH, "/chains", "/", model, "/", type, "/sub-$(f_sub)_pllh.jld2")
    llh = JLD2.load(pllh_file, "pllh")
    chain_df[:, :llh] = vcat(sum(llh)...)
    return chain_df
end
    
function norm_llh!(chain_df; temp = 5.0)
    max_llh = maximum(chain_df.llh)
    w = exp.((chain_df.llh .- max_llh)/temp)
    w ./= sum(w)
    chain_df[:, "wllh"] = w
    return chain_df
end

function make_chaindf(subject, model; type = "",  burn = 1000, temp = 5.0)
    chain = get_chain(subject, model, type = type)
    chain_df = DataFrame(chain)
    get_pllh!(subject, chain_df, model, type = type)
    filter!(x -> x.iteration > burn, chain_df)
    norm_llh!(chain_df, temp = temp)
    return chain_df
end

function get_weighted_average(chain_df, param_cols)
    weighted_params = DataFrame(Dict(col => mean(chain_df[:,col], Weights(chain_df.wllh)) for col in param_cols))
    return weighted_params
end

function get_sampled_params(chain_df, param_cols, resample = 100)
    sampled_param = chain_df[sample(axes(chain_df, 1), Weights(chain_df.wllh), resample), param_cols]
    return sampled_param
end

function get_params(chain_df,; mode = "average", resample = 100, cols_rm = ["lp", "iteration", "chain", "llh", "wllh"])
    param_cols = setdiff(names(chain_df), cols_rm)
    if mode == "average"
        output = get_weighted_average(chain_df, param_cols)
    elseif mode == "sample"
        output = get_sampled_params(chain_df, param_cols, resample)
    end
    return output
end

function get_param_set(row)
    ω_ = collect(row[["ω[1]", "ω[2]", "ω[3]", "ω[4]", "ω[5]"]])
    ω = softmax(ω_)
    cols = setdiff(names(row), ["ω[1]", "ω[2]", "ω[3]", "ω[4]", "ω[5]"])
    ω_tmp = (ω = ω,)
    cols_nt = NamedTuple(Symbol(col) => row[col] for col in cols)
    return merge(ω_tmp, cols_nt)
end

function init_simu_df(df; col_toadd = [:selected_strategy, :reliability, :switch, :action_value, :update_reliability, :rpe, :entropy, :counterfactual, :update_counterfactual, :criterion, :post_criterion, :joint_counterfactual])
    new = deepcopy(df)
    for col in col_toadd
        new[!, col] = Vector{Union{Missing, Float64}}(missing, nrow(new))
    end
    return new
end

function run_simulation(param, na, ns, df)
    Γ, tmp_gamma, Π, Φ, counter, ϕ₀, q, P = init_internals(param.ρ, param.ω₀, na, ns)
    idx_newblock = findall(((df.new_block .== 1)) .|| (df.trial .== 1))
    idx_newblock = vcat(idx_newblock, nrow(df)+1)
    for i in 1:(length(idx_newblock)-1)
        idx_start = idx_newblock[i]
        idx_end = idx_newblock[i+1] - 1   
        run_episode!(idx_start, idx_end, df, Γ, tmp_gamma, Π, Φ, counter, ϕ₀, q, P, param)
    end
    return df #df[df.trial_succeed .== 1, :]
end

function run_episode!(idx_start, idx_end, df, Γ, tmp_gamma, Π, Φ, counter, ϕ₀, q, P, param)
    good_count = zeros(Int, 3)
    success = 0
    post_perf = 0
    finished_episode = false
    stim_count = zeros(Int, 3)
    idx_epis = idx_start
    selected_strategy = 0
    while (idx_epis <= idx_end) #&& !finished_episode
        s = df.stim[idx_epis]
        stim_count[s] += 1
        df[idx_epis, :stim_pres] = stim_count[s]
        trap = Bool(df[idx_epis, :trap])
        correct_choice = df[idx_epis, :correct_choice]
        trial = df[idx_epis, :trial]

        entropy = (-sum(Φ .* log.(Φ)))/log(NSETS+1)

        q, P = get_action(param.model_type, q, P, na, s, param, Φ)
        q_ = deepcopy(q)
        # a = rand(Categorical(P))
        a = df[idx_epis, :choice]
        correct = df[idx_epis, :correct]
        r = df[idx_epis, :fb]
        # correct = a == correct_choice
        # r = trap ? (a != correct_choice) : (a == correct_choice)
        
        selected_strategy = argmax(Φ)
        # df[idx_epis, :choice] = a
        # df[idx_epis, :correct] = correct
        # df[idx_epis, :fb] = r
        df[idx_epis, :trial_succeed] = true
        df[idx_epis, :selected_strategy] = selected_strategy
        df[idx_epis, :entropy] = entropy
        df[idx_epis, :reliability] = Φ[selected_strategy]
        df[idx_epis, :action_value] = P[a]
        df[idx_epis, :rpe] = r - q[s, a]
        good_count, success, post_perf, finished_episode = criterion_update(
            idx_epis, trial, s, df[:, :correct], good_count, success, post_perf, finished_episode
        )
        

        counterfacts_a = setdiff(1:na, a)
        counterfacts_s = setdiff(1:ns, s)
        max_a, max_s  = Tuple(argmax(q[counterfacts_a, counterfacts_s]))
        joint_strategies = [i for i in 1:NSETS if ((ALL_SETS[i][max_a, max_s] == 1) && (ALL_SETS[i][a,s] == 1))]
        df[idx_epis, :joint_counterfactual] = sum(Φ[joint_strategies])
        
        df[idx_epis, :counterfactual] = maximum(P[counterfacts_a])

        df[idx_epis, :criterion] = post_perf==1
        df[idx_epis, :post_criterion] = post_perf

        update_memory_trace!(counter, Γ, tmp_gamma, Φ, param.ω₀, ϕ₀)
        update_prior!(Π, Φ, param.ω, param.ωᵣ)
        update_reliabilities!(Π, Φ, Γ, param.ω, param.ωᵣ, a, s, r, param.ρ, na)
        
        df[idx_epis, :update_reliability] = Φ[selected_strategy] - df[idx_epis, :reliability]
        q_tmp, _ = get_action(param.model_type, q, P, na, s, param, Φ)
        df[idx_epis, :update_counterfactual] = q_tmp[max_a, max_s] - q_[max_a, max_s]
         
        idx_epis += 1
    end
    return df
end

function get_action(model_type, q, P, na, s, param, Φ)
    if model_type == "selection"
        q = compute_q!(softmax(param.β.*Φ), q, na)
        P .= @views(q[:,s])
        P .= epsilon_greedy!(P, param.ϵ)
    elseif model_type == "transition"
        q = compute_q!(Φ, q, na)
        P .= custom_softmax!(P, @views(q[:,s]), param.β, param.ϵ)
    end
    return q, P
end