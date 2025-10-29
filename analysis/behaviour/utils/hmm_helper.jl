function add_switch!(df::DataFrame, subject::Int, save_path::String)
    strat_map = load_latent_states(save_path, subject)
    chn = load_chains(save_path, subject)
    w = get_llh_weight(chn)
    df.hmm_strat .= mode.(eachrow(strat_map))
    process_switch!(df)
    return df
end

function load_latent_states(save_path, subject::Int)
    latent_file = joinpath(save_path, "sub-$(@sprintf("%03d", subject))_latent.jld2")
    latent = load(latent_file)["latent"]
    return hcat(latent...)
end

function load_chains(save_path, subject::Int)
    chain_file = joinpath(save_path, "sub-$(@sprintf("%03d", subject))_chains.jls")
    return deserialize(chain_file)
end

function get_llh_weight(chn)
    chn_df = DataFrame(chn)
    w = exp.(chn_df.lp .- maximum(chn_df.lp))
    w ./= sum(w)
    return w
end

function initialize_switch_columns!(df::DataFrame)
    df.good_strat .= df.hmm_strat .== df.rule
    df.hmm_switch .= vcat(0, diff(df.hmm_strat) .!= 0)
    
    df.goodswitch .= (df.good_strat .== 1) .& (df.hmm_switch .== 1)
    df.time_goodswitch .= 0
    df.time_firstswitch .= 0

    df.firstswitch .= 0
    df.switch_count .= 0

    df.firstsw_pres .= 0
    df.firstsw_trial .= 0

    df.goodsw_pres .= 0
    df.goodsw_trial .= 0

    df.is_random .= df.hmm_strat .== 28
    df.random_switch .= (df.is_random .== 1) .& (df.hmm_switch .== 1)

    df.randomsw_pres .= 0
    df.randomsw_trial .= 0

    df.persev_hmm .= 0
    df.explor_hmm .= 0

    df.firstsw_type .= ""
    df.othersw_type .= ""
    df.goodsw_type .= ""
    df.randomsw_type .= ""
end

function forward_counters!(df::DataFrame, switch_idx::Int, colname::String, persev_choices::Vector{Int64}, current_strat::Int, switch_type::String)
    j = switch_idx
    counter_stim = zeros(Int, 3)
    trial_counter = 0
    trial_colname = string(colname, "_trial")
    pres_colname = string(colname, "_pres")
    type_colname = string(colname, "_type")
    while (minimum(counter_stim) < 3) && (j < nrow(df)) && (df[j, :hmm_strat] == current_strat)
        counter_stim[df[j, :stim]] += 1
        df[j, pres_colname] = counter_stim[df[j, :stim]]
        trial_counter += 1
        df[j, trial_colname] = trial_counter
        df[j, :persev_hmm] = persev_choices[df[j, :stim]] == df[j, :choice]
        explor = filter!(e -> !(e in [df[j, :persev_hmm], df[j, :correct_choice]]), [1,2,3])
        df[j, :explor_hmm] = df[j, :choice] ∈ explor
        df[j, type_colname] = df[j, type_colname] == "" ? switch_type : df[j, type_colname]
        j += 1
    end
end

function backward_counters!(df::DataFrame, switch_idx::Int, colname::String, persev_choices::Vector{Int64}, prev_strat::Int, switch_type::String)
    i = switch_idx - 1
    counter_stim = zeros(Int, 3)
    trial_counter = 0
    trial_colname = string(colname, "_trial")
    pres_colname = string(colname, "_pres")
    type_colname = string(colname, "_type")
    while (abs(minimum(counter_stim)) < 3) && (i > 0) && (df[i, :hmm_strat] == prev_strat)
        counter_stim[df[i, :stim]] -= 1
        df[i, pres_colname] = counter_stim[df[i, :stim]]
        trial_counter += 1
        df[i, trial_colname] = -trial_counter
        df[i, :persev_hmm] = persev_choices[df[i, :stim]] == df[i, :choice]
        explor = filter!(e -> !(e in [df[i, :persev_hmm], df[i, :correct_choice]]), [1,2,3])
        df[i, :explor_hmm] = df[i, :choice] ∈ explor
        df[i, type_colname] = df[i, type_colname] == "" ? switch_type : df[i, type_colname]
        i -= 1
    end
end

function around_switch_counts!(df::DataFrame, switch_idx::Int, colname::String, switch_type::String, persev_choices::Vector{Int64})
    current_strat = df[switch_idx, :hmm_strat]
    if switch_idx > 1
        prev_strat = df[switch_idx-1, :hmm_strat]
    else
        prev_strat = -1
    end
    # explor_choice = df[switch_idx, :explor_choice]
    forward_counters!(df, switch_idx, colname, persev_choices, current_strat, switch_type)
    backward_counters!(df, switch_idx, colname, persev_choices, prev_strat, switch_type)
end



function process_switch!(df::DataFrame)
    initialize_switch_columns!(df)
    firstsw = false
    goodsw = false
    switch_counters = 0
    active_rule = Array(df[:, 1:3])
    persev_choices = zeros(Int, 3)
    for idx in 2:nrow(df)
        if (df.trial[idx] == 1) || (df.new_block[idx] == 1)
            switch_counters = 0
            firstsw = false
            goodsw = false
            persev_choices = active_rule[idx - 1, :]
        end
        if !firstsw && df.hmm_switch[idx] == 1
            firstsw = true
            df.firstswitch[idx] = 1
            df.time_firstswitch[idx] = df.trial[idx]
            switch_counters += 1
            switch_type = SWITCH_TYPES[HTRANS[df.hmm_strat[idx-1], df.hmm_strat[idx]]]
            around_switch_counts!(df, idx, "firstsw", switch_type, persev_choices)
            df.switch_count[idx] = switch_counters[1]
        elseif df.hmm_switch[idx] == 1
            switch_counters += 1
            df.switch_count[idx] = switch_counters
            df.othersw_type[idx] = SWITCH_TYPES[HTRANS[df.hmm_strat[idx-1], df.hmm_strat[idx]]]
        end
        if !goodsw && df.goodswitch[idx] == 1
            goodsw = true
            df.time_goodswitch[idx] = df.trial[idx]
            df.goodswitch[idx] = 1
            switch_type = SWITCH_TYPES[HTRANS[df.hmm_strat[idx-1], df.hmm_strat[idx]]]
            around_switch_counts!(df, idx, "goodsw", switch_type, persev_choices)
        end
        if df.random_switch[idx] == 1
            df.random_switch[idx] = 1
            around_switch_counts!(df, idx, "randomsw", df.othersw_type[idx-1], persev_choices)
        end
    end    
    return df
end

            





















