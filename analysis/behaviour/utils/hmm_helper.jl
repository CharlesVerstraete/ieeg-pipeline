function add_switch!(df::DataFrame, subject::Int = 0, save_path::String = "")
    # Load and process latent states
    if subject > 0
        strat_map = load_latent_states(save_path, subject)
        df.hmm_strat .= mode.(eachrow(strat_map))
    end
    initialize_switch_columns!(df)
    
    # # Process each episode
    # for episode_df in groupby(df, :epis)
    #     process_episode!(episode_df)
    # end

    # Label switch types
    # label_switch_types!(df)
    return df
end

function load_latent_states(save_path, subject::Int)
    latent_file = string(save_path, "sub-$(@sprintf("%03d", subject))_latent.jld2")
    latent = load(latent_file)["latent"]
    return hcat(latent...)
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

function forward_counters!(df::DataFrame, switch_idx::Int, colname::String, persev_choice::Int, explor_choice::Int, current_strat::Int)
    j = switch_idx
    counter_stim = zeros(Int, 3)
    trial_counter = 0
    trial_colname = string(colname, "_trial")
    pres_colname = string(colname, "_pres")
    while (minimum(counter_stim) < 8) && (j < nrow(df)) && (df[j, :hmm_strat] == current_strat)
        counter_stim[df[j, :stim]] += 1
        df[j, pres_colname] = counter_stim[df[j, :stim]]
        trial_counter += 1
        df[j, trial_colname] = trial_counter
        df[j, :persev_hmm] = persev_choice == df[j, :choice]
        df[j, :explor_hmm] = explor_choice == df[j, :choice]
        j += 1
    end
    return df, j, trial_counter
end

function backward_counters!(df::DataFrame, switch_idx::Int, colname::String, persev_choice::Int, explor_choice::Int, prev_strat::Int)
    i = switch_idx - 1
    counter_stim = zeros(Int, 3)
    trial_counter = 0
    trial_colname = string(colname, "_trial")
    pres_colname = string(colname, "_pres")
    while (abs(minimum(counter_stim)) < 8) && (i > 0) && (df[i, :hmm_strat] == prev_strat)
        counter_stim[df[i, :stim]] -= 1
        df[i, pres_colname] = counter_stim[df[i, :stim]]
        trial_counter += 1
        df[i, trial_colname] = -trial_counter
        df[i, :persev_hmm] = persev_choice == df[i, :choice]
        df[i, :explor_hmm] = explor_choice == df[i, :choice]
        i -= 1
    end
    return df, i, trial_counter
end
    
function around_switch_counts!(df::DataFrame, switch_idx::Int, colname::String)
    current_strat = df[switch_idx, :hmm_strat]
    if switch_idx > 1
        prev_strat = df[switch_idx-1, :hmm_strat]
    else
        prev_strat = -1
    end
    persev_choice = df[switch_idx, :persev_choice]
    explor_choice = df[switch_idx, :explor_choice]
    df, j, trial_counter = forward_counters!(df, switch_idx, colname, persev_choice, explor_choice, current_strat)
    df, i, trial_counter = backward_counters!(df, switch_idx, colname, persev_choice, explor_choice, prev_strat)
end



function process_switch!(df::DataFrame)
    initialize_switch_columns!(df)
    firstsw = false
    goodsw = false
    switch_counters = 0

    for idx in 2:nrow(df)
        if df.trial[idx] == 1
            switch_counters = 0
            firstsw = false
            goodsw = false
        end
        if !firstsw && df.hmm_switch[idx] == 1
            firstsw = true
            df.firstswitch[idx] = 1
            df.time_firstswitch[idx] = df.trial[idx]
            switch_counters += 1
            around_switch_counts!(df, idx, "firstsw")
            df.switch_count[idx] = switch_counters[1]
            df.firstsw_type[idx] = SWITCH_TYPES[HTRANS[df.hmm_strat[idx-1], df.hmm_strat[idx]]]
        elseif df.hmm_switch[idx] == 1
            switch_counters += 1
            df.switch_count[idx] = switch_counters
            df.othersw_type[idx] = SWITCH_TYPES[HTRANS[df.hmm_strat[idx-1], df.hmm_strat[idx]]]
        end
        if !goodsw && df.goodswitch[idx] == 1
            goodsw = true
            df.time_goodswitch[idx] = df.trial[idx]
            df.goodswitch[idx] = 1
            around_switch_counts!(df, idx, "goodsw")
            df.goodsw_type[idx] = SWITCH_TYPES[HTRANS[df.hmm_strat[idx-1], df.hmm_strat[idx]]]
        end
        if df.random_switch[idx] == 1
            df.random_switch[idx] = 1
            around_switch_counts!(df, idx, "randomsw")
            df.randomsw_type[idx] = df.othersw_type[idx-1]
        end
    end    
    return df
end

            







a

















# function process_episode!(episode_df::SubDataFrame)
#     # Find first switch
#     switch_idx = find_first_switch(episode_df)
    
#     # If switch found, process pre/post switch trials
#     if switch_idx < nrow(episode_df)
#         episode_df[switch_idx, :time2switch] = switch_idx
#         episode_df[switch_idx, :firstswitch] = 1
#         update_switch_counters!(episode_df)
#     end
# end

# function find_first_switch(df::SubDataFrame)
#     idx = 1
#     while (idx < nrow(df)) && (df[idx, :hmm_switch] != 1)
#         idx += 1
#     end
#     return idx
# end

# function update_switch_counters!(df::SubDataFrame)
#     for t in 1:nrow(df)
#         if df[t, :firstswitch] == 1
#             update_pre_switch_counts!(df, t)
#             update_post_switch_counts!(df, t)
#         end
#     end
# end


# function update_post_switch_counts!(df::SubDataFrame, switch_idx::Int)
#     counter = zeros(Int, 3)
#     j = switch_idx + 1
#     while minimum(counter) < MIN_OBSERVATIONS && (j < nrow(df))
#         counter[df[j, :stim]] += 1
#         df[j, :post_hmmsw_pres] = counter[df[j, :stim]]
#         j += 1
#     end
# end

# function label_switch_types!(df::DataFrame)
#     switch_type = ""
#     random_type = ""
#     for t in 2:nrow(df)
#         if df[t, :firstswitch] == 1
#             switch_type = SWITCH_TYPES[HTRANS[df.hmm_strat[t-1], df.hmm_strat[t]]]
#             random_type = get_random_type(df.hmm_strat[t-1], df.hmm_strat[t])
#         end
#         df.switch_type[t] = switch_type
#         df.random_type[t] = random_type
#     end
# end

# function get_random_type(prev_state::Int, curr_state::Int)::String
#     if curr_state == RANDOM_STATE
#         out =  "to_random"
#     elseif prev_state == RANDOM_STATE
#         out = "from_random"
#     else
#         out = ""
#     end
#     return out
# end