function criterion_update(t, t_epis, state, corrects, good_count, success, post_perf, finished_episode)
    if t_epis > 8
        success = sum(corrects[t-8:t])
        good_count[state] = (good_count[state] + corrects[t]) * corrects[t]
    end
    perf_criterion = (success >= 5) && (sum(good_count .> 1) == 3)
    if perf_criterion || post_perf > 0
        post_perf += 1
    end
    finished_episode = post_perf > 5
    return good_count, success, post_perf, finished_episode
end

function add_criterion!(df)
    df[!, :criterion] .= false
    df[!, :post_criterion] .= 0
    df[!, :finished] .= false
    nt = nrow(df)
    corrects = df.correct
    good_count = zeros(Int, 3)
    success = 0
    post_perf = 0
    finished_episode = false
    t_epis = 0
    for t in 1:nt
        if (df.new_block[t] == 1) || (df.trial[t] == 1)
            good_count = zeros(Int, 3)
            success = 0
            post_perf = 0
            finished_episode = false
            t_epis = 0
        end
        state = df.stim[t]
        good_count, success, post_perf, finished_episode = criterion_update(t,t_epis, state, corrects, good_count, success, post_perf, finished_episode)
        df.criterion[t] = post_perf == 1
        if t > 1
            df.criterion[t-1] = df.criterion[t]
        end
        df.post_criterion[t] = post_perf
        df.finished[t] = finished_episode
        t_epis += 1
    end
    return df
end

function recount_trials!(df)
    nt = nrow(df)
    counter_trial = 0
    counter_stim = zeros(Int, 3)
    df.new_block[1] = 1
    state = df.stim[1]
    counter_stim[state] += 1
    counter_trial += 1
    df.trial[1] = counter_trial
    df.stim_pres[1] = counter_stim[state]
    for t in 2:nt
        if (df.new_block[t] == 1) || (df.epis[t] != df.epis[t-1])
            df.new_block[t] = 1
            counter_trial = 0
            counter_stim = zeros(Int, 3)
        end
        state = df.stim[t]
        counter_stim[state] += 1
        counter_trial += 1
        df.trial[t] = counter_trial
        df.stim_pres[t] = counter_stim[state]
        if counter_trial > 1
            df.new_block[t] = 0
        end
    end
    return df
end

function add_before!(df)
    df.before_pres .= 0
    df.before_trial .= 0
    df.next_stable .= fill(-1, nrow(df))    
    for i in findall(df.new_block .== 1)
        if df.epis[i] <= 1
            continue
        end        
        counter_stim = zeros(Int, 3)
        count_trial = 0        
        j = i - 1
        while j >= 1 && (df.new_block[j] .!= 1)
            state = df.stim[j]
            counter_stim[state] -= 1
            count_trial -= 1
            
            df.before_pres[j] = counter_stim[state]
            df.before_trial[j] = count_trial            
            if df.is_partial[i] == 1
                df.next_stable[j] = Int(df.stim[j] == df.who_stable[i])
            end            
            j -= 1
        end
    end
    return df
end

function add_persev_explor!(df)
    rule = Array(df[:, 1:3])
    
    df.persev_choice .= 0
    df.explor_choice .= 0

    past_rule = [0 0 0]
    for i in 1:nrow(df)
        if df[i, :epis] > 0
            if df[i, :new_block] == 1
                past_rule = rule[i-1, :]
            end
            df.persev_choice[i] = past_rule[df.stim[i]]
        end
    end
    
    explor = [(filter!(e -> !(e in [a,b]), [1,2,3])) for (a,b) in zip(df.persev_choice,  df.correct_choice)]
    # df.explor_choice = [length(x) == 1 ? x[1] : 0 for x in explor]
    df.explor_choice = explor
    df.persev .= df.persev_choice .== df.choice
    # df.explor = df.explor_choice .== df.choice
    df.explor .= [df.choice[i] in explor[i] for i in 1:nrow(df)]

    return df
end