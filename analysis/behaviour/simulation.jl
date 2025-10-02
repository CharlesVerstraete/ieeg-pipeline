# julia 1.10
# -*- coding: utf-8 -*-

"""
Description:
    main script to run the analysis

Author :
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2025-05
"""

include("librairy.jl")

ns, na = 3, 3
sub_list = [2, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29]
DATA_PATH = "/Users/charles.verstraete/Documents/w3_iEEG/behaviour"
subject = sub_list[1]
f_sub = @sprintf("%03d", subject)

# model_type = "transition"
all_sub_df_simu = DataFrame()
all_sub_df = DataFrame()

# file_path = string(DATA_PATH, "/hmm", "/sub-$(f_sub)_task-stratinf_beh-hmm.csv")
# data = CSV.read(file_path, DataFrame, delim = ",")
# data.rule = [findfirst(x -> x == collect(rule), HIDDEN_STATES) for rule in eachrow(data[:, ["rule_1", "rule_2", "rule_3"]])]


for subject in sub_list
# subject = sub_list[1]
# subejct = 3
    f_sub = @sprintf("%03d", subject)
    file_path = string(DATA_PATH, "/hmm", "/sub-$(f_sub)_task-stratinf_beh-hmm.csv")
    data = CSV.read(file_path, DataFrame, delim = ",")
    data.rule = [findfirst(x -> x == collect(rule), HIDDEN_STATES) for rule in eachrow(data[:, ["rule_1", "rule_2", "rule_3"]])]
    # CSV.write(string(DATA_PATH, "/hmm", "/sub-$(f_sub)_task-stratinf_beh-hmm.csv"), data)
    add_criterion!(data)
    add_persev_explor!(data)
    # data.subject .= subject
    clean_data = filter(x -> (x.trial_succeed .== 1), data)
    recount_trials!(clean_data)
    clean_data.prev_fb .= vcat(0, clean_data.fb[1:end-1])
    process_switch!(clean_data)
    filter!(x -> x.training .== 0, clean_data)
    clean_data[ismissing.(clean_data.is_stimstable), :is_stimstable] .= 2
    clean_data[clean_data.rt .> 5, :rt] .= NaN
    clean_data[clean_data.rt .< 0, :rt] .= NaN

    rt_values = clean_data.rt[.!isnan.(clean_data.rt)]
    mean_rt = mean(rt_values)
    std_rt = std(rt_values)

    clean_data.rt_zscore .= 0.0
    valid_rt_mask = .!isnan.(clean_data.rt)
    clean_data[valid_rt_mask, :rt_zscore] .= (clean_data[valid_rt_mask, :rt] .- mean_rt) ./ std_rt 
    
    all_sub_df = vcat(all_sub_df, clean_data)
end




plot(all_sub_df.rt_zscore)








test = combine(groupby(all_sub_df, [:randomsw_trial, :is_stimstable]), :explor => mean => :mean)
test_pos = test[test.randomsw_trial .> 0, :]
test_pos = test_pos[test_pos.randomsw_trial .< 10, :]
test_neg = test[test.randomsw_trial .< 0, :]
test_neg = test_neg[test_neg.randomsw_trial .> -10, :]
@df test_neg plot(:randomsw_trial, :mean, group = :is_stimstable, linewidth = 2, palette = :Dark2_3, ylims = (0, 1), label = false)
@df test_pos plot!(:randomsw_trial, :mean, group = :is_stimstable, linewidth = 2, palette = :Dark2_3)



test = combine(groupby(all_sub_df, [:goodsw_trial]), :rt_zscore => mean => :mean)
test_pos = test[test.goodsw_trial .> 0, :]
test_pos = test_pos[test_pos.goodsw_trial .< 8, :]
test_neg = test[test.goodsw_trial .< 0, :]
test_neg = test_neg[test_neg.goodsw_trial .> -8, :]
@df test_neg bar(:goodsw_trial, :mean,  linewidth = 2, color = :blue, ylims = (-0.1, 0.2), label = false, alpha = 0.3)
@df test_pos bar!(:goodsw_trial, :mean, linewidth = 2, color = :blue, alpha = 0.3, label = "good switch")



test = combine(groupby(all_sub_df, [:firstsw_trial]), :rt_zscore => mean => :mean)
test_pos = test[test.firstsw_trial .> 0, :]
test_pos = test_pos[test_pos.firstsw_trial .< 8, :]
test_neg = test[test.firstsw_trial .< 0, :]
test_neg = test_neg[test_neg.firstsw_trial .> -8, :]
@df test_neg bar!(:firstsw_trial, :mean,  linewidth = 2, color = :red, ylims = (-0.2, 0.3), label = false, alpha = 0.3)
@df test_pos bar!(:firstsw_trial, :mean, linewidth = 2, color = :red, alpha = 0.3, label = "first switch")

test = combine(groupby(all_sub_df, [:randomsw_trial]), :rt_zscore => mean => :mean)
test_pos = test[test.randomsw_trial .> 0, :]
test_pos = test_pos[test_pos.randomsw_trial .< 8, :]
test_neg = test[test.randomsw_trial .< 0, :]
test_neg = test_neg[test_neg.randomsw_trial .> -8, :]
@df test_neg bar!(:randomsw_trial, :mean,  linewidth = 2, color = :green, ylims = (-0.1, 0.4), label = false, alpha = 0.3)
@df test_pos bar!(:randomsw_trial, :mean, linewidth = 2, color = :green, alpha = 0.3, label = "random switch")



test = combine(groupby(all_sub_df, [:epis, :is_partial, :subject]), :switch_count => maximum => :count)
@df test histogram(:is_partial, :count, group = :is_partial, palette = :Dark2_3, alpha = 0.5, nbins = 8, label = false)


histogram(all_sub_df[all_sub_df.time_goodswitch .!= 0, :time_goodswitch], bins = 20)

combine(groupby(all_sub_df, [:epis, :is_partial, :subject]), :switch_count => maximum => :count)

all_sub_df[all_sub_df.goodswitch .!= 0, :switch_count]


dotplot(all_sub_df[all_sub_df.goodswitch .!= 0, :switch_count])


all_sub_df[all_sub_df.firstsw_trial .!= 0, :]

good_sw = all_sub_df[all_sub_df.goodswitch .!= 0, end-10:end]

test = combine(groupby(good_sw, [:subject]), :switch_count => mean => :mean)
# combine(groupby(test, [:subject]), :mean => sum => :total_bysubject)
histogram(test.mean, nbins = 10)
@df test violin(:is_partial, :mean, group = :is_partial, alpha = 0.5)

for subject in sub_list
# subject = sub_list[1]
# subejct = 3
f_sub = @sprintf("%03d", subject)
file_path = string(DATA_PATH, "/hmm", "/sub-$(f_sub)_task-stratinf_beh-hmm.csv")
data = CSV.read(file_path, DataFrame, delim = ",")
data.rule = [findfirst(x -> x == collect(rule), HIDDEN_STATES) for rule in eachrow(data[:, ["rule_1", "rule_2", "rule_3"]])]
CSV.write(string(DATA_PATH, "/hmm", "/sub-$(f_sub)_task-stratinf_beh-hmm.csv"), data)
end
# # add_criterion!(data)
# # add_persev_explor!(data)
# # data.subject .= subject
# # clean_data = filter(x -> (x.trial_succeed .== 1), data)
# # recount_trials!(clean_data)
# # add_switch!(clean_data, subject, string(DATA_PATH, "/chains", "/", "HMM", "/"))
# all_sub_df = vcat(all_sub_df, data)
# # end

# println("Subject: ", subject)
# empty_path = string(DATA_PATH, "/empty", "/sub-$(f_sub)_task-stratinf_beh_empty.csv")
# empty = CSV.read(empty_path, DataFrame, delim = ",")

chain_df = make_chaindf(subject, "si", type = "", burn = 1000, temp = 5.0)
param_df = get_params(chain_df, mode = "sample", resample = 10)
param_df[!, :model_type] .= model_type
complete_simu = DataFrame()
for rank in 1:10

param = get_param_set(param_df[rank, :])
simu_df = init_simu_df(data)
simu_df = run_simulation(param, na, ns, simu_df)
simu_df[!, :switch] .= vcat([0], diff(simu_df.selected_strategy) .!= 0)
# save_path = string(DATA_PATH, "/simu", "/$model_type", "/sub-$(f_sub)_task-stratinf_sim-free.csv")
# CSV.write(save_path, simu_df)
    complete_simu = vcat(complete_simu, simu_df)
end
# scatter(complete_simu.rpe, complete_simu.update_counterfactual)












# simu_df[!, :rank] .= rank
    # complete_simu = vcat(complete_simu, simu_df)
# end
# combined = combine(groupby(complete_simu, [:stim_pres]), :correct => mean => :mean)
# p = @df combined plot(:stim_pres, :mean, linewidth = 2, plot_title = subject, label = "Simulation")
# combined = combine(groupby(data[data.trial_succeed .== 1, :], [:stim_pres]), :correct => mean => :mean)
# @df combined plot!(p, :stim_pres, :mean, linewidth = 2, label = "Real data")
# display(p)
# save_path = string(DATA_PATH, "/simu_rank", "/sub-$(f_sub)_task-stratinf_sim-ranked.csv")
# CSV.write(save_path, complete_simu)
all_sub_df_simu = vcat(all_sub_df_simu, complete_simu)
end

scatter(all_sub_df_simu.rpe, all_sub_df_simu.update_counterfactual, ms = 0.5, msc = 0, alpha = 0.5)

density(all_sub_df_simu.joint_counterfactual)


all_sub_df_simu[:, :joint_group] .= ""
all_sub_df_simu[all_sub_df_simu[:, :joint_counterfactual] .< 0.04, :joint_group] .= "low"
all_sub_df_simu[all_sub_df_simu[:, :joint_counterfactual] .>= 0.04, :joint_group] .= "mid"
all_sub_df_simu[all_sub_df_simu[:, :joint_counterfactual] .> 0.17, :joint_group] .= "high"





test_combined = deepcopy(all_sub_df_simu)
test_combined.rpe .= round.(test_combined.rpe, digits = 5)
# test_combined.update_counterfactual .= round.(test_combined.update_counterfactual, digits = 2)
# test_combined = test_combined[(test_combined.update_counterfactual .>= 0.1) .|| (test_combined.update_counterfactual .<= -0.1), :]
combined = combine(groupby(test_combined, [:joint_group, :rpe]), :update_counterfactual => mean => :mean)




p = []
for joint_group in ["low", "mid", "high"]
    test = combined[combined.joint_group .== joint_group, :]
    p_ = scatter(test.rpe, test.mean, ms = 0.2, msc = 0, title = joint_group, ylim = (-1, 1))
    push!(p, p_)
end
plot(p..., layout = (3, 1), size = (400, 1200), xlabel = "rpe", ylabel = "update_counterfactual")

scatter(all_sub_df_simu.rpe, all_sub_df_simu.update_counterfactual, ms = 0.2, msc = 0)

names(all_sub_df_simu)
scatter(all_sub_df_simu[all_sub_df_simu[:, :joint_counterfactual] .< 0.2, :][:, :rpe], all_sub_df_simu[all_sub_df_simu[:, :joint_counterfactual] .< 0.2, :][:, :update_counterfactual], ms = 0.5, msc = 0)
all_sub_df_simuall_sub_df_simu[:, :update_counterfactual]
complete_simu
all_sub_df_simu[ismissing.(all_sub_df_simu.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df_simu, [:stim_pres, :is_stimstable]), :correct => mean)
filter!(x -> x.is_stimstable .!= -1, combined)
filter!(x -> x.stim_pres .<= 12, combined)
p1 =  @df combined plot(:stim_pres, :correct_mean, linewidth = 2, group = :is_stimstable, ylim = (0, 1), legend = false, title = "entropy")

all_sub_df[ismissing.(all_sub_df.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df[all_sub_df.trial_succeed .== 1, :], [:stim_pres, :is_stimstable]), :correct => mean)
filter!(x -> x.is_stimstable .!= -1, combined)
filter!(x -> x.stim_pres .<= 12, combined)
p3 = @df combined plot(:stim_pres, :correct_mean, linewidth = 2, group = :is_stimstable, ylim = (0, 1), legend = false, title = "real data")

plot(p1, p3, layout = (1, 2), size = (1200, 400))



all_sub_df_simu[ismissing.(all_sub_df_simu.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df_simu, [:stim_pres, :is_stimstable]), :rpe => mean => :mean)
filter!(x -> x.is_stimstable .!= -1, combined)
filter!(x -> x.post_criterion .> 1, combined)
combined.post_criterion .-= 7
@df combined plot(:post_criterion, :mean, linewidth = 2, group = :is_stimstable, legend = false)


all_sub_df_simu[ismissing.(all_sub_df_simu.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df_simu, [:stim_pres, :is_stimstable]), :reliability => mean => :mean)
filter!(x -> x.is_stimstable .!= -1, combined)
filter!(x -> x.stim_pres .<= 12, combined)
@df combined plot(:stim_pres, :mean, linewidth = 2, group = :is_stimstable, legend = false)


all_sub_df[ismissing.(all_sub_df.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df, [:post_criterion, :is_stimstable]), :correct => mean => :mean)
filter!(x -> x.is_stimstable .!= -1, combined)
filter!(x -> x.post_criterion .> 1, combined)
filter!(x -> x.post_criterion .< 6, combined)
combined.post_criterion .-= 7
@df combined plot(:post_criterion, :mean, linewidth = 2, group = :is_stimstable, legend = false)


all_sub_df[ismissing.(all_sub_df.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df, [:stim_pres, :is_stimstable]), :correct => mean => :mean)
filter!(x -> x.is_stimstable .!= -1, combined)
filter!(x -> x.stim_pres .<= 12, combined)
@df combined plot!(:stim_pres, :mean, linewidth = 2, group = :is_stimstable, ylim = (0,1), legend = false)




all_sub_df_simu[~ismissing.(all_sub_df_simu.switch), :]

all_sub_df_simu

model_type = "transition"


# for subject in sub_list
chn = get_chain(subject, "SI", type = model_type)
display(plot(chn, plot_title = subject))
# end


chn_df = DataFrame(chn)
scatter(chn_df[:, "ω[4]"], chn_df[:, "ω[5]"])


data


all_sub_df_simu_transition = DataFrame()
all_sub_df_simu_selection = DataFrame()
all_sub_df_simu_entropy = DataFrame()

# all_sub_df = DataFrame()
for subject in sub_list
# subject = sub_list[1]
f_sub = @sprintf("%03d", subject)
print("Subject: ", subject)
file_path = string(DATA_PATH, "/beh", "/sub-$(f_sub)_task-stratinf_beh.tsv")
data = CSV.read(file_path, DataFrame, delim = ",")
data.subject .= subject
# data.rule = [findfirst(x -> x == collect(rule), HIDDEN_STATES) for rule in eachrow(data[:, ["rule_1", "rule_2", "rule_3"]])]
clean_data = filter(x -> (x.trial_succeed .== 1), data)
# all_sub_df = vcat(all_sub_df, data)
empty_path = string(DATA_PATH, "/empty", "/sub-$(f_sub)_task-empty_beh.tsv")
empty = CSV.read(empty_path, DataFrame, delim = ",")

chain_df = make_chaindf(subject, "SI",type = "greedy")
param_df = get_params(chain_df, mode = "sample", resample = 10)


complete_simu = DataFrame()
for rank in 1:10
# ranked = 1
    param = get_param_set(param_df[ranked, :])
    simu_df = deepcopy(empty)
    simu_df[!, :selected_strategy] .= 0
    simu_df[!, :entropy] .= 0.0
    simu_df[!, :switch] .= 0
    simu_df[!, :reliability] .= 0.0
    simu_df[!, :action_value] .= 0.0
    Γ, tmp_gamma, Π, Φ, counter, ϕ₀, q, P = init_internals(param.ρ, param.ω₀, na, ns)
    entropy = (-sum(Φ .* log.(Φ)))/log(NSETS+1)
    selected_strategy = rand(Categorical(softmax(param.β*@views(Φ))))
    # selected_strategy = rand(Categorical(softmax(@views(Φ))))
    idx_newblock = findall(empty.new_block .== 1)
    for (i, epis) in enumerate(idx_newblock[1:end-1])
    # i = 1
    # epis = idx_newblock[i]
        idx_epis = idx_newblock[i]
        good_count = zeros(Int, 3)
        success = 0
        post_perf = 0
        finished_episode = false
        stim_count = zeros(Int, 3)
        while (idx_epis < idx_newblock[i+1]) && !finished_episode
            s = simu_df[idx_epis, :stim]
            stim_count[s] += 1
            simu_df[idx_epis, :stim_pres] = stim_count[s]
            trap = Bool(simu_df[idx_epis, :trap])
            correct_choice = simu_df[idx_epis, :correct_choice]
            trial = simu_df[idx_epis, :trial]
            entropy = (-sum(Φ .* log.(Φ)))/log(NSETS+1)
            a, correct, r, selected_strategy, P = get_action(entropy, selected_strategy,correct_choice,trap, s, param.β, param.ϵ, Φ)
            # selected_strategy = argmax(Φ)
            # q = compute_q!(Φ, q, na)
            # P .= custom_softmax!(P, @views(q[:,s]), param.β, param.ϵ)
            # a = rand(Categorical(P))
            # correct = a == correct_choice
            # r = trap ? (a != correct_choice) : (a == correct_choice)
            simu_df[idx_epis, :choice] = a
            simu_df[idx_epis, :correct] = correct
            simu_df[idx_epis, :fb] = r
            simu_df[idx_epis, :trial_succeed] = true
            simu_df[idx_epis, :selected_strategy] = selected_strategy
            simu_df[idx_epis, :entropy] = entropy
            simu_df[idx_epis, :reliability] = Φ[selected_strategy]
            # simu_df[idx_epis, :action_value] = q[a, s]
            simu_df[idx_epis, :action_value] = P[a]

            if idx_epis > 1
                simu_df[idx_epis, :switch] = selected_strategy != simu_df[idx_epis-1, :selected_strategy]
            end
            good_count, success, post_perf, finished_episode = criterion_update(idx_epis, trial, s, simu_df[:, :correct], good_count, success, post_perf, finished_episode)
            update_memory_trace!(counter, Γ, tmp_gamma, Φ, param.ω₀, ϕ₀)
            update_prior!(Π, Φ, param.ω, param.ωᵣ)
            update_reliabilities!(Π, Φ, Γ, param.ω, param.ωᵣ, a, s, r, param.ρ, na)
            idx_epis += 1
        end
    end
    simu_df[!, :rank] .= rank
    complete_simu = vcat(complete_simu, simu_df[simu_df.trial_succeed .== 1, :])
end

# all_sub_df_simu_transition = vcat(all_sub_df_simu_transition, complete_simu)
all_sub_df_simu_selection = vcat(all_sub_df_simu_selection, complete_simu)
# all_sub_df_simu_entropy = vcat(all_sub_df_simu_entropy, complete_simu)
combined = combine(groupby(complete_simu, [:stim_pres]), :correct => mean => :mean)
p = @df combined plot(:stim_pres, :mean, linewidth = 2, plot_title = subject, label = "Simulation")

combined = combine(groupby(data[data.trial_succeed .== 1, :], [:stim_pres]), :correct => mean => :mean)
@df combined plot!(p, :stim_pres, :mean, linewidth = 2, label = "Real data")
display(p)
end

mean(all_sub_df_simu_entropy.correct)
mean(all_sub_df_simu_selection.correct)
mean(all_sub_df_simu_transition.correct)
mean(all_sub_df.correct)
# P = zeros(na)

all_sub_df_simu_selection[ismissing.(all_sub_df_simu_selection.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df_simu_selection, [:stim_pres, :is_stimstable]), :correct => mean)
filter!(x -> x.is_stimstable .!= -1, combined)
filter!(x -> x.stim_pres .<= 12, combined)
p1 = @df combined plot(:stim_pres, :correct_mean, linewidth = 2, group = :is_stimstable, ylim = (0, 1), legend = false, title = "entropy")

all_sub_df_simu_transition[ismissing.(all_sub_df_simu_transition.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df_simu_transition, [:stim_pres, :is_stimstable]), :correct => mean)
filter!(x -> x.is_stimstable .!= -1, combined)
filter!(x -> x.stim_pres .<= 12, combined)
p2 = @df combined plot(:stim_pres, :correct_mean, linewidth = 2, group = :is_stimstable, ylim = (0, 1), legend = false, title = "transition")

all_sub_df[ismissing.(all_sub_df.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df[all_sub_df.trial_succeed .== 1, :], [:stim_pres, :is_stimstable]), :correct => mean)
filter!(x -> x.is_stimstable .!= -1, combined)
filter!(x -> x.stim_pres .<= 12, combined)
p3 = @df combined plot(:stim_pres, :correct_mean, linewidth = 2, group = :is_stimstable, ylim = (0, 1), legend = false, title = "real data")

plot(p1, p2, p3, layout = (1, 3), size = (1200, 400))


all_sub_df_simu_selection[ismissing.(all_sub_df_simu_selection.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df_simu_selection, [:stim_pres, :is_stimstable]), :action_value => mean => :mean)
filter!(x -> x.is_stimstable .!= -1, combined)
p1 = @df combined plot(:stim_pres, :mean, linewidth = 2, group = :is_stimstable, ylim = (0, 1), legend = false, title = "entropy")


# all_sub_df_simu_selection[ismissing.(all_sub_df_simu_selection.is_stimstable), :is_stimstable] .= 2
# combined = combine(groupby(all_sub_df_simu_selection, [:stim_pres, :is_stimstable]), :action_value => mean => :mean)
# filter!(x -> x.is_stimstable .!= -1, combined)
# p2 = @df combined plot(:stim_pres, :mean, linewidth = 2, group = :is_stimstable, ylim = (0, 1), legend = false, title = "selection")


all_sub_df_simu_transition[ismissing.(all_sub_df_simu_transition.is_stimstable), :is_stimstable] .= 2
combined = combine(groupby(all_sub_df_simu_transition, [:stim_pres, :is_stimstable]), :action_value => mean => :mean)
filter!(x -> x.is_stimstable .!= -1, combined)
p3 = @df combined plot(:stim_pres, :mean, linewidth = 2, group = :is_stimstable, ylim = (0, 1), legend = false, title = "transition")

plot(p1, p3, layout = (1, 2), size = (1200, 400))


Q = Matrix(fill(1/na, na, ns))
U = Vector(fill(1/(NSETS+1), NSETS+1))

function get_action(entropy, selected_strategy, correct_choices,trap, s, β,ϵ, Φ, α = nothing)
    # if ~isnothing(α)
    #     if entropy > 0.6
    #         selected_strategy = rand(Categorical(softmax(β*@views(Φ))))
    #     end
    # # selected_strategy = rand(Categorical(softmax(β*@views(Φ))))        
    #     # end
    #     else
        # selected_strategy = rand(Categorical(softmax(β*@views(Φ))))
    # end
    U = fill(1/(NSETS+1), NSETS+1)
    Q = zeros(na, na)
    U .= softmax!(param.β .* Φ)
    for i in 1:NSETS
        Q .+= U[i] .* ALL_SETS[i]
    end
    Q .+= U[end]/na
    P = Q[:, s]
    P .*= (1 - ϵ)
    P .+= ϵ / length(P)
    a = rand(Categorical(Q[:, s]))
    correct = a == correct_choices
    r = trap ? (a != correct_choices) : (a == correct_choices)
    return a, correct, r, selected_strategy, Q
end

add_criterion!(data)
cleaned_data = filter(x -> (x.trial_succeed .== 1), data)


softmax(param.β*@views(Φ))


condition_names = Dict(-1 => "Complete", 0 => "Partial", 1 => "Stable")

q = zeros(3, 3)
q .+= ALL_SETS[selected_strategy]
P = q[:, s]
P .*= 0.5
P .+= (1-0.5) / length(P)





states = cleaned_data.stim
actions = cleaned_data.choice
feedbacks = cleaned_data.fb
subject = 12
chain = get_chain(subject, "SI", "entropy")
plot(chain)













# for subject in [2, 4, 5, 8, 12, 14, 16, 19, 20, 23, 25, 28]
chain = get_chain(subject, "SI", "entropy")
display(plot(chain, plot_title = subject))
# end

chain_df = DataFrame(chain)

max_llh = maximum(chain_df.lp)
w = exp.((chain_df.lp .- max_llh))
w ./= sum(w)
chain_df[:, "wllh"] = w

# ranks = sortperm(sortperm(chain_df.lp, rev=true))
# w = 1.0 ./ ranks
# w ./= sum(w)

param_cols = setdiff(names(chain_df), ["lp", "iteration", "chain", "llh", "wllh"])
weighted_params = DataFrame(Dict(col => mean(chain_df[:,col], Weights(chain_df.wllh)) for col in param_cols))
sampled_param = chain_df[sample(axes(chain_df, 1), Weights(chain_df.wllh), 100), :]

cleaned_data[ismissing.(cleaned_data.is_stimstable), :is_stimstable] .= 2
states = cleaned_data.stim
actions = cleaned_data.choice
feedbacks = cleaned_data.fb

combined = combine(groupby(cleaned_data, [:stim_pres, :is_stimstable]), :correct => mean)
filter!(x -> x.is_stimstable .!= -1, combined)
p = [plot() for _ in 1:3]  # Create 3 empty plots
for (i, group) in enumerate(groupby(combined, [:is_stimstable]))
    p[i] = @df group plot(:stim_pres, :correct_mean, legend = false, linewidth = 2, color = :black, title = "$(group.is_stimstable[1])")
end

complete_test = DataFrame()

for rank in 1:100
# row = param_df[rank, :]
row = param_df[1, :]
ω = collect(row[["ω[1]", "ω[2]", "ω[3]", "ω[4]", "ω[5]"]])
ρ = row["ρ"]
ω₀ = row["ω₀"]
ωᵣ = row["ωᵣ"]
ϵ = row["ϵ"]
β = row["β"]
α = row["α"]
ω_ = softmax(ω)
nt = length(states)
ns, na = 3, 3
Γ, tmp_gamma, Π, Φ, counter, ϕ₀, q, P = init_internals(ρ, ω₀, na, ns)
la = zeros(typeof(ω₀), nt)
actions = zeros(Int, nt)
correct_choices = cleaned_data.correct_choice
trap = Bool.(cleaned_data.trap)
correct = zeros(Int, nt)
entropy_ = (-sum(Φ .* log.(Φ)))/log(NSETS+1)
selected_strategy = zeros(Int, nt)
# entropy_ =  0
selected_strategy[1] = 28
for t = 1:nt
    s = states[t]

    # q = compute_q!(Φ, q, na)
    # P .= custom_softmax!(P, @views(q[:,s]), β, ϵ)
    # entropy_ = (-sum(Φ .* log.(Φ)))/log(NSETS+1)
    if entropy_ > α
    selected_strategy[t] = rand(Categorical(softmax(β*@views(Φ))))
    #     # println("Entropy: ", entropy_)
    else
        selected_strategy[t] = selected_strategy[t-1]
    end
    if selected_strategy[t] == 28
        P .= 1.0 / na
        P .*= (1 - ϵ)
        P .+= ϵ / length(P)
    else
        q .= 0.0
        q .+= ALL_SETS[selected_strategy[t]]
        P = q[:, s]
        P .*= (1 - ϵ)
        P .+= ϵ / length(P)
    end
    # selected_strategy[t] = rand(Categorical(@views(Φ)))
    # q = compute_q!(Φ, q, na)
    # P .= custom_softmax!(P, @views(q[:,s]), β, ϵ)
    actions[t] = a = rand(Categorical(P))
    correct[t] = r = trap[t] ? (actions[t] != correct_choices[t]) : (actions[t] == correct_choices[t])
    
    update_memory_trace!(counter, Γ, tmp_gamma, Φ, ω₀, ϕ₀)
    update_prior!(Π, Φ, ω, ωᵣ)
    update_reliabilities!(Π, Φ, Γ, ω, ωᵣ, a, s, r, ρ, na)
end

test_data = deepcopy(cleaned_data)
test_data[!, :action] = actions
test_data[!, :correct] = correct
test_data[!, :selected_strategy] = selected_strategy
recount_trials!(test_data)
filter!(x -> x.training .!= 1, test_data)
combined = combine(groupby(test_data, [:stim_pres, :is_stimstable]), :correct => mean)
filter!(x -> x.is_stimstable .!= -1, combined)
complete_test = vcat(complete_test, test_data)
# display(@df combined plot(:stim_pres, :correct_mean, legend = false, linewidth = 1, group = :is_stimstable, palette = :auto))
for (i, group) in enumerate(groupby(combined, [:is_stimstable]))
    p[i] = @df group plot!(p[i], :stim_pres, :correct_mean, legend = false, linewidth = 0.5, color = :red, alpha = 0.5)
end


end

plot(p[1], p[2], p[3], layout = (1, 3), size = (1200, 400))
complete_test[!, :switch] = vcat([0], diff(complete_test.selected_strategy) .!= 0)
complete_test[!, :good_strat] = complete_test.rule .== complete_test.selected_strategy
complete_test[!, :explor_test] = complete_test.rule .!= complete_test.selected_strategy

combined = combine(groupby(complete_test, [:trial]), :switch => mean)
@df combined plot(:trial, :switch_mean, legend = false)

combined = combine(groupby(cleaned_data, [:stim_pres, :is_stimstable]), :correct => mean)
@df combined plot(:stim_pres, :correct_mean, legend = false, linewidth = 0.1, group = :is_stimstable, palette = :auto)
for (i, group) in enumerate(groupby(combined, [:is_stimstable]))
    p_ = @df group plot(:stim_pres, :correct_mean, legend = false, linewidth = 2, color = :black)
    push!(p[i], p_)
end
a
@views(Φ)

test = zeros(10000)
for i = 1:10000
    test[i] = rand(Categorical(softmax!(β.*@views(Φ))))
end

histogram(test, bins = 1:28)
epsilon_greedy!(P, ϵ)

P = zeros(na)
P .+= ALL_SETS[1][:, 1]
P .*= (1 - ϵ)
P .+= ϵ / length(P)
s = 2

P .= (1 - ϵ) * q_[:,s] .+ ϵ/na

a
# subject = parse(Int, ENV["SUBJECT"])
# nsample = parse(Int, ENV["N_SAMPLES"])
# nchains = parse(Int, ENV["N_CHAINS"])

# f_sub = @sprintf("%03d", subject)
# file_path = string(DATA_PATH, "/beh", "/sub-$(f_sub)_task-stratinf_beh.tsv")
# data = CSV.read(file_path, DataFrame)
# println("Data loaded from $(file_path)")

# cleaned_data = filter(x -> (x.miss .== 0) && (x.choice .!= 0), data)

# sampler = Gibbs(NUTS(100, 0.65, init_ϵ = 0.05, :ρ, :ω₀, :ωᵣ, :ϵ, :β), ESS(:ω))
# states = cleaned_data.stim
# actions = cleaned_data.choice
# feedbacks = cleaned_data.fb

# mdl = stratinf_transition(states, actions, feedbacks)
# chain_data = sample(mdl, sampler, MCMCThreads(), nsample, nchains)
# chains_params = MCMCChains.get_sections(chain_data, :parameters)
# latent = generated_quantities(mdl, chains_params)
# pllh = collect(values(pointwise_loglikelihoods(mdl, chains_params)))

# chain_file = string(DATA_PATH, "/chains", "/SI", "/sub-$(f_sub)_chains.jls")
# latent_file = string(DATA_PATH, "/chains", "/SI", "/sub-$(f_sub)_latent.jld2")
# pllh_file = string(DATA_PATH, "/chains", "/SI", "/sub-$(f_sub)_pllh.jld2")

# serialize(chain_file, chain_data)
# JLD2.save(pllh_file, "pllh", pllh)
# JLD2.save(latent_file, "latent", latent)

# println("SI completed")


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


