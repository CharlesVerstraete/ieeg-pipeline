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



# subject = 25
complete_beh_crit = DataFrame()
complete_beh_nocrit = DataFrame()
count_prop_remove = Dict()
full_beh = DataFrame()
for subject in [2, 3, 4, 5, 8, 9, 12, 14, 16, 18, 19, 20, 23, 25, 28]
    # subject = 12
    f_sub = @sprintf("%03d", subject)
    file_path = string(DATA_PATH, "/beh", "/sub-$(f_sub)_task-stratinf_beh.tsv")
    data = CSV.read(file_path, DataFrame)
    data.subject .= subject
    println("Data loaded from $(file_path)")
    
    add_criterion!(data)
    cleaned_data = filter(x -> (x.trial_succeed .== 1), data)
    filter!(x -> x.breakblock .== 0, cleaned_data)
    filter!(x -> x.after_break .== 0, cleaned_data)
    filter!(x -> x.training .!= 1, cleaned_data)  
    filter!(x -> ~x.criterion, cleaned_data)
    # epis = combine(groupby(cleaned_data, :epis), nrow)
    # epis_nocrit = epis[epis.nrow .> 45 , :epis]
    # # filter!(x -> x.post_criterion .!= 1, cleaned_data)
    # filter!(x -> x.post_criterion .!= 2, cleaned_data)
    recount_trials!(cleaned_data)
    add_before!(cleaned_data)
    cleaned_data[ismissing.(cleaned_data.is_stimstable), :is_stimstable] .= -1
    # nocrit_set = Set(epis_nocrit)
    # tmp_crit = cleaned_data[[!(e in nocrit_set) for e in cleaned_data.epis], :]
    # tmp_nocrit = cleaned_data[[e in nocrit_set for e in cleaned_data.epis], :]
    full_beh = vcat(full_beh, cleaned_data)
    # complete_beh_crit = vcat(complete_beh_crit, tmp_crit)
    # complete_beh_nocrit = vcat(complete_beh_nocrit, tmp_nocrit)
    # count_prop_remove[subject] = length(epis_nocrit) / maximum(cleaned_data.epis)
end

n_epis_bysubject = combine(groupby(full_beh, [:subject]), :epis => maximum)


complete_beh_crit
test = unique(complete_beh_nocrit[:, [:epis, :subject, :is_partial]], [:epis, :subject, :is_partial])
# test = unique(complete_beh_nocrit[:, [:epis, :subject]], [:epis, :subject])
# count = combine(groupby(test, [:subject]), nrow)
count = combine(groupby(test, [:subject, :is_partial]), nrow)



# First make sure the subject column is consistently typed in both DataFrames
count_ratios = leftjoin(count, n_epis_bysubject, on = :subject)
count_ratios.ratio = count_ratios.nrow ./ count_ratios.epis_maximum

@df count_ratios groupedbar(string.(:subject), :ratio, group = :is_partial, legend = true, ylim = (0, 1))







before_pres_combined = combine(groupby(full_beh, [:before_pres, :next_stable]), 
    :correct => mean => :correct_mean,
    :correct => (x -> std(x)/sqrt(length(x))) => :correct_sem)
filter!(x -> x.before_pres .!= 0, before_pres_combined)
filter!(x -> x.before_pres .>= -3, before_pres_combined)

after_pres_combined = combine(groupby(full_beh, [:stim_pres, :is_stimstable]), 
    :correct => mean => :correct_mean,
    :correct => (x -> std(x)/sqrt(length(x))) => :correct_sem)
# filter!(x -> x.stim_pres .<= 20, after_pres_combined)



plateau = mean(mean(full_beh[full_beh.post_criterion .!= 0, :].correct))
# Create a single plot
p = plot(xlabel="Stimulus Presentations", ylabel="Proportion Correct", 
         ylim=(0, 1), legend=:bottomright, size=(800, 500))

# Add plateau line across entire plot
hline!([plateau], color=:gray, linewidth=1, linestyle=:dash, 
        alpha=0.5, label=false)

# Add vertical line at x=0 to separate before/after
vline!([0], color=:black, linewidth=1, linestyle=:dash, label=false)

# Plot after_pres data with error bars
for (i, group) in enumerate(groupby(after_pres_combined, [:is_stimstable]))
    is_stim = group.is_stimstable[1]
    @df group plot!(p, :stim_pres, :correct_mean, 
                    ribbon=:correct_sem, fillalpha=0.2,
                    linewidth=3, color=switchcode_map[is_stim], 
                    label=condition_names[is_stim])
end

# Plot before_pres data with error bars
for (i, group) in enumerate(groupby(before_pres_combined, [:next_stable]))
    is_stim = group.next_stable[1]
    @df group plot!(p, :before_pres, :correct_mean, 
                    ribbon=:correct_sem, fillalpha=0.2,
                    linewidth=3, color=switchcode_map[is_stim],
                    linestyle=:dash, label=nothing)  # No label for before to avoid duplicates
end

display(p)

