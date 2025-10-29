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

sub_list = [2, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 32]

# chain_hmm_path = joinpath(CHAIN_PATH, "hmm")


for subject in sub_list
    f_sub = @sprintf("%03d", subject)
    println("Processing subject $f_sub")
    file_path = joinpath(DATA_PATH, "hmm", "sub-$(f_sub)_task-stratinf_beh-hmm.csv")
    # file_path = joinpath(DATA_PATH, "beh", "sub-$(f_sub)_task-stratinf_beh.csv")
    df = CSV.read(file_path, DataFrame)
    process_switch!(df)
    CSV.write(file_path, df)

    file_path = joinpath(DATA_PATH, "hmm_w", "sub-$(f_sub)_task-stratinf_beh-hmm-weighted.csv")
    # file_path = joinpath(DATA_PATH, "beh", "sub-$(f_sub)_task-stratinf_beh.csv")
    df = CSV.read(file_path, DataFrame)
    process_switch!(df)
    CSV.write(file_path, df)
    # df.trial_count .= 1:nrow(df)
    # df.rule = [findfirst(x -> x == collect(rule), HIDDEN_STATES) for rule in eachrow(df[:, ["rule_1", "rule_2", "rule_3"]])]
    # df[ismissing.(df.is_stimstable), :is_stimstable] .= 2
    # add_criterion!(df)
    # add_persev_explor!(df)
    # only_succeed = filter(row -> row.trial_succeed == 1, df)
    # recount_trials!(only_succeed)
    # add_before!(only_succeed)
    # only_succeed.prev_fb .= vcat(0, only_succeed.fb[1:end-1])
    # add_switch!(only_succeed, subject, chain_hmm_path)
    # save_path = joinpath(DATA_PATH, "hmm", "sub-$(f_sub)_task-stratinf_beh-hmm.csv")
    # CSV.write(save_path, only_succeed)
end


