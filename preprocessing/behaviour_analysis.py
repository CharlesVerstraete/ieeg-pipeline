
#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Behavioral analysis, plots and statistics
"""

# Import libraries
from preprocessing.config import *
from preprocessing.utils.data_helper import *
from preprocessing.utils.beh_helper import *
from preprocessing.utils.plot_helper import *

import seaborn as sns
import matplotlib.pyplot as plt

# matplotlib.use('Qt5Agg')
plt.style.use('seaborn-v0_8-poster') 


BEH_PATH = "/Users/charles.verstraete/Documents/w3_iEEG/behaviour/hmm"
fileslist = get_fileslist(BEH_PATH, ".csv")

df = get_df(fileslist, True, True, True)

df_test = df[np.isin(df["subject"], [13, 15, 17], invert=True)]

plot_indiv_lineplot(df, "stim_pres", "correct", xlim=15, chance=0.33, ylim=(0, 1))
plot_indiv_histplot(df, ["subject", "epis"])


epis_count_complete = df_test.groupby(["subject", "epis"]).size().reset_index(name="count")
plot_boxplot(epis_count_complete, "subject", "count", hue_var="subject")


epis_count, removed_episodes, synthese_removed = filter_episodes(epis_count_complete, min_count=15, max_count=55)

sns.barplot(data=synthese_removed, x="subject", y="n_total", color="green", alpha=0.5)
sns.barplot(data=synthese_removed, x="subject", y="removed_count", color="red", alpha=0.5)
plt.show()

sns.histplot(data=epis_count, x="count", bins=20, kde=True)
plt.show()

y_var = "correct"
x_var_pre = "before_pres"
x_var_post = "stim_pres"

summary_before, summary_after = get_switch_summary(df_test, x_var_pre, x_var_post, y_var, -5, 15)
avr_perf = summary_before[summary_before["before_pres"] > -4]["mean"].mean()
plot_around_switch(
    summary_before, summary_after, x_var_pre, x_var_post, xticks=np.arange(-6, 16, 2), avr_line=avr_perf,
)


filtered_df = df_test.merge(epis_count, on=["subject", "epis"], how="right")
filtered_df = filtered_df[filtered_df["criterion"] == 0].copy().reset_index(drop=True)


figures_beh_path = os.path.join(FIGURES_DIR, "behaviour")

x_var_pre = "before_pres"
x_var_post = "stim_pres"
keys = [-1, 0, 1]

y_var = "correct"
file_path = os.path.join(figures_beh_path, f"{x_var_post}_rule_{y_var}.pdf")
summary_before, summary_after = get_switch_summary(filtered_df, x_var_pre, x_var_post, y_var, -5, 15, "is_stimstable", "is_stimstable")
avr_perf = summary_before[summary_before[x_var_pre] > -4]["mean"].mean()
plot_around_switch(
    summary_before, summary_after, x_var_pre, x_var_post,
    hue_pre="is_stimstable", hue_post="is_stimstable", avr_line=avr_perf, ylim=(0, 1),
    keys=keys, palette=palette_dict, xticks=np.arange(-4, 16, 2), save_path=file_path
)

y_var = "explor"
file_path = os.path.join(figures_beh_path, f"{x_var_post}_rule_{y_var}.pdf")
summary_before, summary_after = get_switch_summary(filtered_df, x_var_pre, x_var_post, y_var, -5, 15, "is_stimstable", "is_stimstable")
plot_around_switch(
    summary_before, summary_after, x_var_pre, x_var_post,
    hue_pre="is_stimstable", hue_post="is_stimstable", ylim=(0, 0.5), ylabel="Proportion exploration",
    keys=keys, palette=palette_dict, xticks=np.arange(-4, 16, 2), save_path=file_path
)

y_var = "persev"
file_path = os.path.join(figures_beh_path, f"{x_var_post}_rule_{y_var}.pdf")
summary_before, _ = get_switch_summary(filtered_df, x_var_pre, x_var_post, "correct", -5, 15, "is_stimstable", "is_stimstable")
_, summary_after = get_switch_summary(filtered_df, x_var_pre, x_var_post, y_var, -5, 15, "is_stimstable", "is_stimstable")
avr_perf = summary_before[summary_before[x_var_pre] > -4]["mean"].mean()
plot_around_switch(
    summary_before, summary_after, x_var_pre, x_var_post,
    hue_pre="is_stimstable", hue_post="is_stimstable", avr_line=avr_perf, ylim=(0, 1), ylabel="Proportion perseveration",
    keys=keys, palette=palette_dict, xticks=np.arange(-4, 16, 2), save_path=file_path
)

x_var_pre = "before_trial"
x_var_post = "trial"

y_var = "correct"
file_path = os.path.join(figures_beh_path, f"{x_var_post}_rule_{y_var}.pdf")
summary_before, summary_after = get_switch_summary(filtered_df, x_var_pre, x_var_post, y_var, -15, 45, "is_stimstable", "is_stimstable")
avr_perf = summary_before[summary_before[x_var_pre] > -12]["mean"].mean()
plot_around_switch(
    summary_before, summary_after, x_var_pre, x_var_post,
    hue_pre="is_stimstable", hue_post="is_stimstable", avr_line=avr_perf, ylim=(0, 1), xlabel="Trial",
    ylabel = "Proportion correct", keys=keys, palette=palette_dict, xticks=np.arange(-15, 46, 5), save_path=file_path
)

y_var = "explor"
file_path = os.path.join(figures_beh_path, f"{x_var_post}_rule_{y_var}.pdf")
summary_before, summary_after = get_switch_summary(filtered_df, x_var_pre, x_var_post, y_var, -15, 45, "is_stimstable", "is_stimstable")
plot_around_switch(
    summary_before, summary_after, x_var_pre, x_var_post,
    hue_pre="is_stimstable", hue_post="is_stimstable", ylim=(0, 0.5), xlabel="Trial",
    ylabel = "Proportion exploration", keys=keys, palette=palette_dict, xticks=np.arange(-15, 46, 5), save_path=file_path
)

y_var = "persev"
file_path = os.path.join(figures_beh_path, f"{x_var_post}_rule_{y_var}.pdf")
summary_before, _ = get_switch_summary(filtered_df, x_var_pre, x_var_post, "correct",-15, 45, "is_stimstable", "is_stimstable")
_, summary_after = get_switch_summary(filtered_df, x_var_pre, x_var_post, y_var, -15, 45, "is_stimstable", "is_stimstable")
avr_perf = summary_before[summary_before[x_var_pre] > -12]["mean"].mean()
plot_around_switch(
    summary_before, summary_after, x_var_pre, x_var_post,
    hue_pre="is_stimstable", hue_post="is_stimstable", avr_line=avr_perf, ylim=(0, 1), xlabel="Trial",
    ylabel = "Proportion perseveration", keys=keys, palette=palette_dict, xticks=np.arange(-15, 46, 5), save_path=file_path
)







x_var_pre = "pre_hmmsw_pres"
x_var_post = "post_hmmsw_pres"
keys = ["random", "global", "overlap"]


y_var = "correct"
file_path = os.path.join(figures_beh_path, f"{x_var_post}_rule_{y_var}.pdf")
summary_before, summary_after = get_switch_summary(filtered_df, x_var_pre, x_var_post, y_var, -5, 7, "switch_type", "switch_type")
plot_around_switch(
    summary_before, summary_after, x_var_pre, x_var_post,
    hue_pre="switch_type", hue_post="switch_type", ylim=(0, 1),
    keys=keys, palette=palette_dict, xticks=np.arange(-4, 8, 2), save_path=file_path
)

y_var = "explor"
file_path = os.path.join(figures_beh_path, f"{x_var_post}_rule_{y_var}.pdf")
summary_before, summary_after = get_switch_summary(filtered_df, x_var_pre, x_var_post, y_var, -5, 15, "switch_type", "switch_type")
plot_around_switch(
    summary_before, summary_after, x_var_pre, x_var_post,
    hue_pre="switch_type", hue_post="switch_type", ylim=(0, 0.5), ylabel="Proportion exploration",
    keys=keys, palette=palette_dict, xticks=np.arange(-4, 16, 2), save_path=file_path
)

y_var = "persev"
file_path = os.path.join(figures_beh_path, f"{x_var_post}_rule_{y_var}.pdf")
summary_after, summary_after = get_switch_summary(filtered_df, x_var_pre, x_var_post, y_var, -5, 15, "switch_type", "switch_type")
plot_around_switch(
    summary_before, summary_after, x_var_pre, x_var_post,
    hue_pre="switch_type", hue_post="switch_type", ylim=(0, 1), ylabel="Proportion perseveration",
    keys=keys, palette=palette_dict, xticks=np.arange(-4, 16, 2), save_path=file_path
)
























# def rolling_rt_zscore(df, window_size=10):
#     """
#     Calculate the rolling z-score of reaction times in the behavioral DataFrame.
#     """
#     rolling_mean = df["rt"].rolling(window=window_size, min_periods=1).mean()
#     rolling_std = df["rt"].rolling(window=window_size, min_periods=1).std()
#     rolling_std = rolling_std.replace(0, np.nan)

#     df["rt_zscore_sliding"] = (df["rt"] - rolling_mean) / rolling_std
#     df.fillna({"rt_zscore_sliding": 0}, inplace=True)
#     return df

BEH_PATH = "/Users/charles.verstraete/Documents/w3_iEEG/behaviour/hmm"
fileslist = get_fileslist(BEH_PATH, ".csv")

fileslist
df = pd.DataFrame()
for file in fileslist:
    beh_df = pd.read_csv(file)
    # beh_df = find_criterion(beh_df)
    # beh_df = beh_df[beh_df["trial_succeed"] == 1]
    beh_df.loc[beh_df["is_stimstable"].isna(), "is_stimstable"] = -1
    beh_df["fb_prev"] = 0
    beh_df.loc[1:, "fb_prev"] = beh_df["fb"].values[:-1]
    # beh_df = beh_df[beh_df["rt"] > 0]
    # beh_df = beh_df[beh_df["rt"] < 5]

    beh_df = beh_df[beh_df["training"] == 0]
    # beh_df = beh_df[beh_df["criterion"] == 0].copy().reset_index(drop=True)
    beh_df = add_before_pres(beh_df)
    beh_df = add_before_trial(beh_df)
    df = pd.concat([df, beh_df], ignore_index=True)



fig, axs = plt.subplots(4, 6, figsize=(21, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, subject in enumerate(df["subject"].unique()):
    subject_df = df[df["subject"] == subject]
    subject_df = subject_df[subject_df["stim_pres"] < 15]
    sns.lineplot(data=subject_df, x="stim_pres", y="correct", ax=axs[i], color="black")
    axs[i].set_title(f"Subject {subject}")
    axs[i].set_xlabel("Stimulus presentation")
    axs[i].axhline(y=0.33, color="red", linestyle="--", label="Chance level")
for j in range(i+1, 25):
    fig.delaxes(axs[j])
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


df["before_pres"]








SIMU_PATH = "/Users/charles.verstraete/Documents/w3_iEEG/behaviour/simu_rank"
fileslist = get_fileslist(SIMU_PATH, ".csv")

simu = pd.DataFrame()
for file in fileslist:
    # file = fileslist[0]
    beh_df = pd.read_csv(file)
    # beh_df = find_criterion(beh_df)
    # beh_df = beh_df[beh_df["trial_succeed"] == 1]
    # beh_df = beh_df[beh_df["rt"] > 0]
    # beh_df = beh_df[beh_df["rt"] < 5]
    for idx, sdf in beh_df.groupby("rank") :
        tmp = pd.DataFrame(sdf.copy())
        sdf.loc[sdf["is_stimstable"].isna(), "is_stimstable"] = -1
        sdf = sdf[sdf["training"] == 0]
        sdf = sdf[sdf["criterion"] == 0].copy().reset_index(drop=True)
        sdf = add_before_pres(sdf)
        sdf = add_before_trial(sdf)
        simu = pd.concat([simu, sdf], ignore_index=True)


SIMU_PATH = "/Users/charles.verstraete/Documents/w3_iEEG/behaviour/simu_rank"
fileslist = get_fileslist(SIMU_PATH, ".csv")

simu = pd.DataFrame()
for file in fileslist:
    beh_df = pd.read_csv(file)
    beh_df = find_criterion(beh_df)
    # beh_df = beh_df[beh_df["trial_succeed"] == 1]
    # beh_df = beh_df[beh_df["rt"] > 0]
    # beh_df = beh_df[beh_df["rt"] < 5]
    beh_df = rolling_rt_zscore(beh_df, window_size=10)
    beh_df = beh_df[beh_df["training"] == 0]
    beh_df = beh_df[beh_df["criterion"] == 0].copy().reset_index(drop=True)
    beh_df = add_before_pres(beh_df)
    beh_df = add_before_trial(beh_df)
    simu = pd.concat([simu, beh_df], ignore_index=True)




# for subject in range(1, 31):
#     if subject not in [1, 7] : 
#         path = os.path.join(ORIGINAL_DATA_DIR, f"{subject}", "Behavior") 
#         if os.path.exists(path) :  
#             behav_files = get_fileslist(os.path.join(ORIGINAL_DATA_DIR, f"{subject}", "Behavior"), ".csv")
#             if len(behav_files) != 0 :
#                 beh_file = [f for f in behav_files if "resTable" in f][0]
#                 empty_file = os.path.join(ORIGINAL_DATA_DIR, f"{subject}", "Behavior", f"{subject}.csv")
#                 beh_df = pd.read_csv(beh_file)
#                 empty = pd.read_csv(empty_file)
#                 beh_df["subject"] = subject
#                 empty["subject"] = subject
#                 beh_df.to_csv(os.path.join(NEW_PATH, "task", f"sub-{subject:03}_task-stratinf_beh.csv"), index=False)
#                 empty.to_csv(os.path.join(NEW_PATH, "empty", f"sub-{subject:03}_task-stratinf_beh_empty.csv"), index=False)
    #             beh_df = find_criterion(beh_df)
    #             beh_df = beh_df[beh_df["trial_succeed"] == 1]
    #             beh_df = beh_df[beh_df["rt"] > 0]
    #             beh_df = beh_df[beh_df["rt"] < 5]
    #             beh_df = rolling_rt_zscore(beh_df, window_size=10)
    #             beh_df = beh_df[beh_df["training"] == 0]
    #             beh_df = beh_df[beh_df["criterion"] == 0].copy().reset_index(drop=True)
    #             beh_df = add_before_pres(beh_df)
    #             beh_df = add_before_trial(beh_df)
    #             df = pd.concat([df, beh_df], ignore_index=True)

good_df = pd.DataFrame()
fig, axs = plt.subplots(5, 5, figsize=(21, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, subject in enumerate(df["subject"].unique()):
    # if subject not in [9, 10, 13, 15, 17, 28] : 
    subject_df = df[df["subject"] == subject]
    # good_df = pd.concat([good_df, subject_df], ignore_index=True)
    sns.lineplot(data=subject_df, x="stim_pres", y="correct", ax=axs[i], color="black")
    axs[i].set_title(f"Subject {subject}")
    axs[i].set_xlabel("Stimulus presentation")
    axs[i].axhline(y=0.33, color="red", linestyle="--", label="Chance level")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()




simu[simu["is_partial"] == 0]


filtered_df = df.copy()
# filtered_df.loc[filtered_df["is_partial"] == 0, "is_stimstable"] = -1
epis_count = filtered_df.groupby(["subject", "epis"]).size().reset_index(name="count")
plt.hist(epis_count["count"], bins=100)
plt.show()
epis_count = epis_count[epis_count["count"] < 55]
epis_count = epis_count[epis_count["count"] > 15]
filtered_df = filtered_df.merge(epis_count, on=["subject", "epis"], how="right")


epis_count = simu.groupby(["subject", "epis", "rank"]).size().reset_index(name="count")
epis_count = epis_count[epis_count["count"] < 50]
epis_count = epis_count[epis_count["count"] > 15]
simu = simu.merge(epis_count, on=["subject", "epis", "rank"], how="right")

simu

filtered_df["subject"].unique()


n_sub = len(filtered_df["subject"].unique())

y_var = "correct"
x_var_post = "stim_pres"
x_var_pre = "before_pres"

summary_after_indiv = simu.groupby(["subject", x_var_post, "is_stimstable"])[y_var].mean().reset_index()
summary_after = summary_after_indiv.groupby([x_var_post, "is_stimstable"]).agg({y_var: ["mean", "std"]}).reset_index()
summary_after.columns = [x_var_post, "is_stimstable", "mean", "std"]
summary_after["sem"] = summary_after["std"] / np.sqrt(n_sub)

summary_before_indiv = simu.groupby(["subject", x_var_pre, "next_stable"])[y_var].mean().reset_index()
summary_before = summary_before_indiv.groupby([x_var_pre, "next_stable"]).agg({y_var: ["mean", "std"]}).reset_index()
summary_before.columns = [x_var_pre, "next_stable", "mean", "std"]
summary_before["sem"] = summary_before["std"] / np.sqrt(n_sub)

summary_after = summary_after[summary_after[x_var_post] > 0]
summary_after = summary_after[summary_after[x_var_post] < 15]

summary_before = summary_before[summary_before[x_var_pre] < 0]
summary_before = summary_before[summary_before[x_var_pre] > -4]
avr_perf = summary_before["mean"].mean()
# summary_before = summary_before[summary_before["before_pres"] > -4]

ref_tmp = summary_before_indiv[(summary_before_indiv["next_stable"] == 1)& (summary_before_indiv["before_pres"] > -4)]
ref = ref_tmp.groupby(["subject"])[y_var].mean().reset_index()
after_stable = summary_after_indiv[(summary_after_indiv["is_stimstable"] == 1)]
test = after_stable.merge(ref, on=["subject"], how="right")
test["interference"] = test["correct_x"] - test["correct_y"]
test = test[test["stim_pres"] < 10]
sns.barplot(data=test, x="stim_pres", y="interference", color="dimgray", alpha=0.5)
sns.stripplot(data=test, x="stim_pres", y="interference", color="black", alpha=0.5)
# plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "interference.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()



plt.figure(figsize=(21, 12))
for key in [-1, 0, 1]:
    after_grp = summary_after[summary_after["is_stimstable"] == key]
    before_grp = summary_before[summary_before["next_stable"] == key]
    plt.plot(after_grp[x_var_post], after_grp["mean"], color=palette_dict[key])
    plt.fill_between(after_grp[x_var_post], after_grp["mean"] - after_grp["sem"], after_grp["mean"] + after_grp["sem"], alpha=0.2, color=palette_dict[key])
    plt.plot(before_grp[x_var_pre], before_grp["mean"], color=palette_dict[key])
    plt.fill_between(before_grp[x_var_pre], before_grp["mean"] - before_grp["sem"], before_grp["mean"] + before_grp["sem"], alpha=0.2, color=palette_dict[key])

plt.xticks(np.arange(-4, 11, 2), rotation=45)
plt.ylim(0, 1)
plt.axhline(y=avr_perf, color="black", linestyle="--", label="Average performance")
plt.title("Behavioral performance by stimulus presentation")
plt.xlabel("Stimulus presentation")
plt.ylabel("Proportion of correct responses")
# plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "stim_pres_perf.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()




summary_after_indiv = filtered_df.groupby(["subject", x_var_post, "is_stimstable"])[y_var].mean().reset_index()
summary_after = summary_after_indiv.groupby([x_var_post, "is_stimstable"]).agg({y_var: ["mean", "std"]}).reset_index()
summary_after.columns = [x_var_post, "is_stimstable", "mean", "std"]
summary_after["sem"] = summary_after["std"] / np.sqrt(n_sub)



simu.columns

y_var = "action_value"
x_var_post = "stim_pres"

summary_after_indiv = simu.groupby(["subject", x_var_post])[y_var].mean().reset_index()
summary_after = summary_after_indiv.groupby([x_var_post]).agg({y_var: ["mean", "std"]}).reset_index()
summary_after.columns = [x_var_post, "mean", "std"]
summary_after["sem"] = summary_after["std"] / np.sqrt(n_sub)

plt.plot(summary_after[x_var_post], summary_after["mean"])
plt.fill_between(summary_after[x_var_post], summary_after["mean"] - summary_after["sem"], summary_after["mean"] + summary_after["sem"], alpha=0.2)
plt.show()




sns.lineplot(data=simu, x="before_pres", y="entropy")
sns.lineplot(data=simu, x="stim_pres", y="entropy")

plt.show()


y_var = "persev"
x_var_post = "post_hmmsw_trial"
x_var_pre = "pre_hmmsw_trial"

summary_after_indiv = filtered_df.groupby(["subject", x_var_post, "switch_type"])[y_var].mean().reset_index()
summary_after = summary_after_indiv.groupby([x_var_post, "switch_type"]).agg({y_var: ["mean", "std"]}).reset_index()
summary_after.columns = [x_var_post, "switch_type", "mean", "std"]
summary_after["sem"] = summary_after["std"] / np.sqrt(n_sub)

summary_before_indiv = filtered_df.groupby(["subject", x_var_pre, "switch_type"])[y_var].mean().reset_index()
summary_before = summary_before_indiv.groupby([x_var_pre, "switch_type"]).agg({y_var: ["mean", "std"]}).reset_index()
summary_before.columns = [x_var_pre, "switch_type", "mean", "std"]
summary_before["sem"] = summary_before["std"] / np.sqrt(n_sub)

summary_after = summary_after[summary_after[x_var_post] > 0]
summary_after = summary_after[summary_after[x_var_post] < 7]

summary_before = summary_before[summary_before[x_var_pre] < 0]
summary_before = summary_before[summary_before[x_var_pre] > -4]
# avr_perf = summary_before["mean"].mean()
# summary_before = summary_before[summary_before["before_pres"] > -4]



plt.figure(figsize=(21, 12))
for key in ["random", "global", "overlap"]: #[-1, 0, 1]:
    after_grp = summary_after[summary_after["switch_type"] == key]
    before_grp = summary_before[summary_before["switch_type"] == key]
    plt.plot(after_grp[x_var_post], after_grp["mean"], color=palette_dict[key])
    plt.fill_between(after_grp[x_var_post], after_grp["mean"] - after_grp["sem"], after_grp["mean"] + after_grp["sem"], alpha=0.2, color=palette_dict[key])
    plt.plot(before_grp[x_var_pre], before_grp["mean"], color=palette_dict[key])
    plt.fill_between(before_grp[x_var_pre], before_grp["mean"] - before_grp["sem"], before_grp["mean"] + before_grp["sem"], alpha=0.2, color=palette_dict[key])

plt.xticks(np.arange(-4, 11, 2), rotation=45)
# plt.ylim(0, 1)
# plt.axhline(y=0.33, color="black", linestyle="--", label="Average rt z-score")
plt.title("Behavioral performance by stimulus presentation")
plt.xlabel("Stimulus presentation")
plt.ylabel("Proportion of correct responses")
# plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "stim_pres_rt.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()

filtered_df.columns

rt_swtich = pd.DataFrame({
    "rt" : filtered_df[filtered_df["firstswitch"] == 1]["rt"].values,
    "pos" : 0,
    "fb" : filtered_df[filtered_df["firstswitch"] == 1]["fb"].values,
    "switch_type" : filtered_df[filtered_df["firstswitch"] == 1]["switch_type"].values
})
rt_postswitch = pd.DataFrame({
    "rt" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["rt"].values,
    "pos" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["post_hmmsw_trial"].values,
    "fb" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["fb"].values,  
    "switch_type" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["switch_type"].values
})
rt_preswitch = pd.DataFrame({
    "rt" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["rt"].values,
    "pos" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["pre_hmmsw_trial"].values,
    "fb" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["fb"].values,
    "switch_type" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["switch_type"].values
})

full_rt = pd.concat([rt_swtich, rt_postswitch, rt_preswitch], ignore_index=True)

fig, axs = plt.subplots(2, 3, figsize=(21, 12), sharex=True)
# axs = axs.flatten()
for i, switch_type in enumerate(full_rt["switch_type"].unique()):
    switch_df = full_rt[full_rt["switch_type"] == switch_type]
    switch_df_pos = switch_df[switch_df["fb"] == 1]
    switch_df_neg = switch_df[switch_df["fb"] == 0]
    sns.barplot(data=switch_df_pos, x="pos", y="rt", ax=axs[0, i], color="blue", alpha=0.5, label="FB+")
    sns.stripplot(data=switch_df_pos, x="pos", y="rt", ax=axs[0, i], color="blue", alpha=0.5)
    sns.barplot(data=switch_df_neg, x="pos", y="rt", ax=axs[1, i], color="red", alpha=0.5, label="FB-")
    sns.stripplot(data=switch_df_neg, x="pos", y="rt", ax=axs[1, i], color="red", alpha=0.5)
    axs[0, i].set_title(f"Switch type: {switch_type}")
    axs[1, i].set_xlabel("Switch position")
    axs[1, i].legend()
plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "rt_switch_type.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()



















def get_around_switch(df, pre_type, post_type, lim_pre, lim_post):
    pre_switch = df[(df[pre_type] >= lim_pre) & (df[pre_type] < 0)].copy()
    post_switch = df[(df[post_type] > 0) & (df[post_type] <= lim_post)].copy()
    return pre_switch, post_switch

def prepare_subbdf(df, var, group_cols, new_names, period) : 
    out_df = df.groupby(group_cols)[var].mean().reset_index()
    out_df = out_df.rename(columns=new_names)
    out_df["period"] = period
    return out_df

def prepare_around_switch(df, var, switch_type, pres_type, lim_pre, lim_post):
    if switch_type == 'rule':
        pre_type = "before_pres" if pres_type == "pres" else "before_trial"
        post_type = "stim_pres" if pres_type == "pres" else "trial"
    else :
        pre_type = "pre_hmmsw_pres" if pres_type == "pres" else "pre_hmmsw_trial"
        post_type = "post_hmmsw_pres" if pres_type == "pres" else "post_hmmsw_trial"
    
    pre_data, post_post = get_around_switch(df, pre_type, post_type, lim_pre, lim_post)

    pre_data = prepare_subbdf(pre_data, var, 
                              ["subject", "next_stable", pre_type], 
                              {"next_stable": "stable_condition", pre_type: "position", var: "accuracy"}, "pre")
    
    post_data = prepare_subbdf(post_post, var, 
                               ["subject", "is_stimstable", post_type], 
                               {"is_stimstable": "stable_condition", post_type: "position", var: "accuracy"}, "post")
    return pd.concat([pre_data, post_data])








def get_merged_around(pre, post, var):
    pre_perf_avg = pre.groupby(["subject", "next_stable"])[var].mean().reset_index()
    post_perf_by_pres = post.groupby(["subject", "is_stimstable", "stim_pres"])[var].mean().reset_index()
    pre_perf_avg = pre_perf_avg.rename(columns={"next_stable": "stable_condition", var: "pre_accuracy"})
    post_perf_by_pres = post_perf_by_pres.rename(columns={"is_stimstable": "stable_condition", var: "post_accuracy"})
    return pd.merge(post_perf_by_pres, pre_perf_avg, on=["subject", "stable_condition"])

def get_concatenated_around(pre, post, var):
    pre_perf_avg = pre.groupby(["subject", "next_stable"])[var].mean().reset_index()
    pre_perf_avg = pre_perf_avg.rename(columns={"next_stable": "stable_condition", var: "pre_accuracy"})
    post_perf_avg = post.groupby(["subject", "is_stimstable", "stim_pres"])[var].mean().reset_index()
    post_perf_avg = post_perf_avg.rename(columns={"is_stimstable": "stable_condition", var: "post_accuracy"})
    pre_data = pre_perf_avg.copy().rename(columns={'next_stable': 'stable_condition', 'before_pres': 'position', var: 'accuracy'})
    pre_data['period'] = 'pre'
    post_data = post_perf_avg.copy().rename(columns={'is_stimstable': 'stable_condition', 'stim_pres': 'position', var: 'accuracy'})
    post_data['period'] = 'post'
    return pd.concat([pre_data, post_data])

def plot_perf_by_pres(pre_data, post_data, palette_dict, stable_last_perf, condition, ax):
    color = palette_dict[condition]
    pre_mean = pre_data[pre_data['stable_condition'] == condition].groupby('position')['accuracy'].mean()
    ax.plot(pre_mean.index, pre_mean.values, color=color, lw=3)

    post_mean = post_data[post_data['stable_condition'] == condition].groupby('position')['accuracy'].mean()
    ax.plot(post_mean.index, post_mean.values, color=color, lw=3)
    ax.axhline(stable_last_perf, color='dimgrey', ls='--', alpha=0.5)
    ax.set_ylim(0, 1)
    return ax




##########################################################################################
#### interference



pre_machine, post_machine = get_around_switch(filtered_df, "before_pres", "stim_pres", -3, 12)

pre_perf_avg = pre_machine.groupby(["subject", "next_stable"])["correct"].mean().reset_index()
pre_perf_avg = pre_perf_avg.rename(columns={"next_stable": "stable_condition", "correct": "pre_accuracy"})

# 2. Calculer la performance pour chaque présentation post-switch
post_perf_by_pres = post_machine.groupby(["subject", "is_stimstable", "stim_pres"])["correct"].mean().reset_index()
post_perf_by_pres = post_perf_by_pres.rename(columns={"is_stimstable": "stable_condition", "correct": "post_accuracy"})

combined = pd.concat([pre_perf_avg, post_perf_by_pres])
combined
interference_by_pres = pd.merge(post_perf_by_pres, pre_perf_avg, on=["subject", "stable_condition"])
interference_by_pres["interference_drop"] = interference_by_pres["post_accuracy"] - interference_by_pres["pre_accuracy"]

plt.figure(figsize=(15, 12))
sns.barplot(x="stim_pres", y="interference_drop", data=interference_by_pres[interference_by_pres["stable_condition"] == 1],  errorbar=("se", 0.95), color="grey", alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
sns.stripplot(x="stim_pres", y="interference_drop", data=interference_by_pres[interference_by_pres["stable_condition"] == 1], size=6, alpha=0.3, color="black", legend=False)
# plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "interference_by_pres.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()



##########################################################################################
#### perf

pre_machine, post_machine = get_around_switch(simu,"before_pres", "stim_pres", -3, 12)


pre_perf_avg = pre_machine.groupby(["subject", "next_stable", "before_pres"])["correct"].mean().reset_index()
post_perf_avg = post_machine.groupby(["subject", "is_stimstable", "stim_pres"])["correct"].mean().reset_index()

pre_data_beh = pre_perf_avg.copy().rename(columns={'next_stable': 'stable_condition', 'before_pres': 'position', 'correct': 'accuracy'})
pre_data_beh['period'] = 'pre'
post_data_beh = post_perf_avg.copy().rename(columns={'is_stimstable': 'stable_condition', 'stim_pres': 'position', 'correct': 'accuracy'})
post_data_beh['period'] = 'post'
combined = pd.concat([pre_data_beh, post_data_beh])

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
condition_axes = {0: axs[0], 1: axs[1], -1: axs[2]}

for condition, ax in condition_axes.items():
    color = palette_dict[condition]
    for subject in combined['subject'].unique():
        pre_subj = combined[(combined['subject'] == subject) & 
                           (combined['stable_condition'] == condition) & 
                           (combined['period'] == 'pre')]
        ax.plot(pre_subj['position'], pre_subj['accuracy'], color=color, alpha=0.4, lw=1)
        post_subj = combined[(combined['subject'] == subject) & 
                            (combined['stable_condition'] == condition) & 
                            (combined['period'] == 'post')]
        ax.plot(post_subj['position'], post_subj['accuracy'], color=color, alpha=0.4, lw=1)


stable_last_perf = combined[(combined['stable_condition'] == 1) & (combined['position'] == -1)]["accuracy"].mean()
axs[1].axhline(stable_last_perf, color='dimgrey', ls='--', alpha=0.5)

for condition, ax in condition_axes.items():
    color = palette_dict[condition]
    pre_mean = combined[(combined['stable_condition'] == condition) & (combined['period'] == 'pre')].groupby('position')['accuracy'].mean()
    ax.plot(pre_mean.index, pre_mean.values, color=color, lw=3)

    post_mean = combined[(combined['stable_condition'] == condition) & (combined['period'] == 'post')].groupby('position')['accuracy'].mean()
    ax.plot(post_mean.index, post_mean.values, color=color, lw=3)
    ax.axhline(0, color='black', ls='--')
    
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


pre_machine_simu, post_machine_simu = get_around_switch(simu, "before_pres", "stim_pres", -3, 10)
pre_machine_beh, post_machine_beh = get_around_switch(filtered_df,"before_pres", "stim_pres", -3, 10)

pre_perf_beh = pre_machine_beh.groupby(["subject", "next_stable", "before_pres"])["correct"].mean().reset_index()
post_perf_beh = post_machine_beh.groupby(["subject", "is_stimstable", "stim_pres"])["correct"].mean().reset_index()

pre_data_beh = pre_perf_beh.copy().rename(columns={'next_stable': 'stable_condition', 'before_pres': 'position', 'correct': 'accuracy'})
post_data_beh = post_perf_beh.copy().rename(columns={'is_stimstable': 'stable_condition', 'stim_pres': 'position', 'correct': 'accuracy'})

# Préparer les données pour les simulations
pre_perf_sim = pre_machine_simu.groupby(["subject", "next_stable", "before_pres"])["correct"].mean().reset_index()
post_perf_sim = post_machine_simu.groupby(["subject", "is_stimstable", "stim_pres"])["correct"].mean().reset_index()

pre_data_sim = pre_perf_sim.copy().rename(columns={'next_stable': 'stable_condition', 'before_pres': 'position', 'correct': 'accuracy'})
post_data_sim = post_perf_sim.copy().rename(columns={'is_stimstable': 'stable_condition', 'stim_pres': 'position', 'correct': 'accuracy'})


root_norm = np.sqrt(len(filtered_df['subject'].unique()))

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
condition_axes = {0: axs[0], 1: axs[1], -1: axs[2]}

for condition, ax in condition_axes.items():
    color = palette_dict[condition]    
    pre_stats_beh = pre_data_beh[pre_data_beh['stable_condition'] == condition].groupby('position')['accuracy'].agg(['mean', 'std']).reset_index()
    ax.plot(pre_stats_beh['position'], pre_stats_beh['mean'], color=color, lw=1)
    ax.fill_between(pre_stats_beh['position'], 
                    pre_stats_beh['mean'] - pre_stats_beh['std']/root_norm, 
                    pre_stats_beh['mean'] + pre_stats_beh['std']/root_norm, 
                    color=color, alpha=0.2)
    
    # COMPORTEMENT RÉEL APRÈS SWITCH (avec ribbons)
    post_stats_beh = post_data_beh[post_data_beh['stable_condition'] == condition].groupby('position')['accuracy'].agg(['mean', 'std']).reset_index()
    ax.plot(post_stats_beh['position'], post_stats_beh['mean'], color=color, lw=1)
    ax.fill_between(post_stats_beh['position'], 
                    post_stats_beh['mean'] - post_stats_beh['std']/root_norm, 
                    post_stats_beh['mean'] + post_stats_beh['std']/root_norm, 
                    color=color, alpha=0.2)

# Pour chaque condition
for condition, ax in condition_axes.items():
    color = palette_dict[condition]
    pre_stats_sim = pre_data_sim[pre_data_sim['stable_condition'] == condition].groupby('position')['accuracy'].agg(['mean', 'std']).reset_index()
    ax.errorbar(pre_stats_sim['position'], pre_stats_sim['mean'], yerr=pre_stats_sim['std']/root_norm, color=color, lw=2.5, capsize=4, linestyle = "--")
    
    post_stats_sim = post_data_sim[post_data_sim['stable_condition'] == condition].groupby('position')['accuracy'].agg(['mean', 'std']).reset_index()
    ax.errorbar(post_stats_sim['position'], post_stats_sim['mean'], yerr=post_stats_sim['std']/root_norm, color=color, lw=2.5, linestyle = "--", capsize=4)


    
# Ajouter des titres communs
fig.text(0.5, 0.01, 'Position relative au changement de règle', ha='center', fontsize=14)
fig.text(0.01, 0.5, 'Précision', va='center', rotation='vertical', fontsize=14)
plt.suptitle('Comparaison comportement réel vs simulation', fontsize=16, y=0.98)

plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "stim_pres_performance_simu.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()



##########################################################################################
#### hmm switch

hmm_switch = filtered_df[filtered_df["firstswitch"] == 1]
count_transition = hmm_switch.groupby(["switch_type", "is_partial", "subject"]).size().reset_index()
count_transition["proportion"] = count_transition[0] / count_transition.groupby(["is_partial", "subject"])[0].transform("sum")

plt.figure(figsize=(13, 10))
sns.barplot(data=count_transition, x = "switch_type", y = "proportion", hue="is_partial", palette = [complete_color, partial_color], errorbar=("se", 0.95), alpha=0.5)
sns.stripplot(data=count_transition, x = "switch_type", y = "proportion", hue="is_partial", palette = [complete_color, partial_color], dodge=True, alpha=0.8, legend=False)
plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "hmm_switch_prop.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()

pre_hmm, post_hmm = get_around_switch(filtered_df, "pre_hmmsw_pres", "post_hmmsw_pres", -3, 6)
# post_hmm = filtered_df[filtered_df["firstswitch"] == 1]
pre_overlap = pre_hmm[pre_hmm['switch_type'] == "overlap"].copy()
pre_nonoverlap = pre_hmm[pre_hmm['switch_type'] != "overlap"].copy()

post_overlap = post_hmm[post_hmm['switch_type'] == "overlap"].copy()
post_nonoverlap = post_hmm[post_hmm['switch_type'] != "overlap"].copy()

pre_overlap_stable = pre_overlap[pre_overlap['is_stimstable'] == 1].copy()
pre_nonoverlap_stable = pre_nonoverlap[pre_nonoverlap['is_stimstable'] == 1].copy()

post_overlap_stable = post_overlap[post_overlap['is_stimstable'] == 1].copy()
post_nonoverlap_nostable = post_nonoverlap[post_nonoverlap['is_stimstable'] == 1].copy()

plt.figure(figsize=(21, 12))

sns.lineplot(data = pre_overlap_stable, x = "pre_hmmsw_pres", y = "persev", errorbar=("se", 0.95), color = stable_color)
sns.lineplot(data = pre_nonoverlap_stable, x = "pre_hmmsw_pres", y = "persev", errorbar=("se", 0.95), color = stable_color, linestyle="--")

sns.lineplot(data = post_overlap_stable, x = "post_hmmsw_pres", y = "persev", errorbar=("se", 0.95), color = stable_color)
sns.lineplot(data = post_nonoverlap_nostable, x = "post_hmmsw_pres", y = "persev", errorbar=("se", 0.95), color = stable_color, linestyle="--")

plt.axhline( pre_hmm[pre_hmm['is_stimstable'] == 1]["correct"].mean(), color = 'dimgrey', linestyle = '--', alpha = 0.5)

plt.ylim(0, 1)
plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "hmm_switch_stable__overlap_persev.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "hmm_switch_stable__overlap_persev.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()


fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True, sharex=True)
sns.barplot(data = pre_overlap_stable, x = "pre_hmmsw_pres", y = "rt_zscore_sliding", errorbar=("se", 0.95), color = stable_color, ax=axs[0])
sns.barplot(data = pre_nonoverlap_stable, x = "pre_hmmsw_pres", y = "rt_zscore_sliding", errorbar=("se", 0.95), color = stable_color, linestyle="--", ax=axs[1])
sns.barplot(data = post_overlap_stable, x = "post_hmmsw_pres", y = "rt_zscore_sliding", errorbar=("se", 0.95), color = stable_color, ax=axs[0])
sns.barplot(data = post_nonoverlap_nostable, x = "post_hmmsw_pres", y = "rt_zscore_sliding", errorbar=("se", 0.95), color = stable_color, linestyle="--", ax=axs[1])
axs[0].set_title("Overlap")
axs[1].set_title("Non-overlap")

sns.barplot(data = pre_overlap_stable, x = "pre_hmmsw_pres", y = "correct", errorbar=("se", 0.95), color = stable_color)
sns.barplot(data = pre_nonoverlap_stable, x = "pre_hmmsw_pres", y = "rt_zscore_sliding", errorbar=("se", 0.95), color = stable_color, linestyle="--")

sns.barplot(data = post_overlap_stable, x = "post_hmmsw_pres", y = "rt_zscore_sliding", errorbar=("se", 0.95), color = stable_color)
sns.barplot(data = post_nonoverlap_nostable, x = "post_hmmsw_pres", y = "rt_zscore_sliding", errorbar=("se", 0.95), color = stable_color, linestyle="--")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "hmm_stable_switch_rt.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(21, 12))
sns.lineplot(data = pre_hmm, x = "pre_hmmsw_pres", y = "persev", errorbar=("se", 0.95), palette = palette_dict, hue="is_stimstable", legend=False)
sns.lineplot(data = post_hmm, x = "post_hmmsw_pres", y = "persev", errorbar=("se", 0.95), palette = palette_dict, hue="is_stimstable", legend=False)
plt.axhline( pre_hmm[pre_hmm['is_stimstable'] == 1]["correct"].mean(), color = 'dimgrey', linestyle = '--', alpha = 0.5)
# plt.axhline(0.33, color = 'black', linestyle = '--')
plt.ylim(0, 1)
plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "hmm_switch_persev.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()


##########################################################################################
#### random strategy

pre_hmm, post_hmm = get_around_switch(filtered_df, "pre_hmmsw_pres", "post_hmmsw_pres", -4, 6)

pre_random = pre_hmm[pre_hmm["switch_type"] == "random"]
pre_nonrandom = pre_hmm[pre_hmm["switch_type"] != "random"]

post_random = post_hmm[post_hmm["switch_type"] == "random"]
post_nonrandom = post_hmm[post_hmm["switch_type"] != "random"]

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

sns.lineplot(data = pre_random, x = "pre_hmmsw_pres", y = "correct", errorbar=("ci", 95), ax = axs[0], color = 'grey')
sns.lineplot(data = post_nonrandom, x = "post_hmmsw_pres", y = "correct",  errorbar=("ci", 95), ax = axs[0], color = 'grey')
axs[0].axhline(0.33, color = 'black', linestyle = '--')
axs[0].set_ylim(0, 1)

sns.lineplot(data = pre_nonrandom, x = "pre_hmmsw_pres", y = "persev", errorbar=("ci", 95), ax = axs[1], color = 'grey')
sns.lineplot(data = post_random, x = "post_hmmsw_pres", y = "persev",  errorbar=("ci", 95), ax = axs[1], color = 'grey')
axs[1].axhline(0.33, color = 'black', linestyle = '--')
axs[1].set_ylim(0, 1)


plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "random_switch_correct.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()




random_duration = pre_random.groupby(["subject", "epis"])["stim_pres"].max()
random_duration = random_duration.reset_index()

random_duration_proba = random_duration.groupby(["subject","stim_pres"]).size().reset_index()
random_duration_proba = random_duration_proba.rename(columns={0: "count"})
random_duration_proba["proba"] = random_duration_proba["count"] / random_duration_proba.groupby("subject")["count"].transform("sum")
random_duration_proba = random_duration_proba[random_duration_proba["stim_pres"] < 8] 

fig, axs = plt.subplots(2, 1, figsize=(15, 5))

sns.barplot(data = random_duration_proba, x = "stim_pres", errorbar="ci", color="grey", alpha=0.5, ax = axs[0])
sns.stripplot(data = random_duration_proba, x = "stim_pres", size=6, alpha=0.3, color="black", legend=False, ax = axs[0])

sns.lineplot(data = random_duration_proba, x = "stim_pres", y = "proba", color = 'grey', ax = axs[1])
axs[1].set_ylim(0, 1)
plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "random_duration.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()




