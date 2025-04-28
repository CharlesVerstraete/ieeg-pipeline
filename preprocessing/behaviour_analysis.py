
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

import seaborn as sns
import matplotlib.pyplot as plt




df = pd.DataFrame()

for subject in range(1, 29) :
    if subject not in [1, 7] : 
        path = os.path.join(ORIGINAL_DATA_DIR, f"{subject}", "Behavior") 
        if os.path.exists(path) :  
            behav_files = get_fileslist(os.path.join(ORIGINAL_DATA_DIR, f"{subject}", "Behavior"), ".csv")
            print(behav_files)
            if len(behav_files) != 0 :
                beh_df = pd.read_csv(behav_files[-1])
                beh_df["subject"] = subject
                beh_df = find_criterion(beh_df)
                beh_df = beh_df[beh_df["criterion"] == 0].copy().reset_index(drop=True)
                beh_df = add_before_pres(beh_df)
                beh_df = add_before_trial(beh_df)
                df = pd.concat([df, beh_df], ignore_index=True)


filtered_df = df[df["training"] == 0]
# filtered_df = filtered_df[filtered_df["criterion"] == 0]
filtered_df.loc[filtered_df["is_partial"] == 0, "is_stimstable"] = -1
epis_count = filtered_df.groupby(["subject", "epis"]).size().reset_index(name="count")
epis_count = epis_count[epis_count["count"] < 55]

filtered_df = filtered_df.merge(epis_count, on=["subject", "epis"], how="right")
n_sub = len(filtered_df["subject"].unique())

summary_after = filtered_df.groupby(["subject", "stim_pres", "is_stimstable"])["correct"].mean().reset_index()
summary_after = summary_after.groupby(["stim_pres", "is_stimstable"]).agg({"correct": ["mean", "std"]}).reset_index()
summary_after.columns = ["stim_pres", "is_stimstable", "mean", "std"]
summary_after["sem"] = summary_after["std"] / np.sqrt(n_sub)

summary_before = filtered_df.groupby(["subject", "before_pres", "next_stable"])["correct"].mean().reset_index()
summary_before = summary_before.groupby(["before_pres", "next_stable"]).agg({"correct": ["mean", "std"]}).reset_index()
summary_before.columns = ["before_pres", "next_stable", "mean", "std"]
summary_before["sem"] = summary_before["std"] / np.sqrt(n_sub)

summary_after = summary_after[summary_after["stim_pres"] > 0]
summary_after = summary_after[summary_after["stim_pres"] < 11]

summary_before = summary_before[summary_before["before_pres"] < 0]
summary_before = summary_before[summary_before["before_pres"] > -5]
avr_perf = summary_before["mean"].mean()
summary_before = summary_before[summary_before["before_pres"] > -4]


full_before = filtered_df.groupby(["subject", "before_pres"])["correct"].mean().reset_index()
full_before = full_before.groupby(["before_pres"]).agg({"correct": ["mean", "std"]}).reset_index()
full_before.columns = ["before_pres", "mean", "std"]
full_before["sem"] = full_before["std"] / np.sqrt(n_sub)

full_before = full_before[full_before["before_pres"] < 0]
full_before = full_before[full_before["before_pres"] > -5]
avr_perf = full_before["mean"].mean()




for key in [-1, 0, 1]:
    after_grp = summary_after[summary_after["is_stimstable"] == key]
    #before_grp = summary_before[summary_before["next_stable"] == key]
    plt.plot(after_grp["stim_pres"], after_grp["mean"], color=palette_dict[key], marker="o", markersize=2)
    plt.fill_between(after_grp["stim_pres"], after_grp["mean"] - after_grp["sem"], after_grp["mean"] + after_grp["sem"], alpha=0.2, color=palette_dict[key])
    # plt.plot(before_grp["before_pres"], before_grp["mean"], color=palette_dict[key], marker="o", markersize=2, linestyle="--")
    # plt.fill_between(before_grp["before_pres"], before_grp["mean"] - before_grp["sem"], before_grp["mean"] + before_grp["sem"], alpha=0.2, color=palette_dict[key])
plt.plot(full_before["before_pres"], full_before["mean"], color="dimgrey", marker="o", markersize=2, linestyle="--")
plt.fill_between(full_before["before_pres"], full_before["mean"] - full_before["sem"], full_before["mean"] + full_before["sem"], alpha=0.2, color="dimgrey")    
plt.ylim(0, 1)
plt.axhline(y=avr_perf, color="black", linestyle="--", label="Average performance")
plt.title("Behavioral performance by stimulus presentation")
plt.xlabel("Stimulus presentation")
plt.ylabel("Proportion of correct responses")
plt.grid()
plt.show()




sns.lineplot(data=filtered_df, x="before_pres", y="correct", hue="next_stable", palette=palette_dict)




