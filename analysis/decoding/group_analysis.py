# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""
Module for group analysis of decoding
"""


from analysis.decoding.config import *
from analysis.decoding.loader import GroupLoader
from analysis.decoding.decoding_analysis import DecodingAnalysis
from analysis.decoding.process_features import Features
from analysis.decoding.geometry import Geometry
from tqdm import tqdm

plt.style.use('seaborn-v0_8-poster') 




model_name = "ridgecv"
fold = 5
subjects = [2, 3, 4, 5, 8, 9, 12, 14, 16, 19, 20, 23, 25, 28]


geometry = Geometry(subjects, "transition", group_var=["is_stimstable", "explor_exploit", "switch_type"])
geometry.fill_group_power("transition")


geometry

sns.barplot(data = geometry.trial_count, x = "group", y = "count")
plt.show()




plt.imshow(geometry.X_group[(1.0, 'explor', 'random')][:, -1, :], aspect='auto', cmap='jet', origin='lower')
plt.show()









geometry.var + ['subject']

[geometry.group_loaders.beh_data[var].dropna().unique().tolist() for var in geometry.var]

# for subject in self.group_loaders.subjects:
subject = geometry.group_loaders.subjects[1]
X_power = geometry._get_powerband("transition", subject, baseline=True, n_jobs=-1)
idx_chan = 0
df = geometry.group_loaders.beh_data[geometry.group_loaders.beh_data['subject'] == subject]
n_channels = X_power.shape[1]
for group, sdf in df.groupby(geometry.group_var):
    beh_idx = sdf.index.tolist()
    geometry.X_group[group][idx_chan:idx_chan+n_channels] += np.mean(X_power[beh_idx, :, :], axis=0)
idx_chan += n_channels
np.mean(X_power[beh_idx, :, :], axis=0).shape

fig, axs = plt.subplots(4, 5, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, (group, power_data) in enumerate(geometry.X_group.items()):
    im = axs[i].imshow(power_data[:, -1, :], aspect='auto', cmap='jet', origin='lower')
    axs[i].set_title(group)
    plt.colorbar(im, ax=axs[i])
plt.tight_layout()
plt.show()

n_bands = len(FREQUENCY_BANDS)
n_channels = geometry.group_loaders.n_channels

X = np.zeros((len(geometry.X_group) * n_timepoints * n_bands, n_channels))
group_labels = []
time_labels = []
band_labels = []    
for i, (group_name, power_data) in enumerate(geometry.X_group.items()):
    for b in range(n_bands):
        for t in range(n_timepoints):
            # Calculer l'indice de ligne
            row_idx = (i * n_timepoints * n_bands) + (b * n_timepoints) + t                
            X[row_idx, :] = power_data[:, b, t]
            group_labels.append(group_name)
            time_labels.append(t)
            band_labels.append(b)
group_labels = np.array(group_labels)
time_labels = np.array(time_labels)
band_labels = np.array(band_labels)


np.save("pca_matrix.npy", X)

df_pca = pd.DataFrame({"time": time_labels, "band": band_labels})
df_pca["stimstable"] = group_labels[:, 0]
df_pca["explor"] = group_labels[:, 1]
df_pca["switch_type"] = group_labels[:, 2]

df_pca.to_csv("pca_metadata.csv", index=False)





plt.imshow(geometry.X_band[group][:48, 2, :], aspect='auto', cmap='jet', origin='lower')
plt.colorbar()
plt.show()

df = geometry.format_exploration(2, correct_treshold=10)
X_power = geometry._get_powerband("transition", 2, n_jobs=-1)

anat = geometry.group_loaders.anatomy_data.copy()
anat = anat[anat["subject"] == 2]
vmpfc_idx = anat[anat["region"] == "OFC"]["chan_idx"].values


switch = df[df["firstswitch"] == 1].index

switch_vmpfc = X_power[switch, :, :, :].copy()
switch_vmpfc = switch_vmpfc[:, vmpfc_idx, :, :]
switch_vmpfc = np.mean(switch_vmpfc, axis=(0, 1))

preswitch = switch - 1
preswitch = preswitch[preswitch >= 0]

preswitch_vmpfc = X_power[preswitch, :, :, :].copy()
preswitch_vmpfc = preswitch_vmpfc[:, vmpfc_idx, :, :]
preswitch_vmpfc = np.mean(preswitch_vmpfc, axis=(0, 1))


fig, axs = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, fr_band in enumerate(FREQUENCY_BANDS):
    axs[i].plot(switch_vmpfc[i, :])
    axs[i].plot(preswitch_vmpfc[i, :], linestyle='--')
    axs[i].set_title(f"VMPFC - {fr_band}")
    axs[i].axhline(0, color='black', linestyle='--')
plt.show()


group_indices = {}
for name, group in df.groupby(["is_stimstable", "test_explor_exploit", "switch_type"]):
    group_indices[name] = group.index.tolist()

group_power = {}
for name, indices in group_indices.items():
    group_power[name] = np.mean(X_power[indices, :, :, :], axis=0)

X = np.zeros((len(group_power) * n_timepoints * n_frband, len(anat)))
group_labels = []
time_labels = []
band_labels = []    
for i, (group_name, power_data) in enumerate(group_power.items()):
    for b in range(n_frband):
        for t in range(n_timepoints):
            row_idx = (i * n_timepoints * n_frband) + (b * n_timepoints) + t                
            X[row_idx, :] = power_data[:, b, t]
            group_labels.append(group_name)
            time_labels.append(t)
            band_labels.append(b)
group_labels = np.array(group_labels)
time_labels = np.array(time_labels)
band_labels = np.array(band_labels)


from sklearn.decomposition import PCA

n_components = 5
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio*100:.2f}%")
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])


pca_df['stim'] = group_labels[:, 0]
pca_df['explor'] = group_labels[:, 1]
pca_df['switch'] = group_labels[:, 2]
pca_df['time'] = time_labels
pca_df["stim"] = pd.to_numeric(pca_df["stim"])

pca_df["band"] = band_labels
pca_df['band_name'] = [list(FREQUENCY_BANDS.keys())[b] for b in band_labels]




period_vector = []
for i in range(1280):
    if i < geometry.group_loaders.event["fb_prev"]:
        period_vector.append("early")
    elif i <  geometry.group_loaders.event["onset"]:
        period_vector.append("pre_stim")
    elif i <  geometry.group_loaders.event["rt"]:
        period_vector.append("action_selection")
    elif i <  geometry.group_loaders.event["fb_next"]:
        period_vector.append("fb_waiting")
    else:
        period_vector.append("end")

pca_df["period"] = period_vector*int(len(pca_df)/len(period_vector))



pca_df.to_csv("pca_df.csv", index=False)


pca_df = pd.read_csv("pca_df.csv")





for (j, (band, ssdf)) in enumerate(pca_df.groupby(["band_name"])):
    fig = plt.figure(figsize=(30, 18))
    fig.suptitle(f"Band: {band}")
    for i, (idx, sdf) in enumerate(ssdf.groupby(["period", "switch", "explor"])):
        # ax = fig.add_subplot(6, 8, i+1)
        ax = fig.add_subplot(5, 6, i+1, projection='3d')
        ax.set_title(f"{idx}")
        for group in sdf['stim'].unique():
            color = palette_dict[group]
            group_data = sdf[sdf['stim'] == group].reset_index()
            group_data = group_data.sort_values('time') 
            ax.scatter(group_data['PC1'], group_data['PC2'],group_data['PC3'], s = 5, color = color)
            ax.plot(group_data['PC1'], group_data['PC2'],group_data['PC3'], label=group, alpha=0.7, lw = 2, color = color)
            #add_events(events_dict, ax) 
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, "dynamic", f"pca_{band}.pdf"), transparent=True, bbox_inches='tight')
    plt.close(fig)
    # plt.show()

contribution = pd.DataFrame({f'PC{i+1}' : pca.components_[i, :] for i in range(n_components)})
contribution["channel"] = geometry.group_loaders.anatomy_data["name"].values
contribution["roi_name"] = geometry.group_loaders.anatomy_data["region"].values
contribution["channel_position"] = geometry.group_loaders.anatomy_data["chan_idx_global"].values
contribution["subject"] = geometry.group_loaders.anatomy_data["subject"].values



contribution["area_order"] = geometry.group_loaders.anatomy_data["area_order"].values
ordered_sdf = contribution.sort_values(by='area_order')
mat = ordered_sdf[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']].values
roi_names = ordered_sdf["roi_name"].values
change_indices = [0]  # Premier élément toujours affiché
for i in range(1, len(roi_names)):
    if roi_names[i] != roi_names[i-1]:
        change_indices.append(i)
custom_labels = [""] * len(roi_names)
for idx in change_indices:
    custom_labels[idx] = roi_names[idx]



sns.heatmap(np.abs(mat), cmap='turbo', vmin=0, vmax=0.35, yticklabels=custom_labels)
plt.show()
















analysis = DecodingAnalysis(subjects, "transition")

analysis.load_matrix(VAR_LIST)

n_clusters = 5
for var in VAR_LIST:
    analysis.clustering_timecourses(var, n_clusters)

# var = "entropy"
for var in VAR_LIST:
    analysis.create_brainplot(var, "Spectral_r", metric="percent_region", save=True)

for subject in subjects:
    for var in ["reliability", "entropy", "action_value", "update_reliability"]:
        analysis.load_cross_decoding_matrix(subject, var)



for subject in subjects:
    analysis.compute_cross_decoding_matrix(subject, ["update_reliability", "rpe"])

analysis

fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for (i, var) in enumerate(["reliability", "entropy", "action_value", "update_reliability"]) :
    mat = np.mean([analysis.cross_decoding_indiv[(subject, var)] for subject in subjects], axis=0)
    # mat = analysis.cross_decoding_indiv[(4, var)]
    lim = np.max(np.abs(mat))
    im = axs[i].imshow(np.arctanh(mat), cmap='jet', origin='lower', aspect='auto', vmin=0, vmax=lim)
    plt.colorbar(im, ax=axs[i])
    axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)
    axs[i].set_yticks(ticks=arranged_timearray, labels=timearray)
    axs[i].set_title(var)
    axs[i].axvline(onset, color='black', linestyle='--')
    axs[i].axvline(fb_prev, color='dimgrey', linestyle='--', lw = 1)
    axs[i].axvline(rt_idx, color='dimgrey', linestyle='--', lw = 1)
    axs[i].axvline(fb, color='dimgrey', linestyle='--', lw = 1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "decoding", "group", "cross_decoding.pdf"), transparent=True, bbox_inches='tight')
plt.show()


from sklearn.metrics import silhouette_score, calinski_harabasz_score

var = "reliability"
df_score = pd.DataFrame()
# for test_met in ["complete", "ward"]:
    # for i in range(2, 6):







fig, axs = plt.subplots(n_clusters, 8, figsize=(20, 12), sharex=True)
# fig, axs = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
# axs = axs.flatten()
for j, var in enumerate(VAR_LIST):
    X = analysis.group_loaders.channel_score[var].T
    lim = np.max(np.abs(X))
    for i in range(1, n_clusters+1):
        chan_idx = analysis.cluster_df[analysis.cluster_df[f"cluster_{var}"] == i]["chan_idx_global"].values
        X_cluster = X[chan_idx]
        # mean = np.mean(X_cluster, axis=0)
        # axs[j].plot(mean, linewidth=2)
        im = axs[i-1, j].imshow(X_cluster, aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=lim)
        plt.colorbar(im, ax=axs[i-1, j])
        axs[i-1, j].set_title(f"Cluster {i} - {var}")

plt.tight_layout()
plt.show()


VAR_LIST


total_count = analysis.cluster_df.groupby(["region"]).size().reset_index(name="total_count")





analysis.cluster_df.groupby(["region", f"cluster_{var}"]).size().reset_index(name="count")

i = 2
# test = analysis.cluster_df[(analysis.cluster_df[f"cluster_{var}"] ==  1) | (analysis.cluster_df[f"cluster_{var}"] == 3)]
test = analysis.cluster_df[(analysis.cluster_df[f"cluster_{var}"] == i)]
count = analysis.cluster_df.groupby(["region"])[f"max_{var}"].mean().reset_index(name="count")
count
count = test.groupby(["region"]).size().reset_index(name="count")
count = count.merge(total_count, on="region", how="left")
count["ratio"] = count["count"] / count["total_count"]
count["ratio_cluster"] = count["count"] / count["count"].sum()
count
# custom_cmap = create_customcmap(count["ratio"].min(), count["ratio"].max(), , "jet")
atlas_filtered = np.zeros(resampled_atlas.shape)
for (i, (idx, row)) in  enumerate(count.iterrows()) :
    sdf = atlas_ref[atlas_ref["ROI_glasser_2"] == row["region"]]
    # atlas_filtered[resampled_atlas == row["roi_n"]] = row["count"]
    for roi_n in sdf["ROI_n"].values:
        atlas_filtered[resampled_atlas == roi_n] = row["count"]
atlas_filtered[atlas_filtered == 0] = np.nan


brain = mne.viz.Brain(
    "fsaverage",
    "rh",
    "pial",
    cortex="ivory",
    background="white",
    size=(800, 500),
    alpha=0.5,
)
brain.add_data(
    atlas_filtered,
    colormap="viridis", 
    alpha=1, 
    colorbar=True,
    fmin = count["count"].min(),
    fmax = count["count"].max(),
)




















        # score_ch = calinski_harabasz_score(analysis.group_loaders.channel_score[var].T, clusters_label)
        # score_silhouette = silhouette_score(analysis.group_loaders.channel_score[var].T, clusters_label)
        # tmp = pd.DataFrame({
        #     "n_clusters": i,
        #     "calinski_harabasz_score": score_ch,
        #     "silhouette_score": score_silhouette,
        #     "variable": var
        # }, index=[0])
        # df_score = pd.concat([df_score, tmp], ignore_index=True)
        # print(f"n_clusters: {i}, Variable: {var}, Calinski-Harabasz score: {score_ch}, Silhouette score: {score_silhouette}")



sns.stripplot(data=df_score, x="variable", y="silhouette_score", hue="n_clusters")
plt.show()

print(f"n_clusters: {i}, Clustering method: {test_met}, Calinski-Harabasz score: {score}")
analysis.plot_clustered_tc(var, clusters_label)


n_clusters = len(np.unique(clusters_label))

X = analysis.group_loaders.channel_score[var]
lim = np.max(np.abs(X))
fig, axs = plt.subplots(1, n_clusters, figsize=(21, 12))
axs = axs.flatten() 
for cluster in range(1, n_clusters + 1):
    cluster_idx = np.where(clusters_label == cluster)[0]
    cluster_data = X[:, cluster_idx]
    im = axs[cluster - 1].imshow(cluster_data.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=axs[cluster - 1])
plt.tight_layout()
plt.show()



unique, counts = np.unique(clusters_label, return_counts=True)
cluster_counts = dict(zip(unique, counts))
print("Cluster counts:", cluster_counts)

    # analysis.plot_clustered_tc("reliability", clusters_label)



    # analysis.plot_clustered_tc("reliability", clusters_label)














subject = 2
var = "reliability"

file_path = os.path.join(OUTPUT_DIR, f"{model_name}", "transition", f"sub-{int(subject):03}_{var}_{model_name}_fold-{fold}_betas-global.npy")
# betas.shape

# mat = np.mean([analysis.cross_decoding_indiv[(subject, var)] for subject in subjects], axis=0)
# ind = np.unravel_index(np.argsort(mat, axis=None), mat.shape)


analysis.group_loaders.anatomy_data.sort_values(by=["area_order"], inplace=True)
regions = analysis.group_loaders.anatomy_data["region"]
fr_band = list(FREQUENCY_BANDS.keys())


fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, var in enumerate(["reliability", "entropy", "action_value", "update_reliability"]):
    betas = analysis.group_loaders.global_betas[var]
    tc = np.mean(analysis.group_loaders.global_score[var], axis=0)
    idx = np.argmax(tc)
    betas_group_byregion = np.zeros((len(regions.unique()), analysis.group_loaders.n_frband))
    for j, region in enumerate(regions.unique()):
        idx_ch = analysis.group_loaders.anatomy_data[analysis.group_loaders.anatomy_data["region"] == region]["chan_idx_global"].values
        tmp = betas[idx-5:idx+5, :, :]
        betas_group_byregion[j] = np.mean(np.abs(tmp[:, idx_ch , :]), axis=(0, 1))
    sns.heatmap(betas_group_byregion, cmap='jet', cbar=True, yticklabels=regions.unique(), xticklabels=fr_band, vmin=0, ax = axs[i])
    axs[i].set_title(f"{var} - {idx}")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "decoding", "group", f"betas_average10_best.pdf"), transparent = True, bbox_inches='tight')
plt.show()


# cross_decoding_matrix = analysis.group_loaders.cross_decoding_matrices[(subject, var)]

subject = 2
var = "rpe"
analysis.group_loaders.data_loaders[subject].ch_names
power_data = analysis.group_loaders.data_loaders[6].load_power()
beh, event_to_keep = analysis.group_loaders.data_loaders[subject].load_behavior("transition")
features = Features(power_data, None)
features.extract_power_bands()
X_power = features.power_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()

indiv_anat =  analysis.group_loaders.anatomy_data[analysis.group_loaders.anatomy_data["subject"] == subject].copy()
analysis.group_loaders.data_loaders[subject].load_anatomy()

idx_chan = indiv_anat["chan_idx_global"].values
idx_chan.sort()

indiv_betas = analysis.group_loaders.global_betas[var][:, idx_chan, :]

y = beh[var].values
y_mean = np.mean(y)

betas = indiv_betas.reshape(indiv_betas.shape[0], -1)
cross_decoding_matrix = np.zeros((n_timepoints, n_timepoints))
X_test_current_time.shape









# for t_train_idx in tqdm(range(n_timepoints), desc=f"Cross decoding subject {subject} for variable {var}"):
t_train_idx = 1
X_test_current_time = X_power[:, :, :, t_train_idx].reshape(X_power.shape[0], -1)
y_pred_all = X_test_current_time @ betas.T
betas.shape

y_pred_means = np.mean(y_pred_all, axis=0) 
cov = np.dot(y_mean, y_pred_means)
corr = cov / (np.std(y) * np.std(y_pred_all))
cross_decoding_matrix[t_train_idx, :] = np.arctanh(corr)



group_selection = GroupLoader(subjects)
group_selection.load_anatomy()
group_selection.load_behavior("selection")

df = group_selection.beh_data.copy()

df = df[df["rt"] > 0]
df = df[df["training"] == 0]

epis_count = df.groupby(["subject", "epis"]).size().reset_index(name="count")
epis_count = epis_count[epis_count["count"] < 55]
epis_count = epis_count[epis_count["count"] > 15]
filtered_df = df.merge(epis_count, on=["subject", "epis"], how="right")


filtered_df["rt_zscore"] = 0.0
for subject, sdf in filtered_df.groupby("subject"):
    filtered_df.loc[sdf.index, "rt_zscore"] = (sdf["rt"] - np.mean(sdf["rt"])) / np.std(sdf["rt"])

sns.barplot(data=filtered_df, x="fb_prev", y="rt_zscore", alpha=0.5)
plt.show()


mean_rt = np.mean(filtered_df["rt"])
std_rt = np.std(filtered_df["rt"])

filtered_df["rt_unormalized"] = filtered_df["rt_zscore"]*std_rt + mean_rt

sns.barplot(data=filtered_df, x="fb_prev", y="rt_unormalized", alpha=0.5)
plt.show()

from scipy.stats import ttest_ind, ttest_rel

# Test t indépendant entre FB+ et FB-
fb_pos = filtered_df[filtered_df["fb_prev"] == 1]["rt_zscore"]
fb_neg = filtered_df[filtered_df["fb_prev"] == 0]["rt_zscore"]

t_stat, p_value = ttest_ind(fb_pos, fb_neg)
print(f"T-test between FB+ and FB-: t-statistic = {t_stat}, p-value = {p_value}")


sns.barplot(data=filtered_df, x="fb_prev", y="rt_zscore", hue = "explor_exploit")
plt.show()


rt_swtich = pd.DataFrame({
    "rt_zscore" : filtered_df[filtered_df["firstswitch"] == 1]["rt_zscore"].values,
    "switch_position" : 0,
    "fb_prev" : filtered_df[filtered_df["firstswitch"] == 1]["fb_prev"].values,
    "switch_type" : filtered_df[filtered_df["firstswitch"] == 1]["switch_type"].values,
    "stimstable" : filtered_df[filtered_df["firstswitch"] == 1]["is_stimstable"].values
})
rt_postswitch = pd.DataFrame({
    "rt_zscore" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["rt_zscore"].values,
    "switch_position" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["post_hmmsw_trial"].values,
    "fb_prev" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["fb_prev"].values,  
    "switch_type" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["switch_type"].values,
    "stimstable" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["is_stimstable"].values
})
rt_preswitch = pd.DataFrame({
    "rt_zscore" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["rt_zscore"].values,
    "switch_position" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["pre_hmmsw_trial"].values,
    "fb_prev" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["fb_prev"].values,
    "switch_type" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["switch_type"].values,
    "stimstable" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["is_stimstable"].values
})


full_rt = pd.concat([rt_swtich, rt_postswitch, rt_preswitch], ignore_index=True)

sns.barplot(data=full_rt, x="switch_position", y="rt_zscore", alpha=0.5)
plt.show()



fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, switch_type in enumerate(full_rt["switch_type"].unique()):
    subset = full_rt[full_rt["switch_type"] == switch_type]
    color = switch_palette[switch_type]
    sns.barplot(data=subset, x="switch_position", y="rt_zscore", 
                alpha=0.5, ax=axes[i], color=color, label = switch_type)

plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, stable in enumerate(full_rt["stimstable"].unique()):
    subset = full_rt[full_rt["stimstable"] == stable]
    color = palette_dict[stable]
    sns.barplot(data=subset, x="switch_position", y="rt_zscore", 
                alpha=0.5, ax=axes[i], color=color, label = rule_changes[stable])

plt.tight_layout()
plt.show()


subset = full_rt[full_rt["stimstable"] == 1]

non_overlap = subset[subset["switch_type"] != "overlap"]
overlap = subset[subset["switch_type"] == "overlap"]

sns.lineplot(data=non_overlap, x="switch_position", y="rt_zscore",  color=stable_color, label="Non-overlap", errorbar="se", linestyle='--')
sns.lineplot(data=overlap, x="switch_position", y="rt_zscore", color=stable_color, label="Overlap", errorbar="se")
plt.axhline(0, color='black', linestyle='--', lw = 0.5)
plt.legend()
plt.show()













sns.kdeplot(df_rtfiltered["rt_zscore"])
plt.show()

df_rtfiltered["rt_test"] = (df_rtfiltered["rt_zscore"]*np.std(df_rtfiltered["rt"]) + np.mean(df_rtfiltered["rt"]))                          

df_rtfiltered = df_rtfiltered.merge(
    df.groupby("subject")["rt"].agg(['mean', 'std']).reset_index(),
    on="subject",
    suffixes=('', '_original')
)

# Réintégrer vectoriellement
df_rtfiltered["rt_reintegrated"] = (
    df_rtfiltered["rt_zscore"] * df_rtfiltered["std"] + df_rtfiltered["mean"]
)


sns.barplot(data=filtered_df, x="fb_prev", y="rt_zscore")
plt.show()






rt_swtich = pd.DataFrame({
    "rt" : filtered_df[filtered_df["firstswitch"] == 1]["rt_zscore"].values,
    "pos" : 0,
    "fb_prev" : filtered_df[filtered_df["firstswitch"] == 1]["fb_prev"].values,
    "switch_type" : filtered_df[filtered_df["firstswitch"] == 1]["switch_type"].values,
    "stimstable" : filtered_df[filtered_df["firstswitch"] == 1]["is_stimstable"].values
})
rt_postswitch = pd.DataFrame({
    "rt" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["rt_zscore"].values,
    "pos" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["post_hmmsw_trial"].values,
    "fb_prev" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["fb_prev"].values,  
    "switch_type" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["switch_type"].values,
    "stimstable" : filtered_df[(filtered_df["post_hmmsw_trial"] == 1) | (filtered_df["post_hmmsw_trial"] == 2)]["is_stimstable"].values
})
rt_preswitch = pd.DataFrame({
    "rt" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["rt_zscore"].values,
    "pos" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["pre_hmmsw_trial"].values,
    "fb_prev" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["fb_prev"].values,
    "switch_type" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["switch_type"].values,
    "stimstable" : filtered_df[(filtered_df["pre_hmmsw_trial"] == -1) | (filtered_df["pre_hmmsw_trial"] == -2)]["is_stimstable"].values
})

full_rt = pd.concat([rt_swtich, rt_postswitch, rt_preswitch], ignore_index=True)

fig, axs = plt.subplots(2, 3, figsize=(21, 12), sharex=True)
# axs = axs.flatten()
for i, switch_type in enumerate(full_rt["stimstable"].unique()):
    switch_df = full_rt[full_rt["stimstable"] == switch_type]
    switch_df_pos = switch_df[switch_df["fb_prev"] == 1]
    switch_df_neg = switch_df[switch_df["fb_prev"] == 0]
    sns.barplot(data=switch_df_pos, x="pos", y="rt", ax=axs[0, i], color="forestgreen", alpha=0.5, label="FB+")
    sns.stripplot(data=switch_df_pos, x="pos", y="rt", ax=axs[0, i], color="forestgreen", alpha=0.5)
    sns.barplot(data=switch_df_neg, x="pos", y="rt", ax=axs[1, i], color="firebrick", alpha=0.5, label="FB-")
    sns.stripplot(data=switch_df_neg, x="pos", y="rt", ax=axs[1, i], color="firebrick", alpha=0.5)
    axs[0, i].set_title(f"Switch type: {switch_type}")
    axs[1, i].set_xlabel("Switch position")
    axs[1, i].legend()
plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "rt_switch_type.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()





fig, axs = plt.subplots(1, 3, figsize=(21, 12), sharex=True)
# axs = axs.flatten()
for i, switch_type in enumerate(full_rt["stimstable"].unique()):
    switch_df = full_rt[full_rt["stimstable"] == switch_type]
    sns.barplot(data=switch_df, x="pos", y="rt", ax=axs[i], color="royalblue", alpha=0.5, label="FB+")
    sns.stripplot(data=switch_df, x="pos", y="rt", ax=axs[i], color="royalblue", alpha=0.5)
    axs[i].set_title(f"Switch type: {switch_type}")
    axs[i].set_xlabel("Switch position")
    axs[i].legend()
plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "rt_switch_type.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()



fig, axs = plt.subplots(3, 3, figsize=(21, 12), sharex=True)
axs = axs.flatten()
for (i, (cat, sdf)) in enumerate(full_rt.groupby(["stimstable","switch_type"])):
    sns.barplot(data=sdf, x="pos", y="rt", ax=axs[i], color="royalblue", alpha=0.5)
    sns.stripplot(data=sdf, x="pos", y="rt", ax=axs[i], color="royalblue", alpha=0.5)
    axs[i].set_title(cat)
plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "behaviour", "rt_switch_type.pdf"), transparent=True, format='pdf', bbox_inches='tight')
plt.show()














import statsmodels.api as sm

df_model = filtered_df.copy()
df_model["log_rt"] = np.log(df_model["rt"])
sns.kdeplot(df_model["log_rt"])
plt.show()

df_model["rt_zscore"]

formula = 'zlog_rt ~ fb_prev'
model_basic = smf.ols(formula, data=df_model).fit()

print("=== MODÈLE DE BASE ===")
print(model_basic.summary())

model_basic.params[model_basic.params.index.str.contains('fb_prev')]
print("\nCoefficients des feedbacks:")
print(fb_coeffs.sort_values(ascending=False))

from statsmodels.regression.mixed_linear_model import MixedLM

df_model

model_mixed = MixedLM.from_formula(
    'log_rt ~ fb_prev', 
    data=df_model, 
    groups=df_model['subject']
).fit()

print("=== MODÈLE À EFFETS MIXTES ===")
print(model_mixed.summary())


fig, axes = plt.subplots(2, 2, figsize=(12, 10))


sns.boxplot(data=df_model, x='fb_prev', y='log_rt')
sns.stripplot(data=df_model, x='fb_prev', y='log_rt', size=1, alpha=0.1, color='black')
plt.show()


# Graphique des effets principaux
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Effet du feedback précédent
sns.boxplot(data=df_model, x='fb_prev', y='rt', ax=axes[0,0])
axes[0,0].set_title('Effet du feedback précédent sur RT')
axes[0,0].set_ylabel('Temps de réaction (s)')

plt.bar(range(len(fb_coeffs)), fb_coeffs.values)
plt.show()

# 2. Coefficients du modèle
# fb_coeffs_clean = fb_coeffs.drop(fb_coeffs.index[0])  # Enlever l'intercept
axes[0,1].bar(range(len(fb_coeffs)), fb_coeffs.values)
axes[0,1].set_xticks(range(len(fb_coeffs)))
axes[0,1].set_xticklabels(fb_coeffs.index, rotation=45)
axes[0,1].set_title('Coefficients des feedbacks')
axes[0,1].set_ylabel('Coefficient (effet sur RT)')

# 3. Résidus vs prédictions
axes[1,0].scatter(model_basic.fittedvalues, model_basic.resid, alpha=0.5)
axes[1,0].axhline(y=0, color='red', linestyle='--')
axes[1,0].set_xlabel('Valeurs prédites')
axes[1,0].set_ylabel('Résidus')
axes[1,0].set_title('Diagnostic: Résidus vs Prédictions')

# 4. Q-Q plot des résidus
sm.qqplot(model_basic.resid, line='s', ax=axes[1,1])
axes[1,1].set_title('Q-Q Plot des résidus')

plt.tight_layout()
plt.show()




models = {
    'Basic': model_basic,
    'Mixed': model_mixed
}

comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    # 'R²_adj': [m.rsquared_adj for m in models.values()],
    'AIC': [m.aic for m in models.values()],
    'BIC': [m.bic for m in models.values()]
})

print("\n=== COMPARAISON DES MODÈLES ===")
print(comparison_df)








group_transition = GroupLoader(SUBJECTS)
group_transition.load_anatomy()
group_transition.load_behavior("transition")

for var in VAR_LIST:
    # group_selection.load_decoding_metric(fold, var, model_name, "selection", "channel")
    # group_selection.load_decoding_metric(fold, var, model_name, "selection", "global")
    group_transition.load_decoding_metric(fold, var, model_name, "transition", "channel")
    group_transition.load_decoding_metric(fold, var, model_name, "transition", "global")
    # group_selection.load_decoding_betas(fold, var, model_name,"transition", "global")
    # group_selection.load_decoding_betas(fold, var, model_name,  "transition", "channel")
    # group_transition.load_decoding_betas(fold, var, model_name,"transition", "global")
    # group_transition.load_decoding_betas(fold, var, model_name,  "transition", "channel")
















group_transition.anatomy_data.sort_values(by=["chan_idx_global"], inplace=True)

rt = analysis.group_loaders.beh_data[analysis.group_loaders.beh_data["rt"] > 0]["rt"].values
mean_rt = np.mean(rt)
sem_rt = np.std(rt)/np.sqrt(len(subjects))
sem__rt_idx = int(sem_rt*sr_decimated)
rt_idx = int(onset + mean_rt*sr_decimated)
fb_prev = onset - sr_decimated
fb = int((rt_idx + sr_decimated))
jitter_fb = 0.2*sr_decimated

sns.pairplot(group_selection.beh_data[VAR_LIST], 
             vars=VAR_LIST, 
             kind="scatter",
             plot_kws={'alpha': 0.3, 's': 1, 'color': 'black'},
             diag_kind="kde",
             diag_kws={'fill': True, 'color': 'royalblue', 'alpha': 0.5},
             height=1.5,
             aspect=1.5)
plt.savefig(os.path.join(FIGURES_DIR, "behaviour", f"pairplot_selection.png"), transparent=True, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(3, 3, figsize=(20, 12))
axs = axs.flatten()
for i, var in enumerate(VAR_LIST):
    x = group_transition.beh_data[var]
    # x_norm = (x - np.mean(x)) / np.std(x)
    y = group_selection.beh_data[var]
    # y_norm = (y - np.mean(y)) / np.std(y)
    axs[i].scatter(x, y, color="black", alpha=0.3, s = 1)
    axs[i].set_title(var)
    axs[i].set_xlabel("Transition")
    axs[i].set_ylabel("Selection")
axs[-1].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "behaviour", f"cov_selection_transition_.pdf"), transparent =True, bbox_inches='tight')
plt.show()


plt.fill_between

fig, axs = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, var in enumerate(VAR_LIST):
    indiv_t = group_transition.global_score[var]
    average_t = np.mean(indiv_t, axis=0)
    sem_t = np.std(indiv_t, axis=0) / np.sqrt(indiv_t.shape[0])
    axs[i].plot(indiv_t.T, color="dimgray", linewidth=0.5, alpha=0.3)
    axs[i].plot(average_t, color="black", linewidth=2)
    axs[i].fill_between(np.arange(len(average_t)), average_t - sem_t, average_t + sem_t, alpha=0.2, color="black")
    axs[i].set_title(var)
    axs[i].axvline(onset, color='red', linestyle='--')
    axs[i].axhline(0, color='black', linestyle='--')
    axs[i].axvline(fb_prev, color='dimgrey', linestyle='--', lw = 1)
    axs[i].fill_between(np.arange(rt_idx - sem__rt_idx, rt_idx +sem__rt_idx), -0.15, 0.75, alpha=0.2, color='dimgrey')
    axs[i].axvline(rt_idx, color='dimgrey', linestyle='--', lw = 1)
    axs[i].axvline(fb, color='dimgrey', linestyle='--', lw = 1)
axs[0].set_xticks(ticks=arranged_timearray, labels=timearray)
axs[0].set_ylim(-0.15, 0.75)
axs[-1].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "decoding", "group", f"average_global_selection_transition.pdf"), transparent =True, bbox_inches='tight')
plt.show()


fig, axs = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, var in enumerate(VAR_LIST):
    # mat_s = group_selection.channel_score[var][:, group_selection.anatomy_data["chan_idx_global"]]
    mat_t = group_transition.channel_score[var][:, group_transition.anatomy_data["chan_idx_global"]]
    lim = np.max([np.abs(mat_s), np.abs(mat_t)])
    # im_s = axs[i].imshow(mat_s.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
    im_t = axs[i].imshow(mat_t.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim, alpha=0.5)
    plt.colorbar(im_s, ax=axs[i])
    axs[i].set_title(var)
    axs[i].axvline(onset, color='black', linestyle='--')
    axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)
    # axs[i].set_yticks(ticks=np.arange(len(group.anatomy_data)), labels=group.anatomy_data["region"])
plt.tight_layout()
plt.show()



np.max([np.abs(mat_s), np.abs(mat_t)])





fig, axs = plt.subplots(1, len(VAR_LIST) , figsize=(3*len(VAR_LIST), 12), sharex=True, sharey=True)
# lim = np.max([np.abs(group_transition.channel_score[var]) for var in VAR_LIST])
for i, var in enumerate(VAR_LIST):
    mat_s = group_transition.channel_score[var][:, group_transition.anatomy_data["chan_idx_global"]]
    lim = np.max(np.abs(mat_s))
    im_s = axs[i].imshow(mat_s.T, aspect='auto', cmap='jet',         
                          interpolation='nearest', vmin=0, vmax=lim)
    axs[i].set_title(f"{var}")
    axs[i].axvline(fb_prev, color='white', linestyle='--')
    axs[i].axvline(onset, color='white', linestyle='-')
    axs[i].axvline(rt_idx, color='white', linestyle='--')
    axs[i].axvline(fb, color='white', linestyle='--')
    plt.colorbar(im_s, ax=axs[i])

        # mat_t = group_transition.channel_score[var][:, group_transition.anatomy_data["chan_idx_global"]]
    # mat_diff = mat_s - mat_t  # Différence entre sélection et transition
    # lim = np.max([np.abs(mat_s), np.abs(mat_t)])
    # diff_lim = np.max(np.abs(mat_diff))
    # im_t = axs[1, i].imshow(mat_t.T, aspect='auto', cmap='jet',
    #                       interpolation='nearest', vmin=-lim, vmax=lim)
    # plt.colorbar(im_t, ax=axs[1, i])
    # axs[1, i].set_title(f"{var} - Transition")
    # axs[1, i].axvline(onset, color='black', linestyle='--')

    # im_diff = axs[2, i].imshow(mat_diff.T, aspect='auto', cmap='RdBu_r',
    #                          interpolation='nearest', vmin=-diff_lim, vmax=diff_lim)
    # plt.colorbar(im_diff, ax=axs[2, i])
    # axs[2, i].set_title(f"{var} - Différence (Sélection - Transition)")
    # axs[2, i].axvline(onset, color='black', linestyle='--')

# for i, var in enumerate(VAR_LIST):
#     for j in range(3):
# plt.colorbar(im_s, ax=axs[-1])
axs[0].set_xticks(ticks=arranged_timearray, labels=timearray)
    # if i == 0:  # Seulement pour la première ligne
    #     for j in range(3):
regions = group_transition.anatomy_data["region"].values

unique_regions = np.unique(regions)
# find where new regions start from regions vector
first_occurrences_idx = np.where([regions[i] != regions[i-1] for i in range(1, len(regions))])[0]
first_occurrences_idx = np.insert(first_occurrences_idx,len(first_occurrences_idx), len(regions)-1)
axs[0].set_yticks(first_occurrences_idx)
axs[0].set_yticklabels(regions[first_occurrences_idx], rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "decoding", "group", f"channel_decoding_transition.pdf"), 
           transparent=True, bbox_inches='tight', dpi=300)
plt.show()




group_selection.load_decoding_betas(fold, "reliability", model_name, "channel")
group.load_decoding_metric(fold, "reliability_max", model_name, "channel")

fig, axs = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, var in enumerate(VAR_LIST):
    test = group.build_region_average(var, "channel", "metric")
    for key, value in test.items():
        max_value = np.max(value)
        if max_value > 0.05:
            color = area_colors[key]
            axs[i].plot(value, label=key, color=color)
    axs[i].legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    axs[i].set_title(var)
plt.tight_layout()
plt.show()
np.argmax(np.mean(group.global_score["reliability"], axis=0))   

group.channel_betas["reliability_max"]

fr_band = list(FREQUENCY_BANDS.keys())
regions = analysis.group_loaders.anatomy_data["region"].values  

fig, axs = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, var in enumerate(VAR_LIST) : 
    # idx = np.argmax(np.mean(group_selection.global_score[var], axis = 0))
    idx = np.argsort(np.mean(group_transition.global_score[var], axis = 0))[-10:]
    betas_group_byregion = np.zeros((len(regions.unique()), group_transition.n_frband))
    for j, region in enumerate(regions.unique()):
        idx_ch = analysis.group_loaders.anatomy_data[analysis.group_loaders.anatomy_data["region"] == region]["chan_idx_global"].values
        tmp = betas[ind, :, :]
        betas_group_byregion[j] = np.mean(np.abs(tmp[:,idx_ch , :]), axis=(0, 1))
    sns.heatmap(betas_group_byregion, cmap='jet', cbar=True, yticklabels=regions.unique(), xticklabels=fr_band, ax=axs[i])
    axs[i].set_title(var)
axs[-1].axis('off')
plt.tight_layout()
# fig.savefig(os.path.join(FIGURES_DIR, "decoding", "group", f"betas_average10_transition.pdf"), transparent =True, bbox_inches='tight')
plt.show()

group_transition.global_betas[var]

np.mean(np.abs(group_selection.channel_betas[var][idx, idx_ch, :]), axis=(0, 1))
np.argsort(np.mean(group_selection.global_score[var], axis = 0))[-10:]
group_selection.channel_betas[var][idx, idx_ch, :]

np.mean(np.abs(group_selection.channel_betas[var][idx, idx_ch, :]), axis=(0, 1))

group_selection.channel_betas[var].shape


area_order = [area_dict[roi] for roi in group_transition.anatomy_data["region"]]
group_transition.anatomy_data["area_order"] = area_order

group_selection.anatomy_data.sort_values(by=["area_order"], inplace=True)


fig, axs = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, var in enumerate(VAR_LIST):
    to_plot = group.channel_score[var][:, group.anatomy_data["chan_idx_global"]]
    lim = 0.8*np.max(np.abs(to_plot))
    im = axs[i].imshow(to_plot.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=axs[i])
    axs[i].set_title(var)
    axs[i].axvline(onset, color='black', linestyle='--')
    axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)
    # axs[i].set_yticks(ticks=np.arange(len(group.anatomy_data)), labels=group.anatomy_data["region"])
plt.tight_layout()
plt.show()



fig, axs = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, var in enumerate(VAR_LIST):
    indiv = group.global_score[var]
    average = np.mean(indiv, axis=0)
    sem = np.std(indiv, axis=0) / np.sqrt(indiv.shape[0])
    axs[i].plot(indiv.T, label="Individual", color="dimgrey", linewidth=0.3, alpha=0.3)
    axs[i].plot(average, label="Average", color="black", linewidth=2)
    axs[i].fill_between(np.arange(len(average)), average - sem, average + sem, alpha=0.2, color="black")
    axs[i].set_title(var)
    axs[i].set_ylim(-0.1, 0.7)
    axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)
    axs[i].axvline(onset, color='red', linestyle='--')
    axs[i].axhline(0, color='black', linestyle='--')
plt.tight_layout()
plt.show()





sns.heatmap(group.channel_score["reliability"][group.anatomy_data["chan_idx_global"]].T, cmap='jet', cbar=True,)
plt.show()




plt.figure(figsize=(12, 12))
sns.heatmap(betas_group_byregion, cmap='jet', cbar=True, yticklabels=regions.unique(), xticklabels=fr_band)
plt.show()

from analysis.decoding.loader import iEEGDataLoader
from analysis.decoding.process_features import Features
subject = 2
loader = iEEGDataLoader(subject)
power_data = loader.load_power()
beh, event_to_keep = loader.load_behavior("transition")


features = Features(power_data, None)

features.extract_power_bands()
X_power = features.power_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()




from tqdm import tqdm
from scipy.stats import pearsonr

idx_chan = test_anat["chan_idx_global"].values
idx_chan.sort()

test_anat =  group_transition.anatomy_data[group_transition.anatomy_data["subject"] == 2]
test_betas = group_transition.global_betas["rpe"][:, idx_chan, :]
# test_betas = group_transition.global_betas["rpe"][:, test_anat["chan_idx_global"].values, :]
test_beh = group_transition.beh_data[group_transition.beh_data["subject"] == 2]

y = beh["rpe"].values

betas = test_betas.reshape(test_betas.shape[0], -1)
cross_decoding_matrix = np.zeros((n_timepoints, n_timepoints))


for t_train_idx in tqdm(range(n_timepoints), desc="Temps d'entraînement"):
    X_test_current_time = X_power[:, :, :, t_train_idx].reshape(X_power.shape[0], -1) 
    for t_test_idx in range(n_timepoints):
        current_betas = betas[t_test_idx, :]  
        y_pred = X_test_current_time @ current_betas
        score, _ = pearsonr(y, y_pred)
        cross_decoding_matrix[t_train_idx, t_test_idx] = score






betas = test_betas.reshape(test_betas.shape[0], -1)  # (n_timepoints, n_features)
n_timepoints = betas.shape[0]
cross_decoding_matrix = np.zeros((n_timepoints, n_timepoints))

# Pré-calcul des valeurs pour y (cible)
y_mean = np.mean(y)
y_centered = y - y_mean
y_norm = np.sqrt(np.sum(y_centered**2))


for t_train_idx in tqdm(range(n_timepoints), desc="Temps d'entraînement (vectorisé)"):
    X_test_current_time = X_power[:, :, :, t_train_idx].reshape(X_power.shape[0], -1)
    y_pred_all = X_test_current_time @ betas.T
    y_pred_means = np.mean(y_pred_all, axis=0) 
    y_pred_centered = y_pred_all - y_pred_means
    y_pred_norms = np.sqrt(np.sum(y_pred_centered**2, axis=0))  # (n_timepoints,)
    numerators = np.dot(y_centered, y_pred_centered)

    mask_valid = (y_pred_norms != 0)
    scores = np.zeros(n_timepoints)
    scores[mask_valid] = numerators[mask_valid] / (y_norm * y_pred_norms[mask_valid])
    
    cross_decoding_matrix[t_train_idx, :] = scores

max_score = np.max(cross_decoding_matrix)
percent_cross_decoding = np.abs(cross_decoding_matrix)/max_score
np.where(percent_cross_decoding > 0.8)
plt.imshow(percent_cross_decoding, cmap='jet', origin = 'lower', aspect='auto')
plt.colorbar()
plt.show()


from sklearn.preprocessing import StandardScaler


X_test_current_time = X_power[:, :, :, 528].reshape(X_power.shape[0], -1)
X_test_scaled = StandardScaler().fit_transform(X_test_current_time)

y_pred = X_test_scaled @ betas[528, :]

score, _ = pearsonr(y, y_pred)
plt.scatter(y, y_pred, alpha=0.3, s=1)
plt.show()

np.argmax(group_transition.global_score["entropy"][0])





from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X = group_transition.channel_score["reliability"].T
# area_ordered = group_transition.anatomy_data["chan_idx_global"].values
# X_ordered = X[area_ordered, :]
dist_matrix = pdist(X, metric='euclidean')
dist_square = squareform(dist_matrix)

plt.imshow(dist_square, cmap='jet', origin='lower', aspect='auto')
plt.colorbar()
plt.show()

group_transition.anatomy_data.sort_values(by=["chan_idx_global"], inplace=True)


group_transition.anatomy_data[group_transition.anatomy_data['cluster'] == 3]["region"]
# Obtenir le nombre de clusters uniques

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score


all_data = []
for var in ["reliability", "entropy"]:
    print(f"Clustering for variable: {var}")
    X = group_transition.channel_score[var].T
    X_truncated = X
    all_data.append(X_truncated)



X_combined = np.concatenate(all_data, axis=1)
X_combined

plt.imshow(dist_square, cmap='jet', origin='lower', aspect='auto')
plt.colorbar()
plt.show()



# df_clusters = pd.DataFrame()
# for var in VAR_LIST:
# print(f"Clustering for variable: {var}")
var = "entropy"

for var in VAR_LIST:
    X = group_transition.channel_score[var].T
    X_timebin = np.array([np.mean(X[:, moment_cut[i]:moment_cut[i+1]], axis=1) for i in range(len(moment_cut)-1)]).T
    dist_matrix = pdist(X_timebin, metric='sqeuclidean')
    Z = linkage(dist_matrix, method='ward')
    fclusters = fcluster(Z, t=3, criterion='maxclust')
    group_transition.anatomy_data[f'cluster_{var}'] = fclusters

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
fig, axs = plt.subplots(3, 8, figsize=(20, 12), sharex=True)
# axs = axs.flatten()
for i, var in enumerate(VAR_LIST):
    X = group_transition.channel_score[var].T
    fclusters = group_transition.anatomy_data[f'cluster_{var}'].values
    n_clusters = len(np.unique(fclusters))
    lim = np.max(np.abs(X))
    for j in range(1, n_clusters+1):
        cluster_channels = np.where(fclusters == j)[0]
        profile = X[cluster_channels, :]
        # axs[j-1, i].plot(np.linspace(-WOI_start_time, WOI_end_time, n_timepoints), profile.T, color=colors[j-1], alpha=0.1, linewidth=0.1)
        # mean_profile = np.mean(X[cluster_channels, :], axis=0)
        # axs[j-1, i].plot(np.linspace(-WOI_start_time, WOI_end_time, n_timepoints ), mean_profile, color=colors[j-1], linewidth=2)
        axs[j-1, i].imshow(profile, aspect='auto', cmap='jet', origin = 'lower', vmin=0, vmax=lim)
        axs[j-1, i].set_title(f"{var} - Cluster {j}")


plt.tight_layout()
plt.show()


idx = group_transition.anatomy_data[(group_transition.anatomy_data['cluster_reliability'] == 2) | (group_transition.anatomy_data['cluster_entropy'] == 2)]["chan_idx_global"].values

X_e = group_transition.channel_score["entropy"].T[idx, :]
X_r = group_transition.channel_score["reliability"].T[idx, :]
plt.imshow(X_e-X_r, aspect='auto', cmap='jet', origin = 'lower', vmin=-0.1, vmax=0.1)
plt.show()








plt.imshow(X, cmap='jet', origin='lower', aspect='auto')
plt.show()


moment_cut = [0, fb_prev, onset, rt_idx, fb, 1280]
X_timebin = np.array([np.mean(X[:, moment_cut[i]:moment_cut[i+1]], axis=1) for i in range(len(moment_cut)-1)]).T
X_timebin_norm = (X_timebin - np.mean(X_timebin, axis=0)) / np.std(X_timebin, axis=0)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
plt.plot(np.abs(pca), 'o', markersize=2, alpha=0.3)
plt.axvline(onset, color='red', linestyle='--')
plt.axvline(fb_prev, color='dimgrey', linestyle='--', lw = 1)
plt.axvline(rt_idx, color='dimgrey', linestyle='--', lw = 1)
plt.axvline(fb, color='dimgrey', linestyle='--', lw = 1)
plt.show()



sns.heatmap(X_timebin_norm, cmap='jet', cbar=True)
# plt.imshow(X_timebin, cmap='jet', origin='lower', aspect='auto')
# plt.colorbar()
plt.show()


ax = plt.figure().add_subplot(projection='3d')
for coord in X_timebin:
    ax.scatter(*coord, color='black', alpha=0.3)
plt.show()


# X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0) 
# X_trunc = X[:, fb_prev:fb]


plt.imshow(dist_square, cmap='jet', origin='lower', aspect='auto')
plt.colorbar()
plt.show()




# for n_clusters in range(2, 10):
n_clusters = 3
spectral = SpectralClustering(
    n_clusters=n_clusters,
    n_neighbors=50,
    affinity='precomputed_nearest_neighbors',
    assign_labels='kmeans',
    random_state=42
)
cluster_labels = spectral.fit_predict(dist_square)
ch_avg = calinski_harabasz_score(X, cluster_labels)
print(f" n clusters: {n_clusters}, calinski score: {ch_avg:.4f}")




    # df_clusters = pd.concat([df_clusters, pd.DataFrame({
    #     'variable': [var],
    #     'n_clusters': [n_clusters],
    #     'silhouette_score': [silhouette_avg]
    # })], ignore_index=True)


# sns.barplot(data=df_clusters, x='n_clusters', y='silhouette_score', hue='variable', palette='deep')
# plt.show()

unique_clusters = np.unique(fclusters)
n_clusters = len(unique_clusters)


# Affichage des régions par cluster
print("Distribution des régions par cluster:")
for cluster_id in np.unique(fclusters):
    regions_in_cluster = group_transition.anatomy_data[
        group_transition.anatomy_data['cluster'] == cluster_id
    ]["region"].value_counts()
    print(f"\nCluster {cluster_id} ({np.sum(fclusters == cluster_id)} canaux):")
    print(regions_in_cluster.to_string())


# for k, var in enumerate(VAR_LIST):
    # X = group_transition.channel_score[var].T

# group_transition.anatomy_data['cluster'] = cluster_labels

# X = group_transition.channel_score[var].T
fig, axs = plt.subplots(1, n_clusters, figsize=(20, 12), sharex=True, sharey=True)

for i in range(1, n_clusters+1):
    cluster_channels = np.where(fclusters == i)[0]
    # for j, chan_idx in enumerate(cluster_channels):
    profile = X[cluster_channels, :]
    axs[i-1].plot(np.linspace(-WOI_start_time, WOI_end_time, n_timepoints), profile.T, color="grey", alpha=0.3, linewidth=0.3)
    mean_profile = np.mean(X[cluster_channels, :], axis=0)
    axs[i-1].plot(np.linspace(-WOI_start_time, WOI_end_time, n_timepoints ), mean_profile, color='black', linewidth=2)
    axs[i-1].set_title(f"{var} - Cluster {i}")
    axs[i-1].axhline(0, color='red', linestyle='--')

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, n_clusters, figsize=(20, 12), sharex=True)
lim = np.max(np.abs(X))
for i in range(1, n_clusters+1):
    cluster_channels = np.where(fclusters == i)[0]
    # for j, chan_idx in enumerate(cluster_channels):
    profile = X[cluster_channels, :]
    im = axs[i-1].imshow(profile, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
plt.colorbar(im, ax=axs[i-1])
plt.tight_layout()
plt.show()






# 4. Visualisations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 4a. Dendrogramme
ax1 = axes[0, 0]
dendro = dendrogram(Z, ax=ax1, truncate_mode='lastp', p=30, 
                   show_leaf_counts=True, leaf_font_size=8)
ax1.set_title('Dendrogramme du clustering hiérarchique')
ax1.set_xlabel('Index des clusters ou taille des clusters')
ax1.set_ylabel('Distance de fusion')

# Ligne horizontale pour montrer où on coupe pour obtenir 4 clusters
threshold = Z[-n_clusters+1, 2]  # Distance de fusion pour obtenir n_clusters
ax1.axhline(y=threshold, color='red', linestyle='--', 
           label=f'Seuil pour {n_clusters} clusters')
ax1.legend()

# 4b. Matrice de distance ordonnée par clusters
ax2 = axes[0, 1]
cluster_order = np.argsort(fclusters)
dist_ordered = dist_square[cluster_order][:, cluster_order]
im1 = ax2.imshow(dist_ordered, cmap='viridis', aspect='auto')
ax2.set_title('Matrice de distance ordonnée par clusters')
plt.colorbar(im1, ax=ax2)

# Ajouter des lignes pour séparer les clusters
cluster_boundaries = []
current_cluster = fclusters[cluster_order[0]]
for i, cluster_id in enumerate(fclusters[cluster_order]):
    if cluster_id != current_cluster:
        cluster_boundaries.append(i)
        current_cluster = cluster_id

for boundary in cluster_boundaries:
    ax2.axhline(y=boundary, color='white', linewidth=2)
    ax2.axvline(x=boundary, color='white', linewidth=2)

# 4c. Profils temporels par cluster
ax3 = axes[1, 0]
colors = ['red', 'blue', 'green', 'orange', 'purple'][:n_clusters]

for cluster_id in range(1, n_clusters+1):
    cluster_mask = fclusters == cluster_id
    cluster_data = X[cluster_mask, :]
    
    # Profils individuels en transparence
    for i in range(min(20, cluster_data.shape[0])):  # Limiter à 20 profils par cluster
        ax3.plot(cluster_data[i, :], color=colors[cluster_id-1], alpha=0.2, linewidth=0.5)
    
    # Profil moyen
    mean_profile = np.mean(cluster_data, axis=0)
    ax3.plot(mean_profile, color=colors[cluster_id-1], linewidth=3, 
            label=f'Cluster {cluster_id} (n={np.sum(cluster_mask)})')

ax3.axvline(x=onset, color='black', linestyle='--', alpha=0.7, label='Stimulus onset')
ax3.set_title('Profils temporels par cluster')
ax3.set_xlabel('Temps')
ax3.set_ylabel('Score de décodage')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4d. Heatmap des profils ordonnés par cluster
ax4 = axes[1, 1]
X_ordered = X[cluster_order, :]
im2 = ax4.imshow(X_ordered, cmap='RdBu_r', aspect='auto')
ax4.set_title('Profils temporels ordonnés par cluster')
ax4.set_xlabel('Temps')
ax4.set_ylabel('Canaux (ordonnés par cluster)')
plt.colorbar(im2, ax=ax4)

# Ajouter des lignes pour séparer les clusters
for boundary in cluster_boundaries:
    ax4.axhline(y=boundary, color='black', linewidth=1)

plt.tight_layout()
plt.show()




silhouette_avg = silhouette_score(X, fclusters)
calinski_score = calinski_harabasz_score(X, fclusters)

print(f"\nMétriques de qualité:")
print(f"  Score de silhouette: {silhouette_avg:.3f}")
print(f"  Score de Calinski-Harabasz: {calinski_score:.3f}")

for dist_metric in ['euclidean', 'sqeuclidean', 'seuclidean', "cosine"]:
    print(f"\nClustering avec la distance '{dist_metric}':")
    dist_matrix = pdist(X_timebin, metric=dist_metric)
    Z = linkage(dist_matrix, method='ward')
    for n in range(2, 10):
        print(f"  Nombre de clusters: {n}")
        fclusters = fcluster(Z, t=n, criterion='maxclust')
        
        silhouette_avg = silhouette_score(X, fclusters)
        calinski_score = calinski_harabasz_score(X, fclusters)
        
        print(f"  Score de silhouette: {silhouette_avg:.3f} | score de Calinski-Harabasz: {calinski_score:.3f}")








for n in range(2, 10):
    clusters_n = fcluster(Z, t=n, criterion='maxclust')
    sil_score = silhouette_score(X, clusters_n)
    calinski_score = calinski_harabasz_score(X, clusters_n)
    print(f"  {n} clusters: silhouette = {sil_score:.3f}, calinski = {calinski_score:.3f}")