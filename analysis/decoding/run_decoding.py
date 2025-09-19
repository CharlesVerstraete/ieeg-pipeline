# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""
Module pour charger les données iEEG pour l'analyse
"""

import matplotlib
matplotlib.use('Qt5Agg')

# from decoding.loader import iEEGDataLoader
# from decoding.process_features import Features
# from decoding.decoding import Decoding
# from decoding.config import *


from analysis.decoding.loader import iEEGDataLoader
from analysis.decoding.process_features import Features
from analysis.decoding.decoding import Decoding
from analysis.decoding.config import *

plt.style.use('seaborn-v0_8-paper') 

full_df_selection = pd.DataFrame()
full_df_transtion = pd.DataFrame()
for subject in [2, 3, 4, 5, 8, 9, 12, 14, 16, 19, 20, 23, 25, 28]:
    loader = iEEGDataLoader(subject)
    df, _ = loader.load_behavior("transition")
    df = add_before_pres(df)
    df = add_before_trial(df)
    full_df_transtion = pd.concat([full_df_transtion, df], ignore_index=True)
    df, _ = loader.load_behavior("selection")
    full_df_selection = pd.concat([full_df_selection, df], ignore_index=True)



# for subject in [2, 3, 4, 5, 8, 9, 12, 14, 16, 19, 20, 23, 25, 28]:
    loader = iEEGDataLoader(subject)
    power_data = loader.load_power()
    phase_data = loader.load_phase()

    behavior_df_selection, event_to_keep = loader.load_behavior("selection")
    behavior_df_transition, _ = loader.load_behavior("transition")
    anatomy_df = loader.load_anatomy()
    features = Features(power_data, phase_data)
    features.extract_power_bands()
    X_power = features.power_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()

    decode_selection = Decoding(X_power, behavior_df_selection, anatomy_df)
    decode_selection.create_pipeline(n_folds=5, alphas=np.logspace(-1, 4, 6), classification=False)

    decode_transition = Decoding(X_power, behavior_df_transition, anatomy_df)
    decode_transition.create_pipeline(n_folds=5, alphas=np.logspace(-1, 4, 6), classification=False)


    for i, var in enumerate(VAR_LIST) : 
        decode_transition.run_decoding(var, n_jobs = -1, model_type = "transition")
        decode_transition.run_decoding(var, n_jobs = -1, mode = "channel", model_type = "transition")
        decode_selection.run_decoding(var, n_jobs = -1, model_type = "selection")
        decode_selection.run_decoding(var, n_jobs = -1, mode = "channel", model_type = "selection")
    decode_selection.plot_multi_tc("selection")
    decode_selection.plot_multi_heatmap("selection")
    decode_transition.plot_multi_tc("transition")
    decode_transition.plot_multi_heatmap("transition")


var = "rpe"

decode_selection.run_decoding(var, n_jobs = -1, save_results = False)
decode_transition.run_decoding(var, n_jobs = -1, save_results = False)



plt.plot(decode_transition.results['timepoint'], decode_transition.metric_global_dict[var], lw = 2, color = "forestgreen", label = "transition")
plt.plot(decode_selection.results['timepoint'], decode_selection.metric_global_dict[var], lw = 2, color = "royalblue", label = "selection")
plt.axhline(0.0, color='black', linestyle='--')
plt.axvline(onset, color='red', linestyle='--')
plt.ylim(-0.2, 1)
plt.xticks(ticks=arranged_timearray, labels=timearray)
plt.tight_layout()
plt.legend()
plt.show()

decode_selection.plot_multi_tc(figsize=(18, 10), save = False)


var = "update_counterfactual"

decode_selection.run_decoding(var, n_jobs = -1, mode = "channel",  save_results = False)
decode_transition.run_decoding(var, n_jobs = -1, mode = "channel", save_results = False)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), sharex=True, sharey=True)

to_plot = decode_transition.metric_channel_dict[var][:, decode_transition.ordered_regions_idx]
lim = np.max(np.abs(to_plot))
im = ax1.imshow(to_plot.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
ax1.set_yticks(ticks=np.arange(decode_transition.n_channels), labels=decode_transition.ordered_regions)
plt.colorbar(im, ax = ax1)
ax1.set_title("Transition")

to_plot = decode_selection.metric_channel_dict[var][:, decode_selection.ordered_regions_idx]
lim = np.max(np.abs(to_plot))
im = ax2.imshow(to_plot.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
plt.colorbar(im, ax = ax2)
ax2.set_title("Selection")

plt.tight_layout()
plt.legend()
plt.show()





plt.scatter(behavior_df_selection["action_value"], behavior_df_transition["action_value"], s=5, alpha=0.4)
plt.show()






behavior_df_transition.columns

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), sharex=True, sharey=True)

ax1.scatter(behavior_df_selection["rpe"], behavior_df_selection["update_counterfactual"], s=5, alpha=0.4)
ax1.set_title("Selection")
ax2.scatter(behavior_df_transition["rpe"], behavior_df_transition["update_counterfactual"], s=5, alpha=0.4)
ax2.set_title("Transition")
plt.tight_layout()
plt.show()






full_df_selection




decode_selection.plot_multi_tc(save = False, figsize = (20, 12))







decode.run_decoding(var, n_jobs = -1, mode = "channel",  save_results = False)

to_plot = decode.metric_channel_dict[var][:, decode.ordered_regions_idx]
lim = np.max(np.abs(to_plot))
plt.imshow(to_plot.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
plt.colorbar()
plt.show()





simu_path = os.path.join(CLUSTER_DIR, 'simu', 'selection',f"sub-{subject:03}_task-stratinf_sim-forced.csv")
df = pd.read_csv(simu_path)
df.columns

beh_path = os.path.join(CLUSTER_DIR, 'beh', f"sub-{subject:03}_task-stratinf_beh.tsv")
df = pd.read_csv(beh_path)



decode.results

fig, axs = plt.subplots(nrows=2, ncols=2)
axs = axs.flatten()


fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
axs = axs.flatten()



from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
from scipy.stats import pearsonr
from joblib import Parallel, delayed



def compute_score_power_chann(X_band, t, y, pipeline, cv):
    """Compute the score for a specific time point"""
    tmp_results = {}
    for ch in range(X_band.shape[1]):
        X = X_band[:, ch, :, t].reshape(X_band.shape[0], -1)
        fold_metrics = []
        fold_betas = []
        fold_score = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            r, _ = pearsonr(y_test, y_pred)
            fold_score.append(mean_squared_error(y_test, y_pred))
            fold_metrics.append(np.arctanh(r))
            fold_betas.append(pipeline.named_steps['ridgecv'].coef_)
        tmp_results[ch] = {
            'score': np.mean(fold_score),
            'metric': np.mean(fold_metrics),
            'betas': np.mean(fold_betas, axis=0)
        }
    return {
        't': t,
        'test': tmp_results
    }





alphas = np.logspace(-1, 4, 6)
pipeline = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas)
)
n_splits = 5
cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

n_timepoints = X_power.shape[-1]
n_channels = X_power.shape[1]

tasks = [(t, ch) for t in range(n_timepoints) for ch in range(n_channels)]

y = behavior_df["reliability_max"].values

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(compute_score_power_chann)(X_power, t , y, pipeline, cv) for t in range(n_timepoints)
)


# Réorganiser les résultats dans des matrices
scores_chan = np.zeros((n_timepoints, n_channels))
metrics_chan = np.zeros((n_timepoints, n_channels))

for result in results:
    t = result['t']
    tmp_results = result['test']
    for ch, tmp_result in tmp_results.items():
        scores_chan[t, ch] = tmp_result['score']
        metrics_chan[t, ch] = tmp_result['metric']


lim = np.max(np.abs(metrics_chan))
plt.imshow(metrics_chan.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
plt.colorbar()
scores_chan

regions = anatomy_df['region'].values

# Version améliorée - Grouper par région avec des séparations visuelles
unique_regions = np.unique(regions)
region_boundaries = []
current_pos = 0

# Créer une figure plus grande
plt.figure(figsize=(30, 22))

# Réorganiser les canaux par région
ordered_idx = []
ordered_labels = []

for region in sorted(unique_regions):
    # Trouver les indices des canaux pour cette région
    region_indices = np.where(regions == region)[0]
    
    # Ajouter tous les canaux de cette région à notre liste d'ordre
    ordered_idx.extend(region_indices)
    
    # Marquer la limite de la région
    current_pos += len(region_indices)
    region_boundaries.append(current_pos - 0.5)
    
    # Ajouter une étiquette pour chaque canal
    for i in region_indices:
        ordered_labels.append(f"{region}")

# Réorganiser les données
ordered_chanmap = metrics_chan[:, ordered_idx]

plt.figure(figsize=(20, 12))

# Créer la heatmap
ax = sns.heatmap(ordered_chanmap.T, cmap='jet', center=0, 
                vmin=-lim, vmax=lim, cbar=True)

# Ajouter des lignes horizontales pour séparer les régions
for boundary in region_boundaries[:-1]:  # Éviter d'ajouter une ligne à la fin
    plt.axhline(y=boundary, color='black', linestyle='-', linewidth=4)

# Étiqueter les axes
plt.title("Canal-wise Decoding Performance by Region", fontsize=18)
plt.xlabel("Time (ms)", fontsize=16)
plt.ylabel("Brain Regions", fontsize=16)

# Configurer les ticks des axes
plt.xticks(ticks=arranged_timearray, labels=timearray)
plt.axvline(x=onset, color='black', linestyle='--', linewidth=2)
plt.yticks(ticks=np.arange(len(regions)), labels=np.array(regions)[ordered_idx], rotation=0)


plt.tight_layout()















# behavior_df["subject"]

# fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 12))
# axs = axs.flatten()
# for i, var in enumerate(['reliability_max', 'reliability_choosen', 'update_reliability_max', 'entropy','q_choosen', "rpe"]) :
#     file_path = os.path.join(OUTPUT_DIR, "ridge", f"sub-{int(subject):03}_{var}_ridge_fold-5_metric-all.npy")
#     data = np.load(file_path)
#     axs[i].plot(data)
#     axs[i].axhline(0, color='black', linestyle='--')
#     axs[i].axvline(onset, color='red', linestyle='--')
#     axs[i].set_title(f"{var} time course")
#     axs[i].set_xlabel("Time (ms)")
#     axs[i].set_ylabel("Metric")
#     axs[i].set_ylim(-0.2, 1)
#     axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)

# fig.tight_layout()
# plt.savefig(f"accuracy_tc_ridge_{subject}.png")





# var_list = list(self.metric_dict.keys())
# nvar = len(var_list)
# nrows = int(np.sqrt(nvar))
# ncols = int(np.ceil(nvar / nrows))
# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
# axs = axs.flatten()
# for i, var in enumerate(var_list):
#     axs[i].plot(self.results['timepoint'], self.metric_dict[var])
#     axs[i].axhline(0, color='black', linestyle='--')
#     axs[i].axvline(onset, color='red', linestyle='--')
#     axs[i].set_title(f"{var} {y} time course")
#     axs[i].set_xlabel("Time (ms)")
#     axs[i].set_ylabel(y)
#     axs[i].set_ylim(-0.1, 1)
#     axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)







# fig, ax = plt.subplots(figsize=(30, 19))
# plt.plot(decode.results['timepoint'], decode.results["metric"])
# plt.axhline(0, color='black', linestyle='--')
# plt.axvline(onset, color='red', linestyle='--')
# plt.xlabel("Time (ms)")
# plt.ylim(-0.1, 1)
# plt.xticks(ticks=arranged_timearray, labels=timearray)
# plt.tight_layout()
# plt.savefig(f"accuracy_tc_ridge_{subject}.png")


# betas = decode.results['betas']  # shape (n_timepoints, n_channels, n_freqs)
# y = behavior_df["entropy"].values

# # Préparer les tableaux pour les résultats
# n_timepoints = betas.shape[0]
# n_channels = betas.shape[1]
# channel_r2 = np.zeros((n_timepoints, n_channels))
# channel_corr = np.zeros((n_timepoints, n_channels))

# for t in range(n_timepoints):
#     for ch in range(n_channels):
#         X = X_power[:, ch, :, t].reshape(X_power.shape[0], -1)
#         X_scaled = StandardScaler().fit_transform(X)
#         beta_ch = betas[t, ch, :]
#         y_pred = X_scaled @ beta_ch
#         r, _ = pearsonr(y, y_pred)
#         r2 = r2_score(y, y_pred)
#         channel_r2[t, ch] = r2
#         channel_corr[t, ch] = np.arctanh(r)  # Fisher z-transformation


# lim = np.max(np.abs(channel_corr))
# plt.figure(figsize=(30, 19))
# plt.imshow(channel_corr.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# plt.colorbar()
# plt.savefig(f"channel_correlation_{subject}.png")


# lim = np.max(np.abs(metrics_chan))
# plt.figure(figsize=(30, 19))
# plt.imshow(metrics_chan.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# plt.colorbar()
# plt.savefig(f"channel_correlation_refit_{subject}.png")

# #compare 2 method for channel correlation
# lim = np.max((np.abs(channel_corr), np.abs(metrics_chan)))
# fig, axs = plt.subplots(1, 3, figsize=(40, 19))
# ax1 = axs[0].imshow(channel_corr.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# axs[0].set_title("Direct from betas")
# ax2 = axs[1].imshow(metrics_chan.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# axs[1].set_title("Refit betas")
# difference = channel_corr.T - metrics_chan.T
# lim = np.max(np.abs(difference))
# ax3 = axs[2].imshow(difference, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# axs[2].set_title("Difference")
# plt.colorbar(ax1, ax=axs[0])
# plt.colorbar(ax2, ax=axs[1])
# plt.colorbar(ax3, ax=axs[2])
# plt.savefig(f"channel_correlation_comparison_{subject}.png")



# lim = np.max(np.abs(metrics_chan))
# ch_names = loader.metadata["ch_names"]
# regions = anatomy_df['region'].values
# ordered_idx = np.argsort(regions)
# ordered_chanmap = metrics_chan[:, ordered_idx]

# plt.figure(figsize=(30, 19))
# sns.heatmap(ordered_chanmap.T, cmap='jet', center=0, vmin=-lim, vmax=lim, cbar=True)
# plt.title("Refit betas")
# plt.xlabel("Time (ms)")
# plt.ylabel("Channels")
# plt.xticks(ticks=arranged_timearray, labels=timearray)
# plt.yticks(ticks=np.arange(len(regions)), labels=np.array(regions)[ordered_idx])
# plt.tight_layout()
# plt.savefig(f"channel_correlation_refit_{subject}.png")



# group_corr = np.zeros((len(np.unique(regions)), n_timepoints))
# for i, region in enumerate(np.unique(regions)):
#     idx_r = np.where(regions == region)[0]
#     group_corr[i, :] = np.mean(metrics_chan[:, idx_r], axis=1)

# lim = np.max(np.abs(group_corr))
# plt.figure(figsize=(20, 12))
# plt.imshow(group_corr, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# plt.colorbar()
# plt.axvline(x=onset, color='black', linestyle='--')
# plt.xticks(ticks=arranged_timearray, labels=timearray)
# plt.yticks(ticks=np.arange((len(np.unique(regions)))), labels=np.unique(regions))
# plt.title("Refit betas")
# plt.xlabel("Frequency bands")
# plt.ylabel("Regions")
# plt.tight_layout()
# plt.savefig(f"channel_correlation_refit_group_{subject}.png")




# from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.linear_model import RidgeCV
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, roc_auc_score, f1_score
# from scipy.stats import pearsonr
# from joblib import Parallel, delayed




# def process_timepoint_ridge(t_idx, ch_idx, X_data, y, cv, pipeline):
#     """Calcule le score pour un canal spécifique à un point temporel donné"""
#     X = X_data[:, ch_idx, :, t_idx].reshape(X_data.shape[0], -1)
#     fold_scores = []
#     fold_corr = []
#     fold_betas = []
    
#     for train_idx, test_idx in cv.split(X, y):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
        
#         pipeline.fit(X_train, y_train)
#         y_pred = pipeline.predict(X_test)
#         r, _ = pearsonr(y_test, y_pred)
#         score = r2_score(y_test, y_pred)
#         fold_scores.append(score)
#         fold_corr.append(np.arctanh(r))
#         fold_betas.append(pipeline.named_steps['ridgecv'].coef_)
    
#     return {
#         'timepoint': t_idx,
#         'channel': ch_idx,
#         'score': np.mean(fold_scores),
#         'metric': np.mean(fold_corr)
#     }

# def process_timepoint_ridge(t_idx, X_data, y, cv, pipeline):
#     """Traite un point temporel spécifique"""
#     X = X_data[..., t_idx].reshape(X_data.shape[0], -1)
#     fold_scores = []
#     fold_corr = []
#     fold_betas = []
    
#     for train_idx, test_idx in cv.split(X, y):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
        
#         pipeline.fit(X_train, y_train)
#         y_pred = pipeline.predict(X_test)
#         r, _ = pearsonr(y_test, y_pred)
#         score = r2_score(y_test, y_pred)
#         fold_scores.append(score)
#         fold_corr.append(np.arctanh(r))
#         fold_betas.append(pipeline.named_steps['ridgecv'].coef_)
    
#     return {
#         'timepoint': t_idx,
#         'score': np.mean(fold_scores),
#         'corr': np.mean(fold_corr),
#         'beta': np.mean(fold_betas, axis=0)
#     }


# # def process_timepoint_svm(t_idx, X_data, y, cv, pipeline):
# #     """Traite un point temporel spécifique"""
# #     X = X_data[..., t_idx].reshape(X_data.shape[0], -1)
# #     fold_scores = []
# #     fold_roc_auc = []
# #     fold_betas = []
    
# #     for train_idx, test_idx in cv.split(X, y):
# #         X_train, X_test = X[train_idx], X[test_idx]
# #         y_train, y_test = y[train_idx], y[test_idx]
        
# #         pipeline.fit(X_train, y_train)
# #         y_pred = pipeline.predict(X_test)
# #         y_proba = pipeline.decision_function(X_test)
# #         f1 = f1_score(y_test, y_pred, average='weighted')
# #         roc_auc = roc_auc_score(y_test, y_proba)
# #         fold_scores.append(f1)
# #         fold_roc_auc.append(roc_auc)
# #         fold_betas.append(pipeline.named_steps['linearsvc'].coef_)
    
# #     return {
# #         'timepoint': t_idx,
# #         'score': np.mean(fold_scores),
# #         'roc_auc': np.mean(fold_roc_auc),
# #         'beta': np.mean(fold_betas, axis=0)
# #     }





# alphas = np.logspace(-1, 4, 6)
# pipeline = make_pipeline(
#     StandardScaler(),
#     RidgeCV(alphas)
# )
# n_splits = 5
# cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# n_timepoints = X_power.shape[-1]
# n_channels = X_power.shape[1]

# tasks = [(t, ch) for t in range(n_timepoints) for ch in range(n_channels)]

# y = behavior_df["entropy"].values

# results = Parallel(n_jobs=-1, verbose=10)(
#     delayed(process_timepoint_ridge)(*task, X_power, y, cv, pipeline) for task in tasks
# )


#     # Réorganiser les résultats dans des matrices
# scores_chan = np.zeros((n_timepoints, n_channels))
# metrics_chan = np.zeros((n_timepoints, n_channels))

# for result in results:
#     t = result['timepoint']
#     ch = result['channel']
#     scores_chan[t, ch] = result['score']
#     metrics_chan[t, ch] = result['metric']




# # pipeline = make_pipeline(
# #     StandardScaler(),
# #     LinearSVC()
# # )
# # n_splits = 5
# # cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# # X_power = features.power_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()
# # # X_sin = features.phase_sin_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()
# # # X_cos = features.phase_cos_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()


# # # X_power = power_data[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()
# # # X_phase = phase_data[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()
# # n_timepoints = X_power.shape[-1]
# # y = behavior_df["fb"].values

# # behavior_df["fb"]

# # # Exécution parallèle sur tous les points temporels
# # results = Parallel(n_jobs=-1, verbose=10)(
# #     delayed(process_timepoint_ridge)(t_idx, X_power, y, cv, pipeline) 
# #     for t_idx in range(n_timepoints)
# # )

# # results = Parallel(n_jobs=-1, verbose=10)(
# #     delayed(process_timepoint_svm)(t_idx, X_power, y, cv, pipeline) 
# #     for t_idx in range(n_timepoints)
# # )


# # # Récupération des résultats dans le bon ordre
# # scores = np.zeros(n_timepoints)
# # corr = np.zeros(n_timepoints)
# # betas = np.zeros((n_timepoints, X_power.shape[1], X_power.shape[2]))

# # for result in results:
# #     i = result['timepoint']
# #     scores[i] = result['score']
# #     corr[i] = result['corr']
# #     betas[i] = result['beta'].reshape(X_power.shape[1], X_power.shape[2])


# # X_power.shape
# # scores = np.zeros(n_timepoints)
# # roc = np.zeros(n_timepoints)
# # betas = np.zeros((n_timepoints, X_power.shape[1], X_power.shape[2]))

# # for result in results:
# #     i = result['timepoint']
# #     scores[i] = result['score']
# #     roc[i] = result['roc_auc']
# #     betas[i] = result['beta'].reshape(X_power.shape[1], X_power.shape[2])




# # plt.figure(figsize=(10, 6))
# # plt.plot(roc, label='ROC AUC')
# # plt.plot(scores, label='F1 Score')
# # plt.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
# # plt.axvline(x=onset, color='g', linestyle='--', label='Onset')
# # plt.xticks(ticks=arranged_timearray, labels=timearray)
# # plt.legend()
# # plt.savefig(f"accuracy_tc_svm_{subject}.png")


# # band_names = list(FREQUENCY_BANDS.keys())

# # ch_names = loader.metadata["ch_names"]
# # regions = anatomy_df['region'].values
# # ordered_idx = np.argsort(regions)

# # max_corr_idx = np.argmax(roc)
# # max_corr = roc[max_corr_idx]
# # orederd_betas = np.abs(betas[max_corr_idx, ordered_idx])
# # plt.figure(figsize=(20, 12))
# # plt.imshow(orederd_betas, aspect='auto', cmap='jet', interpolation='nearest')
# # plt.colorbar()
# # plt.xticks(ticks=np.arange(len(band_names)), labels=band_names, rotation=45)
# # plt.yticks(ticks=np.arange(len(regions)), labels=np.array(regions)[ordered_idx])
# # plt.title(f"time : {max_corr_idx} | corr : {max_corr:.2f}")
# # plt.savefig(f"beta_coefficients_svm_{subject}.png")


# # best_corr_idx = np.argsort(roc)[::-1][:25]
# # plt.figure(figsize=(40, 30))
# # for i, idx in enumerate(best_corr_idx):
# #     plt.subplot(5, 5, i+1)
# #     betas_prepared = np.abs(betas[idx])
# #     group_betas = np.zeros((len(np.unique(regions)), len(band_names)))
# #     for i, region in enumerate(np.unique(regions)):
# #         idx_r = np.where(regions == region)[0]
# #         group_betas[i] = np.mean(betas_prepared[idx_r], axis=0)
# #     plt.imshow(group_betas, aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=0.3)
# #     plt.colorbar()
# #     plt.xticks(ticks=np.arange(len(band_names)), labels=band_names, rotation=45)
# #     plt.yticks(ticks=np.arange((len(np.unique(regions)))), labels=np.unique(regions))
# #     plt.title(f"Time: {idx} | Corr: {roc[idx]:.2f}")
# # plt.tight_layout()
# # plt.savefig(f"beta_coefficients_best_corr_svm_{subject}.png")



