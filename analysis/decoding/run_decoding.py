# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""
Module pour charger les données iEEG pour l'analyse
"""

# import matplotlib
# matplotlib.use('Qt5Agg')

# from decoding.loader import iEEGDataLoader
# from decoding.process_features import Features
# from decoding.decoding import Decoding
# from decoding.config import *

import sys, os
PROJECT_ROOT = os.path.abspath("/Users/charles/Documents/PhD/Analysis/ieeg-pipeline")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.decoding.core_classes.loader import iEEGDataLoader
from analysis.decoding.core_classes.process_features import Features
from analysis.decoding.core_classes.decoding import Decoding
from analysis.decoding.config import *

from scipy.ndimage import gaussian_filter

from joblib import Parallel, delayed

plt.style.use('seaborn-v0_8-poster') 

# full_df_selection = pd.DataFrame()
# full_df_transtion = pd.DataFrame()
# for subject in [2, 3, 4, 5, 8, 9, 12, 14, 16, 19, 20, 23, 25, 28]:
    # loader = iEEGDataLoader(subject)
    # df, _ = loader.load_behavior("transition")
    # # df = add_before_pres(df)
    # # df = add_before_trial(df)
    # full_df_transtion = pd.concat([full_df_transtion, df], ignore_index=True)
    # df, _ = loader.load_behavior("selection")
    # full_df_selection = pd.concat([full_df_selection, df], ignore_index=True)


event_name = "fb"
tmin = -1.5
tmax = 1.5

# subjects = [3, 4, 5, 8, 9, 12, 14, 16, 19, 20, 23, 25, 28]
subjects = [14, 16]
for subject in subjects:
    print(f"Processing subject {subject}")
    power_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_tfr-realign-{event_name}_{tmin}-{tmax}_power.npy")
    metadata_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_tfr-realign-{event_name}-{tmin}-{tmax}_metadata.json"))
    with open(metadata_path, 'r', encoding='utf-8') as f: 
        metadata = json.load(f)
    n_epochs = metadata["n_epochs"]
    n_channels = metadata["n_channels"]
    event_to_keep = metadata["keep_events"]
    power_data = np.load(power_path, mmap_mode='r')
    features = Features(subject)

    power_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_tfr-realign-{event_name}_{tmin}-{tmax}_power.npy")
    features.load_data(phase=False, path=power_path)

    # features.baseline_signal()
    features.extract_power_bands(baselined=False)

    anat_df = features.loader.load_anatomy()
    test_decoding = Decoding(features.power_bands, features.beh, anat_df, tmin=tmin, tmax=tmax)

    test_decoding.create_pipeline(n_folds=5, classification=True)

    for var in ["choice", "fb"] :
        #"firstswitch", "goodswitch", "good_strat", "is_random", "is_stimstable"] :
        print(f"Decoding variable: {var}")
        test_decoding.run_decoding(var, n_jobs = -1, model_type = "hmm")
        # test_decoding.run_decoding(var, mode = "channel", n_jobs = -1, model_type = "hmm")
    fig, _ = test_decoding.plot_multi_tc("", y = "score", ylim=(0.4, 0.7))
    # fig.close()
    plt.close(fig)
    # fig, _ = test_decoding.plot_multi_heatmap("", y = "score", save=False)
    # plt.close(fig)


# event_name = "fb"
# tmin = -1.5 
# tmax = 1.5
# for subject in [2, 3, 4, 5, 8, 9, 12, 14, 16, 19, 20, 23, 25, 28]:
#     metadata_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_tfr-realign-{event_name}-{tmin}-{tmax}_metadata.json"))
#     with open(metadata_path, 'r', encoding='utf-8') as f: 
#         metadata = json.load(f)
#     beh_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_task-stratinf_beh-aligned.csv"))
#     event_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_events-aligned.csv"))
#     beh = pd.read_csv(beh_path)
#     events = pd.read_csv(event_path)
#     trials = np.array(beh["trial_count"])
#     pad = np.array(metadata["padded_event"])
#     idx_padded = np.where(pad != 0)[0]
#     n_idx_padded = len(idx_padded)
#     print("#"*50)
#     print(f"Subject {subject} has {n_idx_padded} epochs with padding correspond to {n_idx_padded/len(pad)*100:.2f}% of the data, mean padding {np.mean(pad[idx_padded]):.2f} samples and max padding {np.max(pad[idx_padded])} samples")
#     # if n_idx_padded > 0:
#     #     for trial in trials[idx_padded]:
#     #         print(f" - Trial {trial} with padding {pad[trials==trial][0]} samples")
#     #         print(events[events["trial_count"]==trial])
#     #     print(" "*50)
#     #     print(" "*50)


#     # print(f"Subject {subject} has {n_idx_padded} epochs with padding correspond to {n_idx_padded/len(pad)*100:.2f}% of the data")


# def process_subject(subject):
#     features = Features(subject)
#     features.load_data(phase=False)
#     features.prepare_data()
#     features.export_realign_data("fb", tmin=-1.5, tmax=1.5)
#     features.export_realign_data("action", tmin=-2, tmax=1.5)
#     beh_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_task-stratinf_beh-aligned.csv"))
#     features.beh.to_csv(beh_path, index=False)
#     events_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_events-aligned.csv"))
#     features.events_df.to_csv(events_path, index=False)
#     del features



# subjects = [2, 3, 4, 5, 8, 9, 12, 14, 16, 19, 20, 23, 25, 28]
# Parallel(n_jobs=-1, prefer="threads")(delayed(process_subject)(subj) for subj in subjects)

# # for subject in [2, 3, 4, 5, 8, 9, 12, 14, 16, 19, 20, 23, 25, 28]:
# subject = 2
# features = Features(subject)
# features.load_data(phase=False)
# features.prepare_data()
# realign_power_data, realign_phase_data = features._realign_data("fb", -1.5, 1.5, False)
# power_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_tfr-realign-{event_name}_{tmin}-{tmax}_power.npy")
# test_w = np.memmap(power_path, mode='w+', shape=realign_power_data.shape, dtype=np.float16)
# test_w[:] = realign_power_data[:]
# test_w.flush()
# np.save(power_path, realign_power_data, allow_pickle=False)
# test_load = np.load(power_path, mmap_mode='r')

# n_epochs = features.n_epochs
# n_channels = features.n_channels
# test = np.memmap(power_path, mode='r', shape=(n_epochs, n_channels, n_freqs, int((tmax - tmin)*sr_decimated)), dtype=np.float16)

# features.export_realign_data("fb", tmin=-1.5, tmax=1.5)
# features.export_realign_data("action", tmin=-2, tmax=1.5)
# beh_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_task-stratinf_beh-aligned.csv"))
# features.beh.to_csv(beh_path, index=False)
# events_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "aligned", f"sub-{int(subject):03}_events-aligned.csv"))
# features.events_df.to_csv(events_path, index=False)
# del features


# # for subject in [2, 3, 4, 5, 8, 9, 12, 14, 16, 19, 20, 23, 25, 28]:
# subject = 2

# event_name = "fb"
# tmin = -1.5
# tmax = 1.5

# event_samples = features.events_dict[event_name]


# start_offset = int(tmin * sr_decimated)
# end_offset = int(tmax * sr_decimated)
# win_len = end_offset - start_offset
# realign_power_data = np.zeros((features.n_epochs, features.n_channels, n_freqs, win_len), dtype=np.float16)
# padded_list = []
# # for i in range(features.n_epochs):
# i = 1020
# start_idx = event_samples[i] + start_offset
# end_idx = event_samples[i] + end_offset
# if end_idx > n_times_decimed:
#     syntetic_padding = end_idx - n_times_decimed
#     mu = features.baseline[i].astype(np.float32)
#     sigma = np.std(features.power_data[i].astype(np.float32), axis=-1, keepdims=True)
#     syntetic_data = np.random.normal(mu, sigma, (features.n_channels, n_freqs, syntetic_padding))
#     realign_power_data[i, :, :, :win_len-syntetic_padding] = features.power_data[i, :, :, start_idx:end_idx]
#     realign_power_data[i, :, :, win_len-syntetic_padding:] = syntetic_data
#     padded_list.append(syntetic_padding)
# else :
#     realign_power_data[i, :, :, :] = features.power_data[i, :, :, start_idx:end_idx]
#     padded_list.append(0)


# fig, axs = plt.subplots(7, 7, figsize=(30, 21), sharex=True, sharey=True)
# for ax, ch_idx in zip(axs.flatten(), range(48)):
#     mat = test_load[1020, ch_idx]-features.baseline[1020, ch_idx]
#     lim = np.max(np.abs(mat))
#     ax.imshow(mat, aspect='auto', origin='lower', cmap='jet', vmin=-lim, vmax=lim)
# fig.tight_layout()
# fig.show()





# features.events_dict["is_missing_fb"].sum()

# features.get_event_dict()

# np.where(features.events_dict[event_name]>n_times_decimed)
# features.events_dict
# event_name = "fb"
# tmin = -1.5
# tmax = 1.5

# sns.histplot(features.events_dict["fb"]-features.events_dict["action"], kde=True, bins=100)
# plt.show()
# np.where((features.events_dict["fb"]-features.events_dict["action"]) > 2*sr_decimated)
# event_samples = z
# features.events_dict["trial"][features.events_dict["fb"] > n_times_decimed]


# start_offset = int(tmin * sr_decimated)
# end_offset = int(tmax * sr_decimated)
# win_len = end_offset - start_offset
# realign_power_data = np.zeros((features.n_epochs, features.n_channels, n_freqs, win_len), dtype=np.float16)

# for i in range(features.n_epochs):
#     print(f"Processing epoch {i+1}")
#     start_idx = event_samples[i] + start_offset
#     end_idx = event_samples[i] + end_offset
#     if end_idx > n_times_decimed:
#         syntetic_padding = end_idx - n_times_decimed
#         syntetic_data = np.random.normal(mu[..., None], sigma[..., None], (features.n_channels, n_freqs, syntetic_padding))
#         realign_power_data[i, :, :, :win_len-syntetic_padding] = features.power_data[i, :, :, start_idx:end_idx]
#         realign_power_data[i, :, :, :syntetic_padding] = syntetic_data
#     else :
#         realign_power_data[i, :, :, :end_idx] = features.power_data[i, :, :, start_idx:end_idx]

# win_len
# syntetic_padding = end_idx - n_times_decimed

# baseline_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "timefreq", f"sub-{int(subject):03}_tfr-baseline.npy")
# baseline = np.load(baseline_path)
# baseline = baseline[features.event_to_keep]
# realign_power_data = realign_power_data - baseline
# features.events_dict["beh_rt"][261]

# event_samples[i] + start_offset
# event_samples[i] + end_offset
# event_samples[i]

# features.event_to_keep
# syntetic_padding
# end_idx-n_times_decimed
# win_len
# test_trial = features.events_df.loc[features.events_df["trial_count"]==781, :]
# test_trial
# test_trial["decimated_sample"].diff()/sr_decimated
# test_trial
# list_number = np.arange(0, 10)

# list_number[:]
# realign_power_data[i, :, :, :win_len-syntetic_padding].shape
# features.power_data[i, :, :, start_idx:end_idx].shape
# anatomy_df = features.loader.load_anatomy()
# anatomy_df
# mu = np.mean(features.power_data.astype(np.float32), axis=(0, -1))
# test = np.random.normal(mu[..., None], sigma[..., None], (features.n_channels, n_freqs, 500))
# test = realign_power_data[17, 10, :, :]
# lim = np.max(np.abs(test))
# plt.imshow(test, aspect='auto', origin='lower', cmap='jet', vmin=-lim, vmax=lim)
# plt.colorbar()
# # plt.yticks(ticks=np.arange(len(anatomy_df)), labels=anatomy_df["name"].values)
# plt.show()
# sigma = np.std(features.power_data.astype(np.float32), axis=(0, -1))

# np.random.normal(mu, sigma, (features.n_channels, n_freqs))

# features.power_data[0, 10, :, :]

# sigma

# realign_power_data[:148].shape
# features.power_data[i, :, :, start_idx:end_idx].shape
# realign_power_data[i, :, :, syntetic_padding:].shape


# # syntetic_data = 
# np.random.normal(mu, sigma, size=(features.n_channels, n_freqs, syntetic_padding)).astype(np.float16)




# win_len-syntetic_padding
















# loader = iEEGDataLoader(subject)
# power_data = loader.load_power()
# # phase_data = loader.load_phase()
# anatomy_df = loader.load_anatomy()

# loader.ch_names
# behavior_df, event_to_keep, events_df = loader.load_behavior("")
# power_data = power_data[event_to_keep, :, :, :]

# baseline_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "timefreq", f"sub-{int(subject):03}_tfr-baseline.npy")
# baseline = np.load(baseline_path)
# baseline = baseline[event_to_keep]
# power_data_baselined = power_data - baseline
# del baseline, power_data
# gc.collect()



# trial_beh = behavior_df["trial_count"].values

# test_events = events[events["trial_count"].isin(trial_beh)].reset_index(drop = True)
# test_events["decimated_sample"] = (test_events["sample_update"] / decimation).astype(int)

# test = loader.get_events_dict(behavior_df, event_to_keep, test_events)



# test_events
# events_dict = {
#     "index_trial": [],
#     "onset" : [],
#     "action" : [],
#     "fb_current" : [],
# }
# zero_epoch = int(epoch_padding * sr_decimated)

# for trial in behavior_df["trial_count"].values:
#     event_trial = test_events[test_events["trial_count"] == trial]
#     events_dict["index_trial"].append(trial)
#     event_seq = event_trial["decimated_sample"].diff().values
#     event_seq[np.isnan(event_seq)] = 0
#     event_seq = np.cumsum(event_seq)
#     event_seq = event_seq.astype(int)
#     event_seq += zero_epoch
#     events_dict["onset"].append(zero_epoch)
#     events_dict["action"].append(event_seq[1])
#     events_dict["fb_current"].append(event_seq[2])



# behavior_df
# events_dict["onset"] = np.array(events_dict["onset"])
# events_dict["action"] = np.array(events_dict["action"])
# events_dict["fb_current"] = np.array(events_dict["fb_current"])
# events_dict["index_trial"] = np.array(events_dict["index_trial"])

# events_dict



# zero_epoch = int(epoch_padding * sr_decimated)

# zero_epoch
# times_decimed[zero_epoch]
# fb_sample = []
# action_sample = []
# for idx, trial_df in test_events.groupby("trial_count"):
#     event_seq = trial_df["decimated_sample"].diff().values
#     event_seq[np.isnan(event_seq)] = 0
#     event_seq = np.cumsum(event_seq)
#     event_seq = event_seq.astype(int)
#     event_seq += zero_epoch
#     fb_sample.append(event_seq[2])
#     action_sample.append(event_seq[1])

# fb_sample = np.array(fb_sample)
# action_sample = np.array(action_sample)

# fb_sample


# WOI_start_time = 1.5
# WOI_end_time = 3.5
# WOI_start_idx= int((epoch_padding - WOI_start_time) * sr_decimated)
# WOI_end_idx = int((WOI_end_time + epoch_padding) * sr_decimated)

# n_timepoints = int((WOI_start_time + WOI_end_time) * sr_decimated)
# timearray = np.linspace(-WOI_start_time, WOI_end_time, int((WOI_start_time + WOI_end_time)*2+1))
# arranged_timearray = np.arange(0, n_timepoints+1, n_timepoints/(len(timearray)-1))

# onset = int(WOI_start_time * sr_decimated)




# zero_epoch

# fb_sample = features.events_dict["fb"]
# np.where(features.events_dict["events_rt"]<0)

# behavior_df = features.beh
# features.events_dict["trial"][[381]]
# features.events_df[features.events_df["trial_count"].isin([406.])]

# sns.histplot((features.events_dict["beh_rt"][(features.events_dict["beh_rt"]>0) & (features.events_dict["events_rt"]>0)]-features.events_dict["events_rt"][(features.events_dict["beh_rt"]>0) & (features.events_dict["events_rt"]>0)])*1000)
# plt.show()

# good_switch = behavior_df[(behavior_df["firstswitch"] == 1) & (behavior_df["goodswitch"] == 1)].index
# good_switch
# b4_good = behavior_df[(behavior_df["firstswitch"] == 0) & (behavior_df["goodswitch"] == 1)].index
# # good_switch = behavior_df[(behavior_df["firstswitch"] == 1) & (behavior_df["goodswitch"] == 1)].index
# good_switch = behavior_df[(behavior_df["firstswitch"] == 1)].index
# b4_good = behavior_df[(behavior_df["firstswitch"].shift(-1) == 1)].index
# # b4_good = behavior_df[behavior_df["stim"]== 3].index
# anatomy_df = features.loader.load_anatomy()
# fb_aligned = [features.power_data_baselined[i, :, :, fb_sample[i]-300:fb_sample[i]+300] for i in good_switch if fb_sample[i]-300 >=0 and fb_sample[i]+300 <= n_times_decimed]
# mat = np.mean(realign_power_data[b4_good[:5]], axis=0)#np.mean(np.asarray(fb_aligned), axis=0)
# fb_aligned = [features.power_data_baselined[i, :, :, fb_sample[i]-300:fb_sample[i]+300] for i in b4_good if fb_sample[i]-300 >=0 and fb_sample[i]+300 <= n_times_decimed]
# mat_b4 = np.mean(np.asarray(fb_aligned), axis=0)
# diff = mat - mat_b4
# fig, axs = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
# axs = axs.flatten()


# # anatomy_df[anatomy_df["region"] == "VMPFC"].index
# # lim = np.max(np.abs(mat[anatomy_df[anatomy_df["region"] == "VMPFC"].index]))
# for i, (region, index) in enumerate(anatomy_df.groupby("region").groups.items()):
# # for i, ch_idx in enumerate(anatomy_df[anatomy_df["region"] == "VMPFC"].index) : 
#     channel_mat = gaussian_filter(np.mean(mat[index], axis=0).astype(np.float32), sigma=3)
#     # channel_mat = gaussian_filter(mat[index].astype(np.float32), sigma=3)
#     lim = np.max(np.abs(channel_mat))
#     im = axs[i].imshow(channel_mat, aspect='auto', origin='lower', cmap='jet', vmin=-lim, vmax=lim, interpolation='nearest')
#     plt.colorbar(im, ax=axs[i])
#     # axs[i].axvline(x=300, color='k')
#     # axs[i].axvline(x=44, color='dimgrey', linestyle='--', lw=0.5)
#     # axs[i].axvspan(xmin=0, xmax=100, color='lightgrey', alpha=0.5)
#     axs[i].hlines(y=band_indices[1:], xmin=0, xmax=600, colors='dimgrey', linestyles='--', linewidth=0.5)
#     axs[i].set_xticks(ticks=ticks, labels=label)
#     axs[i].set_yticks(ticks=band_indices[1:], labels=list(FREQUENCY_BANDS.items()))
#     axs[i].set_title(region)
#     # axs[i].set_title(f"{anatomy_df.loc[ch_idx, "region"]} - {anatomy_df.loc[ch_idx, "name"]}")
# plt.tight_layout()
# plt.show()

# np.arange(0, 601, 0.5*sr_decimated)

# 0.2 * sr_decimated
# # ...existing code...
# width = channel_mat.shape[1]                    # ex: 600
# step = int(0.5 * sr_decimated)                  # pas en échantillons pour 0.5s
# # base ticks every 0.5s, ensure sample 1, feedback(300) and last sample are present

# ticks = [300-2*step, 300 - step, 300, 300 + step, 300+2*step]
# # labels in seconds with 0 at sample 300
# label = [-(300 - t) / sr_decimated for t in ticks]


# len(anatomy_df["region"].unique())

# test_signal_rt = test_action["time_signal"].values - test_onset["time_signal"].values

# delta_rt = test_signal_rt - behavior_df["rt"].values
# delta_rt[delta_rt>1] = np.nan
# plt.hist(delta_rt, bins=300)
# sns.kdeplot(delta_rt, bw=0.5)
# plt.show()




# behavior_df

# behavior_df[~behavior_df["trial_count"].isin(common_trials)].reset_index(drop = True)
# onset_events = onset_events[onset_events["align_eeg"] != "-"].reset_index(drop = True)

# beh_df["trial_count"] = np.arange(1, len(beh_df) + 1)
# # beh_df = beh_df[beh_df["training"] == 0]
# beh_orig_eegmap = beh_df[beh_df["trial_count"].isin(onset_events["trial_count"])].reset_index(drop = True)
# common_trials = np.intersect1d(beh_orig_eegmap["trial_count"], simu_df["trial_count"])
# event_tokeep = beh_orig_eegmap[beh_orig_eegmap["trial_count"].isin(common_trials)].index
# simu_behav_clean = simu_df[simu_df["trial_count"].isin(common_trials)].reset_index(drop = True)









# # behavior_df_transition, _ = load
# # er.load_behavior("transition")

# features = Features(power_data, [], subject)
# features.baseline_signal()
# features.extract_power_bands()
# X_power = features.power_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()



# test_decoding = Decoding(X_power, behavior_df, anatomy_df)
# test_decoding.create_pipeline(n_folds=5, classification=True)


# # test_decoding = Decoding(X_power, behavior_df, anatomy_df)
# # test_decoding.create_pipeline(n_folds=5, classification=True)
# for var in ["goodswitch", "firstswitch", "good_strat", "is_random"]:
#     print(f"Decoding variable: {var}")
#     test_decoding.run_decoding(var, n_jobs = 8, model_type = "hmm")
#     test_decoding.plot_tc(var, y = "score", ylim = (0.4, 0.8))
#     test_decoding.run_decoding(var, n_jobs = 8, mode = "channel", model_type = "hmm")
# test_decoding.plot_multi_tc("", y = "score", ylim = (0.4, 0.8))
# test_decoding.plot_multi_heatmap("", y = "score")

# plt.show()
# ch = 0
# t = 100  # time point index
# y = test_decoding.beh["is_stimstable"].values
# np.where(y == -1)
# y = y[60:]

# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score, f1_score

# clf = CalibratedClassifierCV(LinearSVC(max_iter=10000), cv=3)
# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# pipeline = make_pipeline(StandardScaler(), clf)
# test_roc = []
# test_f1 = []

# for t in range(0, 1200, 2):
#     X = test_decoding.data[60:, 6:10, :, t].reshape(test_decoding.n_epochs - 60, -1)
#     tmp_roc = []
#     tmp_f1 = []
#     for train_idx, test_idx in cv.split(X, y):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
#         pipeline.fit(X_train, y_train)
#         y_proba = pipeline.predict_proba(X_test)
#         y_pred = pipeline.predict(X_test)
#         roc = roc_auc_score(y_test, y_proba, multi_class="ovr")
#         f1 = f1_score(y_test, y_pred, average="weighted")
#         tmp_roc.append(roc)
#         tmp_f1.append(f1)
#     roc = np.mean(tmp_roc)
#     f1 = np.mean(tmp_f1)
#     print(f"Time {t}: ROC AUC = {roc:.3f}, F1 Score = {f1:.3f}")
#     test_roc.append(roc)
#     test_f1.append(f1)


# plt.plot(test_roc)
# plt.plot(test_f1)
# plt.show()
# # for var in ["fb", "prev_fb", "is_partial", "goodswitch", "firstswitch", "good_strat", "is_random"]:
# #     test_decoding.load_results(var, mode = "global", model_type = "hmm")
# #     test_decoding.load_results(var, mode = "channel", model_type = "hmm")




# behavior_df[behavior_df["fb"] == 1].index
# before_sw = X_power[behavior_df[behavior_df["fb"] == 1].index-1, :, :, :]
# before_sw_mean = np.mean(before_sw, axis=0)
# good_sw = X_power[behavior_df[behavior_df["fb"] == 1].index, :, :, :]
# good_sw_mean = np.mean(good_sw, axis=0)

# fig, axs = plt.subplots(3, 2, figsize=(18, 10))
# axs = axs.flatten()
# for i, band in enumerate(FREQUENCY_BANDS.keys()):
#     sig = good_sw_mean[test_decoding.ordered_regions_idx, i, :] - before_sw_mean[test_decoding.ordered_regions_idx, i, :]
#     lim = np.max(np.abs(sig))
#     im = axs[i].imshow(sig, aspect='auto', cmap='jet', vmin=-lim, vmax=lim, interpolation='nearest')
#     axs[i].axvline(onset, color='black', linestyle='--')
#     plt.colorbar(im, ax=axs[i])
#     axs[i].set_title(f"{band} band")
#     axs[i].set_yticks(np.arange(len(test_decoding.regions)), labels=test_decoding.ordered_regions)
# plt.show()

# # behavior_df.loc[(behavior_df["is_partial"]==-1), "is_partial"] = 0
# # behavior_df.columns
# # test_decoding = Decoding(X_power, behavior_df, anatomy_df)
# # test_decoding.create_pipeline(n_folds=5, classification=True)
# var = "goodswitch"
# test_decoding.run_decoding(var, n_jobs = -1, model_type = "hmm")

# plt.plot(test_decoding.results["score"])
# plt.show()

# test_decoding.run_decoding(var, n_jobs = -1, mode = "channel", model_type = "hmm")
# mat = test_decoding.results["score"][:, test_decoding.ordered_regions_idx]

# plt.imshow(mat.T, aspect='auto', cmap='jet', interpolation='nearest')
# plt.yticks(np.arange(len(test_decoding.regions)), labels=test_decoding.ordered_regions)
# plt.colorbar()
# plt.show()
# test_decoding.results["metric"]


# betas = test_decoding.betas_channel_dict["goodswitch"]  # shape (n_timepoints, n_channels, n_freqs)
# fig, axs = plt.subplots(4, 2, figsize=(18, 10))
# axs = axs.flatten()
# for i, band in enumerate(FREQUENCY_BANDS.keys()):
#     lim = np.max(np.abs(betas[:, :, i]))
#     val = betas[:, test_decoding.ordered_regions_idx, i].T
#     im = axs[i].imshow(val, cmap='jet', aspect='auto', vmin=-lim, vmax=lim, interpolation='nearest')
#     axs[i].axvline(onset, color='black', linestyle='--')
#     axs[i].set_yticks(np.arange(len(test_decoding.regions)), labels=test_decoding.ordered_regions)
#     plt.colorbar(im, ax = axs[i])
#     axs[i].set_title(f"{band} band")
# plt.tight_layout()
# plt.show()



# # test_decoding.plot_multi_tc("", y = "score", ylim = (0.4, 0.6), save = False)


# var_list = list(test_decoding.score_global_dict.keys())


# nvar = len(var_list)
# nrows = int(np.sqrt(nvar))
# ncols = int(np.ceil(nvar / nrows))

# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 19))
# axs = axs.flatten()
# for i, var in enumerate(var_list):
#     # val = gaussian_filter1d(test_decoding.score_global_dict[var], sigma=5)
#     val = test_decoding.score_global_dict[var]
#     axs[i].plot(test_decoding.results['timepoint'], val, lw = 0.5)
#     axs[i].axhline(0.5, color='black', linestyle='--')
#     axs[i].axvline(onset, color='red', linestyle='--')
#     axs[i].set_xlabel("Time (ms)")
#     axs[i].set_ylabel("Accuracy")
#     axs[i].set_title(f"{var}", fontsize=16, fontweight='bold')
#     axs[i].set_ylim(0.45, 0.65)
#     axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)
# # plt.tight_layout()
# plt.subplots_adjust(hspace=0.5)
# plt.show()




# var_list = list(test_decoding.score_channel_dict.keys())

# nvar = len(var_list)
# nrows = int(np.sqrt(nvar))
# ncols = int(np.ceil(nvar / nrows))

# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 19), sharex=True, sharey=True)
# axs = axs.flatten()
# for i, var in enumerate(var_list):
#     to_plot = test_decoding.score_channel_dict[var]
#     # lim = np.max(np.abs(to_plot))
#     im = axs[i].imshow(to_plot.T, aspect='auto', cmap='jet', interpolation='nearest')
#     plt.colorbar(im, ax=axs[i])
#     axs[i].axvline(onset, color='black', linestyle='--')
#     axs[i].set_title(f"{var} channel accuracy")
#     axs[i].set_xlabel("Time (ms)")
#     axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)
#     axs[i].set_yticks(np.arange(len(test_decoding.regions)), labels=test_decoding.regions)
# plt.subplots_adjust(hspace=0.5)
# plt.show()

# # decode_selection = Decoding(X_power, behavior_df_selection, anatomy_df)
# # decode_selection.create_pipeline(n_folds=5, alphas=np.logspace(-1, 4, 6), classification=False)

# # decode_transition = Decoding(X_power, behavior_df_transition, anatomy_df)
# # decode_transition.create_pipeline(n_folds=5, alphas=np.logspace(-1, 4, 6), classification=False)


# # for i, var in enumerate(VAR_LIST) : 
# #     decode_transition.run_decoding(var, n_jobs = -1, model_type = "transition")
# #     decode_transition.run_decoding(var, n_jobs = -1, mode = "channel", model_type = "transition")
# #     decode_selection.run_decoding(var, n_jobs = -1, model_type = "selection")
# #     decode_selection.run_decoding(var, n_jobs = -1, mode = "channel", model_type = "selection")
# # decode_selection.plot_multi_tc("selection")
# # decode_selection.plot_multi_heatmap("selection")
# # decode_transition.plot_multi_tc("transition")
# # decode_transition.plot_multi_heatmap("transition")


# # var = "rpe"

# # decode_selection.run_decoding(var, n_jobs = -1, save_results = False)
# # decode_transition.run_decoding(var, n_jobs = -1, save_results = False)



# # plt.plot(decode_transition.results['timepoint'], decode_transition.metric_global_dict[var], lw = 2, color = "forestgreen", label = "transition")
# # plt.plot(decode_selection.results['timepoint'], decode_selection.metric_global_dict[var], lw = 2, color = "royalblue", label = "selection")
# # plt.axhline(0.0, color='black', linestyle='--')
# # plt.axvline(onset, color='red', linestyle='--')
# # plt.ylim(-0.2, 1)
# # plt.xticks(ticks=arranged_timearray, labels=timearray)
# # plt.tight_layout()
# # plt.legend()
# # plt.show()

# # decode_selection.plot_multi_tc(figsize=(18, 10), save = False)


# # var = "update_counterfactual"

# # decode_selection.run_decoding(var, n_jobs = -1, mode = "channel",  save_results = False)
# # decode_transition.run_decoding(var, n_jobs = -1, mode = "channel", save_results = False)



# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), sharex=True, sharey=True)

# # to_plot = decode_transition.metric_channel_dict[var][:, decode_transition.ordered_regions_idx]
# # lim = np.max(np.abs(to_plot))
# # im = ax1.imshow(to_plot.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# # ax1.set_yticks(ticks=np.arange(decode_transition.n_channels), labels=decode_transition.ordered_regions)
# # plt.colorbar(im, ax = ax1)
# # ax1.set_title("Transition")

# # to_plot = decode_selection.metric_channel_dict[var][:, decode_selection.ordered_regions_idx]
# # lim = np.max(np.abs(to_plot))
# # im = ax2.imshow(to_plot.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# # plt.colorbar(im, ax = ax2)
# # ax2.set_title("Selection")

# # plt.tight_layout()
# # plt.legend()
# # plt.show()





# # plt.scatter(behavior_df_selection["action_value"], behavior_df_transition["action_value"], s=5, alpha=0.4)
# # plt.show()






# # behavior_df_transition.columns

# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), sharex=True, sharey=True)

# # ax1.scatter(behavior_df_selection["rpe"], behavior_df_selection["update_counterfactual"], s=5, alpha=0.4)
# # ax1.set_title("Selection")
# # ax2.scatter(behavior_df_transition["rpe"], behavior_df_transition["update_counterfactual"], s=5, alpha=0.4)
# # ax2.set_title("Transition")
# # plt.tight_layout()
# # plt.show()






# # full_df_selection




# # decode_selection.plot_multi_tc(save = False, figsize = (20, 12))







# # decode.run_decoding(var, n_jobs = -1, mode = "channel",  save_results = False)

# # to_plot = decode.metric_channel_dict[var][:, decode.ordered_regions_idx]
# # lim = np.max(np.abs(to_plot))
# # plt.imshow(to_plot.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# # plt.colorbar()
# # plt.show()





# # simu_path = os.path.join(CLUSTER_DIR, 'simu', 'selection',f"sub-{subject:03}_task-stratinf_sim-forced.csv")
# # df = pd.read_csv(simu_path)
# # df.columns

# # beh_path = os.path.join(CLUSTER_DIR, 'beh', f"sub-{subject:03}_task-stratinf_beh.tsv")
# # df = pd.read_csv(beh_path)



# # decode.results

# # fig, axs = plt.subplots(nrows=2, ncols=2)
# # axs = axs.flatten()


# # fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
# # axs = axs.flatten()



# # from sklearn.model_selection import KFold, StratifiedKFold
# # from sklearn.linear_model import RidgeCV
# # from sklearn.svm import LinearSVC
# # from sklearn.pipeline import make_pipeline
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
# # from scipy.stats import pearsonr
# # from joblib import Parallel, delayed



# # def compute_score_power_chann(X_band, t, y, pipeline, cv):
# #     """Compute the score for a specific time point"""
# #     tmp_results = {}
# #     for ch in range(X_band.shape[1]):
# #         X = X_band[:, ch, :, t].reshape(X_band.shape[0], -1)
# #         fold_metrics = []
# #         fold_betas = []
# #         fold_score = []
# #         for train_idx, test_idx in cv.split(X, y):
# #             X_train, X_test = X[train_idx], X[test_idx]
# #             y_train, y_test = y[train_idx], y[test_idx]
# #             pipeline.fit(X_train, y_train)
# #             y_pred = pipeline.predict(X_test)
# #             r, _ = pearsonr(y_test, y_pred)
# #             fold_score.append(mean_squared_error(y_test, y_pred))
# #             fold_metrics.append(np.arctanh(r))
# #             fold_betas.append(pipeline.named_steps['ridgecv'].coef_)
# #         tmp_results[ch] = {
# #             'score': np.mean(fold_score),
# #             'metric': np.mean(fold_metrics),
# #             'betas': np.mean(fold_betas, axis=0)
# #         }
# #     return {
# #         't': t,
# #         'test': tmp_results
# #     }





# # alphas = np.logspace(-1, 4, 6)
# # pipeline = make_pipeline(
# #     StandardScaler(),
# #     RidgeCV(alphas)
# # )
# # n_splits = 5
# # cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# # n_timepoints = X_power.shape[-1]
# # n_channels = X_power.shape[1]

# # tasks = [(t, ch) for t in range(n_timepoints) for ch in range(n_channels)]

# # y = behavior_df["reliability_max"].values

# # results = Parallel(n_jobs=-1, verbose=10)(
# #     delayed(compute_score_power_chann)(X_power, t , y, pipeline, cv) for t in range(n_timepoints)
# # )


# # # Réorganiser les résultats dans des matrices
# # scores_chan = np.zeros((n_timepoints, n_channels))
# # metrics_chan = np.zeros((n_timepoints, n_channels))

# # for result in results:
# #     t = result['t']
# #     tmp_results = result['test']
# #     for ch, tmp_result in tmp_results.items():
# #         scores_chan[t, ch] = tmp_result['score']
# #         metrics_chan[t, ch] = tmp_result['metric']


# # lim = np.max(np.abs(metrics_chan))
# # plt.imshow(metrics_chan.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# # plt.colorbar()
# # scores_chan

# # regions = anatomy_df['region'].values

# # # Version améliorée - Grouper par région avec des séparations visuelles
# # unique_regions = np.unique(regions)
# # region_boundaries = []
# # current_pos = 0

# # # Créer une figure plus grande
# # plt.figure(figsize=(30, 22))

# # # Réorganiser les canaux par région
# # ordered_idx = []
# # ordered_labels = []

# # for region in sorted(unique_regions):
# #     # Trouver les indices des canaux pour cette région
# #     region_indices = np.where(regions == region)[0]
    
# #     # Ajouter tous les canaux de cette région à notre liste d'ordre
# #     ordered_idx.extend(region_indices)
    
# #     # Marquer la limite de la région
# #     current_pos += len(region_indices)
# #     region_boundaries.append(current_pos - 0.5)
    
# #     # Ajouter une étiquette pour chaque canal
# #     for i in region_indices:
# #         ordered_labels.append(f"{region}")

# # # Réorganiser les données
# # ordered_chanmap = metrics_chan[:, ordered_idx]

# # plt.figure(figsize=(20, 12))

# # # Créer la heatmap
# # ax = sns.heatmap(ordered_chanmap.T, cmap='jet', center=0, 
# #                 vmin=-lim, vmax=lim, cbar=True)

# # # Ajouter des lignes horizontales pour séparer les régions
# # for boundary in region_boundaries[:-1]:  # Éviter d'ajouter une ligne à la fin
# #     plt.axhline(y=boundary, color='black', linestyle='-', linewidth=4)

# # # Étiqueter les axes
# # plt.title("Canal-wise Decoding Performance by Region", fontsize=18)
# # plt.xlabel("Time (ms)", fontsize=16)
# # plt.ylabel("Brain Regions", fontsize=16)

# # # Configurer les ticks des axes
# # plt.xticks(ticks=arranged_timearray, labels=timearray)
# # plt.axvline(x=onset, color='black', linestyle='--', linewidth=2)
# # plt.yticks(ticks=np.arange(len(regions)), labels=np.array(regions)[ordered_idx], rotation=0)


# # plt.tight_layout()















# # # behavior_df["subject"]

# # # fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 12))
# # # axs = axs.flatten()
# # # for i, var in enumerate(['reliability_max', 'reliability_choosen', 'update_reliability_max', 'entropy','q_choosen', "rpe"]) :
# # #     file_path = os.path.join(OUTPUT_DIR, "ridge", f"sub-{int(subject):03}_{var}_ridge_fold-5_metric-all.npy")
# # #     data = np.load(file_path)
# # #     axs[i].plot(data)
# # #     axs[i].axhline(0, color='black', linestyle='--')
# # #     axs[i].axvline(onset, color='red', linestyle='--')
# # #     axs[i].set_title(f"{var} time course")
# # #     axs[i].set_xlabel("Time (ms)")
# # #     axs[i].set_ylabel("Metric")
# # #     axs[i].set_ylim(-0.2, 1)
# # #     axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)

# # # fig.tight_layout()
# # # plt.savefig(f"accuracy_tc_ridge_{subject}.png")





# # # var_list = list(self.metric_dict.keys())
# # # nvar = len(var_list)
# # # nrows = int(np.sqrt(nvar))
# # # ncols = int(np.ceil(nvar / nrows))
# # # fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
# # # axs = axs.flatten()
# # # for i, var in enumerate(var_list):
# # #     axs[i].plot(self.results['timepoint'], self.metric_dict[var])
# # #     axs[i].axhline(0, color='black', linestyle='--')
# # #     axs[i].axvline(onset, color='red', linestyle='--')
# # #     axs[i].set_title(f"{var} {y} time course")
# # #     axs[i].set_xlabel("Time (ms)")
# # #     axs[i].set_ylabel(y)
# # #     axs[i].set_ylim(-0.1, 1)
# # #     axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)







# # # fig, ax = plt.subplots(figsize=(30, 19))
# # # plt.plot(decode.results['timepoint'], decode.results["metric"])
# # # plt.axhline(0, color='black', linestyle='--')
# # # plt.axvline(onset, color='red', linestyle='--')
# # # plt.xlabel("Time (ms)")
# # # plt.ylim(-0.1, 1)
# # # plt.xticks(ticks=arranged_timearray, labels=timearray)
# # # plt.tight_layout()
# # # plt.savefig(f"accuracy_tc_ridge_{subject}.png")


# # # betas = decode.results['betas']  # shape (n_timepoints, n_channels, n_freqs)
# # # y = behavior_df["entropy"].values

# # # # Préparer les tableaux pour les résultats
# # # n_timepoints = betas.shape[0]
# # # n_channels = betas.shape[1]
# # # channel_r2 = np.zeros((n_timepoints, n_channels))
# # # channel_corr = np.zeros((n_timepoints, n_channels))

# # # for t in range(n_timepoints):
# # #     for ch in range(n_channels):
# # #         X = X_power[:, ch, :, t].reshape(X_power.shape[0], -1)
# # #         X_scaled = StandardScaler().fit_transform(X)
# # #         beta_ch = betas[t, ch, :]
# # #         y_pred = X_scaled @ beta_ch
# # #         r, _ = pearsonr(y, y_pred)
# # #         r2 = r2_score(y, y_pred)
# # #         channel_r2[t, ch] = r2
# # #         channel_corr[t, ch] = np.arctanh(r)  # Fisher z-transformation


# # # lim = np.max(np.abs(channel_corr))
# # # plt.figure(figsize=(30, 19))
# # # plt.imshow(channel_corr.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# # # plt.colorbar()
# # # plt.savefig(f"channel_correlation_{subject}.png")


# # # lim = np.max(np.abs(metrics_chan))
# # # plt.figure(figsize=(30, 19))
# # # plt.imshow(metrics_chan.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# # # plt.colorbar()
# # # plt.savefig(f"channel_correlation_refit_{subject}.png")

# # # #compare 2 method for channel correlation
# # # lim = np.max((np.abs(channel_corr), np.abs(metrics_chan)))
# # # fig, axs = plt.subplots(1, 3, figsize=(40, 19))
# # # ax1 = axs[0].imshow(channel_corr.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# # # axs[0].set_title("Direct from betas")
# # # ax2 = axs[1].imshow(metrics_chan.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# # # axs[1].set_title("Refit betas")
# # # difference = channel_corr.T - metrics_chan.T
# # # lim = np.max(np.abs(difference))
# # # ax3 = axs[2].imshow(difference, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# # # axs[2].set_title("Difference")
# # # plt.colorbar(ax1, ax=axs[0])
# # # plt.colorbar(ax2, ax=axs[1])
# # # plt.colorbar(ax3, ax=axs[2])
# # # plt.savefig(f"channel_correlation_comparison_{subject}.png")



# # # lim = np.max(np.abs(metrics_chan))
# # # ch_names = loader.metadata["ch_names"]
# # # regions = anatomy_df['region'].values
# # # ordered_idx = np.argsort(regions)
# # # ordered_chanmap = metrics_chan[:, ordered_idx]

# # # plt.figure(figsize=(30, 19))
# # # sns.heatmap(ordered_chanmap.T, cmap='jet', center=0, vmin=-lim, vmax=lim, cbar=True)
# # # plt.title("Refit betas")
# # # plt.xlabel("Time (ms)")
# # # plt.ylabel("Channels")
# # # plt.xticks(ticks=arranged_timearray, labels=timearray)
# # # plt.yticks(ticks=np.arange(len(regions)), labels=np.array(regions)[ordered_idx])
# # # plt.tight_layout()
# # # plt.savefig(f"channel_correlation_refit_{subject}.png")



# # # group_corr = np.zeros((len(np.unique(regions)), n_timepoints))
# # # for i, region in enumerate(np.unique(regions)):
# # #     idx_r = np.where(regions == region)[0]
# # #     group_corr[i, :] = np.mean(metrics_chan[:, idx_r], axis=1)

# # # lim = np.max(np.abs(group_corr))
# # # plt.figure(figsize=(20, 12))
# # # plt.imshow(group_corr, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
# # # plt.colorbar()
# # # plt.axvline(x=onset, color='black', linestyle='--')
# # # plt.xticks(ticks=arranged_timearray, labels=timearray)
# # # plt.yticks(ticks=np.arange((len(np.unique(regions)))), labels=np.unique(regions))
# # # plt.title("Refit betas")
# # # plt.xlabel("Frequency bands")
# # # plt.ylabel("Regions")
# # # plt.tight_layout()
# # # plt.savefig(f"channel_correlation_refit_group_{subject}.png")




# # # from sklearn.model_selection import KFold, StratifiedKFold
# # # from sklearn.linear_model import RidgeCV
# # # from sklearn.svm import LinearSVC
# # # from sklearn.pipeline import make_pipeline
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.metrics import r2_score, roc_auc_score, f1_score
# # # from scipy.stats import pearsonr
# # # from joblib import Parallel, delayed




# # # def process_timepoint_ridge(t_idx, ch_idx, X_data, y, cv, pipeline):
# # #     """Calcule le score pour un canal spécifique à un point temporel donné"""
# # #     X = X_data[:, ch_idx, :, t_idx].reshape(X_data.shape[0], -1)
# # #     fold_scores = []
# # #     fold_corr = []
# # #     fold_betas = []
    
# # #     for train_idx, test_idx in cv.split(X, y):
# # #         X_train, X_test = X[train_idx], X[test_idx]
# # #         y_train, y_test = y[train_idx], y[test_idx]
        
# # #         pipeline.fit(X_train, y_train)
# # #         y_pred = pipeline.predict(X_test)
# # #         r, _ = pearsonr(y_test, y_pred)
# # #         score = r2_score(y_test, y_pred)
# # #         fold_scores.append(score)
# # #         fold_corr.append(np.arctanh(r))
# # #         fold_betas.append(pipeline.named_steps['ridgecv'].coef_)
    
# # #     return {
# # #         'timepoint': t_idx,
# # #         'channel': ch_idx,
# # #         'score': np.mean(fold_scores),
# # #         'metric': np.mean(fold_corr)
# # #     }

# # # def process_timepoint_ridge(t_idx, X_data, y, cv, pipeline):
# # #     """Traite un point temporel spécifique"""
# # #     X = X_data[..., t_idx].reshape(X_data.shape[0], -1)
# # #     fold_scores = []
# # #     fold_corr = []
# # #     fold_betas = []
    
# # #     for train_idx, test_idx in cv.split(X, y):
# # #         X_train, X_test = X[train_idx], X[test_idx]
# # #         y_train, y_test = y[train_idx], y[test_idx]
        
# # #         pipeline.fit(X_train, y_train)
# # #         y_pred = pipeline.predict(X_test)
# # #         r, _ = pearsonr(y_test, y_pred)
# # #         score = r2_score(y_test, y_pred)
# # #         fold_scores.append(score)
# # #         fold_corr.append(np.arctanh(r))
# # #         fold_betas.append(pipeline.named_steps['ridgecv'].coef_)
    
# # #     return {
# # #         'timepoint': t_idx,
# # #         'score': np.mean(fold_scores),
# # #         'corr': np.mean(fold_corr),
# # #         'beta': np.mean(fold_betas, axis=0)
# # #     }


# # # # def process_timepoint_svm(t_idx, X_data, y, cv, pipeline):
# # # #     """Traite un point temporel spécifique"""
# # # #     X = X_data[..., t_idx].reshape(X_data.shape[0], -1)
# # # #     fold_scores = []
# # # #     fold_roc_auc = []
# # # #     fold_betas = []
    
# # # #     for train_idx, test_idx in cv.split(X, y):
# # # #         X_train, X_test = X[train_idx], X[test_idx]
# # # #         y_train, y_test = y[train_idx], y[test_idx]
        
# # # #         pipeline.fit(X_train, y_train)
# # # #         y_pred = pipeline.predict(X_test)
# # # #         y_proba = pipeline.decision_function(X_test)
# # # #         f1 = f1_score(y_test, y_pred, average='weighted')
# # # #         roc_auc = roc_auc_score(y_test, y_proba)
# # # #         fold_scores.append(f1)
# # # #         fold_roc_auc.append(roc_auc)
# # # #         fold_betas.append(pipeline.named_steps['linearsvc'].coef_)
    
# # # #     return {
# # # #         'timepoint': t_idx,
# # # #         'score': np.mean(fold_scores),
# # # #         'roc_auc': np.mean(fold_roc_auc),
# # # #         'beta': np.mean(fold_betas, axis=0)
# # # #     }





# # # alphas = np.logspace(-1, 4, 6)
# # # pipeline = make_pipeline(
# # #     StandardScaler(),
# # #     RidgeCV(alphas)
# # # )
# # # n_splits = 5
# # # cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# # # n_timepoints = X_power.shape[-1]
# # # n_channels = X_power.shape[1]

# # # tasks = [(t, ch) for t in range(n_timepoints) for ch in range(n_channels)]

# # # y = behavior_df["entropy"].values

# # # results = Parallel(n_jobs=-1, verbose=10)(
# # #     delayed(process_timepoint_ridge)(*task, X_power, y, cv, pipeline) for task in tasks
# # # )


# # #     # Réorganiser les résultats dans des matrices
# # # scores_chan = np.zeros((n_timepoints, n_channels))
# # # metrics_chan = np.zeros((n_timepoints, n_channels))

# # # for result in results:
# # #     t = result['timepoint']
# # #     ch = result['channel']
# # #     scores_chan[t, ch] = result['score']
# # #     metrics_chan[t, ch] = result['metric']




# # # # pipeline = make_pipeline(
# # # #     StandardScaler(),
# # # #     LinearSVC()
# # # # )
# # # # n_splits = 5
# # # # cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# # # # X_power = features.power_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()
# # # # # X_sin = features.phase_sin_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()
# # # # # X_cos = features.phase_cos_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()


# # # # # X_power = power_data[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()
# # # # # X_phase = phase_data[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()
# # # # n_timepoints = X_power.shape[-1]
# # # # y = behavior_df["fb"].values

# # # # behavior_df["fb"]

# # # # # Exécution parallèle sur tous les points temporels
# # # # results = Parallel(n_jobs=-1, verbose=10)(
# # # #     delayed(process_timepoint_ridge)(t_idx, X_power, y, cv, pipeline) 
# # # #     for t_idx in range(n_timepoints)
# # # # )

# # # # results = Parallel(n_jobs=-1, verbose=10)(
# # # #     delayed(process_timepoint_svm)(t_idx, X_power, y, cv, pipeline) 
# # # #     for t_idx in range(n_timepoints)
# # # # )


# # # # # Récupération des résultats dans le bon ordre
# # # # scores = np.zeros(n_timepoints)
# # # # corr = np.zeros(n_timepoints)
# # # # betas = np.zeros((n_timepoints, X_power.shape[1], X_power.shape[2]))

# # # # for result in results:
# # # #     i = result['timepoint']
# # # #     scores[i] = result['score']
# # # #     corr[i] = result['corr']
# # # #     betas[i] = result['beta'].reshape(X_power.shape[1], X_power.shape[2])


# # # # X_power.shape
# # # # scores = np.zeros(n_timepoints)
# # # # roc = np.zeros(n_timepoints)
# # # # betas = np.zeros((n_timepoints, X_power.shape[1], X_power.shape[2]))

# # # # for result in results:
# # # #     i = result['timepoint']
# # # #     scores[i] = result['score']
# # # #     roc[i] = result['roc_auc']
# # # #     betas[i] = result['beta'].reshape(X_power.shape[1], X_power.shape[2])




# # # # plt.figure(figsize=(10, 6))
# # # # plt.plot(roc, label='ROC AUC')
# # # # plt.plot(scores, label='F1 Score')
# # # # plt.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
# # # # plt.axvline(x=onset, color='g', linestyle='--', label='Onset')
# # # # plt.xticks(ticks=arranged_timearray, labels=timearray)
# # # # plt.legend()
# # # # plt.savefig(f"accuracy_tc_svm_{subject}.png")


# # # # band_names = list(FREQUENCY_BANDS.keys())

# # # # ch_names = loader.metadata["ch_names"]
# # # # regions = anatomy_df['region'].values
# # # # ordered_idx = np.argsort(regions)

# # # # max_corr_idx = np.argmax(roc)
# # # # max_corr = roc[max_corr_idx]
# # # # orederd_betas = np.abs(betas[max_corr_idx, ordered_idx])
# # # # plt.figure(figsize=(20, 12))
# # # # plt.imshow(orederd_betas, aspect='auto', cmap='jet', interpolation='nearest')
# # # # plt.colorbar()
# # # # plt.xticks(ticks=np.arange(len(band_names)), labels=band_names, rotation=45)
# # # # plt.yticks(ticks=np.arange(len(regions)), labels=np.array(regions)[ordered_idx])
# # # # plt.title(f"time : {max_corr_idx} | corr : {max_corr:.2f}")
# # # # plt.savefig(f"beta_coefficients_svm_{subject}.png")


# # # # best_corr_idx = np.argsort(roc)[::-1][:25]
# # # # plt.figure(figsize=(40, 30))
# # # # for i, idx in enumerate(best_corr_idx):
# # # #     plt.subplot(5, 5, i+1)
# # # #     betas_prepared = np.abs(betas[idx])
# # # #     group_betas = np.zeros((len(np.unique(regions)), len(band_names)))
# # # #     for i, region in enumerate(np.unique(regions)):
# # # #         idx_r = np.where(regions == region)[0]
# # # #         group_betas[i] = np.mean(betas_prepared[idx_r], axis=0)
# # # #     plt.imshow(group_betas, aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=0.3)
# # # #     plt.colorbar()
# # # #     plt.xticks(ticks=np.arange(len(band_names)), labels=band_names, rotation=45)
# # # #     plt.yticks(ticks=np.arange((len(np.unique(regions)))), labels=np.unique(regions))
# # # #     plt.title(f"Time: {idx} | Corr: {roc[idx]:.2f}")
# # # # plt.tight_layout()
# # # # plt.savefig(f"beta_coefficients_best_corr_svm_{subject}.png")



