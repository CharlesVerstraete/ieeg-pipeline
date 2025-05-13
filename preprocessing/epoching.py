#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Epoching the signal
"""

# Import libraries
from preprocessing.config import *
from preprocessing.utils.data_helper import *
from preprocessing.utils.plot_helper import *
from preprocessing.utils.signal_helper import *
import json
############################################################################################################################################################################
############################################################################################################################################################################
# Plotting the alignment and the bad trials
############################################################################################################################################################################
############################################################################################################################################################################

alignement_matrix = np.zeros((len(SUBJECTS), 1600))

for (i, subject) in enumerate(SUBJECTS):
    print(f"Processing subject {subject}")
    behav_files = get_fileslist(os.path.join(SECOND_ANALYSIS__DATA_DIR, f"sub-{subject:03}", "raw", "beh"), "stratinf_beh.tsv")
    beh_df = pd.read_csv(behav_files[0], sep="\t")
    events_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "ieeg", f"sub-{int(subject):03}_events.tsv")
    events = pd.read_csv(events_path, sep = "\t")

    bad_trial = events[events["align_eeg"] == "-"]["trial_count"].unique().astype(int)
    good_trial = events[events["align_eeg"] != "-"]["trial_count"].unique().astype(int)
    alignement_matrix[i, good_trial] = 1
    alignement_matrix[i, bad_trial] = -1


n_bad = alignement_matrix[alignement_matrix == -1].shape[0]
n_good = alignement_matrix[alignement_matrix == 1].shape[0]
print(f"Number of bad trials : {n_bad}")
print(f"Number of good trials : {n_good}")
percent_bad = (n_bad / (n_good + n_bad))*100
print(f"Ratio of good trials : {percent_bad:.2f}%")

correct_cmap = ListedColormap(['#d73027', '#f7f7f7', '#1a9850'])  # Rouge, blanc, vert (bon contraste)

fig, ax = plt.figure(figsize=(21, 12)), plt.subplot(111)
im = ax.imshow(alignement_matrix, aspect="auto", cmap=correct_cmap, origin="lower", interpolation="nearest", vmin=-1, vmax=1)
ax.set_xlabel("Trials")
ax.set_ylabel("Subjects")

y_ticks = np.arange(0, len(SUBJECTS), 1)
y_labels = [f"{SUBJECTS[i]:d}" for i in range(len(SUBJECTS))]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "preprocessing", "alignement_matrix.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
plt.show()


plt.figure(figsize=(21, 12))
for (i, subject) in enumerate(SUBJECTS):
    plt.subplot(3, 5, i + 1)
    subject = SUBJECTS[0]
    events_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "ieeg", f"sub-{int(subject):03}_events.tsv")
    events = pd.read_csv(events_path, sep = "\t")
    #events = events[events["align_eeg"] != "-"].copy().reset_index(drop=True)
    events = events[events["run"] == 1].copy().reset_index(drop=True)
    plt.plot(np.abs(events["time_log"].diff() - (events["sample"].diff()/SAMPLING_RATE)))

    plt.title(f"sub-{int(subject):03}")
plt.tight_layout()
plt.show()




############################################################################################################################################################################
############################################################################################################################################################################
# Create the epochs
############################################################################################################################################################################
############################################################################################################################################################################

for subject in [25, 28]:
    print(f"Processing subject {subject}")
    electrode_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "anat", f"sub-{int(subject):03}_electrodes-bipolar.csv")
    electrode = pd.read_csv(electrode_path)

    events_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "ieeg", f"sub-{int(subject):03}_events.tsv")
    events = pd.read_csv(events_path, sep = "\t")
    events_updated = events.copy()
    events_updated["sample_update"] = 0
    n_run = events["run"].max().astype(int)

    epochs = []
    pads = []
    offsets = []

    for run in range(1, n_run+1) :
        raw_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "filtered", f"sub-{int(subject):03}_run-{run:02}_bipolar-denoised.fif")
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

        clean_electrode = electrode[electrode["name"].isin(raw.ch_names)]
        clean_electrode = clean_electrode[clean_electrode["is_inmask"]]
        clean_electrode = clean_electrode[(clean_electrode["lobe"] == "Fr") | (clean_electrode["lobe"] == "Ins")]

        picks = mne.pick_channels(raw.ch_names, clean_electrode["name"].values.astype(str).tolist())

        events_run = events[events["run"] == run].copy().reset_index(drop=True)

        first_event_sample = events_run["sample"].values[0]
        last_event_sample = events_run["sample"].values[-1]

        raw, pad_added, start_offset, end_offset = check_lenght(raw, SAMPLING_RATE, first_event_sample, last_event_sample, epoch_padding)

        pads.append(pad_added)
        offsets.append([int(start_offset), int(end_offset)])

        events_run["sample"] = events_run["sample"] + start_offset
        events_signal = np.stack([events_run["sample"].values, np.zeros(len(events_run)), events_run["trigger_value"].values]).T.astype(int)
        events_updated.loc[events_updated["run"] == run, "sample_update"] = events_run["sample"].values
        tmp = mne.Epochs(raw, events_signal, tmin=-epoch_padding, tmax=epoch_padding,  baseline=None, event_id = [10, 20, 30, 11, 21, 31, 12, 22, 32], preload = True, on_missing = "ignore", picks = picks, reject_by_annotation=False, reject=None) 
        epochs.append(tmp)
        del tmp, raw
        gc.collect()

    epoch_complete = mne.concatenate_epochs(epochs)
    ch_names = epoch_complete.info["ch_names"]
    bad_trials = events_updated.loc[events_updated["align_eeg"] == '-', "trial_count"].unique().tolist()

    meta_data = {
        "n_run": int(n_run),
        "bad_trials": bad_trials,
        'sfreq': SAMPLING_RATE,
        'n_epochs': len(epoch_complete),
        'n_channels': len(ch_names),
        'ch_names': ch_names,
        'padding added': pads,
        'offsets': offsets
    }

    epochs_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "epochs", f"sub-{int(subject):03}_ieeg-epochs.fif")

    events_update_path =  os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "epochs",f"sub-{int(subject):03}_events-updated.tsv")
    metadata_path =  os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "epochs", f"sub-{int(subject):03}_meta-data.json")

    with open(metadata_path, 'w', encoding='utf-8') as f: 
        json.dump(meta_data, f, ensure_ascii=False, indent=4)

    events_updated.to_csv(events_update_path, sep = "\t", index = False)
    epoch_complete.save(epochs_path, overwrite = True, verbose='error')
    print(f"Saving epochs to {epochs_path}")

    del epoch_complete
    gc.collect()
