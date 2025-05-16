# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""
Module pour charger les données iEEG pour l'analyse
"""
from decoding.config import *
# from config import *


class iEEGDataLoader:
    
    def __init__(self, subject):
        self.subject = subject
        self.metadata = None
        self.ch_names = None
        self.times = None
        self.freqlist = None
        self._load_metadata()
        
    def _load_metadata(self):
        metadata_path = os.path.join(DATA_DIR, 'tfr', f"sub-{int(self.subject):03}_tfr-metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.ch_names = self.metadata['ch_names']
    
    def load_power(self):
        power_path = os.path.join(DATA_DIR, 'tfr', f"sub-{int(self.subject):03}_tfr-power.npy")
        power = np.memmap(
            power_path, dtype='float16', mode='r', 
            shape=(
                self.metadata['n_epochs'], 
                self.metadata['n_channels'], 
                self.metadata['n_freqs'], 
                len(self.metadata['times_decimed']),
            )
        )
        print(f"Données de puissance chargées: {power.shape}")
        return power

    
    def load_phase(self):
        phase_path = os.path.join(DATA_DIR, 'tfr',f"sub-{int(self.subject):03}_tfr-phase.npy")
        phase = np.memmap(
            phase_path, dtype='float16', mode='r', 
            shape=(
                self.metadata['n_epochs'], 
                self.metadata['n_channels'], 
                n_freqs, 
                n_times_decimed
            )
        )
        print(f"Données de phase chargées: {phase.shape}")
        return phase

    
    def load_behavior(self):        
        beh_path = os.path.join(DATA_DIR, 'beh', f"sub-{self.subject:03}_task-stratinf_beh.tsv")
        simu_path = os.path.join(DATA_DIR, 'beh',f"sub-{self.subject:03}_task-stratinf_sim-forced.tsv")
        events_path = os.path.join(DATA_DIR, 'events', f"sub-{self.subject:03}_events-updated.tsv")

        beh_df = pd.read_csv(beh_path, sep="\t")
        simu_df = pd.read_csv(simu_path)
        events = pd.read_csv(events_path, sep="\t")

        onset_events = events[events["trigger_value"].isin(stim_ids)].reset_index(drop = True)
        onset_events = onset_events[onset_events["align_eeg"] != "-"].reset_index(drop = True)

        beh_orig_eegmap = beh_df[beh_df["trial_count"].isin(onset_events["trial_count"])].reset_index(drop = True)
        common_trials = np.intersect1d(beh_orig_eegmap["trial_count"], simu_df["trial_count"])
        event_tokeep = beh_orig_eegmap[beh_orig_eegmap["trial_count"].isin(common_trials)].index
        simu_behav_clean = simu_df[simu_df["trial_count"].isin(common_trials)].reset_index(drop = True)
        
        if "rpe" not in simu_behav_clean.columns:
            simu_behav_clean["rpe"] = simu_behav_clean["fb"] - simu_behav_clean["q_choosen"]

        return simu_behav_clean, event_tokeep
    
    def load_anatomy(self):
        anat_path = os.path.join(DATA_DIR, 'anat', f"sub-{int(self.subject):03}_electrodes-bipolar.csv")
        anat_df = pd.read_csv(anat_path)
        anat_df = anat_df[anat_df["name"].isin(self.ch_names)].reset_index(drop = True)
        anat_df["chan_idx"] = anat_df["name"].apply(lambda x: self.ch_names.index(x))
        return anat_df
