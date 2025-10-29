# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""
Module pour charger les données iEEG pour l'analyse
"""
# from decoding.config import *
from analysis.decoding.config import *


class iEEGDataLoader:
    
    def __init__(self, subject):
        self.subject = subject
        self.metadata = None
        self.ch_names = None
        self.times = None
        self.freqlist = None
        self.n_epochs = None
        self._load_metadata()
        
    def _load_metadata(self):
        # metadata_path = os.path.join(DATA_DIR, 'tfr', f"sub-{int(self.subject):03}_tfr-metadata.json")
        metadata_path = os.path.join(DATA_DIR, f"sub-{int(self.subject):03}", "preprocessed", "timefreq", f"sub-{int(self.subject):03}_tfr-metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.ch_names = self.metadata['ch_names']
        self.times = np.array(self.metadata['times_decimed'])
        self.freqlist = np.array(self.metadata['freqlist'])
        self.n_epochs = self.metadata['n_epochs']
        self.n_channels = self.metadata['n_channels']
    
    def load_power(self, path=None):
        if path is None:
            path = os.path.join(DATA_DIR, f"sub-{int(self.subject):03}", "preprocessed", "timefreq", f"sub-{int(self.subject):03}_tfr-power.npy")
            power = np.memmap(
                path, dtype='float16', mode='r', 
                shape=(
                    self.n_epochs, 
                    self.n_channels, 
                    n_freqs, 
                    n_times_decimed,
                )
        )
        else:
            power = np.load(path, mmap_mode='r')
        print(f"Données de puissance chargées: {power.shape}")
        return np.array(power, copy=True)

    def load_baseline(self, path = None):
        if path is None:
            path = os.path.join(DATA_DIR, f"sub-{int(self.subject):03}", "preprocessed", "timefreq", f"sub-{int(self.subject):03}_tfr-baseline.npy")
        baseline = np.load(path)
        print(f"Données de baseline chargées: {baseline.shape}")
        return baseline
    
    def load_phase(self):
        phase_path =os.path.join(DATA_DIR, f"sub-{int(self.subject):03}", "preprocessed", "timefreq", f"sub-{int(self.subject):03}_tfr-phase.npy")
        phase = np.memmap(
            phase_path, dtype='float16', mode='r', 
            shape=(
                self.n_epochs, 
                self.n_channels, 
                n_freqs, 
                n_times_decimed
            )
        )
        print(f"Données de phase chargées: {phase.shape}")
        return np.array(phase, copy=True)

    
    def load_behavior(self, model_type):        
        beh_path = os.path.join(ROOT_DIR, 'tmp_data', "beh", f"sub-{self.subject:03}_task-stratinf_beh.csv")
        # simu_path = os.path.join(BEH_DIR, 'forced', f"sub-{self.subject:03}_task-stratinf_sim-forced.csv")
        simu_path = os.path.join(ROOT_DIR, 'tmp_data', "hmm", f"sub-{self.subject:03}_task-stratinf_beh-hmm.csv")
        events_path = os.path.join(DATA_DIR, f"sub-{int(self.subject):03}", "preprocessed", "epochs", f"sub-{self.subject:03}_events-updated.tsv")

        beh_df = pd.read_csv(beh_path, sep=",")
        simu_df = pd.read_csv(simu_path, sep=",")
        events = pd.read_csv(events_path, sep="\t")

        onset_events = events[events["trigger_value"].isin(stim_ids)].reset_index(drop = True)
        onset_events = onset_events[onset_events["align_eeg"] != "-"].reset_index(drop = True)

        beh_df["trial_count"] = np.arange(1, len(beh_df) + 1)
        
        simu_df = simu_df[simu_df["training"] == 0]
        beh_orig_eegmap = beh_df[beh_df["trial_count"].isin(onset_events["trial_count"])].reset_index(drop = True)
        common_trials = np.intersect1d(beh_orig_eegmap["trial_count"], simu_df["trial_count"])
        event_tokeep = beh_orig_eegmap[beh_orig_eegmap["trial_count"].isin(common_trials)].index

        simu_behav_clean = simu_df[simu_df["trial_count"].isin(common_trials)].reset_index(drop = True)
        events_df = events[events["trial_count"].isin(common_trials)].reset_index(drop = True)
        events_df["decimated_sample"] = (events_df["sample_update"] / decimation).astype(int)

        return simu_behav_clean, event_tokeep, events_df

    def get_events_dict(self, beh_df, events):
        """Get a dictionary of events"""
        events_dict = {
            "trial": [],
            "onset" : [],
            "action" : [],
            "fb" : [],
            "is_missing_fb" : [],
            "is_missing_action" : [],
            "beh_rt" : [],
            "events_rt" : [],
            "padded_fb" : [],
            "padded_action" : []
        }
        zero_epoch = int(epoch_padding * sr_decimated)
        for trial, beh_rt in beh_df[["trial_count", "rt"]].values:
            event_trial = events[events["trial_count"] == trial]
            event_seq = event_trial["decimated_sample"].values
            events_dict["trial"].append(trial)
            events_dict["onset"].append(zero_epoch)
            if event_seq[1] == 0 or (event_seq[0] > event_seq[1]) or (event_seq[1] - event_seq[0] > 5*sr_decimated):
                events_dict["is_missing_action"].append(1)
                estimated_rt = int(zero_epoch + beh_rt * sr_decimated)
                events_dict["action"].append(estimated_rt)
                events_dict["events_rt"].append(beh_rt)
            else :
                events_dict["is_missing_action"].append(0)
                events_dict["action"].append((event_seq[1]-event_seq[0]) + zero_epoch)
                events_dict["events_rt"].append((event_seq[1]-event_seq[0])/sr_decimated)
            if event_seq[2] == 0 or (event_seq[0] > event_seq[2]) or (event_seq[2] - event_seq[1] > 2*sr_decimated):
                events_dict["is_missing_fb"].append(1)
                estimated_fb = int(zero_epoch + (beh_rt + 1) * sr_decimated)
                events_dict["fb"].append(estimated_fb)
            else :
                events_dict["is_missing_fb"].append(0)
                events_dict["fb"].append((event_seq[2]-event_seq[1]) + events_dict["action"][-1])
            events_dict["beh_rt"].append(beh_rt)
        for key in events_dict.keys():
            events_dict[key] = np.array(events_dict[key])
        return events_dict

    
    def load_anatomy(self):
        anat_path = os.path.join(DATA_DIR, f"sub-{int(self.subject):03}", "raw", 'anat', f"sub-{int(self.subject):03}_electrodes-bipolar.csv")
        anat_df = pd.read_csv(anat_path)
        anat_df = anat_df[anat_df["name"].isin(self.ch_names)].reset_index(drop = True)
        anat_df["chan_idx"] = anat_df["name"].apply(lambda x: self.ch_names.index(x))
        return anat_df

    def extract_window_of_interest(self, data, start_idx=WOI_start_idx, end_idx=WOI_end_idx):
        """Extract the window of interest from the data"""
        return data[:, :, :, start_idx:end_idx]


class GroupLoader:

    def __init__(self, subjects, tmin = -WOI_start_time, tmax = WOI_end_time):
        self.subjects = subjects
        self.n_subjects = len(subjects)
        self.data_loaders = {subject : iEEGDataLoader(subject) for subject in subjects}
        self.n_timepoints = int((np.abs(tmin) + np.abs(tmax)) * sr_decimated)
        self.n_channels = np.sum([loader.metadata['n_channels'] for loader in self.data_loaders.values()])
        self.n_frband = n_frband
        self.anatomy_data = None
        self.beh_data = None
        self.global_score = {}
        self.global_metric = {}
        self.channel_score = {}
        self.channel_metric = {}
        self.global_betas = {}
        self.channel_betas = {}
        self.event = {}

    def load_decoding_metric(self, fold, var, model_name, model_type, mode):
        """Load the decoding data for a specific subject and variable"""
        if mode == "channel":
            self.channel_metric[var] = np.zeros((self.n_timepoints, self.n_channels))
            start_idx = 0
            for i, subject in enumerate(self.subjects) :
                file_path = os.path.join(OUTPUT_DIR, f"{model_name}", f"{model_type}", f"sub-{int(subject):03}_{var}_{model_name}_fold-{fold}_metric-{mode}.npy")
                score = np.load(file_path)
                end_idx = start_idx + score.shape[-1]
                self.channel_metric[var][:, start_idx:end_idx] = score
                start_idx = end_idx
        else:
            self.global_metric[var] = np.zeros((len(self.subjects), self.n_timepoints))
            for i, subject in enumerate(self.subjects) :
                file_path = os.path.join(OUTPUT_DIR, f"{model_name}", f"{model_type}", f"sub-{int(subject):03}_{var}_{model_name}_fold-{fold}_metric-{mode}.npy")
                score = np.load(file_path)
                self.global_metric[var][i] = score

    def load_decoding_score(self, fold, var, model_name, model_type, mode):
        """Load the decoding data for a specific subject and variable"""
        if mode == "channel":
            self.channel_score[var] = np.zeros((self.n_timepoints, self.n_channels))
            start_idx = 0
            for i, subject in enumerate(self.subjects) :
                file_path = os.path.join(OUTPUT_DIR, f"{model_name}", f"{model_type}", f"sub-{int(subject):03}_{var}_{model_name}_fold-{fold}_score-{mode}.npy")
                score = np.load(file_path)
                end_idx = start_idx + score.shape[-1]
                self.channel_score[var][:, start_idx:end_idx] = score
                start_idx = end_idx
        else:
            self.global_score[var] = np.zeros((len(self.subjects), self.n_timepoints))
            for i, subject in enumerate(self.subjects) :
                file_path = os.path.join(OUTPUT_DIR, f"{model_name}", f"{model_type}", f"sub-{int(subject):03}_{var}_{model_name}_fold-{fold}_score-{mode}.npy")
                score = np.load(file_path)
                self.global_score[var][i] = score
            
    def load_decoding_betas(self, fold, var, model_name, model_type, mode):
        """Load the decoding betas for a specific subject and variable"""
        if mode == "channel":
            self.channel_betas[var] = np.zeros((self.n_timepoints, self.n_channels, self.n_frband))
        else:
            self.global_betas[var] = np.zeros((self.n_timepoints, self.n_channels, self.n_frband))
        start_idx = 0
        for i, subject in enumerate(self.subjects) :
            file_path = os.path.join(OUTPUT_DIR, f"{model_name}", f"{model_type}", f"sub-{int(subject):03}_{var}_{model_name}_fold-{fold}_betas-{mode}.npy")
            betas = np.load(file_path)
            end_idx = start_idx + betas.shape[1]
            if mode == "channel":
                self.channel_betas[var][:, start_idx:end_idx] = betas
            else:
                self.global_betas[var][:, start_idx:end_idx] = betas
            start_idx = end_idx

    def _format_exploration(self, df, correct_treshold=12):
        """
        Format the exploration data by checking for new blocks and categorizing exploration vs exploitation.
        """
        df["explor_exploit"] = ""
        correct_count = 0
        for idx, row in df.iterrows():
            if row["new_block"] == 1:
                correct_count = 0
            else:
                correct_count += row["correct"]
            if (((row["hmm_strat"] != row["selected_strategy"]) & (correct_count <= correct_treshold))):
                df.at[idx, "explor_exploit"] = "explor"
            elif ((correct_count > correct_treshold) | (row["hmm_strat"] == row["selected_strategy"])) :
                df.at[idx,"explor_exploit"] = "exploit" 
        return df

    def load_behavior(self, model_type):
        """Load the behavior data for all subjects"""
        self.beh_data = pd.DataFrame()
        for subject in self.subjects:
            simu_behav_clean, _, _ = self.data_loaders[subject].load_behavior(model_type)
            # simu_behav_clean = self._format_exploration(simu_behav_clean)
            self.beh_data = pd.concat([self.beh_data, simu_behav_clean], ignore_index=False)
        self.get_events_dict()

    def load_anatomy(self, sort=False):
        """Load the anatomy data for all subjects"""
        self.anatomy_data = pd.DataFrame()
        for subject in self.subjects:
            anatomy_data = self.data_loaders[subject].load_anatomy()
            anatomy_data["subject"] = subject
            anatomy_data.sort_values(by=["chan_idx"], inplace=True)
            self.anatomy_data = pd.concat([self.anatomy_data, anatomy_data], ignore_index=True)
        self.anatomy_data["chan_idx_global"] = np.arange(0, self.n_channels)
        self.anatomy_data["area_order"] = [area_dict[roi] for roi in self.anatomy_data["region"]]
        if sort:
            self.anatomy_data.sort_values(by=["area_order"], inplace=True)

    def get_events_dict(self):
        """Get a dictionary of events"""
        rt = self.beh_data[(self.beh_data["rt"] > 0) & (self.beh_data["rt"] < 5)]["rt"].values
        mean_rt = np.mean(rt)
        sem_rt = np.std(rt)/np.sqrt(self.n_subjects)
        sem__rt_idx = int(sem_rt*sr_decimated)
        rt_idx = int(onset + mean_rt*sr_decimated)
        fb_prev = onset - sr_decimated
        fb = int((rt_idx + sr_decimated))
        self.event = {
            "fb_prev": fb_prev,
            "onset": onset,
            "rt": rt_idx,
            "fb_next": fb,
            "sem_rt": sem__rt_idx
        }
    
    


    #     """Save the results to a file"""
    #     if mode == "channel":
    #         score_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_score-channel.npy")
    #         metric_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_metric-channel.npy")
    #         betas_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_betas-channel.npy")
    #     else:
    #         score_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_score-global.npy")
    #         metric_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_metric-global.npy")
    #         betas_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_betas-global.npy")
    #     np.save(score_file, self.results['score'])
    #     np.save(metric_file, self.results['metric'])
    #     np.save(betas_file, self.results['betas'])