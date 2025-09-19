# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""
Module pca and manifold manipulation
"""
# from decoding.config import *
from analysis.decoding.config import *
from analysis.decoding.loader import GroupLoader
from analysis.decoding.process_features import Features
from sklearn.decomposition import PCA


class Geometry():
    def __init__(self, subjects, model_type, group_var):
        self.n_bands = len(FREQUENCY_BANDS)
        self.group_loaders = GroupLoader(subjects)
        self.group_loaders.load_anatomy()
        self.group_loaders.load_behavior(model_type)
        self.n_channels = self.group_loaders.n_channels
        self.n_subjects = self.group_loaders.n_subjects
        self.n_timepoints = self.group_loaders.n_timepoints
        self.group_var = group_var
        self.X_group = {group : np.zeros((self.n_channels, len(FREQUENCY_BANDS), self.n_timepoints)) 
                       for group, _ in self.group_loaders.beh_data.groupby(group_var)}
        self.trial_count = self.count_trials()
        self.X = np.zeros((len(self.X_group) * self.n_timepoints * self.n_bands, self.n_channels))
        self.group_labels = []
        self.time_labels = []
        self.band_labels = []  
        
        self.X_pca = None
        self.pca_components = None
        self.pca_explained_variance = None

    def count_trials(self):
        """
        Count the number of trials for each group.
        """
        group_count = self.group_loaders.beh_data.groupby(['subject']+self.group_var).size().reset_index(name="count")
        group_count["group"] = ['_'.join(map(str, x)) for x in group_count[self.group_var].values]
        return group_count

    def _get_powerband(self, model_type, subject, baseline = True, n_jobs=-1):
        """
        Get the power for a specific frequency band across all subjects.
        """
        loader = self.group_loaders.data_loaders[subject]
        power_data = loader.load_power()

        _, event_to_keep = loader.load_behavior(model_type)
        features = Features(power_data, None, subject)
        if baseline:
            features.baseline_signal()
        features.extract_power_bands(n_jobs=n_jobs)
        X_power = features.power_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()
        del features, power_data
        gc.collect()
        return X_power
    
    def fill_group_power(self, model_type, baseline=True, n_jobs=-1):
        """
        Fill the group power data for each group.
        """
        idx_chan = 0
        for subject in self.group_loaders.subjects:
            X_power = self._get_powerband(model_type, subject, baseline=baseline, n_jobs=n_jobs)
            df = self.group_loaders.beh_data[self.group_loaders.beh_data['subject'] == subject]
            n_channels = X_power.shape[1]
            for group, sdf in df.groupby(self.group_var):
                beh_idx = sdf.index.tolist()
                self.X_group[group][idx_chan:idx_chan+n_channels] += np.mean(X_power[beh_idx, :, :], axis=0)
            idx_chan += n_channels

    def make_pcamatrix(self):
        """
        Create a matrix for PCA analysis.
        """
        for i, (group_name, power_data) in enumerate(self.X_group.items()):
            for b in range(self.n_bands):
                for t in range(self.n_timepoints):
                    row_idx = (i * self.n_timepoints * self.n_bands) + (b * self.n_timepoints) + t                
                    self.X[row_idx, :] = power_data[:, b, t]
                    self.group_labels.append(group_name)
                    self.time_labels.append(t)
                    self.band_labels.append(b)
        self.group_labels = np.array(self.group_labels)
        self.time_labels = np.array(self.time_labels)
        self.band_labels = np.array(self.band_labels)

    def apply_pca(self, n_components=5):
        """
        Apply PCA to the data.
        """
        pca = PCA(n_components=n_components)
        self.X_pca = pca.fit_transform(self.X)
        self.pca_components = pca.components_
        self.pca_explained_variance = pca.explained_variance_ratio_
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            print(f"Component {i+1}: Explained Variance Ratio = {ratio*100:.2f}%")
    
    def create_pca_df(self):
        """
        Create a DataFrame for PCA results.
        """
        pca_df = pd.DataFrame(self.X_pca, columns=[f'PC{i+1}' for i in range(self.X_pca.shape[1])])
        pca_df['stim'] = self.group_labels[:, 0]
        pca_df['explor'] = self.group_labels[:, 1]
        pca_df['switch'] = self.group_labels[:, 2]
        pca_df['time'] = self.time_labels
        pca_df["stim"] = pd.to_numeric(pca_df["stim"])
        pca_df["band"] = self.band_labels
        pca_df['band_name'] = [list(FREQUENCY_BANDS.keys())[b] for b in self.band_labels]
        period_vector = self.find_period()
        pca_df['period'] = period_vector*int(len(pca_df)/len(period_vector))
        return pca_df
                
    def find_period(self) : 
        period_vector = []
        for i in range(1280):
            if i < self.group_loaders.event["fb_prev"]:
                period_vector.append("early")
            elif i <  self.group_loaders.event["onset"]:
                period_vector.append("pre_stim")
            elif i <  self.group_loaders.event["rt"]:
                period_vector.append("action_selection")
            elif i <  self.group_loaders.event["fb_next"]:
                period_vector.append("fb_waiting")
            else:
                period_vector.append("end")
        return period_vector
    
    def create_contribution_df(self, n_components=5):
        """
        Create a DataFrame showing the contribution of each channel to the PCA components.
        """
        contribution = pd.DataFrame({f'PC{i+1}' : pca.components_[i, :] for i in range(n_components)})
        contribution["channel"] = self.group_loaders.anatomy_data["name"].values
        contribution["roi_name"] = self.group_loaders.anatomy_data["region"].values
        contribution["channel_position"] = self.group_loaders.anatomy_data["chan_idx_global"].values
        contribution["subject"] = self.group_loaders.anatomy_data["subject"].values
        contribution["area_order"] = self.group_loaders.anatomy_data["area_order"].values
        contribution.sort_values(by="area_order", inplace=True)



    


    
