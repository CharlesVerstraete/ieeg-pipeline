# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""
Module for decoding analysis of iEEG data
"""
# from decoding.config import *

from analysis.decoding.config import *
from analysis.decoding.loader import GroupLoader
from analysis.decoding.process_features import Features
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from preprocessing.utils.data_helper import *
from preprocessing.utils.anat_helper import *
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


atlas_ref, atlas_img, atlas_data = get_atlas_data(ATLAS_DIR)

atlas = surface.load_surf_data(ATLAS_DIR + "/rh.HCP-MMP1.annot")
fsaverage_orig = datasets.fetch_surf_fsaverage("fsaverage")
fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
coords_orig, _ = surface.load_surf_data(fsaverage_orig[f"pial_right"])
coords_target, _ = surface.load_surf_mesh(fsaverage[f"pial_right"])
resampled_atlas = surface_resample(atlas, coords_orig, coords_target)


class DecodingAnalysis:
    def __init__(self, subjects, model_type):
        self.subjects = subjects
        self.model_type = model_type
        self.group_loaders = GroupLoader(subjects)
        self.group_loaders.load_anatomy()
        self.group_loaders.load_behavior(model_type)
        self.cross_decoding_indiv = {}
        self.cluster_df = self.group_loaders.anatomy_data[["chan_idx_global", "x", "y", "z", "region", "subject", "chan_idx", "roi_n"]].copy()

    def load_matrix(self, var_list, decoding_type = "ridgecv", fold=5):
        """
        Load matrix for all subjects and store them in the group loader.
        """
        for var in var_list:
            self.group_loaders.load_decoding_metric(fold, var, decoding_type, self.model_type, "channel")
            self.group_loaders.load_decoding_metric(fold, var, decoding_type, self.model_type, "global")
            self.group_loaders.load_decoding_betas(fold, var, decoding_type, self.model_type, "channel")
            self.group_loaders.load_decoding_betas(fold, var, decoding_type, self.model_type, "global")

    def save_cross_decoding_matrix(self, matrix, subject, var):
        """
        Save the cross-decoding matrix for all subjects and variable list.
        """
        file_path = os.path.join(OUTPUT_DIR, "cross_decoding", f"sub-{int(subject):03}_{var}_cross_decoding.npy")
        np.save(file_path, matrix)

    def load_cross_decoding_matrix(self, subject, var):
        """
        Load the cross-decoding matrix for a given subject and variable.
        """
        file_path = os.path.join(OUTPUT_DIR, "cross_decoding", f"sub-{int(subject):03}_{var}_cross_decoding.npy")
        self.cross_decoding_indiv[(subject, var)] = np.load(file_path)

    def compute_cross_decoding_matrix(self, subject, var_list, save=True):
        """
        Compute the cross-decoding matrix for a given subjects and variable list.
        """
        power_data = self.group_loaders.data_loaders[subject].load_power()
        beh, event_to_keep = self.group_loaders.data_loaders[subject].load_behavior("transition")
        features = Features(power_data, None)
        features.extract_power_bands()
        X_power = features.power_bands[event_to_keep, :,:, WOI_start_idx:WOI_end_idx].copy()

        indiv_anat =  self.group_loaders.anatomy_data[self.group_loaders.anatomy_data["subject"] == subject].copy()
        idx_chan = indiv_anat["chan_idx_global"].values
        idx_chan.sort()

        for var in var_list:
            indiv_betas = self.group_loaders.global_betas[var][:, idx_chan, :]
            y = beh[var].values
            y_mean = np.mean(y)
            y_centered = y - y_mean
            y_norm = np.sqrt(np.sum(y_centered**2))

            betas = indiv_betas.reshape(indiv_betas.shape[0], -1)
            cross_decoding_matrix = np.zeros((n_timepoints, n_timepoints))
            for t_train_idx in tqdm(range(n_timepoints), desc=f"Cross decoding subject {subject} for variable {var}"):
                X_test_current_time = X_power[:, :, :, t_train_idx].reshape(X_power.shape[0], -1)
                y_pred_all = X_test_current_time @ betas.T
                y_pred_means = np.mean(y_pred_all, axis=0)  
                y_pred_centered = y_pred_all - y_pred_means 
                y_pred_norms = np.sqrt(np.sum(y_pred_centered**2, axis=0)) 
                
                numerators = np.dot(y_centered, y_pred_centered)  
                correlations = numerators / (y_norm * y_pred_norms)
                cross_decoding_matrix[t_train_idx, :] = correlations
                if save:
                    self.save_cross_decoding_matrix(cross_decoding_matrix, subject, var)
            self.cross_decoding_indiv[(subject, var)] = cross_decoding_matrix

    def _compute_distance_matrix(self, var, method='cut', metric='sqeuclidean'):
        """
        Compute the distance matrix for a given variable.
        """
        X = self.group_loaders.channel_score[var].T
        dist_matrix = None
        if method == 'cut':
            event = self.group_loaders.event
            moment_cut = [0,
                        event["fb_prev"],
                        int((event["fb_prev"]+event["onset"])/2),
                        event["onset"],
                        int((event["onset"]+event["rt"])/2),
                        event["rt"],
                        int((event["rt"]+event["fb_next"])/2),
                        event["fb_next"],
                        n_timepoints]
            X_timebin = np.array([np.mean(X[:, moment_cut[i]:moment_cut[i+1]], axis=1) for i in range(len(moment_cut)-1)]).T
            dist_matrix = pdist(X_timebin, metric=metric)
        elif method == 'full':
            dist_matrix = pdist(X, metric=metric)

        return X, dist_matrix

    def clustering_timecourses(self, var, n_clusters, dist_metric='sqeuclidean', method='ward'):
        """
        Perform hierarchical clustering on the cross-decoding matrix for a given variable.
        """
        X, dist_matrix = self._compute_distance_matrix(var, method=method, metric=dist_metric)
        Z = linkage(dist_matrix, method=method)
        self.cluster_df[f'cluster_{var}'] = fcluster(Z, n_clusters, criterion='maxclust')
        self.cluster_df[f'max_{var}'] = np.max(X, axis=1)


    def count_clusters_regions(self, var):
        """
        Count the number of clusters per region for a given variable.
        """
        count = self.cluster_df.groupby(["region", f"cluster_{var}"]).size().reset_index(name="count")
        count["percent_region"] = count["count"] / count.groupby("region")["count"].transform("sum") * 100
        count["percent_cluster"] = count["count"] / count.groupby(f"cluster_{var}")["count"].transform("sum") * 100
        return count
    
    def create_brainmap(self, count, metric="count") : 
        """
        Create a brain map for a given variable.
        """
        atlas_filtered = np.zeros(resampled_atlas.shape)
        for (i, (_, row)) in  enumerate(count.iterrows()) :
            sdf = atlas_ref[atlas_ref["ROI_glasser_2"] == row["region"]]
            for roi_n in sdf["ROI_n"].values:
                atlas_filtered[resampled_atlas == roi_n] = row[metric]
        atlas_filtered[atlas_filtered == 0] = np.nan
        return atlas_filtered

    def make_brain(self, count, cmap="viridis", metric="count"):
        """
        Create a brain map for a given variable and hemisphere.
        """
        atlas_filtered = self.create_brainmap(count, metric=metric)
        brain = mne.viz.Brain(
            "fsaverage",
            "rh",
            "pial",
            cortex="ivory",
            background="white",
            size=(4096, 2160),
            alpha=0.8,
        )
        brain.add_data(
            atlas_filtered,
            colormap=cmap, 
            alpha=1, 
            colorbar=False,
            fmin = 0,
            fmax = np.nanmax(atlas_filtered),
        )
        collection_img = {}
        for view in ["lateral", "medial"]:
            brain.show_view(view)
            screenshot = brain.screenshot()
            nonwhite_pix = (screenshot != 255).any(-1)
            nonwhite_row = nonwhite_pix.any(1)
            nonwhite_col = nonwhite_pix.any(0)
            screenshot = screenshot[nonwhite_row][:, nonwhite_col]
            collection_img[view] = screenshot
        brain.close()
        return collection_img

    def create_brainplot(self, var, cmap, metric="count", save=False):
        """
        Create a brain plot for a given variable.
        """
        count = self.count_clusters_regions(var)
        X = self.group_loaders.channel_score[var].T
        lim = np.max(np.abs(X))
        event = self.group_loaders.event
        clusters = sorted(count[f"cluster_{var}"].unique())
        n_clusters = len(clusters)
        fig, axs = plt.subplots(n_clusters, 3, figsize=(35, 20))
        for i, cluster in enumerate(clusters):
            cluster_count = count[count[f"cluster_{var}"] == cluster]
            mat_idx = self.cluster_df[self.cluster_df[f'cluster_{var}'] == cluster]["chan_idx_global"].values
            collection_img = self.make_brain(cluster_count, metric=metric, cmap=cmap)
            
            axs[i, 0].imshow(collection_img["lateral"])
            axs[i, 0].axis('off')
            axs[i, 1].imshow(collection_img["medial"])
            axs[i, 1].axis('off')
            self._create_colorbar(fig, axs[i, 0], cluster_count, cmap, metric=metric)
            
            im = axs[i, 2].imshow(X[mat_idx], cmap="jet", aspect='auto', origin="lower", vmin=0, vmax=lim)
            axs[i, 2].set_xticks(ticks=arranged_timearray, labels=timearray)
            axs[i, 2].axvline(event["fb_prev"], color='white', linestyle='--')
            axs[i, 2].axvline(event["onset"], color='white', linestyle='-')
            axs[i, 2].axvline(event["rt"], color='white', linestyle='--')
            axs[i, 2].axvline(event["fb_next"], color='white', linestyle='--')
            plt.colorbar(im, ax=axs[i, 2])

        plt.subplots_adjust(
            left=0.02,    
            right=0.98,   
            top=0.98,     
            bottom=0.02,  
            hspace=0.02,   
            wspace=0.01    
        )
        if save:
            plt.savefig(os.path.join(FIGURES_DIR, "decoding", "group", f"{var}_clustered.pdf"),  transparent=True, format='pdf',  bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def _create_colorbar(self, fig, ax1, count,cmap, metric="count"):
        pos_lateral = ax1.get_position()

        cbar_left = pos_lateral.x1
        cbar_bottom = pos_lateral.y0 
        cbar_width = 0.01
        cbar_height = pos_lateral.height
        
        norm = Normalize(vmin=0, vmax=count[metric].max())
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        
        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        _ = plt.colorbar(mappable, cax=cax, orientation="vertical")
        