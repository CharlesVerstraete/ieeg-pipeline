# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""

"""

from decoding.config import *
# from config import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from scipy.stats import pearsonr
from joblib import Parallel, delayed

class Decoding:
    def __init__(self, data, beh, anat):
        self.data = data
        self.beh = beh
        self.anat = anat
        self.ch_names = anat["name"].values
        self.regions = anat["region"].values
        self.ordered_regions_idx = np.argsort(self.regions)
        self.ordered_regions = self.regions[self.ordered_regions_idx]
        self.n_regions = len(np.unique(self.regions))
        self.subject = beh["subject"].values[0]
        self.n_timepoint = data.shape[-1]
        self.n_epochs = data.shape[0]
        self.n_channels = data.shape[1]
        self.n_freqs = data.shape[2]
        self.model_name = None
        self.alphas = None
        self.classification = None
        self.n_folds = None
        self.cv = None
        self.estimator = None
        self.pipeline = None
        self.results = None
        self.metric_global_dict = {}
        self.metric_channel_dict = {}
        self.score_global_dict = {}
        self.score_channel_dict = {}
        self.betas_global_dict = {}
        self.betas_channel_dict = {}
    
    def create_pipeline(self, n_folds, alphas=None, classification=False):
        """Create a pipeline for Ridge regression with cross-validation"""
        self.n_folds = n_folds
        self.classification = classification
        if self.classification:
            self.estimator = LinearSVC()
            self.cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            self.model_name = "linearsvc"
        else:
            self.alphas = alphas
            self.estimator = RidgeCV(alphas=self.alphas)
            self.cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            self.model_name = "ridgecv"
        self.pipeline = make_pipeline(StandardScaler(), self.estimator)

    def _compute_fold(self, X, y):
        """Compute the score for a specific fold"""
        fold_metrics = []
        fold_betas = []
        fold_score = []
        for train_idx, test_idx in self.cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            self.pipeline.fit(X_train, y_train)
            y_pred = self.pipeline.predict(X_test)
            if not self.classification:  # Régression
                r, _ = pearsonr(y_test, y_pred)
                fold_score.append(mean_squared_error(y_test, y_pred))
                fold_metrics.append(np.arctanh(r))
            else:
                fold_score.append(roc_auc_score(y_test, y_pred))
                fold_metrics.append(f1_score(y_test, y_pred, average='weighted'))
            fold_betas.append(self.pipeline.named_steps[self.model_name].coef_)
        return {
            'score': np.mean(fold_score), 
            'metric': np.mean(fold_metrics), 
            'betas': np.mean(fold_betas, axis=0)
        }

    def compute_score_power(self, y, t, mode = "global"):
        """
        Compute the score for a specific time point.
        """
        if mode == "channel":
            channel_results = {}
            for ch in range(self.n_channels):   
                X = self.data[:, ch, :, t].reshape(self.n_epochs, -1)
                channel_results[ch] = self._compute_fold(X, y)
            results = {t : channel_results}
        elif mode == "global":
            X = self.data[..., t].reshape(self.n_epochs, -1)
            tmp = self._compute_fold(X, y)
            results = {t : tmp}
        
        return results

    def _format_results(self, results, var, mode):
        """Format the results"""
        if mode == "channel":
            self.results = {label: np.zeros((self.n_timepoint, self.n_channels)) for label in ["score", "metric"]}
            self.results['betas'] = np.zeros((self.n_timepoint, self.n_freqs))
            for result in results:
                t = result.keys()
                for ch in range(self.n_channels):
                    self.results['score'][t, ch] = result[t][ch]['score']
                    self.results['metric'][t, ch] = result[t][ch]['metric']
                    self.results['betas'][t, ch] = result[t][ch]['betas'].reshape(self.n_freqs)
            self.metric_channel_dict[var] = self.results['metric']
            self.score_channel_dict[var] = self.results['score']
            self.betas_channel_dict[var] = self.results['betas']
        elif mode == "global":
            self.results = {label: np.zeros(self.n_timepoint) for label in ["score", "metric"]}
            self.results['betas'] = np.zeros((self.n_timepoint, self.n_channels, self.n_freqs))
            for result in results:
                t = result.keys()
                self.results['score'][t] = result[t]['score']
                self.results['metric'][t] = result[t]['metric']
                self.results['betas'][t] = result[t]['betas'].reshape(self.n_channels, self.n_freqs)
            self.metric_global_dict[var] = self.results['metric']
            self.score_global_dict[var] = self.results['score']
            self.betas_global_dict[var] = self.results['betas']
        self.results['timepoint'] = np.arange(self.n_timepoint)



    def run_decoding(self, var, mode = "global", n_jobs = -1, save_results = True, normalize = False):
        """Run the decoding process"""
        y = self.beh[var].values
        if normalize:
            y = (y - np.mean(y)) / np.std(y)
        if mode == "channel":
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.compute_score_power)(y, t, mode = mode) for t in range(self.n_timepoint)
            )
        elif mode == "global":
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.compute_score_power)(y, t, mode = mode) for t in range(self.n_timepoint)
            )
        self._format_results(results, var, mode)
        if save_results:
            self.save_results(var)
    
    def save_results(self, var, mode = "global"):
        """Save the results to a file"""
        if mode == "channel":
            score_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_score-channel.npy")
            metric_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_metric-channel.npy")
            betas_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_betas-channel.npy")
        else:
            score_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_score-global.npy")
            metric_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_metric-global.npy")
            betas_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_betas-global.npy")
        np.save(score_file, self.results['score'])
        np.save(metric_file, self.results['metric'])
        np.save(betas_file, self.results['betas'])

    def plot_tc(self, var, y = "metric", save = True, figsize = (30, 19), extension = "png"):
        """Plot the accuracy time course"""
        baseline = 0.5 if self.classification else 0
        fig, _ = plt.subplots(figsize=figsize)
        plt.plot(self.results['timepoint'], self.results[y])
        plt.axhline(baseline, color='black', linestyle='--')
        plt.axvline(onset, color='red', linestyle='--')
        plt.title(f"{var} {y} time course")
        plt.xlabel("Time (ms)")
        plt.ylabel(y)
        plt.ylim(-0.2, 1)
        plt.xticks(ticks=arranged_timearray, labels=timearray)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(FIGURES_DIR, f"sub-{int(self.subject):03}_{var}_accuracy-tc.{extension}"))
        else:
            plt.show()
        plt.close(fig)

    def plot_multi_tc(self, y = "metric", save = True, figsize = (30, 19), extension = "png"):
        """Plot multiple accuracy time courses"""
        var_list = list(self.metric_global_dict.keys())
        nvar = len(var_list)
        nrows = int(np.sqrt(nvar))
        ncols = int(np.ceil(nvar / nrows))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.flatten()
        for i, var in enumerate(var_list):
            axs[i].plot(self.results['timepoint'], self.metric_global_dict[var])
            axs[i].axhline(0, color='black', linestyle='--')
            axs[i].axvline(onset, color='red', linestyle='--')
            axs[i].set_title(f"{var} {y} global accuracy")
            axs[i].set_xlabel("Time (ms)")
            axs[i].set_ylabel(y)
            axs[i].set_ylim(-0.2, 1)
            axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)
        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axs[j // ncols, j % ncols])
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(FIGURES_DIR, f"sub-{int(self.subject):03}_multi_accuracy-tc.{extension}"))
        else:
            plt.show()
        plt.close(fig)


    def plot_multi_heatmap(self, y = "metric", save = True, figsize = (30, 19), extension = "png"):
        """Plot multiple accuracy time courses"""
        var_list = list(self.metric_channel_dict.keys())
        nvar = len(var_list)
        nrows = int(np.sqrt(nvar))
        ncols = int(np.ceil(nvar / nrows))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
        axs = axs.flatten()
        for i, var in enumerate(var_list):
            to_plot = self.metric_channel_dict[var][:, self.ordered_regions_idx]
            lim = np.max(np.abs(to_plot))
            im = axs[i].imshow(to_plot, aspect='auto', cmap='jet', interpolation='nearest', vmin=-lim, vmax=lim)
            plt.colorbar(im, ax=axs[i])
            axs[i].axvline(onset, color='black', linestyle='--')
            axs[i].set_title(f"{var} {y} channel accuracy")
            axs[i].set_xlabel("Time (ms)")
            axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)
            axs[i].set_yticks(ticks=np.arange(self.n_channels), labels=self.ordered_regions)
        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axs[j // ncols, j % ncols])
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(FIGURES_DIR, f"sub-{int(self.subject):03}_multi_accuracy-tc.{extension}"))
        else:
            plt.show()
        plt.close(fig)


