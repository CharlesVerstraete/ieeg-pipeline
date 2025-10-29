# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""

"""

# from decoding.config import *
from analysis.decoding.config import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.svm import LinearSVC
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from scipy.stats import pearsonr
from joblib import Parallel, delayed

class Decoding:
    def __init__(self, data, beh, anat, tmin=-1.5, tmax=1.5):
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
        self.timearray = np.linspace(tmin, tmax, int((np.abs(tmin) + tmax)*2+1))
        self.arranged_timearray = np.arange(0, self.n_timepoint +1, self.n_timepoint/(len(self.timearray)-1))
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
            # self.estimator = LinearSVC(max_iter=10000)
            self.cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            self.estimator = LinearSVC(max_iter=10000)
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
            if not self.classification:  # Régression
                y_pred = self.pipeline.predict(X_test)
                r, _ = pearsonr(y_test, y_pred)
                fold_score.append(mean_squared_error(y_test, y_pred))
                fold_metrics.append(np.arctanh(r))
            else:
                frozen_clf = FrozenEstimator(self.pipeline)
                calib = CalibratedClassifierCV(estimator=frozen_clf, method="sigmoid", n_jobs=-1)
                calib.fit(X_train, y_train)
                y_pred = calib.predict(X_test)
                y_proba = calib.predict_proba(X_test)
                n_classes = np.unique(y_train).size
                coef = self.pipeline.named_steps[self.model_name].coef_
                if n_classes == 2:
                    roc = roc_auc_score(y_test, y_proba[:, 1])
                    beta = coef
                else:
                    roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
                    beta = np.mean(coef, axis=0)
                f1 = f1_score(y_test, y_pred, average="weighted")
                fold_score.append(roc)
                fold_metrics.append(f1)
            fold_betas.append(beta)
        return {
            'score': np.mean(fold_score), 
            'metric': np.mean(fold_metrics), 
            'betas': np.mean(fold_betas, axis=0)
        }

    def compute_score_power(self, y, t, mode):
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

########################################################################
# To modify and optimize

    def _format_results(self, results, var, mode):
        """Format the results"""
        if mode == "channel":
            self.results = {label: np.zeros((self.n_timepoint, self.n_channels)) for label in ["score", "metric"]}
            self.results['betas'] = np.zeros((self.n_timepoint, self.n_channels, self.n_freqs))
            for result in results:
                t = list(result.keys())[0]
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
                t = list(result.keys())[0]
                self.results['score'][t] = result[t]['score']
                self.results['metric'][t] = result[t]['metric']
                self.results['betas'][t] = result[t]['betas'].reshape(self.n_channels, self.n_freqs)
            self.metric_global_dict[var] = self.results['metric']
            self.score_global_dict[var] = self.results['score']
            self.betas_global_dict[var] = self.results['betas']
        self.results['timepoint'] = np.arange(self.n_timepoint)

    # def _format_results(self, results, var, mode):
    #     """Format the results"""



    def run_decoding(self, var, mode = "global", n_jobs = -1, model_type = "", save_results = True, normalize = False):
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
            self.save_results(var, mode, model_type)

    def save_results(self, var, mode, model_type):
        """Save the results to a file"""
        score_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"{model_type}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_score-{mode}.npy")
        metric_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"{model_type}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_metric-{mode}.npy")
        betas_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"{model_type}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_betas-{mode}.npy")
        np.save(score_file, self.results['score'])
        np.save(metric_file, self.results['metric'])
        np.save(betas_file, self.results['betas'])

    def load_results(self, var, mode, model_type):
        """Load the results from a file"""
        score_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"{model_type}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_score-{mode}.npy")
        metric_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"{model_type}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_metric-{mode}.npy")
        betas_file = os.path.join(OUTPUT_DIR, f"{self.model_name}", f"{model_type}", f"sub-{int(self.subject):03}_{var}_{self.model_name}_fold-{self.n_folds}_betas-{mode}.npy")
        self.results = {
            'score': np.load(score_file),
            'metric': np.load(metric_file),
            'betas': np.load(betas_file)
        }
        self.n_timepoint = np.array(self.results['metric']).shape[0]
        if mode == "channel":
            self.metric_channel_dict[var] = self.results['metric']
            self.score_channel_dict[var] = self.results['score']
            self.betas_channel_dict[var] = self.results['betas']
        elif mode == "global":
            self.metric_global_dict[var] = self.results['metric']
            self.score_global_dict[var] = self.results['score']
            self.betas_global_dict[var] = self.results['betas']
        


    def plot_tc(self, var, y = "metric", save = True, figsize = (30, 19), ylim = (-0.2, 1), extension = "png", fig = None, ax = None):
        """Plot the accuracy time course"""
        baseline = 0.5 if self.classification else 0
        # fig, ax = plt.subplots(figsize=figsize)
        if y == "metric" :
            val = self.metric_global_dict[var]
        else:
            val = self.score_global_dict[var]
        ax.plot(np.arange(self.n_timepoint), val)
        ax.axhline(baseline, color='black', linestyle='--')
        ax.axvline(onset, color='red', linestyle='--')
        ax.set_title(f"{var} {y} time course")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(y)
        ax.set_ylim(ylim)
        ax.set_xticks(ticks=self.arranged_timearray, labels=self.timearray)
        if save:
            fig.savefig(os.path.join(FIGURES_DIR, f"sub-{int(self.subject):03}_{var}_accuracy-tc.{extension}"))
        # else:
            # plt.show()
        # plt.close(fig)
        return fig, ax

    def plot_multi_tc(self, model_type, y = "metric", save = True, ylim = (-0.2, 1), figsize = (30, 19), extension = "png"):
        """Plot multiple accuracy time courses"""
        var_list = list(self.metric_global_dict.keys())
        nvar = len(var_list)
        nrows = int(np.sqrt(nvar))
        ncols = int(np.ceil(nvar / nrows))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.flatten()
        for i, var in enumerate(var_list):
            fig, axs[i] = self.plot_tc(var, y=y, save=False, figsize=figsize, ylim=ylim, extension=extension, fig=fig, ax=axs[i])
            # axs[i].plot(self.results['timepoint'], val)
            # axs[i].axhline(baseline, color='black', linestyle='--')
            # axs[i].axvline(onset, color='red', linestyle='--')
            # axs[i].set_title(f"{var} {y} global accuracy", fontweight='bold')
            # axs[i].set_xlabel("Time (ms)")
            # axs[i].set_ylabel(y)
            # axs[i].set_ylim(ylim)
            # axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)
        # for j in range(i + 1, nrows * ncols):
        #     fig.delaxes(axs[j // ncols, j % ncols])
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        if save:
            fig.savefig(os.path.join(FIGURES_DIR, "decoding", f"sub-{int(self.subject):03}_multi_accuracy-tc-{model_type}.{extension}"))
        # else:
        #     plt.show()
        # plt.close(fig)
        return fig, axs

    def plot_heatmap(self, var, y = "metric", save = True, extension = "png", fig = None, ax = None):
        """Plot the accuracy time course as heatmap"""
        if y == "metric" :
            val = self.metric_channel_dict[var]
        else:
            val = self.score_channel_dict[var]
        lim = np.max(np.abs(val))
        to_plot = val[:, self.ordered_regions_idx]
        lim_neg = 0.4 if self.classification else -lim
        im = ax.imshow(to_plot.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=lim_neg, vmax=lim)
        plt.colorbar(im, ax=ax)
        ax.axvline(onset, color='black', linestyle='--')
        ax.set_title(f"{var} {y} channel accuracy")
        ax.set_xlabel("Time (ms)")
        ax.set_xticks(ticks=arranged_timearray, labels=timearray)
        ax.set_yticks(ticks=np.arange(self.n_channels), labels=self.ordered_regions)
        if save:
            fig.savefig(os.path.join(FIGURES_DIR, f"sub-{int(self.subject):03}_{var}_accuracy-ht.{extension}"))
        return fig, ax


    def plot_multi_heatmap(self, model_type, y = "metric", save = True, figsize = (30, 19), extension = "png"):
        """Plot multiple accuracy time courses"""
        var_list = list(self.metric_channel_dict.keys())
        nvar = len(var_list)
        nrows = int(np.sqrt(nvar))
        ncols = int(np.ceil(nvar / nrows))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
        axs = axs.flatten()
        for i, var in enumerate(var_list):
            fig, axs[i] = self.plot_heatmap(var, y=y, save=False,  extension=extension, fig=fig, ax=axs[i])
            # if y == "metric" :
            #     val = self.metric_channel_dict[var]
            # else:
            #     val = self.score_channel_dict[var]
            # lim = np.max(np.abs(val))
            # to_plot = val[:, self.ordered_regions_idx]
            # lim_neg = 0.4 if self.classification else -lim
            # im = axs[i].imshow(to_plot.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=lim_neg, vmax=lim)
            # plt.colorbar(im, ax=axs[i])
            # axs[i].axvline(onset, color='black', linestyle='--')
            # axs[i].set_title(f"{var} {y} channel accuracy")
            # axs[i].set_xlabel("Time (ms)")
            # axs[i].set_xticks(ticks=arranged_timearray, labels=timearray)
            # axs[i].set_yticks(ticks=np.arange(self.n_channels), labels=self.ordered_regions)
        # for j in range(i + 1, nrows * ncols):
        #     fig.delaxes(axs[j // ncols, j % ncols])
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(FIGURES_DIR, "decoding", f"sub-{int(self.subject):03}_multi_accuracy-ht-{model_type}.{extension}"))
        # else:
        #     plt.show()
        # plt.close(fig)
        return fig, axs


