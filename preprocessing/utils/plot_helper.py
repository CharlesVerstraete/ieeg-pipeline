#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Signal processing helper functions
"""
# Import libraries

from preprocessing.config import *
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('Qt5Agg')
# plt.style.use('seaborn-v0_8-poster') 





def plot_denoised(f, pxx, pxx_clean, path = None) :
    fig, axs = plt.subplots(3, 1, figsize=(21, 12))
    axs[0].plot(f, 10 * np.log10(pxx).T, color = "black", lw =0.5, alpha = 0.5)
    axs[0].set_title("PSD original")
    axs[1].plot(f, 10 * np.log10(pxx_clean).T,  color = "blue",  lw =0.5, alpha = 0.5)
    axs[1].set_title("PSD cleaned")
    axs[2].plot(f, ((10 * np.log10(pxx)) - (10 * np.log10(pxx_clean))).T, color = "firebrick",  lw =0.5, alpha = 0.5)
    axs[2].set_title("Difference")
    plt.tight_layout()
    if path is not None :
        fig.savefig(path, transparent=True, format='pdf',  bbox_inches='tight')
        plt.close(fig)
    else :
        plt.show()


def plot_indiv_lineplot(df, x_var, y_var, xlim = None, chance = None, ylim = (0, 1), save_path=None):
    """
    Plot individual performance
    """
    subjects = sorted(df["subject"].unique())
    rows = int(np.ceil(len(subjects) / 6))
    fig, axs = plt.subplots(rows, 6, figsize=(21, 12), sharex=True, sharey=True)
    axs = axs.flatten()
    for i, subject in enumerate(subjects):
        s_df = df[(df["subject"] == subject)]
        if xlim is not None:
            s_df = s_df[(s_df[x_var] < xlim)]
        sns.lineplot(data=s_df, x=x_var, y=y_var, ax=axs[i], color="black")
        axs[i].set_title(f"Subject {subject}")
        if chance is not None:
            axs[i].axhline(y=chance, color="red", linestyle="--")
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    plt.ylim(ylim)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, transparent=True, format='pdf',  bbox_inches='tight')
    plt.show()
    

def plot_indiv_histplot(df, vars, save_path=None):
    """
    Plot individual distribution
    """
    count = df.groupby(vars).size().reset_index(name="count")
    subjects = sorted(df["subject"].unique())
    rows = int(np.ceil(len(subjects) / 6))
    fig, axs = plt.subplots(rows, 6, figsize=(21, 12), sharex=True, sharey=True)
    axs = axs.flatten()
    for i, subject in enumerate(subjects):
        s_df = count[(count["subject"] == subject)]
        sns.histplot(data=s_df, x="count", ax=axs[i], bins=10, kde=True)
        axs[i].set_title(f"Subject {subject}")
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, transparent=True, format='pdf',  bbox_inches='tight')
    plt.show()

def plot_boxplot(df, x_var, y_var, box_alpha=0.5, palette = "husl", hue_var=None, save_path=None):
    """
    Plot boxplot
    """
    df[x_var] = df[x_var].astype(str)
    sns.boxplot(
        data=df, x=x_var, y=y_var, hue=hue_var, palette=palette, showfliers=False,
        boxprops={"alpha": box_alpha}, whiskerprops={"alpha": box_alpha},
        capprops={"alpha": box_alpha}, medianprops={"alpha": box_alpha}
    )
    sns.stripplot(data=df, x=x_var, y=y_var, hue=hue_var, palette=palette, alpha=0.8)
    if save_path is not None:
        plt.savefig(save_path, transparent=True, format='pdf',  bbox_inches='tight')
    plt.show()





def plot_timecourse_summary(summary, x, ymean="mean", yerr="sem", hue=None, keys=None,
                            palette=None, ax=None, ribbon_alpha=0.2):
    """
    Trace un time-course (ligne + bande sem) à partir d'un summary (mean/sem).
    Gère hue=None (sans groupe) ou une liste de keys spécifique.
    """
    if hue is None:
        d = summary.sort_values(x)
        ax.plot(d[x], d[ymean], color="black")
        ax.fill_between(d[x], d[ymean] - d[yerr], d[ymean] + d[yerr], alpha=ribbon_alpha, color="black")
    else:
        levels = keys if keys is not None else list(summary[hue].dropna().unique())
        for k in levels:
            d = summary[summary[hue] == k].sort_values(x)
            color = (palette.get(k) if isinstance(palette, dict) else None) or "C0"
            ax.plot(d[x], d[ymean], color=color)
            ax.fill_between(d[x], d[ymean] - d[yerr], d[ymean] + d[yerr], alpha=ribbon_alpha, color=color)
    return ax


def plot_around_switch(summary_before, summary_after, x_pre, x_post, avr_line = None,
                       hue_pre=None, hue_post=None, keys=None, palette=None, ylim=None,
                        xticks=None, xlabel="Stimulus presentation", ylabel="Proportion correct",
                       save_path=None):
    """
    Superpose pré (x_pre) et post (x_post) sur le même axe, réutilisable pour tout time-course.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_timecourse_summary(summary_after, x=x_post, hue=hue_post, keys=keys, palette=palette,
                            ax=ax, ribbon_alpha=0.2)
    plot_timecourse_summary(summary_before, x=x_pre, hue=hue_pre, keys=keys, palette=palette,
                            ax=ax, ribbon_alpha=0.2)
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)
    if xticks is not None:
        ax.set_xticks(xticks)
    if ylim is not None:
        ax.set_ylim(ylim)
    if avr_line is not None:
        ax.axhline(avr_line, color="black", lw=2, ls="--", alpha=0.8)
    ax.axvline(0, color="red", lw=3, alpha=0.8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=True, format="pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax
