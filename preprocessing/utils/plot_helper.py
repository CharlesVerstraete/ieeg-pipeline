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

