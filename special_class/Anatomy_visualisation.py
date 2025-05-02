#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Child class for anatomical visualisation
"""

# Import libraries
from preprocessing.config import *
from preprocessing.utils.data_helper import *
from preprocessing.utils.anat_helper import *
from special_class.Anatomy import *

from matplotlib.widgets import Slider
from matplotlib.widgets import Button



class Anatomy_visualisation(Anatomy):
    def __init__(self, anat_file, atlas_path, color_map, subject_num, figsize=(21, 12)):
        """
        Initialize the AnatomyVisualisation class with anatomical data and parameters.
        Parameters:
        - anat_file: Path to the anatomical NIfTI file.
        """
        super().__init__(anat_file, atlas_path, subject_num)
        self.color_map = color_map
        self.init_fig(figsize)
        self.init_sliders()

    def init_fig(self, figsize):
        """
        Initialize the figure for anatomical visualisation.
        """
        self.fig, self.axs = plt.subplots(1, 3, figsize=figsize)
        self.axs[0].set_title("Sagittal view")
        self.axs[1].set_title("Coronal view")
        self.axs[2].set_title("Axial view")
        self.axs[0].axis("off")
        self.axs[1].axis("off")
        self.axs[2].axis("off")
        plt.subplots_adjust(left=0.01, right=0.93, bottom=0.12, top=0.99, hspace=0.01, wspace=0.01)

    def init_sliders(self):
        """
        Initialize sliders for slice navigation.
        """
        axcolor = "lightgoldenrodyellow"
        shape = self.anat_data.shape
        self.sliders = []
        for i, (orientation, max_slice) in enumerate(zip(['x', 'y', 'z'], shape)):
            ax_slider = plt.axes([0.05, 0.1 - i * 0.04, 0.9, 0.02], facecolor=axcolor)
            slider = Slider(ax_slider, f"{orientation}-slice", 0, max_slice - 1, valinit=max_slice // 2, valstep=1)
            slider.on_changed(self.navigate)
            self.sliders.append(slider)


    def plot_masked_slice(self, ax, slice_data):
        """
        Show a masked slice of the anatomical data.
        """
        masked_slice = np.ma.masked_where(slice_data == 0, slice_data)
        ax.imshow(masked_slice.T, origin="lower", cmap=self.color_map)

    def navigate(self, event=None):
        """
        Navigate through slices of the anatomical data with overlays.
        """
        slice_indices = [int(slider.val) for slider in self.sliders]
        mni_indices = coordinate_transformation(np.array(slice_indices), np.linalg.inv(self.anat.affine))
        slice_planes = [
        self.anat_data[slice_indices[0], :, :],  
        self.anat_data[: , slice_indices[1], :],  
        self.anat_data[: , :, slice_indices[2]], 
        ]
        for i, ax in enumerate(self.axs):
            self.plot_masked_slice(ax, slice_planes[i])
            ax.set_title(f"Voxel {slice_indices[i]} - MNI {mni_indices[i]}")
            ax.axis("off")

    def plot(self):
        """
        Plot the anatomical data with overlays.
        """
        self.navigate()
        plt.show()

    