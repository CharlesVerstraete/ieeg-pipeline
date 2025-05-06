#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Class for anatomical organisation and localisation
"""

# Import libraries
from preprocessing.config import *
from preprocessing.utils.data_helper import *
from preprocessing.utils.anat_helper import *

from scipy.ndimage import binary_dilation


class Anatomy:
    def __init__(self, atlas_path, subject_num):
        """
        Initialize the Anatomy class with anatomical data and parameters.
        """
        self.subject_num = subject_num
        self.atlas_ref, self.atlas_img, self.atlas_data = get_atlas_data(atlas_path)
        self.anat_img, self.anat_data = None, None
        self.sulcus, self.sulcus_data = None, None
        self.segmentation, self.segmentation_data = None, None
        self.enlarged_segmentation = None
        self.electrode_df = None
        self.bipolar_df = None

    def load_nifti(self, file, reverse = False):
        """
        Load a NIfTI file and return the image and data.
        """
        nifti = nib.load(file)
        if reverse:
            nifti = nifti.slicer[::-1]
        return nifti, nifti.get_fdata()
    
    def load_anatomical(self, anat_file):
        """
        Load the anatomical NIfTI file
        """
        self.anat_img, self.anat_data = self.load_nifti(anat_file)

    def binarise(self, data):
        """
        Binarise the data by setting all non-zero values to 1.
        """
        data[data != 0] = 1
        return data

    def merge_sulci(self, right, left):
        """
        Merge the sulci data from left and right hemispheres.
        """
        _, sulcusL_data = self.load_nifti(left, reverse=True)
        _, sulcusR_data = self.load_nifti(right, reverse=True)
        merged_sulcus = np.zeros(self.anat_data.shape)
        if sulcusR_data is not None:
            merged_sulcus[:sulcusR_data.shape[0], -sulcusR_data.shape[1]:, -sulcusR_data.shape[2]:] += sulcusR_data
        if sulcusL_data is not None:
            merged_sulcus[:sulcusL_data.shape[0], -sulcusL_data.shape[1]:, -sulcusL_data.shape[2]:] += sulcusL_data
        self.sulcus_data = self.binarise(merged_sulcus).astype(np.uint8)
        self.sulcus = nib.Nifti1Image(self.sulcus_data, self.anat_img.affine)
    
    def create_segmented_mask(self, right, left):
        """
        Create a white matter mask by merging the left and right grey matter files.
        """
        _, segmentedL_data = self.load_nifti(left, reverse=True)
        _, segmentedR_data = self.load_nifti(right, reverse=True)
        segmentedL_data[segmentedL_data != 255] = 0
        segmentedR_data[segmentedR_data != 255] = 0
        merged_data = (segmentedL_data + segmentedR_data).astype(np.uint8)
        self.segmentation_data = self.binarise(merged_data)
        self.segmentation = nib.Nifti1Image(self.segmentation_data, self.anat_img.affine)
        self.enlarged_segmentation = self.enlarge_data(self.segmentation_data)
    
    def enlarge_data(self, data, dilatation = np.ones((3, 3, 3))):
        """
        Enlarge the data using binary dilation.
        """
        dilated_data = binary_dilation(data > 0, structure = dilatation).astype(np.uint8)
        return dilated_data

    def check_good_electrodes(self):
        """
        Check if the electrodes are in the image.
        """
        coords_mni = self.electrode_df[['x', 'y', 'z']].values
        coords_vox = coordinate_transformation(coords_mni, self.anat_img.affine)
        coords_vox = np.round(coords_vox).astype(int)
        for i in range(3):
            out_of_bounds = np.where((coords_vox[:, i] < 0) | (coords_vox[:, i] >= self.anat_data.shape[i]))[0]
            if len(out_of_bounds) > 0:
                self.electrode_df.drop(index = out_of_bounds, inplace = True)
        self.electrode_df.reset_index(drop = True, inplace = True)

    def load_electrodes(self, implant_file):
        """
        Load the electrode coordinates from the implant file.
        """
        name_path = [f for f in implant_file if "Name" in f][0]
        pos_path = [f for f in implant_file if "Pos" in f][0]
        coords = np.loadtxt(pos_path)
        self.electrode_df = pd.DataFrame(coords, columns = ['x', 'y', 'z'])
        with open(name_path) as f:
            name_elec = [line.rstrip('\n') for line in f]
        self.electrode_df["name"] = name_elec
        self.electrode_df['electrode'] = [match.group(1) for match in [re.match(r"([A-Za-z']+)([ \d]+)", name) for name in self.electrode_df['name'].values]]
        self.electrode_df["subject"] = self.subject_num
        self.check_good_electrodes()

    def get_electrodes_inmask(self, df):
        """
        Check if the electrodes are in the mask.
        """
        coords_mni = df[['x', 'y', 'z']].values
        coords_vox = coordinate_transformation(coords_mni, self.anat_img.affine)
        coords_vox = np.round(coords_vox).astype(int)
        pos = np.ravel_multi_index(coords_vox.T, self.enlarged_segmentation.shape)
        inmask = self.enlarged_segmentation.take(pos)
        idx_inmask = np.where(inmask == 1)[0]
        df["is_inmask"] = False
        df.loc[idx_inmask, "is_inmask"] = True
        return df

    def get_electrodes_parcels(self, df):
        """
        Get the parcels of the electrodes.
        """
        coords_mni = df[['x', 'y', 'z']].values
        coords_vox = coordinate_transformation(coords_mni, self.atlas_img.affine)
        coords_vox = np.round(coords_vox).astype(int)
        atlas_voxels, atlas_values = format_atlas_data(self.atlas_data)
        distances, parcels = find_nearest_neighbors(coords_vox, atlas_voxels, atlas_values)
        df["roi_n"] = parcels
        df["distances"] = distances
        df = assign_labels(df, self.atlas_ref)
        return df

    def make_bipolar_df(self):
        """
        Create the bipolar dataframe.
        """
        bipolar_rows = []
        for electrode, sub_df in self.electrode_df.groupby("electrode"):
            for i in range(len(sub_df) - 1):
                anode = sub_df.iloc[i]
                cathode = sub_df.iloc[i + 1]
                bipolar_rows.append({
                    "electrode": anode["electrode"],
                    "name": f"{anode['name']}-{cathode['name']}",
                    "anode": anode["name"],
                    "cathode": cathode["name"],
                    "x": (anode["x"] + cathode["x"]) / 2,
                    "y": (anode["y"] + cathode["y"]) / 2,
                    "z": (anode["z"] + cathode["z"]) / 2
                })
        self.bipolar_df = pd.DataFrame(bipolar_rows)

    def add_bipolar_contacts_info(self):
        """Add information about individual contacts to the bipolar dataframe"""
        name_to_info = {row["name"]: row for _, row in self.electrode_df.iterrows()}
        for col in ["roi_n", "roi_name", "region", "is_inmask", "lobe", "distances"]:
            self.bipolar_df[f"anode_{col}"] = self.bipolar_df["anode"].map(
                lambda x: name_to_info.get(x, {}).get(col, None))
            self.bipolar_df[f"cathode_{col}"] = self.bipolar_df["cathode"].map(
                lambda x: name_to_info.get(x, {}).get(col, None))

    def process_electrodes(self, df_type = 'unipolar'):
        """
        Process the electrodes and create the bipolar dataframe.
        """
        if df_type == 'unipolar':
            self.electrode_df = self.get_electrodes_inmask(self.electrode_df)
            self.electrode_df = self.get_electrodes_parcels(self.electrode_df)
        elif df_type == 'bipolar':
            self.bipolar_df = self.get_electrodes_inmask(self.bipolar_df)
            self.bipolar_df = self.get_electrodes_parcels(self.bipolar_df)
        
    def load_electrodes_from_grenoble(self, file):
        """
        Load the electrodes from the Grenoble implant file.
        """
        df = get_grenoble_df(file)
        self.electrode_df = format_grenoble_df(df)
        self.electrode_df["subject"] = self.subject_num
        self.make_bipolar_df()
        self.electrode_df = self.get_electrodes_parcels(self.electrode_df)
        self.bipolar_df = self.get_electrodes_parcels(self.bipolar_df)
        self.add_bipolar_contacts_info()
        self.bipolar_df["is_inmask"] = (self.bipolar_df["cathode_is_inmask"] | self.bipolar_df["anode_is_inmask"])
    
    def load_electrodes_from_bordeaux(self, file):
        """
        Load the electrodes from the Bordeaux implant file.
        """
        self.electrode_df = get_bordeaux_df(file)
        self.electrode_df["subject"] = self.subject_num
        self.make_bipolar_df()
        self.electrode_df = self.get_electrodes_parcels(self.electrode_df)
        self.bipolar_df = self.get_electrodes_parcels(self.bipolar_df)
        self.add_bipolar_contacts_info()
        self.bipolar_df["is_inmask"] = (self.bipolar_df["cathode_is_inmask"] | self.bipolar_df["anode_is_inmask"])

    def save_electrodes(self, path, df_type = 'unipolar'):
        """
        Save the electrodes dataframe to a CSV file.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        if df_type == 'unipolar':
            save_path = os.path.join(path, f"sub-{self.subject_num:03}_electrodes-unipolar.csv")
            self.electrode_df.to_csv(save_path, index = False)
        elif df_type == 'bipolar':
            save_path = os.path.join(path, f"sub-{self.subject_num:03}_electrodes-bipolar.csv")
            self.bipolar_df.to_csv(save_path, index = False)

    def save_files(self, path):
        """
        Save the anatomical files to the specified path.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, f"sub-{self.subject_num:03}_T1w.nii")
        nib.save(self.anat_img, save_path)
        save_path = os.path.join(path, f"sub-{self.subject_num:03}_sulcus.nii")
        nib.save(self.sulcus, save_path)
        save_path = os.path.join(path, f"sub-{self.subject_num:03}_dseg.nii")
        nib.save(self.segmentation, save_path)