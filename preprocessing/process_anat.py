#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Build the anatomical data
"""

# Import libraries

from preprocessing.config import *
from preprocessing.utils.data_helper import *
from preprocessing.utils.anat_helper import *

atlas_path = "/Users/charles.verstraete/Documents/w3_iEEG/analysis/data/anatomical_atlas"
atlas_ref, atlas_img, atlas_data = get_atlas_data(atlas_path)
atlas_voxels, atlas_values = format_atlas_data(atlas_data)

gre_path_list = ["/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/3/Localisation/Gre_2024_GUIa.xlsx",
                 "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/23/MRI/Par_2024_XX7.csv",
                 "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/28/MRI/Par_2025_GHAa.csv"]

full = pd.DataFrame()

bdx_path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/25/BRA_THA/MRIcentered.txt"
df = get_bordeaux_df(bdx_path)


coords = coordinate_transformation(df[["x", "y", "z"]].values.astype(float), atlas_img.affine).astype(int)
distances, parcels = find_nearest_neighbors(coords, atlas_voxels, atlas_values)
df["roi_n"] = parcels
df["distances"] = distances
df = assign_labels(df, atlas_ref)



