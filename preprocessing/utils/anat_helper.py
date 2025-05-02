#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Anatomical organisation and localisation helper functions
"""

from preprocessing.config import *
from preprocessing.utils.data_helper import *
import numpy as np
import pandas as pd
import os
import re
from scipy.spatial import cKDTree, KDTree
from nilearn import datasets, surface, plotting
from matplotlib.colors import ListedColormap


def get_grenoble_df(path):
    """
    Get the electrodes coordinates from the Grenoble database
    """
    if os.path.splitext(path)[1] == ".csv" :
        df = pd.read_csv(path, sep=";")
    elif os.path.splitext(path)[1] == ".xlsx" :
        df = pd.read_excel(path)
    cut_idx = np.where(df["Contacts Positions"].isna())[0][0]
    df = df.iloc[:cut_idx]
    df.columns = df.loc[1].values
    df.drop(index=[0, 1], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_bordeaux_df(path):
    """
    Get the electrodes coordinates from the Bordeaux database
    """
    df = pd.read_csv(path, sep="\t")
    df.drop(index=len(df)-1, inplace=True)
    df["is_inmask"] = df["matter"] == "grey"
    df.rename(columns={"Electrode": "electrode"}, inplace=True)
    df["name"] = [f"{elec}{n}" for elec, n in df[["electrode", "Contact"]].values]
    df.reset_index(drop=True)
    return df[["electrode", "name", "x", "y", "z", "is_inmask"]] 


def format_grenoble_df(df):
    """
    Format the Grenoble dataframe
    """
    new_df = pd.DataFrame()
    coords = df["MNI"].str.strip('[]').str.split(',', expand=True)
    for i, col in enumerate(["x", "y", "z"]):
        new_df[col] = coords[i].astype(float)
    new_df["name"] = df["contact"]
    new_df["electrode"] = [match.group(1) for match in [re.match(r"([A-Za-z']+)([ \d]+)", name) for name in new_df['name'].values]]
    new_df["is_inmask"] = df["GreyWhite"] == "GreyMatter"
    return new_df

def coordinate_transformation(coords, affine):
    """
    Transform coordinates to another space using the affine transformation matrix
    """
    coords_array = np.asarray(coords).astype(float)
    if coords_array.ndim == 1:
        out_coords = np.dot(np.append(coords_array, 1), np.linalg.inv(affine).T)[:3]
    else:
        homogeneous_coords = np.hstack((coords_array, np.ones((len(coords_array), 1))))
        out_coords = np.dot(homogeneous_coords, np.linalg.inv(affine).T)[:, :3]
    return out_coords

def format_atlas_data(atlas_data) :
    """
    Format the atlas data to get the valid voxels and their values
    """
    x, y, z = np.indices(atlas_data.shape)
    voxel_coords = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    values = atlas_data.flatten()
    valid = ~np.isnan(values) & (values > 0)
    valid_voxels = voxel_coords[valid]
    valid_values = values[valid]
    return valid_voxels, valid_values

def find_nearest_neighbors(coord, voxels, values):
    """
    Find the nearest neighbors of the given coordinates in the atlas space
    """
    voxel_tree = cKDTree(voxels)
    distances, indices = voxel_tree.query(coord)
    parcels = values[indices].astype(int)
    return distances, parcels

def assign_labels(df, atlas_ref):
    """
    Assign labels to the dataframe based on the atlas reference
    """
    df["roi_name"] = [atlas_ref["ROI_glasser_1"].values[x-1] if x != 0 else " " for x in df["roi_n"].values]
    df["region"] = [atlas_ref["ROI_glasser_2"].values[x-1] if x != 0 else " " for x in df["roi_n"].values]
    df["lobe"] = [atlas_ref["Lobe"].values[x-1] if x != 0 else " " for x in df["roi_n"].values]
    return df


def get_atlas_data(atlas_path):
    """
    Get the atlas data
    """
    atlas_ref_path = os.path.join(atlas_path, "HCP-MMP1_labels.csv")
    atlas_ref = pd.read_csv(atlas_ref_path, delimiter=';')
    atlas_img_path = os.path.join(atlas_path, "HCP-MMP1_resampled.nii")
    atlas_img = nib.load(atlas_img_path)
    atlas_data = atlas_img.get_fdata()
    return atlas_ref, atlas_img, atlas_data

def surface_resample(surf_to_resample, orig_coord, target_coord):
    """
    Resample the surface data to match the target coordinates
    """
    tree = KDTree(orig_coord)
    _, indices = tree.query(target_coord)
    return surf_to_resample[indices]

def create_customcmap(vmin, vmax, center, cmap_orig):
    """
    Create a custom colormap
    """
    base_cmap = plt.get_cmap(cmap_orig)
    x = np.linspace(0, 1, 1024)
    center_norm = (center - vmin) / (vmax - vmin)
    idx = np.where(
        x < center_norm,
        0.5 * (x / center_norm) ** 2,
        1 - 0.5 * ((1 - x) / (1 - center_norm)) ** 2
    )
    return ListedColormap(base_cmap(idx))


