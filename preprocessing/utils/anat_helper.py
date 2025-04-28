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
from scipy.spatial import cKDTree


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
    return df.reset_index(drop=True)


def format_grenoble_df(df, atlas_ref):
    """
    Format the Grenoble dataframe
    """
    new_df = pd.DataFrame()
    new_df["x"] = [x.replace("[", "").replace("]", "").split(",")[0] for x in df["MNI"].values]
    new_df["y"] = [x.replace("[", "").replace("]", "").split(",")[1] for x in df["MNI"].values]
    new_df["z"] = [x.replace("[", "").replace("]", "").split(",")[2] for x in df["MNI"].values]
    new_df["name"] = df["contact"]
    new_df["electrode"] = [match.group(1) for match in [re.match(r"([A-Za-z']+)([ \d]+)", name) for name in new_df['name'].values]]
    new_df["is_inmask"] = df["GreyWhite"] == "GreyMatter"
    return new_df

def coordinate_transformation(coords, affine):
    """
    Transform coordinates to another space using the affine transformation matrix
    """
    coords = np.hstack((coords, np.ones((len(coords), 1))))
    return np.dot(coords, np.linalg.inv(affine).T)[:, :3]

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
    atlas_img_path = os.path.join(atlas_path, "HCP-MMP1.nii")
    atlas_img = nib.load(atlas_img_path)
    atlas_data = atlas_img.get_fdata()
    return atlas_ref, atlas_img, atlas_data









# df = df[(df["lobe"] == "Fr") | (df["lobe"] == "Ins")]
# df = df[df["is_inmask"]]
# full = pd.concat([full, df], ignore_index=True)

# path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/23/MRI/Par_2024_XX7.csv"
# df = get_grenoble_df(path)
# df = format_grenoble_df(df, atlas_ref)
# coords = coordinate_transformation(df[["x", "y", "z"]].values.astype(float), atlas_img.affine).astype(int)
# distances, parcels = find_nearest_neighbors(coords, atlas_voxels, atlas_values)
# df["roi_n"] = parcels
# #df["distances"] = distances
# df = assign_labels(df, atlas_ref)
# df = df[(df["lobe"] == "Fr") | (df["lobe"] == "Ins")]
# df = df[df["is_inmask"]]
# full = pd.concat([full, df], ignore_index=True)

# path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/28/MRI/Par_2025_GHAa.csv"
# df = get_grenoble_df(path)
# df = format_grenoble_df(df, atlas_ref)
# coords = coordinate_transformation(df[["x", "y", "z"]].values.astype(float), atlas_img.affine).astype(int)
# distances, parcels = find_nearest_neighbors(coords, atlas_voxels, atlas_values)
# df["roi_n"] = parcels
# #df["distances"] = distances
# df = assign_labels(df, atlas_ref)
# df = df[(df["lobe"] == "Fr") | (df["lobe"] == "Ins")]
# df = df[df["is_inmask"]]
# col_tokeep = df.columns 
# full = pd.concat([full, df], ignore_index=True)


# initial_v1path = "/Users/charles.verstraete/Documents/w3_iEEG/analysis/data"
# full = pd.DataFrame()
# for subject in ["02", "04", "05", "08", "09", "12", "14", "16", "18", "19", "20"]:
#     df = pd.read_csv(initial_v1path + f"/preprocessed/electrode_position/subject_{subject}_electrodes.csv")
#     df = df[(df["lobe"] == "Fr") | (df["lobe"] == "Ins")]
#     df = df[df["is_inmask"]]
#     full = pd.concat([full, df], ignore_index=True)
# full["roi_n"] = full["roi"]
# full = full[col_tokeep]



# count = full.groupby(["roi_n"]).size().reset_index(name="count")


# from nilearn import plotting
# from nilearn import datasets, surface, plotting


# atlas = surface.load_surf_data(initial_v1path + "/anatomical_atlas/rh.HCP-MMP1.annot")

# fsaverage_orig = datasets.fetch_surf_fsaverage("fsaverage")
# fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
# coords_orig, _ = surface.load_surf_data(fsaverage_orig[f"pial_right"])
# coords_target, _ = surface.load_surf_mesh(fsaverage[f"pial_right"])
# resampled_atlas = surface_resample(atlas, coords_orig, coords_target)

# atlas_filtered = np.zeros(resampled_atlas.shape)
# roi_refdf = pd.DataFrame({'roi' : full["region"].unique(), "idx" : 0})
# for (i, (idx, row)) in  enumerate(count.iterrows()) :
#     atlas_filtered[resampled_atlas == row["roi_n"]] = row["count"]

# atlas_filtered[atlas_filtered == 0] = np.nan

# #for view in ["lateral", "medial", "dorsal", "ventral", "anterior", "posterior"]:
# view = "lateral"
# plotting.plot_surf_stat_map(
#     surf_mesh=fsaverage["pial_right"],
#     stat_map=atlas_filtered,
#     bg_map=fsaverage["sulc_right"],
#     hemi='right',
#     view=view,
#     cmap=custom_cmap,
#     colorbar=True,
#     bg_on_data=True
# )
# plt.show()
# plt.savefig(f"electrodes_density_{view}.pdf", transparent=True, format='pdf',  bbox_inches='tight')
# plt.close()

# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="classic",
#     background="white",
#     size=(800, 600),
#     alpha=0.8,
# )


# brain.add_data(
#     atlas_filtered,
#     colormap=custom_cmap, 
#     alpha=0.8, 
#     colorbar=True,
#     fmin=0,
#     fmax=count["count"].max(),
#     colorbar_kwargs={
#         "n_labels": 6,
#         "fmt": "%.f",
#     },
#     verbose=True
# )


# brain.add_foci(full[["x", "y", "z"]].values.astype(float), scale_factor=0.2, color="black")









# plotting.plot_surf_stat_map(
#     surf_mesh=fsaverage["pial_right"],
#     stat_map=atlas_filtered,
#     bg_map=fsaverage["sulc_right"],
#     hemi='right',
#     view='medial',
#     cmap=custom_cmap,
#     colorbar=False,
#     axes=axes[0,1],
#     bg_on_data=True
# )

# plotting.plot_surf_stat_map(
#     surf_mesh=fsaverage["pial_right"],
#     stat_map=atlas_filtered,
#     bg_map=fsaverage["sulc_right"],
#     hemi='right',
#     view='ventral',
#     cmap=custom_cmap,
#     colorbar=False,
#     axes=axes[1,0],
#     bg_on_data=True
# )

# plotting.plot_surf_stat_map(
#     surf_mesh=fsaverage["pial_right"],
#     stat_map=atlas_filtered,
#     bg_map=fsaverage["sulc_right"],
#     hemi='right',
#     view='dorsal',
#     cmap=custom_cmap,
#     colorbar=True,
#     axes=axes[1,1],
#     bg_on_data=True
# )

# #plt.tight_layout()
# plt.show()

# # Create a vertex-level data array (initialize with zeros)
# vtx_data = np.zeros(163842)  # Number of vertices in fsaverage's right hemisphere

# # Get a list of all vertex indices for each label region and assign them values
# for idx, row in count.iterrows():
#     count_value = row["count"]
#     label_name = row["roi_name"]
#     label_idx = np.where(np.array(labels_name) == label_name)[0]
#     if len(label_idx) > 0:
#         label = labels[label_idx[0]]
#         vtx_data[label.vertices] = count_value

# vtx_data[vtx_data == 0] = np.nan  # Set non-label vertices to NaN
# # Now add the data with proper vertex mapping

# vtx_data


# surface.load_surf_data(fsaverage["pial_right"])


# plotting.plot_surf(
#     surf_mesh = vtx_data,
# )
# plt.show()



# from scipy.spatial import KDTree


# def surface_resample(surf_to_resample, orig_coord, target_coord):
#     tree = KDTree(orig_coord)
#     _, indices = tree.query(target_coord)
#     return surf_to_resample[indices]

# fsaverage_orig = datasets.fetch_surf_fsaverage("fsaverage")
# fsaverage = datasets.fetch_surf_fsaverage("fsaverage4")
# coords_orig, _ = surface.load_surf_data(fsaverage_orig[f"pial_right"])
# coords_target, _ = surface.load_surf_mesh(fsaverage[f"pial_right"])
# resampled_atlas = surface_resample(vtx_data, coords_orig, coords_target)









# from matplotlib.colors import hsv_to_rgb, ListedColormap


# vmin = count["count"].min()
# vmax = count["count"].max()

# # vmax = 20
# # vmin = 5

# base_cmap = plt.get_cmap("jet")

# # Create array of values to sample colors from
# x = np.linspace(0, 1, 256)

# # Create a custom normalization centered at the specified value
# center_norm = (10 - vmin) / (vmax - vmin)

# # Create gamma-like function to emphasize values around center
# idx = np.where(
#     x < center_norm,
#     0.5 * (x / center_norm) ** 2,
#     1 - 0.5 * ((1 - x) / (1 - center_norm)) ** 2
# )
# custom_cmap = ListedColormap(base_cmap(idx))



# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np



# max_points = 50000
# if len(valid_voxels) > max_points:
#     indices = np.random.choice(len(valid_voxels), max_points, replace=False)
#     sample_points = valid_voxels[indices]
#     sample_values = valid_values[indices]
# else:
#     sample_points = valid_voxels
#     sample_values = valid_values





# # Create 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Color points by their atlas values
# scatter = ax.scatter(
#     sample_points[:, 0], 
#     sample_points[:, 1], 
#     sample_points[:, 2],
#     c=sample_values,
#     cmap='tab20',
#     alpha=0.8,
#     s=5
# )



# # Add a colorbar
# cbar = plt.colorbar(scatter)
# cbar.set_label('Atlas Region ID')

# # Set labels and title
# ax.set_xlabel('X (voxel)')
# ax.set_ylabel('Y (voxel)')
# ax.set_zlabel('Z (voxel)')
# ax.set_title('Valid Voxels in Atlas Space')

# plt.tight_layout()
# plt.show()




# # Create a mask of valid voxels
# mask = np.zeros_like(atlas_data, dtype=bool)
# for x, y, z in valid_voxels:
#     mask[int(x), int(y), int(z)] = True

# # Show slices through the atlas
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# axes = axes.flatten()

# # Get the center of mass for valid voxels
# center = np.mean(valid_voxels, axis=0).astype(int)


# # Axial slices (z-planes)
# axes[0].imshow(atlas_data[:, :, center[2]].T, cmap='tab20', origin='lower')
# axes[0].set_title(f'Axial Slice (z={center[2]})')
# axes[0].set_xlabel('X')
# axes[0].set_ylabel('Y')

# # Coronal slices (y-planes)
# axes[1].imshow(atlas_data[:, center[1], :].T, cmap='tab20', origin='lower')  
# axes[1].set_title(f'Coronal Slice (y={center[1]})')
# axes[1].set_xlabel('X')
# axes[1].set_ylabel('Z')

# # Sagittal slices (x-planes)
# axes[2].imshow(atlas_data[center[0], :, :].T, cmap='tab20', origin='lower')
# axes[2].set_title(f'Sagittal Slice (x={center[0]})')
# axes[2].set_xlabel('Y')
# axes[2].set_ylabel('Z')

# # Show mask of valid voxels for each slice
# axes[3].imshow(mask[:, :, center[2]].T, cmap='Reds', origin='lower')
# axes[3].set_title('Valid Voxels (Axial)')

# axes[4].imshow(mask[:, center[1], :].T, cmap='Reds', origin='lower')
# axes[4].set_title('Valid Voxels (Coronal)')

# axes[5].imshow(mask[center[0], :, :].T, cmap='Reds', origin='lower')
# axes[5].set_title('Valid Voxels (Sagittal)')

# plt.tight_layout()
# plt.show()


# test_points = vox_coords[-3:]

# # Find k nearest neighbors
# k = 100
# fig = plt.figure(figsize=(15, 5))

# for i, test_point in enumerate(test_points):
#     distances, indices = voxel_tree.query(test_point, k=k)
#     nearest_points = valid_voxels[indices]
    
#     # Plot the test point and its neighbors
#     ax = fig.add_subplot(1, 3, i+1, projection='3d')
    
#     # Plot the test point
#     ax.scatter([test_point[0]], [test_point[1]], [test_point[2]], 
#                color='red', s=100, marker='*', label='Query Point')
    
#     # Plot the k nearest neighbors
#     for j in range(k):
#         value = valid_values[indices[j]]
#         color = custom_cmap(value)
#         ax.scatter(nearest_points[j, 0], nearest_points[j, 1], nearest_points[j, 2], 
#                    color=color, s=10, alpha=0.5)
#     #ax.scatter(nearest_points[:, 0], nearest_points[:, 1], nearest_points[:, 2], s=50, alpha=0.6, label='Nearest Neighbors', cmap = "hsv")
    
#     # Set axis ranges to focus on the points
#     margin = 20
#     ax.set_xlim(test_point[0] - margin, test_point[0] + margin)
#     ax.set_ylim(test_point[1] - margin, test_point[1] + margin)
#     ax.set_zlim(test_point[2] - margin, test_point[2] + margin)
    
#     ax.set_title(f'Query {i+1}: {k} Nearest Neighbors')
#     ax.legend()

# plt.tight_layout()
# plt.show()

# custom_cmap = plt.get_cmap('hsv', 180)  # Create a colormap with 20 discrete colors


# df
# atlas_data = atlas_img.get_fdata()
# affine = atlas_img.affine

# # Créer une liste des coordonnées de voxels valides et leurs valeurs
# x, y, z = np.indices(atlas_data.shape)
# voxel_coords = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
# values = atlas_data.flatten()

# # Ne garder que les voxels avec des étiquettes valides (non-NaN, non-zéro)
# valid = ~np.isnan(values) & (values > 0)
# valid_voxels = voxel_coords[valid]
# valid_values = values[valid]

# # Construire un KDTree avec les voxels valides
# voxel_tree = cKDTree(valid_voxels)
# electrode_coords = df[["x", "y", "z"]].values.astype(float)

# homogeneous_coords = np.hstack((electrode_coords, np.ones((len(electrode_coords), 1))))
# transformed_coords = np.dot(homogeneous_coords, np.linalg.inv(affine).T)
# vox_coords = np.round(transformed_coords[:, :3]).astype(int)
# vox_coords

# distances, indices = voxel_tree.query(vox_coords)
# parcels = valid_values[indices].astype(int)
















# subject = 12
# implant_path = os.path.join(FIRST_ANALYSIS__DATA_DIR, "raw",f"subject_{subject}", "anatomical", "implantation")
# implant_files = get_fileslist(implant_path, '.txt')
# name_file = [f for f in implant_files if "Name" in f][0]
# pos_file = [f for f in implant_files if "Pos" in f][0]
# new_posfile = os.path.join(implant_path, "sub12_Pos2.txt")

# coords = np.loadtxt(pos_file)
# df = pd.DataFrame(coords, columns = ['x', 'y', 'z'])
# with open(name_file) as f:
#     name_elec = [line.rstrip('\n') for line in f]
# df["name"] = name_elec
# df['electrode'] = [match.group(1) for match in [re.match(r"([A-Za-z']+)([ \d]+)", name) for name in df['name'].values]]

# anat_path = os.path.join(FIRST_ANALYSIS__DATA_DIR, "raw",f"subject_{subject}", "anatomical")
# anat_file = get_fileslist(anat_path, '.nii')[0]
# anat_img = nib.load(anat_file)
# anat_data = anat_img.get_fdata()
# affine = anat_img.affine

# ct_path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/12/sub12/ct/CTpost_2024-4-2/sub12-CTpost_2024-4-2.nii"
# ct_img = nib.load(ct_path)  
# ct_data = ct_img.get_fdata()
# ct_affine = ct_img.affine

# fig, axs = plt.subplots(3, 4, figsize=(10, 10))
# for i, elec in enumerate(df["electrode"].unique()) : 
#     ax = axs.flatten()[i]
#     coords = df[df["electrode"] == elec][["x", "y_test", "z_test"]].values
#     coord_vox = np.array([np.linalg.inv(ct_affine).dot(np.append(coord, 1))[:3].astype(int) for coord in coords])
#     fixed = np.mean(coord_vox[:, 1]).astype(int)
#     ax.imshow(anat_data[:, fixed, :].T, cmap="gray", origin="lower")
#     ax.imshow(ct_data[:, fixed, :].T, cmap="turbo", origin="lower", alpha=0.3)
#     ax.scatter(coord_vox[:, 0], coord_vox[:, 2], c="red", s=20, alpha = 0.3)
# plt.tight_layout()
# plt.show()

# coords = df[["x", "y", "z"]].values
# coords[:, 2] += 20
# coords[:, 1] -= 17
# df["y_test"] = coords[:, 1]
# df["z_test"] = coords[:, 2]
# coords = df[["x", "y_test", "z_test"]].values
# coord_vox = np.array([np.linalg.inv(ct_affine).dot(np.append(coord, 1))[:3] for coord in coords])
# coord_vox_anat = np.array([affine.dot(np.append(coord, 1))[:3] for coord in coord_vox])


# np.savetxt(new_posfile, coord_vox_anat)







# coords_reoriented = np.array([affine.dot(np.append(coord, 1))[:3].astype(int) for coord in coords])
# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="low_contrast",
#     background="white",
#     size=(800, 600),
#     alpha=0.8,
# )
# brain.add_foci(coord_vox_anat, scale_factor=0.2, color="red")
































# path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/3/Localisation/Gre_2024_GUIa.xlsx"
# df = pd.read_excel(path, nrows=223)
# df.columns = df.loc[1].values
# df.drop(index=[0, 1], inplace=True)
# df.reset_index(drop=True, inplace=True)

# coords = np.array([x.replace("[", "").replace("]", "").split(",") for x in df["MNI"].values])
# coords = coords.astype(float)

# df["MNI-HCP-MMP1"].value


# labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "rh")
# label_data = df["MNI-HCP-MMP1"].unique()
# label_data = label_data[~pd.isna(label_data)]

# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="low_contrast",
#     background="white",
#     size=(800, 600),
#     alpha=0.8,
# )

# for label in labels:
#     name = label.name.split("-")[0].replace("ROI", "")[:-1]
#     name = name.replace("R", "L")
#     if name in label_data:
#         brain.add_label(label, color="blue", alpha=0.5)  
# for i, coord in enumerate(coords):
#     brain.add_foci(coord.astype(int), scale_factor=0.2, color="red")



# brain.close()







# path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/23/MRI/Par_2024_XX7.csv"
# df = pd.read_csv(path, sep=";")
# df.columns = df.loc[1].values
# df.drop(index=[0, 1], inplace=True)
# df.reset_index(drop=True, inplace=True)
# df = df[:184].copy()
# coords = np.array([x.replace("[", "").replace("]", "").split(",") for x in df["MNI"].values])
# coords = coords.astype(float)

# label_data = df["MNI-HCP-MMP1"].unique()
# label_data = label_data[~pd.isna(label_data)]

# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="low_contrast",
#     background="white",
#     size=(800, 600),
#     alpha=0.8,
# )

# for label in labels:
#     name = label.name.split("-")[0].replace("ROI", "")[:-1]
#     name = name.replace("R", "L")
#     if name in label_data:
#         brain.add_label(label, color="blue", alpha=0.5)  
# for i, coord in enumerate(coords):
#     brain.add_foci(coord.astype(int), scale_factor=0.2, color="red")


# path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/28/MRI/Par_2025_GHAa.csv"
# df = pd.read_csv(path, sep=";")
# df.columns = df.loc[1].values
# df.drop(index=[0, 1], inplace=True)
# df.reset_index(drop=True, inplace=True)
# df = df[:215].copy()
# coords = np.array([x.replace("[", "").replace("]", "").split(",") for x in df["MNI"].values])
# coords = coords.astype(float)

# atlas_ref = pd.read_csv("/Users/charles.verstraete/Documents/w3_iEEG/analysis/data/anatomical_atlas/HCP-MMP1_labels.csv", delimiter=';')

# atlas_ref

# full = pd.DataFrame()

# path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/3/Localisation/Gre_2024_GUIa.xlsx"
# df = pd.read_excel(path, nrows=225)
# df.columns = df.loc[1].values
# df.drop(index=[0, 1], inplace=True)
# df.reset_index(drop=True, inplace=True)

# new_df = pd.DataFrame()
# new_df["x"] = [x.replace("[", "").replace("]", "").split(",")[0] for x in df["MNI"].values]
# new_df["y"] = [x.replace("[", "").replace("]", "").split(",")[1] for x in df["MNI"].values]
# new_df["z"] = [x.replace("[", "").replace("]", "").split(",")[2] for x in df["MNI"].values]
# new_df["name"] = df["contact"]
# new_df["electrode"] = [match.group(1) for match in [re.match(r"([A-Za-z']+)([ \d]+)", name) for name in new_df['name'].values]]
# new_df["is_inmask"] = df["GreyWhite"] == "GreyMatter"
# new_df["roi_name"] = [row["MNI-HCP-MMP1"].split("_")[-1] if not pd.isna(row["MNI-HCP-MMP1"]) else " " for idx, row in df.iterrows()]
# new_df["roi"] = [np.where(atlas_ref["ROI_glasser_1"].values == x)[0][0] + 1 if x != " " else 0 for x in new_df["roi_name"].values ]
# new_df["region"] = [atlas_ref["ROI_glasser_2"].values[x-1] if x != 0 else " " for x in new_df["roi"].values]
# new_df["lobe"] = [atlas_ref["Lobe"].values[x-1] if x != 0 else " " for x in new_df["roi"].values]


# full = pd.concat([full, new_df], ignore_index=True)

# path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/23/MRI/Par_2024_XX7.csv"
# df = pd.read_csv(path, sep=";")
# df.columns = df.loc[1].values
# df.drop(index=[0, 1], inplace=True)
# df.reset_index(drop=True, inplace=True)
# df = df[:184].copy()

# new_df = pd.DataFrame()
# new_df["x"] = [x.replace("[", "").replace("]", "").split(",")[0] for x in df["MNI"].values]
# new_df["y"] = [x.replace("[", "").replace("]", "").split(",")[1] for x in df["MNI"].values]
# new_df["z"] = [x.replace("[", "").replace("]", "").split(",")[2] for x in df["MNI"].values]
# new_df["name"] = df["contact"]
# new_df["electrode"] = [match.group(1) for match in [re.match(r"([A-Za-z']+)([ \d]+)", name) for name in new_df['name'].values]]
# new_df["is_inmask"] = df["GreyWhite"] == "GreyMatter"
# new_df["roi_name"] = [row["MNI-HCP-MMP1"].split("_")[-1] if not pd.isna(row["MNI-HCP-MMP1"]) else " " for idx, row in df.iterrows()]
# new_df["roi"] = [np.where(atlas_ref["ROI_glasser_1"].values == x)[0][0] + 1 if x != " " else 0 for x in new_df["roi_name"].values ]
# new_df["region"] = [atlas_ref["ROI_glasser_2"].values[x-1] if x != 0 else " " for x in new_df["roi"].values]
# new_df["lobe"] = [atlas_ref["Lobe"].values[x-1] if x != 0 else " " for x in new_df["roi"].values]



# full = pd.concat([full, df], ignore_index=True)

# path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/28/MRI/Par_2025_GHAa.csv"
# df = pd.read_csv(path, sep=";")
# df.columns = df.loc[1].values
# df.drop(index=[0, 1], inplace=True)
# df.reset_index(drop=True, inplace=True)
# df = df[:213].copy()
# full = pd.concat([full, df], ignore_index=True)

# coords = np.array([x.replace("[", "").replace("]", "").split(",") for x in full["MNI"].values])
# coords = coords.astype(float)

# label_data = full["MNI-HCP-MMP1"].unique()
# label_data = label_data[~pd.isna(label_data)]


# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="low_contrast",
#     background="white",
#     size=(800, 600),
#     alpha=0.8,
# )
# i = 1
# palette = sns.color_palette("husl", len(label_data))
# for label in labels:
#     name = label.name.split("-")[0].replace("ROI", "")[:-1]
#     name = name.replace("R", "L")
#     if name in label_data:
#         color = palette[i]
#         brain.add_label(label, color=color, alpha=0.5) 
#         i += 1


# for i, coord in enumerate(coords):
#     brain.add_foci(coord.astype(int), scale_factor=0.2, color="black")

# from nilearn import plotting
# from nilearn import datasets, surface, plotting

# fsaverage_orig = datasets.fetch_surf_fsaverage("fsaverage5")

# fig, ax = plt.subplots(1, 1, figsize=(8, 4), subplot_kw={"projection": "3d"})
# plotting.plot_surf(fsaverage_orig["pial_left"], axes=ax, alpha=0.5)

# view = plotting.view_markers(final_coord)
# view.open_in_browser()

# ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
# plt.show()


# counts = full.groupby("MNI-HCP-MMP1").size().reset_index(name="count")


# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="low_contrast",
#     background="white",
#     size=(800, 600),
#     alpha=0.8,
# )

# labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "rh")
# labels_name = [label.name.split("-")[0].replace("ROI", "")[2:-1] for label in labels]
# labels_name
# counts2["label"] = [x.split("_")[-1] for x in counts2["MNI-HCP-MMP1"]]

# palette = sns.color_palette("RdBu", counts["count"].max()+1)
# for idx, row in counts.iterrows():
#     color = palette[row["count"]]
#     label_idx = np.where(np.array(labels_name) == row["label"])[0]
#     if len(label_idx) > 0:
#         label = labels[label_idx[0]]
#         brain.add_label(label, color=color, alpha=0.8) 


# brain.add_data(counts["count"].values, colormap="RdBu_r", alpha=0.8, colorbar=True, verbose=True)

# # Create a vertex-level data array (initialize with zeros)
# vtx_data = np.zeros(163842)  # Number of vertices in fsaverage's right hemisphere

# # Get a list of all vertex indices for each label region and assign them values
# for idx, row in final_count.iterrows():
#     count_value = row["count"]
#     label_name = row["true_label"]
#     label_idx = np.where(np.array(labels_name) == label_name)[0]
#     if len(label_idx) > 0:
#         label = labels[label_idx[0]]
#         vtx_data[label.vertices] = count_value

# vtx_data[vtx_data == 0] = np.nan  # Set non-label vertices to NaN
# # Now add the data with proper vertex mapping

# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="low_contrast",
#     background="white",
#     size=(800, 600),
#     alpha=0.8,
# )


# brain.add_data(
#     vtx_data,
#     colormap=custom_cmap, 
#     alpha=0.8, 
#     colorbar=True,
#     fmin=0,
#     fmax=final_count["count"].max(),
#     verbose=True
# )


# brain.add_foci(final_coord, scale_factor=0.2, color="black")



# final_count["count"].sum()

# for label in labels:
#     name = label.name.split("-")[0].replace("ROI", "")[:-1]
#     name = name.replace("R", "L")
#     if name in label_data:
#         color = palette[i]
#         brain.add_label(label, color=color, alpha=0.5) 


# initial_v1path = "/Users/charles.verstraete/Documents/w3_iEEG/analysis/data"
# test_full = pd.DataFrame()
# for subject in ["02", "04", "05", "08", "09", "12", "14", "16", "18", "19", "20"]:
#     #df = pd.read_csv(f"data/preprocessed/filter_bipolarized/filtered_{subject}_electrodes.csv")
#     df = pd.read_csv(initial_v1path + f"/preprocessed/electrode_position/subject_{subject}_electrodes.csv")
#     df = df[(df["lobe"] == "Fr") | (df["lobe"] == "Ins")]
#     df = df[df["is_inmask"]]
#     test_full = pd.concat([test_full, df], ignore_index=True)

# test_full
# full = full[(full["GreyWhite"] == "GreyMatter") & (~full["MNI-HCP-MMP1"].isna())].copy()
# counts = test_full.groupby("roi_name").size().reset_index(name="count")
# full
# counts2 = full.groupby("MNI-HCP-MMP1").size().reset_index(name="count")
# counts2["label"] = [x.split("_")[-1] for x in counts2["MNI-HCP-MMP1"]]

# coords = np.array([x.replace("[", "").replace("]", "").split(",") for x in full["T1pre Scanner Based"].values])
# coords = coords.astype(float)
# coords2 = test_full[["x", "y", "z"]].values

# final_coord = np.concatenate((coords, coords2), axis=0)
# final_coord[:, 0] = np.abs(final_coord[:, 0])
# final_count = pd.merge(counts, counts2, left_on="roi_name", right_on="label", how="outer")
# final_count["count"] = final_count["count_x"].fillna(0) + final_count["count_y"].fillna(0)
# final_count["true_label"] = final_count["roi_name"].fillna(final_count["label"])
# final_count = final_count[["true_label", "count"]]

# final_count["count"].sum()

# full.columns
# from matplotlib.colors import hsv_to_rgb, ListedColormap

# vmin = final_count["count"].min()
# vmax = final_count["count"].max()

# base_cmap = plt.get_cmap("RdBu_r")

# # Create array of values to sample colors from
# x = np.linspace(0, 1, 256)

# # Create a custom normalization centered at the specified value
# center_norm = (10 - vmin) / (vmax - vmin)

# # Create gamma-like function to emphasize values around center
# idx = np.where(
#     x < center_norm,
#     0.5 * (x / center_norm) ** 2,
#     1 - 0.5 * ((1 - x) / (1 - center_norm)) ** 2
# )

# custom_cmap = ListedColormap(base_cmap(idx))


# test_path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/25/Localisation/MRIcentered.txt"
# df = pd.read_csv(test_path, sep="\t")
# df.drop(index=121, inplace=True)
# df.reset_index(drop=True, inplace=True)
# electrode_coords = df[["x", "y", "z"]].values


# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="low_contrast",
#     background="white",
#     size=(800, 600),
#     alpha=0.8,
# )

# brain.add_foci(coord_test)

# path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/25/BRA_THA/ElectrodesAllCoordinates.txt"
# df = pd.read_csv(path, sep="\t")
# df2 = df[244:365].copy().reset_index(drop=True)
# df2.columns = df.columns

# np.where(df2["CT_voxel"] == "MRI_voxel")[0]
# np.where(df2["Unnamed: 1"].isna())[0]

# coord_test = df2[["x", "y", "z"]].values.astype(int)

# coord_test
# atlas_path = "/Users/charles.verstraete/Documents/w3_iEEG/analysis/data/anatomical_atlas/HCP-MMP1.nii"
# atlas_img = nib.load(atlas_path)
# atlas_data = atlas_img.get_fdata()

# atlas_data[~np.isin(atlas_data, test_parcels)] = 0
# atlas_img = nib.Nifti1Image(atlas_data, atlas_img.affine, atlas_img.header)

# view = plotting.plot_img_on_surf(
#     atlas_img,
#     surf_mesh='fsaverage4', 
#     colorbar=True,
#     bg_on_data=True,
#     cmap = "tab20",
#     )

# atlas_mesh = surface.vol_to_surf(atlas_img, fsaverage_orig["pial_left"], mask_img=atlas_img)
# atlas_mesh = atlas_mesh.astype(int)

# for val in test_parcels:
#     if val != 0:
#         sub_atlas = atlas_mesh.copy()
#         sub_atlas[sub_atlas != val] = 0
#         sub_atlas[sub_atlas == val] = 1
#         plotting.plot_surf_roi(roi_map=atlas_mesh,
#             view="lateral",
#             surf_mesh=fsaverage_orig["pial_left"])
#         plt.show()



# atlas_data = atlas_data.astype(int)
# atlas_data[atlas_data == 0] = np.nan
# fig, axs = plt.subplots(3, 3, figsize=(10, 10))
# for i in range(120, 190, 10):
#     ax = axs.flatten()[i // 10 - 12]
#     good_coord = vox_coords[(vox_coords[:, 1] <= i)]
#     ax.imshow(atlas_data[:, i, :].T, cmap="tab20", origin="lower")
#     ax.scatter(good_coord[:, 0], good_coord[:, 2], c="black", s=20)
# plt.colorbar()
# plt.show()

# coord_test
# atlas_data

# vox_coords
# atlas_data = atlas_img.get_fdata()
# affine = atlas_img.affine


# # Créer une liste des coordonnées de voxels valides et leurs valeurs
# x, y, z = np.indices(atlas_data.shape)
# voxel_coords = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
# values = atlas_data.flatten()

# # Ne garder que les voxels avec des étiquettes valides (non-NaN, non-zéro)
# valid = ~np.isnan(values) & (values > 0)
# valid_voxels = voxel_coords[valid]
# valid_values = values[valid]

# # Construire un KDTree avec les voxels valides
# voxel_tree = cKDTree(valid_voxels)

# # Convertir les coordonnées MNI en indices de voxels
# vox_coords = np.zeros((len(electrode_coords), 3), dtype=int)

# for i, coord in enumerate(electrode_coords):
#     # Convertir en espace voxel
#     vox_coord = np.linalg.inv(affine).dot(np.append(coord, 1))[:3]
#     vox_coords[i] = np.round(vox_coord).astype(int)

# # Trouver les parcelles les plus proches
# distances, indices = voxel_tree.query(vox_coords)
# parcels = valid_values[indices].astype(int)
# distances
# test_parcels = np.unique(parcels)

# labels_ref_path = "/Users/charles.verstraete/Documents/w3_iEEG/analysis/data/anatomical_atlas/HCP-MMP1_labels.csv"
# labels_ref = pd.read_csv(labels_ref_path, sep=";")
# labels_ref
# df2["roi_number"] = parcels
# df2["roi_label"] = [labels_ref.loc[labels_ref["ROI_n"] == x, "ROI_glasser_1"].values[0] for x in parcels]

# labels_ref
# parcels

# test_parcels




# from scipy.spatial import cKDTree


# tree = cKDTree(coord_test)

# dist, nearest_idx = tree.query([x, y, z])

# print(f"Dimensions: {voxel_tree.m}")
# print(f"Number of points: {voxel_tree.n}")
# print(f"Min bounds: {voxel_tree.mins}")
# print(f"Max bounds: {voxel_tree.maxes}")

# # Plot a simple representation of points in the tree
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot electrode positions
# ax.scatter(coord_test[:, 0], coord_test[:, 1], coord_test[:, 2], c='blue', s=30)

# # Set labels
# ax.set_xlabel('X (mm)')
# ax.set_ylabel('Y (mm)') 
# ax.set_zlabel('Z (mm)')
# ax.set_title('Electrodes in KD-Tree')

# plt.tight_layout()
# plt.show()