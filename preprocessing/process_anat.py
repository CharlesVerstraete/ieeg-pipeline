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

import matplotlib
matplotlib.use('Qt5Agg')

atlas_path = "/Users/charles.verstraete/Documents/w3_iEEG/analysis/data/anatomical_atlas"
atlas_ref, atlas_img, atlas_data = get_atlas_data(atlas_path)
atlas_voxels, atlas_values = format_atlas_data(atlas_data)

gre_path_list = ["/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/3/Localisation/Gre_2024_GUIa.xlsx",
                 "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/23/MRI/Par_2024_XX7.csv",
                 "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/28/MRI/Par_2025_GHAa.csv"]

full = pd.DataFrame()

bdx_path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/25/BRA_THA/MRIcentered.txt"
df = get_bordeaux_df(bdx_path)
df["Contact_n"] = df["Contact"].astype(int)
df.rename(columns={"Electrode": "electrode"}, inplace=True) 
col_name = ['x', 'y', 'z', 'name', 'electrode', 'is_inmask', 'roi_n', 'roi_name', 'region', 'lobe']

coords = coordinate_transformation(df[["x", "y", "z"]].values.astype(float), atlas_img.affine).astype(int)
distances, parcels = find_nearest_neighbors(coords, atlas_voxels, atlas_values)
df["roi_n"] = parcels
df["distances"] = distances
df = assign_labels(df, atlas_ref)
df["name"] = [f"{elec}{n}" for elec, n in df[["electrode", "Contact_n"]].values]

df = df[(df["lobe"] == "Fr") | (df["lobe"] == "Ins")]
df = df[df["is_inmask"]]
df = df[col_name]

full = pd.concat([full, df], axis=0)

for gre_path in gre_path_list :
    df = get_grenoble_df(gre_path)
    df = format_grenoble_df(df, atlas_ref)
    coords = coordinate_transformation(df[["x", "y", "z"]].values.astype(float), atlas_img.affine).astype(int)
    distances, parcels = find_nearest_neighbors(coords, atlas_voxels, atlas_values)
    df["roi_n"] = parcels
    df["distances"] = distances
    df = assign_labels(df, atlas_ref)
    df = df[(df["lobe"] == "Fr") | (df["lobe"] == "Ins")]
    df = df[df["is_inmask"]]
    df = df[col_name]
    full = pd.concat([full, df], axis=0)

initial_v1path = "/Users/charles.verstraete/Documents/w3_iEEG/analysis/data"
for subject in ["02", "04", "05", "08", "09", "12", "14", "16", "18", "19", "20"]:
    df = pd.read_csv(initial_v1path + f"/preprocessed/electrode_position/subject_{subject}_electrodes.csv")
    df = df[(df["lobe"] == "Fr") | (df["lobe"] == "Ins")]
    df = df[df["is_inmask"]]
    df.rename(columns={"roi": "roi_n"}, inplace=True) 
    df = df[col_name]
    full = pd.concat([full, df], axis=0)




count = full.groupby(["roi_n", "region"]).size().reset_index(name="count")
fig = plt.figure(figsize=(20, 10))
sns.barplot(data=count, x="region", y="count", hue="region", alpha = 0.3, errorbar=None)
sns.stripplot(data=count, x="region", y="count", hue="region", size=12, alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "localisation", "roi_count.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
plt.show()


atlas = surface.load_surf_data(initial_v1path + "/anatomical_atlas/rh.HCP-MMP1.annot")
fsaverage_orig = datasets.fetch_surf_fsaverage("fsaverage")
fsaverage = datasets.fetch_surf_fsaverage("fsaverage6")
coords_orig, _ = surface.load_surf_data(fsaverage_orig[f"pial_right"])
coords_target, _ = surface.load_surf_mesh(fsaverage[f"pial_right"])
resampled_atlas = surface_resample(atlas, coords_orig, coords_target)

atlas_filtered = np.zeros(atlas.shape)
for (i, (idx, row)) in  enumerate(count.iterrows()) :
    atlas_filtered[atlas == row["roi_n"]] = row["count"]
atlas_filtered[atlas_filtered == 0] = np.nan



custom_cmap = create_customcmap(count["count"].min(), count["count"].max(), 5, "Spectral_r")

plotting.plot_surf_stat_map(
    surf_mesh=fsaverage["pial_right"],
    stat_map=atlas_filtered,
    bg_map=fsaverage["sulc_right"],
    cmap=custom_cmap,
    colorbar=True,
    bg_on_data=True,
    avg_method='max'
)
plt.show()

brain = mne.viz.Brain(
    "fsaverage",
    "rh",
    "pial",
    cortex="classic",
    background="white",
    size=(800, 600),
    alpha=0.8,
)


brain.add_data(
    atlas_filtered,
    colormap=custom_cmap, 
    alpha=0.8, 
    colorbar=True,
    fmin=0,
    fmax=count["count"].max(),
    colorbar_kwargs={
        "fmt": "%.f",
        "n_labels": 5,
    }
)

coord = full[["x", "y", "z"]].values.astype(float)
coord[:, 0] = np.abs(coord[:, 0])
brain.add_foci(coord, scale_factor=0.2, color="black")

labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "rh")
