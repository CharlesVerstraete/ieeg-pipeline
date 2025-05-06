#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Electrodes visualisation
"""

# Import libraries

from preprocessing.config import *
from preprocessing.utils.data_helper import *
from preprocessing.utils.anat_helper import *
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import matplotlib
matplotlib.use('Qt5Agg')
# plt.style.use('seaborn-v0_8-poster') 



test = plt.get_cmap('tab20b')  # for up to 20 colors
colors = test(np.linspace(0, 1, len(area_dict.keys())))  # Generate 26 distinct colors
area_colors = {key : colors[val-1] for key, val in area_dict.items()}
atlas_ref, atlas_img, atlas_data = get_atlas_data(ATLAS_DIR)


complete_electrodes = pd.DataFrame()
for subject in SUBJECTS :
    path = os.path.join(DATA_DIR, f'sub-{subject:03d}', 'raw', 'anat', f'sub-{subject:03}_electrodes-bipolar.csv')
    df = pd.read_csv(path)
    df["subject"] = subject
    complete_electrodes = pd.concat([complete_electrodes, df], ignore_index=True)

frontal_electrodes = complete_electrodes[(complete_electrodes["lobe"] == "Fr") | (complete_electrodes["lobe"] == "Ins")].copy()
frontal_electrodes_gm = frontal_electrodes[frontal_electrodes["is_inmask"]].copy()

count = frontal_electrodes_gm.groupby(["region"]).size().reset_index(name="count")
count["area"] = [area_dict[x] for x in count["region"].values]
count.sort_values("area", inplace=True)

fig = plt.figure(figsize=(21, 12))
sns.barplot(data=count, x="region", y="count", hue="region", alpha = 1, errorbar=None, palette=area_colors)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "localisation", "roi_count.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
plt.show()






atlas = surface.load_surf_data(FIRST_ANALYSIS__DATA_DIR + "/anatomical_atlas/rh.HCP-MMP1.annot")
fsaverage_orig = datasets.fetch_surf_fsaverage("fsaverage")
fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
coords_orig, _ = surface.load_surf_data(fsaverage_orig[f"pial_right"])
coords_target, _ = surface.load_surf_mesh(fsaverage[f"pial_right"])
resampled_atlas = surface_resample(atlas, coords_orig, coords_target)

count = frontal_electrodes_gm.groupby(["region", "roi_n"]).size().reset_index(name="count")
custom_cmap = create_customcmap(count["count"].min(), count["count"].max(), 10, "Spectral_r")
atlas_filtered = np.zeros(resampled_atlas.shape)
for (i, (idx, row)) in  enumerate(count.iterrows()) :
    atlas_filtered[resampled_atlas == row["roi_n"]] = row["count"]
atlas_filtered[atlas_filtered == 0] = np.nan

brain = mne.viz.Brain(
    "fsaverage",
    "rh",
    "pial",
    cortex="ivory",
    background="white",
    size=(4096, 2160),
    alpha=0.5,
)

collection_img = {}
for view in ["lateral", "medial", "dorsal", "ventral", "rostral", "frontal"]:
    brain.add_data(
        atlas_filtered,
        colormap=custom_cmap, 
        alpha=1, 
        colorbar=False,
        fmin = 0,
        fmax = count["count"].max() + 1,
    )
    brain.show_view(view)
    screenshot = brain.screenshot()
    collection_img[view] = screenshot

brain.close()

norm = Normalize(vmin=count["count"].min(), vmax=count["count"].max() + 1)
mappable = ScalarMappable(norm=norm, cmap=custom_cmap)
mappable.set_array([])  # Needed when not mapping to a specific array

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(21, 12),
)

for (i, (view, ax)) in enumerate(zip(collection_img.keys(), axes.flatten())):
    nonwhite_pix = (collection_img[view] != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = collection_img[view][nonwhite_row][:, nonwhite_col]
    ax.imshow(cropped_screenshot)
    ax.set_title(view)
    ax.axis("off")
cbar = plt.colorbar(
    mappable=mappable,
    cax=fig.add_axes([0.95, 0.15, 0.02, 0.7]),
    label="Brain Region",
    orientation="vertical"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "localisation", "electrode_density.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
plt.show()

brain = mne.viz.Brain(
    "fsaverage",
    "rh",
    "pial",
    cortex="ivory",
    background="white",
    size=(4096, 2160),
    alpha=0.8
)
for (i, (region, sdf)) in enumerate(frontal_electrodes_gm.groupby("region")) :
    color = area_colors[region]
    coords = sdf[["x", "y", "z"]].values.astype(float)
    coords[:, 0] = np.abs(coords[:, 0])
    brain.add_foci(
        coords,
        scale_factor=0.2,
        color=color
    )

collection_img = {}
for view in ["lateral", "medial", "dorsal", "ventral", "rostral", "frontal"]:
    brain.show_view(view)
    screenshot = brain.screenshot()
    collection_img[view] = screenshot

brain.close()

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(21, 12),
)
for (i, (view, ax)) in enumerate(zip(collection_img.keys(), axes.flatten())):
    nonwhite_pix = (collection_img[view] != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = collection_img[view][nonwhite_row][:, nonwhite_col]
    ax.imshow(cropped_screenshot)
    ax.set_title(view)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "localisation", "electrodes_positions.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
plt.show()


frontal_atlas = atlas_ref[(atlas_ref["Lobe"] == "Fr") | (atlas_ref["Lobe"] == "Ins")].copy()
frontal_atlas["area"] = [area_dict[x] for x in frontal_atlas["ROI_glasser_2"].values]
labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "rh")
formated_labels = {label.name.split("_")[1] : label for label in labels[1:]}

brain = mne.viz.Brain(
    "fsaverage",
    "rh",
    "pial",
    cortex="ivory",
    background="white",
    size=(4096, 2160),
    alpha=0.8,
)
for (i, (region, sdf)) in enumerate(frontal_atlas.groupby("ROI_glasser_2")) :
    color = area_colors[region]
    for (i, (idx, row)) in enumerate(sdf.iterrows()) :
        name = row["ROI_glasser_1"]
        label = formated_labels[name]
        brain.add_label(label,color=color,alpha=1, hemi="rh")


