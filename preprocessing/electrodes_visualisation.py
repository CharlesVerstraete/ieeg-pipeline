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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


from scipy.ndimage import binary_dilation



atlas_ref, atlas_img, atlas_data = get_atlas_data(ATLAS_DIR)


merged_sulcus = np.zeros(atlas_data.shape)
n_sulcus = 0
for subject in SUBJECTS :
    path = os.path.join(DATA_DIR, f'sub-{subject:03d}', 'raw', 'anat', f'sub-{subject:03}_sulcus.nii')
    if os.path.exists(path) :
        img = nib.load(path)
        data = img.get_fdata()
        data = data.astype(int)
        dilated_data = binary_dilation(data > 0, structure = np.ones((1, 1, 1))).astype(np.uint8)
        merged_sulcus += dilated_data
        n_sulcus += 1

merged_sulcus = merged_sulcus / n_sulcus
merged_sulcus[merged_sulcus == 0] = np.nan
sulcus_img = nib.Nifti1Image(merged_sulcus, atlas_img.affine, atlas_img.header)
nib.save(sulcus_img,  "merged_sulcus.nii")
plt.imshow(np.nanmean(merged_sulcus[85:90, :, :], axis=0, keepdims=True).T, cmap="jet", origin="lower")
plt.colorbar()
plt.show()

complete_electrodes = pd.DataFrame()
for subject in [2, 3, 4, 5, 8, 9, 12, 14, 16, 18, 19, 20, 23, 25, 28] :
    path = os.path.join(DATA_DIR, f'sub-{subject:03d}', 'raw', 'anat', f'sub-{subject:03}_electrodes-bipolar.csv')
    df = pd.read_csv(path)
    df["subject"] = subject
    complete_electrodes = pd.concat([complete_electrodes, df], ignore_index=True)

frontal_electrodes = complete_electrodes[(complete_electrodes["lobe"] == "Fr") | (complete_electrodes["lobe"] == "Ins")].copy()
frontal_electrodes_gm = frontal_electrodes[frontal_electrodes["is_inmask"]].copy()

# frontal_electrodes_gm = group_transition.anatomy_data

complete_electrodes_gm = complete_electrodes[complete_electrodes["is_inmask"]].copy()



# complete_electrodes.columns

complete_electrodes_gm = complete_electrodes_gm.merge(atlas_ref[["cortex", "ROI_n", "ROI_glasser_1", "ROI_glasser_2"]], left_on="roi_name", right_on="ROI_glasser_1", how="left")

count = complete_electrodes_gm.groupby(["cortex", "subject"]).size().reset_index(name="count")

count_subject = count.groupby(["cortex"]).size().reset_index(name="count")
count_subject = count_subject.merge(atlas_ref[["cortex", "ROI_n"]], left_on="cortex", right_on="cortex", how="left")

count_total = count.groupby(["cortex"])["count"].sum().reset_index()
count_total = count_total.merge(atlas_ref[["cortex", "ROI_n"]], left_on="cortex", right_on="cortex", how="left")


count = pd.read_csv("subjects___electrodes_per_brain_area__filtered_to_nomenclature_only_.csv")
count = count.merge(atlas_ref[["cortex", "ROI_n"]], left_on="network", right_on="cortex", how="left")

count

count["area"] = [area_dict[x] for x in count["region"].values]
count.sort_values("area", inplace=True)

fig = plt.figure(figsize=(21, 12))
sns.barplot(data=count, x="cortex", y="count", hue = "subject", alpha = 1, errorbar=None)
plt.xticks(rotation=90)
plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "localisation", "roi_count.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
plt.show()






atlas = surface.load_surf_data(ATLAS_DIR + "/rh.HCP-MMP1.annot")
fsaverage_orig = datasets.fetch_surf_fsaverage("fsaverage")
fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
coords_orig, _ = surface.load_surf_data(fsaverage_orig[f"pial_right"])
coords_target, _ = surface.load_surf_mesh(fsaverage[f"pial_right"])
resampled_atlas = surface_resample(atlas, coords_orig, coords_target)

# count = complete_electrodes_gm.groupby(["region", "roi_n"]).size().reset_index(name="count")
# count = complete_electrodes_gm.groupby(["roi_n"]).size().reset_index(name="count")

custom_cmap = create_customcmap(count_subject["count"].min(), count_subject["count"].max(), 2, "viridis")
base_cmap = plt.get_cmap("viridis")
x = np.linspace(0, 1, 128) 
custom_cmap = ListedColormap(base_cmap(x))


atlas_filtered = np.zeros(resampled_atlas.shape)
for (i, (idx, row)) in  enumerate(count.iterrows()) :
    atlas_filtered[resampled_atlas == row["ROI_n"]] = row["total_electrodes"]
atlas_filtered[atlas_filtered == 0] = np.nan




atlas_filtered = np.zeros(resampled_atlas.shape)
for (i, (region, sdf)) in enumerate(atlas_ref.groupby("cortex")) :
    print(f"{i+1} : {region}")
    for (j, (idx, row)) in  enumerate(sdf.iterrows()) :
        atlas_filtered[resampled_atlas == row["ROI_n"]] = i + 1

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
brain.add_data(
    atlas_filtered,
    colormap=custom_cmap, 
    alpha=1, 
    colorbar=False,
    fmin = 1,
    fmax = count["total_electrodes"].max(),
)


collection_img = {}
for view in ["lateral", "medial", "dorsal", "ventral", "rostral", "frontal"]:
    brain.show_view(view)
    screenshot = brain.screenshot()
    collection_img[view] = screenshot

brain.close()

norm = Normalize(vmin=count["total_electrodes"].min(), vmax=count["total_electrodes"].max() + 1)
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
# plt.savefig(os.path.join(FIGURES_DIR, "localisation", "electrode_density.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
plt.savefig(os.path.join("../for_joao/total_heatmap_2.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
plt.show()




brain = mne.viz.Brain(
    "fsaverage",
    "rh",
    "pial",
    cortex="ivory",
    background="white",
    size=(4096, 2160),
    alpha=0.5
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
    # offscreen=True,
    # show=False,
)
for (i, (region, sdf)) in enumerate(frontal_atlas.groupby("ROI_glasser_2")) :
    color = area_colors[region]
    for (i, (idx, row)) in enumerate(sdf.iterrows()) :
        name = row["ROI_glasser_1"]
        label = formated_labels[name]
        brain.add_label(label,color=color,alpha=1, hemi="rh")

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
plt.savefig(os.path.join(FIGURES_DIR, "localisation", "areas.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
plt.show()