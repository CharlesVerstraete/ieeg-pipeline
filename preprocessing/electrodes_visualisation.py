#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Electrodes visualisation
"""

# Import libraries



# test = plt.get_cmap('tab20b')  # for up to 20 colors
# colors = test(np.linspace(0, 1, 20))  # Generate 26 distinct colors
# area_colors = {key : colors[val-1] for key, val in area_dict.items()}


# count = full.groupby(["roi_n", "roi_name", "region", "subject"]).size().reset_index(name="count")
# fig, axes = plt.subplots(5, 4, figsize=(20, 10), sharey=True)
# for ((region_name, sdf), ax) in zip(count.groupby("region"), axes.flatten()):
#     color = colors[area_dict[region_name]-1]
#     sns.barplot(data=sdf, x="roi_name", y="count", alpha = 0.8, errorbar=None, ax=ax, color=color)
#     sns.stripplot(data=sdf, x="roi_name", y="count", size=5, alpha=1, ax=ax, color=color)
#     ax.set_title(region_name)
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.legend().remove()
# plt.tight_layout()
# plt.show()


# area_colors = {key : colors[val-1] for key, val in area_dict.items()}
# count = full.groupby("region").size().reset_index(name="count")
# count["area"] = [area_dict[x] for x in count["region"].values]
# count.sort_values("area", inplace=True)
# area_colors = {key : colors[val-1] for key, val in area_dict.items()}

# sns.barplot(data=count, x="region", y="count", hue="region", alpha = 0.6, errorbar=None, palette=area_colors)
# sns.stripplot(data=count, x="region", y="count", hue="region", size=5, alpha=1, palette=area_colors)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "localisation", "roi_count.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
# plt.show()


# count = full.groupby(["roi_n", "roi_name", "region"]).size().reset_index(name="count")
# count = full.groupby(["region"]).size().reset_index(name="count")

# atlas = surface.load_surf_data(initial_v1path + "/anatomical_atlas/rh.HCP-MMP1.annot")
# fsaverage_orig = datasets.fetch_surf_fsaverage("fsaverage")
# fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
# coords_orig, _ = surface.load_surf_data(fsaverage_orig[f"pial_right"])
# coords_target, _ = surface.load_surf_mesh(fsaverage[f"pial_right"])
# resampled_atlas = surface_resample(atlas, coords_orig, coords_target)


# atlas_filtered = np.zeros(resampled_atlas.shape)
# for (i, (idx, row)) in  enumerate(count.iterrows()) :
#     sdf = atlas_ref[atlas_ref["ROI_glasser_2"] == row["region"]]
#     for (j, (idx2, row2)) in enumerate(sdf.iterrows()) :
#         atlas_filtered[resampled_atlas == row2["ROI_n"]] = row["count"]
# atlas_filtered[atlas_filtered == 0] = np.nan



# custom_cmap = create_customcmap(count["count"].min(), count["count"].max(), 10, "Spectral_r")

# plotting.plot_surf_roi(
#     surf_mesh=fsaverage["pial_right"],
#     roi_map=atlas_filtered,
#     bg_map=fsaverage["sulc_right"],
#     cmap=custom_cmap,
#     colorbar=True,
#     bg_on_data=True,
#     avg_method='max'
# )
# plt.show()


# coords = full[["x", "y", "z"]].values.astype(float)
# coords[:, 0] = np.abs(coords[:, 0])
# collection_img = {}
# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="ivory",
#     background="white",
#     size=(4096, 2160),
#     alpha=0.5,
# )
# for view in ["lateral", "medial", "dorsal", "ventral", "rostral", "frontal"]:
#     # brain.add_foci(
#     #     coords,
#     #     scale_factor=0.15,
#     #     color="black"
#     # )
#     brain.add_data(
#         atlas_filtered,
#         colormap=custom_cmap, 
#         alpha=1, 
#         colorbar=False,
#         fmin = 0,
#         fmax = count["count"].max() + 1,
#     )
#     brain.show_view(view)
#     screenshot = brain.screenshot()
#     collection_img[view] = screenshot

# brain.close()


# tick_positions = sorted(area_dict.values())  # Numeric positions
# tick_labels = [region for region, value in sorted(area_dict.items(), key=lambda x: x[1])]  # Region names


# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=3,
#     figsize=(21, 12),
# )

# for (i, (view, ax)) in enumerate(zip(collection_img.keys(), axes.flatten())):
#     nonwhite_pix = (collection_img[view] != 255).any(-1)
#     nonwhite_row = nonwhite_pix.any(1)
#     nonwhite_col = nonwhite_pix.any(0)
#     cropped_screenshot = collection_img[view][nonwhite_row][:, nonwhite_col]
#     ax.imshow(cropped_screenshot)
#     ax.set_title(view)
#     ax.axis("off")
# plt.tight_layout()
# cbar = plt.colorbar(
#     mappable=mappable,
#     cax=fig.add_axes([0.92, 0.15, 0.02, 0.7]),
#     label="Brain Region",
#     orientation="vertical"
# )
# cbar.ax.set_yticklabels(tick_labels)  # Use the region names as labels
# cbar.ax.tick_params(labelsize=8)
# plt.savefig(os.path.join(FIGURES_DIR, "localisation", "area_colored.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
# plt.show()

# list(area_dict.keys())

# frontal_atlas = atlas_ref[(atlas_ref["Lobe"] == "Fr") | (atlas_ref["Lobe"] == "Ins")].copy()
# frontal_atlas["area"] = [area_dict[x] for x in frontal_atlas["ROI_glasser_2"].values]

# labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "rh")

# formated_labels = {label.name.split("_")[1] : label for label in labels[1:]}

# frontal_atlas["ROI_glasser_2"]


# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="ivory",
#     background="white",
#     size=(4096, 2160),
#     alpha=0.8,
# )
# 1/17
# test_colors = sns.color_palette("husl", 13)
# frontal_atlas[frontal_atlas["cortex"] == "Dorsolateral_Prefrontal"]
# for (i, (idx, row)) in enumerate(frontal_atlas[frontal_atlas["cortex"] == "Dorsolateral_Prefrontal"].iterrows()) :
#     color = test_colors[i]
#     name = row["ROI_glasser_1"]
#     #region = row["ROI_glasser_2"]
#     label = formated_labels[name]
#     brain.add_label(label,color=color,alpha=1, hemi="rh")
#     #brain.add_text(text = f"{name}-{region}", color=color, x= 0.1, y = 0.02*i)

# frontal_atlas[frontal_atlas["ROI_glasser_2"] == "Area_55-DLPFC"]
# frontal_atlas[frontal_atlas["ROI_glasser_2"] == "Area9_lateral"]
# frontal_atlas

# for (i, (idx, row)) in enumerate(frontal_atlas.iterrows()) :
#     name = row["ROI_glasser_1"]
#     color = colors[row["area"]-1]
#     label = formated_labels[name]
#     brain.add_label(label,color=color,alpha=1, hemi="rh")
#     #brain.add_text(text = row["ROI_glasser_2"], color=color, x= 0.1, y = 0.05*i)


# brain = mne.viz.Brain(
#     "fsaverage",
#     "rh",
#     "pial",
#     cortex="ivory",
#     background="white",
#     size=(4096, 2160),
#     alpha=0.8,
# )
# for (i, (region, sdf)) in enumerate(frontal_atlas.groupby("ROI_glasser_2")) :
#     color = area_colors[region]
#     brain.add_text(text = region, color=color, x= 0.1, y = 0.05*(area_dict[region]-1))
#     for (i, (idx, row)) in enumerate(sdf.iterrows()) :
#         name = row["ROI_glasser_1"]
#         label = formated_labels[name]
#         brain.add_label(label,color=color,alpha=1, hemi="rh")



# 1/20

# collection_img = {}
# for view in ["lateral", "medial", "dorsal", "ventral", "rostral", "frontal"]:
#     brain.show_view(view)
#     screenshot = brain.screenshot()
#     collection_img[view] = screenshot

# brain.close()


# # let's create a figure with all the views for both hemispheres
# views = [
#     "lateral",
#     "medial",
#     "dorsal",
#     "ventral",
#     "anterior",
#     "posterior",
# ]




# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=3,
#     subplot_kw={"projection": "3d"},
#     figsize=(17, 10),
# )
# axes = np.atleast_2d(axes)


# for view, ax in zip(views, axes.flatten()):
#         plotting.plot_surf(
#             surf_mesh=fsaverage["pial_right"],
#             surf_map=atlas_filtered,
#             bg_map=fsaverage["sulc_right"],
#             cmap=custom_cmap,
#             colorbar=False,
#             bg_on_data=True,
#             avg_method='max',
#             view=view,
#             figure=fig,
#             axes=ax,
#             title=view,
#         )
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize

# # cmap_roi = ListedColormap(colors)
# norm = Normalize(vmin=count["count"].min(), vmax=count["count"].max() + 1)
# mappable = ScalarMappable(norm=norm, cmap=custom_cmap)
# mappable.set_array([])  # Needed when not mapping to a specific array

# plt.colorbar(
#     mappable=mappable,
#     cax=fig.add_axes([0.92, 0.15, 0.02, 0.7]),
#     label="Number of electrodes",
#     orientation="vertical",
#     ticks=np.arange(0, count["count"].max() + 1, 10),
#     format="%.f"
# )
# plt.savefig(os.path.join(FIGURES_DIR, "localisation", "electrode_density.pdf"), transparent=True, format='pdf',  bbox_inches='tight')
# plt.savefig(os.path.join(FIGURES_DIR, "localisation", "electrode_density.png"), transparent=True, format='png',  bbox_inches='tight', dpi=500)

# plt.show()

