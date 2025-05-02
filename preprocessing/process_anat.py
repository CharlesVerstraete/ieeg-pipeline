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
from special_class.Anatomy import *
from special_class.Anatomy_visualisation import *

for subject in [2, 4, 5, 8, 9, 12, 14, 16, 18, 19, 20]:
    print(subject)
    init_path = os.path.join(FIRST_ANALYSIS__DATA_DIR, "raw", f"subject_{subject:02}", "anatomical")
    anat_path = os.path.join(init_path, f"sub{subject:02}.nii")
    right_segmentation_path = os.path.join(init_path, "segmentation", f"Rcortex_sub{subject:02}.nii")
    left_segmentation_path = os.path.join(init_path, "segmentation", f"Lcortex_sub{subject:02}.nii")
    left_sulcus_path = os.path.join(init_path, "sulcus", "sulciL.nii")
    right_sulcus_path = os.path.join(init_path, "sulcus", "sulciR.nii")
    implant_files = get_fileslist(os.path.join(init_path, "implantation"), ".txt")
    save_path = os.path.join(DATA_DIR, f'sub-{subject:03d}', 'raw', 'anat')

    anat_object = Anatomy(ATLAS_DIR, subject)
    anat_object.load_anatomical(anat_path)
    anat_object.merge_sulci(right_sulcus_path, left_sulcus_path)
    anat_object.create_segmented_mask(right_segmentation_path, left_segmentation_path)
    anat_object.load_electrodes(implant_files)
    anat_object.make_bipolar_df()
    anat_object.process_electrodes()
    anat_object.process_electrodes(df_type="bipolar")
    anat_object.add_bipolar_contacts_info()
    anat_object.save_electrodes(save_path)
    anat_object.save_electrodes(save_path, df_type="bipolar")
    anat_object.save_files(save_path)


############################################################################################################################################################################
############################################################################################################################################################################
# Custom import from Bordeaux
############################################################################################################################################################################
############################################################################################################################################################################



bdx_path = "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/25/BRA_THA/MRIcentered.txt"
anat_object = Anatomy(ATLAS_DIR, 25)
anat_object.load_electrodes_from_bordeaux(bdx_path)
save_path = os.path.join(DATA_DIR, f'sub-{25:03d}', 'raw', 'anat')
anat_object.save_electrodes(save_path)
anat_object.save_electrodes(save_path, df_type="bipolar")


############################################################################################################################################################################
############################################################################################################################################################################
# Custom import from Grenoble
############################################################################################################################################################################
############################################################################################################################################################################


gre_path_list = ["/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/3/Localisation/Gre_2024_GUIa.xlsx",
                 "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/23/MRI/Par_2024_XX7.csv",
                 "/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/28/MRI/Par_2025_GHAa.csv"]


for subject, gre_path in zip([3, 23, 28], gre_path_list) :
    print(subject)
    anat_object = Anatomy(ATLAS_DIR, subject)
    anat_object.load_electrodes_from_grenoble(gre_path)
    anat_object.save_electrodes(save_path)


