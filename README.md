# iEEG-Pipeline

Processing pipeline for intracranial EEG (iEEG) data analysis.

## Overview

This project provides tools for importing, processing, and analyzing intracranial EEG data. It handles raw data import from various formats, implements BIDS-compatible organization, and includes specialized tools for anatomical localization, event alignment, artifact detection, and signal preprocessing.

## Features

- Import data from raw TRC, EDF or Nx files to BIDS-compatible format
- **Comprehensive anatomical processing pipeline:**
  - Automatic electrode localization from implantation files
  - Segmentation of gray and white matter
  - Sulci identification and analysis
  - Atlas-based parcellation of electrode locations
  - Generation of both monopolar and bipolar electrode configurations
  - 3D visualization of electrode placement
  - Grey/white matter classification of electrode contacts
- Event detection and alignment between signal and behavioral logs
- Signal filtering and preprocessing
- Artifact detection and removal
- Custom processing for special cases (e.g., analog triggers)
- Advanced visualization tools for both signal and anatomical data

## Anatomical Pipeline

The anatomical pipeline provides tools for localizing electrodes and analyzing their placement within brain structures:

1. **Anatomical Data Loading**: Process NIfTI files with the `Anatomy` class
2. **Tissue Segmentation**: Separate gray/white matter and create binary masks
3. **Sulci Analysis**: Merge and process sulci information from both hemispheres
4. **Electrode Localization**: Map electrode coordinates to anatomical structures
5. **Atlas Integration**: Maps electrodes to brain regions using multiple atlases (HCP-MMP1, etc.)
6. **Bipolar Derivation**: Creation of bipolar contact configurations with anatomical mapping
7. **Visualization**: Use the `Anatomy_visualisation` class for interactive 3D plotting

## Directory Structure

```
├── preprocessing/
│   ├── config.py                 # Configuration settings
│   ├── import_data.py            # Data import pipeline
│   ├── process_anat.py           # Anatomical processing workflows
│   ├── utils/
│   │   ├── data_helper.py        # Helper functions
│   │   ├── anat_helper.py        # Anatomical processing utilities
│   │   └── ...
├── special_class/
│   ├── Anatomy.py                # Core anatomical processing class
│   ├── Anatomy_visualisation.py  # Visualization extensions for anatomy
│   └── ...
├── data/
│   ├── sub-XXX/                  # Subject-specific folders
│   │   ├── raw/
│   │   │   ├── ieeg/             # Raw iEEG recordings
│   │   │   ├── anat/             # Anatomical data (NIfTI files)
│   │   │   └── beh/              # Behavioral data
│   │   └── preprocessed/
│   │       ├── aligned/          # Time-aligned data
│   │       ├── filtered/         # Filtered signals
│   │       ├── epochs/           # Epoched data
│   │       └── timefreq/         # Time-frequency analyses
├── figures/
│   ├── localisation/             # Electrode localization visualizations
│   └── ...
└── README.md
```

## Requirements

- Python 3.9+
- NumPy
- Pandas
- MNE
- Neo
- SciPy
- Matplotlib
- Nibabel (for NIfTI file processing)
- Nilearn (for visualization)

## Contact

charles.vrst@gmail.com