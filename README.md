# iEEG-Pipeline

Processing pipeline for intracranial EEG (iEEG) data analysis.

## Overview

This project provides tools for importing, processing, and analyzing intracranial EEG data. It handles raw data import from various formats, implements BIDS-compatible organization, and includes specialized tools for event alignment, artifact detection, and signal preprocessing.

## Features

- Import data from raw TRC, EDF or Nx files to BIDS-compatible format
- Automatically detect and classify electrode types from anat file
- Event detection and alignment between signal and behavioral logs
- Signal filtering and preprocessing
- Artifact detection and removal
- Custom processing for special cases (e.g., analog triggers)
- Visualization tools 

## Directory Structure

```
├── preprocessing/
│   ├── config.py                 # Configuration settings
│   ├── import_data.py            # Data import pipeline
│   ├── utils/
│   │   ├── data_helper.py        # Helper functions
│   │   └── ...
├── data/
│   ├── sub-XXX/                  # Subject-specific folders
│   │   ├── raw/
│   │   │   ├── ieeg/             # Raw iEEG recordings
│   │   │   ├── anat/             # Anatomical data
│   │   │   └── beh/              # Behavioral data
│   │   └── preprocessed/
│   │       ├── aligned/          # Time-aligned data
│   │       ├── filtered/         # Filtered signals
│   │       ├── epochs/           # Epoched data
│   │       └── timefreq/         # Time-frequency analyses
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


## Contact

charles.vrst@gmail.com