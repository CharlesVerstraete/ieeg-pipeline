# iEEG-Pipeline

Processing pipeline for intracranial EEG (iEEG) data analysis. It covers complete preprocessing (signal import, event alignment, anatomical localization) and processing (feature extraction, decoding with ridge/SVM, geometry-based analyses, spectral power/phase).

The repository is organized to keep raw/preprocessed data under `data/`, modular preprocessing utilities under `preprocessing/`, and analysis code under `analysis/`.

- Data organization: BIDS-like per subject under `data/sub-XXX/`
- Code organization: `preprocessing/` (import, alignment, anatomy) and `analysis/` (feature extraction, decoding, group/geometry analyses)
- Figures and derived outputs: `figures/` and `analysis/decoding_outputs/`

Core configuration and constants (paths, sampling, bands) live in [preprocessing/config.py](preprocessing/config.py). Project-wide subject lists, directories, and frequency bands are defined there and are reused across modules.

Directory snapshot:
- [preprocessing/](preprocessing/) — import, alignment, anatomy workflows
- [analysis/](analysis/) — decoding, geometry, group analyses
- [data/](data/) — subject-level raw and preprocessed outputs
- [figures/](figures/) — generated visualizations
- [special_class/](special_class/) — Anatomy classes and visualization helpers

--------------------------------------------------------------------------------

## 1) Preprocessing

This stage turns raw signals and logs into aligned, anatomically localized, epoched time-frequency data ready for decoding.
All configuration and constants: [preprocessing/config.py](preprocessing/config.py)


Signal processing and visualization:
- Import and IO
  - Supported formats: Micromed (.TRC), Neuralynx (.nx), generic (.edf).
  - Loader consolidates channel metadata (names, sampling rate, units) and reads behavioral logs/events.
  - Channel renaming/mapping to a consistent convention happens at import. See [preprocessing/import_data.py](preprocessing/import_data.py).
- Referencing, montage, selection
  - Bipolar referencing strategy.
  - Exclusion of non-neural channels (EKG/EMG) and known-noise channels at load time.
- Line-noise removal and filtering
  - Zapline method for 50 Hz and harmonics removing (de Cheveigné, A. (2019)).
  - High-pass (0.5 Hz)
- Event alignment
  - Alignment of EEG trigger streams to behavioral logs using Needleman–Wunsch sequence alignment.
- Epoching
  - Create epochs around key events (e.g., stimulus onset, feedback).
  - Attach trial metadata (condition labels, correctness, block/episode indices).
- Time–frequency decomposition
  - Multitaper decomposition frequencies and bands from `FREQUENCY_BANDS` (and/or a dense frequency grid).
  - Outputs per trial × channel × frequency × time:
    - Power (magnitude-squared), log-transform (dB) and baseline.
    - Phase (angle).

Persist aligned indices, cleaned continuous data (optional), epoched arrays, and time–frequency tensors under `data/sub-XXX/preprocessed/`.


Anatomy processing and visualization:

- [preprocessing/process_anat.py](preprocessing/process_anat.py)
  - Orchestrates subject-level anatomy workflow:
    - Contact import
    - Surface projection and metrics
    - Atlas labeling
    - Bipolar coordinates
- [special_class/Anatomy.py](special_class/Anatomy.py)
  - Create table from position of the electrodes:
    - Tree method to map channels location with HCP-MMP1 (Glasser, MF. (2016)).
    - Bulding unipolar and bipolar tables.
- [special_class/Anatomy_visualisation.py](special_class/Anatomy_visualisation.py)
  - Visualization utilities:
    - Static brain projections (scatter on surface or glass brain with nilearn).
    - ROI-level summaries.

Outputs (per subject under [data/](data/)):
- raw/ieeg: raw recordings and events (e.g., `events.tsv`)
- preprocessed/filtered: denoised/bipolar data
- preprocessed/aligned: alignment products (indices/mappings between EEG/log/behavior)
- preprocessed/epochs: trial-aligned epochs (selection/transition tasks)
- preprocessed/timefreq: time-frequency decompositions used later for power/phase features

--------------------------------------------------------------------------------

## 2) Processing

This stage extracts features (power/phase), runs decoding (ridge regression and SVM classification), aggregates across subjects, performs geometry-based summaries, and supports cross-temporal analyses.

Core modules:
- Configuration: [analysis/decoding/config.py](analysis/decoding/config.py)
- Data loaders:
  - Subject-level: [`analysis.decoding.loader.iEEGDataLoader`](analysis/decoding/loader.py)
  - Group-level: [`analysis.decoding.loader.GroupLoader`](analysis/decoding/loader.py)
- Feature extraction:
  - [`analysis.decoding.process_features.Features`](analysis/decoding/process_features.py) for power bands and phase matrices
    - Typical output shape for power: (n_trials, n_channels, n_bands, n_timepoints)
- Decoding:
  - [`analysis.decoding.decoding.Decoding`](analysis/decoding/decoding.py)
    - Models: RidgeCV (regression) and Linear SVM (classification)
    - Pipeline: StandardScaler → Estimator (KFold/StratifiedKFold CV)
    - Metrics provided (examples): ROC-AUC, F1, Pearson r, MSE
    - Modalities:
      - Global: channel-aggregated decoding curves over time
      - Channel: per-channel decoding curves (for topography and clustering)
- Orchestration and group analysis:
  - [`analysis.decoding.decoding_analysis.DecodingAnalysis`](analysis/decoding/decoding_analysis.py) (group-level aggregation, plotting, clustering hooks)
- Geometry-based analysis:
  - [`analysis.decoding.geometry.Geometry`](analysis/decoding/geometry.py)
    - Groups epochs by experimental factors (e.g., `is_stimstable`, `explor_exploit`, `switch_type`)
    - Aggregates per-group power over channels/frequency bands
    - Produces matrices for PCA
- Outputs:
  - Time-course decoding per variable (global and per-channel)
  - Cross-decoding matrices across timepoints (per subject and averaged)
  - Channel clustering based on decoding profiles (e.g., using hierarchical methods in group scripts)
  - Geometry/PCA summaries by behavior group and frequency bands

--------------------------------------------------------------------------------

## Directory Structure

```
ieeg-pipeline/
├── preprocessing/                         # Import, alignment, behavior, anatomy workflows
│   ├── config.py                          # Global paths, subjects, sampling, freq bands
│   ├── import_data.py                     # Raw import, filtering, epoching, time–frequency
│   ├── process_anat.py                    # MRI/CT coreg, contact coords, atlas labels, exports
│   ├── behaviour_analysis.py              # Behavioral summaries and plots (switches, RT, HMM)
│   └── utils/
│       ├── beh_helper.py                  # Behavioral feature engineering (persev/explor, criterion)
│       ├── data_helper.py                 # IO helpers, file listing, subject utilities
│       └── align_helper.py                # Event sequence alignment (Needleman–Wunsch, helpers)
│
├── analysis/                              # Decoding and higher-level analyses
│   ├── decoding/
│   │   ├── config.py                      # Decoding-side constants (bands, time windows, CV)
│   │   ├── loader.py                      # Load aligned features/labels (subject/group)
│   │   ├── process_features.py            # Build power/phase tensors from time–frequency
│   │   ├── decoding.py                    # Ridge/SVM pipelines, metrics, cross-decoding
│   │   ├── decoding_analysis.py           # Aggregation, plotting, clustering orchestration
│   │   ├── geometry.py                    # Grouped power, PCA-ready matrices, ROI summaries
│   │   └── group_analysis.py              # Group figures and summaries
│
├── special_class/                         # Anatomy classes and visualizations
│   ├── Anatomy.py                         # Contact/channel table, transforms, ROI filtering
│   └── Anatomy_visualisation.py           # Brain scatter, ROI plots, metrics overlays
│
├── data/                                  # Per-subject data (BIDS-like, raw → preprocessed)
│   ├── sub-XXX/                           # Subject folders
│   │   ├── raw/ieeg                       # Raw EEG and events
│   │   └── preprocessed/                  # Filtered, aligned, epochs, timefreq, anat
│   └── ...                                # Additional subjects
│
├── figures/                               # Generated figures (behavior, decoding, anatomy)
│   ├── behaviour/                         # Behavioral plots (switch, persev/explor, etc.)
│   ├── decoding/                          # Decoding curves, cross-temporal matrices, clusters
│   └── anatomy/                           # Brain plots, ROI summaries
│
│
└── README.md
```

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

## Requirements

- Python 3.9+
- NumPy, SciPy, Pandas
- MNE, Neo
- Matplotlib, Seaborn
- Nibabel, Nilearn (anatomy/visualization)

--------------------------------------------------------------------------------

## Contact

verstraetecarlito@gmail.com