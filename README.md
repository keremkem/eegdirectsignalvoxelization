# EEG-fMRI Direct Signal Voxelization (v5.1.0-dev)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18860370.svg)](https://doi.org/10.5281/zenodo.18860370)
To cite all releases you can use: 10.5281/zenodo.17603483

A research-oriented Python pipeline for mapping scalp EEG activity into 3D voxel-level brain space using a hybrid information-theoretic and spatial-consistency framework.

**Status:** Methodology complete · Large-scale validation in progress  
**License:** CC BY-NC 4.0 (Academic use only)

---

## 🧠 Overview

This repository provides an implementation of a **hybrid mutual information (MI) + Dice coefficient** scoring approach for estimating voxel-level EEG representations, combined with **Leave-One-Subject-Out (LOSO)** cross-validation and **ICC reliability** analyses.

The pipeline enables frequency-specific spatial inference across canonical EEG bands (δ, θ, α, β, γ), with validation procedures designed to reduce circularity and improve generalizability. As of v5.1.0, band power is computed as relative power (normalized by total broadband power), eliminating 1/f covariance artifacts, and Dice similarity is computed on ranked activations, making scores amplitude-invariant. Output volumes are compatible with SPM, CONN, and FreeSurfer.

---

## 🔬 Key Features

- **Hybrid MI–Dice scoring for voxel-level EEG projection
- **Relative band power — eliminates 1/f broadband covariance (v5.1.0)
- **Rank-based Dice similarity — amplitude-invariant voxel overlap (v5.1.0)
- **LOSO cross-validation for subject-level generalization
- **ICC(2,k) reliability analysis for voxel-wise stability
- **7-step circularity control framework
- **Frequency-specific mapping across δ, θ, α, β, γ (1–45 Hz)
- **AR(1) prewhitening for temporal autocorrelation removal
- **HRF convolution (Glover 1999 canonical; CONN/SPM-compatible normalization)
- **CLI interface with argparse for scriptable batch processing
- **Per-subject QC CSV export alongside NIfTI outputs
- **Modular Python architecture for research use

This implementation is intended for **non-commercial academic neuroscience research**.

---

## 📁 Repository Structure

src/ # Core MI-Dice voxelization algorithms
utils/ # Preprocessing + helper functions
data/ # Example EEG matrices (sample subject)
results/ # Output templates (NIfTI, ICC maps)
docs/ # Supplementary documentation (WIP)
LICENSE # CC BY-NC 4.0
CITATION.cff # Citation metadata
README.md # This file

---

## 🛠️ Pipeline Summary

### 1. Preprocessing
- Bandpass filtering (1–45 Hz)
- ICA artifact removal  
- CSD transformation  
- Relative Hilbert envelope extraction (normalized by broadband power)
- Epoching (2 s windows)

### 2. Voxelization
- Gray-matter voxel grid (~200k voxels, MNI space)
- Geodesic Gaussian weighting (band-dependent sigma scaling) 
- Hemisphere normalization  
- **Hybrid score:**  
  \[
  H = \alpha \cdot MI + (1 - \alpha) \cdot Dice
  \]
- Dice computed on ranked activations (amplitude-invariant, v5.1.0)

### 3. Post-processing
- AR(1) prewhitening
- Z-score normalization
- HRF convolution (Glover canonical; no variance scaling — CONN-compatible)
- Temporal consistency filtering
- Raykill artifact cleanup

### 4. Validation
- Leave-One-Subject-Out (LOSO) weight optimization
- ICC(2,k) reliability maps
- Circularity reduction (noise-floor estimation, spatial boundary checks)
- Literature-derived validation targets per band 

### 4. Output
- Frequency-specific voxel maps (NIfTI)   
- Reliability volumes 
- Seed-to-voxel and ROI-level summaries
- Per-subject QC metrics (CSV)

---

💻 Usage
bash# Basic run (all subjects, all bands)
python eegclaudev5EO5.py

# Skip AR(1) prewhitening for faster iteration
python eegclaudev5EO5.py --no-ar1

# Disable LOSO (group K-fold fallback)
python eegclaudev5EO5.py --no-loso

Full usage documentation will be added once LEMON validation is complete.

## 📊 Current Dataset Status

**SPIS Dataset (N=10)**  
✔ Processing complete  
✔ Full validation  
✔ Circularity: r = 0.33  

**LEMON Dataset (N=40)**  
✔ Validation complete  
✔ LOSO weighting estimation and Circularity r=0,4 
⏳ Final ICC maps forthcoming  

---

📖 Citation
If you use this repository, please cite:

Kemik, K., Aykaç, C. (2025).
EEG-fMRI Direct Signal Voxelization Pipeline (v3.4-dev).
GitHub Repository: https://github.com/keremkem/eegdirectsignalvoxelization

---
BibTeX:
@software{Kemik_Aykac_2025_voxelization,
  author    = {Kerem Kemik and Cansu Aykaç},
  title     = {EEG-fMRI Direct Signal Voxelization Pipeline},
  year      = {2025},
  version   = {3.4-dev},
  url       = {https://github.com/keremkem/eegdirectsignalvoxelization},
  note      = {Academic research use only}
}

---
⚖️ License (Academic Use Only)
This software is released under the Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

Permitted:
Academic research

Educational use

Non-commercial studies

Methodological replication

Restricted:
Commercial applications

Proprietary software integration

Clinical or diagnostic use

For commercial licensing inquiries: keremkemik9@gmail.com

---

👤 Authors
Dr. Kerem Kemik
Post-Doctoral Researcher — MD-PhD Neuroscience
📧 keremkemik9@gmail.com
🔗 github.com/keremkem

Dr. Cansu Aykaç - PsYD Neuropsychology

Vibecoded with Claude Sonnet 4.5.

🚧 Development Status

✅ Core MI–Dice algorithm
✅ Relative band power & rank-based Dice (v5.1.0)
✅ LOSO cross-validation module
✅ AR(1) prewhitening
✅ CONN/SPM/FreeSurfer-compatible HRF output
✅ SPIS pilot validation (N=10)
⏳ LEMON large-scale validation (N=40, ongoing)
⏳ Manuscript submission (in preparation)

Data Sources: 
Data were obtained from the MPI-Leipzig Mind-Brain-Body (LEMON) dataset (Babayan et al., 2019; OpenNeuro accession ds000221). The first 40 participants of the young adult cohort were used (sub-032301 to sub-032342, excluding sub-032309 and sub-032335 due to data quality issues), all recorded during eyes-open resting state.
Also : M. Torkamani-Azar, S. D. Kanik, S. Aydin and M. Cetin, "Prediction of Reaction Time and Vigilance Variability From Spatio-Spectral Features of Resting-State EEG in a Long Sustained Attention Task," in IEEE Journal of Biomedical and Health Informatics, vol. 24, no. 9, pp. 2550-2558, Sept. 2020, doi: 10.1109/JBHI.2020.2980056. https://ieeexplore.ieee.org/document/9034192 (https://github.com/mastaneht/SPIS-Resting-State-Dataset).

See CHANGELOG.md for full version history.

Last updated: March 2026
