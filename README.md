# EEG-fMRI Direct Signal Voxelization (v3.4-dev)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17603484.svg)](https://doi.org/10.5281/zenodo.17603484)

A research-oriented Python pipeline for mapping scalp EEG activity into 3D voxel-level brain space using a hybrid information-theoretic and spatial-consistency framework.

**Status:** Methodology complete · Large-scale validation in progress  
**License:** CC BY-NC 4.0 (Academic use only)

---

## 🧠 Overview

This repository provides an implementation of a **hybrid mutual information (MI) + Dice coefficient** scoring approach for estimating voxel-level EEG representations, combined with **Leave-One-Subject-Out (LOSO)** cross-validation and **ICC reliability** analyses.

The pipeline enables frequency-specific spatial inference across canonical EEG bands (δ, θ, α, β, γ), with validation procedures designed to reduce circularity and improve generalizability.

---

## 🔬 Key Features

- **Hybrid MI–Dice scoring** for voxel-level EEG projection  
- **LOSO cross-validation** for subject-level generalization  
- **ICC(2,k) reliability analysis** for voxel-wise stability  
- **7-step circularity control framework**  
- **Frequency-specific mapping** (1–45 Hz)  
- **Modular Python architecture** for research use  

This implementation is intended for **non-commercial academic neuroscience research**.

---

## 📁 Repository Structure

├── src/ # Core MI-Dice voxelization algorithms
├── utils/ # Preprocessing + helper functions
├── data/ # Example EEG matrices (sample subject)
├── results/ # Output templates (NIfTI, ICC maps)
├── docs/ # Supplementary documentation (WIP)
├── LICENSE # CC BY-NC 4.0
├── CITATION.cff # Citation metadata
└── README.md # This file

---

## 🛠️ Pipeline Summary

### 1. Preprocessing
- Bandpass filtering (1–45 Hz)
- ICA artifact removal  
- CSD transformation  
- Hilbert envelope extraction  
- Epoching (2 s windows)

### 2. Voxelization
- Gray-matter voxel grid (~200k voxels, MNI space)
- Geodesic Gaussian weighting  
- Hemisphere normalization  
- **Hybrid score:**  
  \[
  H = \alpha \cdot MI + (1 - \alpha) \cdot Dice
  \]

### 3. Validation
- **Leave-One-Subject-Out optimization**  
- **ICC(2,k)** reliability maps  
- **Circularity reduction** (noise-floor estimation, spatial boundary checks)  

### 4. Output
- Frequency-specific voxel maps  
- Reliability volumes (NIfTI)  
- Seed-to-voxel and ROI-level summaries  

---

## 📊 Current Dataset Status

**SPIS Dataset (N=10)**  
✔ Processing complete  
✔ Full validation  
✔ Circularity: r = 0.33  

**LEMON Dataset (N=40)**  
⏳ Validation in progress  
⏳ LOSO weighting estimation  
⏳ Final ICC maps forthcoming  

---
Detailed usage will be added once LEMON validation is complete.

📖 Citation
If you use this repository, please cite:

Kemik, K., Aykaç, C. (2025).
EEG-fMRI Direct Signal Voxelization Pipeline (v3.4-dev).
GitHub Repository: https://github.com/keremkem/eegdirectsignalvoxelization

BibTeX:

bibtex
Kodu kopyala
@software{Kemik_Aykac_2025_voxelization,
  author    = {Kerem Kemik and Cansu Aykaç},
  title     = {EEG-fMRI Direct Signal Voxelization Pipeline},
  year      = {2025},
  version   = {3.4-dev},
  url       = {https://github.com/keremkem/eegdirectsignalvoxelization},
  note      = {Academic research use only}
}
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

👤 Authors
Dr. Kerem Kemik
Post-Doctoral Researcher — MD-PhD Neuroscience
📧 keremkemik9@gmail.com
🔗 github.com/keremkem

Dr. Cansu Aykaç - PsYD Neuropsychology

Vibecoded with Claude Sonnet 4.5.

🚧 Development Status
 Core MI–Dice algorithm

 LOSO cross-validation module

 SPIS pilot validation

 LEMON large-scale validation (ongoing)

 Manuscript submission (in preparation)

Last updated: November 2025
