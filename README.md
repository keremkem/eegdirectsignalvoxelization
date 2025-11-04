# EEG-fMRI Direct Signal Voxelization (v3.4)

A validated research pipeline for projecting EEG scalp signals onto voxel-level 3D anatomical space, enabling frequency-resolved functional connectivity mapping across five canonical EEG bands (δ, θ, α, β, γ).  
Developed by Dr. Kerem Kemik, in collaboration with Cansu Aykaç and colleagues (Izmir, Türkiye).

---

## Overview

This repository contains the full workflow and validation framework for the **EEG-to-voxel pipeline (v3.4)**.  
The project bridges electrophysiological and hemodynamic domains by mapping EEG-derived oscillatory activity onto the MNI cortical surface using mutual-information and Dice-based hybrid scoring.

Key features:
- Frequency-specific voxelization (1–45 Hz, five canonical bands)
- LOSO (Leave-One-Subject-Out) cross-validation
- MI–Dice hybrid weighting for spatial correspondence
- ICC-based test–retest reliability analysis
- Seven-tier validation against circularity bias (SM3)

This pipeline was designed for **methodological and research purposes** and is not intended for clinical or commercial use.

---

## Citation

If you use this repository, code, or methodology in academic work, please cite the forthcoming publication:

> Aykaç, C., Kemik, K., et al. (2025). *EEG-Derived Volumetric Connectivity: Establishing Frequency-Specific Reliability Baselines for Clinical Biomarker Development.*  
> Under review.

Example BibTeX:
```bibtex
@article{Aykac2025eegvoxel,
  author    = {Cansu Aykaç and Kerem Kemik and collaborators},
  title     = {EEG-Derived Volumetric Connectivity: Establishing Frequency-Specific Reliability Baselines for Clinical Biomarker Development},
  year      = {2025},
  note      = {Under review}
}
Method Summary
Dataset: SPIS Resting-State EEG Dataset (Torkamani-Azar et al., Sabancı University, 2020)
Subjects: 10 healthy adults (N=10)
System: BioSemi ActiveTwo (64 channels, 10-10 montage)
Preprocessing:

1–45 Hz bandpass, 50 Hz notch

Infomax ICA (FastICA, scikit-learn)

Alpha-protection algorithm to preserve occipital rhythms

Current Source Density (CSD) re-referencing
Voxelization:

200k gray matter voxels, Harvard–Oxford atlas mask

Geodesic Gaussian weighting (σ=22 mm)

Hemisphere isolation and normalization

MI–Dice hybrid scoring for voxel projection
Validation:

7-tier anti-circularity framework

ICC analysis for cross-method reliability

LOSO generalization testing

File Structure
/src                → Core voxelization scripts (Python)
/data               → Example EEG input files (MATLAB .mat)
/results            → Group-level ICC, ROI, and seed-to-voxel outputs
/docs               → Supplementary methods (SM1–SM3)
LICENSE             → CC BY-NC 4.0 License
README.md           → Documentation and usage policy
License and Usage Policy
This repository is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC 4.0).
The materials are intended solely for academic and research purposes.

✅ You may copy, modify, and share the code for academic or educational work.

❌ You may not use this repository or derivatives for commercial, diagnostic, or proprietary software.

🧠 You must cite the original publication and retain this license notice in derivative works.

📄 Full license text: CC BY-NC 4.0
© 2025 Dr. Kerem Kemik and Dr. Cansu Aykaç
✉️ Contact: keremkemik9@gmail.com

Acknowledgments
This work builds upon contributions from the SPIS Dataset (Sabancı University, Torkamani-Azar et al., 2020) and related open-source EEG processing frameworks.
Special thanks to Kerem Kemik for methodological development and system integration.

“This code is a bridge between electric silence and hemodynamic rhythm—designed only for research, not for profit.”

## License
This repository is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.  
The code and data are intended **solely for academic and research purposes**.  
Commercial or clinical use requires explicit written permission from the author.  

📄 License text: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)  
✉️ Contact: keremkemik9@gmail.com

