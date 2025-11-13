EEG-fMRI Direct Signal Voxelization (v3.4)

A validated research pipeline for projecting EEG scalp signals into voxel-level 3D anatomical space, enabling frequency-resolved functional connectivity mapping across five canonical EEG bands (δ, θ, α, β, γ).

Developed by Dr. Kerem Kemik (GitHub: keremkem)
in collaboration with Dr. Cansu Aykaç (İzmir, Türkiye)


---

Overview

This repository presents the complete methodological workflow and validation framework for the EEG-to-voxel direct signal projection pipeline (v3.4).
The system bridges electrophysiology and hemodynamics by mapping EEG-derived oscillatory activity onto the MNI cortical surface using a Mutual-Information + Dice hybrid scoring model.

This pipeline is designed exclusively for research and methodological development and is not intended for clinical or commercial use.

## 🛡️ Intellectual Property & Patent Notice

**Core Methodology:** MI-Dice Hybrid Voxelization Framework  
**Inventors:** Dr. Kerem Kemik (primary) & Dr. Cansu Aykaç  
**Prior Art Established:** January 2025  

### Usage Rights:
| Use Case | Status | Contact Required |
|----------|--------|------------------|
| 🎓 Academic Research | ✅ Permitted | Citation required |
| 🏥 Clinical Research (non-commercial) | ✅ Permitted | Citation required |
| 🏢 Commercial Applications | ⚠️ Restricted | keremkemik9@gmail.com |
| 🎯 Defense/Military Systems | ⚠️ Restricted | keremkemik9@gmail.com |
| 📱 Medical Device Integration | ⚠️ Restricted | keremkemik9@gmail.com |
| 🔬 Patent Claims on Derivatives | ❌ Prohibited | Prior art established |

**Patent Status:** Methods and apparatus for information-theoretic spatial inference from distributed sensors are subject to pending intellectual property protections.

**License:** CC BY-NC 4.0 (code) + Proprietary methodological rights (inventors)

📧 Licensing inquiries: keremkemik9@gmail.com

---

Key Features

Frequency-specific voxelization (1–45 Hz; δ/θ/α/β/γ)

LOSO (Leave-One-Subject-Out) cross-validation

MI–Dice hybrid spatial correspondence weighting

ICC-based test–retest reliability analysis

Seven-tier circularity-control framework (SM3)

Robust reproducibility backbone for EEG-to-fMRI translation research



---

Conceptual Note

The statistical framework implemented here—particularly the MI–Dice hybridization—enables the definition of complex forms whose underlying structure can be inferred without direct interaction between multiple sensors and the object itself.
This approach allows latent shapes or fields (e.g., a quantum sphere, a radar interaction volume, or abstract energy distributions) to be characterized through information-theoretic correspondences and spatial-consistency constraints, rather than physical contact.


---

Conceptual Applications

Although the present pipeline is applied to EEG-to-brain mapping, the underlying methodology generalizes to a broader class of problems:

Radar or sonar field reconstruction: inferring volumetric interaction zones from sparse multi-sensor correlations

Quantum-field approximations: estimating latent geometric structures using non-linear information metrics

Electromagnetic or acoustic shape inference: reconstructing hidden volumetric forms from distributed measurements

Spatial coherence modeling in complex systems: identifying topologies where physical contact is limited or impossible


The MI–Dice hybridization supplies the shape-defining signal, while LOSO generalization ensures the shape is real, not an artifact of any specific sensor or subject.


---

Citation

If you use this repository, codebase, or methodology in academic work, please cite:

Kemik, K.,Aykac, C.,et al. (2025).
EEG-Derived Volumetric Connectivity: Establishing Frequency-Specific Reliability Baselines for Clinical Biomarker Development.
(Under review)

BibTeX

@article{KEMİK2025eegvoxel,
  author    = {Cansu Aykaç and Kerem Kemik and collaborators},
  THIS PART WILL BE UPDATED UPON PUBLICATION
  title     = {EEG-Derived Volumetric Connectivity: Establishing Frequency-Specific Reliability Baselines for Clinical Biomarker Development},
  year      = {2025},
  note      = {Under review}
}


---

Method Summary

Dataset

SPIS Resting-State EEG Dataset (Torkamani-Azar et al., Sabancı University, 2020)

N = 10 healthy adults

BioSemi ActiveTwo, 64 electrodes (10–10 montage)


Preprocessing

1–45 Hz bandpass, 50 Hz notch

Infomax ICA (FastICA; scikit-learn)

Alpha-protection algorithm

CSD transformation


Voxelization

~200,000 gray matter voxels (Harvard–Oxford mask)

Geodesic Gaussian weighting (σ = 22 mm)

Hemisphere isolation + normalization

MI–Dice hybrid scoring


Validation

7-layer circularity-control framework

ICC (2,k) reliability analyses

LOSO generalization testing



---

File Structure

/src        → Core voxelization scripts (Python)
/data       → Sample EEG inputs (.mat)
/results    → ICC, ROI, seed-to-voxel outputs
/docs       → Supplementary methods (SM1–SM3)
LICENSE     → Non-Commercial Research License
README.md   → Documentation & usage policy


---

License & Usage Policy

This repository is licensed under the:

Non-Commercial Research and Educational Use License (NCR-EUL)

© 2025 Kerem Kemik & Cansu Aykaç

 Allowed

Non-commercial academic and educational use

Modification and extension for research

Publication with proper attribution


 Not Allowed

Commercial use

Diagnostic or clinical applications

Proprietary or monetized software integration

Redistribution without attribution and license retention


Attribution Requirement

Use of this repository must cite:

“EegDirectSignalVoxelization — Kemik & Aykaç (2025)”

Full License

See the LICENSE file.

Contact

keremkemik9@gmail.com
GitHub: keremkem


---

Acknowledgments

This work builds upon the SPIS Dataset (Sabancı University, Torkamani-Azar et al., 2020) and open-source EEG processing frameworks.
Also working on LEMON database for EEG fMRI validity. They will be published after successfull analysis. 
This code was established with claude sonnet 4.5 vibecoding feature. 

> “This code serves as a quantitative bridge between electric silence and hemodynamic rhythm—built for science, not for profit.”- MI–Dice hybrid weighting for spatial correspondence
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
Claude 4.5 Sonnet was used for vibecoding.

“This code is a bridge between electric silence and hemodynamic rhythm—designed only for research, not for profit.”

## License
This repository is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.  
The code and data are intended **solely for academic and research purposes**.  
Commercial or clinical use requires explicit written permission from the author.  

📄 License text: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)  
✉️ Contact: keremkemik9@gmail.com

