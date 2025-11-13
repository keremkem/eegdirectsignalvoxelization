# EEG-fMRI Direct Signal Voxelization (v3.4)

> A computational pipeline for projecting EEG scalp signals onto voxel-level 3D brain space using hybrid information-theoretic methods.

(https://github.com/keremkem/eegdirectsignalvoxelization)
(https://www.python.org/)
(https://creativecommons.org/licenses/by-nc/4.0/)
**Developed by:** Dr. Kerem Kemik [keremkem](https://github.com/keremkem)  
**In collaboration with:** Dr. Cansu Aykaç (İzmir, Türkiye)
---
## 📊 Overview
This repository implements a validated research pipeline for EEG-fMRI spatial integration, enabling frequency-resolved functional connectivity mapping across canonical EEG bands (δ, θ, α, β, γ).

The framework bridges electrophysiology and hemodynamics by mapping EEG-derived oscillatory activity onto MNI cortical space using a **Mutual Information + Dice hybrid scoring** approach with **Leave-One-Subject-Out (LOSO)** cross-validation.
**Status:** Methodology complete. Large-scale validation (LEMON, N=40) in progress.
---
## 🔬 Key Features
- **Frequency-specific voxelization** (1–45 Hz across 5 canonical bands)
- **MI-Dice hybrid scoring** (information-theoretic + spatial consistency)
- **LOSO cross-validation** (subject-level generalization testing)
- **ICC reliability analysis** (test-retest validation)
- **7-tier circularity control** (spatial autocorrelation < 0.35)
- **Modular Python implementation** (NumPy, SciPy, scikit-learn, MNE)
---
## 🛠️ Method Summary

### Datasets

**SPIS** (Torkamani-Azar et al., Sabancı University, 2020)
- N = 10 healthy adults
- Status: ✅ Analysis complete
- Purpose: Pilot validation, circularity control (r = 0.33)

**LEMON** (Babayan et al., Max Planck Institute, 2019)  
- N = 40 subjects (selected from 227)
- Status: ⏳ Validation in progress
- Purpose: Cross-dataset generalization, large-scale reliability testing
### Pipeline
1. **Preprocessing**
   - Bandpass: 1–45 Hz, notch: 50 Hz
   - ICA artifact removal (FastICA)
   - Alpha-protection algorithm (occipital rhythm preservation)
   - Current Source Density (CSD) referencing

2. **Voxelization**
   - ~200,000 gray matter voxels (Harvard-Oxford atlas)
   - Geodesic Gaussian weighting (σ = 22 mm)
   - Hemisphere isolation + normalization
   - **MI-Dice hybrid scoring:** α·MI(sensor→voxel) + β·Dice(sensor→voxel)

3. **Validation**
   - LOSO generalization (per-subject dropout)
   - ICC(2,k) reliability analysis
   - Circularity bias assessment (r < 0.35)
   - 7-tier validation framework
---
## 📁 Repository Structure
├── src/ # Core voxelization algorithms (Python)
├── utils/ # Helper functions, preprocessing tools
├── data/ # Example data (MATLAB .mat format)
├── results/ # Output templates (ICC maps, seed-to-voxel)
├── docs/ # Supplementary methods (SM1-SM3)
├── LICENSE # CC BY-NC 4.0
├── CITATION.cff # Citation metadata
└── README.md # This file
---
## 📖 Citation
If you use this code or methodology, please cite:
Kemik, K., Aykaç, C., et al. (2025).
EEG-Derived Volumetric Connectivity: Establishing
Frequency-Specific Reliability Baselines for Clinical
Biomarker Development. [Manuscript in preparation]

**BibTeX:**
```bibtex
@software{Kemik2025voxelization,
  author    = {Kerem Kemik and Cansu Aykaç},
  title     = {EEG-fMRI Direct Signal Voxelization Pipeline},
  year      = {2025},
  version   = {3.4},
  url       = {https://github.com/keremkem/eegdirectsignalvoxelization},
  note      = {Manuscript in preparation}
}
⚖️ License & Usage
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

✅ Permitted Uses:
Academic and educational research
Non-commercial neuroscience studies
Methodological replication and extension
Derivative works with attribution
⚠️ Restricted Uses:
Commercial applications (requires licensing agreement)
Medical diagnostic devices (requires regulatory approval)
Proprietary software integration
🔐 Intellectual Property:
This work establishes prior art for the MI-Dice-LOSO methodological combination (November 2025). Authors retain rights to commercial licensing and derivative patents.

For commercial inquiries: keremkemik9@gmail.com

🧠 Technical Details
Core Innovation: MI-Dice Hybrid Framework with LOSO validation.
Traditional approaches rely on single metrics:

Source localization methods (LORETA/sLORETA): Inverse modeling with dipole assumptions
Coherence-based methods: Sensor-space connectivity (no voxel projection)
Spatial overlap only: No information-theoretic weighting
Our approach combines:

Mutual Information (MI): Information-theoretic correspondence
Dice Coefficient: Spatial overlap consistency
LOSO Validation: Subject-level generalization testing
This hybrid framework provides robust spatial inference without inverse modeling assumptions.

🤝 Acknowledgments
SPIS Dataset: Torkamani-Azar et al., Sabancı University (2020)
LEMON Dataset: Babayan et al., Max Planck Institute (2019)
Open-source tools: MNE-Python, scikit-learn, nibabel
Development assistance: Claude Sonnet 4.5 (Anthropic) for methodology validation and code optimization
📧 Contact
Dr. Kerem Kemik
📧 keremkemik9@gmail.com
🔗 GitHub: keremkem

📜 Development Notes
Timeline:

January 2025: Initial development 
July 2025: MI-Dice hybrid framework
September 2025: LOSO breakthrough (mature validation)
October 2025: Research documentation complete
November 2025: Public release, LEMON analysis ongoing
Future Updates:
 LEMON validation complete (Q4 2025)
 Manuscript submission (Q1 2026)
 Tutorial notebooks & documentation
 Community contributions welcome (contact first)
"This code serves as a quantitative bridge between electric silence and hemodynamic rhythm—built for science, not for profit."
© 2025 Dr. Kerem Kemik & Dr. Cansu Aykaç | CC BY-NC 4.0 License
