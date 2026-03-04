# Changelog

All notable changes to the EEG-fMRI Direct Signal Voxelization Pipeline are documented here.

---

## [v5.1.0] — 2025 · "Frequency-Specific" (LEMON Dataset · Eyes-Open)

### New Features
- **Relative band power** (`hilbert_envelope_relative`): Replaces absolute Hilbert envelope. Normalizes each band's power by total broadband power, eliminating 1/f covariance artifacts that previously inflated cross-band correlations.
- **Rank-based Dice similarity**: Amplitude-invariant voxel overlap scoring. Eliminates sensitivity to broadband power fluctuations; particularly beneficial for gamma and delta bands where absolute amplitudes vary widely across subjects.
- **CLI argument parser** (`argparse`): Pipeline can now be invoked from the command line with flags (e.g., `--no-ar1`, `--no-loso`, subject filtering).
- **Per-subject QC CSV export**: Quality-control metrics written to structured `.csv` files alongside NIfTI outputs for downstream statistical review.
- **Thread-safe Logger** (`threading.RLock`): Logger class upgraded with reentrant lock for safe parallel logging across LOSO folds.

### Bug Fixes
- **Gamma spatial prior corrected**: Was mistakenly inheriting the alpha spatial prior (`center_y = -75`). Now uses a frequency-appropriate dual-peak prior (posterior visual + frontal cognitive). Affects all gamma-band voxel maps.
- **HRF variance scaling removed**: Post-convolution variance normalization was incorrectly applied inside the pipeline. CONN/SPM handle normalization internally; double-scaling has been removed. Output NIfTI volumes are now fully CONN/SPM/FreeSurfer-compatible.
- **HRF mmap edge padding added**: Memory-mapped HRF convolution path was missing edge padding, causing boundary artifacts in chunked z-slice processing. Fixed.

### Changes
- **LOSO targets recalibrated**: All literature-derived validation targets (posterior/anterior alpha ratio, sensorimotor beta, frontal-midline theta, gamma posterior emphasis, anterior/posterior delta) recalibrated for the relative power scale. Absolute-power targets from v3.x/v4.x are no longer applicable.
- **Dataset**: SPIS (N=10, eyes-closed) → **LEMON (N=40, eyes-open)**. Subject list updated accordingly (`sub-03230x_EO.mat` format).
- **Sampling rate**: 256 Hz → **250 Hz** (LEMON dataset hardware).
- **Cache version bumped to `v51`**: Incompatible with `cache_v40` and `cache_v50`. Old caches must be regenerated.
- **Additional imports**: `csv`, `argparse`, `threading`, `zscore`, `rankdata`, `gaussian_filter`, `interp1d`, `LinearRegression`, `shutil` added to support new features.
- **Codebase size**: ~2,890 → ~3,483 lines (+593 lines).

---

## [v3.4-dev] — 2024 · SPIS Pilot (Eyes-Closed)

### New Features
- **LOSO cross-validation** (`USE_LOSO = True`): Replaced group-level K-fold with Leave-One-Subject-Out optimization for subject-level generalizability. Parallel fold execution via `joblib`.
- **LOSO outlier detection**: Automatic exclusion of subjects with anomalous alpha posterior/anterior ratio or occipital coverage below configurable thresholds.

### Changes
- **Validation framework**: Group K-fold (v3.3) superseded by LOSO as default mode. K-fold retained as fallback (`USE_LOSO = False`).
- **Dataset**: SPIS (N=10, `S02–S11_restingPre_EC.mat`), eyes-closed resting state.
- **Cache directory**: `cache_v33`.

---

## [v3.3] — 2024 · Group K-fold Edition

### New Features
- **Group-level K-fold optimization**: Pooled MI/Dice computation across all subjects before weight search, ensuring consistent `alpha` weights.
- **Two-phase processing architecture**: Phase 1 computes and caches per-subject MI/Dice scores; Phase 2 pools data for group K-fold weight search; Phase 3 applies optimal weights and builds volumes.
- **Per-subject MI/Dice caching** (`CACHE_MI_DICE = True`): Avoids redundant recomputation on reruns.
- **Posterior-selective CSD alpha preservation** (`CSD_ALPHA_MODE = "posterior_selective"`): Retains alpha amplitude in posterior electrodes while applying full CSD to frontal channels.
- **ROI-aware sparsification**: Band-specific minimum voxel counts and regional coverage constraints (occipital, frontal) to prevent anatomically implausible sparse maps.

### Core Algorithm
- Hybrid MI–Dice scoring: `H = α · MI + (1 − α) · Dice`
- Geodesic Gaussian spatial weighting on gray-matter voxel grid (~200k voxels, MNI space)
- Hemisphere normalization with configurable cross-hemisphere leakage
- Raykill artifact cleanup (minimum cluster size enforcement)
- Frequency bands: δ (1–4), θ (4–8), α (8–13), β (13–30), γ (30–45 Hz)

---

## Version Summary

| Version | Dataset | N | Condition | Key Method | Cache |
|---------|---------|---|-----------|------------|-------|
| v3.3 | SPIS | 10 | Eyes-Closed | Group K-fold | cache_v33 |
| v3.4-dev | SPIS | 10 | Eyes-Closed | LOSO (pilot) | cache_v33 |
| v5.0.0 | LEMON | 40 | Eyes-Open | LOSO + absolute power | cache_v50 |
| v5.1.0 | LEMON | 40 | Eyes-Open | LOSO + relative power + rank Dice | cache_v51 |
