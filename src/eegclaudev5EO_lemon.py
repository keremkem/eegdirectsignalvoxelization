#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-to-fMRI Voxel Projection Pipeline v5.1.0 "Frequency-Specific"
====================================================================
v5.1.0 Changes from v5.0.0:
  - NEW: Relative band power (hilbert_envelope_relative) — eliminates
         broadband 1/f covariance across frequency bands
  - NEW: Rank-based Dice similarity — amplitude-invariant
  - FIX: Gamma spatial prior corrected (was identical to alpha)
  - FIX: HRF variance scaling removed (CONN handles normalization)
  - FIX: HRF mmap edge padding added (was missing)
  - UPDATED: LOSO targets recalibrated for relative power
  - UPDATED: Cache version bumped to v51 (incompatible with v50/v40)

All tunable parameters are optimized per-dataset via Phase 2 LOSO.
Literature-derived targets remain fixed as validation anchors.

Output: SPM/CONN/FreeSurfer-compatible NIfTI volumes.
"""

import os
import sys
import gc
import csv
import time
import warnings
import hashlib
import json
import pickle
import argparse
import re
import threading
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch, hilbert, welch
from scipy.spatial.distance import cdist
from scipy.ndimage import label, uniform_filter1d, gaussian_filter
from scipy.stats import kurtosis, zscore, gamma as gamma_dist, rankdata
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
import nibabel as nib
from nilearn import image, datasets

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION v5.1.0 — FREQUENCY-SPECIFIC
# ============================================================================

class Config:
    """
    Pipeline configuration v5.1.0
    
    Key changes from v5.0.0:
      - Relative band power replaces absolute Hilbert envelope
      - Gamma spatial prior corrected
      - LOSO targets recalibrated for relative power scale
      - Cache version bumped (v51)
    """

    # ── Paths ──────────────────────────────────────────────────
    DATA_PATH = r"C:\Users\kerem\Downloads\eegtest"
    
    SUBJECTS = [
    "sub-032301_EO.mat", "sub-032302_EO.mat", "sub-032303_EO.mat",
    "sub-032304_EO.mat", "sub-032305_EO.mat", "sub-032306_EO.mat",
    "sub-032307_EO.mat", "sub-032308_EO.mat", "sub-032310_EO.mat",
    "sub-032311_EO.mat", "sub-032312_EO.mat", "sub-032313_EO.mat",
    "sub-032314_EO.mat", "sub-032315_EO.mat", "sub-032316_EO.mat",
    "sub-032317_EO.mat", "sub-032318_EO.mat", "sub-032319_EO.mat",
    "sub-032320_EO.mat", "sub-032321_EO.mat", "sub-032322_EO.mat",
    "sub-032323_EO.mat", "sub-032324_EO.mat", "sub-032325_EO.mat",
    "sub-032326_EO.mat", "sub-032327_EO.mat", "sub-032328_EO.mat",
    "sub-032329_EO.mat", "sub-032330_EO.mat", "sub-032331_EO.mat",
    "sub-032332_EO.mat", "sub-032333_EO.mat", "sub-032334_EO.mat",
    "sub-032336_EO.mat", "sub-032337_EO.mat", "sub-032338_EO.mat",
    "sub-032339_EO.mat", "sub-032340_EO.mat", "sub-032341_EO.mat",
    "sub-032342_EO.mat",
]

    CACHE_DIR = "cache_v51"           # NEW: incompatible with v50/v40
    VERSION_TAG = "v5.1.0"

    # ── FIXED: Hardware / Experiment ───────────────────────────
    FS = 250
    SEGMENT_DURATION = 2.0  # seconds, pseudo-TR

    BANDS = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 80),
    }

    # ── FIXED: HRF Canonical Parameters (Glover 1999) ─────────
    HRF_A1 = 6.0
    HRF_A2 = 16.0
    HRF_B1 = 1.0
    HRF_B2 = 1.0
    HRF_C = 1.0 / 6.0
    HRF_LENGTH = 32.0

    # ── FIXED: LOSO Targets (Literature) ───────────────────────
    # RECALIBRATED for relative band power (v5.1.0)
    # Relative power values are typically 0.0–1.0 per band
    # Ratios computed on relative power are closer to 1.0
    LOSO_TARGETS = {
        "alpha": {
            "type": "ratio",
            "target_value": 0.9,           # EO: alpha attenuated, P/A ~0.5-1.0 (Barry et al. 2007 EO)
            "target_range": (0.4, 1.5),    # EO-appropriate; EC refs (2.0-6.0) do not apply
            "confidence": "MEDIUM",        # Downgraded: EO alpha is state-dependent
            "description": "Posterior/Anterior relative alpha ratio (EO)",
            "reference": "Barry et al. 2007 (EO); Klimesch 1999 (EO-adapted)",
            "regions": {
                "numerator":   {"y_max": -40},
                "denominator": {"y_min": 0},
            },
        },
        "beta": {
            "type": "proportion",
            "target_value": 0.22,           # was 0.18
            "target_range": (0.12, 0.40),
            "confidence": "MEDIUM",
            "description": "Sensorimotor cortex relative beta proportion",
            "reference": "Pfurtscheller & Lopes da Silva 1999",
            "region": {
                "y_min": -45, "y_max": 15,
                "z_min": 40,
                "abs_x_min": 10, "abs_x_max": 55,
            },
        },
        "theta": {
            "type": "ratio",
            "target_value": 1.4,            # was 1.2
            "target_range": (0.9, 2.2),
            "confidence": "LOW-MEDIUM",
            "description": "Frontal-midline / lateral relative theta ratio",
            "reference": "Mitchell et al. 2008; Scheeringa et al. 2008",
            "regions": {
                "numerator":   {"abs_x_max": 15, "y_min": 10, "y_max": 60,
                                "z_min": 15, "z_max": 65},
                "denominator": {"abs_x_min": 25, "y_min": 10, "y_max": 60,
                                "z_min": 5},
            },
        },
        "gamma": {
            "type": "proportion_ratio",
            "target_value": 1.8,            # was 2.5
            "target_range": (0.8, 4.0),
            "confidence": "LOW",
            "description": "Posterior emphasis relative gamma ratio",
            "reference": "Fries 2009; Hoogenboom et al. 2006",
            "region": {"y_max": -60, "z_max": 40},
        },
        "delta": {
            "type": "ratio",
            "target_value": 0.85,           # was 0.75
            "target_range": (0.5, 1.4),
            "confidence": "MEDIUM",
            "description": "Anterior/Posterior relative delta ratio",
            "reference": "Harmony et al. 1996",
            "use_mean": True,
            "regions": {
                "numerator":   {"y_min": 0},
                "denominator": {"y_max": -40},
            },
        },
    }

    # ── DEFAULTS: LOSO-L1 Search Centers (per-band) ───────────
    L1_DEFAULTS = {
        "delta": {
            "mi_weight": 0.45, "contrast": 0.90, "boost": 1.7,
            "penalty": 0.8, "softening": 0.40,
            "keep_top_pct": 0.85, "smoothing_fwhm": 1.0,
        },
        "theta": {
            "mi_weight": 0.65, "contrast": 0.85, "boost": 1.6,
            "penalty": 0.7, "softening": 0.55,
            "keep_top_pct": 0.90, "smoothing_fwhm": 1.0,
        },
        "alpha": {
            "mi_weight": 0.65, "contrast": 0.50, "boost": 1.3,
            "penalty": 0.0, "softening": 0.70,  # EO: high softening = data-driven, prior is soft guide only
            "keep_top_pct": 0.90, "smoothing_fwhm": 1.5,
        },
        "beta": {
            "mi_weight": 0.55, "contrast": 0.65, "boost": 1.4,
            "penalty": 0.5, "softening": 0.45,
            "keep_top_pct": 0.85, "smoothing_fwhm": 1.0,
        },
        "gamma": {
            "mi_weight": 0.40, "contrast": 0.90, "boost": 1.2,
            "penalty": 0.3, "softening": 0.70,
            "keep_top_pct": 0.80, "smoothing_fwhm": 1.0,
        },
    }
    
    SIGMA_BASE = 35.0
    CSD_SIGMA = 30.0
    HEMISPHERE_ISOLATION = 0.85
    CROSS_HEMISPHERE_WEIGHT = 0.25
    
    # ── DEFAULTS: LOSO-L2 Search Centers (global) ─────────────
    L2_DEFAULTS = {
        "raykill_min_cluster_mm3": 25.0,
        "hrf_snr": 100.0,
        "temporal_window": 3,
        "ar1_enabled": True,
    }

    # ── LOSO-L1 Search Bounds ─────────────────────────────────
    L1_BOUNDS = {
        "mi_weight":      (0.25, 0.95),
        "contrast":       (0.10, 1.20),
        "boost":          (1.0, 3.0),
        "penalty":        (0.0, 1.0),
        "softening":      (0.1, 1.0),
        "keep_top_pct":   (0.60, 0.98),
        "smoothing_fwhm": (0.0, 3.0),
    }

    # ── Per-band L1 Bounds Overrides ─────────────────────────
    # Overrides L1_BOUNDS for specific bands/params.
    # Alpha: softening floor raised to 0.50 — prevents optimizer from
    # collapsing to near-zero softening which makes the EC-designed
    # posterior gaussian prior dominate EO data (r_raw_prior dropped to 0.38).
    # boost ceiling lowered to 1.8 — reduces prior amplification.
    L1_BOUNDS_OVERRIDE = {
        "alpha": {
            "softening":      (0.50, 1.0),   # EO: prior must remain soft
            "boost":          (1.0, 1.8),    # EO: limit posterior boosting
        },
    }

    # ── LOSO-L2 Search Bounds ─────────────────────────────────
    L2_BOUNDS = {
        "raykill_min_cluster_mm3": (10.0, 80.0),
        "hrf_snr": (10.0, 500.0),
        "temporal_window": (1, 7),
    }

    # ── LOSO Search Settings ──────────────────────────────────
    LOSO_N_CANDIDATES_L1 = 20
    LOSO_N_CANDIDATES_L2 = 15
    LOSO_MAX_REFINEMENT = 3
    LOSO_CONVERGENCE_TOL = 0.001
    LOSO_STABILITY_THRESHOLD = 0.15
    KFOLD_N_SPLITS = 3

    # ── MI Weight Caps (safety bounds from literature) ────────
    MI_WEIGHT_CAPS = {
        "alpha": (0.35, 0.92),
        "beta":  (0.35, 0.65),
        "theta": (0.35, 0.75),
        "gamma": (0.35, 0.75),
        "delta": (0.30, 0.70),
    }

    # ── Spatial Prior Type Definitions ─────────────────────────
    SPATIAL_PRIOR_STRUCTURE = {
        "alpha": {"type": "gaussian", "param_keys": ["center_y", "sigma_y"]},
        "beta":  {"type": "band_z",   "param_keys": ["y_center", "y_sigma",
                  "z_center", "z_sigma", "abs_x_center", "abs_x_sigma"]},
        "theta": {"type": "midline_frontal", "param_keys": ["y_center",
                  "y_sigma", "abs_x_sigma", "z_center", "z_sigma"]},
        "delta": {"type": "frontal_broad", "param_keys": ["y_center", "y_sigma"]},
        "gamma": {"type": "dual_peak", "param_keys": ["center_y_post",     # CHANGED v5.1
                  "sigma_y_post", "center_y_front", "sigma_y_front"]},
    }
    
    # Spatial prior fixed centers (from literature, not optimized)
    # v5.1.0: Gamma corrected — was identical to alpha (center_y=-75)
    SPATIAL_PRIOR_FIXED = {
        "alpha": {"center_y": -40, "sigma_y": 80, "min_weight": 0.60},  # EO: flatter prior, softer posterior bias
        "beta":  {"y_center": -15, "y_sigma": 35, "z_center": 50,
                  "z_sigma": 25, "abs_x_center": 40, "abs_x_sigma": 25,
                  "min_weight": 0.30, "penalty_y_center": -15,
                  "penalty_y_sigma": 20},
        "theta": {"y_center": 35, "y_sigma": 25, "abs_x_sigma": 15,
                  "z_center": 45, "z_sigma": 20, "min_weight": 0.25,
                  "penalty_y_center": -15, "penalty_y_sigma": 30},
        "delta": {"y_center": 40, "y_sigma": 40, "min_weight": 0.25,
                  "penalty_y_center": -15, "penalty_y_sigma": 30},
        # v5.1.0 FIX: Gamma uses dual-peak prior (posterior visual + frontal cognitive)
        # Was: {"center_y": -75, "sigma_y": 40, "min_weight": 0.20}
        "gamma": {"center_y_post": -70, "sigma_y_post": 35,
                  "center_y_front": 25, "sigma_y_front": 30,
                  "post_weight": 0.5, "front_weight": 0.5,
                  "min_weight": 0.35},
    }

    # ── Band-Dependent Sigma Scaling ──────────────────────────
    BAND_SIGMA_SCALE = {
        "delta": 1.15, "theta": 1.05, "alpha": 1.00,
        "beta": 0.90, "gamma": 0.80,
    }

    # ── ICA Settings ──────────────────────────────────────────
    ICA_MODE = "standard"
    ICA_N_COMPONENTS = 62
    ICA_MAX_REMOVE = {"conservative": 5, "standard": 8, "aggressive": 12}

    # ── CSD Alpha Preservation ────────────────────────────────
    CSD_PRESERVE_ALPHA = True
    CSD_ALPHA_MODE = "posterior_selective"
    CSD_POSTERIOR_Y_THRESHOLD = -40.0
    CSD_FRONTAL_Y_THRESHOLD = 20.0
    CSD_CENTRAL_ALPHA_RETENTION = 0.5
    CSD_CENTRAL_BETA_RETENTION = 0.3

    # ── Grid ──────────────────────────────────────────────────
    GRID_SPACING = 2.0
    GRID_BOUNDS = {"x": (-90, 91), "y": (-130, 91), "z": (-72, 109)}
    GM_INCLUDE_SUBCORTICAL = False

    # ── Region Multipliers ────────────────────────────────────
    SIGMA_MULTIPLIERS = {
        "occipital": 1.5, "posterior": 1.3, "parietal": 1.2,
        "central": 1.0, "frontal": 1.1,
    }

    # ── Midline Settings ──────────────────────────────────────
    MIDLINE_WEIGHT = 1.0
    MIDLINE_GAP_MM = 3.0

    # ── Raykill Defaults ──────────────────────────────────────
    RAYKILL_ENABLE = True
    RAYKILL_KEEP_FLOOR_PCT = 0.50
    RAYKILL_KEEP_FLOOR_ABS = 25000

    # ── Z-Score ───────────────────────────────────────────────
    APPLY_ZSCORE = True
    ZSCORE_MODE = "global"

    # ── MI/Dice ───────────────────────────────────────────────
    MI_N_BINS = 4

    # ── Temporal Filtering ────────────────────────────────────
    TEMPORAL_FILTERING = True
    TEMPORAL_MIN_FRAC = 0.3

    # ── Frontal Alpha Protection ──────────────────────────────
    ALPHA_PA_MAX_RATIO = 3.0          # EO: posterior alpha suppressed, max P/A ~1.5-2.0
    FRONTAL_ALPHA_MIN_COVERAGE = 0.0
    FRONTAL_ALPHA_MIN_VOXELS = 2000
    FRONTAL_ALPHA_MIN_VALUE = 0.05

    # ── Min Voxels ────────────────────────────────────────────
    MIN_VOXELS_GLOBAL = {
        "delta": 15000, "theta": 15000, "alpha": 20000,
        "beta": 25000, "gamma": 18000,
    }

    # ── LOSO Outlier Detection ────────────────────────────────
    LOSO_EXCLUDE_OUTLIERS = True
    LOSO_OUTLIER_ALPHA_PA_MIN = 0.3   # EO: P/A ~0.5-1.0, don't exclude normal EO subjects
    LOSO_OUTLIER_ALPHA_OCC_MIN = 3.0  # EO: occipital coverage also lower

    # ── Phase 3 Memory / IO ───────────────────────────────────
    PHASE3_USE_MMAP = True
    PHASE3_MMAP_THRESHOLD_GB = 2.0
    PHASE3_CHUNK_SIZE = 2000
    PHASE3_ENABLE_CHECKPOINT = True

    # ── AR(1) Prewhitening ────────────────────────────────────
    AR1_MAX_ITER = 100

    # ── Parallelism ───────────────────────────────────────────
    N_JOBS = min(8, cpu_count())
    LOSO_N_JOBS = 2
    BATCH_SIZE_SIGNATURES = 5000
    RANDOM_SEED = 42

    # ── Cache ─────────────────────────────────────────────────
    CACHE_MI_DICE = True


# ============================================================================
# THREAD-SAFE LOGGER
# ============================================================================

class Logger:
    _log_file = None
    _lock = threading.RLock()

    @classmethod
    def open_log(cls, path):
        with cls._lock:
            cls._close_log_internal()
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    if Path(path).exists():
                        time.sleep(0.1 * (attempt + 1))
                    cls._log_file = open(path, 'w', encoding='utf-8', buffering=1)
                    break
                except PermissionError:
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)
                    else:
                        cls._log_file = None
                        print(f"[Logger] File logging disabled: {path}")
                        return
        cls.info(f"Log opened: {path}")

    @classmethod
    def _close_log_internal(cls):
        if cls._log_file is not None:
            try:
                cls._log_file.flush()
                cls._log_file.close()
            except Exception:
                pass
            cls._log_file = None

    @classmethod
    def close_log(cls):
        with cls._lock:
            cls._close_log_internal()

    @staticmethod
    def _write(line):
        with Logger._lock:
            print(line)
            if Logger._log_file is not None:
                try:
                    Logger._log_file.write(line + '\n')
                except Exception:
                    pass

    @staticmethod
    def info(msg):
        Logger._write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    @staticmethod
    def warn(msg):
        Logger.info(f"⚠️  {msg}")

    @staticmethod
    def error(msg):
        Logger.info(f"❌ {msg}")

    @staticmethod
    def success(msg):
        Logger.info(f"✅ {msg}")

    @staticmethod
    def section(title):
        Logger.info("\n" + "=" * 70)
        Logger.info(title)
        Logger.info("=" * 70)


# ============================================================================
# HRF CONVOLVER — v5.1.0 FIXED
# ============================================================================

class HRFConvolver:
    """
    HRF convolution: neural → pseudo-BOLD.
    
    v5.1.0 changes:
      - Variance scaling REMOVED (CONN handles normalization)
      - Mmap version now has proper edge padding
    """

    @staticmethod
    def canonical_hrf(tr, length=32.0, config=None):
        if config is not None:
            a1, a2 = config.HRF_A1, config.HRF_A2
            b1, b2 = config.HRF_B1, config.HRF_B2
            c = config.HRF_C
        else:
            a1, a2, b1, b2, c = 6.0, 16.0, 1.0, 1.0, 1.0 / 6.0

        t = np.arange(0, length, tr)
        g1 = gamma_dist.pdf(t, a1, scale=b1)
        g2 = gamma_dist.pdf(t, a2, scale=b2)
        hrf = g1 - c * g2

        hrf_sum = np.sum(hrf)
        if abs(hrf_sum) > 1e-12:
            hrf = hrf / hrf_sum
        else:
            hrf = hrf / (np.max(np.abs(hrf)) + 1e-12)
            Logger.warn("    [HRF] Near-zero net integral, peak-normalized")

        return hrf.astype(np.float32)

    @staticmethod
    def convolve_volume(volume_4d, tr, config=None):
        """
        HRF convolution — v5.1.0: no variance scaling.
        CONN/SPM perform their own normalization.
        """
        x, y, z, T = volume_4d.shape
        hrf = HRFConvolver.canonical_hrf(tr, config=config)

        Logger.info(f"    [HRF-Conv] kernel={len(hrf)} samples, TR={tr}s")
        Logger.info(f"    [HRF-Conv] kernel sum={np.sum(hrf):.4f}, "
                    f"peak={np.max(hrf):.4f}, undershoot={np.min(hrf):.4f}")

        active_3d = np.any(volume_4d != 0, axis=3)
        n_active = int(active_3d.sum())

        if n_active == 0:
            Logger.warn("    [HRF-Conv] No active voxels!")
            return volume_4d

        vol_2d = volume_4d.reshape(-1, T)
        active_1d = active_3d.reshape(-1)
        active_indices = np.where(active_1d)[0]

        Logger.info(f"    [HRF-Conv] Convolving {n_active} voxels...")

        pad_len = len(hrf) - 1
    
        chunk_size = 5000
        for start in tqdm(range(0, len(active_indices), chunk_size),
                         desc="    HRF conv", ncols=80, leave=False):
            end = min(start + chunk_size, len(active_indices))
            idx = active_indices[start:end]
            for i in idx:
                signal = vol_2d[i]
                padded = np.pad(signal, (pad_len, 0), mode='edge')
                convolved = np.convolve(padded, hrf, mode='valid')
                vol_2d[i] = convolved

        bold_volume = vol_2d.reshape(x, y, z, T)

        # v5.1.0: NO variance scaling — CONN handles normalization
        Logger.info("    [HRF-Conv] No variance scaling (CONN-compatible)")

        Logger.success(f"    [HRF-Conv] Range: [{bold_volume.min():.4f}, "
                     f"{bold_volume.max():.4f}]")
        return bold_volume.astype(np.float32)

    @staticmethod
    def convolve_volume_mmap(volume_mmap, tr, config=None):
        """
        HRF convolution for memory-mapped volumes.
        v5.1.0: Added edge padding (was missing), removed variance scaling.
        """
        x, y, z, T = volume_mmap.shape
        hrf = HRFConvolver.canonical_hrf(tr, config=config)
        pad_len = len(hrf) - 1

        Logger.info(f"    [HRF-Conv] Memmap mode (TR={tr}s)")

        chunk_z = max(1, z // 10)
        for z_start in tqdm(range(0, z, chunk_z),
                           desc="    HRF conv (mmap)", ncols=80, leave=False):
            z_end = min(z_start + chunk_z, z)
            chunk = np.array(volume_mmap[:, :, z_start:z_end, :])
            cx, cy, cz, ct = chunk.shape
            chunk_2d = chunk.reshape(-1, ct)

            active = np.any(chunk_2d != 0, axis=1)
            for i in np.where(active)[0]:
                signal = chunk_2d[i]
                # v5.1.0 FIX: Edge padding (was missing in v5.0 mmap path)
                padded = np.pad(signal, (pad_len, 0), mode='edge')
                convolved = np.convolve(padded, hrf, mode='valid')
                chunk_2d[i] = convolved

            volume_mmap[:, :, z_start:z_end, :] = chunk_2d.reshape(cx, cy, cz, ct)
            volume_mmap.flush()

        # v5.1.0: NO variance scaling
        Logger.info("    [HRF-Conv] No variance scaling (CONN-compatible)")
        Logger.success("    [HRF-Conv] Memmap convolution complete")
        return volume_mmap


# ============================================================================
# COORDINATE SYSTEM
# ============================================================================

class CoordinateSystem:
    """MNI coordinate mapping for 64-channel EEG."""
    
    RAW_COORDS = """Fp1 -26 87 -18
Fp2 26 87 -18
Fpz 0 96 -15
AF7 -44 69 -18
AF8 44 69 -18
AF3 -32 70 20
AF4 32 70 20
AFz 0 75 24
F7 -63 38 -10
F8 63 38 -10
F5 -61 44 12
F6 61 44 12
F3 -44 56 30
F4 44 56 30
F1 -21 63 41
F2 21 63 41
Fz 0 63 39
FT7 -69 21 -16
FT8 69 21 -16
FC5 -58 22 25
FC6 58 22 25
FC3 -44 30 38
FC4 44 30 38
FC1 -23 32 52
FC2 23 32 52
FCz 0 33 48
T7 -69 -8 -9
T8 69 -8 -9
C5 -64 -1 27
C6 64 -1 27
C3 -43 0 51
C4 43 0 51
C1 -20 0 65
C2 20 0 65
Cz 0 0 67
CP5 -59 -25 21
CP6 59 -25 21
CP3 -43 -27 40
CP4 43 -27 40
CP1 -23 -23 56
CP2 23 -23 56
CPz 0 -34 53
P7 -62 -48 -5
P8 62 -48 -5
P5 -58 -47 20
P6 58 -47 20
P3 -41 -51 39
P4 41 -51 39
P1 -18 -54 55
P2 18 -54 55
Pz 0 -57 41
P9 -54 -54 -30
P10 54 -54 -30
PO7 -44 -69 -20
PO8 44 -69 -20
PO3 -31 -69 6
PO4 31 -69 6
POz 0 -70 26
O1 -24 -87 -12
O2 24 -87 -12
Oz 0 -85 12
Iz 0 -96 -8"""

    MANUAL_MNI_COORDS = {
        'Oz':  [0, -105, 15],  'O1':  [-27, -108, 10], 'O2': [27, -108, 10],
        'Iz':  [0, -112, 5],   'POz': [0, -95, 45],
        'PO3': [-32, -98, 35], 'PO4': [32, -98, 35],
        'PO7': [-42, -100, 25],'PO8': [42, -100, 25],
        'Pz':  [0, -68, 65],   'P1': [-20, -72, 60],  'P2': [20, -72, 60],
        'P3':  [-42, -72, 50], 'P4':  [42, -72, 50],
        'P5':  [-52, -65, 42], 'P6':  [52, -65, 42],
        'P7':  [-58, -62, 28], 'P8':  [58, -62, 28],
        'P9':  [-50, -70, 10], 'P10': [50, -70, 10],
        'CPz': [0, -35, 75],   'CP1': [-20, -35, 72], 'CP2': [20, -35, 72],
        'CP3': [-40, -35, 65], 'CP4': [40, -35, 65],
        'CP5': [-54, -32, 50], 'CP6': [54, -32, 50],
        'T7':  [-70, -20, -5], 'T8':  [70, -20, -5],
    }

    @classmethod
    def load_coordinates(cls):
        coords = {}
        order = []
        for line in cls.RAW_COORDS.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            name = parts[0]
            x_raw, y_raw, z_raw = float(parts[1]), float(parts[2]), float(parts[3])
            if name in cls.MANUAL_MNI_COORDS:
                coords[name] = np.array(cls.MANUAL_MNI_COORDS[name], dtype=np.float32)
            else:
                scale = 0.88
                coords[name] = np.array([
                    x_raw * scale,
                    y_raw * scale * 1.12 - 5.0,
                    z_raw * scale + 48.0,
                ], dtype=np.float32)
            order.append(name)
        return coords, order


# ============================================================================
# SIGNAL PROCESSING — v5.1.0: RELATIVE BAND POWER
# ============================================================================

class SignalProcessor:
    """
    EEG signal processing utilities.
    
    v5.1.0: Added hilbert_envelope_relative for broadband-normalized
    frequency-specific power estimation.
    """

    @staticmethod
    def bandpass(data, fs, low, high, order=4):
        nyq = fs / 2
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, data, axis=1)

    @staticmethod
    def notch(data, fs, freq=50.0, Q=30):
        b, a = iirnotch(freq / (fs / 2), Q=Q)
        return filtfilt(b, a, data, axis=1)

    @staticmethod
    def hilbert_envelope(segment, band, fs):
        """Original absolute power envelope (kept for backward compat)."""
        filtered = SignalProcessor.bandpass(segment, fs, band[0], band[1])
        analytic = hilbert(filtered, axis=1)
        return np.abs(analytic).mean(axis=1)

    @staticmethod
    def hilbert_envelope_relative(segment, band, fs):
        """
        v5.1.0 NEW: Relative band power — broadband-normalized.
        
        Eliminates 1/f broadband covariance that causes all frequency
        bands to show identical BOLD correlation patterns.
        
        relative_power = band_power / total_power
        
        Reference:
            Klimesch (1999): relative alpha power is more functionally
            specific than absolute power.
            
        Returns
        -------
        relative : np.ndarray (n_channels,)
            Relative band power per channel, values in [0, 1].
        """
        # Target band power
        filtered = SignalProcessor.bandpass(segment, fs, band[0], band[1])
        analytic = hilbert(filtered, axis=1)
        band_power = np.abs(analytic).mean(axis=1)  # (n_channels,)
        
        # Total broadband power (1 Hz to Nyquist-safe)
        nyq = fs / 2
        hi_total = min(80.0, nyq - 1)
        total_filtered = SignalProcessor.bandpass(segment, fs, 1.0, hi_total)
        total_analytic = hilbert(total_filtered, axis=1)
        total_power = np.abs(total_analytic).mean(axis=1)  # (n_channels,)
        
        # Relative power: band / total
        relative = band_power / (total_power + 1e-12)
        
        return relative.astype(np.float32)

    @staticmethod
    def bandpower_welch(signal, fs, fmin, fmax):
        f, Pxx = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
        idx = (f >= fmin) & (f <= fmax)
        return np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0


# ============================================================================
# ICA ARTIFACT REMOVAL — FIXED
# ============================================================================

class ICAArtifactRemover:
    """
    ICA-based artifact removal with band-specific component protection.
    """
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.DATA_PATH) / "ica_cache_v51"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_signature(self, eeg, fs):
        h = hashlib.sha1()
        h.update(np.asarray(eeg.shape, dtype=np.int32).tobytes())
        h.update(np.float32(fs).tobytes())
        sample_len = min(eeg.shape[1], int(fs * 10))
        h.update(np.ascontiguousarray(eeg[:, :sample_len]).astype(np.float32).tobytes())
        h.update(f"v510_{self.config.ICA_MODE}".encode())
        return h.hexdigest()[:16]

    def _extract_features(self, components, fs):
        n_comp = components.shape[0]
        features = {}
        for i in range(n_comp):
            sig = components[i]
            feat = {}
            feat['std'] = np.std(sig)
            feat['kurtosis'] = kurtosis(sig, fisher=False, bias=False)
            feat['median_abs_dev'] = np.median(np.abs(sig - np.median(sig)))

            f, Pxx = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
            total_power = np.trapz(Pxx, f) + 1e-12

            feat['lf_ratio'] = np.trapz(Pxx[f < 2], f[f < 2]) / total_power
            feat['hf_ratio'] = (
                np.trapz(Pxx[(f >= 30) & (f <= 100)], f[(f >= 30) & (f <= 100)])
                / total_power
            )
            feat['delta_ratio'] = np.trapz(Pxx[(f >= 1) & (f <= 4)], f[(f >= 1) & (f <= 4)]) / total_power
            feat['theta_ratio'] = np.trapz(Pxx[(f >= 4) & (f <= 8)], f[(f >= 4) & (f <= 8)]) / total_power
            feat['alpha_ratio'] = np.trapz(Pxx[(f >= 8) & (f <= 13)], f[(f >= 8) & (f <= 13)]) / total_power
            feat['beta_ratio']  = np.trapz(Pxx[(f >= 13) & (f <= 30)], f[(f >= 13) & (f <= 30)]) / total_power
            feat['gamma_ratio'] = np.trapz(Pxx[(f >= 30) & (f <= 80)], f[(f >= 30) & (f <= 80)]) / total_power
            feat['spectral_flatness'] = (
                np.exp(np.mean(np.log(Pxx + 1e-12))) / (np.mean(Pxx) + 1e-12)
            )

            acf = np.correlate(sig - sig.mean(), sig - sig.mean(), mode='full')
            acf = acf[len(acf) // 2:]
            acf = acf / (acf[0] + 1e-12)
            lag_idx = int(fs * 0.02)
            feat['acf_lag1'] = acf[lag_idx] if len(acf) > lag_idx else 0.0
            features[i] = feat
        return features

    @staticmethod
    def _classify_channels_safe(ch_order, ch_coords):
        lh, rh, mid = [], [], []
        for i, name in enumerate(ch_order):
            match = re.search(r'(\d+)$', name)
            if match:
                num = int(match.group(1))
                if num % 2 == 1:
                    lh.append(i)
                else:
                    rh.append(i)
            elif 'z' in name.lower():
                mid.append(i)
            else:
                x = ch_coords[name][0]
                if x < -3.0:
                    lh.append(i)
                elif x > 3.0:
                    rh.append(i)
                else:
                    mid.append(i)
        return lh, rh, mid

    def _classify_artifacts(self, features, mixing_matrix, ch_order, mode="standard"):
        n_comp = len(features)

        posterior_channels = [
            j for j, ch in enumerate(ch_order)
            if ch in {'O1','O2','Oz','PO3','PO4','PO7','PO8','POz',
                      'P1','P2','P3','P4','P7','P8','Pz','Iz'}
        ]
        frontal_channels = [
            j for j, ch in enumerate(ch_order)
            if ch in {'Fp1','Fp2','Fpz','AF7','AF8','AF3','AF4'}
        ]
        central_channels = [
            j for j, ch in enumerate(ch_order)
            if ch in {'C1','C2','C3','C4','C5','C6','Cz',
                      'FC1','FC2','FC3','FC4','FC5','FC6','FCz',
                      'CP1','CP2','CP3','CP4','CP5','CP6','CPz'}
        ]
        frontal_midline_channels = [
            j for j, ch in enumerate(ch_order)
            if ch in {'Fz','FCz','Cz','F1','F2','FC1','FC2','AFz'}
        ]
        
        n_ch = mixing_matrix.shape[0]
        posterior_channels = [j for j in posterior_channels if j < n_ch]
        frontal_channels = [j for j in frontal_channels if j < n_ch]
        central_channels = [j for j in central_channels if j < n_ch]
        frontal_midline_channels = [j for j in frontal_midline_channels if j < n_ch]
        
        feat_keys = ['std', 'kurtosis', 'lf_ratio', 'hf_ratio',
                     'alpha_ratio', 'spectral_flatness', 'acf_lag1']
        feat_matrix = np.array([
            [features[i][k] for k in feat_keys] for i in range(n_comp)
        ])
        feat_z = (feat_matrix - feat_matrix.mean(axis=0)) / (feat_matrix.std(axis=0) + 1e-12)

        bad_idx = []
        reasons = {}

        for i in range(n_comp):
            f = features[i]
            fz = feat_z[i]
            score = 0.0
            reason_list = []
            weights = np.abs(mixing_matrix[:, i])
            overall_w = weights.mean() + 1e-12

            if fz[1] > 2.5 and f['lf_ratio'] > 0.3 and frontal_channels:
                frontal_w = weights[frontal_channels].mean()
                if frontal_w > 1.5 * overall_w:
                    score += 3.0
                    reason_list.append("EOG")

            if f['hf_ratio'] > 0.15 and fz[1] > 1.5:
                score += 2.5
                reason_list.append("EMG")

            if weights.max() > 5 * np.median(weights):
                score += 2.0
                reason_list.append("Channel_noise")

            if f['spectral_flatness'] > 0.8:
                score += 1.5
                reason_list.append("Line_noise")

            if f['acf_lag1'] < 0.3:
                score += 1.0
                reason_list.append("Spiky")

            if f['alpha_ratio'] > 0.25 and posterior_channels:
                post_w = weights[posterior_channels].mean()
                post_dom = post_w / overall_w
                if f['alpha_ratio'] > 0.50 and post_dom > 1.3:
                    score -= 6.0
                    reason_list.append("PROTECT_strong_alpha")
                elif f['alpha_ratio'] > 0.35 and post_dom > 1.1:
                    score -= 4.0
                    reason_list.append("PROTECT_mod_alpha")
                elif post_dom > 1.0:
                    score -= 2.5
                    reason_list.append("PROTECT_weak_alpha")

            if f['beta_ratio'] > 0.25 and central_channels:
                central_w = weights[central_channels].mean()
                central_dom = central_w / overall_w
                if f['beta_ratio'] > 0.40 and central_dom > 1.3:
                    score -= 4.0
                    reason_list.append("PROTECT_strong_beta")
                elif f['beta_ratio'] > 0.30 and central_dom > 1.1:
                    score -= 2.5
                    reason_list.append("PROTECT_mod_beta")
                elif central_dom > 1.0:
                    score -= 1.5
                    reason_list.append("PROTECT_weak_beta")

            if f['theta_ratio'] > 0.25 and frontal_midline_channels:
                fm_w = weights[frontal_midline_channels].mean()
                fm_dom = fm_w / overall_w
                if f['theta_ratio'] > 0.40 and fm_dom > 1.3:
                    score -= 3.5
                    reason_list.append("PROTECT_strong_theta")
                elif f['theta_ratio'] > 0.30 and fm_dom > 1.1:
                    score -= 2.0
                    reason_list.append("PROTECT_mod_theta")
                elif fm_dom > 1.0:
                    score -= 1.0
                    reason_list.append("PROTECT_weak_theta")

            if f['delta_ratio'] > 0.40:
                frontal_w = weights[frontal_channels].mean() if frontal_channels else 0
                frontal_dom = frontal_w / overall_w
                if frontal_dom < 1.3 and f['lf_ratio'] < 0.5:
                    if f['delta_ratio'] > 0.60:
                        score -= 2.0
                        reason_list.append("PROTECT_strong_delta")
                    else:
                        score -= 1.0
                        reason_list.append("PROTECT_weak_delta")

            threshold = {"conservative": 4.0, "standard": 3.0, "aggressive": 2.0}[mode]
            if score >= threshold:
                bad_idx.append(i)
                reasons[i] = f"Score={score:.1f}: {'+'.join(reason_list)}"

        return np.array(bad_idx), reasons

    def clean(self, eeg, fs, ch_order):
        sig = self._compute_signature(eeg, fs)
        cache_path = self.cache_dir / f"ica_{sig}.npz"

        if self.config.CACHE_MI_DICE and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            Logger.info(f"ICA: Cache HIT ({len(data['bad_idx'])} removed)")
            return data['eeg_clean'], {
                'cache': 'HIT', 'bad_idx': data['bad_idx'].tolist()
            }

        import mne
        n_comp = min(self.config.ICA_N_COMPONENTS, eeg.shape[0])
        Logger.info(f"ICA: Decomposing {eeg.shape[0]}ch → {n_comp} components...")

        fit_samples = int(fs * 180)
        rng = np.random.RandomState(self.config.RANDOM_SEED)
        fit_idx = np.sort(rng.choice(eeg.shape[1], min(fit_samples, eeg.shape[1]), replace=False))

        n_ch = eeg.shape[0]
        ch_names = ch_order[:n_ch]  # Sadece 62 kanal
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg', verbose=False)
        raw_fit = mne.io.RawArray(eeg[:, fit_idx], info, verbose=False)
        raw_full = mne.io.RawArray(eeg, info, verbose=False)

        ica_mne = mne.preprocessing.ICA(
            n_components=n_comp,
            method='picard',
            fit_params=dict(tol=5e-3, max_iter=500, extended=False),
            random_state=self.config.RANDOM_SEED,
            verbose=False,
        )
        ica_mne.fit(raw_fit, verbose=False)
        sources = ica_mne.get_sources(raw_full).get_data()  # (n_comp, T)
        mixing = ica_mne.mixing_matrix_                      # (n_ch, n_comp)

        features = self._extract_features(sources, fs)
        bad_idx, reasons = self._classify_artifacts(
            features, mixing, ch_names, self.config.ICA_MODE
        )

        max_remove = self.config.ICA_MAX_REMOVE[self.config.ICA_MODE]
        if len(bad_idx) > max_remove:
            scores = []
            for idx in bad_idx:
                try:
                    scores.append(float(reasons[idx].split('=')[1].split(':')[0]))
                except (ValueError, IndexError):
                    scores.append(0.0)
            top_idx = np.argsort(scores)[-max_remove:]
            bad_idx = bad_idx[top_idx]

        sources_clean = sources.copy()
        sources_clean[bad_idx, :] = 0.0
        eeg_clean = (mixing @ sources_clean).astype(np.float32)  # (n_ch, T)

        if self.config.CACHE_MI_DICE:
            np.savez(
                cache_path,
                eeg_clean=eeg_clean,
                bad_idx=np.array(bad_idx, dtype=np.int32),
            )

        Logger.success(f"ICA: Removed {len(bad_idx)}/{n_comp} components")
        return eeg_clean, {
            'cache': 'MISS',
            'bad_idx': bad_idx.tolist(),
            'reasons': {int(k): v for k, v in reasons.items()},
        }
        
# ============================================================================
# CSD RE-REFERENCING
# ============================================================================

class CSDReferencer:
    @staticmethod
    def apply(eeg, ch_coords, ch_order, config):
        sigma = config.CSD_SIGMA
        Logger.info(f"CSD: σ={sigma:.1f}mm, mode={config.CSD_ALPHA_MODE}")

        coords_array = np.stack([ch_coords[ch] for ch in ch_order])
        distances = cdist(coords_array, coords_array)

        G = np.exp(-(distances ** 2) / (2 * sigma ** 2)).astype(np.float32)

        taper_start = 60.0
        taper_end = 100.0
        taper_mask = (distances > taper_start) & (distances <= taper_end)
        G[taper_mask] *= np.exp(
            -((distances[taper_mask] - taper_start) / 20.0) ** 2
        )
        G[distances > taper_end] = 0.0

        np.fill_diagonal(G, 0.0)
        row_sums = G.sum(axis=1, keepdims=True)
        isolated = row_sums.squeeze() < 1e-6
        if np.any(isolated):
            n_isolated = int(isolated.sum())
            Logger.warn(f"CSD: {n_isolated} isolated channels detected")
            G[isolated, :] = 0.0
            row_sums[isolated] = 1.0
        
        G = G / row_sums

        if not config.CSD_PRESERVE_ALPHA or config.CSD_ALPHA_MODE == "none":
            return eeg - (G @ eeg)

        fs = config.FS
        nyq = fs / 2
        y_coords = coords_array[:, 1]

        posterior = y_coords < config.CSD_POSTERIOR_Y_THRESHOLD
        frontal = y_coords > config.CSD_FRONTAL_Y_THRESHOLD
        central = ~posterior & ~frontal

        if config.CSD_ALPHA_MODE == "posterior_selective":
            b_a, a_a = butter(4, [8 / nyq, 13 / nyq], btype='band')
            eeg_alpha = filtfilt(b_a, a_a, eeg, axis=1)

            eeg_alpha_post = np.zeros_like(eeg)
            eeg_alpha_post[posterior] = eeg_alpha[posterior]

            eeg_alpha_cent = np.zeros_like(eeg)
            eeg_alpha_cent[central] = (
                config.CSD_CENTRAL_ALPHA_RETENTION * eeg_alpha[central]
            )

            eeg_alpha_front = eeg_alpha.copy()
            eeg_alpha_front[posterior] = 0.0
            eeg_alpha_front[central] *= (1.0 - config.CSD_CENTRAL_ALPHA_RETENTION)

            eeg_non_alpha = eeg - eeg_alpha
            eeg_to_csd = eeg_non_alpha + eeg_alpha_front + 0.5 * eeg_alpha_cent
            eeg_csd = ((eeg_to_csd - (G @ eeg_to_csd))
                       + eeg_alpha_post + 0.5 * eeg_alpha_cent)

            gamma_low = config.BANDS["gamma"][0]
            b_b, a_b = butter(4, [13 / nyq, min(gamma_low, nyq - 1) / nyq],
                              btype='band')
            eeg_beta_orig = filtfilt(b_b, a_b, eeg, axis=1)
            beta_restore = np.zeros_like(eeg)
            beta_restore[central] = (
                config.CSD_CENTRAL_BETA_RETENTION * eeg_beta_orig[central]
            )
            eeg_csd += beta_restore

            b_t, a_t = butter(4, [4 / nyq, 8 / nyq], btype='band')
            eeg_theta_orig = filtfilt(b_t, a_t, eeg, axis=1)
            midline = np.abs(coords_array[:, 0]) < 10.0
            theta_restore = np.zeros_like(eeg)
            theta_restore[midline] = 0.10 * eeg_theta_orig[midline]
            eeg_csd += theta_restore

            Logger.success("CSD: Applied (soft taper + multi-band preservation)")
            return eeg_csd

        return eeg - (G @ eeg)


# ============================================================================
# VOXEL GRID
# ============================================================================

class VoxelGrid:
    """MNI-space voxel grid with GM masking and atlas labeling."""

    def __init__(self, config):
        self.config = config
        self.spacing = config.GRID_SPACING
        Logger.info("Grid: Creating 3D voxel grid...")
        self.coords_all, self.shape = self._create_grid()
        self.affine = self._create_affine()
        self.inv_affine = np.linalg.inv(self.affine)
        Logger.info("Grid: Loading atlas...")
        self.gm_mask = self._load_gm_mask()
        self.atlas_idx, self.roi_names = self._load_atlas()
        Logger.info("Grid: Filtering to GM...")
        self.coords_gm = self._filter_to_gm()
        Logger.success(f"Grid: {len(self.coords_gm)} GM voxels")

    def _create_grid(self):
        b = self.config.GRID_BOUNDS
        xs = np.arange(b['x'][0], b['x'][1], self.spacing)
        ys = np.arange(b['y'][0], b['y'][1], self.spacing)
        zs = np.arange(b['z'][0], b['z'][1], self.spacing)
        grid = np.array(np.meshgrid(xs, ys, zs, indexing='ij'))
        return grid.reshape(3, -1).T, (len(xs), len(ys), len(zs))

    def _create_affine(self):
        s = self.spacing
        b = self.config.GRID_BOUNDS
        return np.array([
            [s, 0, 0, b['x'][0]],
            [0, s, 0, b['y'][0]],
            [0, 0, s, b['z'][0]],
            [0, 0, 0, 1],
        ], dtype=np.float64)

    def _load_gm_mask(self):
        target_img = nib.Nifti1Image(
            np.zeros(self.shape, dtype=np.int16), self.affine
        )
        ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        cort_img = ho.maps if isinstance(ho.maps, nib.Nifti1Image) else nib.load(ho.maps)
        resampled = image.resample_to_img(cort_img, target_img, interpolation='nearest')
        gm = resampled.get_fdata() > 0

        if self.config.GM_INCLUDE_SUBCORTICAL:
            ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
            sub_img = (ho_sub.maps if isinstance(ho_sub.maps, nib.Nifti1Image)
                       else nib.load(ho_sub.maps))
            sub_r = image.resample_to_img(sub_img, target_img, interpolation='nearest')
            gm |= (sub_r.get_fdata() > 0)
        return gm

    def _load_atlas(self):
        ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = ho.maps if isinstance(ho.maps, nib.Nifti1Image) else nib.load(ho.maps)
        target_img = nib.Nifti1Image(
            np.zeros(self.shape, dtype=np.int16), self.affine
        )
        resampled = image.resample_to_img(atlas_img, target_img, interpolation='nearest')
        return resampled.get_fdata().astype(int), list(ho.labels)

    def _filter_to_gm(self):
        coords_gm = []
        for coord in tqdm(self.coords_all, desc="  Filtering GM", ncols=80):
            vox = self.inv_affine @ np.append(coord, 1)
            xi, yi, zi = np.round(vox[:3]).astype(int)
            if (0 <= xi < self.shape[0] and 0 <= yi < self.shape[1]
                    and 0 <= zi < self.shape[2] and self.gm_mask[xi, yi, zi]):
                coords_gm.append(coord)
        return np.asarray(coords_gm, dtype=np.float32)

    def mni_to_voxel(self, mni_coords):
        vox = self.inv_affine @ np.append(mni_coords, 1)
        return np.round(vox[:3]).astype(int)


# ============================================================================
# VOXEL SIGNATURES
# ============================================================================

class SignatureComputer:
    """Voxel signatures — spatial mapping from channels to voxels."""

    def __init__(self, config):
        self.config = config

    def _get_adaptive_sigma(self, voxel_y, band_name=None):
        sigma_base = self.config.SIGMA_BASE

        if voxel_y < -90:
            region = "occipital"
        elif voxel_y < -70:
            region = "posterior"
        elif voxel_y < -40:
            region = "parietal"
        elif voxel_y < 0:
            region = "central"
        else:
            region = "frontal"

        sigma = sigma_base * self.config.SIGMA_MULTIPLIERS[region]
        if band_name:
            sigma *= self.config.BAND_SIGMA_SCALE.get(band_name, 1.0)
        return sigma

    def compute(self, voxel_coords, ch_coords, ch_order, band_name=None):
        Logger.section(
            f"Computing Signatures{f' [{band_name}]' if band_name else ''}"
        )
        n_voxels = len(voxel_coords)
        n_channels = len(ch_order)
        ch_coords_array = np.stack([ch_coords[ch] for ch in ch_order])

        lh_idx, rh_idx, mid_idx = ICAArtifactRemover._classify_channels_safe(
            ch_order, ch_coords
        )

        hemi_iso = self.config.HEMISPHERE_ISOLATION
        cross_w = self.config.CROSS_HEMISPHERE_WEIGHT

        signatures = np.zeros((n_voxels, n_channels), dtype=np.float32)
        batch_size = self.config.BATCH_SIZE_SIGNATURES

        for batch_start in tqdm(range(0, n_voxels, batch_size),
                                desc="Signatures", ncols=80):
            batch_end = min(batch_start + batch_size, n_voxels)
            batch_coords = voxel_coords[batch_start:batch_end]
            distances = np.linalg.norm(
                batch_coords[:, None, :] - ch_coords_array[None, :, :], axis=2
            )

            for bi, gi in enumerate(range(batch_start, batch_end)):
                vox = voxel_coords[gi]
                dist = distances[bi]
                sigma = self._get_adaptive_sigma(vox[1], band_name)

                weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                weights[dist > 200] = 0.0

                if np.sum(weights > 0.01) < 8:
                    nearest = np.argsort(dist)[:8]
                    weights[nearest] = np.maximum(weights[nearest], 0.05)

                vox_x = vox[0]
                gap = self.config.MIDLINE_GAP_MM
                if vox_x < -gap:
                    weights[lh_idx] *= hemi_iso
                    weights[rh_idx] *= cross_w
                    weights[mid_idx] *= self.config.MIDLINE_WEIGHT
                elif vox_x > gap:
                    weights[rh_idx] *= hemi_iso
                    weights[lh_idx] *= cross_w
                    weights[mid_idx] *= self.config.MIDLINE_WEIGHT
                else:
                    weights[lh_idx] *= 0.80
                    weights[rh_idx] *= 0.80

                w_sum = weights.sum()
                if w_sum > 1e-12:
                    weights /= w_sum
                else:
                    weights[:] = 0.0
                    weights[np.argmin(dist)] = 1.0

                signatures[gi] = weights

        Logger.success(f"Signatures: {n_voxels} voxels computed")
        return signatures


# ============================================================================
# MI / DICE COMPUTATION — v5.1.0: RANK-BASED DICE
# ============================================================================

class MIDiceComputer:
    """
    Mutual Information and Dice similarity computation.
    
    v5.1.0: Added rank-based Dice for amplitude-invariant similarity.
    Rank-based Dice eliminates sensitivity to broadband power fluctuations.
    """

    def __init__(self, config):
        self.config = config

    def compute(self, voxel_sigs, snapshots):
        Logger.info("Computing MI/Dice scores (rank-based Dice v5.1)...")
        n_voxels, n_channels = voxel_sigs.shape
        n_timepoints = snapshots.shape[0]
        n_bins = self.config.MI_N_BINS

        v_bins = [
            np.histogram_bin_edges(voxel_sigs[i], bins=n_bins)
            for i in tqdm(range(n_voxels), desc="  Bins", ncols=80)
        ]
    
        sig_ranks = np.zeros_like(voxel_sigs)
        for i in range(n_voxels):
            sig_ranks[i] = rankdata(voxel_sigs[i]) / n_channels

        # ── DÖNGÜ DIŞINDA — bir kez hesaplanır ──
        vb_all = np.array([
            np.clip(np.digitize(voxel_sigs[i], v_bins[i], right=False) - 1, 0, n_bins - 1)
            for i in range(n_voxels)
        ], dtype=np.int32)  # (n_voxels, n_channels)

        mi = np.zeros((n_voxels, n_timepoints), dtype=np.float32)
        dice = np.zeros_like(mi)

        for t in tqdm(range(n_timepoints), desc="  Timepoints", ncols=80):
            snapshot = snapshots[t]

            snap_rank = rankdata(snapshot).astype(np.float32) / n_channels
            numerator = 2.0 * (sig_ranks @ snap_rank)
            denominator = (np.sum(sig_ranks ** 2, axis=1)
                          + np.sum(snap_rank ** 2) + 1e-12)
            dice_t = numerator / denominator

            s_bins = np.linspace(snapshot.min(), snapshot.max(), n_bins + 1)
            sb = np.clip(
                np.digitize(snapshot, s_bins, right=False) - 1, 0, n_bins - 1
            )

            joint_idx_all = vb_all * n_bins + sb[np.newaxis, :]  # (n_voxels, n_channels)

            contingency_all = np.zeros((n_voxels, n_bins * n_bins), dtype=np.float32)
            for b in range(n_bins * n_bins):
                contingency_all[:, b] = (joint_idx_all == b).sum(axis=1)
            contingency_all += 0.5

            row_sums = contingency_all.sum(axis=1, keepdims=True)
            pxy_all = contingency_all / row_sums
            px_all = pxy_all.reshape(n_voxels, n_bins, n_bins).sum(axis=2)
            py_all = pxy_all.reshape(n_voxels, n_bins, n_bins).sum(axis=1)
            px_py_all = (px_all[:, :, np.newaxis] * py_all[:, np.newaxis, :]).reshape(n_voxels, n_bins * n_bins)
            mask_all = pxy_all > 0
            log_term = np.where(mask_all, np.log(pxy_all / (px_py_all + 1e-10)), 0.0)
            mi_t = np.maximum(0.0, (pxy_all * log_term).sum(axis=1))

            for arr in [mi_t, dice_t]:
                vmin, vmax = arr.min(), arr.max()
                if vmax > vmin:
                    arr[:] = (arr - vmin) / (vmax - vmin)

            mi[:, t] = mi_t
            dice[:, t] = dice_t

        Logger.success("MI/Dice: Complete (rank-based Dice v5.1)")
        return mi, dice


# ============================================================================
# CACHE MANAGER — v5.1 Compatible
# ============================================================================

class CacheManager:
    """
    Manages MI/Dice cache files.
    
    v5.1.0: Primary cache is v51 (incompatible with v50/v40 due to
    relative band power and rank-based Dice changes).
    
    NOTE: v50/v40 caches are NOT used as fallback because the envelope
    computation changed (absolute → relative). Old caches would give
    wrong results.
    """
    
    FALLBACK_DIRS = ["cache_v51"]  # v5.1: NO fallback to old caches
    
    def __init__(self, config):
        self.config = config
        self.primary_dir = Path(config.DATA_PATH) / config.CACHE_DIR
        self.primary_dir.mkdir(parents=True, exist_ok=True)
    
    def find_cache(self, subject_id, band_name):
        for cache_dir_name in self.FALLBACK_DIRS:
            cache_dir = Path(self.config.DATA_PATH) / cache_dir_name
            cache_path = cache_dir / f"{subject_id}_{band_name}_mi_dice.npz"
            if cache_path.exists():
                return cache_path
        return None
    
    def all_bands_cached(self, subject_id):
        for band_name in self.config.BANDS:
            if self.find_cache(subject_id, band_name) is None:
                return False
        return True
    
    def load_subject(self, subject_id, band_name):
        path = self.find_cache(subject_id, band_name)
        if path is None:
            raise FileNotFoundError(
                f"No cache found for {subject_id}/{band_name} "
                f"in {self.FALLBACK_DIRS}"
            )
        
        source_dir = path.parent.name
        if source_dir != self.config.CACHE_DIR:
            Logger.info(f"    Cache fallback: {source_dir}/{path.name}")
        
        data = np.load(path)
        return data['mi'], data['dice'], data['voxel_coords']
    
    def save(self, subject_id, band_name, mi, dice, voxel_coords):
        path = self.primary_dir / f"{subject_id}_{band_name}_mi_dice.npz"
        np.savez_compressed(path, mi=mi, dice=dice, voxel_coords=voxel_coords)
        Logger.info(f"    Cached: {path.name}")
    
    def load_all_subjects(self, subject_files):
        subject_data = []
        
        for subject_file in subject_files:
            subject_id = subject_file.split('_')[0]
            
            if not self.all_bands_cached(subject_id):
                Logger.warn(f"  {subject_id}: Incomplete cache, skipping")
                continue
            
            mi_dict, dice_dict = {}, {}
            voxel_coords = None
            
            for band_name in self.config.BANDS:
                mi, dice, coords = self.load_subject(subject_id, band_name)
                mi_dict[band_name] = mi
                dice_dict[band_name] = dice
                if voxel_coords is None:
                    voxel_coords = coords
            
            subject_data.append((subject_id, mi_dict, dice_dict, voxel_coords))
            Logger.info(f"  ✅ {subject_id}: Loaded from cache")
        
        return subject_data


# ============================================================================
# SPATIAL PRIOR — v5.1.0: DUAL-PEAK GAMMA
# ============================================================================

class SpatialPrior:
    """
    Spatial prior weights for anatomically-informed voxel scoring.
    
    v5.1.0: Added dual_peak prior type for gamma band.
    Gamma now uses posterior (visual) + frontal (cognitive) peaks
    instead of copying alpha's posterior-only prior.
    """

    def __init__(self, config):
        self.config = config

    def get_weights(self, voxel_coords, band_name, boost=1.5,
                    penalty=0.0, softening=1.0):
        structure = self.config.SPATIAL_PRIOR_STRUCTURE.get(band_name)
        fixed = self.config.SPATIAL_PRIOR_FIXED.get(band_name)

        if structure is None or fixed is None:
            return np.ones(len(voxel_coords), dtype=np.float32)

        x = voxel_coords[:, 0].astype(np.float64)
        y = voxel_coords[:, 1].astype(np.float64)
        z = voxel_coords[:, 2].astype(np.float64)

        prior_type = structure['type']
        min_w = fixed.get('min_weight', 0.05)

        # ── Compute raw spatial weights ──
        if prior_type == "gaussian":
            cy = fixed['center_y']
            sy = fixed['sigma_y']
            weights = np.exp(-((y - cy) ** 2) / (2 * sy ** 2))

        elif prior_type == "band_z":
            wy = np.exp(-((y - fixed['y_center']) ** 2) / (2 * fixed['y_sigma'] ** 2))
            wz = np.exp(-((z - fixed['z_center']) ** 2) / (2 * fixed['z_sigma'] ** 2))
            wx = np.exp(-((np.abs(x) - fixed['abs_x_center']) ** 2) / (2 * fixed['abs_x_sigma'] ** 2))
            weights = wy * wz * wx

        elif prior_type == "midline_frontal":
            wy = np.exp(-((y - fixed['y_center']) ** 2) / (2 * fixed['y_sigma'] ** 2))
            wz = np.exp(-((z - fixed['z_center']) ** 2) / (2 * fixed['z_sigma'] ** 2))
            wx = np.exp(-(x ** 2) / (2 * fixed['abs_x_sigma'] ** 2))
            weights = wy * wz * wx

        elif prior_type == "frontal_broad":
            cy = fixed['y_center']
            sy = fixed['y_sigma']
            weights = np.exp(-((y - cy) ** 2) / (2 * sy ** 2))

        elif prior_type == "dual_peak":
            # v5.1.0 NEW: Dual-peak prior for gamma
            # Peak 1: Posterior (visual gamma)
            cy_post = fixed['center_y_post']
            sy_post = fixed['sigma_y_post']
            w_post = np.exp(-((y - cy_post) ** 2) / (2 * sy_post ** 2))

            # Peak 2: Frontal (cognitive gamma)
            cy_front = fixed['center_y_front']
            sy_front = fixed['sigma_y_front']
            w_front = np.exp(-((y - cy_front) ** 2) / (2 * sy_front ** 2))

            # Weighted combination
            pw = fixed.get('post_weight', 0.5)
            fw = fixed.get('front_weight', 0.5)
            weights = pw * w_post + fw * w_front

        else:
            weights = np.ones(len(voxel_coords), dtype=np.float64)

        # ── Apply penalty for off-target regions ──
        if penalty > 0 and 'penalty_y_center' in fixed:
            pc = fixed['penalty_y_center']
            ps = fixed['penalty_y_sigma']
            penalty_gauss = np.exp(-((y - pc) ** 2) / (2 * ps ** 2))
            suppressor = 1.0 - penalty * penalty_gauss
            weights *= suppressor

        # ── Normalize and scale by boost ──
        w_max = weights.max()
        if w_max > 1e-12:
            norm_weights = weights / w_max
            weights = min_w + (boost - min_w) * norm_weights
        else:
            weights = np.full(len(voxel_coords), min_w, dtype=np.float64)

        # ── Soften: blend with uniform (1.0) ──
        if softening > 0:
            weights = (1.0 - softening) * weights + softening * 1.0

        return weights.astype(np.float32)

    def apply(self, hybrid, voxel_coords, band_name, boost=1.5,
              penalty=0.0, softening=1.0):
        """Apply spatial prior to hybrid scores (2D: voxels × time)."""
        weights = self.get_weights(
            voxel_coords, band_name, boost, penalty, softening
        )
        if hybrid.ndim == 2:
            return hybrid * weights[:, np.newaxis]
        return hybrid * weights


# ============================================================================
# LOSO OPTIMIZER v5.1 — TWO-LEVEL AUTONOMOUS
# ============================================================================

class LOSOOptimizer:
    """
    Two-level Leave-One-Subject-Out optimizer.
    
    Unchanged from v5.0 except version tags and cache references.
    """

    def __init__(self, config):
        self.config = config
        self.cache_mgr = CacheManager(config)
        self.spatial_prior = SpatialPrior(config)
        self._eval_cache = {}

    @staticmethod
    def _build_mask(coords, region_def):
        y = coords[:, 1]
        x = coords[:, 0]
        z = coords[:, 2]
        mask = np.ones(len(coords), dtype=bool)

        if 'y_min' in region_def:
            mask &= (y >= region_def['y_min'])
        if 'y_max' in region_def:
            mask &= (y <= region_def['y_max'])
        if 'z_min' in region_def:
            mask &= (z >= region_def['z_min'])
        if 'z_max' in region_def:
            mask &= (z <= region_def['z_max'])
        if 'abs_x_min' in region_def:
            mask &= (np.abs(x) >= region_def['abs_x_min'])
        if 'abs_x_max' in region_def:
            mask &= (np.abs(x) <= region_def['abs_x_max'])
        return mask

    def _compute_band_metric(self, avg_hybrid, coords, band_name):
        target_cfg = self.config.LOSO_TARGETS.get(band_name)
        if target_cfg is None:
            return 0.0

        target_type = target_cfg['type']
        target_val = target_cfg['target_value']
        target_range = target_cfg.get('target_range', (0, 999))
        y = coords[:, 1]

        if target_type == "ratio":
            num_mask = self._build_mask(coords, target_cfg['regions']['numerator'])
            den_mask = self._build_mask(coords, target_cfg['regions']['denominator'])
            if den_mask.any() and num_mask.any():
                if target_cfg.get('use_mean', False):
                    raw_val = (avg_hybrid[num_mask].mean()
                               / (avg_hybrid[den_mask].mean() + 1e-12))
                else:
                    raw_val = (avg_hybrid[num_mask].sum()
                               / (avg_hybrid[den_mask].sum() + 1e-12))
            else:
                raw_val = 0.0

        elif target_type == "proportion":
            mask = self._build_mask(coords, target_cfg['region'])
            raw_val = (avg_hybrid[mask].sum() / (avg_hybrid.sum() + 1e-12)
                       if mask.any() else 0.0)

        elif target_type == "proportion_ratio":
            mask = self._build_mask(coords, target_cfg['region'])
            if mask.any():
                prop = avg_hybrid[mask].sum() / (avg_hybrid.sum() + 1e-12)
                raw_val = prop / (1.0 - prop + 1e-12)
            else:
                raw_val = 0.0
        else:
            raw_val = 0.0

        target_low, target_high = target_range
        target_std = (target_high - target_low) / 4.0
        if target_std < 1e-6:
            target_std = 1.0

        z_val = (raw_val - target_val) / target_std
        score = 1.0 / (1.0 + z_val ** 2)

        if target_low <= raw_val <= target_high:
            score = min(1.0, score * 1.15)

        penalty_multiplier = 1.0

        if band_name == "delta":
            post_mask = y < -40
            ant_mask = y > 0
            if ant_mask.any() and post_mask.any():
                f_p_ratio = (avg_hybrid[ant_mask].mean()
                             / (avg_hybrid[post_mask].mean() + 1e-12))
                if f_p_ratio > 1.5:
                    penalty_multiplier *= np.exp(
                        -(f_p_ratio - 1.5) ** 2 / (2 * 1.5 ** 2)
                    )
                elif f_p_ratio < 0.8:
                    penalty_multiplier *= np.exp(
                        -(0.8 - f_p_ratio) ** 2 / (2 * 0.5 ** 2)
                    )

        elif band_name == "beta":
            central_mask = (y >= -45) & (y <= 15)
            if central_mask.any():
                central_pct = (100.0 * avg_hybrid[central_mask].sum()
                               / (avg_hybrid.sum() + 1e-12))
                if central_pct > 35:
                    penalty_multiplier *= np.exp(
                        -(central_pct - 35) ** 2 / (2 * 20 ** 2)
                    )

        return float(score * penalty_multiplier)

    def _generate_candidates(self, bounds_dict, n_candidates, seed=None):
        if seed is None:
            seed = self.config.RANDOM_SEED

        param_names = list(bounds_dict.keys())
        n_params = len(param_names)
        bounds = np.array([bounds_dict[p] for p in param_names])

        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=n_params, seed=seed)
            samples = sampler.random(n=n_candidates)
        except ImportError:
            rng = np.random.RandomState(seed)
            samples = rng.uniform(0, 1, size=(n_candidates, n_params))

        candidates = []
        for row in samples:
            candidate = {}
            for i, param in enumerate(param_names):
                lo, hi = bounds[i]
                val = lo + row[i] * (hi - lo)
                if param == 'temporal_window':
                    val = int(round(val))
                candidate[param] = val
            candidates.append(candidate)
        return candidates

    def _evaluate_l1_candidate(self, candidate, mi_pooled, dice_pooled,
                               coords, band_name, fold_idx,
                               kf, subject_boundaries):
        param_hash = hashlib.md5(
            json.dumps(candidate, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        cache_key = (fold_idx, band_name, param_hash)

        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        mi_w = candidate.get('mi_weight', 0.55)
        boost = candidate.get('boost', 1.5)
        penalty = candidate.get('penalty', 0.0)
        contrast = candidate.get('contrast', 0.6)
        softening = candidate.get('softening', 0.5)

        h_raw = mi_w * mi_pooled + (1.0 - mi_w) * dice_pooled

        prior_w = self.spatial_prior.get_weights(
            coords, band_name, boost=boost,
            penalty=penalty, softening=softening,
        )
        h_raw = h_raw * prior_w[:, np.newaxis]

        if contrast != 1.0:
            h_raw = np.power(np.maximum(h_raw, 1e-12), contrast)

        hmin, hmax = h_raw.min(), h_raw.max()
        if hmax > hmin:
            h_norm = (h_raw - hmin) / (hmax - hmin)
        else:
            h_norm = h_raw

        avg_h = h_norm.mean(axis=1)
        main_score = self._compute_band_metric(avg_h, coords, band_name)

        fold_scores = []
        n_subjects = len(subject_boundaries) - 1

        for _, val_subj_idxs in kf.split(range(n_subjects)):
            val_time_indices = []
            for s_idx in val_subj_idxs:
                start = subject_boundaries[s_idx]
                end = subject_boundaries[s_idx + 1]
                val_time_indices.extend(range(start, end))

            if not val_time_indices:
                continue

            f_avg = h_norm[:, val_time_indices].mean(axis=1)
            f_score = self._compute_band_metric(f_avg, coords, band_name)
            fold_scores.append(f_score)

        if fold_scores:
            stability = 1.0 - (np.std(fold_scores)
                                / (np.mean(fold_scores) + 1e-12))
        else:
            stability = 0.5

        stability_penalty = max(
            0, self.config.LOSO_STABILITY_THRESHOLD - stability
        )
        objective = main_score * max(0.1, stability) - stability_penalty

        self._eval_cache[cache_key] = float(objective)
        return float(objective)

    def _run_l1_fold(self, test_idx, all_subjects, band_name):
        test_subject = all_subjects[test_idx]
        test_id = test_subject.split('_')[0]
        train_subjects = [s for i, s in enumerate(all_subjects) if i != test_idx]

        Logger.info(f"\n  ┌{'─'*56}┐")
        Logger.info(f"  │ FOLD {test_idx+1}/{len(all_subjects)}: "
                    f"Test={test_id}, Train={len(train_subjects)} │")
        Logger.info(f"  └{'─'*56}┘")

        # Load each subject's mi/dice as 1D mean (n_voxels,) — avoids huge time-axis concat
        # KFold runs over subjects (not timepoints) — valid for spatial optimization
        mi_subjects, dice_subjects, coords_ref = [], [], None

        for train_subj in train_subjects:
            train_id = train_subj.split('_')[0]
            try:
                mi, dice, coords = self.cache_mgr.load_subject(train_id, band_name)
                mi_1d   = mi.mean(axis=1)   if mi.ndim == 2   else mi
                dice_1d = dice.mean(axis=1) if dice.ndim == 2 else dice
                mi_subjects.append(mi_1d.astype(np.float32))
                dice_subjects.append(dice_1d.astype(np.float32))
                if coords_ref is None:
                    coords_ref = coords.astype(np.float32)
                del mi, dice, mi_1d, dice_1d
            except FileNotFoundError:
                Logger.warn(f"    Cache miss: {train_id}/{band_name}")
                continue

        if not mi_subjects:
            Logger.warn(f"  No train data for fold {test_idx+1}")
            return {
                'fold': test_idx, 'test_subject': test_id,
                'optimal_params': {}, 'test_metrics': {},
            }

        # (n_voxels, n_subjects) — ~20MB vs old (n_voxels, n_time*n_subjects) ~9GB
        mi_pooled   = np.stack(mi_subjects,   axis=1)
        dice_pooled = np.stack(dice_subjects, axis=1)
        n_subjects  = mi_pooled.shape[1]
        subject_boundaries = list(range(n_subjects + 1))

        del mi_subjects, dice_subjects
        gc.collect()

        kf = KFold(
            n_splits=min(self.config.KFOLD_N_SPLITS, n_subjects),
            shuffle=True, random_state=self.config.RANDOM_SEED,
        )

        # ── STAGE 1: Spatial anchoring ──
        Logger.info(f"    Stage 1: Spatial anchoring")

        spatial_bounds = {
            k: v for k, v in self.config.L1_BOUNDS.items()
            if k in ('boost', 'penalty', 'softening')
        }
        # Apply per-band overrides (e.g. alpha softening floor for EO data)
        band_overrides = getattr(self.config, 'L1_BOUNDS_OVERRIDE', {}).get(band_name, {})
        for k, v in band_overrides.items():
            if k in spatial_bounds:
                spatial_bounds[k] = v
        spatial_candidates = self._generate_candidates(
            spatial_bounds, self.config.LOSO_N_CANDIDATES_L1
        )

        defaults = self.config.L1_DEFAULTS[band_name]
        baseline_intensity = {
            'mi_weight': defaults['mi_weight'],
            'contrast': defaults['contrast'],
        }

        best_spatial_score = -np.inf
        best_spatial = {
            'boost': defaults['boost'],
            'penalty': defaults['penalty'],
            'softening': defaults['softening'],
        }

        for cand in spatial_candidates:
            full_cand = {**cand, **baseline_intensity}
            score = self._evaluate_l1_candidate(
                full_cand, mi_pooled, dice_pooled, coords_ref,
                band_name, test_idx, kf, subject_boundaries,
            )
            if score > best_spatial_score:
                best_spatial_score = score
                best_spatial = cand

        Logger.info(f"    → Anchor: B={best_spatial['boost']:.2f}, "
                    f"P={best_spatial['penalty']:.2f}, "
                    f"S={best_spatial['softening']:.2f} "
                    f"(score={best_spatial_score:.4f})")

        # ── STAGE 2: Intensity + volume params refinement ──
        Logger.info(f"    Stage 2: Intensity & volume refinement")

        intensity_bounds = {
            k: v for k, v in self.config.L1_BOUNDS.items()
            if k in ('mi_weight', 'contrast', 'keep_top_pct', 'smoothing_fwhm')
        }

        best_params = {**best_spatial, **baseline_intensity,
                       'keep_top_pct': defaults['keep_top_pct'],
                       'smoothing_fwhm': defaults['smoothing_fwhm']}
        current_best = best_spatial_score

        for r in range(self.config.LOSO_MAX_REFINEMENT):
            n_c = max(4, self.config.LOSO_N_CANDIDATES_L1 // (r + 1))

            if r == 0:
                candidates = self._generate_candidates(
                    intensity_bounds, n_c,
                    seed=self.config.RANDOM_SEED + r,
                )
            else:
                candidates = self._refine_around(
                    best_params, intensity_bounds, n_c, r,
                )

            improved = False
            for cand in candidates:
                full_cand = {**best_spatial, **cand}
                score = self._evaluate_l1_candidate(
                    full_cand, mi_pooled, dice_pooled, coords_ref,
                    band_name, test_idx, kf, subject_boundaries,
                )
                if score > current_best + self.config.LOSO_CONVERGENCE_TOL:
                    current_best = score
                    best_params = full_cand
                    improved = True

            if not improved and r > 0:
                Logger.info(f"    → Converged at round {r+1}")
                break

        lo, hi = self.config.MI_WEIGHT_CAPS.get(band_name, (0.30, 0.85))
        best_params['mi_weight'] = float(np.clip(best_params['mi_weight'], lo, hi))

        Logger.info(f"    ✅ Optimal: MI={best_params['mi_weight']:.3f}, "
                    f"C={best_params['contrast']:.3f}, "
                    f"B={best_params['boost']:.3f}, "
                    f"P={best_params['penalty']:.3f}, "
                    f"S={best_params['softening']:.3f}, "
                    f"K={best_params.get('keep_top_pct', 0):.3f}, "
                    f"FWHM={best_params.get('smoothing_fwhm', 0):.2f}")

        test_metrics = self._evaluate_test(test_id, band_name, best_params)

        del mi_pooled, dice_pooled
        gc.collect()

        return {
            'fold': test_idx,
            'test_subject': test_id,
            'optimal_params': best_params,
            'best_metric': float(current_best),
            'test_metrics': test_metrics,
        }

    def _refine_around(self, current_best, bounds_dict, n_candidates, round_idx):
        candidates = []
        shrink = 0.4 ** round_idx
        param_names = list(bounds_dict.keys())

        for _ in range(n_candidates):
            cand = {}
            for p in param_names:
                lo, hi = bounds_dict[p]
                center = current_best.get(p, (lo + hi) / 2)
                span = (hi - lo) * shrink
                new_lo = max(lo, center - span / 2)
                new_hi = min(hi, center + span / 2)
                val = np.random.uniform(new_lo, new_hi)
                if p == 'temporal_window':
                    val = int(round(val))
                cand[p] = val
            candidates.append(cand)
        return candidates

    def _evaluate_test(self, test_id, band_name, params):
        try:
            mi, dice, coords = self.cache_mgr.load_subject(test_id, band_name)
        except FileNotFoundError:
            Logger.warn(f"    Test eval: cache miss for {test_id}")
            return {}

        mi_w = params.get('mi_weight', 0.55)
        hybrid = mi_w * mi + (1.0 - mi_w) * dice

        hybrid = self.spatial_prior.apply(
            hybrid, coords, band_name,
            boost=params.get('boost', 1.5),
            penalty=params.get('penalty', 0.0),
            softening=params.get('softening', 0.5),
        )

        contrast = params.get('contrast', 1.0)
        if contrast != 1.0:
            hybrid = np.power(np.maximum(hybrid, 1e-12), contrast)

        hmin, hmax = hybrid.min(), hybrid.max()
        if hmax > hmin:
            hybrid = (hybrid - hmin) / (hmax - hmin)

        avg = hybrid.mean(axis=1)
        y = coords[:, 1]

        post_sum = avg[y < -40].sum()
        ant_sum = avg[y > 0].sum()
        pa_ratio = post_sum / (ant_sum + 1e-12)
        occ_sum = avg[y < -70].sum()
        occ_pct = 100.0 * occ_sum / (avg.sum() + 1e-12)
        band_metric = self._compute_band_metric(avg, coords, band_name)

        metrics = {
            'pa_ratio': float(pa_ratio),
            'occipital_pct': float(occ_pct),
            'band_metric': float(band_metric),
        }

        Logger.info(f"    Test: P/A={pa_ratio:.2f}, "
                    f"Occ={occ_pct:.1f}%, metric={band_metric:.4f}")

        del hybrid, mi, dice
        gc.collect()
        return metrics

    def optimize_band_l1(self, band_name, all_subjects):
        target_cfg = self.config.LOSO_TARGETS[band_name]
        Logger.section(f"L1 LOSO: {band_name.upper()}")
        Logger.info(f"  Target: {target_cfg['description']} = "
                    f"{target_cfg['target_value']} "
                    f"[{target_cfg['confidence']}]")
        Logger.info(f"  Reference: {target_cfg.get('reference', 'N/A')}")

        fold_results = []
        for i in range(len(all_subjects)):
            result = self._run_l1_fold(i, all_subjects, band_name)
            fold_results.append(result)
            self._eval_cache.clear()
            gc.collect()

        valid_folds = [r for r in fold_results if r['optimal_params']]
        if not valid_folds:
            Logger.error(f"  No valid folds for {band_name}")
            return self.config.L1_DEFAULTS[band_name], fold_results

        param_keys = valid_folds[0]['optimal_params'].keys()
        final_params = {}
        for k in param_keys:
            vals = [r['optimal_params'][k] for r in valid_folds
                    if k in r['optimal_params']]
            if vals:
                final_params[k] = float(np.median(vals))

        final_params['_mean_metric'] = float(
            np.mean([r['best_metric'] for r in valid_folds])
        )
        final_params['_std_metric'] = float(
            np.std([r['best_metric'] for r in valid_folds])
        )

        Logger.info(f"\n  {'='*55}")
        Logger.info(f"  FINAL L1 {band_name.upper()}:")
        for k, v in sorted(final_params.items()):
            Logger.info(f"    {k:20s}: {v:.4f}")
        Logger.info(f"  {'='*55}")

        return final_params, fold_results

    def _evaluate_l2_candidate(self, l2_candidate, all_subjects,
                               l1_params_per_band):
        raykill_min = l2_candidate.get(
            'raykill_min_cluster_mm3',
            self.config.L2_DEFAULTS['raykill_min_cluster_mm3']
        )
        temporal_window = int(l2_candidate.get(
            'temporal_window',
            self.config.L2_DEFAULTS['temporal_window']
        ))

        total_score = 0.0
        n_evaluations = 0

        for band_name in self.config.BANDS:
            l1_params = l1_params_per_band.get(
                band_name, self.config.L1_DEFAULTS[band_name]
            )
            band_scores = []

            rng = np.random.RandomState(self.config.RANDOM_SEED)
            n_sample = min(3, len(all_subjects))
            indices = rng.choice(len(all_subjects), n_sample, replace=False)
            sample_subjects = [all_subjects[i] for i in indices]

            for subject_file in sample_subjects:
                subject_id = subject_file.split('_')[0]
                try:
                    mi, dice, coords = self.cache_mgr.load_subject(
                        subject_id, band_name
                    )
                except FileNotFoundError:
                    continue

                mi_w = l1_params.get('mi_weight', 0.55)
                hybrid = mi_w * mi + (1.0 - mi_w) * dice

                hybrid = self.spatial_prior.apply(
                    hybrid, coords, band_name,
                    boost=l1_params.get('boost', 1.5),
                    penalty=l1_params.get('penalty', 0.0),
                    softening=l1_params.get('softening', 0.5),
                )

                contrast = l1_params.get('contrast', 1.0)
                if contrast != 1.0:
                    hybrid = np.power(np.maximum(hybrid, 1e-12), contrast)

                eps = float(np.finfo(np.float32).eps)
                hmin, hmax = hybrid.min(), hybrid.max()
                if hmax > hmin + eps:
                    hybrid = (hybrid - hmin) / (hmax - hmin)

                avg = hybrid.mean(axis=1)
                mm3_per_vox = self.config.GRID_SPACING ** 3
                min_voxels = max(4, int(raykill_min / mm3_per_vox))

                if len(coords) > 0:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(coords)
                    radius = self.config.GRID_SPACING * 2.5
                    try:
                        neighbor_counts = tree.query_ball_point(
                            coords, r=radius, return_length=True
                        )
                    except TypeError:
                        neighbor_counts = np.array([
                            len(tree.query_ball_point(c, r=radius))
                            for c in coords
                        ])
                    isolated = neighbor_counts < min_voxels
                    avg[isolated] *= 0.1

                n_time = hybrid.shape[1]
                if n_time > temporal_window:
                    kernel = np.ones(temporal_window) / temporal_window
                    for v in range(0, len(hybrid), 1000):
                        v_end = min(v + 1000, len(hybrid))
                        for vi in range(v, v_end):
                            if np.any(hybrid[vi] > 0):
                                smoothed = np.convolve(
                                    (hybrid[vi] > 0).astype(float),
                                    kernel, mode='same'
                                )
                                drop = smoothed < self.config.TEMPORAL_MIN_FRAC
                                hybrid[vi, drop] = 0

                    avg = hybrid.mean(axis=1)

                score = self._compute_band_metric(avg, coords, band_name)
                band_scores.append(score)

                del hybrid, mi, dice
                gc.collect()

            if band_scores:
                confidence_weight = {
                    'HIGH': 1.0, 'MEDIUM': 0.8,
                    'LOW-MEDIUM': 0.6, 'LOW': 0.4,
                }
                w = confidence_weight.get(
                    self.config.LOSO_TARGETS[band_name].get('confidence', 'MEDIUM'),
                    0.5
                )
                total_score += w * np.mean(band_scores)
                n_evaluations += 1

        return total_score / max(1, n_evaluations)

    def optimize_l2(self, all_subjects, l1_params_per_band):
        Logger.section("L2 OPTIMIZATION (Phase 3 parameters)")
        Logger.info(f"  Optimizing: {list(self.config.L2_BOUNDS.keys())}")
        Logger.info(f"  Fixed (Phase 1): SIGMA_BASE={self.config.SIGMA_BASE}, "
                    f"CSD_SIGMA={self.config.CSD_SIGMA}, "
                    f"HEMI_ISO={self.config.HEMISPHERE_ISOLATION}")

        candidates = self._generate_candidates(
            self.config.L2_BOUNDS,
            self.config.LOSO_N_CANDIDATES_L2,
        )
        candidates.insert(0, dict(self.config.L2_DEFAULTS))

        best_score = -np.inf
        best_l2 = dict(self.config.L2_DEFAULTS)

        for i, cand in enumerate(tqdm(candidates, desc="  L2 candidates", ncols=80)):
            score = self._evaluate_l2_candidate(
                cand, all_subjects, l1_params_per_band
            )
            if score > best_score:
                best_score = score
                best_l2 = cand

        for r in range(2):
            refined = self._refine_around(best_l2, self.config.L2_BOUNDS, 8, r + 1)
            for cand in refined:
                score = self._evaluate_l2_candidate(
                    cand, all_subjects, l1_params_per_band
                )
                if score > best_score + self.config.LOSO_CONVERGENCE_TOL:
                    best_score = score
                    best_l2 = cand

        Logger.info(f"\n  {'='*50}")
        Logger.info(f"  OPTIMAL L2:")
        for k, v in sorted(best_l2.items()):
            Logger.info(f"    {k:30s}: {v}")
        Logger.info(f"  Score: {best_score:.4f}")
        Logger.info(f"  {'='*50}")

        return best_l2

    def optimize_all(self, all_subjects):
        Logger.section("AUTONOMOUS LOSO OPTIMIZATION v5.1")
        Logger.info(f"Subjects: {len(all_subjects)}")
        Logger.info(f"L1 params: {list(self.config.L1_BOUNDS.keys())}")
        Logger.info(f"L2 params: {list(self.config.L2_BOUNDS.keys())}")

        l1_results = {}
        fold_details = {}

        for band_name in self.config.BANDS:
            l1_params, folds = self.optimize_band_l1(band_name, all_subjects)
            l1_results[band_name] = l1_params
            fold_details[band_name] = folds

            self._save_results({
                'l1': l1_results, 'l2': {}, 'fold_details': fold_details,
            }, intermediate=True)
            gc.collect()

        l2_results = self.optimize_l2(all_subjects, l1_results)

        results = {
            'l1': l1_results,
            'l2': l2_results,
            'fold_details': fold_details,
        }
        self._save_results(results, intermediate=False)

        return results

    def _save_results(self, results, intermediate=False):
        tag = self.config.VERSION_TAG
        out = Path(self.config.DATA_PATH)
        suffix = "_intermediate" if intermediate else ""

        json_data = {
            'l1': {
                band: {k: v for k, v in params.items() if not k.startswith('_')}
                for band, params in results.get('l1', {}).items()
            },
            'l2': results.get('l2', {}),
            'version': tag,
            'timestamp': datetime.now().isoformat(),
        }

        json_path = out / f"loso_weights_{tag}{suffix}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=float)

        if not intermediate:
            pkl_path = out / f"loso_details_{tag}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(results, f)

        Logger.success(f"LOSO results saved: {json_path.name}")

    def load_results(self):
        tag = self.config.VERSION_TAG
        out = Path(self.config.DATA_PATH)

        json_path = out / f"loso_weights_{tag}.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            if 'l1' in data:
                Logger.success(f"Loaded v5.1 LOSO results")
                return data

        # v5.1: No backward compat with v5.0/v4.x (different envelope)
        Logger.warn("No v5.1 LOSO results found — will use defaults")
        return None

    def _convert_legacy(self, old_data):
        """Convert v4.x/v5.0 weight format to v5.1 format."""
        l1 = {}
        for band_name in self.config.BANDS:
            band_data = old_data.get(band_name, {})

            if isinstance(band_data, (int, float)):
                l1[band_name] = {
                    **self.config.L1_DEFAULTS[band_name],
                    'mi_weight': float(band_data),
                }
            elif isinstance(band_data, dict):
                merged = dict(self.config.L1_DEFAULTS[band_name])
                for k in ('mi_weight', 'contrast', 'boost', 'penalty', 'softening'):
                    if k in band_data:
                        merged[k] = float(band_data[k])
                l1[band_name] = merged
            else:
                l1[band_name] = dict(self.config.L1_DEFAULTS[band_name])

        return {
            'l1': l1,
            'l2': dict(self.config.L2_DEFAULTS),
        }


# ============================================================================
# AR(1) PREWHITENER
# ============================================================================

class AR1Prewhitener:
    """Vectorized AR(1) prewhitening."""

    @staticmethod
    def apply_chunked(volume, config, mmap_mode=False):
        x, y, z, T = volume.shape

        Logger.info(f"    [AR1] Vectorized prewhitening ({x*y*z} voxels, T={T})...")

        vol_2d = volume.reshape(-1, T)
        N = vol_2d.shape[0]

        active_mask = np.std(vol_2d, axis=1) > 1e-10
        active_indices = np.where(active_mask)[0]
        n_active = len(active_indices)

        if n_active == 0:
            Logger.warn("    [AR1] No active voxels!")
            return volume

        Logger.info(f"    [AR1] Active voxels: {n_active}")

        chunk_size = min(10000, n_active)
        all_phis = []

        for c_start in tqdm(range(0, n_active, chunk_size),
                           desc="    AR(1) vectorized", ncols=80, leave=False):
            c_end = min(c_start + chunk_size, n_active)
            idx = active_indices[c_start:c_end]
            chunk = vol_2d[idx, :].astype(np.float64)

            y_t = chunk[:, 1:]
            y_t1 = chunk[:, :-1]

            mean_yt = np.mean(y_t, axis=1, keepdims=True)
            mean_yt1 = np.mean(y_t1, axis=1, keepdims=True)

            cov_num = np.mean(
                (y_t - mean_yt) * (y_t1 - mean_yt1), axis=1
            )

            var_yt1 = np.var(y_t1, axis=1, ddof=1)
            var_yt1 = np.maximum(var_yt1, 1e-12)

            phi = cov_num / var_yt1
            phi = np.clip(phi, -0.99, 0.99)
            all_phis.extend(phi.tolist())

            c = mean_yt.squeeze() - phi * mean_yt1.squeeze()

            predicted = c[:, None] + phi[:, None] * y_t1
            resid = y_t - predicted

            first_resid = chunk[:, 0] - np.mean(chunk, axis=1)

            new_chunk = np.zeros_like(chunk, dtype=np.float32)
            new_chunk[:, 0] = first_resid
            new_chunk[:, 1:] = resid

            vol_2d[idx] = new_chunk

        active_data = vol_2d[active_mask]

        rng = np.random.RandomState(config.RANDOM_SEED)
        if len(active_data) > 100000:
            sample_idx = rng.choice(len(active_data), 100000, replace=False)
            sample = active_data[sample_idx]
        else:
            sample = active_data

        g_mean = float(np.mean(sample))
        g_std = float(np.std(sample, ddof=1))

        if g_std > 1e-10:
            if mmap_mode and isinstance(volume, np.memmap):
                for s in range(0, N, chunk_size):
                    e = min(s + chunk_size, N)
                    chunk_mask = active_mask[s:e]
                    vol_2d[s:e][chunk_mask] = (
                        (vol_2d[s:e][chunk_mask] - g_mean) / g_std
                    )
                volume.flush()
            else:
                vol_2d[active_mask] = (vol_2d[active_mask] - g_mean) / g_std
                volume = vol_2d.reshape(x, y, z, T)

        Logger.success(f"    [AR1] Done: φ_median={np.median(all_phis):.3f}, "
                      f"μ={g_mean:.4f}, σ={g_std:.4f}")

        return volume.astype(np.float32)


# ============================================================================
# QUALITY CONTROL
# ============================================================================

class QualityControl:
    """Basic QC metrics for generated volumes."""

    def __init__(self, grid, config):
        self.grid = grid
        self.config = config

    def physiological_distribution(self, volume, band_name):
        x, y, z, T = volume.shape
    
        vol_mean_abs = np.mean(np.abs(volume), axis=3)
        active = vol_mean_abs > 0.001

        if not active.any():
            return {'total_active': 0, 'coverage_pct': 0.0}

        y_indices = np.where(active)[1]
        y_coords = self.grid.affine[1, 1] * y_indices + self.grid.affine[1, 3]

        posterior_frac = np.mean(y_coords < -40)
        anterior_frac = np.mean(y_coords > 0)
    
        x_indices = np.where(active)[0]
        x_coords = self.grid.affine[0, 0] * x_indices + self.grid.affine[0, 3]
        left_frac = np.mean(x_coords < 0)

        return {
            'total_active': int(active.sum()),
            'coverage_pct': float(100.0 * active.sum() / active.size),
            'posterior_frac': float(posterior_frac),
            'anterior_frac': float(anterior_frac),
            'left_hemisphere_frac': float(left_frac),
            'mean_value': float(vol_mean_abs[active].mean()),
            'std_value': float(vol_mean_abs[active].std()),
        }


# ============================================================================
# CHUNKED POST-PROCESSOR — v5.1.0: HRF VARIANCE SCALING REMOVED
# ============================================================================

class ChunkedPostProcessor:
    """Post-processing pipeline."""

    def __init__(self, config, grid):
        self.config = config
        self.grid = grid

    def process_volume(self, volume, band_name, l1_params, l2_params,
                       mmap_mode=False, band_idx=0):
        x, y, z, T = volume.shape
        atlas = self.grid.atlas_idx

        smoothing_fwhm = l1_params.get(
            'smoothing_fwhm',
            self.config.L1_DEFAULTS[band_name]['smoothing_fwhm']
        )
        keep_top_pct = l1_params.get(
            'keep_top_pct',
            self.config.L1_DEFAULTS[band_name]['keep_top_pct']
        )
        temporal_window = int(l2_params.get(
            'temporal_window',
            self.config.L2_DEFAULTS['temporal_window']
        ))
        ar1_enabled = l2_params.get(
            'ar1_enabled',
            self.config.L2_DEFAULTS['ar1_enabled']
        )
        raykill_min = l2_params.get(
            'raykill_min_cluster_mm3',
            self.config.L2_DEFAULTS['raykill_min_cluster_mm3']
        )

        Logger.info(f"    PostProc: FWHM={smoothing_fwhm:.1f}, "
                    f"K={keep_top_pct:.2f}, TW={temporal_window}, "
                    f"AR1={ar1_enabled}")

        # 1. AR(1)
        if ar1_enabled:
            volume = AR1Prewhitener.apply_chunked(volume, self.config, mmap_mode)

        # 2. Z-score
        if self.config.APPLY_ZSCORE:
            volume = self._apply_zscore_fixed(volume, mmap_mode, band_idx)

        # 3-6. Per-timepoint
        smooth_sigma = None
        if smoothing_fwhm > 0:
            smooth_sigma = [smoothing_fwhm / 2.355 / self.config.GRID_SPACING] * 3

        Logger.info(f"    Processing {T} timepoints...")
        for t in range(T):
            if t % max(1, T // 10) == 0:
                Logger.info(f"      t={t+1}/{T}")

            vol_t = np.array(volume[:, :, :, t])

            if smooth_sigma is not None:
                vol_t = self._smooth_3d(vol_t, smooth_sigma)

            vol_t = self._sparsify(vol_t, atlas, band_name, keep_top_pct)

            if self.config.RAYKILL_ENABLE:
                vol_t = self._ray_kill(vol_t, raykill_min)

            volume[:, :, :, t] = vol_t

            if mmap_mode and hasattr(volume, 'flush') and t % 5 == 0:
                volume.flush()

            del vol_t

        # 6. Temporal consistency
        if self.config.TEMPORAL_FILTERING:
            volume = self._temporal_consistency(volume, temporal_window, mmap_mode)

        # 7. HRF Convolution (LAST — v5.1: no variance scaling)
        Logger.info("    [HRF] Applying HRF convolution...")
        if mmap_mode and isinstance(volume, np.memmap):
            volume = HRFConvolver.convolve_volume_mmap(
                volume, self.config.SEGMENT_DURATION, self.config
            )
        else:
            volume = HRFConvolver.convolve_volume(
                volume, self.config.SEGMENT_DURATION, self.config
            )
        
        return volume

    def _apply_zscore_fixed(self, volume, mmap_mode, band_idx=0):
        x, y, z, T = volume.shape
        vol_flat = volume.reshape(-1, T)

        active_mask = np.any(vol_flat != 0, axis=1)
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            Logger.warn("    Z-score: No active voxels!")
            return volume

        rng = np.random.RandomState(self.config.RANDOM_SEED + band_idx)
        max_sample = 100000

        if len(active_indices) > max_sample:
            sample_idx = rng.choice(
                len(active_indices), max_sample, replace=False
            )
            sample_indices = active_indices[sample_idx]
        else:
            sample_indices = active_indices

        sample = vol_flat[sample_indices]
        g_mean = float(np.mean(sample))
        g_std = float(np.std(sample, ddof=1))

        if g_std < 1e-10:
            Logger.warn("    Z-score: Near-zero std, skipping")
            return volume

        Logger.info(f"    Z-score: μ={g_mean:.4f}, σ={g_std:.4f}")

        if mmap_mode and isinstance(volume, np.memmap):
            chunk = 5000
            for start in range(0, len(active_indices), chunk):
                end = min(start + chunk, len(active_indices))
                idx = active_indices[start:end]
                vol_flat[idx] = (vol_flat[idx] - g_mean) / g_std
            volume.flush()
        else:
            vol_flat[active_mask] = (vol_flat[active_mask] - g_mean) / g_std
            volume = vol_flat.reshape(x, y, z, T)

        return volume

    def _smooth_3d(self, vol_t, sigma_voxels):
        mask = vol_t > 0
        if not mask.any():
            return vol_t
        smoothed = gaussian_filter(vol_t, sigma=sigma_voxels,
                                   mode='constant', cval=0.0)
        return np.where(mask, smoothed, 0.0)
    
    def _sparsify(self, vol_t, atlas, band_name, keep_top_pct):
        n_gm = int(self.grid.gm_mask.sum())
        min_voxels = self.config.MIN_VOXELS_GLOBAL.get(band_name, 15000)
        target_active = max(int(keep_top_pct * n_gm), min_voxels)
    
        current_active = (vol_t > 0) & self.grid.gm_mask
        n_current = int(current_active.sum())
    
        if n_current <= target_active:
            return vol_t
    
        gm_vals = vol_t[self.grid.gm_mask]
        positive_vals = gm_vals[gm_vals > 0]
    
        if len(positive_vals) == 0:
            return vol_t
    
        k = min(target_active, len(positive_vals))
        thr = np.partition(positive_vals, -k)[-k]
        vol_t[self.grid.gm_mask & (vol_t < thr)] = 0.0
    
        return vol_t

    def _ray_kill(self, vol_t, min_cluster_mm3):
        mm3_per_vox = self.config.GRID_SPACING ** 3
        min_voxels = max(8, int(min_cluster_mm3 / mm3_per_vox))

        active_mask = (vol_t > 0) & self.grid.gm_mask
        if not active_mask.any():
            return vol_t

        labeled, n_clusters = label(active_mask)
        if n_clusters == 0:
            return vol_t

        counts = np.bincount(labeled.ravel())
        small_labels = np.where(counts < min_voxels)[0]
        small_labels = small_labels[small_labels > 0]

        if len(small_labels) > 0:
            vol_t[np.isin(labeled, small_labels)] = 0.0

        n_remaining = int((vol_t > 0).sum())
        n_orig = int(active_mask.sum())
        keep_floor = max(
            int(self.config.RAYKILL_KEEP_FLOOR_PCT * n_orig),
            self.config.RAYKILL_KEEP_FLOOR_ABS,
        )

        if n_remaining < keep_floor and n_orig > 0:
            all_vals = vol_t.copy()
            all_vals[~active_mask] = 0
            active_vals = all_vals[active_mask]
            K = min(n_orig, keep_floor)
            thr = np.partition(active_vals, -K)[-K]
            rescue = active_mask & (all_vals >= thr)
            vol_t = np.where(rescue, all_vals, vol_t)
        return vol_t

    def _temporal_consistency(self, volume, window, mmap_mode):
        x, y, z, T = volume.shape
        chunk_t = 50

        Logger.info(f"    Temporal consistency (window={window})...")

        for t_start in range(0, T, chunk_t):
            t_end = min(t_start + chunk_t, T)
            chunk = np.array(volume[:, :, :, t_start:t_end])

            mask = (chunk > 0).astype(np.float32)
            smoothed = uniform_filter1d(mask, size=window, axis=3, mode='reflect')
            keep = smoothed >= self.config.TEMPORAL_MIN_FRAC
            chunk[~keep] = 0.0

            volume[:, :, :, t_start:t_end] = chunk

            if mmap_mode and hasattr(volume, 'flush'):
                volume.flush()

            del chunk, mask, smoothed, keep

        return volume


# ============================================================================
# MAIN PIPELINE v5.1
# ============================================================================

class GroupEEGtoFMRIPipeline:
    """
    EEG-to-fMRI voxel projection pipeline.
    
    Phase 1: EEG → MI/Dice cache (v5.1: relative band power + rank Dice)
    Phase 2: Two-level LOSO optimization
    Phase 3: Volume building with HRF convolution (no variance scaling)
    
    Output: SPM/CONN/FreeSurfer-compatible NIfTI files.
    """

    def __init__(self, config):
        self.config = config
        self.start_time = datetime.now()
        self.cache_mgr = CacheManager(config)
        np.random.seed(config.RANDOM_SEED)

        Logger.info(f"Pipeline v5.1.0 Frequency-Specific")
        Logger.info(f"  Data: {config.DATA_PATH}")
        Logger.info(f"  Subjects: {len(config.SUBJECTS)}")
        Logger.info(f"  Cache: {config.CACHE_DIR} (v5.1 only, no fallback)")
        Logger.info(f"  Envelope: relative band power (NEW)")
        Logger.info(f"  Dice: rank-based (NEW)")

    # ────────────────────────────────────────────────────────────
    # PHASE 1
    # ────────────────────────────────────────────────────────────

    def _process_one_subject(self, subject_file):
        """Process single subject: EEG → MI/Dice scores."""
        subject_id = subject_file.split('_')[0]
        Logger.section(f"PHASE 1: {subject_id}")

        # Check cache
        if self.cache_mgr.all_bands_cached(subject_id):
            Logger.info(f"  ✅ All bands cached, loading...")
            mi_dict, dice_dict = {}, {}
            voxel_coords = None
            for band in self.config.BANDS:
                mi, dice, coords = self.cache_mgr.load_subject(subject_id, band)
                mi_dict[band] = mi
                dice_dict[band] = dice
                if voxel_coords is None:
                    voxel_coords = coords
            return subject_id, mi_dict, dice_dict, voxel_coords

        # Load EEG
        eeg_path = Path(self.config.DATA_PATH) / subject_file
        if not eeg_path.exists():
            Logger.error(f"  File not found: {eeg_path}")
            return subject_id, None, None, None

        try:
            mat = scipy.io.loadmat(eeg_path, squeeze_me=True, struct_as_record=False)
            from scipy.signal import resample_poly
            eeg_raw = np.array(mat['dataRest']).astype(np.float32)
            eeg_raw = resample_poly(eeg_raw, 250, 2500, axis=1).astype(np.float32)
            Logger.info(f"  EEG shape: {eeg_raw.shape}")
        except Exception as e:
            Logger.error(f"  Load failed: {e}")
            return subject_id, None, None, None

        # Preprocessing
        eeg = SignalProcessor.bandpass(eeg_raw, self.config.FS, 1, 45)
        eeg = SignalProcessor.notch(eeg, self.config.FS, 50.0)

        ch_coords, ch_order = CoordinateSystem.load_coordinates()

        ica = ICAArtifactRemover(self.config)
        eeg, _ = ica.clean(eeg, self.config.FS, ch_order)  # 62ch → 62ch (ICA_N_COMPONENTS=62)

        eeg = CSDReferencer.apply(eeg, ch_coords, ch_order, self.config)

        # Segmentation
        seg_samples = int(self.config.FS * self.config.SEGMENT_DURATION)
        n_segments = eeg.shape[1] // seg_samples
        segments = np.stack([
            eeg[:, i * seg_samples:(i + 1) * seg_samples]
            for i in range(n_segments)
        ])
        Logger.info(f"  Segments: {n_segments} × {seg_samples} samples")

        # Grid + Signatures + MI/Dice
        grid_cache_file = Path(self.config.DATA_PATH) / "grid_cache_v51.pkl"

        if grid_cache_file.exists():
            Logger.info("Loading VoxelGrid from cache...")
            with open(grid_cache_file, 'rb') as f:
                grid = pickle.load(f)
            Logger.success(f"Grid loaded: {len(grid.coords_gm)} GM voxels")
        else:
            Logger.info("Creating VoxelGrid (this will take 5-10 minutes)...")
            grid = VoxelGrid(self.config)
            with open(grid_cache_file, 'wb') as f:
                pickle.dump(grid, f)
            Logger.success(f"Grid cached for future runs")

        sig_computer = SignatureComputer(self.config)
        mi_dict, dice_dict = {}, {}

        for band_name, freq_range in self.config.BANDS.items():
            Logger.info(f"\n  Band: {band_name.upper()} "
                       f"({freq_range[0]}-{freq_range[1]} Hz)")

            try:
                voxel_sigs = sig_computer.compute(
                    grid.coords_gm, ch_coords, ch_order, band_name=band_name
                )

                # v5.1.0: Use RELATIVE band power instead of absolute
                snapshots = np.stack([
                    SignalProcessor.hilbert_envelope_relative(
                        seg, freq_range, self.config.FS
                    )
                    for seg in tqdm(segments, desc="    Envelopes (relative)", ncols=80)
                ])

                Logger.info(f"    Envelope stats: mean={snapshots.mean():.4f}, "
                           f"std={snapshots.std():.4f}, "
                           f"range=[{snapshots.min():.4f}, {snapshots.max():.4f}]")

                mi_computer = MIDiceComputer(self.config)
                mi_scores, dice_scores = mi_computer.compute(voxel_sigs, snapshots)

                mi_dict[band_name] = mi_scores
                dice_dict[band_name] = dice_scores

                self.cache_mgr.save(
                    subject_id, band_name, mi_scores, dice_scores, grid.coords_gm
                )

                del snapshots, voxel_sigs
                gc.collect()

            except Exception as e:
                Logger.error(f"  {band_name} failed: {e}")
                import traceback
                traceback.print_exc()
                n_vox = len(grid.coords_gm)
                mi_dict[band_name] = np.zeros((n_vox, n_segments), dtype=np.float32)
                dice_dict[band_name] = np.zeros_like(mi_dict[band_name])

        Logger.success(f"  {subject_id}: Phase 1 complete")
        return subject_id, mi_dict, dice_dict, grid.coords_gm

    def _phase1(self):
        """Phase 1: Process all subjects."""
        Logger.section("PHASE 1: MI/DICE COMPUTATION (v5.1 relative power)")
        subject_data = []

        for i, sf in enumerate(self.config.SUBJECTS):
            Logger.info(f"\n📍 Subject {i+1}/{len(self.config.SUBJECTS)}: {sf}")
            result = self._process_one_subject(sf)
            if result[1] is not None:
                subject_data.append(result)
            else:
                Logger.warn(f"  Skipping {result[0]}")

        Logger.success(
            f"Phase 1: {len(subject_data)}/{len(self.config.SUBJECTS)} processed"
        )
        return subject_data

    # ────────────────────────────────────────────────────────────
    # PHASE 2
    # ────────────────────────────────────────────────────────────

    def _phase2(self, subject_data):
        """Phase 2: Two-level LOSO optimization."""
        Logger.section("PHASE 2: AUTONOMOUS LOSO OPTIMIZATION")

        all_subjects = [d[0] + "_EO.mat" for d in subject_data]
        Logger.info(f"  Subjects: {len(all_subjects)}")

        optimizer = LOSOOptimizer(self.config)
        results = optimizer.optimize_all(all_subjects)

        if self.config.LOSO_EXCLUDE_OUTLIERS:
            outliers = self._detect_outliers(results.get('fold_details', {}))
            if outliers:
                Logger.warn(f"  Outlier subjects: {outliers}")

        return results

    def _detect_outliers(self, fold_details):
        """Detect outlier subjects from fold results."""
        outliers = []
        alpha_folds = fold_details.get('alpha', [])

        for r in alpha_folds:
            m = r.get('test_metrics', {})
            if not m:
                continue

            pa = m.get('pa_ratio', 999)
            occ = m.get('occipital_pct', 999)
            reasons = []

            if pa < self.config.LOSO_OUTLIER_ALPHA_PA_MIN:
                reasons.append(f"P/A={pa:.2f}")
            if occ < self.config.LOSO_OUTLIER_ALPHA_OCC_MIN:
                reasons.append(f"Occ={occ:.1f}%")

            if reasons:
                subj = r.get('test_subject', '?')
                outliers.append(subj)
                Logger.warn(f"    Outlier {subj}: {', '.join(reasons)}")

        return outliers

    # ────────────────────────────────────────────────────────────
    # PHASE 3
    # ────────────────────────────────────────────────────────────

    def _phase3(self, subject_data, loso_results):
        """Phase 3: Build NIfTI volumes with HRF convolution."""
        Logger.section("PHASE 3: VOLUME BUILDING + HRF CONVOLUTION")

        tag = self.config.VERSION_TAG
        output_path = Path(self.config.DATA_PATH)
        temp_path = output_path / f"temp_mmap_{tag}"
        temp_path.mkdir(exist_ok=True)

        l1_all = loso_results.get('l1', {})
        l2_params = loso_results.get('l2', dict(self.config.L2_DEFAULTS))

        grid = VoxelGrid(self.config)
        spatial_prior = SpatialPrior(self.config)

        use_mmap = self.config.PHASE3_USE_MMAP
        mmap_threshold_gb = self.config.PHASE3_MMAP_THRESHOLD_GB

        checkpoint_file = output_path / f"phase3_checkpoint_{tag}.json"
        completed = self._load_checkpoint(checkpoint_file)
        remaining = [
            (i, sd) for i, sd in enumerate(subject_data)
            if sd[0] not in completed
        ]

        if not remaining:
            Logger.success("  All subjects already processed!")
            return

        Logger.info(f"  To process: {len(remaining)}/{len(subject_data)}")
        Logger.info(f"  L2 params: {json.dumps(l2_params, indent=2, default=str)}")

        for idx, (orig_idx, subject_info) in enumerate(remaining):
            subject_id = subject_info[0]
            Logger.info(f"\n{'═'*60}")
            Logger.info(f"  [{idx+1}/{len(remaining)}] {subject_id}")
            Logger.info(f"{'═'*60}")

            try:
                self._process_subject_phase3(
                    subject_id, subject_info,
                    l1_all, l2_params,
                    grid, spatial_prior,
                    output_path, temp_path, tag,
                    use_mmap, mmap_threshold_gb,
                )

                completed.add(subject_id)
                self._save_checkpoint(checkpoint_file, completed)
                gc.collect()

            except Exception as e:
                Logger.error(f"  ❌ {subject_id} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        self._phase3_summary(subject_data, completed, output_path, tag)

        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)

    def _process_subject_phase3(self, subject_id, subject_info,
                                l1_all, l2_params,
                                grid, spatial_prior,
                                output_path, temp_path, tag,
                                use_mmap, mmap_threshold_gb):
        """Process one subject through Phase 3."""
        mi_dict = subject_info[1]
        dice_dict = subject_info[2]
        voxel_coords = subject_info[3]

        n_voxels = len(voxel_coords)
        n_time = next(iter(mi_dict.values())).shape[1]
        volume_shape = (*grid.shape, n_time)
        volume_size_gb = np.prod(volume_shape) * 4 / (1024 ** 3)

        Logger.info(f"    Voxels: {n_voxels}, Timepoints: {n_time}")
        Logger.info(f"    Dense volume: {volume_shape} = {volume_size_gb:.2f} GB")

        for b_idx, band_name in enumerate(self.config.BANDS):
            Logger.info(f"\n    🎵 [{b_idx+1}/{len(self.config.BANDS)}] "
                       f"{band_name.upper()}")

            l1_params = l1_all.get(
                band_name, self.config.L1_DEFAULTS[band_name]
            )

            Logger.info(f"      L1: MI={l1_params.get('mi_weight', 0.55):.3f}, "
                       f"C={l1_params.get('contrast', 0.6):.3f}, "
                       f"B={l1_params.get('boost', 1.5):.3f}, "
                       f"K={l1_params.get('keep_top_pct', 0.4):.3f}, "
                       f"FWHM={l1_params.get('smoothing_fwhm', 1.0):.2f}")

            mi = mi_dict[band_name]
            dice = dice_dict[band_name]

            hybrid = self._compute_hybrid(
                mi, dice, l1_params, spatial_prior, voxel_coords, band_name
            )

            is_mmap = False
            mmap_file = None
            volume = None

            try:
                if use_mmap and volume_size_gb > mmap_threshold_gb:
                    Logger.info(f"      Using memory-mapped volume")
                    volume, mmap_file = self._build_volume_mmap(
                        hybrid, voxel_coords, grid, temp_path, band_name
                    )
                    is_mmap = True
                else:
                    volume = self._build_volume_dense(
                        hybrid, voxel_coords, grid
                    )

                del hybrid
                gc.collect()

                processor = ChunkedPostProcessor(self.config, grid)
                volume = processor.process_volume(
                    volume, band_name, l1_params, l2_params,
                    mmap_mode=is_mmap, band_idx=b_idx,
                )

                qc = QualityControl(grid, self.config)
                qc_metrics = qc.physiological_distribution(volume, band_name)
                Logger.info(f"      QC: coverage={qc_metrics.get('coverage_pct', 0):.1f}%, "
                           f"post={qc_metrics.get('posterior_frac', 0):.2f}, "
                           f"ant={qc_metrics.get('anterior_frac', 0):.2f}")

                self._save_nifti(
                    volume, subject_id, band_name, grid,
                    qc_metrics, l1_params, l2_params,
                    output_path, tag,
                )

                Logger.success(f"      ✅ {band_name} complete")

            finally:
                del volume
                gc.collect()

                if is_mmap and mmap_file is not None and mmap_file.exists():
                    try:
                        mmap_file.unlink(missing_ok=True)
                    except Exception:
                        pass

    def _compute_hybrid(self, mi, dice, l1_params, spatial_prior,
                        coords, band_name):
        mi_w = l1_params.get('mi_weight', 0.55)
        boost = l1_params.get('boost', 1.5)
        penalty = l1_params.get('penalty', 0.0)
        softening = l1_params.get('softening', 0.5)
        contrast = l1_params.get('contrast', 1.0)

        hybrid = mi_w * mi.astype(np.float32) + (1.0 - mi_w) * dice.astype(np.float32)

        hybrid = spatial_prior.apply(
            hybrid, coords, band_name,
            boost=boost, penalty=penalty, softening=softening,
        )

        if contrast != 1.0:
            np.power(np.maximum(hybrid, 1e-12), contrast, out=hybrid)

        hmin, hmax = hybrid.min(), hybrid.max()
        eps = float(np.finfo(np.float32).eps)
        if hmax > hmin + eps:
            hybrid = (hybrid - hmin) / (hmax - hmin)
            np.clip(hybrid, 0, 1, out=hybrid)
        else:
            Logger.warn(f"      ⚠ Flat hybrid for {band_name}")
            hybrid.fill(0)

        return hybrid

    def _build_volume_dense(self, hybrid_data, voxel_coords, grid):
        """Build dense numpy volume from hybrid scores."""
        n_voxels, n_time = hybrid_data.shape
        x, y, z = grid.shape

        volume = np.zeros((x, y, z, n_time), dtype=np.float32)

        coords_homo = np.hstack([voxel_coords, np.ones((n_voxels, 1))])
        vox_indices = (grid.inv_affine @ coords_homo.T).T[:, :3].astype(int)

        valid = (
            (vox_indices[:, 0] >= 0) & (vox_indices[:, 0] < x) &
            (vox_indices[:, 1] >= 0) & (vox_indices[:, 1] < y) &
            (vox_indices[:, 2] >= 0) & (vox_indices[:, 2] < z)
        )

        volume[
            vox_indices[valid, 0],
            vox_indices[valid, 1],
            vox_indices[valid, 2], :
        ] = hybrid_data[valid]

        Logger.info(f"      Dense: {100.0 * valid.sum() / n_voxels:.1f}% placed")
        return volume

    def _build_volume_mmap(self, hybrid_data, voxel_coords, grid,
                           temp_path, band_name):
        """Build memory-mapped volume from hybrid scores."""
        n_voxels, n_time = hybrid_data.shape
        x, y, z = grid.shape

        mmap_file = temp_path / f"vol_{band_name}_{int(time.time()*1000)}.dat"
        volume = np.memmap(
            mmap_file, dtype='float32', mode='w+', shape=(x, y, z, n_time)
        )

        chunk_size = self.config.PHASE3_CHUNK_SIZE

        for start in tqdm(range(0, n_voxels, chunk_size),
                         desc="      Mmap fill", ncols=80, leave=False):
            end = min(start + chunk_size, n_voxels)
            chunk_coords = voxel_coords[start:end]
            chunk_data = hybrid_data[start:end, :]

            coords_homo = np.hstack([
                chunk_coords, np.ones((len(chunk_coords), 1))
            ])
            vox_indices = (grid.inv_affine @ coords_homo.T).T[:, :3].astype(int)

            valid = (
                (vox_indices[:, 0] >= 0) & (vox_indices[:, 0] < x) &
                (vox_indices[:, 1] >= 0) & (vox_indices[:, 1] < y) &
                (vox_indices[:, 2] >= 0) & (vox_indices[:, 2] < z)
            )

            volume[
                vox_indices[valid, 0],
                vox_indices[valid, 1],
                vox_indices[valid, 2], :
            ] = chunk_data[valid]

        volume.flush()
        Logger.info(f"      Mmap volume flushed: {mmap_file.name}")
        return volume, mmap_file

    # ────────────────────────────────────────────────────────────
    # SAVE OUTPUTS
    # ────────────────────────────────────────────────────────────

    def _save_nifti(self, volume, subject_id, band_name, grid,
                    qc_metrics, l1_params, l2_params,
                    output_path, tag):
        """Save NIfTI + QC metrics + parameter log."""
        base = f"{subject_id}_{band_name}"

        vol_data = np.array(volume) if isinstance(volume, np.memmap) else volume
        nifti_img = nib.Nifti1Image(vol_data, affine=grid.affine)

        nifti_img.header['pixdim'][4] = float(self.config.SEGMENT_DURATION)
        nifti_img.header.set_xyzt_units('mm', 'sec')
        nifti_img.header.set_qform(grid.affine, code=1)
        nifti_img.header.set_sform(grid.affine, code=1)

        desc = (f"EEG2fMRI-v5.1 {band_name} "
                f"MI={l1_params.get('mi_weight', 0):.2f} "
                f"HRF-conv TR={self.config.SEGMENT_DURATION}s "
                f"RelPow RankDice")
        nifti_img.header['descrip'] = desc.encode('ascii')[:80]

        # Save .nii.gz (CONN, general use)
        nii_gz_path = output_path / f"{base}_bold_{tag}.nii.gz"
        nib.save(nifti_img, nii_gz_path)

        # Save .nii (SPM compatibility)
        nii_path = output_path / f"{base}_bold_{tag}.nii"
        nib.save(nifti_img, nii_path)

        Logger.info(f"      Saved: {nii_gz_path.name} + {nii_path.name}")

        # QC metrics CSV
        csv_path = output_path / f"{base}_qc_{tag}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for k, v in sorted(qc_metrics.items()):
                writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])

        # Parameter log JSON
        param_log = {
            'subject': subject_id,
            'band': band_name,
            'version': tag,
            'l1_params': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in l1_params.items()},
            'l2_params': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in l2_params.items()},
            'qc': qc_metrics,
            'timestamp': datetime.now().isoformat(),
            'hrf_convolved': True,
            'hrf_variance_scaling': False,  # v5.1: explicitly noted
            'envelope_type': 'relative_band_power',  # v5.1
            'dice_type': 'rank_based',  # v5.1
            'tr_seconds': self.config.SEGMENT_DURATION,
        }
        json_path = output_path / f"{base}_params_{tag}.json"
        with open(json_path, 'w') as f:
            json.dump(param_log, f, indent=2, default=str)

    # ────────────────────────────────────────────────────────────
    # CHECKPOINT
    # ────────────────────────────────────────────────────────────

    def _load_checkpoint(self, checkpoint_file):
        if not checkpoint_file.exists():
            return set()
        try:
            with open(checkpoint_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                return set(data)
            Logger.warn(f"  Checkpoint unexpected format: {type(data)}")
            return set()
        except (json.JSONDecodeError, TypeError, IOError) as e:
            Logger.warn(f"  Checkpoint read error: {e}")
            return set()

    def _save_checkpoint(self, checkpoint_file, completed):
        """Atomic checkpoint save."""
        try:
            temp = checkpoint_file.parent / f".ckpt_tmp_{os.getpid()}"
            with open(temp, 'w') as f:
                json.dump(list(completed), f)
                f.flush()
                if hasattr(os, 'fsync'):
                    os.fsync(f.fileno())
            temp.replace(checkpoint_file)
        except Exception as e:
            Logger.error(f"  Checkpoint save error: {e}")

    # ────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────

    def _phase3_summary(self, subject_data, completed, output_path, tag):
        """Print Phase 3 summary."""
        Logger.section("PHASE 3 SUMMARY")
        total = len(subject_data)
        n_done = len(completed)
        n_fail = total - n_done

        Logger.info(f"  Total: {total}")
        Logger.info(f"  Completed: {n_done}")
        Logger.info(f"  Failed: {n_fail}")

        nifti_files = list(output_path.glob(f"*_bold_{tag}.nii.gz"))
        if nifti_files:
            total_gb = sum(f.stat().st_size for f in nifti_files) / (1024 ** 3)
            Logger.info(f"  Output files: {len(nifti_files)} NIfTI, {total_gb:.2f} GB")

        Logger.info(f"\n  Pipeline configuration v5.1.0:")
        Logger.info(f"    Envelope: relative band power (broadband-normalized)")
        Logger.info(f"    Dice: rank-based (amplitude-invariant)")
        Logger.info(f"    Gamma prior: dual-peak (posterior + frontal)")
        Logger.info(f"    HRF Convolution: ✅ (canonical Glover 1999)")
        Logger.info(f"    HRF Variance Scaling: ❌ (CONN handles normalization)")
        Logger.info(f"    Direction: EEG-neural → HRF conv → pseudo-BOLD")
        Logger.info(f"    TR: {self.config.SEGMENT_DURATION}s")
        Logger.info(f"    Formats: .nii.gz (CONN) + .nii (SPM)")
        Logger.info(f"    L1 optimized: {list(self.config.L1_BOUNDS.keys())}")
        Logger.info(f"    L2 optimized: {list(self.config.L2_BOUNDS.keys())}")

    # ────────────────────────────────────────────────────────────
    # RUN
    # ────────────────────────────────────────────────────────────

    def run(self, start_from_phase=1):
        """
        Main pipeline entry point.
        
        Parameters
        ----------
        start_from_phase : int (1, 2, or 3)
            1: Full run (EEG processing → optimization → volumes)
            2: Skip Phase 1 (use cached MI/Dice — must be v5.1 cache!)
            3: Skip Phase 1+2 (use cached MI/Dice + saved LOSO weights)
        """
        Logger.section("EEG-to-fMRI Pipeline v5.1.0 — Frequency-Specific")
        Logger.info(f"  Start from Phase: {start_from_phase}")
        Logger.info(f"  Timestamp: {self.start_time.isoformat()}")

        # ── PHASE 1 ──
        if start_from_phase <= 1:
            subject_data = self._phase1()
            if not subject_data:
                Logger.error("No subjects processed. Aborting.")
                return 1
        else:
            Logger.section("PHASE 1: SKIPPED (using v5.1 cache)")
            subject_data = self.cache_mgr.load_all_subjects(self.config.SUBJECTS)
            if not subject_data:
                Logger.error("No v5.1 cached data found. Run Phase 1 first.")
                Logger.error("  (v5.0/v4.x caches are incompatible — "
                           "relative power changed)")
                return 1
            Logger.success(f"  Loaded {len(subject_data)} subjects from cache")

        # ── PHASE 2 ──
        if start_from_phase <= 2:
            loso_results = self._phase2(subject_data)
            if not loso_results or not loso_results.get('l1'):
                Logger.warn("LOSO returned empty, using defaults")
                loso_results = {
                    'l1': dict(self.config.L1_DEFAULTS),
                    'l2': dict(self.config.L2_DEFAULTS),
                }
        else:
            Logger.section("PHASE 2: SKIPPED (using saved weights)")
            optimizer = LOSOOptimizer(self.config)
            loso_results = optimizer.load_results()
            if loso_results is None:
                Logger.warn("No saved v5.1 LOSO results, using defaults")
                loso_results = {
                    'l1': dict(self.config.L1_DEFAULTS),
                    'l2': dict(self.config.L2_DEFAULTS),
                }
            else:
                Logger.success("  Loaded LOSO results")

        # Log final params
        Logger.section("OPTIMIZATION RESULTS")
        for band in self.config.BANDS:
            params = loso_results['l1'].get(band, {})
            Logger.info(f"  {band:6s}: MI={params.get('mi_weight', '?'):>5}, "
                       f"C={params.get('contrast', '?'):>5}, "
                       f"B={params.get('boost', '?'):>5}, "
                       f"P={params.get('penalty', '?'):>5}, "
                       f"K={params.get('keep_top_pct', '?'):>5}, "
                       f"FWHM={params.get('smoothing_fwhm', '?'):>4}")

        # ── PHASE 3 ──
        if start_from_phase <= 3:
            self._phase3(subject_data, loso_results)

        # ── DONE ──
        duration = (datetime.now() - self.start_time).total_seconds()
        Logger.section("PIPELINE COMPLETED")
        Logger.success(
            f"  Time: {int(duration//3600)}h "
            f"{int((duration%3600)//60)}m "
            f"{int(duration%60)}s"
        )
        Logger.info(f"  Output: SPM (.nii) + CONN (.nii.gz) compatible")
        Logger.info(f"  HRF: Canonical convolution (no variance scaling)")
        Logger.info(f"  Envelope: Relative band power (frequency-specific)")
        Logger.info(f"  Dice: Rank-based (amplitude-invariant)")
        Logger.info(f"  All parameters: LOSO-optimized per-dataset")

        return 0


def auto_detect_subjects_from_cache(config):
    cache_path = Path(config.DATA_PATH) / config.CACHE_DIR
    if not cache_path.exists():
        return
    subjects = set()
    for f in cache_path.glob("*_alpha_mi_dice.npz"):
        subj = f.stem.split('_')[0]
        subjects.add(f"{subj}_restingPre_EC.mat")
    if subjects:
        config.SUBJECTS = sorted(subjects)
        print(f"✅ Cache'den {len(subjects)} subject bulundu: {config.SUBJECTS}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='EEG-to-fMRI Pipeline v5.1.0 — Frequency-Specific',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run from scratch (Phase 1 required — v5.0 cache incompatible)
  python pipeline.py --start-from 1
  
  # Use v5.1 cache, run LOSO + volumes
  python pipeline.py --start-from 2
  
  # Use cache + saved weights, only build volumes
  python pipeline.py --start-from 3
  
  # Disable AR(1) for speed
  python pipeline.py --start-from 2 --no-ar1

NOTE: v5.1 uses relative band power and rank-based Dice.
      v5.0/v4.x caches are NOT compatible — run Phase 1 fresh.
        """,
    )

    parser.add_argument(
        '--start-from', type=int, default=1, choices=[1, 2, 3],
        help='Start from phase (1=full, 2=skip EEG processing, 3=skip optimization)',
    )
    parser.add_argument(
        '--no-ar1', action='store_true',
        help='Disable AR(1) prewhitening (faster)',
    )
    parser.add_argument(
        '--no-hrf', action='store_true',
        help='Disable HRF convolution (output = neural activity, not pseudo-BOLD)',
    )
    parser.add_argument(
        '--log', type=str, default=None,
        help='Log file path (default: auto-generated)',
    )

    args = parser.parse_args()

    config = Config()

    # Validate paths
    if not Path(config.DATA_PATH).exists():
        print(f"❌ Data path not found: {config.DATA_PATH}")
        return 1

    missing = [
        f for f in config.SUBJECTS
        if not (Path(config.DATA_PATH) / f).exists()
    ]
    if missing:
        print(f"⚠️  Missing {len(missing)} files:")
        for f in missing:
            print(f"    - {f}")
        config.SUBJECTS = [f for f in config.SUBJECTS if f not in missing]
        if not config.SUBJECTS:
            print("❌ No valid subjects. Aborting.")
            return 1

    # Apply CLI overrides
    if args.no_ar1:
        config.L2_DEFAULTS['ar1_enabled'] = False

    # Open log
    if args.log:
        log_path = args.log
    else:
        log_path = (
            Path(config.DATA_PATH)
            / f"pipeline_{config.VERSION_TAG}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    Logger.open_log(str(log_path))

    try:
        # Handle --no-hrf by monkey-patching HRF convolution
        if args.no_hrf:
            Logger.warn("HRF convolution DISABLED — output is neural activity")
            HRFConvolver.convolve_volume = staticmethod(
                lambda vol, tr, config=None: vol
            )
            HRFConvolver.convolve_volume_mmap = staticmethod(
                lambda vol, tr, config=None: vol
            )

        pipeline = GroupEEGtoFMRIPipeline(config)
        exit_code = pipeline.run(start_from_phase=args.start_from)

    except KeyboardInterrupt:
        Logger.warn("Interrupted by user")
        exit_code = 130

    except Exception as e:
        Logger.error(f"Pipeline failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    finally:
        Logger.close_log()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())