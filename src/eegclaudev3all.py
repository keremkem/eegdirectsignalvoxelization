#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-to-fMRI Voxel Projection Pipeline v3.3 (Group Analysis Edition)
===================================================================

MAJOR FEATURES:
✅ Group-level K-fold optimization (pooled data across subjects)
✅ Consistent MI weights across all subjects
✅ Two-phase processing: (1) MI/Dice computation, (2) Volume building
✅ Robust outlier handling
✅ Literature-validated metrics

Workflow:
---------
PHASE 1: Process all subjects → compute MI/Dice → cache
PHASE 2: Pool data → group K-fold → find optimal weights
PHASE 3: Apply optimal weights → build volumes for all subjects

Author: [Your Name]
Version: 3.3
Date: 2024
"""

import os
import sys
import gc
import warnings
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count
import pickle
import hashlib
import json
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch, hilbert, welch
from scipy.spatial.distance import cdist
from scipy.ndimage import label, uniform_filter1d
from scipy.stats import kurtosis

import nibabel as nib
from nilearn import image, datasets
from nilearn.maskers import NiftiLabelsMasker

from sklearn.decomposition import FastICA
from sklearn.metrics import mutual_info_score
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold

from joblib import Parallel, delayed
from tqdm import tqdm
import psutil

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Pipeline configuration with group-level optimization"""
    
    # ===== PATHS =====
    DATA_PATH = r"C:\Users\kerem\Downloads\eegyedek"
    
    # ===== SUBJECT LIST =====
    SUBJECTS = [
        "S02_restingPre_EC.mat",
        "S03_restingPre_EC.mat",
        "S04_restingPre_EC.mat",
        "S05_restingPre_EC.mat",
        "S06_restingPre_EC.mat",
        "S07_restingPre_EC.mat",
        "S08_restingPre_EC.mat",
        "S09_restingPre_EC.mat",
        "S10_restingPre_EC.mat",
        "S11_restingPre_EC.mat",
    ]
    
    # ===== PROCESSING MODE =====
    GROUP_KFOLD = True  # Use group-level K-fold optimization
    CACHE_MI_DICE = True  # Cache MI/Dice per subject
    CACHE_DIR = "cache_v33"
    
    # ===== CSD PARAMETERS =====
    CSD_SIGMA = 30.0
    CSD_PRESERVE_ALPHA = True
    CSD_ALPHA_MODE = "posterior_selective"  # "none" / "global" / "posterior_selective"
    CSD_POSTERIOR_Y_THRESHOLD = -40.0
    CSD_FRONTAL_Y_THRESHOLD = 20.0
    CSD_CENTRAL_ALPHA_RETENTION = 0.5
    
    # ===== EEG PARAMETERS =====
    FS = 256  # Sampling rate (Hz)
    SEGMENT_DURATION = 2.0  # TR (seconds)
    
    BANDS = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    
    # ===== ICA ARTIFACT REMOVAL =====
    ICA_MODE = "standard"  # conservative / standard / aggressive
    ICA_N_COMPONENTS = 64
    ICA_MAX_REMOVE = {
        "conservative": 5,
        "standard": 8,
        "aggressive": 12
    }
    
    # ===== SPATIAL PROJECTION =====
    SIGMA_BASE = 22.0  # mm (Gevins et al., 1994)
    
    SIGMA_MULTIPLIERS = {
        "occipital": 1.5,
        "posterior": 1.3,
        "parietal": 1.2,
        "central": 1.0,
        "frontal": 1.1,
    }
    
    APPLY_REGIONAL_BOOST = False  # Literature compatibility
    
    HEMISPHERE_ISOLATION = 0.70
    CROSS_HEMISPHERE_WEIGHT = 0.25
    MIDLINE_WEIGHT = 0.85
    MIDLINE_GAP_MM = 3.0
    
    # ===== VOXELIZATION =====
    GRID_SPACING = 2.0  # mm
    GRID_BOUNDS = {
        "x": (-90, 91),
        "y": (-130, 91),
        "z": (-72, 109),
    }
    
    GM_INCLUDE_SUBCORTICAL = False
    
    # ===== ROI-AWARE SPARSIFICATION =====
    KEEP_TOP_PCT = {
        "delta": 0.50,
        "theta": 0.55,
        "alpha": 0.90,
        "beta": 0.50,
        "gamma": 0.40,
    }
    
    MIN_VOXELS_GLOBAL = {
        "delta": 20000,
        "theta": 22000,
        "alpha": 120000,
        "beta": 22000,
        "gamma": 16000,
    }
    
    PER_ROI_KEEP_PCT = 0.15
    PER_ROI_MIN_VOXELS = 500
    
    # ✅ RELAXED OCCIPITAL PROTECTION (realistic)
    OCCIPITAL_ROI_INDICES = {22, 23, 24, 32, 36, 39, 40, 47, 48}
    OCCIPITAL_MIN_COVERAGE = 0.65  # %65 (was 0.95)
    OCCIPITAL_MIN_VOXELS = 1500    # 1500 (was 3500)
    
    FRONTAL_ROI_INDICES = {3, 4, 5, 6, 7, 25}
    FRONTAL_MIN_COVERAGE = 0.85
    FRONTAL_MIN_VOXELS = 3000
    
    POSTERIOR_ROI_BOOST = {
        "delta": 1.4,
        "theta": 1.3,
        "alpha": 2.0,
        "beta": 1.0,
        "gamma": 1.0,
    }
    
    POSTERIOR_ROI_INDICES = {11, 13, 20, 21, 31}
    
    # ===== TEMPORAL CONSISTENCY =====
    TEMPORAL_FILTERING = True
    TEMPORAL_WINDOW = 3
    TEMPORAL_MIN_FRAC = 0.35
    
    # ===== ARTIFACT CLEANUP =====
    RAYKILL_ENABLE = True
    RAYKILL_MIN_CLUSTER_MM3 = 25.0
    RAYKILL_KEEP_FLOOR_PCT = 0.20
    RAYKILL_KEEP_FLOOR_ABS = 28000
    
    # ===== NORMALIZATION =====
    APPLY_ZSCORE = True
    ZSCORE_MODE = "global"
    
    # ===== MI/DICE =====
    MI_N_BINS = 10
    
    # ===== LOSO PARAMETERS (YENİ EKLE) =====
    USE_LOSO = True  # False = Group K-fold (v3.3), True = LOSO (v3.4)
    GROUP_KFOLD = False  # Artık kullanılmıyor (backward compat)
    LOSO_N_JOBS = 6  # Paralel fold sayısı
    LOSO_EXCLUDE_OUTLIERS = True  # Otomatik outlier detection
    LOSO_OUTLIER_ALPHA_PA_MIN = 1.5  # Alpha P/A ratio minimum
    LOSO_OUTLIER_ALPHA_OCC_MIN = 12.0  # Alpha occipital % minimum
    
    # ✅ K-FOLD PARAMETERS
    KFOLD_N_SPLITS = 3
    KFOLD_MI_WEIGHTS_TO_TEST = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    
    # ===== CONNECTIVITY =====
    CONN_STANDARDIZE = False
    CONN_DETREND = True
    CONN_GSR = True
    CONN_PARTIAL = True
    CONN_FISHER_Z = True
    
    # ===== PERFORMANCE =====
    N_JOBS = min(8, cpu_count())
    BATCH_SIZE_SIGNATURES = 5000

# ============================================================================
# LOGGER
# ============================================================================

class Logger:
    """Simple logger with timestamps"""
    
    @staticmethod
    def info(msg):
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {msg}")
    
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
    def debug(msg):
        Logger.info(f"🔍 {msg}")
    
    @staticmethod
    def section(title):
        Logger.info("\n" + "="*70)
        Logger.info(title)
        Logger.info("="*70)

# ============================================================================
# COORDINATE SYSTEM (unchanged from v3.2)
# ============================================================================

class CoordinateSystem:
    """EEG 10-20 to MNI coordinate transformation"""
    
    RAW_COORDS = """
Fp1 -26 87 -18
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
TP7 -66 -30 -13
TP8 66 -30 -13
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
Iz 0 -96 -8
"""
    
    MANUAL_MNI_COORDS = {
        'Oz':  [0,   -105, 15],
        'O1':  [-27, -108, 10],
        'O2':  [27,  -108, 10],
        'Iz':  [0,   -112, 5],
        'POz': [0,   -95,  45],
        'PO3': [-32, -98,  35],
        'PO4': [32,  -98,  35],
        'PO7': [-42, -100, 25],
        'PO8': [42,  -100, 25],
        'Pz':  [0,   -68,  65],
        'P1':  [-20, -72,  60],
        'P2':  [20,  -72,  60],
        'P3':  [-42, -72,  50],
        'P4':  [42,  -72,  50],
        'P5':  [-52, -65,  42],
        'P6':  [52,  -65,  42],
        'P7':  [-58, -62,  28],
        'P8':  [58,  -62,  28],
        'P9':  [-50, -70,  10],
        'P10': [50,  -70,  10],
        'CPz': [0,   -35,  75],
        'CP1': [-20, -35,  72],
        'CP2': [20,  -35,  72],
        'CP3': [-40, -35,  65],
        'CP4': [40,  -35,  65],
        'CP5': [-54, -32,  50],
        'CP6': [54,  -32,  50],
        'T7':  [-70, -20, -5],
        'T8':  [70,  -20, -5],
        'TP7': [-68, -38, -8],
        'TP8': [68,  -38, -8],
    }
    
    @classmethod
    def load_coordinates(cls):
        coords = {}
        order = []
        
        for line in cls.RAW_COORDS.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            
            name, x, y, z = parts
            
            if name in cls.MANUAL_MNI_COORDS:
                coords[name] = np.array(cls.MANUAL_MNI_COORDS[name], dtype=np.float32)
                order.append(name)
                continue
            
            x_eeg, y_eeg, z_eeg = float(x), float(y), float(z)
            
            scale = 0.88
            x_mni = x_eeg * scale
            y_mni = y_eeg * scale * 1.12 - 5.0
            z_mni = z_eeg * scale + 48.0
            
            coords[name] = np.array([x_mni, y_mni, z_mni], dtype=np.float32)
            order.append(name)
        
        return coords, order

# ============================================================================
# SIGNAL PROCESSING (unchanged)
# ============================================================================

class SignalProcessor:
    """EEG signal preprocessing utilities"""
    
    @staticmethod
    def bandpass(data, fs, low, high, order=4):
        nyq = fs / 2
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, data, axis=1)
    
    @staticmethod
    def notch(data, fs, freq=50.0, Q=30):
        nyq = fs / 2
        b, a = iirnotch(freq/nyq, Q=Q)
        return filtfilt(b, a, data, axis=1)
    
    @staticmethod
    def hilbert_envelope(segment, band, fs):
        filtered = SignalProcessor.bandpass(segment, fs, band[0], band[1])
        analytic = hilbert(filtered, axis=1)
        envelope = np.abs(analytic).mean(axis=1)
        return envelope
    
    @staticmethod
    def bandpower_welch(signal, fs, fmin, fmax):
        f, Pxx = welch(signal, fs=fs, nperseg=min(4096, len(signal)))
        idx = (f >= fmin) & (f <= fmax)
        if not np.any(idx):
            return 0.0
        return np.trapz(Pxx[idx], f[idx])

# ============================================================================
# ICA ARTIFACT REMOVAL (unchanged from v3.2)
# ============================================================================

class ICAArtifactRemover:
    """Literature-based ICA artifact detection and removal"""
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.DATA_PATH) / "ica_cache_v33"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_signature(self, eeg, fs):
        h = hashlib.sha1()
        h.update(np.asarray(eeg.shape, dtype=np.int32).tobytes())
        h.update(np.float32(fs).tobytes())
        sample_len = min(eeg.shape[1], int(fs * 10))
        h.update(np.ascontiguousarray(eeg[:, :sample_len]).astype(np.float32).tobytes())
        h.update(f"v33_{self.config.ICA_MODE}".encode())
        return h.hexdigest()[:16]
    
    def _extract_features(self, components, fs):
        n_comp = components.shape[0]
        features = {}
        
        for i in range(n_comp):
            signal = components[i]
            feat = {}
            
            feat['std'] = np.std(signal)
            feat['kurtosis'] = kurtosis(signal, fisher=False, bias=False)
            feat['median_abs_dev'] = np.median(np.abs(signal - np.median(signal)))
            
            f, Pxx = welch(signal, fs=fs, nperseg=min(4096, len(signal)))
            total_power = np.trapz(Pxx, f) + 1e-12
            
            lf_power = np.trapz(Pxx[f < 2], f[f < 2])
            feat['lf_ratio'] = lf_power / total_power
            
            hf_power = np.trapz(Pxx[(f >= 30) & (f <= 100)], f[(f >= 30) & (f <= 100)])
            feat['hf_ratio'] = hf_power / total_power
            
            alpha_power = np.trapz(Pxx[(f >= 8) & (f <= 13)], f[(f >= 8) & (f <= 13)])
            feat['alpha_ratio'] = alpha_power / total_power
            
            feat['spectral_flatness'] = np.exp(np.mean(np.log(Pxx + 1e-12))) / (np.mean(Pxx) + 1e-12)
            
            acf = np.correlate(signal - signal.mean(), signal - signal.mean(), mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / (acf[0] + 1e-12)
            lag_idx = int(fs * 0.02)
            feat['acf_lag1'] = acf[lag_idx] if len(acf) > lag_idx else 0.0
            
            features[i] = feat
        
        return features
    
    def _classify_artifacts(self, features, mixing_matrix, ch_order, mode="standard"):
        n_comp = len(features)
        
        feat_keys = ['std', 'kurtosis', 'lf_ratio', 'hf_ratio', 'alpha_ratio', 
                     'spectral_flatness', 'acf_lag1']
        feat_matrix = np.array([[features[i][k] for k in feat_keys] for i in range(n_comp)])
        
        feat_z = (feat_matrix - feat_matrix.mean(axis=0)) / (feat_matrix.std(axis=0) + 1e-12)
        
        posterior_channels = [j for j, ch in enumerate(ch_order) 
                             if ch in ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 
                                      'P1', 'P2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'Iz']]
        
        bad_idx = []
        reasons = {}
        
        for i in range(n_comp):
            f = features[i]
            fz = feat_z[i]
            score = 0.0
            reason_list = []
            
            # EOG Detection
            if fz[1] > 2.5 and f['lf_ratio'] > 0.3:
                frontal_channels = [j for j, ch in enumerate(ch_order) 
                                   if ch in ['Fp1', 'Fp2', 'Fpz', 'AF7', 'AF8', 'AF3', 'AF4']]
                if frontal_channels:
                    weights = np.abs(mixing_matrix[:, i])
                    frontal_weight = weights[frontal_channels].mean()
                    overall_weight = weights.mean()
                    if frontal_weight > 1.5 * overall_weight:
                        score += 3.0
                        reason_list.append("EOG")
            
            # EMG Detection
            if f['hf_ratio'] > 0.15 and fz[1] > 1.5:
                score += 2.5
                reason_list.append("EMG")
            
            # Channel Noise
            weights = np.abs(mixing_matrix[:, i])
            if weights.max() > 5 * np.median(weights):
                score += 2.0
                reason_list.append("Channel_noise")
            
            # Line Noise
            if f['spectral_flatness'] > 0.8:
                score += 1.5
                reason_list.append("Line_noise")
            
            # Spiky Signal
            if f['acf_lag1'] < 0.3:
                score += 1.0
                reason_list.append("Spiky")
            
            # ENHANCED ALPHA PROTECTION
            if f['alpha_ratio'] > 0.25:
                weights = np.abs(mixing_matrix[:, i])
                
                if posterior_channels:
                    post_weight = weights[posterior_channels].mean()
                    overall_weight = weights.mean()
                    post_dominance = post_weight / (overall_weight + 1e-12)
                    
                    if f['alpha_ratio'] > 0.50 and post_dominance > 1.3:
                        score -= 6.0
                        reason_list.append("Protected_STRONG_posterior_alpha")
                    elif f['alpha_ratio'] > 0.35 and post_dominance > 1.1:
                        score -= 4.0
                        reason_list.append("Protected_moderate_posterior_alpha")
                    elif post_dominance > 1.0:
                        score -= 2.5
                        reason_list.append("Protected_weak_posterior_alpha")
                    else:
                        score -= 1.0
                        reason_list.append("Weak_alpha_protection")
                else:
                    score -= 2.0
                    reason_list.append("Protected_alpha")
            
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
            Logger.info(f"ICA: Cache HIT ({len(data['bad_idx'])} components removed)")
            return data['eeg_clean'], {
                'cache': 'HIT',
                'bad_idx': data['bad_idx'].tolist(),
                'reasons': data['reasons'].item() if 'reasons' in data else {}
            }
        
        Logger.info(f"ICA: Decomposing {eeg.shape[0]} channels...")
        n_comp = min(self.config.ICA_N_COMPONENTS, eeg.shape[0])
        
        ica = FastICA(n_components=n_comp, random_state=0, max_iter=1000, 
                      tol=1e-3, whiten='unit-variance')
        
        components_time = ica.fit_transform(eeg.T).T
        mixing = ica.mixing_
        
        Logger.info("ICA: Extracting features...")
        features = self._extract_features(components_time, fs)
        
        Logger.info(f"ICA: Classifying artifacts (mode={self.config.ICA_MODE})...")
        bad_idx, reasons = self._classify_artifacts(
            features, mixing, ch_order, mode=self.config.ICA_MODE
        )
        
        max_remove = self.config.ICA_MAX_REMOVE[self.config.ICA_MODE]
        if len(bad_idx) > max_remove:
            scores = [float(reasons[i].split('=')[1].split(':')[0]) for i in bad_idx]
            top_idx = np.argsort(scores)[-max_remove:]
            bad_idx = bad_idx[top_idx]
            reasons = {i: reasons[i] for i in bad_idx}
        
        components_clean = components_time.copy()
        components_clean[bad_idx, :] = 0.0
        
        eeg_clean = (components_clean.T @ mixing.T + ica.mean_).T.astype(np.float32)
        
        if self.config.CACHE_MI_DICE:
            np.savez(cache_path, 
                     eeg_clean=eeg_clean,
                     bad_idx=np.array(bad_idx, dtype=np.int32),
                     reasons=reasons)
        
        Logger.success(f"ICA: Removed {len(bad_idx)}/{n_comp} components")
        
        return eeg_clean, {
            'cache': 'MISS',
            'bad_idx': bad_idx.tolist(),
            'reasons': reasons
        }

# ============================================================================
# CSD RE-REFERENCING (unchanged from v3.2)
# ============================================================================

class CSDReferencer:
    """Current Source Density re-referencing with posterior-selective alpha preservation"""
    
    @staticmethod
    def apply(eeg, ch_coords, ch_order, config):
        sigma = config.CSD_SIGMA
        preserve_alpha = config.CSD_PRESERVE_ALPHA
        alpha_mode = config.CSD_ALPHA_MODE
        
        Logger.info(f"CSD: Applying surface Laplacian (σ={sigma}mm, "
                    f"alpha_preserve={preserve_alpha}, mode={alpha_mode})...")
        
        coords_array = np.stack([ch_coords[ch] for ch in ch_order])
        
        distances = cdist(coords_array, coords_array)
        
        G = np.exp(-(distances**2) / (2 * sigma**2)).astype(np.float32)
        G[distances > 80.0] = 0.0
        np.fill_diagonal(G, 0.0)
        
        row_sums = G.sum(axis=1, keepdims=True) + 1e-10
        G = G / row_sums
        
        if not preserve_alpha or alpha_mode == "none":
            eeg_csd = eeg - (G @ eeg)
            Logger.success("CSD: Applied (standard, full suppression)")
            return eeg_csd
        
        if alpha_mode == "global":
            b_alpha, a_alpha = butter(4, [8/128, 13/128], btype='band')
            eeg_alpha = filtfilt(b_alpha, a_alpha, eeg, axis=1)
            eeg_non_alpha = eeg - eeg_alpha
            
            eeg_non_alpha_csd = eeg_non_alpha - (G @ eeg_non_alpha)
            
            eeg_csd = eeg_non_alpha_csd + eeg_alpha
            
            Logger.success("CSD: Applied (global alpha preservation)")
            return eeg_csd
        
        if alpha_mode == "posterior_selective":
            y_coords = coords_array[:, 1]
            posterior_mask = (y_coords < config.CSD_POSTERIOR_Y_THRESHOLD)
            frontal_mask = (y_coords > config.CSD_FRONTAL_Y_THRESHOLD)
            central_mask = (~posterior_mask) & (~frontal_mask)
            
            n_posterior = int(posterior_mask.sum())
            n_central = int(central_mask.sum())
            n_frontal = int(frontal_mask.sum())
            
            Logger.info(f"  Channel regions: Posterior={n_posterior}, "
                       f"Central={n_central}, Frontal={n_frontal}")
            
            fs = 256
            nyq = fs / 2
            b_alpha, a_alpha = butter(4, [8/nyq, 13/nyq], btype='band')
            eeg_alpha_all = filtfilt(b_alpha, a_alpha, eeg, axis=1)
            
            eeg_alpha_posterior = np.zeros_like(eeg)
            eeg_alpha_posterior[posterior_mask, :] = eeg_alpha_all[posterior_mask, :]
            
            eeg_alpha_central = np.zeros_like(eeg)
            eeg_alpha_central[central_mask, :] = config.CSD_CENTRAL_ALPHA_RETENTION * eeg_alpha_all[central_mask, :]
            
            eeg_alpha_frontal = eeg_alpha_all.copy()
            eeg_alpha_frontal[posterior_mask, :] = 0.0
            eeg_alpha_frontal[central_mask, :] *= (1.0 - config.CSD_CENTRAL_ALPHA_RETENTION)
            
            eeg_non_alpha = eeg - eeg_alpha_all
            
            eeg_to_csd = eeg_non_alpha + eeg_alpha_frontal + (0.5 * eeg_alpha_central)
            eeg_to_csd_csd = eeg_to_csd - (G @ eeg_to_csd)
            
            eeg_csd = eeg_to_csd_csd + eeg_alpha_posterior + (0.5 * eeg_alpha_central)
            
            if n_posterior > 0:
                post_alpha_before = np.abs(eeg_alpha_all[posterior_mask]).mean()
                post_alpha_after = np.abs(eeg_alpha_posterior[posterior_mask]).mean()
                retention = post_alpha_after / (post_alpha_before + 1e-12)
                Logger.info(f"  Posterior alpha retention: {100*retention:.1f}%")
            
            if n_frontal > 0:
                front_alpha_before = np.abs(eeg_alpha_all[frontal_mask]).mean()
                eeg_alpha_final = filtfilt(b_alpha, a_alpha, eeg_csd, axis=1)
                front_alpha_after = np.abs(eeg_alpha_final[frontal_mask]).mean()
                suppression = 1.0 - front_alpha_after / (front_alpha_before + 1e-12)
                Logger.info(f"  Frontal alpha suppression: {100*suppression:.1f}%")
            
            Logger.success("CSD: Applied (posterior-selective alpha preservation)")
            return eeg_csd
        
        Logger.warn(f"Unknown alpha_mode: {alpha_mode}, using standard CSD")
        return eeg - (G @ eeg)

# ============================================================================
# GRID AND MASK (unchanged)
# ============================================================================

class VoxelGrid:
    """3D voxel grid with gray matter masking"""
    
    def __init__(self, config):
        self.config = config
        self.spacing = config.GRID_SPACING
        
        Logger.info("Grid: Creating 3D voxel grid...")
        self.coords_all, self.shape = self._create_grid()
        
        self.affine = self._create_affine()
        self.inv_affine = np.linalg.inv(self.affine)
        
        Logger.info("Grid: Loading Harvard-Oxford atlas...")
        self.gm_mask = self._load_gm_mask()
        self.atlas_idx, self.roi_names = self._load_atlas()
        
        Logger.info("Grid: Filtering to gray matter...")
        self.coords_gm = self._filter_to_gm()
        
        Logger.success(f"Grid: {len(self.coords_gm)} GM voxels (shape={self.shape})")
    
    def _create_grid(self):
        bounds = self.config.GRID_BOUNDS
        
        xs = np.arange(bounds['x'][0], bounds['x'][1], self.spacing)
        ys = np.arange(bounds['y'][0], bounds['y'][1], self.spacing)
        zs = np.arange(bounds['z'][0], bounds['z'][1], self.spacing)
        
        grid = np.array(np.meshgrid(xs, ys, zs, indexing='ij'))
        coords = grid.reshape(3, -1).T
        shape = (len(xs), len(ys), len(zs))
        
        return coords, shape
    
    def _create_affine(self):
        s = self.spacing
        bounds = self.config.GRID_BOUNDS
        
        affine = np.array([
            [ s,  0,  0,  bounds['x'][0]],
            [ 0,  s,  0,  bounds['y'][0]],
            [ 0,  0,  s,  bounds['z'][0]],
            [ 0,  0,  0,  1]
        ], dtype=np.float32)
        
        return affine
    
    def _load_gm_mask(self):
        target_img = nib.Nifti1Image(np.zeros(self.shape, dtype=np.int16), self.affine)
        
        ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        cort_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
        cort_resampled = image.resample_to_img(cort_img, target_img, interpolation='nearest')
        gm_mask = cort_resampled.get_fdata() > 0
        
        if self.config.GM_INCLUDE_SUBCORTICAL:
            ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
            sub_img = ho_sub.maps if isinstance(ho_sub.maps, nib.Nifti1Image) else nib.load(ho_sub.maps)
            sub_resampled = image.resample_to_img(sub_img, target_img, interpolation='nearest')
            gm_mask |= (sub_resampled.get_fdata() > 0)
        
        Logger.info(f"  GM mask: {int(gm_mask.sum())} voxels")
        return gm_mask
    
    def _load_atlas(self):
        ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
        
        target_img = nib.Nifti1Image(np.zeros(self.shape, dtype=np.int16), self.affine)
        atlas_resampled = image.resample_to_img(atlas_img, target_img, interpolation='nearest')
        atlas_idx = atlas_resampled.get_fdata().astype(int)
        
        roi_names = list(ho_cort.labels)
        
        Logger.info(f"  Atlas: {len(roi_names)} ROIs loaded")
        return atlas_idx, roi_names
    
    def _filter_to_gm(self):
        coords_gm = []
        
        for coord in tqdm(self.coords_all, desc="  Filtering GM", ncols=80):
            vox = self.inv_affine @ np.append(coord, 1)
            xi, yi, zi = np.round(vox[:3]).astype(int)
            
            if (0 <= xi < self.shape[0] and
                0 <= yi < self.shape[1] and
                0 <= zi < self.shape[2] and
                self.gm_mask[xi, yi, zi]):
                coords_gm.append(coord)
        
        return np.asarray(coords_gm, dtype=np.float32)
    
    def mni_to_voxel(self, mni_coords):
        vox = self.inv_affine @ np.append(mni_coords, 1)
        return np.round(vox[:3]).astype(int)

# ============================================================================
# VOXEL SIGNATURE COMPUTATION (unchanged)
# ============================================================================

class SignatureComputer:
    """Compute voxel signatures with hemisphere balancing"""
    
    def __init__(self, config):
        self.config = config
    
    def _get_adaptive_sigma(self, voxel_y):
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
        
        multiplier = self.config.SIGMA_MULTIPLIERS[region]
        return self.config.SIGMA_BASE * multiplier
    
    def _classify_channels(self, ch_names, ch_coords):
        lh, rh, mid = [], [], []
        
        for i, name in enumerate(ch_names):
            x = ch_coords[name][0]
            
            if any(s in name for s in ['1', '3', '5', '7', '9']):
                lh.append(i)
            elif any(s in name for s in ['2', '4', '6', '8', '10']):
                rh.append(i)
            elif 'z' in name.lower() or abs(x) <= self.config.MIDLINE_GAP_MM:
                mid.append(i)
            else:
                (lh if x < -self.config.MIDLINE_GAP_MM else rh).append(i)
        
        return lh, rh, mid
    
    def _classify_voxel_hemisphere(self, voxel_x):
        if abs(voxel_x) <= self.config.MIDLINE_GAP_MM:
            return 'MID'
        elif voxel_x < -self.config.MIDLINE_GAP_MM:
            return 'LH'
        else:
            return 'RH'
    
    def compute(self, voxel_coords, ch_coords, ch_order):
        Logger.section("Computing Voxel Signatures")
        
        n_voxels = len(voxel_coords)
        n_channels = len(ch_order)
        
        ch_coords_array = np.stack([ch_coords[ch] for ch in ch_order])
        
        lh_idx, rh_idx, mid_idx = self._classify_channels(ch_order, ch_coords)
        Logger.info(f"Channels: LH={len(lh_idx)}, RH={len(rh_idx)}, MID={len(mid_idx)}")
        
        signatures = np.zeros((n_voxels, n_channels), dtype=np.float32)
        
        batch_size = self.config.BATCH_SIZE_SIGNATURES
        
        for batch_start in tqdm(range(0, n_voxels, batch_size), 
                               desc="Signatures", ncols=80):
            batch_end = min(batch_start + batch_size, n_voxels)
            batch_coords = voxel_coords[batch_start:batch_end]
            
            distances = np.linalg.norm(
                batch_coords[:, None, :] - ch_coords_array[None, :, :], 
                axis=2
            )
            
            for bi, global_idx in enumerate(range(batch_start, batch_end)):
                vox_coord = voxel_coords[global_idx]
                vox_x, vox_y = vox_coord[0], vox_coord[1]
                dist = distances[bi]
                
                sigma = self._get_adaptive_sigma(vox_y)
                
                weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                weights[dist > 200] = 0.0
                
                if np.sum(weights > 0.01) < 8:
                    nearest = np.argsort(dist)[:8]
                    weights[nearest] = np.maximum(weights[nearest], 0.05)
                
                hemi = self._classify_voxel_hemisphere(vox_x)
                
                if hemi == 'LH':
                    weights[lh_idx] *= self.config.HEMISPHERE_ISOLATION
                    weights[rh_idx] *= self.config.CROSS_HEMISPHERE_WEIGHT
                    weights[mid_idx] *= self.config.MIDLINE_WEIGHT
                elif hemi == 'RH':
                    weights[rh_idx] *= self.config.HEMISPHERE_ISOLATION
                    weights[lh_idx] *= self.config.CROSS_HEMISPHERE_WEIGHT
                    weights[mid_idx] *= self.config.MIDLINE_WEIGHT
                elif hemi == 'MID':
                    weights[lh_idx] *= 0.80
                    weights[rh_idx] *= 0.80
                    weights[mid_idx] *= 1.0
                
                weight_sum = weights.sum()
                if weight_sum > 1e-12:
                    weights /= weight_sum
                else:
                    nearest_ch = np.argmin(dist)
                    weights[:] = 0.0
                    weights[nearest_ch] = 1.0
                
                signatures[global_idx] = weights
        
        Logger.success(f"Signatures: Computed for {n_voxels} voxels")
        
        return signatures

# ============================================================================
# MI/DICE COMPUTATION (unchanged)
# ============================================================================

class MIDiceComputer:
    """Compute Mutual Information and Dice similarity"""
    
    def __init__(self, config):
        self.config = config
    
    def _compute_one_timepoint(self, voxel_sigs, snapshot, v_bins, t):
        n_bins = self.config.MI_N_BINS
        n_voxels = len(voxel_sigs)
        
        # DICE
        numerator = 2.0 * (voxel_sigs @ snapshot)
        denominator = (np.sum(voxel_sigs**2, axis=1) + np.sum(snapshot**2) + 1e-12)
        dice_scores = numerator / denominator
        
        # MI
        s_min, s_max = snapshot.min(), snapshot.max()
        s_bins = np.linspace(s_min, s_max, n_bins + 1)
        
        sb = np.digitize(snapshot, bins=s_bins, right=False)
        sb = np.clip(sb - 1, 0, n_bins - 1)
        
        mi_scores = np.zeros(n_voxels, dtype=np.float32)
        
        chunk_size = 5000
        
        for start in range(0, n_voxels, chunk_size):
            end = min(start + chunk_size, n_voxels)
            
            chunk_data = voxel_sigs[start:end]
            
            for i in range(end - start):
                vox_idx = start + i
                
                v_min, v_max = chunk_data[i].min(), chunk_data[i].max()
                if v_max <= v_min:
                    mi_scores[vox_idx] = 0.0
                    continue
                
                vb_bins = v_bins[vox_idx]
                vb = np.digitize(chunk_data[i], bins=vb_bins, right=False)
                vb = np.clip(vb - 1, 0, n_bins - 1)
                
                contingency = np.zeros((n_bins, n_bins), dtype=np.float32)
                np.add.at(contingency, (vb, sb), 1)
                
                mi_scores[vox_idx] = self._fast_mi_from_contingency(contingency)
        
        mi_norm = self._normalize(mi_scores)
        dice_norm = self._normalize(dice_scores)
        
        return t, mi_norm, dice_norm
    
    def _fast_mi_from_contingency(self, contingency):
        contingency = contingency + 1e-12
        
        pxy = contingency / contingency.sum()
        
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        
        px_py = px * py
        
        mask = pxy > 0
        mi = np.sum(pxy[mask] * np.log(pxy[mask] / (px_py[mask] + 1e-12)))
        
        return max(0.0, mi)
    
    def _normalize(self, scores):
        vmin, vmax = scores.min(), scores.max()
        if vmax > vmin:
            return (scores - vmin) / (vmax - vmin)
        return scores
    
    def compute(self, voxel_sigs, snapshots):
        Logger.info("Computing MI/Dice scores...")
        
        n_voxels, n_channels = voxel_sigs.shape
        n_timepoints = snapshots.shape[0]
        
        Logger.info("  Pre-computing bins...")
        v_bins = [
            np.histogram_bin_edges(voxel_sigs[i], bins=self.config.MI_N_BINS)
            for i in tqdm(range(n_voxels), desc="  Bins", ncols=80)
        ]
        
        Logger.info(f"  Computing {n_timepoints} timepoints (sequential)...")
        
        results = []
        for t in tqdm(range(n_timepoints), desc="  Timepoints", ncols=80):
            result = self._compute_one_timepoint(voxel_sigs, snapshots[t], v_bins, t)
            results.append(result)
        
        results.sort(key=lambda x: x[0])
        
        mi = np.zeros((n_voxels, n_timepoints), dtype=np.float32)
        dice = np.zeros_like(mi)
        
        for t, mi_t, dice_t in results:
            mi[:, t] = mi_t
            dice[:, t] = dice_t
        
        Logger.success("MI/Dice: Computed")
        
        return mi, dice

# ============================================================================
# ✅ GROUP K-FOLD OPTIMIZER (NEW!)
# ============================================================================

class GroupKFoldOptimizer:
    """
    Group-level K-fold optimization on pooled data.
    
    Key difference from v3.2:
    - Optimizes on POOLED data from ALL subjects
    - Returns SINGLE weight for each band (applied to all subjects)
    """
    
    def __init__(self, config):
        self.config = config
        self.n_folds = config.KFOLD_N_SPLITS
        self.mi_weights_to_test = config.KFOLD_MI_WEIGHTS_TO_TEST
        self.voxel_coords_pooled = None
    
    def set_pooled_voxel_coords(self, coords_list):
        """
        Set pooled voxel coordinates from all subjects.
        
        Args:
            coords_list: List of (n_voxels_i, 3) arrays, one per subject
        """
        self.voxel_coords_pooled = np.concatenate(coords_list, axis=0)
        Logger.info(f"Group K-fold: Pooled {len(self.voxel_coords_pooled)} voxels "
                   f"from {len(coords_list)} subjects")
    
    def optimize_band(self, mi_pooled, dice_pooled, band_name):
        """
        Optimize MI weight for one band using pooled data.
        
        Args:
            mi_pooled: (n_voxels_total, n_timepoints_total) pooled MI
            dice_pooled: (n_voxels_total, n_timepoints_total) pooled Dice
            band_name: Band name (e.g., "alpha")
        
        Returns:
            optimal_weight: Single float (applied to ALL subjects)
        """
        Logger.info(f"\n🔍 GROUP K-FOLD: {band_name.upper()}")
        Logger.info(f"  Pooled data shape: {mi_pooled.shape}")
        
        if self.voxel_coords_pooled is None:
            Logger.error("❌ Voxel coordinates not set! Call set_pooled_voxel_coords() first")
            return 0.5
        
        # Band-specific optimization
        if band_name == "alpha":
            best_weight = self._optimize_alpha(mi_pooled, dice_pooled)
        elif band_name == "beta":
            best_weight = self._optimize_beta(mi_pooled, dice_pooled)
        elif band_name == "delta":
            best_weight = self._optimize_delta(mi_pooled, dice_pooled)
        elif band_name == "theta":
            best_weight = self._optimize_theta(mi_pooled, dice_pooled)
        elif band_name == "gamma":
            best_weight = self._optimize_gamma(mi_pooled, dice_pooled)
        else:
            best_weight = 0.5
        
        Logger.info(f"  ✅ GROUP optimal weight: MI={best_weight:.2f}, Dice={1-best_weight:.2f}")
        
        return best_weight
    
    def _optimize_alpha(self, mi_scores, dice_scores):
        """Alpha: Maximize posterior/anterior ratio"""
        y_coords = self.voxel_coords_pooled[:, 1]
        posterior_mask = (y_coords < -40)
        anterior_mask = (y_coords > 0)
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        best_weight = 0.5
        best_metric = -np.inf
        
        for mi_weight in tqdm(self.mi_weights_to_test, desc="  Testing", ncols=80):
            hybrid = mi_weight * mi_scores + (1-mi_weight) * dice_scores
            avg_hybrid = hybrid.mean(axis=1)
            
            # Posterior/Anterior ratio
            post_sum = avg_hybrid[posterior_mask].sum()
            ant_sum = avg_hybrid[anterior_mask].sum()
            ratio = post_sum / (ant_sum + 1e-12)
            
            # K-fold stability
            fold_ratios = []
            for _, test_idx in kf.split(range(hybrid.shape[1])):
                fold_hybrid = hybrid[:, test_idx].mean(axis=1)
                fold_post = fold_hybrid[posterior_mask].sum()
                fold_ant = fold_hybrid[anterior_mask].sum()
                fold_ratios.append(fold_post / (fold_ant + 1e-12))
            
            stability = 1.0 - np.std(fold_ratios) / (np.mean(fold_ratios) + 1e-12)
            
            # Target: ratio=6.0
            target_ratio = 6.0
            ratio_score = 1.0 - min(1.0, abs(ratio - target_ratio) / target_ratio)
            
            metric = ratio_score * stability
            
            if metric > best_metric:
                best_metric = metric
                best_weight = mi_weight
        
        # Cap at 0.85
        if best_weight > 0.85:
            Logger.warn(f"  Alpha MI={best_weight:.2f} too high, capping at 0.85")
            best_weight = 0.85
        
        return best_weight
    
    def _optimize_beta(self, mi_scores, dice_scores):
        """Beta: Maximize central concentration"""
        y_coords = self.voxel_coords_pooled[:, 1]
        central_mask = (y_coords > -40) & (y_coords < 0)
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        best_weight = 0.5
        best_metric = -np.inf
        
        for mi_weight in tqdm(self.mi_weights_to_test, desc="  Testing", ncols=80):
            hybrid = mi_weight * mi_scores + (1-mi_weight) * dice_scores
            avg_hybrid = hybrid.mean(axis=1)
            
            # Central percentage
            central_pct = avg_hybrid[central_mask].sum() / (avg_hybrid.sum() + 1e-12)
            
            # K-fold stability
            fold_centrals = []
            for _, test_idx in kf.split(range(hybrid.shape[1])):
                fold_hybrid = hybrid[:, test_idx].mean(axis=1)
                fold_central = fold_hybrid[central_mask].sum() / (fold_hybrid.sum() + 1e-12)
                fold_centrals.append(fold_central)
            
            stability = 1.0 - np.std(fold_centrals) / (np.mean(fold_centrals) + 1e-12)
            
            # Target: 57.5%
            target_central = 0.575
            central_score = 1.0 - min(1.0, abs(central_pct - target_central) / target_central)
            
            metric = central_score * stability
            
            if metric > best_metric:
                best_metric = metric
                best_weight = mi_weight
        
        if best_weight < 0.45:
            Logger.warn(f"  Beta MI={best_weight:.2f} too low, raising to 0.45")
            best_weight = 0.45
        
        return best_weight
    
    def _optimize_delta(self, mi_scores, dice_scores):
        """Delta: Maximize diffuse distribution"""
        from scipy.stats import entropy
        
        y_coords = self.voxel_coords_pooled[:, 1]
        regions = {
            'posterior': y_coords < -40,
            'central': (y_coords >= -40) & (y_coords < 0),
            'anterior': y_coords >= 0
        }
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        best_weight = 0.5
        best_metric = -np.inf
        
        for mi_weight in tqdm(self.mi_weights_to_test, desc="  Testing", ncols=80):
            hybrid = mi_weight * mi_scores + (1-mi_weight) * dice_scores
            avg_hybrid = hybrid.mean(axis=1)
            
            # Regional entropy
            region_sums = [avg_hybrid[mask].sum() for mask in regions.values()]
            region_probs = np.array(region_sums) / (sum(region_sums) + 1e-12)
            dist_entropy = entropy(region_probs)
            
            # K-fold stability
            fold_entropies = []
            for _, test_idx in kf.split(range(hybrid.shape[1])):
                fold_hybrid = hybrid[:, test_idx].mean(axis=1)
                fold_sums = [fold_hybrid[mask].sum() for mask in regions.values()]
                fold_probs = np.array(fold_sums) / (sum(fold_sums) + 1e-12)
                fold_entropies.append(entropy(fold_probs))
            
            stability = 1.0 - np.std(fold_entropies) / (np.mean(fold_entropies) + 1e-12)
            
            metric = dist_entropy * stability
            
            if metric > best_metric:
                best_metric = metric
                best_weight = mi_weight
        
        return best_weight
    
    def _optimize_theta(self, mi_scores, dice_scores):
        """Theta: Maximize frontal-midline concentration"""
        y_coords = self.voxel_coords_pooled[:, 1]
        x_coords = self.voxel_coords_pooled[:, 0]
        
        # ✅ Broader frontal-midline mask
        frontal_midline_mask = (
            (y_coords > -10) & 
            (y_coords < 60) & 
            (np.abs(x_coords) < 40)
        )
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        best_weight = 0.5
        best_metric = -np.inf
        
        for mi_weight in tqdm(self.mi_weights_to_test, desc="  Testing", ncols=80):
            hybrid = mi_weight * mi_scores + (1-mi_weight) * dice_scores
            avg_hybrid = hybrid.mean(axis=1)
            
            # Frontal-midline percentage
            fm_pct = avg_hybrid[frontal_midline_mask].sum() / (avg_hybrid.sum() + 1e-12)
            
            # K-fold stability
            fold_fms = []
            for _, test_idx in kf.split(range(hybrid.shape[1])):
                fold_hybrid = hybrid[:, test_idx].mean(axis=1)
                fold_fm = fold_hybrid[frontal_midline_mask].sum() / (fold_hybrid.sum() + 1e-12)
                fold_fms.append(fold_fm)
            
            stability = 1.0 - np.std(fold_fms) / (np.mean(fold_fms) + 1e-12)
            
            # Target: 20%
            target_fm = 0.20
            fm_score = 1.0 - min(1.0, abs(fm_pct - target_fm) / target_fm)
            
            metric = fm_score * stability
            
            if metric > best_metric:
                best_metric = metric
                best_weight = mi_weight
        
        # Cap at 0.70
        if best_weight > 0.70:
            Logger.warn(f"  Theta MI={best_weight:.2f} too high, capping at 0.70")
            best_weight = 0.70
        
        return best_weight
    
    def _optimize_gamma(self, mi_scores, dice_scores):
        """Gamma: Maximize focal distribution"""
        from scipy.stats import kurtosis
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        best_weight = 0.5
        best_metric = -np.inf
        
        for mi_weight in tqdm(self.mi_weights_to_test, desc="  Testing", ncols=80):
            hybrid = mi_weight * mi_scores + (1-mi_weight) * dice_scores
            avg_hybrid = hybrid.mean(axis=1)
            
            # Spatial kurtosis
            spatial_kurt = kurtosis(avg_hybrid)
            
            # K-fold stability
            fold_kurts = []
            for _, test_idx in kf.split(range(hybrid.shape[1])):
                fold_hybrid = hybrid[:, test_idx].mean(axis=1)
                fold_kurts.append(kurtosis(fold_hybrid))
            
            stability = 1.0 - np.std(fold_kurts) / (np.mean(fold_kurts) + 1e-12)
            
            metric = spatial_kurt * stability
            
            if metric > best_metric:
                best_metric = metric
                best_weight = mi_weight
        
        return best_weight
    
    def optimize_all_bands(self, mi_dict_pooled, dice_dict_pooled):
        """
        Optimize all bands on pooled data.
        
        Args:
            mi_dict_pooled: {band: pooled MI array}
            dice_dict_pooled: {band: pooled Dice array}
        
        Returns:
            optimal_weights: {band: float}
        """
        Logger.section("GROUP K-FOLD OPTIMIZATION")
        
        optimal_weights = {}
        
        for band_name in mi_dict_pooled.keys():
            optimal_weights[band_name] = self.optimize_band(
                mi_dict_pooled[band_name],
                dice_dict_pooled[band_name],
                band_name
            )
        
        # Save results
        import json
        weights_path = Path(self.config.DATA_PATH) / "group_optimal_weights_v33.json"
        with open(weights_path, 'w') as f:
            json.dump(optimal_weights, f, indent=2)
        
        Logger.success(f"Group weights saved: {weights_path.name}")
        Logger.info("\n📊 GROUP OPTIMAL WEIGHTS:")
        for band, weight in optimal_weights.items():
            Logger.info(f"  {band:8s}: MI={weight:.2f}, Dice={1-weight:.2f}")
        
        return optimal_weights

# ============================================================================
# ✅ LOSO OPTIMIZER (v3.4 NEW!)
# ============================================================================

class LOSOOptimizer:
    """
    Leave-One-Subject-Out cross-validation optimizer.
    
    Difference from GroupKFold:
    - Each fold: Train on N-1 subjects, test on 1 subject
    - Returns MEDIAN weight across folds (more robust)
    - Can run in parallel (n_jobs=8)
    """
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.DATA_PATH) / config.CACHE_DIR
        self.n_jobs = config.LOSO_N_JOBS
    
    def optimize_band_loso(self, band_name, all_subjects):
        """
        Run LOSO for one band.
        
        Args:
            band_name: e.g., "theta"
            all_subjects: List of subject files
        
        Returns:
            optimal_weight: float (median across folds)
            fold_details: List[dict] (per-fold results)
        """
        
        Logger.section(f"LOSO OPTIMIZATION: {band_name.upper()}")
        Logger.info(f"N_FOLDS = {len(all_subjects)} (one per subject)")
        Logger.info(f"N_JOBS = {self.n_jobs} (parallel execution)")
        
        n_folds = len(all_subjects)
        
        # Parallel LOSO folds
        fold_results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._run_one_fold)(
                test_idx=i,
                all_subjects=all_subjects,
                band_name=band_name
            )
            for i in range(n_folds)
        )
        
        # Aggregate results
        weights = [r['optimal_weight'] for r in fold_results]
        test_metrics = [r['test_metrics'] for r in fold_results]
        
        optimal_weight = float(np.median(weights))
        
        Logger.info(f"\n{'='*70}")
        Logger.info(f"{band_name.upper()} LOSO RESULTS:")
        Logger.info(f"{'='*70}")
        
        for i, (w, subj) in enumerate(zip(weights, all_subjects)):
            subj_id = subj.split('_')[0]
            Logger.info(f"  Fold {i+1:2d} (Test={subj_id}): MI={w:.2f}")
        
        Logger.info(f"\n  Median:  {optimal_weight:.2f}")
        Logger.info(f"  Mean:    {np.mean(weights):.2f}")
        Logger.info(f"  Std:     {np.std(weights):.3f}")
        Logger.info(f"  Min/Max: {np.min(weights):.2f} / {np.max(weights):.2f}")
        Logger.info(f"{'='*70}\n")
        
        return optimal_weight, fold_results
    
    def _run_one_fold(self, test_idx, all_subjects, band_name):
        """
        Run one LOSO fold (train on N-1, test on 1).
        
        Returns:
            {
                'fold': int,
                'test_subject': str,
                'optimal_weight': float,
                'test_metrics': dict
            }
        """
        
        test_subject = all_subjects[test_idx]
        test_id = test_subject.split('_')[0]
        
        train_subjects = [
            s for i, s in enumerate(all_subjects) 
            if i != test_idx
        ]
        
        Logger.info(f"\n┌{'─'*68}┐")
        Logger.info(f"│ FOLD {test_idx+1}/{len(all_subjects)}: "
                   f"Test={test_id:3s}, Train={len(train_subjects)} subjects "
                   f"{'':20s}│")
        Logger.info(f"└{'─'*68}┘")
        
        # 1. Load train data (MI/Dice from cache)
        mi_list, dice_list, coords_list = [], [], []
        
        for train_subj in train_subjects:
            train_id = train_subj.split('_')[0]
            cache_path = self.cache_dir / f"{train_id}_{band_name}_mi_dice.pkl"
            
            if not cache_path.exists():
                Logger.error(f"  ❌ Cache not found: {cache_path.name}")
                continue
            
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                # ✅ Float64 → Float32 (RAM yarıya iner!)
                mi_list.append(data['mi'].astype(np.float32))
                dice_list.append(data['dice'].astype(np.float32))
                coords_list.append(data['voxel_coords'].astype(np.float32))
        
        if len(mi_list) == 0:
            Logger.error(f"  ❌ No train data loaded!")
            return {
                'fold': test_idx,
                'test_subject': test_id,
                'optimal_weight': 0.5,
                'test_metrics': {}
            }
        
        # 2. Pool train data
        mi_pooled = np.concatenate(mi_list, axis=0)
        dice_pooled = np.concatenate(dice_list, axis=0)
        coords_pooled = np.concatenate(coords_list, axis=0)
        
        Logger.info(f"  Train data pooled: {mi_pooled.shape}")
        
        # 3. K-fold optimize on TRAIN data
        optimizer = GroupKFoldOptimizer(self.config)
        optimizer.set_pooled_voxel_coords(coords_list)
        
        # Optimize this one band
        optimal_weight = optimizer.optimize_band(
            mi_pooled, 
            dice_pooled, 
            band_name
        )
        
        Logger.info(f"  ✅ Fold {test_idx+1} optimal: MI={optimal_weight:.2f}, "
                   f"Dice={1-optimal_weight:.2f}")
        
        # 4. Evaluate on test subject (optional, for outlier detection)
        test_metrics = self._evaluate_test_subject(
            test_subject, band_name, optimal_weight
        )
        
        # Cleanup
        del mi_pooled, dice_pooled, coords_pooled, mi_list, dice_list, coords_list
        gc.collect()
        
        return {
            'fold': test_idx,
            'test_subject': test_id,
            'optimal_weight': optimal_weight,
            'test_metrics': test_metrics
        }
    
    def _evaluate_test_subject(self, test_subject, band_name, mi_weight):
        """
        Build volume for test subject and compute QC metrics.
        
        Returns:
            metrics: dict (e.g., P/A ratio, occipital %)
        """
        
        test_id = test_subject.split('_')[0]
        
        try:
            # Load test subject's MI/Dice
            cache_path = self.cache_dir / f"{test_id}_{band_name}_mi_dice.pkl"
            
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                mi = data['mi']
                dice = data['dice']
                voxel_coords = data['voxel_coords']
            
            # Apply weight
            hybrid = mi_weight * mi + (1 - mi_weight) * dice
            
            # Normalize
            hmin, hmax = hybrid.min(), hybrid.max()
            if hmax > hmin:
                hybrid = (hybrid - hmin) / (hmax - hmin)
            
            # Quick metrics (simplified, no full volume build)
            # Just check posterior/anterior distribution
            y_coords = voxel_coords[:, 1]
            
            avg_hybrid = hybrid.mean(axis=1)
            
            posterior_mask = (y_coords < -40)
            anterior_mask = (y_coords > 0)
            
            post_sum = avg_hybrid[posterior_mask].sum()
            ant_sum = avg_hybrid[anterior_mask].sum()
            
            pa_ratio = post_sum / (ant_sum + 1e-12)
            
            # Occipital
            occ_mask = (y_coords < -70)
            occ_pct = 100.0 * avg_hybrid[occ_mask].sum() / (avg_hybrid.sum() + 1e-12)
            
            metrics = {
                'pa_ratio': float(pa_ratio),
                'occipital_%': float(occ_pct)
            }
            
            Logger.info(f"  Test metrics: P/A={pa_ratio:.2f}, Occ={occ_pct:.1f}%")
            
            del hybrid, mi, dice
            gc.collect()
            
            return metrics
        
        except Exception as e:
            Logger.warn(f"  ⚠️ Could not evaluate test subject: {e}")
            return {}
    
    def optimize_all_bands_loso(self, all_subjects):
        """
        Run LOSO for ALL bands sequentially (MEMORY-EFFICIENT).
    
        Key change: Process ONE band at a time with reduced parallelism.
        """
    
        Logger.section("="*70)
        Logger.section("LOSO OPTIMIZATION - ALL BANDS (Memory-Efficient)")
        Logger.section("="*70)
    
        optimal_weights = {}
        all_fold_results = {}
    
        # ✅ RAM-SAFE SETTINGS
        original_n_jobs = self.n_jobs
    
        # Dynamically adjust based on available memory
        import psutil
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
    
        if available_ram_gb < 6:
            safe_n_jobs = 2
            Logger.warn(f"⚠️  Low RAM ({available_ram_gb:.1f}GB) - using n_jobs=2")
        elif available_ram_gb < 10:
            safe_n_jobs = 4
            Logger.info(f"💾 Moderate RAM ({available_ram_gb:.1f}GB) - using n_jobs=4")
        else:
            safe_n_jobs = min(6, original_n_jobs)
            Logger.info(f"💾 Good RAM ({available_ram_gb:.1f}GB) - using n_jobs={safe_n_jobs}")
    
        self.n_jobs = safe_n_jobs
    
        for band_idx, band_name in enumerate(self.config.BANDS.keys()):
            Logger.info(f"\n{'='*70}")
            Logger.info(f"BAND {band_idx+1}/5: {band_name.upper()}")
            Logger.info(f"{'='*70}\n")
        
            try:
                optimal_weight, fold_results = self.optimize_band_loso(
                    band_name, all_subjects
               )
            
                optimal_weights[band_name] = optimal_weight
                all_fold_results[band_name] = fold_results
            
                # Save intermediate (in case of crash)
                self._save_intermediate(optimal_weights, all_fold_results)
            
                Logger.success(f"✅ {band_name.upper()} completed: MI={optimal_weight:.2f}\n")
            
                # Force garbage collection between bands
                import gc
                gc.collect()
            
                # Small break to let system recover
                import time
                time.sleep(3)
            
            except Exception as e:
               Logger.error(f"❌ {band_name} failed: {e}")
               Logger.warn(f"   Skipping {band_name}, continuing with next band...")
               optimal_weights[band_name] = 0.50  # Default fallback
               all_fold_results[band_name] = []
    
        # Restore original n_jobs
        self.n_jobs = original_n_jobs
    
        # Save final results
        self._save_final(optimal_weights, all_fold_results)
    
        Logger.success("\n✅ LOSO OPTIMIZATION COMPLETED FOR ALL BANDS")
        Logger.info("\n📊 FINAL OPTIMAL WEIGHTS (LOSO MEDIAN):")
        for band, weight in optimal_weights.items():
            Logger.info(f"  {band:8s}: MI={weight:.2f}, Dice={1-weight:.2f}")
    
        return optimal_weights, all_fold_results
    
    def _save_intermediate(self, optimal_weights, fold_results):
        """Save intermediate results"""
        output_dir = Path(self.config.DATA_PATH)
        
        weights_path = output_dir / "loso_weights_intermediate_v34.json"
        with open(weights_path, 'w') as f:
            json.dump(optimal_weights, f, indent=2)
    
    def _save_final(self, optimal_weights, fold_results):
        """Save final LOSO results"""
        output_dir = Path(self.config.DATA_PATH)
        
        # Weights
        weights_path = output_dir / "loso_optimal_weights_v34.json"
        with open(weights_path, 'w') as f:
            json.dump(optimal_weights, f, indent=2)
        
        Logger.success(f"LOSO weights saved: {weights_path.name}")
        
        # Detailed results
        results_path = output_dir / "loso_fold_details_v34.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump({
                'optimal_weights': optimal_weights,
                'fold_results': fold_results,
                'config': {
                    'n_folds': len(self.config.SUBJECTS),
                    'n_jobs': self.n_jobs,
                    'bands': list(self.config.BANDS.keys())
                }
            }, f)
        
        Logger.success(f"LOSO details saved: {results_path.name}")
        
# ============================================================================
# VOLUME BUILDER (unchanged)
# ============================================================================

class VolumeBuilder:
    """Build 4D NIfTI volumes from voxel data"""
    
    def __init__(self, grid):
        self.grid = grid
    
    def build(self, voxel_data, voxel_coords):
        n_voxels, n_timepoints = voxel_data.shape
        x, y, z = self.grid.shape
        
        volume = np.zeros((x, y, z, n_timepoints), dtype=np.float32)
        
        placed = 0
        for i in tqdm(range(n_voxels), desc="Building volume", ncols=80):
            mni_coord = voxel_coords[i]
            xi, yi, zi = self.grid.mni_to_voxel(mni_coord)
            
            if 0 <= xi < x and 0 <= yi < y and 0 <= zi < z:
                volume[xi, yi, zi, :] = voxel_data[i, :]
                placed += 1
        
        Logger.info(f"  Placed {placed}/{n_voxels} voxels into volume")
        
        return volume

# ============================================================================
# POST-PROCESSING (from v3.2, keeping all methods)
# ============================================================================

class PostProcessor:
    """Volume post-processing"""
    
    def __init__(self, config, grid):
        self.config = config
        self.grid = grid
    
    def zscore_normalize(self, volume, mode="global"):
        Logger.info(f"Z-score normalization (mode={mode})...")
        
        x, y, z, T = volume.shape
        
        if mode == "global":
            active_mask = (volume > 0)
            active_values = volume[active_mask]
            
            if active_values.size > 0:
                mean_global = active_values.mean()
                std_global = active_values.std()
                
                if std_global > 0:
                    volume[active_mask] = (volume[active_mask] - mean_global) / std_global
        
        elif mode == "active":
            for t in range(T):
                vol_t = volume[:, :, :, t]
                active = (vol_t > 0)
                
                if active.any():
                    mean_t = vol_t[active].mean()
                    std_t = vol_t[active].std()
                    
                    if std_t > 0:
                        vol_t[active] = (vol_t[active] - mean_t) / std_t
                
                volume[:, :, :, t] = vol_t
        
        elif mode == "timepoint":
            for t in range(T):
                vol_t = volume[:, :, :, t]
                mean_t = vol_t.mean()
                std_t = vol_t.std()
                
                if std_t > 0:
                    volume[:, :, :, t] = (vol_t - mean_t) / std_t
        
        Logger.success("Z-score: Applied")
        return volume
    
    def temporal_consistency(self, volume):
        if not self.config.TEMPORAL_FILTERING:
            return volume
        
        Logger.info(f"Temporal consistency (window={self.config.TEMPORAL_WINDOW})...")
        
        mask = (volume > 0).astype(np.float32)
        
        smoothed = uniform_filter1d(mask, size=self.config.TEMPORAL_WINDOW, 
                                   axis=3, mode='reflect')
        
        keep_mask = smoothed >= self.config.TEMPORAL_MIN_FRAC
        
        volume[~keep_mask] = 0.0
        
        removed = (~keep_mask & (mask > 0)).sum()
        Logger.info(f"  Removed {removed} temporally inconsistent voxels")
        
        return volume
    
    def roi_balanced_sparsify(self, volume, band_name):
        Logger.info(f"ROI-balanced sparsification [{band_name}]...")
        
        x, y, z, T = volume.shape
        atlas = self.grid.atlas_idx
        roi_names = self.grid.roi_names
        
        global_pct = self.config.KEEP_TOP_PCT.get(band_name, 0.40)
        global_min = self.config.MIN_VOXELS_GLOBAL.get(band_name, 20000)
        posterior_boost = self.config.POSTERIOR_ROI_BOOST.get(band_name, 1.0)
        
        roi_labels = np.unique(atlas)
        roi_labels = roi_labels[roi_labels > 0]
        
        total_before = int((volume > 0).sum())
        total_after = 0
        
        for t in range(T):
            vol_t = volume[:, :, :, t]
            active_mask = (vol_t > 0)
            n_active = int(active_mask.sum())
            
            if n_active == 0:
                continue
            
            final_mask = np.zeros_like(active_mask, dtype=bool)
            
            # OCCIPITAL PROTECTION
            for roi_label in self.config.OCCIPITAL_ROI_INDICES:
                roi_mask = (atlas == roi_label) & active_mask
                n_roi = int(roi_mask.sum())
                
                if n_roi > 0:
                    K_occ = max(
                        int(self.config.OCCIPITAL_MIN_COVERAGE * n_roi),
                        min(self.config.OCCIPITAL_MIN_VOXELS, n_roi)
                    )
                    
                    roi_values = vol_t[roi_mask]
                    if len(roi_values) >= K_occ:
                        threshold_occ = np.partition(roi_values, -K_occ)[-K_occ]
                        occ_keep = roi_mask & (vol_t >= threshold_occ)
                    else:
                        occ_keep = roi_mask
                    
                    final_mask |= occ_keep
            
            # FRONTAL PROTECTION
            if hasattr(self.config, 'FRONTAL_ROI_INDICES'):
                for roi_label in self.config.FRONTAL_ROI_INDICES:
                    roi_mask = (atlas == roi_label) & active_mask
                    n_roi = int(roi_mask.sum())
                    
                    if n_roi > 0:
                        K_frontal = max(
                            int(self.config.FRONTAL_MIN_COVERAGE * n_roi),
                            min(self.config.FRONTAL_MIN_VOXELS, n_roi)
                        )
                        
                        roi_values = vol_t[roi_mask]
                        if len(roi_values) >= K_frontal:
                            threshold_frontal = np.partition(roi_values, -K_frontal)[-K_frontal]
                            frontal_keep = roi_mask & (vol_t >= threshold_frontal)
                        else:
                            frontal_keep = roi_mask
                        
                        final_mask |= frontal_keep
            
            # GLOBAL TOP-K
            non_occ_mask = active_mask & (~final_mask)
            n_non_occ = int(non_occ_mask.sum())
            
            if n_non_occ > 0:
                K_global = max(
                    int(global_pct * n_non_occ),
                    global_min - int(final_mask.sum())
                )
                K_global = min(K_global, n_non_occ)
                
                if K_global > 0:
                    non_occ_values = vol_t[non_occ_mask]
                    threshold_global = np.partition(non_occ_values, -K_global)[-K_global]
                    global_keep = non_occ_mask & (vol_t >= threshold_global)
                    final_mask |= global_keep
            
            # OTHER ROI MINIMUM QUOTA
            for roi_label in roi_labels:
                if roi_label in self.config.OCCIPITAL_ROI_INDICES:
                    continue
                
                roi_mask = (atlas == roi_label) & active_mask & (~final_mask)
                n_roi = int(roi_mask.sum())
                
                if n_roi > 0:
                    roi_pct = self.config.PER_ROI_KEEP_PCT
                    roi_min = self.config.PER_ROI_MIN_VOXELS
                    
                    if roi_label in self.config.POSTERIOR_ROI_INDICES:
                        roi_pct *= posterior_boost
                        roi_min = int(roi_min * posterior_boost)
                    
                    K_roi = max(int(roi_pct * n_roi), min(roi_min, n_roi))
                    
                    roi_values = vol_t[roi_mask]
                    if len(roi_values) >= K_roi:
                        threshold_roi = np.partition(roi_values, -K_roi)[-K_roi]
                        roi_keep = roi_mask & (vol_t >= threshold_roi)
                    else:
                        roi_keep = roi_mask
                    
                    final_mask |= roi_keep
            
            kept = int(final_mask.sum())
            total_after += kept
            
            vol_t[~final_mask] = 0.0
            volume[:, :, :, t] = vol_t
        
        Logger.info(f"  Sparsified: {total_before} → {total_after} voxels "
                   f"({100*total_after/max(total_before,1):.1f}%)")
        
        return volume
    
    def anisotropic_smooth(self, volume, fwhm_xyz=(3.0, 3.0, 3.0)):
        Logger.info(f"Anisotropic smoothing (FWHM_XYZ={fwhm_xyz}mm)...")
        
        from scipy.ndimage import gaussian_filter
        
        sigma_mm = [f / 2.355 for f in fwhm_xyz]
        sigma_voxels = [s / self.config.GRID_SPACING for s in sigma_mm]
        
        Logger.info(f"  Sigma (voxels): X={sigma_voxels[0]:.2f}, Y={sigma_voxels[1]:.2f}, Z={sigma_voxels[2]:.2f}")
        
        x, y, z, T = volume.shape
        smoothed = np.zeros_like(volume)
        
        for t in range(T):
            vol_t = volume[:, :, :, t]
            mask = (vol_t > 0)
            
            if not mask.any():
                continue
            
            smoothed_t = gaussian_filter(
                vol_t,
                sigma=sigma_voxels,
                mode='constant',
                cval=0.0
            )
            
            smoothed_t[~mask] = 0.0
            
            blended = 0.85 * vol_t + 0.15 * smoothed_t
            blended[~mask] = 0.0
            
            smoothed[:, :, :, t] = blended
        
        Logger.success("Anisotropic smoothing: Applied")
        return smoothed
    
    def ray_killer_cleanup(self, volume):
        if not self.config.RAYKILL_ENABLE:
            return volume
        
        Logger.info("Ray-killer: Removing isolated clusters...")
        
        x, y, z, T = volume.shape
        gm_mask = self.grid.gm_mask
        
        mm3_per_voxel = self.config.GRID_SPACING ** 3
        min_voxels = max(8, int(self.config.RAYKILL_MIN_CLUSTER_MM3 / mm3_per_voxel))
        
        kept_total = 0
        removed_total = 0
        
        for t in range(T):
            vol_t = volume[:, :, :, t]
            
            if not np.any(vol_t > 0):
                continue
            
            vol_orig = vol_t.copy()
            active_mask = (vol_t > 0) & gm_mask
            n_active = int(active_mask.sum())
            
            if n_active == 0:
                volume[:, :, :, t] = 0.0
                continue
            
            labeled, n_clusters = label(active_mask)
            
            if n_clusters > 0:
                counts = np.bincount(labeled.ravel())
                
                drop_labels = np.where(counts < min_voxels)[0]
                
                if len(drop_labels) > 0:
                    drop_mask = np.isin(labeled, drop_labels)
                    vol_t[drop_mask] = 0.0
            
            kept = int((vol_t > 0).sum())
            
            keep_floor = max(
                int(self.config.RAYKILL_KEEP_FLOOR_PCT * n_active),
                self.config.RAYKILL_KEEP_FLOOR_ABS
            )
            
            if kept < keep_floor:
                rescue_values = vol_orig[active_mask]
                K_rescue = min(len(rescue_values), keep_floor)
                threshold_rescue = np.partition(rescue_values, -K_rescue)[-K_rescue]
                
                rescue_mask = active_mask & (vol_orig >= threshold_rescue)
                vol_t = np.where(rescue_mask, vol_orig, 0.0)
                kept = int((vol_t > 0).sum())
            
            kept_total += kept
            removed_total += (n_active - kept)
            
            volume[:, :, :, t] = vol_t
        
        Logger.info(f"  Kept: {kept_total}, Removed: {removed_total}")
        
        return volume

# ============================================================================
# QUALITY CONTROL (unchanged from v3.2)
# ============================================================================

class QualityControl:
    """Quality control metrics and visualization"""
    
    def __init__(self, grid):
        self.grid = grid
    
    def physiological_distribution(self, volume, band_name):
        Logger.info(f"\n{'='*70}")
        Logger.info(f"QC: Physiological Distribution [{band_name.upper()}]")
        Logger.info(f"{'='*70}")
        
        affine = self.grid.affine
        gm_mask = self.grid.gm_mask
        
        x, y, z = self.grid.shape
        yi = np.arange(y)
        Y_coords = (affine @ np.vstack([
            np.zeros_like(yi), yi, np.zeros_like(yi), np.ones_like(yi)
        ]))[1, :]
        
        Y_grid = Y_coords[np.newaxis, :, np.newaxis, np.newaxis]
        gm_4d = gm_mask[..., np.newaxis]
        
        regions = {
            "very_posterior": (Y_grid <= -85),
            "occipital": (Y_grid > -85) & (Y_grid <= -70),
            "posterior": (Y_grid > -70) & (Y_grid <= -40),
            "central": (Y_grid > -40) & (Y_grid <= 0),
            "anterior": (Y_grid > 0) & (Y_grid <= 40),
            "frontal": (Y_grid > 40)
        }
        
        V = volume.copy()
        V[V < 0] = 0
        
        total = V[gm_4d.squeeze(-1)].sum() + 1e-12
        
        results = {}
        
        Logger.info(f"{'Region':<20} {'%':>8} {'Voxels':>10} {'Mean':>10}")
        Logger.info(f"{'-'*52}")
        
        for region_name, region_mask in regions.items():
            region_data = V[region_mask.squeeze(-1) & gm_4d.squeeze(-1)]
            
            region_sum = region_data.sum()
            pct = 100.0 * region_sum / total
            n_active = int((region_data > 0).sum())
            mean_act = region_data[region_data > 0].mean() if n_active > 0 else 0.0
            
            results[f"{band_name}_{region_name}_%"] = float(pct)
            results[f"{band_name}_{region_name}_nvoxels"] = n_active
            results[f"{band_name}_{region_name}_mean"] = float(mean_act)
            
            Logger.info(f"{region_name.replace('_', ' ').title():<20} {pct:>6.1f}%  {n_active:>8d}  {mean_act:>8.3f}")
        
        if band_name == "alpha":
            post_sum = (V[regions["very_posterior"].squeeze(-1) & gm_4d.squeeze(-1)].sum() +
                       V[regions["occipital"].squeeze(-1) & gm_4d.squeeze(-1)].sum() +
                       V[regions["posterior"].squeeze(-1) & gm_4d.squeeze(-1)].sum())
            
            ant_sum = (V[regions["anterior"].squeeze(-1) & gm_4d.squeeze(-1)].sum() +
                      V[regions["frontal"].squeeze(-1) & gm_4d.squeeze(-1)].sum())
            
            ratio = post_sum / max(ant_sum, 1e-12)
            results["alpha_post_ant_ratio"] = float(ratio)
            
            occ_pct = results[f"{band_name}_very_posterior_%"] + results[f"{band_name}_occipital_%"]
            
            Logger.info(f"\n{'='*52}")
            Logger.info(f"📊 ALPHA POSTERIOR/ANTERIOR RATIO: {ratio:.2f}")
            Logger.info(f"   Literature: 2.0-15.0 (Klimesch 1999)")
            
            if 2.0 <= ratio <= 15.0:
                Logger.success(f"   ✅ WITHIN literature range")
            elif ratio < 2.0:
                Logger.warn(f"   ⚠️  BELOW range (too diffuse)")
            else:
                Logger.warn(f"   ⚠️  ABOVE range (too focal)")
            
            Logger.info(f"\n📊 ALPHA OCCIPITAL COVERAGE: {occ_pct:.1f}%")
            Logger.info(f"   (Very Posterior + Occipital regions)")
            Logger.info(f"   Literature: 25-40% (Klimesch 1999)")
            
            if 18.0 <= occ_pct <= 40.0:
                Logger.success(f"   ✅ ACCEPTABLE (conservative estimate)")
            elif occ_pct < 18.0:
                Logger.error(f"   ❌ TOO LOW - check parameters")
            else:
                Logger.success(f"   ✅ EXCELLENT coverage")
            
            Logger.info(f"{'='*52}")
        
        Logger.info("")
        
        return results
    
    def roi_coverage_analysis(self, volume, band_name):
        Logger.info(f"\n=== ROI Coverage Analysis [{band_name}] ===")
        
        atlas = self.grid.atlas_idx
        roi_names = self.grid.roi_names
        
        roi_labels = np.unique(atlas)
        roi_labels = roi_labels[roi_labels > 0]
        
        roi_stats = {}
        
        for roi_label in roi_labels:
            roi_name = roi_names[roi_label] if roi_label < len(roi_names) else f"ROI_{roi_label}"
            roi_mask = (atlas == roi_label)
            
            active_any_time = np.any(volume > 0, axis=3)
            active_voxels = int(np.sum(active_any_time & roi_mask))
            total_voxels = int(np.sum(roi_mask))
            
            roi_mask_4d = roi_mask[..., np.newaxis]
            roi_data_all = volume[np.broadcast_to(roi_mask_4d, volume.shape)]
            roi_data_active = roi_data_all[roi_data_all > 0]
            
            mean_act = float(roi_data_active.mean()) if roi_data_active.size > 0 else 0.0
            max_act = float(roi_data_active.max()) if roi_data_active.size > 0 else 0.0
            
            roi_stats[roi_label] = {
                'name': roi_name,
                'total_voxels': total_voxels,
                'active_voxels': active_voxels,
                'coverage_%': float(100.0 * active_voxels / max(total_voxels, 1)),
                'mean_activation': mean_act,
                'max_activation': max_act
            }
        
        covered = {k: v for k, v in roi_stats.items() if v['active_voxels'] > 100}
        weak = {k: v for k, v in roi_stats.items() if 0 < v['active_voxels'] <= 100}
        missing = {k: v for k, v in roi_stats.items() if v['active_voxels'] == 0}
        
        Logger.info(f"Total ROIs: {len(roi_labels)}")
        Logger.info(f"✅ Covered (>100 voxels): {len(covered)}")
        Logger.info(f"⚠️  Weak (1-100 voxels): {len(weak)}")
        Logger.info(f"❌ Missing (0 voxels): {len(missing)}")
        
        if missing:
            Logger.info("\n❌ MISSING ROIs:")
            for label, info in sorted(missing.items(), 
                                     key=lambda x: x[1]['total_voxels'], 
                                     reverse=True)[:10]:
                Logger.info(f"  [{label:2d}] {info['name'][:45]:45s} (GM: {info['total_voxels']:4d})")
        
        Logger.info("\n✅ TOP 10 Covered ROIs:")
        top = sorted(covered.items(), 
                    key=lambda x: x[1]['active_voxels'], 
                    reverse=True)[:10]
        for label, info in top:
            Logger.info(f"  [{label:2d}] {info['name'][:35]:35s} | "
                       f"Active: {info['active_voxels']:6d} | "
                       f"Mean: {info['mean_activation']:.3f}")
        
        return covered, weak, missing, roi_stats
    
    def export_coverage_csv(self, roi_stats, band_name, output_dir):
        import csv
        
        csv_path = Path(output_dir) / f"{band_name}_roi_coverage.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'ROI_ID', 'ROI_Name', 'Total_GM_Voxels', 'Active_Voxels',
                'Coverage_%', 'Mean_Activation', 'Max_Activation', 'Status'
            ])
            
            for label, info in sorted(roi_stats.items()):
                if info['active_voxels'] > 100:
                    status = 'Covered'
                elif info['active_voxels'] > 0:
                    status = 'Weak'
                else:
                    status = 'Missing'
                
                writer.writerow([
                    label,
                    info['name'],
                    info['total_voxels'],
                    info['active_voxels'],
                    f"{info['coverage_%']:.2f}",
                    f"{info['mean_activation']:.4f}",
                    f"{info['max_activation']:.4f}",
                    status
                ])
        
        Logger.success(f"CSV: {csv_path.name}")

# ============================================================================
# CONNECTIVITY ANALYSIS (unchanged)
# ============================================================================

class ConnectivityAnalyzer:
    """CONN-compatible connectivity matrices"""
    
    def __init__(self, config):
        self.config = config
    
    def compute(self, nifti_img, band_name, output_dir):
        Logger.info(f"CONN: Computing connectivity [{band_name}]...")
        
        try:
            ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            labels_img = (ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) 
                         else nib.load(ho_cort.maps))
            roi_names = list(ho_cort.labels)[1:]
            
            masker = NiftiLabelsMasker(
                labels_img=labels_img,
                labels=roi_names,
                background_label=0,
                standardize=self.config.CONN_STANDARDIZE,
                detrend=self.config.CONN_DETREND,
                verbose=0
            )
            
            timeseries = masker.fit_transform(nifti_img)
            
            active_cols = np.where(np.any(timeseries != 0, axis=0))[0]
            timeseries = timeseries[:, active_cols]
            active_names = [roi_names[i] for i in active_cols]
            
            Logger.info(f"  Active ROIs: {len(active_names)}/{len(roi_names)}")
            
            if self.config.CONN_GSR and timeseries.shape[1] > 0:
                Logger.info("  Applying GSR...")
                global_signal = timeseries.mean(axis=1, keepdims=True)
                beta = ((global_signal * timeseries).sum(axis=0) / 
                       ((global_signal * global_signal).sum() + 1e-12))
                timeseries = timeseries - global_signal * beta
            
            if timeseries.shape[1] >= 2:
                R = np.corrcoef(timeseries.T)
                
                if self.config.CONN_FISHER_Z:
                    Rz = np.arctanh(np.clip(R, -0.999999, 0.999999))
                else:
                    Rz = R.copy()
                
                if self.config.CONN_PARTIAL:
                    Logger.info("  Computing partial correlations...")
                    lw = LedoitWolf().fit(timeseries)
                    precision = lw.precision_
                    diag_sqrt = np.sqrt(np.outer(np.diag(precision), np.diag(precision))) + 1e-12
                    partial_corr = -precision / diag_sqrt
                    np.fill_diagonal(partial_corr, 1.0)
                else:
                    partial_corr = np.zeros_like(R)
            else:
                R = np.zeros((1, 1))
                Rz = R.copy()
                partial_corr = R.copy()
            
            # Save files
            ts_path = Path(output_dir) / f"{band_name}_voxel_conn.mat"
            scipy.io.savemat(ts_path, {
                'roi_timeseries': timeseries,
                'roi_names': np.array(active_names, dtype=object),
                'n_timepoints': int(timeseries.shape[0]),
                'n_rois': int(timeseries.shape[1]),
                'TR': float(self.config.SEGMENT_DURATION),
            })
            
            conn_path = Path(output_dir) / f"{band_name}_voxel_connectivity.mat"
            scipy.io.savemat(conn_path, {
                'connectivity_pearson_r': R,
                'connectivity_pearson_z': Rz,
                'connectivity_partial': partial_corr,
                'n_rois': int(timeseries.shape[1]),
            })
            
            Logger.success(f"CONN: Saved {ts_path.name}, {conn_path.name}")
            
        except Exception as e:
            Logger.error(f"CONN error: {e}")

# ============================================================================
# ✅ MAIN GROUP PIPELINE (NEW!)
# ============================================================================

class GroupEEGtoFMRIPipeline:
    """
    Group-level EEG-to-fMRI pipeline with pooled K-fold optimization.
    
    Processing flow:
    ----------------
    PHASE 1: Process all subjects → compute MI/Dice → cache
    PHASE 2: Pool data → group K-fold → find optimal weights
    PHASE 3: Apply optimal weights → build volumes for all subjects
    """
    
    def __init__(self, config):
        self.config = config
        self.start_time = datetime.now()
        
        # Create cache directory
        self.cache_dir = Path(config.DATA_PATH) / config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        Logger.info(f"Cache directory: {self.cache_dir}")
    
    def _get_cache_path(self, subject_id, band_name):
        """Get cache file path for MI/Dice data"""
        return self.cache_dir / f"{subject_id}_{band_name}_mi_dice.pkl"
    
    def _process_one_subject_mi_dice(self, subject_file):
        """
        Process one subject up to MI/Dice computation.
        
        Returns:
            subject_id: str
            mi_dict: {band_name: (n_voxels, n_timepoints)}
            dice_dict: {band_name: (n_voxels, n_timepoints)}
            voxel_coords: (n_voxels, 3)
        """
        subject_id = subject_file.split('_')[0]
        
        Logger.section(f"PHASE 1: Processing {subject_id}")
        
        # Check if already cached
        cached = True
        for band_name in self.config.BANDS.keys():
            cache_path = self._get_cache_path(subject_id, band_name)
            if not cache_path.exists():
                cached = False
                break
        
        if cached and self.config.CACHE_MI_DICE:
            Logger.info(f"✅ {subject_id}: Loading from cache...")
            
            mi_dict = {}
            dice_dict = {}
            voxel_coords = None
            
            for band_name in self.config.BANDS.keys():
                cache_path = self._get_cache_path(subject_id, band_name)
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    mi_dict[band_name] = data['mi']
                    dice_dict[band_name] = data['dice']
                    if voxel_coords is None:
                        voxel_coords = data['voxel_coords']
            
            Logger.success(f"{subject_id}: Loaded from cache")
            return subject_id, mi_dict, dice_dict, voxel_coords
        
        # Load EEG
        Logger.info(f"{subject_id}: Loading EEG...")
        eeg_path = Path(self.config.DATA_PATH) / subject_file
        
        if not eeg_path.exists():
            Logger.error(f"❌ File not found: {eeg_path}")
            return subject_id, None, None, None
        
        try:
            mat_data = scipy.io.loadmat(eeg_path, squeeze_me=True, struct_as_record=False)
            eeg_raw = np.array(mat_data['dataRest']).astype(np.float32)[:64]
            Logger.info(f"  EEG shape: {eeg_raw.shape}")
        except Exception as e:
            Logger.error(f"❌ Failed to load EEG: {e}")
            return subject_id, None, None, None
        
        # Preprocessing
        Logger.info(f"{subject_id}: Preprocessing...")
        eeg_filtered = SignalProcessor.bandpass(eeg_raw, self.config.FS, 1, 45)
        eeg_filtered = SignalProcessor.notch(eeg_filtered, self.config.FS, 50.0)
        
        # Coordinates
        ch_coords, ch_order = CoordinateSystem.load_coordinates()
        
        # ICA
        Logger.info(f"{subject_id}: ICA artifact removal...")
        ica_remover = ICAArtifactRemover(self.config)
        eeg_clean, ica_info = ica_remover.clean(eeg_filtered, self.config.FS, ch_order)
        
        if ica_info['reasons']:
            Logger.info(f"  Removed {len(ica_info['bad_idx'])} components")
        
        # CSD
        Logger.info(f"{subject_id}: CSD re-referencing...")
        eeg_csd = CSDReferencer.apply(eeg_clean, ch_coords, ch_order, self.config)
        
        # Segmentation
        Logger.info(f"{subject_id}: Segmentation...")
        segment_samples = int(self.config.FS * self.config.SEGMENT_DURATION)
        n_segments = eeg_csd.shape[1] // segment_samples
        
        segments = np.stack([
            eeg_csd[:, i*segment_samples:(i+1)*segment_samples]
            for i in range(n_segments)
        ])
        
        Logger.info(f"  Segments: {n_segments}")
        
        # Grid & Signatures
        Logger.info(f"{subject_id}: Creating voxel grid...")
        grid = VoxelGrid(self.config)
        
        Logger.info(f"{subject_id}: Computing signatures...")
        signature_computer = SignatureComputer(self.config)
        voxel_sigs = signature_computer.compute(grid.coords_gm, ch_coords, ch_order)
        
        # Compute MI/Dice for all bands
        Logger.info(f"{subject_id}: Computing MI/Dice for all bands...")
        mi_dict = {}
        dice_dict = {}
        
        for band_name, freq_range in self.config.BANDS.items():
            Logger.info(f"\n  Band: {band_name.upper()} ({freq_range[0]}-{freq_range[1]} Hz)")
            
            try:
                # Hilbert envelope
                snapshots = np.stack([
                    SignalProcessor.hilbert_envelope(seg, freq_range, self.config.FS)
                    for seg in tqdm(segments, desc=f"    Envelopes", ncols=80)
                ])
                
                # MI/Dice
                mi_dice_computer = MIDiceComputer(self.config)
                mi_scores, dice_scores = mi_dice_computer.compute(voxel_sigs, snapshots)
                
                mi_dict[band_name] = mi_scores
                dice_dict[band_name] = dice_scores
                
                # Cache
                if self.config.CACHE_MI_DICE:
                    cache_path = self._get_cache_path(subject_id, band_name)
                    with open(cache_path, 'wb') as f:
                        pickle.dump({
                            'mi': mi_scores,
                            'dice': dice_scores,
                            'voxel_coords': grid.coords_gm
                        }, f)
                    Logger.info(f"    Cached: {cache_path.name}")
                
                Logger.success(f"  ✅ {band_name}: MI/Dice computed")
                
                # Cleanup
                del snapshots, mi_scores, dice_scores
                gc.collect()
                
            except Exception as e:
                Logger.error(f"  ❌ Failed on {band_name}: {e}")
                mi_dict[band_name] = np.zeros((len(grid.coords_gm), n_segments), dtype=np.float32)
                dice_dict[band_name] = np.zeros_like(mi_dict[band_name])
        
        Logger.success(f"{subject_id}: PHASE 1 completed")
        
        return subject_id, mi_dict, dice_dict, grid.coords_gm
    
    def _phase1_process_all_subjects(self):
        """
        PHASE 1: Process all subjects and compute MI/Dice.
        
        Returns:
            subject_data: List of (subject_id, mi_dict, dice_dict, voxel_coords)
        """
        Logger.section("="*70)
        Logger.section("PHASE 1: PROCESSING ALL SUBJECTS (MI/DICE COMPUTATION)")
        Logger.section("="*70)
        
        subject_data = []
        
        for i, subject_file in enumerate(self.config.SUBJECTS):
            Logger.info(f"\n📍 Subject {i+1}/{len(self.config.SUBJECTS)}: {subject_file}")
            
            result = self._process_one_subject_mi_dice(subject_file)
            
            if result[1] is not None:  # mi_dict not None
                subject_data.append(result)
            else:
                Logger.warn(f"⚠️  Skipping {result[0]} (processing failed)")
        
        Logger.success(f"\n✅ PHASE 1 COMPLETED: {len(subject_data)}/{len(self.config.SUBJECTS)} subjects processed")
        
        return subject_data
    
    
    def _phase2_optimization(self, subject_data):
        """
        PHASE 2: Optimize weights (Group K-fold OR LOSO).
    
        Args:
            subject_data: List of (subject_id, mi_dict, dice_dict, voxel_coords)
    
        Returns:
            optimal_weights: {band_name: float}
        """
    
        if self.config.USE_LOSO:
            # v3.4: LOSO
            return self._phase2_loso(subject_data)
        else:
            # v3.3: Group K-fold
           return self._phase2_group_kfold(subject_data)

    def _phase2_loso(self, subject_data):
        """
        PHASE 2: LOSO optimization (v3.4).
        """
    
        Logger.section("="*70)
        Logger.section("PHASE 2: LOSO OPTIMIZATION")
        Logger.section("="*70)
    
        # Extract subject list
        all_subjects = [data[0] + "_restingPre_EC.mat" for data in subject_data]
    
        Logger.info(f"LOSO: {len(all_subjects)} subjects")
        Logger.info(f"Parallel: {self.config.LOSO_N_JOBS} jobs")
    
        # Run LOSO
        optimizer = LOSOOptimizer(self.config)
        optimal_weights, fold_results = optimizer.optimize_all_bands_loso(all_subjects)
    
        # Outlier detection
        if self.config.LOSO_EXCLUDE_OUTLIERS:
            outliers = self._detect_outliers(fold_results)
        
            if outliers:
                Logger.warn(f"\n⚠️  OUTLIERS DETECTED: {outliers}")
                Logger.warn("Consider excluding these subjects in final analysis.")
    
        Logger.success(f"\n✅ PHASE 2 COMPLETED: LOSO optimal weights determined")
    
        return optimal_weights

    def _detect_outliers(self, fold_results):
        """
        Detect outlier subjects based on test metrics.
    
        Returns:
            outliers: List[str] (subject IDs)
        """
    
        outliers = []
    
        # Check alpha fold results
        alpha_results = fold_results.get('alpha', [])
    
        for result in alpha_results:
            subj_id = result['test_subject']
            metrics = result.get('test_metrics', {})
        
            if not metrics:
                continue
        
            pa_ratio = metrics.get('pa_ratio', 999)
            occ_pct = metrics.get('occipital_%', 999)
        
            is_outlier = False
            reasons = []
        
            if pa_ratio < self.config.LOSO_OUTLIER_ALPHA_PA_MIN:
                is_outlier = True
                reasons.append(f"P/A={pa_ratio:.2f}<{self.config.LOSO_OUTLIER_ALPHA_PA_MIN}")
        
            if occ_pct < self.config.LOSO_OUTLIER_ALPHA_OCC_MIN:
                is_outlier = True
                reasons.append(f"Occ={occ_pct:.1f}%<{self.config.LOSO_OUTLIER_ALPHA_OCC_MIN}%")
        
            if is_outlier:
                outliers.append(subj_id)
            Logger.warn(f"  ⚠️ {subj_id}: OUTLIER - {', '.join(reasons)}")
    
        return outliers

    def _phase2_group_kfold(self, subject_data):
        """
        PHASE 2: Group K-fold optimization (v3.3 - backward compatibility).
        """
    
        Logger.section("="*70)
        Logger.section("PHASE 2: GROUP K-FOLD OPTIMIZATION")
        Logger.section("="*70)
    
        # Pool data (mevcut kod aynı kalacak)
        Logger.info("Pooling data from all subjects...")
    
        mi_pooled = {}
        dice_pooled = {}
        coords_list = []
    
        for band_name in self.config.BANDS.keys():
            mi_list = []
            dice_list = []
        
            for subject_id, mi_dict, dice_dict, voxel_coords in subject_data:
                mi_list.append(mi_dict[band_name])
                dice_list.append(dice_dict[band_name])
        
            mi_pooled[band_name] = np.concatenate(mi_list, axis=0)
            dice_pooled[band_name] = np.concatenate(dice_list, axis=0)
        
            Logger.info(f"  {band_name}: {mi_pooled[band_name].shape}")
    
        for subject_id, mi_dict, dice_dict, voxel_coords in subject_data:
            coords_list.append(voxel_coords)
    
        # Run group K-fold
        optimizer = GroupKFoldOptimizer(self.config)
        optimizer.set_pooled_voxel_coords(coords_list)
    
        optimal_weights = optimizer.optimize_all_bands(mi_pooled, dice_pooled)
    
        Logger.success(f"\n✅ PHASE 2 COMPLETED: Group K-fold weights determined")
    
        del mi_pooled, dice_pooled
        gc.collect()
    
        return optimal_weights
    
    def _phase3_build_volumes(self, subject_data, optimal_weights):
        """
        PHASE 3: Build volumes for all subjects using group optimal weights.
        
        Args:
            subject_data: List of (subject_id, mi_dict, dice_dict, voxel_coords)
            optimal_weights: {band_name: float}
        """
        Logger.section("="*70)
        Logger.section("PHASE 3: BUILDING VOLUMES WITH GROUP WEIGHTS")
        Logger.section("="*70)
        
        version_tag = "v34" if self.config.USE_LOSO else "v33"
        
        for i, (subject_id, mi_dict, dice_dict, voxel_coords) in enumerate(subject_data):
            Logger.info(f"\n📍 Subject {i+1}/{len(subject_data)}: {subject_id}")
            
            # Create grid (needed for post-processing)
            grid = VoxelGrid(self.config)
            
            # Process each band
            for band_name in self.config.BANDS.keys():
                Logger.info(f"\n{'='*70}")
                Logger.info(f"BAND: {subject_id} - {band_name.upper()}")
                Logger.info(f"{'='*70}")
                
                # Get MI/Dice for this subject
                mi_scores = mi_dict[band_name]
                dice_scores = dice_dict[band_name]
                
                # Use GROUP optimal weight
                mi_weight = optimal_weights[band_name]
                dice_weight = 1.0 - mi_weight
                
                Logger.info(f"GROUP weights: MI={mi_weight:.3f}, Dice={dice_weight:.3f}")
                
                # Compute hybrid score
                hybrid = mi_weight * mi_scores + dice_weight * dice_scores
                
                # Normalize
                hmin, hmax = hybrid.min(), hybrid.max()
                if hmax > hmin:
                    hybrid = (hybrid - hmin) / (hmax - hmin)
                
                # Contrast enhancement
                hybrid = np.power(hybrid, 0.7)
                
                # Build 4D volume
                Logger.info("Building 4D volume...")
                volume_builder = VolumeBuilder(grid)
                volume = volume_builder.build(hybrid, voxel_coords)
                
                # Post-processing
                post_processor = PostProcessor(self.config, grid)
                
                if self.config.APPLY_ZSCORE:
                    volume = post_processor.zscore_normalize(volume, mode=self.config.ZSCORE_MODE)
                
                volume = post_processor.anisotropic_smooth(volume, fwhm_xyz=(3.0, 3.0, 3.0))
                volume = post_processor.temporal_consistency(volume)
                volume = post_processor.roi_balanced_sparsify(volume, band_name)
                volume = post_processor.ray_killer_cleanup(volume)
                
                # QC
                qc = QualityControl(grid)
                
                phys_metrics = qc.physiological_distribution(volume, band_name)
                
                # Save metrics
                import csv
                metrics_path = Path(self.config.DATA_PATH) / f"{subject_id}_{band_name}_phys_metrics_v33.csv"
                with open(metrics_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Value'])
                    for key, val in sorted(phys_metrics.items()):
                        writer.writerow([key, f"{val:.4f}"])
                
                Logger.success(f"Metrics saved: {metrics_path.name}")
                
                # ROI coverage
                covered, weak, missing, roi_stats = qc.roi_coverage_analysis(volume, band_name)
                qc.export_coverage_csv(roi_stats, f"{subject_id}_{band_name}", self.config.DATA_PATH)
                
                # Save NIfTI
                Logger.info("Saving NIfTI files...")
                
                nifti_img = nib.Nifti1Image(volume, affine=grid.affine)
                nifti_img.header['pixdim'][4] = self.config.SEGMENT_DURATION
                nifti_img.header.set_qform(grid.affine, code=1)
                nifti_img.header.set_sform(grid.affine, code=1)
                
                # Compressed NIfTI
                nii_path = Path(self.config.DATA_PATH) / f"{subject_id}_{band_name}_voxel_{version_tag}.nii.gz"
                nib.save(nifti_img, nii_path)
                Logger.success(f"NIfTI: {nii_path.name}")  

                # Uncompressed (SPM-compatible)
                spm_path = Path(self.config.DATA_PATH) / f"{subject_id}_{band_name}_voxel_{version_tag}_spm.nii"
                nib.save(nifti_img, spm_path)
                Logger.success(f"SPM: {spm_path.name}")

                # Metrics
                metrics_path = Path(self.config.DATA_PATH) / f"{subject_id}_{band_name}_phys_metrics_{version_tag}.csv"
                
                # Connectivity
                conn_analyzer = ConnectivityAnalyzer(self.config)
                conn_analyzer.compute(nifti_img, f"{subject_id}_{band_name}", self.config.DATA_PATH)
                
                # Stats
                active = (volume != 0)
                if active.any():
                    vals = volume[active]
                    Logger.info(f"Stats: min={vals.min():.3f}, max={vals.max():.3f}, "
                              f"mean={vals.mean():.3f}, std={vals.std():.3f}")
                
                # Cleanup
                del volume, hybrid
                gc.collect()
            
            Logger.success(f"✅ {subject_id}: All bands completed")
        
        Logger.success(f"\n✅ PHASE 3 COMPLETED: All volumes built")
    
    def run(self):
        """Execute full pipeline (Group K-fold OR LOSO)"""
    
        mode = "LOSO" if self.config.USE_LOSO else "Group K-Fold"
    
        Logger.section(f"EEG-to-fMRI Pipeline v3.4 ({mode} Analysis)")
        Logger.info(f"Data path: {self.config.DATA_PATH}")
        Logger.info(f"Subjects: {len(self.config.SUBJECTS)}")
        Logger.info(f"Mode: {mode}")
    
        if self.config.USE_LOSO:
            Logger.info(f"LOSO N_JOBS: {self.config.LOSO_N_JOBS}")
    
        # PHASE 1: Process all subjects (MI/Dice)
        subject_data = self._phase1_process_all_subjects()
    
        if len(subject_data) == 0:
            Logger.error("❌ No subjects processed successfully. Aborting.")
            return 1
    
        # PHASE 2: Optimization (Group K-fold OR LOSO)
        optimal_weights = self._phase2_optimization(subject_data)
    
        # PHASE 3: Build volumes with optimal weights
        self._phase3_build_volumes(subject_data, optimal_weights)
    
        # Summary
        duration = (datetime.now() - self.start_time).total_seconds()
    
        Logger.section("="*70)
        Logger.section("PIPELINE COMPLETED")
        Logger.section("="*70)
        Logger.success(f"Total time: {int(duration//3600)}h {int((duration%3600)//60)}m {int(duration%60)}s")
        Logger.info(f"Mode: {mode}")
        Logger.info(f"Subjects processed: {len(subject_data)}/{len(self.config.SUBJECTS)}")
        Logger.info(f"Output directory: {self.config.DATA_PATH}")
    
        weights_file = "loso_optimal_weights_v34.json" if self.config.USE_LOSO else "group_optimal_weights_v33.json"
        weights_path = Path(self.config.DATA_PATH) / weights_file
        if weights_path.exists():
            Logger.info(f"Optimal weights: {weights_file}")
    
        Logger.info("\n📊 FINAL OPTIMAL WEIGHTS:")
        for band, weight in optimal_weights.items():
            Logger.info(f"  {band:8s}: MI={weight:.2f}, Dice={1-weight:.2f}")
    
        # Outlier report (LOSO only)
        if self.config.USE_LOSO and self.config.LOSO_EXCLUDE_OUTLIERS:
            outlier_file = Path(self.config.DATA_PATH) / "loso_outliers_v34.txt"
            if outlier_file.exists():
                Logger.info(f"\n⚠️  Outlier report: {outlier_file.name}")
    
        Logger.info("\n📁 OUTPUT FILES PER SUBJECT:")
        Logger.info("  - {subject}_{band}_voxel_v34.nii.gz (compressed)")
        Logger.info("  - {subject}_{band}_voxel_v34_spm.nii (SPM-compatible)")
        Logger.info("  - {subject}_{band}_phys_metrics_v34.csv")
        Logger.info("  - {subject}_{band}_roi_coverage.csv")
        Logger.info("  - {subject}_{band}_voxel_conn.mat")
        Logger.info("  - {subject}_{band}_voxel_connectivity.mat")
    
        return 0

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    # Initialize configuration
    config = Config()
    
    # Validate environment
    if not Path(config.DATA_PATH).exists():
        Logger.error(f"Data path does not exist: {config.DATA_PATH}")
        return 1
    
    # Check subject files
    missing_files = []
    for subject_file in config.SUBJECTS:
        eeg_path = Path(config.DATA_PATH) / subject_file
        if not eeg_path.exists():
            missing_files.append(subject_file)
    
    if missing_files:
        Logger.warn(f"⚠️  Missing files ({len(missing_files)}):")
        for f in missing_files:
            Logger.warn(f"  - {f}")
        
        Logger.info(f"\nContinuing with {len(config.SUBJECTS) - len(missing_files)} available subjects...")
        
        # Remove missing files from config
        config.SUBJECTS = [f for f in config.SUBJECTS if f not in missing_files]
        
        if len(config.SUBJECTS) == 0:
            Logger.error("❌ No valid subject files found. Aborting.")
            return 1
    
    try:
        # Run pipeline
        pipeline = GroupEEGtoFMRIPipeline(config)
        return pipeline.run()
    
    except KeyboardInterrupt:
        Logger.warn("\n⚠️  Pipeline interrupted by user")
        return 130
    
    except Exception as e:
        Logger.error(f"Pipeline failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())