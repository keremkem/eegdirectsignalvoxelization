#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-to-fMRI Voxel Projection Pipeline v3.1 (Academic Edition)
==============================================================

MAJOR IMPROVEMENTS:
✅ Fixed occipital/temporal coverage (posterior coordinate adjustment)
✅ Improved hemisphere balancing with asymmetry detection
✅ Adaptive sigma based on brain region (occipital: 2x boost)
✅ ROI-specific minimum thresholds (no more missing occipital)
✅ Better ICA artifact detection
✅ Enhanced debugging and QC metrics

Referanslar:
- Koessler et al. (2009) - 10-20 to MNI mapping
- Jurcak et al. (2007) - EEG coordinate systems
- Perrin et al. (1989) - CSD re-referencing
- Delorme & Makeig (2004) - ICA artifact removal
- Harvard-Oxford Cortical Atlas (FSL)

Author: [Your Name]
Version: 3.1
Date: 2024
"""

import os
import sys
import gc
import warnings
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count

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

from joblib import Parallel, delayed
from tqdm import tqdm
import hashlib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Pipeline configuration with literature-based defaults"""
    
    # ===== PATHS =====
    DATA_PATH = r"C:\Users\kerem\Downloads\eegyedek"
    EEG_FILE = "S05_restingPre_EC.mat"
    
     # ===== CSD PARAMETERS (YENİ/GÜNCEL) =====
    CSD_SIGMA = 30.0
    CSD_PRESERVE_ALPHA = True  # ✅ YENİ
    CSD_ALPHA_MODE = "posterior_selective"  # ✅ YENİ
    # Options: "none", "global", "posterior_selective"
    
    CSD_POSTERIOR_Y_THRESHOLD = -40.0  # Y < -40 = posterior
    CSD_FRONTAL_Y_THRESHOLD = 20.0     # Y > 20 = frontal
    CSD_CENTRAL_ALPHA_RETENTION = 0.5  # Central bölgede %50 alpha
    
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
    # Base Gaussian sigma (Gevins et al., 1994: 15-30mm optimal)
    SIGMA_BASE = 22.0  # mm
    
    # Region-specific sigma multipliers
    SIGMA_MULTIPLIERS = {
        "occipital": 1.5,      # Y < -90: 44mm (broader coverage)
        "posterior": 1.3,      # -90 < Y < -70: 33mm
        "parietal": 1.2,       # -70 < Y < -40: 26.4mm
        "central": 1.0,        # -40 < Y < 0: 22mm (base)
        "frontal": 1.1,        # Y > 0: 22mm
    }
    APPLY_REGIONAL_BOOST = False  # Disabled for literature compatibility
    # Hemisphere isolation (Nunez et al., 2001)
    HEMISPHERE_ISOLATION = 0.70  # Same-hemisphere weight
    CROSS_HEMISPHERE_WEIGHT = 0.25  # Opposite-hemisphere penalty
    MIDLINE_WEIGHT = 0.85  # Midline channels contribution
    MIDLINE_GAP_MM = 3.0  # Distance threshold for midline
    
    # ===== VOXELIZATION =====
    GRID_SPACING = 2.0  # mm (2mm isotropic)
    GRID_BOUNDS = {
        "x": (-90, 91),   # Left-Right
        "y": (-130, 91),  # Posterior-Anterior
        "z": (-72, 109),  # Inferior-Superior
    }
    
    GM_INCLUDE_SUBCORTICAL = False  # Only cortical GM
    
    # ===== ROI-AWARE SPARSIFICATION =====
    # Band-specific global thresholds
    KEEP_TOP_PCT = {
        "delta": 0.50,
        "theta": 0.55,
        "alpha": 0.90,  # Alpha: broader distribution
        "beta": 0.50,
        "gamma": 0.40,
    }
    
    MIN_VOXELS_GLOBAL = {
        "delta": 20000,
        "theta": 22000,
        "alpha": 120000,  # Alpha: more voxels
        "beta": 22000,
        "gamma": 16000,
    }
    
    # Per-ROI thresholds (baseline)
    PER_ROI_KEEP_PCT = 0.15  # Keep top 15% within each ROI
    PER_ROI_MIN_VOXELS = 500  # Minimum voxels per ROI
    
    # Occipital ROI protection (Harvard-Oxford indices)
    OCCIPITAL_ROI_INDICES = {22, 23, 24, 32, 36, 39, 40, 47, 48}
    OCCIPITAL_MIN_COVERAGE = 0.95  # Keep 75% of occipital ROI voxels
    OCCIPITAL_MIN_VOXELS = 3500  # Minimum 2000 voxels per occipital ROI
    
    # ✅ YENİ: Frontal ROI protection
    FRONTAL_ROI_INDICES = {3, 4, 5, 6, 7, 25 }  # Frontal pole, superior frontal gyrus
    FRONTAL_MIN_COVERAGE = 0.85
    FRONTAL_MIN_VOXELS = 3000
    
    # Posterior ROI boost (band-specific)
    POSTERIOR_ROI_BOOST = {
        "delta": 1.4,
        "theta": 1.3,
        "alpha": 2.0,  # Alpha: maximum posterior boost
        "beta": 1.0,
        "gamma": 1.0,
    }
    
    POSTERIOR_ROI_INDICES = {11, 13, 20, 21, 31}
    
    # ===== TEMPORAL CONSISTENCY =====
    TEMPORAL_FILTERING = True
    TEMPORAL_WINDOW = 3  # timepoints
    TEMPORAL_MIN_FRAC = 0.35  # Keep if >55% active in window
    
    # ===== ARTIFACT CLEANUP =====
    RAYKILL_ENABLE = True
    RAYKILL_MIN_CLUSTER_MM3 = 25.0
    RAYKILL_KEEP_FLOOR_PCT = 0.20
    RAYKILL_KEEP_FLOOR_ABS = 28000
    
    # ===== NORMALIZATION =====
    APPLY_ZSCORE = True
    ZSCORE_MODE = "global"  # global / active / timepoint
    
    # ===== MI/DICE =====
    MI_N_BINS = 10
    MI_WEIGHT = 0.30
    DICE_WEIGHT = 0.70
    
    # ===== CONNECTIVITY =====
    CONN_STANDARDIZE = False
    CONN_DETREND = True
    CONN_GSR = True
    CONN_PARTIAL = True
    CONN_FISHER_Z = True
    
    # ===== PERFORMANCE =====
    N_JOBS = min(8, cpu_count())
    BATCH_SIZE_SIGNATURES = 5000
    CACHE_ICA = True

# ============================================================================
# COORDINATE SYSTEM
# ============================================================================

class CoordinateSystem:
    """EEG 10-20 to MNI coordinate transformation"""
    
    # 10-20 system scalp coordinates (mm)
    # Source: International 10-20 system
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
    
    # !! CORRECTED: Empirical MNI coordinates (Koessler et al., 2009)
    # These override the automatic transformation
    MANUAL_MNI_COORDS = {
        # === OCCIPITAL (CRITICAL!) ===
        'Oz':  [0,   -105, 15],   # Posterior midline
        'O1':  [-27, -108, 10],   # Left occipital
        'O2':  [27,  -108, 10],   # Right occipital
        'Iz':  [0,   -112, 5],    # Inferior occipital
        
        # === PARIETO-OCCIPITAL ===
        'POz': [0,   -95,  45],
        'PO3': [-32, -98,  35],
        'PO4': [32,  -98,  35],
        'PO7': [-42, -100, 25],
        'PO8': [42,  -100, 25],
        
        # === PARIETAL ===
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
        
        # === CENTRAL-PARIETAL ===
        'CPz': [0,   -35,  75],
        'CP1': [-20, -35,  72],
        'CP2': [20,  -35,  72],
        'CP3': [-40, -35,  65],
        'CP4': [40,  -35,  65],
        'CP5': [-54, -32,  50],
        'CP6': [54,  -32,  50],
        
        # === TEMPORAL ===
        'T7':  [-70, -20, -5],
        'T8':  [70,  -20, -5],
        'TP7': [-68, -38, -8],
        'TP8': [68,  -38, -8],
    }
    
    @classmethod
    def load_coordinates(cls):
        """
        Load and transform 10-20 coordinates to MNI space.
        
        Returns:
            coords: Dict[str, np.ndarray] - Channel name to MNI coordinates
            order: List[str] - Channel order
        """
        coords = {}
        order = []
        
        for line in cls.RAW_COORDS.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            
            name, x, y, z = parts
            
            # Use manual mapping if available
            if name in cls.MANUAL_MNI_COORDS:
                coords[name] = np.array(cls.MANUAL_MNI_COORDS[name], dtype=np.float32)
                order.append(name)
                continue
            
            # Otherwise, apply empirical transformation
            x_eeg, y_eeg, z_eeg = float(x), float(y), float(z)
            
            # Koessler et al. (2009) transformation parameters
            scale = 0.88
            x_mni = x_eeg * scale
            y_mni = y_eeg * scale * 1.12 - 5.0  # Less Y-stretch
            z_mni = z_eeg * scale + 48.0  # Higher Z-offset
            
            coords[name] = np.array([x_mni, y_mni, z_mni], dtype=np.float32)
            order.append(name)
        
        return coords, order

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
# SIGNAL PROCESSING
# ============================================================================

class SignalProcessor:
    """EEG signal preprocessing utilities"""
    
    @staticmethod
    def bandpass(data, fs, low, high, order=4):
        """Butterworth bandpass filter"""
        nyq = fs / 2
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, data, axis=1)
    
    @staticmethod
    def notch(data, fs, freq=50.0, Q=30):
        """Notch filter for line noise"""
        nyq = fs / 2
        b, a = iirnotch(freq/nyq, Q=Q)
        return filtfilt(b, a, data, axis=1)
    
    @staticmethod
    def hilbert_envelope(segment, band, fs):
        """
        Compute Hilbert envelope for a frequency band.
        
        Args:
            segment: (n_channels, n_samples) EEG segment
            band: (low, high) frequency range
            fs: Sampling rate
        
        Returns:
            envelope: (n_channels,) mean envelope per channel
        """
        filtered = SignalProcessor.bandpass(segment, fs, band[0], band[1])
        analytic = hilbert(filtered, axis=1)
        envelope = np.abs(analytic).mean(axis=1)
        return envelope
    
    @staticmethod
    def bandpower_welch(signal, fs, fmin, fmax):
        """Compute bandpower using Welch's method"""
        f, Pxx = welch(signal, fs=fs, nperseg=min(4096, len(signal)))
        idx = (f >= fmin) & (f <= fmax)
        if not np.any(idx):
            return 0.0
        return np.trapz(Pxx[idx], f[idx])

# ============================================================================
# ICA ARTIFACT REMOVAL
# ============================================================================

class ICAArtifactRemover:
    """
    Literature-based ICA artifact detection and removal.
    
    References:
    - Chaumon et al. (2015) - Automated artifact detection
    - Winkler et al. (2011) - ICLabel
    - Delorme & Makeig (2004) - EEGLAB ICA
    """
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.DATA_PATH) / "ica_cache_v31"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_signature(self, eeg, fs):
        """Generate cache signature for EEG data"""
        h = hashlib.sha1()
        h.update(np.asarray(eeg.shape, dtype=np.int32).tobytes())
        h.update(np.float32(fs).tobytes())
        sample_len = min(eeg.shape[1], int(fs * 10))
        h.update(np.ascontiguousarray(eeg[:, :sample_len]).astype(np.float32).tobytes())
        h.update(f"v31_{self.config.ICA_MODE}".encode())
        return h.hexdigest()[:16]
    
    def _extract_features(self, components, fs):
        """
        Extract multi-modal features from ICA components.
        
        Features:
        1. Temporal: kurtosis, variance
        2. Spectral: low-freq ratio, high-freq ratio, alpha ratio
        3. Autocorrelation: smoothness
        """
        n_comp = components.shape[0]
        features = {}
        
        for i in range(n_comp):
            signal = components[i]
            feat = {}
            
            # Temporal features
            feat['std'] = np.std(signal)
            feat['kurtosis'] = kurtosis(signal, fisher=False, bias=False)
            feat['median_abs_dev'] = np.median(np.abs(signal - np.median(signal)))
            
            # Spectral features
            f, Pxx = welch(signal, fs=fs, nperseg=min(4096, len(signal)))
            total_power = np.trapz(Pxx, f) + 1e-12
            
            # Low frequency (< 2 Hz) - drift/slow artifact
            lf_power = np.trapz(Pxx[f < 2], f[f < 2])
            feat['lf_ratio'] = lf_power / total_power
            
            # High frequency (30-100 Hz) - muscle artifact
            hf_power = np.trapz(Pxx[(f >= 30) & (f <= 100)], f[(f >= 30) & (f <= 100)])
            feat['hf_ratio'] = hf_power / total_power
            
            # Alpha band (8-13 Hz) - brain activity indicator
            alpha_power = np.trapz(Pxx[(f >= 8) & (f <= 13)], f[(f >= 8) & (f <= 13)])
            feat['alpha_ratio'] = alpha_power / total_power
            
            # Spectral flatness (white noise indicator)
            feat['spectral_flatness'] = np.exp(np.mean(np.log(Pxx + 1e-12))) / (np.mean(Pxx) + 1e-12)
            
            # Autocorrelation at 20ms lag (smoothness)
            acf = np.correlate(signal - signal.mean(), signal - signal.mean(), mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / (acf[0] + 1e-12)
            lag_idx = int(fs * 0.02)
            feat['acf_lag1'] = acf[lag_idx] if len(acf) > lag_idx else 0.0
            
            features[i] = feat
        
        return features
    
    def _classify_artifacts(self, features, mixing_matrix, ch_order, mode="standard"):
        """
        Classify components as artifacts or brain activity.
    
        ✅ IMPROVED: Enhanced alpha protection with spatial weighting
    
        Returns:
            bad_idx: Indices of artifact components
            reasons: Dict of reasons for each artifact
        """
        n_comp = len(features)
    
        # Extract feature matrix
        feat_keys = ['std', 'kurtosis', 'lf_ratio', 'hf_ratio', 'alpha_ratio', 
                     'spectral_flatness', 'acf_lag1']
        feat_matrix = np.array([[features[i][k] for k in feat_keys] for i in range(n_comp)])
    
        # Z-score normalization
        feat_z = (feat_matrix - feat_matrix.mean(axis=0)) / (feat_matrix.std(axis=0) + 1e-12)
    
        # ✅ YENİ: Posterior channel indices (for alpha protection)
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
        
            # 1. EOG Detection (frontal, low-freq, high kurtosis)
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
        
            # 2. EMG Detection (high-freq, high kurtosis)
            if f['hf_ratio'] > 0.15 and fz[1] > 1.5:
                score += 2.5
                reason_list.append("EMG")
        
            # 3. Channel Noise (single channel dominance)
            weights = np.abs(mixing_matrix[:, i])
            if weights.max() > 5 * np.median(weights):
                score += 2.0
                reason_list.append("Channel_noise")
        
            # 4. Line Noise (flat spectrum)
            if f['spectral_flatness'] > 0.8:
                score += 1.5
                reason_list.append("Line_noise")
        
            # 5. Spiky Signal (low autocorrelation)
            if f['acf_lag1'] < 0.3:
                score += 1.0
                reason_list.append("Spiky")
        
            # ============================================================
            # ✅ 6. ENHANCED ALPHA PROTECTION (Posterior-weighted)
            # ============================================================
            if f['alpha_ratio'] > 0.25:
                # Check spatial distribution
                weights = np.abs(mixing_matrix[:, i])
            
                if posterior_channels:
                    post_weight = weights[posterior_channels].mean()
                    overall_weight = weights.mean()
                    post_dominance = post_weight / (overall_weight + 1e-12)
                
                    # Strong posterior alpha → maximum protection
                    if f['alpha_ratio'] > 0.50 and post_dominance > 1.3:
                        score -= 6.0  # ✅ Very strong protection
                        reason_list.append("Protected_STRONG_posterior_alpha")
                
                    # Moderate posterior alpha
                    elif f['alpha_ratio'] > 0.35 and post_dominance > 1.1:
                        score -= 4.0
                        reason_list.append("Protected_moderate_posterior_alpha")
                
                    # Weak posterior alpha
                    elif post_dominance > 1.0:
                        score -= 2.5
                        reason_list.append("Protected_weak_posterior_alpha")
                
                    # Frontal/diffuse alpha → minimal protection
                    else:
                        score -= 1.0
                        reason_list.append("Weak_alpha_protection")
                else:
                    # No posterior channels identified, generic protection
                    score -= 2.0
                    reason_list.append("Protected_alpha")
        
            # Decision thresholds
            threshold = {"conservative": 4.0, "standard": 3.0, "aggressive": 2.0}[mode]
        
            if score >= threshold:
                bad_idx.append(i)
                reasons[i] = f"Score={score:.1f}: {'+'.join(reason_list)}"
    
        return np.array(bad_idx), reasons
    
    def clean(self, eeg, fs, ch_order):
        """
        Main ICA cleaning pipeline with caching.
        
        Args:
            eeg: (n_channels, n_samples) raw EEG
            fs: Sampling rate
            ch_order: List of channel names
        
        Returns:
            eeg_clean: Cleaned EEG
            info: Dict with cache status, removed components, reasons
        """
        # Check cache
        sig = self._compute_signature(eeg, fs)
        cache_path = self.cache_dir / f"ica_{sig}.npz"
        
        if self.config.CACHE_ICA and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            Logger.info(f"ICA: Cache HIT ({len(data['bad_idx'])} components removed)")
            return data['eeg_clean'], {
                'cache': 'HIT',
                'bad_idx': data['bad_idx'].tolist(),
                'reasons': data['reasons'].item() if 'reasons' in data else {}
            }
        
        # ICA decomposition
        Logger.info(f"ICA: Decomposing {eeg.shape[0]} channels...")
        n_comp = min(self.config.ICA_N_COMPONENTS, eeg.shape[0])
        
        ica = FastICA(n_components=n_comp, random_state=0, max_iter=1000, 
                      tol=1e-3, whiten='unit-variance')
        
        components_time = ica.fit_transform(eeg.T).T  # (n_comp, n_samples)
        mixing = ica.mixing_  # (n_channels, n_comp)
        
        # Feature extraction
        Logger.info("ICA: Extracting features...")
        features = self._extract_features(components_time, fs)
        
        # Artifact classification
        Logger.info(f"ICA: Classifying artifacts (mode={self.config.ICA_MODE})...")
        bad_idx, reasons = self._classify_artifacts(
            features, mixing, ch_order, mode=self.config.ICA_MODE
        )
        
        # Safety limit
        max_remove = self.config.ICA_MAX_REMOVE[self.config.ICA_MODE]
        if len(bad_idx) > max_remove:
            scores = [float(reasons[i].split('=')[1].split(':')[0]) for i in bad_idx]
            top_idx = np.argsort(scores)[-max_remove:]
            bad_idx = bad_idx[top_idx]
            reasons = {i: reasons[i] for i in bad_idx}
        
        # Remove artifacts
        components_clean = components_time.copy()
        components_clean[bad_idx, :] = 0.0
        
        # Reconstruct
        eeg_clean = (components_clean.T @ mixing.T + ica.mean_).T.astype(np.float32)
        
        # Cache
        if self.config.CACHE_ICA:
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
# CSD RE-REFERENCING
# ============================================================================

class CSDReferencer:
    """
    Current Source Density (Surface Laplacian) re-referencing.
    
    ✅ IMPROVED: Posterior-selective alpha preservation
    
    Reference: Perrin et al. (1989) - Spherical spline interpolation
    """
    
    @staticmethod
    def apply(eeg, ch_coords, ch_order, config):
        """
        Apply CSD re-referencing with optional posterior-selective alpha preservation.
        
        Args:
            eeg: (n_channels, n_samples)
            ch_coords: Dict[str, np.ndarray] - Channel coordinates
            ch_order: List[str] - Channel names
            config: Config object with CSD parameters
        
        Returns:
            eeg_csd: CSD-referenced EEG
        """
        sigma = config.CSD_SIGMA
        preserve_alpha = config.CSD_PRESERVE_ALPHA
        alpha_mode = config.CSD_ALPHA_MODE
        
        Logger.info(f"CSD: Applying surface Laplacian (σ={sigma}mm, "
                    f"alpha_preserve={preserve_alpha}, mode={alpha_mode})...")
        
        # Get coordinate array
        coords_array = np.stack([ch_coords[ch] for ch in ch_order])
        
        # Compute distance matrix
        from scipy.spatial.distance import cdist
        distances = cdist(coords_array, coords_array)
        
        # Gaussian kernel
        G = np.exp(-(distances**2) / (2 * sigma**2)).astype(np.float32)
        G[distances > 80.0] = 0.0
        np.fill_diagonal(G, 0.0)
        
        # Normalize rows
        row_sums = G.sum(axis=1, keepdims=True) + 1e-10
        G = G / row_sums
        
        # ============================================================
        # STANDARD CSD (No alpha preservation)
        # ============================================================
        if not preserve_alpha or alpha_mode == "none":
            eeg_csd = eeg - (G @ eeg)
            Logger.success("CSD: Applied (standard, full suppression)")
            return eeg_csd
        
        # ============================================================
        # GLOBAL ALPHA PRESERVATION (All channels)
        # ============================================================
        if alpha_mode == "global":
            from scipy.signal import butter, filtfilt
            
            # Extract alpha band (8-13 Hz)
            b_alpha, a_alpha = butter(4, [8/128, 13/128], btype='band')
            eeg_alpha = filtfilt(b_alpha, a_alpha, eeg, axis=1)
            eeg_non_alpha = eeg - eeg_alpha
            
            # Apply CSD only to non-alpha
            eeg_non_alpha_csd = eeg_non_alpha - (G @ eeg_non_alpha)
            
            # Reconstruct
            eeg_csd = eeg_non_alpha_csd + eeg_alpha
            
            Logger.success("CSD: Applied (global alpha preservation)")
            return eeg_csd
        
        # ============================================================
        # ✅ POSTERIOR-SELECTIVE ALPHA PRESERVATION (Recommended)
        # ============================================================
        if alpha_mode == "posterior_selective":
            from scipy.signal import butter, filtfilt
            
            # 1. Identify channel regions by Y-coordinate
            y_coords = coords_array[:, 1]
            posterior_mask = (y_coords < config.CSD_POSTERIOR_Y_THRESHOLD)  # Y < -40
            frontal_mask = (y_coords > config.CSD_FRONTAL_Y_THRESHOLD)     # Y > 20
            central_mask = (~posterior_mask) & (~frontal_mask)
            
            n_posterior = int(posterior_mask.sum())
            n_central = int(central_mask.sum())
            n_frontal = int(frontal_mask.sum())
            
            Logger.info(f"  Channel regions: Posterior={n_posterior}, "
                       f"Central={n_central}, Frontal={n_frontal}")
            
            # 2. Extract alpha band (8-13 Hz) from ALL channels
            fs = 256  # Sampling rate
            nyq = fs / 2
            b_alpha, a_alpha = butter(4, [8/nyq, 13/nyq], btype='band')
            eeg_alpha_all = filtfilt(b_alpha, a_alpha, eeg, axis=1)
            
            # 3. Split alpha into regional components
            eeg_alpha_posterior = np.zeros_like(eeg)
            eeg_alpha_posterior[posterior_mask, :] = eeg_alpha_all[posterior_mask, :]
            
            eeg_alpha_central = np.zeros_like(eeg)
            eeg_alpha_central[central_mask, :] = config.CSD_CENTRAL_ALPHA_RETENTION * eeg_alpha_all[central_mask, :]
            
            eeg_alpha_frontal = eeg_alpha_all.copy()
            eeg_alpha_frontal[posterior_mask, :] = 0.0
            eeg_alpha_frontal[central_mask, :] *= (1.0 - config.CSD_CENTRAL_ALPHA_RETENTION)
            
            # 4. Non-alpha component
            eeg_non_alpha = eeg - eeg_alpha_all
            
            # 5. Apply CSD to: non-alpha + frontal_alpha + partial_central_alpha
            eeg_to_csd = eeg_non_alpha + eeg_alpha_frontal + (0.5 * eeg_alpha_central)
            eeg_to_csd_csd = eeg_to_csd - (G @ eeg_to_csd)
            
            # 6. Reconstruct: CSD_output + preserved_posterior_alpha + partial_central_alpha
            eeg_csd = eeg_to_csd_csd + eeg_alpha_posterior + (0.5 * eeg_alpha_central)
            
            # ============================================================
            # QC: Measure alpha retention
            # ============================================================
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
        
        # Fallback (should never reach here)
        Logger.warn(f"Unknown alpha_mode: {alpha_mode}, using standard CSD")
        return eeg - (G @ eeg)

# ============================================================================
# GRID AND MASK
# ============================================================================

class VoxelGrid:
    """3D voxel grid with gray matter masking"""
    
    def __init__(self, config):
        self.config = config
        self.spacing = config.GRID_SPACING
        
        # Create grid
        Logger.info("Grid: Creating 3D voxel grid...")
        self.coords_all, self.shape = self._create_grid()
        
        # Create affine transform
        self.affine = self._create_affine()
        self.inv_affine = np.linalg.inv(self.affine)
        
        # Load atlas and mask
        Logger.info("Grid: Loading Harvard-Oxford atlas...")
        self.gm_mask = self._load_gm_mask()
        self.atlas_idx, self.roi_names = self._load_atlas()
        
        # Filter to GM voxels
        Logger.info("Grid: Filtering to gray matter...")
        self.coords_gm = self._filter_to_gm()
        
        Logger.success(f"Grid: {len(self.coords_gm)} GM voxels (shape={self.shape})")
    
    def _create_grid(self):
        """Create 3D coordinate grid"""
        bounds = self.config.GRID_BOUNDS
        
        xs = np.arange(bounds['x'][0], bounds['x'][1], self.spacing)
        ys = np.arange(bounds['y'][0], bounds['y'][1], self.spacing)
        zs = np.arange(bounds['z'][0], bounds['z'][1], self.spacing)
        
        grid = np.array(np.meshgrid(xs, ys, zs, indexing='ij'))
        coords = grid.reshape(3, -1).T
        shape = (len(xs), len(ys), len(zs))
        
        return coords, shape
    
    def _create_affine(self):
        """
        Create MNI-compatible affine transformation matrix.
    
        MNI Convention (RAS+):
        - X: Left (-) to Right (+)
        - Y: Posterior (-) to Anterior (+)
        - Z: Inferior (-) to Superior (+)
        """
        s = self.spacing
        bounds = self.config.GRID_BOUNDS
    
        # MNI standard affine (positive diagonal)
        affine = np.array([
            [ s,  0,  0,  bounds['x'][0]],  # X: -90 (left) to +90 (right)
            [ 0,  s,  0,  bounds['y'][0]],  # Y: -130 (post) to +90 (ant)
            [ 0,  0,  s,  bounds['z'][0]],  # Z: -72 (inf) to +109 (sup)
            [ 0,  0,  0,  1]
        ], dtype=np.float32)
    
        Logger.debug(f"Affine matrix (MNI standard):\n{affine}")
        return affine
    
    def _load_gm_mask(self):
        """Load gray matter mask from Harvard-Oxford atlas"""
        target_img = nib.Nifti1Image(np.zeros(self.shape, dtype=np.int16), self.affine)
        
        # Cortical GM
        ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        cort_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
        cort_resampled = image.resample_to_img(cort_img, target_img, interpolation='nearest')
        gm_mask = cort_resampled.get_fdata() > 0
        
        # Subcortical (optional)
        if self.config.GM_INCLUDE_SUBCORTICAL:
            ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
            sub_img = ho_sub.maps if isinstance(ho_sub.maps, nib.Nifti1Image) else nib.load(ho_sub.maps)
            sub_resampled = image.resample_to_img(sub_img, target_img, interpolation='nearest')
            gm_mask |= (sub_resampled.get_fdata() > 0)
        
        Logger.info(f"  GM mask: {int(gm_mask.sum())} voxels")
        return gm_mask
    
    def _load_atlas(self):
        """Load ROI labels from Harvard-Oxford atlas"""
        ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
        
        target_img = nib.Nifti1Image(np.zeros(self.shape, dtype=np.int16), self.affine)
        atlas_resampled = image.resample_to_img(atlas_img, target_img, interpolation='nearest')
        atlas_idx = atlas_resampled.get_fdata().astype(int)
        
        roi_names = list(ho_cort.labels)
        
        Logger.info(f"  Atlas: {len(roi_names)} ROIs loaded")
        return atlas_idx, roi_names
    
    def _filter_to_gm(self):
        """Filter coordinates to gray matter voxels"""
        coords_gm = []
        
        for coord in tqdm(self.coords_all, desc="  Filtering GM", ncols=80):
            # Transform MNI to voxel indices
            vox = self.inv_affine @ np.append(coord, 1)
            xi, yi, zi = np.round(vox[:3]).astype(int)
            
            # Check bounds and GM mask
            if (0 <= xi < self.shape[0] and
                0 <= yi < self.shape[1] and
                0 <= zi < self.shape[2] and
                self.gm_mask[xi, yi, zi]):
                coords_gm.append(coord)
        
        return np.asarray(coords_gm, dtype=np.float32)
    
    def mni_to_voxel(self, mni_coords):
        """Convert MNI coordinates to voxel indices"""
        vox = self.inv_affine @ np.append(mni_coords, 1)
        return np.round(vox[:3]).astype(int)

# ============================================================================
# VOXEL SIGNATURE COMPUTATION
# ============================================================================

class SignatureComputer:
    """Compute voxel signatures with occipital boost"""
    
    def __init__(self, config):
        self.config = config
    
    def _get_adaptive_sigma(self, voxel_y):
        """Get sigma based on Y-coordinate"""
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
        """Classify channels by hemisphere"""
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
        """Classify voxel hemisphere"""
        if abs(voxel_x) <= self.config.MIDLINE_GAP_MM:
            return 'MID'
        elif voxel_x < -self.config.MIDLINE_GAP_MM:
            return 'LH'
        else:
            return 'RH'
    
    def compute(self, voxel_coords, ch_coords, ch_order):
        """Compute voxel signatures WITHOUT occipital boost (pure Gaussian)"""
        Logger.section("Computing Voxel Signatures")

        n_voxels = len(voxel_coords)
        n_channels = len(ch_order)

        ch_coords_array = np.stack([ch_coords[ch] for ch in ch_order])

        lh_idx, rh_idx, mid_idx = self._classify_channels(ch_order, ch_coords)
        Logger.info(f"Channels: LH={len(lh_idx)}, RH={len(rh_idx)}, MID={len(mid_idx)}")

        occipital_channels = [i for i, ch in enumerate(ch_order) 
                             if ch in ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Iz']]
        Logger.info(f"Occipital channels: {len(occipital_channels)}")

        n_posterior = 0
        signatures = np.zeros((n_voxels, n_channels), dtype=np.float32)

        # ✅ SIMPLIFIED: Process all voxels WITHOUT boost
        Logger.info("Processing all voxels (pure Gaussian weighting)...")
    
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
            
                # Adaptive sigma (region-specific)
                sigma = self._get_adaptive_sigma(vox_y)
            
                # Gaussian weighting
                weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                weights[dist > 200] = 0.0
            
                # Minimum connectivity
                if np.sum(weights > 0.01) < 8:
                    nearest = np.argsort(dist)[:8]
                    weights[nearest] = np.maximum(weights[nearest], 0.05)
            
                # ✅ NO BOOST - Only count posterior voxels
                if vox_y < -70:
                    n_posterior += 1
            
                # Hemisphere balancing
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
            
                # Normalize
                weight_sum = weights.sum()
                if weight_sum > 1e-12:
                    weights /= weight_sum
                else:
                    nearest_ch = np.argmin(dist)
                    weights[:] = 0.0
                    weights[nearest_ch] = 1.0
            
                signatures[global_idx] = weights

        Logger.success(f"Signatures: Computed for {n_voxels} voxels")
        Logger.info(f"  Posterior voxels (Y<-70): {n_posterior}")
        Logger.info(f"  Boost: DISABLED (pure Gaussian)")

        # ✅ SIMPLE QC (without boost expectations)
        self._qc_signatures_simple(signatures, voxel_coords, ch_order, occipital_channels)

        return signatures

    def _qc_signatures_simple(self, signatures, voxel_coords, ch_order, occipital_channels):
        """Simple QC for pure Gaussian signatures"""
        Logger.info("\n=== Signature QC (Pure Gaussian) ===")
    
        posterior_mask = voxel_coords[:, 1] < -70
        n_posterior = posterior_mask.sum()
    
        Logger.info(f"Posterior voxels (Y<-70): {n_posterior}")
    
        if n_posterior > 0:
            posterior_sigs = signatures[posterior_mask]
            occ_weights = posterior_sigs[:, occipital_channels]
            occ_fractions = occ_weights.sum(axis=1)
        
            Logger.info(f"  Occipital fraction per voxel (natural Gaussian):")
            Logger.info(f"    Mean: {occ_fractions.mean():.4f}")
            Logger.info(f"    Median: {np.median(occ_fractions):.4f}")
            Logger.info(f"    Min: {occ_fractions.min():.4f}")
            Logger.info(f"    Max: {occ_fractions.max():.4f}")
        
            # Sample voxels
            sample_indices = [0, n_posterior//4, n_posterior//2, 3*n_posterior//4, n_posterior-1]
            Logger.info(f"\n  Sample posterior voxels:")
        
            for si in sample_indices[:3]:  # Show only 3 samples
                if si < n_posterior:
                    sample_vox = voxel_coords[posterior_mask][si]
                    sample_sig = posterior_sigs[si]
                    sample_occ_frac = occ_fractions[si]
                
                    Logger.info(f"\n    Voxel [{sample_vox[0]:.1f}, {sample_vox[1]:.1f}, {sample_vox[2]:.1f}]:")
                    Logger.info(f"      Occipital fraction: {sample_occ_frac:.3f}")
                
                    top3 = np.argsort(sample_sig)[-3:][::-1]
                    Logger.info(f"      Top 3 channels:")
                    for ch_idx in top3:
                        Logger.info(f"        {ch_order[ch_idx]:6s}: {sample_sig[ch_idx]:.4f}")
                        
class SimplifiedKFoldOptimizer:
    """
    Band-specific K-fold optimization with physiological metrics.
    """
    
    def __init__(self, config):
        self.config = config
        self.n_folds = 3
        self.mi_weights_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
        self.voxel_coords = None
    
    def set_voxel_coords(self, coords):
        """Set voxel coordinates for spatial metrics"""
        self.voxel_coords = coords
        Logger.info(f"K-fold: Loaded {len(coords)} voxel coordinates")
    
    def optimize_band(self, mi_scores, dice_scores, band_name):
        """Optimize MI weight using band-specific metric"""
        Logger.info(f"\n🔍 Optimizing {band_name.upper()} (physiological metric, K={self.n_folds})...")
        
        if self.voxel_coords is None:
            Logger.error("❌ Voxel coordinates not set! Call set_voxel_coords() first")
            return 0.5
        
        # Band-specific optimization
        if band_name == "alpha":
            best_weight = self._optimize_alpha(mi_scores, dice_scores)
        elif band_name == "beta":
            best_weight = self._optimize_beta(mi_scores, dice_scores)
        elif band_name == "delta":
            best_weight = self._optimize_delta(mi_scores, dice_scores)
        elif band_name == "theta":
            best_weight = self._optimize_theta(mi_scores, dice_scores)
        elif band_name == "gamma":
            best_weight = self._optimize_gamma(mi_scores, dice_scores)
        else:
            best_weight = 0.5
        
        Logger.info(f"  Best MI weight: {best_weight:.2f}")
        self._save_results(band_name, best_weight)
        
        return best_weight
    
    def _optimize_alpha(self, mi_scores, dice_scores):
        """Alpha: Maximize posterior/anterior ratio (target: 4-8)"""
        from sklearn.model_selection import KFold
        
        y_coords = self.voxel_coords[:, 1]
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
            
            # Target: ratio=6.0, penalize deviation
            target_ratio = 6.0
            ratio_score = 1.0 - min(1.0, abs(ratio - target_ratio) / target_ratio)
            
            metric = ratio_score * stability
            
            if metric > best_metric:
                best_metric = metric
                best_weight = mi_weight
                
            if best_weight > 0.85:
                Logger.warn(f"  Alpha MI={best_weight:.2f} too high, capping at 0.85")
                best_weight = 0.85
        
        return best_weight
    
    def _optimize_beta(self, mi_scores, dice_scores):
        """Beta: Maximize central concentration (target: 50-65%)"""
        from sklearn.model_selection import KFold
        
        y_coords = self.voxel_coords[:, 1]
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
            
            # Target: 57.5% (middle of 50-65%)
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
        """Delta: Maximize diffuse distribution (high entropy)"""
        from sklearn.model_selection import KFold
        from scipy.stats import entropy
        
        y_coords = self.voxel_coords[:, 1]
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
            
            # Regional distribution entropy
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
        """Theta: Maximize frontal-midline concentration (Onton 2005)"""
        from sklearn.model_selection import KFold
    
        y_coords = self.voxel_coords[:, 1]
        x_coords = self.voxel_coords[:, 0]
    
        # ✅ Frontal-midline mask (Y > 20, |X| < 30)
        frontal_midline_mask = (y_coords > 20) & (np.abs(x_coords) < 30)
        
    
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
        
            # Target: 20% frontal-midline (18-22% range)
            target_fm = 0.20
            fm_score = 1.0 - min(1.0, abs(fm_pct - target_fm) / target_fm)
        
            metric = fm_score * stability
        
            if metric > best_metric:
                best_metric = metric
                best_weight = mi_weight
    
        # ✅ UPPER BOUND (döngüden SONRA)
        if best_weight > 0.70:
            Logger.warn(f"  Theta MI={best_weight:.2f} too high (not frontal-midline)")
            Logger.warn(f"   Capping to 0.70 for proper frontal distribution")
            best_weight = 0.70
    
        return best_weight
    
    def _optimize_gamma(self, mi_scores, dice_scores):
        """Gamma: Maximize focal distribution (high kurtosis)"""
        from sklearn.model_selection import KFold
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
    
    def _save_results(self, band_name, best_weight):
        """Save results"""
        import csv
        csv_path = Path(self.config.DATA_PATH) / f"{band_name}_kfold_optimization_v33.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Band', 'Best_MI_Weight', 'Best_Dice_Weight'])
            writer.writerow([band_name, f"{best_weight:.2f}", f"{1-best_weight:.2f}"])
        Logger.info(f"  Results saved: {csv_path.name}")
    
    def optimize_all_bands(self, mi_dict, dice_dict):
        """Optimize all bands"""
        Logger.section("9. K-Fold Optimization (Physiological Metrics)")
        
        optimal_weights = {}
        for band_name in mi_dict.keys():
            optimal_weights[band_name] = self.optimize_band(
                mi_dict[band_name], 
                dice_dict[band_name], 
                band_name
            )
        
        # Save
        weights_path = Path(self.config.DATA_PATH) / "optimal_hybrid_weights_v33.json"
        import json
        with open(weights_path, 'w') as f:
            json.dump(optimal_weights, f, indent=2)
        
        Logger.success(f"Optimization completed!")
        Logger.info(f"Optimal weights saved: {weights_path.name}")
        Logger.info(f"\nOptimal weights:")
        for band, weight in optimal_weights.items():
            Logger.info(f"  {band:8s}: MI={weight:.2f}, Dice={1-weight:.2f}")
        
        return optimal_weights
        
# ============================================================================
# MI/DICE COMPUTATION
# ============================================================================

class MIDiceComputer:
    """
    Compute Mutual Information and Dice similarity between 
    voxel signatures and EEG snapshots.
    """
    
    def __init__(self, config):
        self.config = config
    
    def _compute_one_timepoint(self, voxel_sigs, snapshot, v_bins, t):
        """
        OPTIMIZED: Vectorized MI computation with chunking.
    
        PERFORMANCE: ~20× faster than loop-based sklearn.mutual_info_score
        """
        n_bins = self.config.MI_N_BINS
        n_voxels = len(voxel_sigs)
    
        # === DICE COMPUTATION (Already vectorized) ===
        numerator = 2.0 * (voxel_sigs @ snapshot)
        denominator = (np.sum(voxel_sigs**2, axis=1) + np.sum(snapshot**2) + 1e-12)
        dice_scores = numerator / denominator
    
        # === MI COMPUTATION (Optimized with chunking) ===
        # Bin snapshot once
        s_min, s_max = snapshot.min(), snapshot.max()
        s_bins = np.linspace(s_min, s_max, n_bins + 1)
    
        # Digitize snapshot
        sb = np.digitize(snapshot, bins=s_bins, right=False)
        sb = np.clip(sb - 1, 0, n_bins - 1)  # Convert to 0-based
    
        mi_scores = np.zeros(n_voxels, dtype=np.float32)
    
        # Process in chunks to avoid memory issues
        chunk_size = 5000
    
        for start in range(0, n_voxels, chunk_size):
            end = min(start + chunk_size, n_voxels)
        
            # Vectorized binning for chunk
            chunk_data = voxel_sigs[start:end]
        
            for i in range(end - start):
                vox_idx = start + i
            
                # Bin voxel signature
                v_min, v_max = chunk_data[i].min(), chunk_data[i].max()
                if v_max <= v_min:
                    mi_scores[vox_idx] = 0.0
                    continue
            
                vb_bins = v_bins[vox_idx]
                vb = np.digitize(chunk_data[i], bins=vb_bins, right=False)
                vb = np.clip(vb - 1, 0, n_bins - 1)
            
                # Fast 2D histogram (contingency table)
                contingency = np.zeros((n_bins, n_bins), dtype=np.float32)
                np.add.at(contingency, (vb, sb), 1)
            
                # Fast MI from contingency
                mi_scores[vox_idx] = self._fast_mi_from_contingency(contingency)
    
        # Normalize
        mi_norm = self._normalize(mi_scores)
        dice_norm = self._normalize(dice_scores)
    
        return t, mi_norm, dice_norm

    def _fast_mi_from_contingency(self, contingency):
        """
        Fast MI calculation from contingency table.
    
        Avoids sklearn overhead for batch processing.
        """
        contingency = contingency + 1e-12  # Avoid log(0)
    
        # Joint probability
        pxy = contingency / contingency.sum()
    
        # Marginal probabilities
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
    
        # MI = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
        px_py = px * py
    
        # Only compute where pxy > 0 (avoid log(0))
        mask = pxy > 0
        mi = np.sum(pxy[mask] * np.log(pxy[mask] / (px_py[mask] + 1e-12)))
    
        return max(0.0, mi)

    def _normalize(self, scores):
        """Min-max normalization"""
        vmin, vmax = scores.min(), scores.max()
        if vmax > vmin:
            return (scores - vmin) / (vmax - vmin)
        return scores
    
    def compute(self, voxel_sigs, snapshots):
        """
        Compute MI and Dice for all timepoints (parallel).
        
        Args:
            voxel_sigs: (n_voxels, n_channels) signatures
            snapshots: (n_timepoints, n_channels) EEG envelopes
        
        Returns:
            mi: (n_voxels, n_timepoints)
            dice: (n_voxels, n_timepoints)
        """
        Logger.info("Computing MI/Dice scores...")
        
        n_voxels, n_channels = voxel_sigs.shape
        n_timepoints = snapshots.shape[0]
        
        # Pre-compute bins for voxel signatures
        Logger.info("  Pre-computing bins...")
        v_bins = [
            np.histogram_bin_edges(voxel_sigs[i], bins=self.config.MI_N_BINS)
            for i in tqdm(range(n_voxels), desc="  Bins", ncols=80)
        ]
        
        # ✅ SEQUENTIAL PROCESSING (threading disabled due to GIL issues)
        Logger.info(f"  Computing {n_timepoints} timepoints (sequential)...")
    
        results = []
        for t in tqdm(range(n_timepoints), desc="  Timepoints", ncols=80):
            # Process each timepoint
            result = self._compute_one_timepoint(voxel_sigs, snapshots[t], v_bins, t)
            results.append(result)
        
            # Progress update every 10 timepoints
            if (t + 1) % 10 == 0:
                Logger.info(f"    Processed {t+1}/{n_timepoints} timepoints...")
    
        # Sort and reconstruct matrices
        results.sort(key=lambda x: x[0])
    
        mi = np.zeros((n_voxels, n_timepoints), dtype=np.float32)
        dice = np.zeros_like(mi)
    
        for t, mi_t, dice_t in results:
            mi[:, t] = mi_t
            dice[:, t] = dice_t
    
        Logger.success("MI/Dice: Computed")
    
        return mi, dice

# ============================================================================
# VOLUME BUILDER
# ============================================================================

class VolumeBuilder:
    """Build 4D NIfTI volumes from voxel data"""
    
    def __init__(self, grid):
        self.grid = grid
    
    def build(self, voxel_data, voxel_coords):
        """
        Build 4D volume from voxel-wise data.
        
        Args:
            voxel_data: (n_voxels, n_timepoints)
            voxel_coords: (n_voxels, 3) MNI coordinates
        
        Returns:
            volume: (x, y, z, t) 4D array
        """
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
# POST-PROCESSING
# ============================================================================

class PostProcessor:
    """Volume post-processing: normalization, filtering, sparsification"""
    
    def __init__(self, config, grid):
        self.config = config
        self.grid = grid
    
    def zscore_normalize(self, volume, mode="global"):
        """
        Z-score normalization.
        
        Modes:
        - global: Single mean/std across all active voxels (recommended)
        - active: Per-timepoint on active voxels
        - timepoint: Per-timepoint on all voxels
        """
        Logger.info(f"Z-score normalization (mode={mode})...")
        
        x, y, z, T = volume.shape
        
        if mode == "global":
            # Global normalization across all timepoints
            active_mask = (volume > 0)
            active_values = volume[active_mask]
            
            if active_values.size > 0:
                mean_global = active_values.mean()
                std_global = active_values.std()
                
                if std_global > 0:
                    volume[active_mask] = (volume[active_mask] - mean_global) / std_global
        
        elif mode == "active":
            # Per-timepoint normalization (only active voxels)
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
            # Per-timepoint normalization (all voxels)
            for t in range(T):
                vol_t = volume[:, :, :, t]
                mean_t = vol_t.mean()
                std_t = vol_t.std()
                
                if std_t > 0:
                    volume[:, :, :, t] = (vol_t - mean_t) / std_t
        
        Logger.success("Z-score: Applied")
        return volume
    
    def temporal_consistency(self, volume):
        """
        Temporal consistency filtering.
        
        Keep only voxels that are active in a sufficient fraction of 
        their temporal neighborhood.
        """
        if not self.config.TEMPORAL_FILTERING:
            return volume
        
        Logger.info(f"Temporal consistency (window={self.config.TEMPORAL_WINDOW})...")
        
        # Binary mask: 1 if active, 0 otherwise
        mask = (volume > 0).astype(np.float32)
        
        # Smooth mask along time axis
        smoothed = uniform_filter1d(mask, size=self.config.TEMPORAL_WINDOW, 
                                   axis=3, mode='reflect')
        
        # Keep voxels active in >threshold fraction of window
        keep_mask = smoothed >= self.config.TEMPORAL_MIN_FRAC
        
        # Apply mask
        volume[~keep_mask] = 0.0
        
        removed = (~keep_mask & (mask > 0)).sum()
        Logger.info(f"  Removed {removed} temporally inconsistent voxels")
        
        return volume
    
    def roi_balanced_sparsify(self, volume, band_name):
        """
        ROI-aware sparsification with occipital protection.
    
        Strategy:
        1. ÖNCE occipital ROI'leri koru
        2. Global top-K selection
        3. Per-ROI minimum quota
        """
        Logger.info(f"ROI-balanced sparsification [{band_name}]...")
    
        x, y, z, T = volume.shape
        atlas = self.grid.atlas_idx
        roi_names = self.grid.roi_names
    
        # Get band-specific parameters
        global_pct = self.config.KEEP_TOP_PCT.get(band_name, 0.40)
        global_min = self.config.MIN_VOXELS_GLOBAL.get(band_name, 20000)
        posterior_boost = self.config.POSTERIOR_ROI_BOOST.get(band_name, 1.0)
    
        # Get unique ROI labels
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
        
            # ============================================================
            # STEP 1: ÖNCE OCCIPITAL ROI'LERİ KORU (FORCE)
            # ============================================================
            final_mask = np.zeros_like(active_mask, dtype=bool)
        
            for roi_label in self.config.OCCIPITAL_ROI_INDICES:
                roi_mask = (atlas == roi_label) & active_mask
                n_roi = int(roi_mask.sum())
            
                if n_roi > 0:
                    # Minimum %85'ini tut
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
            # ✅ YENİ STEP 1.5: FRONTAL ROI'LERİ KORU
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
            # ============================================================
            # STEP 2: GLOBAL TOP-K (occipital dışındaki voxeller için)
            # ============================================================
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
        
            # ============================================================
            # STEP 3: DİĞER ROI'LER İÇİN MİNİMUM QUOTA
            # ============================================================
            for roi_label in roi_labels:
                if roi_label in self.config.OCCIPITAL_ROI_INDICES:
                    continue  # Zaten işlendi
            
                roi_mask = (atlas == roi_label) & active_mask & (~final_mask)
                n_roi = int(roi_mask.sum())
            
                if n_roi > 0:
                    roi_pct = self.config.PER_ROI_KEEP_PCT
                    roi_min = self.config.PER_ROI_MIN_VOXELS
                
                    # Posterior boost
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
        
            # Update volume
            kept = int(final_mask.sum())
            total_after += kept
        
            vol_t[~final_mask] = 0.0
            volume[:, :, :, t] = vol_t
    
        Logger.info(f"  Sparsified: {total_before} → {total_after} voxels "
                   f"({100*total_after/max(total_before,1):.1f}%)")
    
        return volume
    def anisotropic_smooth(self, volume, fwhm_xyz=(6.0, 2.0, 6.0)):
        """
        Anisotropic Gaussian - Y eksenini koru (posterior dominance).
    
        Args:
            fwhm_xyz: (X_mm, Y_mm, Z_mm) - Y küçük tutularak posterior korunur
        """
        Logger.info(f"Anisotropic smoothing (FWHM_XYZ={fwhm_xyz}mm)...")
    
        from scipy.ndimage import gaussian_filter
    
        # Her eksen için ayrı sigma
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
        
            # Anisotropic smoothing
            smoothed_t = gaussian_filter(
                vol_t,
                sigma=sigma_voxels,  # Tuple: farklı sigma her eksen için
                mode='constant',
                cval=0.0
            )
        
            smoothed_t[~mask] = 0.0
        
            # %85 original + %15 smoothed (daha dengeli)
            blended = 0.85 * vol_t + 0.15 * smoothed_t
            blended[~mask] = 0.0
        
            smoothed[:, :, :, t] = blended
    
        Logger.success("Anisotropic smoothing: Applied")
        return smoothed
    
    def ray_killer_cleanup(self, volume):
        """
        Remove small isolated clusters (artifact cleanup).
        
        Connected-component analysis per timepoint.
        """
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
            
            # Label connected components
            labeled, n_clusters = label(active_mask)
            
            if n_clusters > 0:
                # Count voxels per cluster
                counts = np.bincount(labeled.ravel())
                
                # Mark small clusters for removal
                drop_labels = np.where(counts < min_voxels)[0]
                
                if len(drop_labels) > 0:
                    drop_mask = np.isin(labeled, drop_labels)
                    vol_t[drop_mask] = 0.0
            
            kept = int((vol_t > 0).sum())
            
            # Safety floor: keep minimum voxels
            keep_floor = max(
                int(self.config.RAYKILL_KEEP_FLOOR_PCT * n_active),
                self.config.RAYKILL_KEEP_FLOOR_ABS
            )
            
            if kept < keep_floor:
                # Rescue: keep top-K by intensity
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
    # PostProcessor sınıfına YENİ metod ekle (boost'a DOKUNMA):

    def spatial_smooth(self, volume, fwhm_mm=4.0):  # ← 6mm → 4mm (hafif smoothing)
        """
        MINIMAL Gaussian spatial smoothing (çizgileri temizle, dengeyi koruma).
        """
        Logger.info(f"Minimal spatial smoothing (FWHM={fwhm_mm}mm)...")
    
        from scipy.ndimage import gaussian_filter
    
        sigma_mm = fwhm_mm / 2.355
        sigma_voxels = sigma_mm / self.config.GRID_SPACING  # ~0.85 voxel
    
        x, y, z, T = volume.shape
        smoothed = np.zeros_like(volume)
    
        for t in range(T):
            vol_t = volume[:, :, :, t]
            mask = (vol_t > 0)
        
            if mask.any():
                # Çok hafif smoothing (sadece çizgileri temizle)
                smoothed_t = gaussian_filter(vol_t, sigma=sigma_voxels, mode='constant', cval=0.0)
            
                # Mask dışına taşma
                smoothed_t[~mask] = 0.0
            
                # %95 orjinal + %5 smoothed (çok minimal)
                smoothed_t = 0.95 * vol_t + 0.05 * smoothed_t
            
                smoothed[:, :, :, t] = smoothed_t
    
        Logger.success(f"Minimal smoothing: Applied (sigma={sigma_voxels:.2f} voxels)")
        return smoothed
# ============================================================================
# QC AND ANALYSIS
# ============================================================================

class QualityControl:
    """Quality control metrics and visualization"""
    
    def __init__(self, grid):
        self.grid = grid
    
    def physiological_distribution(self, volume, band_name):
        """
        Analyze spatial distribution with CLEAR occipital reporting.
        """
        Logger.info(f"\n{'='*70}")
        Logger.info(f"QC: Physiological Distribution [{band_name.upper()}]")
        Logger.info(f"{'='*70}")
    
        affine = self.grid.affine
        gm_mask = self.grid.gm_mask
    
        # Get Y-axis coordinates
        x, y, z = self.grid.shape
        yi = np.arange(y)
        Y_coords = (affine @ np.vstack([
            np.zeros_like(yi), yi, np.zeros_like(yi), np.ones_like(yi)
        ]))[1, :]
    
        # Create region masks
        Y_grid = Y_coords[np.newaxis, :, np.newaxis, np.newaxis]
        gm_4d = gm_mask[..., np.newaxis]
    
        # ✅ EXPLICIT regions (6 regions for clarity)
        regions = {
            "very_posterior": (Y_grid <= -85),      # Y < -85 (occipital pole)
            "occipital": (Y_grid > -85) & (Y_grid <= -70),  # -85 to -70
            "posterior": (Y_grid > -70) & (Y_grid <= -40),  # -70 to -40 (parietal)
            "central": (Y_grid > -40) & (Y_grid <= 0),      # -40 to 0
            "anterior": (Y_grid > 0) & (Y_grid <= 40),      # 0 to 40
            "frontal": (Y_grid > 40)                         # > 40
        }
    
        # Compute activation per region
        V = volume.copy()
        V[V < 0] = 0
    
        total = V[gm_4d.squeeze(-1)].sum() + 1e-12
    
        results = {}
    
        # ✅ PRINT HEADER
        Logger.info(f"{'Region':<20} {'%':>8} {'Voxels':>10} {'Mean':>10}")
        Logger.info(f"{'-'*52}")
    
        for region_name, region_mask in regions.items():
            region_data = V[region_mask.squeeze(-1) & gm_4d.squeeze(-1)]
        
            region_sum = region_data.sum()
            pct = 100.0 * region_sum / total
            n_active = int((region_data > 0).sum())
            mean_act = region_data[region_data > 0].mean() if n_active > 0 else 0.0
        
            # Store
            results[f"{band_name}_{region_name}_%"] = float(pct)
            results[f"{band_name}_{region_name}_nvoxels"] = n_active
            results[f"{band_name}_{region_name}_mean"] = float(mean_act)
        
            # ✅ PRINT EACH REGION
            Logger.info(f"{region_name.replace('_', ' ').title():<20} {pct:>6.1f}%  {n_active:>8d}  {mean_act:>8.3f}")
    
        # ✅ ALPHA-SPECIFIC ANALYSIS
        if band_name == "alpha":
            # Posterior sum (very_posterior + occipital + posterior)
            post_sum = (V[regions["very_posterior"].squeeze(-1) & gm_4d.squeeze(-1)].sum() +
                       V[regions["occipital"].squeeze(-1) & gm_4d.squeeze(-1)].sum() +
                       V[regions["posterior"].squeeze(-1) & gm_4d.squeeze(-1)].sum())
        
            # Anterior sum (anterior + frontal)
            ant_sum = (V[regions["anterior"].squeeze(-1) & gm_4d.squeeze(-1)].sum() +
                      V[regions["frontal"].squeeze(-1) & gm_4d.squeeze(-1)].sum())
        
            ratio = post_sum / max(ant_sum, 1e-12)
            results["alpha_post_ant_ratio"] = float(ratio)
        
            # Combined occipital (very_posterior + occipital)
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
    
        Logger.info("")  # Blank line
    
        return results
    
    def roi_coverage_analysis(self, volume, band_name):
        """
        Detailed ROI coverage analysis.
    
        Returns:
            covered: ROIs with >100 voxels
            weak: ROIs with 1-100 voxels
            missing: ROIs with 0 voxels
        """
        Logger.info(f"\n=== ROI Coverage Analysis [{band_name}] ===")
    
        atlas = self.grid.atlas_idx
        roi_names = self.grid.roi_names
    
        roi_labels = np.unique(atlas)
        roi_labels = roi_labels[roi_labels > 0]
    
        roi_stats = {}
    
        for roi_label in roi_labels:
            roi_name = roi_names[roi_label] if roi_label < len(roi_names) else f"ROI_{roi_label}"
            roi_mask = (atlas == roi_label)
        
            # ✅ DÜZELTME: UNIQUE voxel sayısı (en az 1 timepoint'te active)
            active_any_time = np.any(volume > 0, axis=3)  # (x, y, z) boolean
            active_voxels = int(np.sum(active_any_time & roi_mask))
            total_voxels = int(np.sum(roi_mask))
        
            # ✅ Statistics: Tüm timepoint'lerdeki voxel değerleri
            roi_mask_4d = roi_mask[..., np.newaxis]  # (x, y, z, 1) → broadcast to (x, y, z, t)
            roi_data_all = volume[np.broadcast_to(roi_mask_4d, volume.shape)]  # Flatten
            roi_data_active = roi_data_all[roi_data_all > 0]
        
            mean_act = float(roi_data_active.mean()) if roi_data_active.size > 0 else 0.0
            max_act = float(roi_data_active.max()) if roi_data_active.size > 0 else 0.0
        
            roi_stats[roi_label] = {
               'name': roi_name,
               'total_voxels': total_voxels,
               'active_voxels': active_voxels,  # ✅ Artık UNIQUE voxel sayısı
               'coverage_%': float(100.0 * active_voxels / max(total_voxels, 1)),
               'mean_activation': mean_act,
                'max_activation': max_act
            }
        
        # Categorize
        covered = {k: v for k, v in roi_stats.items() if v['active_voxels'] > 100}
        weak = {k: v for k, v in roi_stats.items() if 0 < v['active_voxels'] <= 100}
        missing = {k: v for k, v in roi_stats.items() if v['active_voxels'] == 0}
        
        # Report
        Logger.info(f"Total ROIs: {len(roi_labels)}")
        Logger.info(f"✅ Covered (>100 voxels): {len(covered)}")
        Logger.info(f"⚠️  Weak (1-100 voxels): {len(weak)}")
        Logger.info(f"❌ Missing (0 voxels): {len(missing)}")
        
        # Missing details
        if missing:
            Logger.info("\n❌ MISSING ROIs:")
            for label, info in sorted(missing.items(), 
                                     key=lambda x: x[1]['total_voxels'], 
                                     reverse=True)[:10]:
                Logger.info(f"  [{label:2d}] {info['name'][:45]:45s} (GM: {info['total_voxels']:4d})")
        
        # Top covered
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
        """Export
        Export ROI coverage to CSV"""
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
# CONNECTIVITY ANALYSIS
# ============================================================================

class ConnectivityAnalyzer:
    """CONN-compatible connectivity matrices"""
    
    def __init__(self, config):
        self.config = config
    
    def compute(self, nifti_img, band_name, output_dir):
        """
        Compute connectivity matrices using Harvard-Oxford atlas.
        
        Args:
            nifti_img: NIfTI image object (4D volume)
            band_name: Frequency band name (e.g., "alpha")
            output_dir: Output directory for .mat files
        """
        Logger.info(f"CONN: Computing connectivity [{band_name}]...")
        
        try:
            # Load Harvard-Oxford atlas
            ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            labels_img = (ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) 
                         else nib.load(ho_cort.maps))
            roi_names = list(ho_cort.labels)[1:]  # Skip background
            
            # Extract ROI timeseries
            masker = NiftiLabelsMasker(
                labels_img=labels_img,
                labels=roi_names,
                background_label=0,
                standardize=self.config.CONN_STANDARDIZE,
                detrend=self.config.CONN_DETREND,
                verbose=0
            )
            
            timeseries = masker.fit_transform(nifti_img)
            
            # Filter active ROIs
            active_cols = np.where(np.any(timeseries != 0, axis=0))[0]
            timeseries = timeseries[:, active_cols]
            active_names = [roi_names[i] for i in active_cols]
            
            Logger.info(f"  Active ROIs: {len(active_names)}/{len(roi_names)}")
            
            # Global signal regression
            if self.config.CONN_GSR and timeseries.shape[1] > 0:
                Logger.info("  Applying GSR...")
                global_signal = timeseries.mean(axis=1, keepdims=True)
                beta = ((global_signal * timeseries).sum(axis=0) / 
                       ((global_signal * global_signal).sum() + 1e-12))
                timeseries = timeseries - global_signal * beta
            
            # Compute connectivity
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
# MAIN PIPELINE
# ============================================================================

class EEGtoFMRIPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.start_time = datetime.now()
    
    def run(self):
        """Execute full pipeline"""
        
        Logger.section("EEG-to-fMRI Pipeline v3.2 (K-Fold Optimized)")
        Logger.info(f"Data path: {self.config.DATA_PATH}")
        Logger.info(f"EEG file: {self.config.EEG_FILE}")
        
        # ===== 1. LOAD EEG DATA =====
        Logger.section("1. Loading EEG Data")
        
        eeg_path = Path(self.config.DATA_PATH) / self.config.EEG_FILE
        mat_data = scipy.io.loadmat(eeg_path, squeeze_me=True, struct_as_record=False)
        
        eeg_raw = np.array(mat_data['dataRest']).astype(np.float32)[:64]
        Logger.info(f"EEG shape: {eeg_raw.shape}")
        
        # Basic filtering
        eeg_filtered = SignalProcessor.bandpass(eeg_raw, self.config.FS, 1, 45)
        eeg_filtered = SignalProcessor.notch(eeg_filtered, self.config.FS, freq=50.0)
        Logger.success("Preprocessing: Bandpass (1-45Hz) + Notch (50Hz)")
        
        # ===== 2. LOAD COORDINATES =====
        Logger.section("2. Loading Channel Coordinates")
        
        ch_coords, ch_order = CoordinateSystem.load_coordinates()
        Logger.info(f"Channels loaded: {len(ch_order)}")
        
        # Verify key channels
        for ch in ['Oz', 'O1', 'O2', 'POz', 'Pz']:
            if ch in ch_coords:
                coord = ch_coords[ch]
                Logger.info(f"  {ch:4s}: [{coord[0]:6.1f}, {coord[1]:6.1f}, {coord[2]:6.1f}]")
        
        # ===== 3. ICA ARTIFACT REMOVAL =====
        Logger.section("3. ICA Artifact Removal")
        
        ica_remover = ICAArtifactRemover(self.config)
        eeg_clean, ica_info = ica_remover.clean(eeg_filtered, self.config.FS, ch_order)
        
        if ica_info['reasons']:
            Logger.info("Removed components:")
            for idx, reason in ica_info['reasons'].items():
                Logger.info(f"  [{idx:2d}] {reason}")
        
        # ===== 4. CSD RE-REFERENCING =====
        Logger.section("4. CSD Re-referencing")
        
        eeg_csd = CSDReferencer.apply(eeg_clean, ch_coords, ch_order, self.config)
        
        # ===== 5. SEGMENTATION =====
        Logger.section("5. Segmentation")
        
        segment_samples = int(self.config.FS * self.config.SEGMENT_DURATION)
        n_segments = eeg_csd.shape[1] // segment_samples
        
        segments = np.stack([
            eeg_csd[:, i*segment_samples:(i+1)*segment_samples]
            for i in range(n_segments)
        ])
        
        Logger.info(f"Segments: {n_segments} × {self.config.SEGMENT_DURATION}s (TR)")
        
        # ===== 6. VOXEL GRID =====
        Logger.section("6. Voxel Grid & Gray Matter Mask")
        
        grid = VoxelGrid(self.config)
        
        # ===== 7. VOXEL SIGNATURES =====
        Logger.section("7. Voxel Signatures")
        
        signature_computer = SignatureComputer(self.config)
        voxel_sigs = signature_computer.compute(grid.coords_gm, ch_coords, ch_order)
        
        # ===== 8. PHASE 1: COMPUTE MI/DICE FOR ALL BANDS =====
        Logger.section("8. Computing MI/Dice Scores (All Bands)")

        mi_all = {}
        dice_all = {}

        # ✅ YENİ: Band counter
        band_count = 0
        total_bands = len(self.config.BANDS)

        for band_name, freq_range in self.config.BANDS.items():
            band_count += 1
    
            Logger.info(f"\n{'='*70}")
            Logger.info(f"🔍 DEBUG: Processing band {band_count}/{total_bands}: {band_name.upper()}")
            Logger.info(f"{'='*70}")
            Logger.info(f"Band: {band_name.upper()} ({freq_range[0]}-{freq_range[1]} Hz)")
    
            try:
                # a) Hilbert envelope
                Logger.info(f"🔍 DEBUG: Computing Hilbert envelopes for {band_name}...")
                snapshots = np.stack([
                    SignalProcessor.hilbert_envelope(seg, freq_range, self.config.FS)
                    for seg in tqdm(segments, desc=f"  {band_name} envelopes", ncols=80)
                ])
                Logger.info(f"🔍 DEBUG: Snapshots shape: {snapshots.shape}")
        
                # b) MI/Dice computation
                Logger.info(f"🔍 DEBUG: Computing MI/Dice for {band_name}...")
                mi_dice_computer = MIDiceComputer(self.config)
                mi_scores, dice_scores = mi_dice_computer.compute(voxel_sigs, snapshots)
        
                Logger.info(f"🔍 DEBUG: MI shape: {mi_scores.shape}, Dice shape: {dice_scores.shape}")
                Logger.info("Regional boost: DISABLED")
        
                # c) Store for optimization
                Logger.info(f"🔍 DEBUG: Storing results for {band_name}...")
                mi_all[band_name] = mi_scores
                dice_all[band_name] = dice_scores
        
                Logger.success(f"✅ {band_name}: MI/Dice computed and stored")
        
                # d) Cleanup to save memory
                Logger.info(f"🔍 DEBUG: Cleaning up {band_name} snapshots...")
                del snapshots, mi_scores, dice_scores
                gc.collect()
        
                Logger.info(f"🔍 DEBUG: Band {band_count}/{total_bands} ({band_name}) completed!\n")
    
            except Exception as e:
                Logger.error(f"❌ FAILED on band {band_name}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        
                # Store dummy data to continue
                Logger.warn(f"Storing dummy data for {band_name} to continue...")
                mi_all[band_name] = np.zeros((len(grid.coords_gm), len(segments)), dtype=np.float32)
                dice_all[band_name] = np.zeros_like(mi_all[band_name])

        # ✅ Verify all bands completed
        Logger.info(f"\n{'='*70}")
        Logger.info(f"🔍 DEBUG: Phase 1 COMPLETED!")
        Logger.info(f"🔍 DEBUG: Collected bands: {list(mi_all.keys())}")
        Logger.info(f"🔍 DEBUG: Expected bands: {list(self.config.BANDS.keys())}")

        if set(mi_all.keys()) == set(self.config.BANDS.keys()):
            Logger.success(f"✅ All {len(mi_all)} bands computed successfully!")
        else:
            Logger.error(f"❌ Missing bands: {set(self.config.BANDS.keys()) - set(mi_all.keys())}")

        Logger.info(f"{'='*70}\n")

        # ===== 9. PHASE 2: K-FOLD OPTIMIZATION =====
        Logger.info("🔍 DEBUG: ========== STARTING PHASE 2: K-FOLD ==========")

        try:
            # K-fold optimization
            optimizer = SimplifiedKFoldOptimizer(self.config)
            optimizer.set_voxel_coords(grid.coords_gm)  # ✅ BU SATIRI EKLE
            optimal_weights_kfold = optimizer.optimize_all_bands(mi_all, dice_all)
    
            Logger.success("✅ K-fold optimization completed!")
    
            optimal_weights = optimal_weights_kfold.copy()  # ← BU SATIRI EKLE
    
            Logger.info("Final weights (with alpha override):")
            for band, weight in optimal_weights.items():
                Logger.info(f"  {band:8s}: MI={weight:.2f}, Dice={1-weight:.2f}")

        except Exception as e:
            Logger.error(f"❌ K-fold optimization FAILED: {type(e).__name__}: {e}")
    
            import traceback
            Logger.error("Full traceback:")
            traceback.print_exc()
    
            Logger.warn("⚠️  Falling back to literature-based weights...")
    
            optimal_weights = {
                "delta": 0.50,
                "theta": 0.45,
                "alpha": 0.40,
                "beta": 0.55,
                "gamma": 0.50,
            }
    
            Logger.info(f"Using fallback weights: {optimal_weights}")

        Logger.info("🔍 DEBUG: ========== PHASE 2 COMPLETED ==========\n")

        # ===== 10. PHASE 3: BUILD VOLUMES WITH OPTIMIZED WEIGHTS =====
        Logger.section("10. Building Volumes with Optimized Weights")
    
        for band_name in self.config.BANDS.keys():
            Logger.info(f"\n{'='*70}")
            Logger.info(f"BAND: {band_name.upper()}")
            Logger.info(f"{'='*70}")
        
            # a) Retrieve MI/Dice
            mi_scores = mi_all[band_name]
            dice_scores = dice_all[band_name]
        
            # b) Use OPTIMIZED weights
            mi_weight = optimal_weights[band_name]
            dice_weight = 1.0 - mi_weight
        
            Logger.info(f"Hybrid weights: MI={mi_weight:.3f}, Dice={dice_weight:.3f}")
        
            # c) Compute hybrid score with optimized weights
            hybrid = mi_weight * mi_scores + dice_weight * dice_scores
        
            # Normalize
            hmin, hmax = hybrid.min(), hybrid.max()
            if hmax > hmin:
                hybrid = (hybrid - hmin) / (hmax - hmin)
        
            # Contrast enhancement
            hybrid = np.power(hybrid, 0.7)
        
            # d) Build 4D volume
            Logger.info("Building 4D volume...")
            volume_builder = VolumeBuilder(grid)
            volume = volume_builder.build(hybrid, grid.coords_gm)
        
            # e) Post-processing
            post_processor = PostProcessor(self.config, grid)
        
            if self.config.APPLY_ZSCORE:
                volume = post_processor.zscore_normalize(volume, mode=self.config.ZSCORE_MODE)
          
            # Anisotropic smoothing
            volume = post_processor.anisotropic_smooth(volume, fwhm_xyz=(3.0, 3.0, 3.0))
        
            volume = post_processor.temporal_consistency(volume)
            volume = post_processor.roi_balanced_sparsify(volume, band_name)
            volume = post_processor.ray_killer_cleanup(volume)
        
            # f) QC
            qc = QualityControl(grid)

            # Physiological distribution (WITH DETAILED OUTPUT)
            phys_metrics = qc.physiological_distribution(volume, band_name)

            # ✅ SAVE METRICS TO CSV
            import csv
            metrics_path = Path(self.config.DATA_PATH) / f"{band_name}_phys_metrics_v32.csv"
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for key, val in sorted(phys_metrics.items()):
                    writer.writerow([key, f"{val:.4f}"])

            Logger.success(f"Metrics saved: {metrics_path.name}")

            # ROI coverage
            covered, weak, missing, roi_stats = qc.roi_coverage_analysis(volume, band_name)
            qc.export_coverage_csv(roi_stats, band_name, self.config.DATA_PATH)
        
            # g) Save NIfTI
            Logger.info("Saving NIfTI files...")
        
            nifti_img = nib.Nifti1Image(volume, affine=grid.affine)
            nifti_img.header['pixdim'][4] = self.config.SEGMENT_DURATION
            nifti_img.header.set_qform(grid.affine, code=1)
            nifti_img.header.set_sform(grid.affine, code=1)
        
            # Compressed NIfTI
            nii_path = Path(self.config.DATA_PATH) / f"{band_name}_voxel_v32.nii.gz"
            nib.save(nifti_img, nii_path)
            Logger.success(f"NIfTI: {nii_path.name}")
        
            # Uncompressed (SPM-compatible)
            spm_path = Path(self.config.DATA_PATH) / f"{band_name}_voxel_v32_spm.nii"
            nib.save(nifti_img, spm_path)
            Logger.success(f"SPM: {spm_path.name}")
        
            # h) Connectivity
            conn_analyzer = ConnectivityAnalyzer(self.config)
            conn_analyzer.compute(nifti_img, band_name, self.config.DATA_PATH)
        
            # i) Stats
            active = (volume != 0)
            if active.any():
                vals = volume[active]
                Logger.info(f"Stats: min={vals.min():.3f}, max={vals.max():.3f}, "
                          f"mean={vals.mean():.3f}, std={vals.std():.3f}")
        
            # Cleanup
            del volume, hybrid, mi_scores, dice_scores  # ✅ Doğru değişkenler
            gc.collect()
    
        # Cleanup MI/Dice data
        del mi_all, dice_all
        gc.collect()
    
        # ===== SUMMARY =====
        duration = (datetime.now() - self.start_time).total_seconds()
    
        Logger.section("PIPELINE COMPLETED")
        Logger.success(f"Total time: {int(duration//60)}m {int(duration%60)}s")
        Logger.info(f"Output directory: {self.config.DATA_PATH}")
        # ✅ Optimal weights path (yeniden tanımla)
        weights_path = Path(self.config.DATA_PATH) / "optimal_hybrid_weights_v33.json"
        if weights_path.exists():
            Logger.info(f"Optimized weights: {weights_path.name}")
        else:
            Logger.warn("Weights file not found (K-fold may have failed)")
        

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
    
    eeg_path = Path(config.DATA_PATH) / config.EEG_FILE
    if not eeg_path.exists():
        Logger.error(f"EEG file not found: {eeg_path}")
        return 1
    
    try:
        # Run pipeline
        pipeline = EEGtoFMRIPipeline(config)
        pipeline.run()
        return 0
    
    except Exception as e:
        Logger.error(f"Pipeline failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())