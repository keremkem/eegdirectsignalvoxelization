"""
SHARED PATH CONFIGURATION FOR VALIDATION SCRIPTS
=================================================
All validation analyses import paths from here.

Author: [Your name]
Date: 2024
EEGClaude v3.4 Pipeline
"""

from pathlib import Path
import json
import pickle
import numpy as np

# ============================================================================
# BASE PATHS (SENIN SİSTEMİNE GÖRE)
# ============================================================================

# Ana çalışma dizini
BASE_DIR = Path(r"C:\Users\kerem\downloads\eegyedek")

# Cache dizini (v33 - K-fold sonuçları, LOSO'da kullanıldı)
CACHE_DIR = BASE_DIR / "cache_v33"

# LOSO sonuçları dizini
LOSO_DIR = BASE_DIR / "LOSO"

# LOSO intermediate results (optional)
LOSO_INTERMEDIATE = LOSO_DIR / "loso_weights_intermediate_v34"

# Synthetic test cache dizini (yaratılacak)
SYNTHETIC_CACHE = BASE_DIR / "cache_synthetic"

# Validation outputs (yaratılacak)
VALIDATION_DIR = BASE_DIR / "validation_outputs"

# Create directories if they don't exist
SYNTHETIC_CACHE.mkdir(exist_ok=True)
VALIDATION_DIR.mkdir(exist_ok=True)

# ============================================================================
# KEY FILES
# ============================================================================

# LOSO optimal weights (JSON)
LOSO_WEIGHTS_FILE = LOSO_DIR / "loso_optimal_weights_v34.json"

# LOSO fold details (pickle)
LOSO_FOLD_DETAILS = LOSO_DIR / "loso_fold_details_v34.pkl"

# ============================================================================
# SUBJECT/BAND INFO
# ============================================================================

SUBJECTS = [f"S{i:02d}" for i in range(2, 12)]  # S02-S11
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
N_SUBJECTS = len(SUBJECTS)  # 10
N_BANDS = len(BANDS)  # 5

# ============================================================================
# FILE EXISTENCE CHECKS
# ============================================================================

def check_required_files():
    """
    Check if required files exist for validation analyses
    """
    
    print("\n" + "="*70)
    print("CHECKING REQUIRED FILES FOR VALIDATION")
    print("="*70)
    
    all_ok = True
    
    # ========================================================================
    # 1. Check CACHE_DIR
    # ========================================================================
    print("\n1️⃣ CACHE DIRECTORY (K-fold results used by LOSO)")
    print("-"*70)
    
    if not CACHE_DIR.exists():
        print(f"❌ Cache directory not found: {CACHE_DIR}")
        all_ok = False
    else:
        print(f" Cache directory found: {CACHE_DIR}")
        
        # Count cache files
        cache_files = list(CACHE_DIR.glob("S*_*_mi_dice.pkl"))
        expected = N_SUBJECTS * N_BANDS  # 50
        
        print(f"   Found: {len(cache_files)} cache files")
        print(f"   Expected: {expected} files ({N_SUBJECTS} subjects × {N_BANDS} bands)")
        
        if len(cache_files) < expected:
            print(f"   ⚠️ Missing {expected - len(cache_files)} cache files")
            
            # Show which ones are missing
            missing = []
            for subject in SUBJECTS:
                for band in BANDS:
                    cache_file = get_cache_file(subject, band)
                    if not cache_file.exists():
                        missing.append(f"{subject}_{band}")
            
            if len(missing) <= 10:
                print(f"   Missing: {', '.join(missing)}")
            else:
                print(f"   (Too many to list)")
        else:
            print(f"   ✅ All {expected} cache files present")
    
    # ========================================================================
    # 2. Check LOSO_DIR
    # ========================================================================
    print("\n2️⃣ LOSO RESULTS DIRECTORY")
    print("-"*70)
    
    if not LOSO_DIR.exists():
        print(f"❌ LOSO directory not found: {LOSO_DIR}")
        all_ok = False
    else:
        print(f"✅ LOSO directory found: {LOSO_DIR}")
        
        # Check key files
        key_files = {
            'LOSO weights (JSON)': LOSO_WEIGHTS_FILE,
            'LOSO fold details (pickle)': LOSO_FOLD_DETAILS
        }
        
        for name, path in key_files.items():
            if path.exists():
                print(f"   ✅ {name}: {path.name}")
            else:
                print(f"   ⚠️ {name} not found: {path.name}")
        
        # Count NIfTI files
        nii_files = list(LOSO_DIR.glob("*.nii*"))
        csv_files = list(LOSO_DIR.glob("*.csv"))
        
        print(f"\n   Additional files:")
        print(f"     NIfTI volumes: {len(nii_files)}")
        print(f"     CSV files:     {len(csv_files)}")
    
    # ========================================================================
    # 3. Check VALIDATION_DIR (will be created)
    # ========================================================================
    print("\n3️⃣ VALIDATION OUTPUT DIRECTORY")
    print("-"*70)
    print(f"✅ Validation outputs will be saved to: {VALIDATION_DIR}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    
    if all_ok:
        print("✅ ALL REQUIRED FILES PRESENT")
        print("\nYou can run all validation scripts with real data!")
    else:
        print("⚠️ SOME FILES MISSING")
        print("\nValidation scripts will use simulated data where needed.")
    
    print("="*70)
    
    return all_ok

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_cache_file(subject, band):
    """
    Get cache file path for a subject/band
    
    Example: get_cache_file("S02", "alpha") 
             → C:/Users/kerem/.../cache_v33/S02_alpha_mi_dice.pkl
    """
    return CACHE_DIR / f"{subject}_{band}_mi_dice.pkl"

def get_output_path(filename):
    """
    Get output path in validation directory
    
    Example: get_output_path("orthogonal_qc.png")
             → C:/Users/kerem/.../validation_outputs/orthogonal_qc.png
    """
    return VALIDATION_DIR / filename

def load_loso_optimal_weights():
    """
    Load LOSO optimal weights from JSON
    
    Returns:
        dict: {'alpha': 0.85, 'beta': 0.20, ...}
    """
    
    if not LOSO_WEIGHTS_FILE.exists():
        print(f"⚠️ LOSO weights file not found: {LOSO_WEIGHTS_FILE}")
        return None
    
    with open(LOSO_WEIGHTS_FILE, 'r') as f:
        weights = json.load(f)
    
    return weights

def load_loso_fold_details():
    """
    Load LOSO fold-level details from pickle
    
    Returns:
        dict: {
            'fold_results': {
                'alpha': [...],
                'beta': [...],
                ...
            },
            ...
        }
    """
    
    if not LOSO_FOLD_DETAILS.exists():
        print(f"⚠️ LOSO fold details not found: {LOSO_FOLD_DETAILS}")
        return None
    
    with open(LOSO_FOLD_DETAILS, 'rb') as f:
        details = pickle.load(f)
    
    return details

def load_cache_data(subject, band):
    """
    Load cache data for a subject/band
    
    Returns:
        dict: {
            'mi': array (n_voxels, 75),
            'dice': array (n_voxels, 75),
            'voxel_coords': array (n_voxels, 3)
        }
        or None if file doesn't exist
    """
    
    cache_file = get_cache_file(subject, band)
    
    if not cache_file.exists():
        return None
    
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    
    return cache

def get_all_cache_data(band):
    """
    Load and concatenate cache data across all subjects for a band
    
    Returns:
        tuple: (mi_values, dice_values, voxel_coords)
        or (None, None, None) if no data
    """
    
    all_mi = []
    all_dice = []
    all_coords = []
    
    for subject in SUBJECTS:
        cache = load_cache_data(subject, band)
        
        if cache is not None:
            all_mi.append(cache['mi'])
            all_dice.append(cache['dice'])
            all_coords.append(cache['voxel_coords'])
    
    if len(all_mi) == 0:
        return None, None, None
    
    # Concatenate
    mi_concat = np.vstack(all_mi)  # (total_voxels, 75)
    dice_concat = np.vstack(all_dice)
    coords_concat = np.vstack(all_coords)
    
    return mi_concat, dice_concat, coords_concat

# ============================================================================
# DATA AVAILABILITY FLAGS
# ============================================================================

def get_data_availability():
    """
    Check what data is available for validation scripts
    
    Returns:
        dict: {
            'cache_available': bool,
            'loso_weights_available': bool,
            'loso_details_available': bool
        }
    """
    
    availability = {
        'cache_available': CACHE_DIR.exists() and len(list(CACHE_DIR.glob("*.pkl"))) > 0,
        'loso_weights_available': LOSO_WEIGHTS_FILE.exists(),
        'loso_details_available': LOSO_FOLD_DETAILS.exists()
    }
    
    return availability

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

def print_config():
    """
    Print current configuration
    """
    
    print("\n" + "="*70)
    print("VALIDATION CONFIGURATION")
    print("="*70)
    
    print(f"\nBase directory:      {BASE_DIR}")
    print(f"Cache directory:     {CACHE_DIR}")
    print(f"LOSO directory:      {LOSO_DIR}")
    print(f"Validation outputs:  {VALIDATION_DIR}")
    
    print(f"\nSubjects: {N_SUBJECTS} ({SUBJECTS[0]} - {SUBJECTS[-1]})")
    print(f"Bands:    {N_BANDS} ({', '.join(BANDS)})")
    
    avail = get_data_availability()
    
    print(f"\nData availability:")
    print(f"  Cache files:       {'✅ Available' if avail['cache_available'] else '❌ Not found'}")
    print(f"  LOSO weights:      {'✅ Available' if avail['loso_weights_available'] else '❌ Not found'}")
    print(f"  LOSO fold details: {'✅ Available' if avail['loso_details_available'] else '❌ Not found'}")
    
    print("="*70)

# ============================================================================
# REAL DATA EXAMPLE VALUES (From your LOSO results)
# ============================================================================

# Your actual LOSO optimal weights (v34)
REAL_LOSO_WEIGHTS = {
    'alpha': 0.85,
    'theta': 0.34,
    'beta': 0.20,
    'gamma': 0.18,
    'delta': 0.16
}

# Your actual P/A ratios (from randomization test)
REAL_PA_RATIOS = {
    'alpha': 3.69,
    'theta': 1.44,
    'beta': 1.28
}

# ============================================================================
# RUN CHECK ON IMPORT
# ============================================================================

if __name__ == "__main__":
    print_config()
    print()

    check_required_files()
