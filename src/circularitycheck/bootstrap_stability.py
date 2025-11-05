# bootstrap_stability_ROBUST.py
"""
ROBUST BOOTSTRAP ANALYSIS
=========================
Uses median + MAD instead of mean + std to handle high inter-subject variability.
Provides honest estimates with per-subject breakdown.
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import gc

CACHE_DIR = Path(r"C:\Users\kerem\Downloads\eegyedek\cache_v33")
DATA_PATH = Path(r"C:\Users\kerem\Downloads\eegyedek\LOSO")

SUBJECTS = [f"S{i:02d}" for i in range(2, 12)]
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_voxel_cache(subject, band):
    """Load voxel-level MI/Dice cache"""
    cache_file = CACHE_DIR / f"{subject}_{band}_mi_dice.pkl"
    
    if not cache_file.exists():
        return None
    
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def compute_optimal_weight_from_voxels(mi_array, dice_array):
    """Find optimal weight by aggregating voxel scores"""
    avg_mi = np.mean(mi_array, axis=0)  # (75,)
    optimal_idx = np.argmax(avg_mi)
    return optimal_idx

def bootstrap_single_fold_optimized(train_subjects, band, n_bootstrap=100):
    """
    Bootstrap by resampling SUBJECTS (not voxels)
    Memory-efficient version
    """
    
    weight_grid = np.linspace(0, 1, 75)
    bootstrap_weights = []
    
    # Load all train caches once
    train_caches = []
    for subj in train_subjects:
        cache = load_voxel_cache(subj, band)
        if cache is not None:
            train_caches.append({
                'mi': cache['mi'],
                'dice': cache['dice']
            })
    
    if len(train_caches) == 0:
        return None
    
    n_subjects = len(train_caches)
    
    # Bootstrap by resampling subjects
    for _ in range(n_bootstrap):
        subject_indices = np.random.choice(n_subjects, size=n_subjects, replace=True)
        
        mi_sum = None
        dice_sum = None
        total_voxels = 0
        
        for idx in subject_indices:
            mi_array = train_caches[idx]['mi']
            dice_array = train_caches[idx]['dice']
            
            n_voxels = mi_array.shape[0]
            
            if mi_sum is None:
                mi_sum = np.sum(mi_array, axis=0)
                dice_sum = np.sum(dice_array, axis=0)
            else:
                mi_sum += np.sum(mi_array, axis=0)
                dice_sum += np.sum(dice_array, axis=0)
            
            total_voxels += n_voxels
        
        avg_mi = mi_sum / total_voxels
        optimal_idx = np.argmax(avg_mi)
        bootstrap_weights.append(weight_grid[optimal_idx])
    
    del train_caches
    gc.collect()
    
    return np.array(bootstrap_weights)

def simulate_loso_lightweight(band, n_bootstrap=100):
    """LOSO simulation with subject-level bootstrap"""
    
    print(f"\n{'='*70}")
    print(f"{band.upper()} - SUBJECT-LEVEL BOOTSTRAP")
    print(f"{'='*70}")
    
    fold_results = []
    
    for test_subject in tqdm(SUBJECTS, desc=f"{band}"):
        train_subjects = [s for s in SUBJECTS if s != test_subject]
        
        bootstrap_weights = bootstrap_single_fold_optimized(
            train_subjects, band, n_bootstrap
        )
        
        if bootstrap_weights is None:
            continue
        
        fold_result = {
            'test_subject': test_subject,
            'mean_weight': bootstrap_weights.mean(),
            'median_weight': np.median(bootstrap_weights),
            'std_weight': bootstrap_weights.std(),
            'ci_lower': np.percentile(bootstrap_weights, 2.5),
            'ci_upper': np.percentile(bootstrap_weights, 97.5)
        }
        
        fold_results.append(fold_result)
    
    return fold_results

# ============================================================================
# ROBUST STATISTICS
# ============================================================================

def aggregate_loso_results_ROBUST(fold_results):
    """
    ROBUST aggregation using MEDIAN + MAD
    
    MAD (Median Absolute Deviation) is robust to outliers
    """
    
    fold_means = np.array([f['mean_weight'] for f in fold_results])
    fold_medians = np.array([f['median_weight'] for f in fold_results])
    
    # Overall MEDIAN (robust central tendency)
    overall_median = np.median(fold_medians)
    
    # MAD (Median Absolute Deviation)
    mad = np.median(np.abs(fold_medians - overall_median))
    
    # Convert MAD to std equivalent (for normal distribution)
    # Factor 1.4826 assumes normality
    std_equivalent = mad * 1.4826
    
    # 95% CI using MAD
    ci_lower = overall_median - 1.96 * std_equivalent
    ci_upper = overall_median + 1.96 * std_equivalent
    
    # Clamp to valid range [0, 1]
    ci_lower = max(0.0, ci_lower)
    ci_upper = min(1.0, ci_upper)
    
    # Also compute traditional stats for comparison
    traditional_mean = fold_means.mean()
    traditional_std = fold_means.std()
    
    return {
        'median': overall_median,
        'mad': mad,
        'std_equivalent': std_equivalent,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'n_folds': len(fold_results),
        # Traditional stats for comparison
        'traditional_mean': traditional_mean,
        'traditional_std': traditional_std,
        # Per-subject breakdown
        'fold_medians': fold_medians,
        'fold_means': fold_means
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_robust_results(results):
    """
    Visualization with:
    1. Median + MAD error bars
    2. Per-subject distribution (violin plot style)
    """
    
    print("\n" + "="*70)
    print("GENERATING ROBUST VISUALIZATION...")
    print("="*70)
    
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    bands = [b for b in bands if b in results]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    
    # ========================================================================
    # PLOT 1: Median with MAD error bars
    # ========================================================================
    ax = axes[0]
    x = np.arange(len(bands))
    
    medians = [results[b]['median'] for b in bands]
    mads = [results[b]['mad'] for b in bands]
    ci_lowers = [results[b]['ci_lower'] for b in bands]
    ci_uppers = [results[b]['ci_upper'] for b in bands]
    
    # Error bars using MAD
    ax.errorbar(x, medians, yerr=mads, 
                fmt='o', markersize=14, capsize=12, capthick=3,
                color='steelblue', ecolor='steelblue', linewidth=3,
                label='Median ± MAD', alpha=0.8)
    
    # Add CI range as shaded area
    for i, band in enumerate(bands):
        ax.fill_between([i-0.2, i+0.2], 
                        ci_lowers[i], ci_uppers[i],
                        alpha=0.2, color='steelblue')
    
    ax.set_ylabel('Optimal MI Weight', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency Band', fontsize=14, fontweight='bold')
    ax.set_title('Robust Estimates: Median ± MAD (N=10 subjects)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bands], fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, 
               label='Neutral', linewidth=2)
    ax.legend(fontsize=11, loc='upper right')
    
    # Add MAD values as text
    for i, band in enumerate(bands):
        mad = mads[i]
        ax.text(i, medians[i] + 0.08, f'MAD={mad:.3f}', 
               ha='center', fontsize=9, color='darkred',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Warning text
    ax.text(0.5, 0.05, 
            '⚠ High variability reflects individual differences across subjects',
            transform=ax.transAxes, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # ========================================================================
    # PLOT 2: Per-subject distribution (strip plot)
    # ========================================================================
    ax = axes[1]
    
    for i, band in enumerate(bands):
        fold_medians = results[band]['fold_medians']
        
        # Jitter for visibility
        x_jitter = np.random.normal(i, 0.04, size=len(fold_medians))
        
        # Plot individual subjects
        ax.scatter(x_jitter, fold_medians, 
                  alpha=0.6, s=80, color='coral', edgecolors='black', linewidth=1)
        
        # Plot overall median
        median = results[band]['median']
        ax.plot([i-0.3, i+0.3], [median, median], 
               color='steelblue', linewidth=3, alpha=0.8)
    
    ax.set_ylabel('Optimal MI Weight', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency Band', fontsize=14, fontweight='bold')
    ax.set_title('Per-Subject Variability (Each dot = 1 subject)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bands], fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # ========================================================================
    # PLOT 3: Comparison - Robust vs Traditional
    # ========================================================================
    ax = axes[2]
    
    x_offset = np.array([-0.15, 0.15])
    
    for i, band in enumerate(bands):
        # Robust (median + MAD)
        robust_median = results[band]['median']
        robust_std = results[band]['std_equivalent']
        
        # Traditional (mean + std)
        trad_mean = results[band]['traditional_mean']
        trad_std = results[band]['traditional_std']
        
        # Plot both
        ax.errorbar(i + x_offset[0], robust_median, yerr=robust_std,
                   fmt='o', markersize=10, capsize=8, capthick=2,
                   color='steelblue', ecolor='steelblue', 
                   label='Robust (Median±MAD)' if i == 0 else '')
        
        ax.errorbar(i + x_offset[1], trad_mean, yerr=trad_std,
                   fmt='s', markersize=10, capsize=8, capthick=2,
                   color='coral', ecolor='coral',
                   label='Traditional (Mean±SD)' if i == 0 else '')
    
    ax.set_ylabel('Optimal MI Weight', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency Band', fontsize=14, fontweight='bold')
    ax.set_title('Robust vs Traditional Statistics', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bands], fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    
    fig_path = DATA_PATH / "bootstrap_ROBUST_results.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {fig_path.name}")
    
    plt.show()

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_robust_bootstrap(n_bootstrap=100):
    """
    Main robust bootstrap analysis
    """
    
    print("="*70)
    print("ROBUST BOOTSTRAP ANALYSIS")
    print("="*70)
    print(f"\nMethod: Subject-level resampling")
    print(f"Bootstrap iterations per fold: {n_bootstrap}")
    print(f"LOSO folds: {len(SUBJECTS)}")
    print(f"Statistics: MEDIAN ± MAD (robust to outliers)")
    print()
    
    all_results = {}
    
    for band in BANDS:
        fold_results = simulate_loso_lightweight(band, n_bootstrap)
        
        if len(fold_results) > 0:
            aggregated = aggregate_loso_results_ROBUST(fold_results)
            all_results[band] = aggregated
            
            print(f"\n{band.upper()} ROBUST SUMMARY:")
            print(f"  Median: {aggregated['median']:.3f}")
            print(f"  MAD:    {aggregated['mad']:.4f}")
            print(f"  95% CI: [{aggregated['ci_lower']:.3f}, {aggregated['ci_upper']:.3f}]")
            print(f"  CI Width: {aggregated['ci_width']:.4f}")
            print(f"  (Traditional: Mean={aggregated['traditional_mean']:.3f}, "
                  f"SD={aggregated['traditional_std']:.4f})")
    
    return all_results

def save_robust_results(results):
    """Save results to text file"""
    
    output_file = DATA_PATH / "bootstrap_ROBUST_results.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ROBUST BOOTSTRAP ANALYSIS RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write("Method: Subject-level resampling with robust statistics\n")
        f.write("Estimator: MEDIAN (robust to outliers)\n")
        f.write("Dispersion: MAD (Median Absolute Deviation)\n")
        f.write("Sample: N=10 subjects\n\n")
        
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            if band not in results:
                continue
            
            r = results[band]
            
            f.write(f"{band.upper()}:\n")
            f.write(f"  Median:     {r['median']:.3f}\n")
            f.write(f"  MAD:        {r['mad']:.4f}\n")
            f.write(f"  Std equiv:  {r['std_equivalent']:.4f}\n")
            f.write(f"  95% CI:     [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]\n")
            f.write(f"  CI Width:   {r['ci_width']:.4f}\n")
            f.write(f"\n  Traditional comparison:\n")
            f.write(f"  Mean:       {r['traditional_mean']:.3f}\n")
            f.write(f"  SD:         {r['traditional_std']:.4f}\n\n")
        
        f.write("\nINTERPRETATION:\n")
        f.write("-"*70 + "\n")
        f.write("• MEDIAN is more robust than MEAN to outlier subjects\n")
        f.write("• MAD (Median Absolute Deviation) is robust to extreme values\n")
        f.write("• High variability reflects genuine inter-subject differences\n")
        f.write("• Caution: Small sample size (N=10) limits precision\n\n")
        
        f.write("RECOMMENDATION:\n")
        f.write("-"*70 + "\n")
        f.write("Report as: 'Median optimal weight = X.XX (MAD = X.XX)'\n")
        f.write("Acknowledge: 'Substantial inter-subject variability observed'\n")
    
    print(f"✅ Results saved: {output_file.name}")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    
    print("\n🚀 Starting ROBUST bootstrap analysis...\n")
    
    # Run with 100 bootstrap iterations per fold
    results = run_robust_bootstrap(n_bootstrap=100)
    
    if results:
        visualize_robust_results(results)
        save_robust_results(results)
        
        print("\n" + "="*70)
        print("✅ ROBUST ANALYSIS COMPLETE")
        print("="*70)
        print("\nKey Points:")
        print("  • Used MEDIAN instead of MEAN (robust to outliers)")
        print("  • Used MAD instead of SD (robust dispersion)")
        print("  • High variability is REAL (inter-subject differences)")
        print("  • Report honestly: acknowledge variability in paper")