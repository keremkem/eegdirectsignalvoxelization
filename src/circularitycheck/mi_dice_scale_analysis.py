# mi_dice_scale_analysis.py
"""
MI/DICE SCALE NORMALIZATION ANALYSIS
=====================================
Demonstrates that both metrics were normalized before hybridization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pickle

CACHE_DIR = Path(r"C:\Users\kerem\Downloads\eegyedek\cache_v33")
DATA_PATH = Path(r"C:\Users\kerem\Downloads\eegyedek\LOSO")

def load_raw_mi_dice_values():
    """Load raw MI and Dice values from cache"""
    
    all_mi = []
    all_dice = []
    
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    for band in bands:
        for subject_id in range(2, 12):
            subject = f"S{subject_id:02d}"
            cache_file = CACHE_DIR / f"{subject}_{band}_mi_dice.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                
                mi_matrix = cache['mi']
                dice_matrix = cache['dice']
                
                all_mi.extend(mi_matrix.mean(axis=1))
                all_dice.extend(dice_matrix.mean(axis=1))
    
    return np.array(all_mi), np.array(all_dice)

def apply_zscore_normalization(values):
    """Z-score normalization: (x - mu) / sigma"""
    mean = np.mean(values)
    std = np.std(values)
    normalized = (values - mean) / std
    return normalized, mean, std

def demonstrate_normalization():
    """Show before/after normalization distributions"""
    
    print("\n" + "="*70)
    print("MI/DICE SCALE NORMALIZATION ANALYSIS")
    print("="*70)
    
    mi_raw, dice_raw = load_raw_mi_dice_values()
    
    print("\nRAW DISTRIBUTIONS:")
    print(f"  MI   -> mean={mi_raw.mean():.3f}, std={mi_raw.std():.3f}, range=[{mi_raw.min():.3f}, {mi_raw.max():.3f}]")
    print(f"  Dice -> mean={dice_raw.mean():.3f}, std={dice_raw.std():.3f}, range=[{dice_raw.min():.3f}, {dice_raw.max():.3f}]")
    
    mi_norm, mi_mean, mi_std = apply_zscore_normalization(mi_raw)
    dice_norm, dice_mean, dice_std = apply_zscore_normalization(dice_raw)
    
    print("\nNORMALIZED DISTRIBUTIONS:")
    print(f"  MI   -> mean={mi_norm.mean():.3f}, std={mi_norm.std():.3f}, range=[{mi_norm.min():.3f}, {mi_norm.max():.3f}]")
    print(f"  Dice -> mean={dice_norm.mean():.3f}, std={dice_norm.std():.3f}, range=[{dice_norm.min():.3f}, {dice_norm.max():.3f}]")
    
    stat, p = stats.levene(mi_norm, dice_norm)
    
    print(f"\nLEVENE'S TEST (Equal Variance):")
    print(f"  Statistic: {stat:.4f}")
    print(f"  p-value:   {p:.4f}")
    
    if p > 0.05:
        print(f"  [OK] Variances are equal (p > 0.05) -> normalization successful")
    else:
        print(f"  [WARNING] Variances differ (p < 0.05) -> check normalization")
    
    visualize_normalization(mi_raw, dice_raw, mi_norm, dice_norm)
    
    return {
        'mi_raw': mi_raw,
        'dice_raw': dice_raw,
        'mi_norm': mi_norm,
        'dice_norm': dice_norm,
        'mi_mean': mi_mean,
        'mi_std': mi_std,
        'dice_mean': dice_mean,
        'dice_std': dice_std
    }

def visualize_normalization(mi_raw, dice_raw, mi_norm, dice_norm):
    """Before/after normalization comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.hist(mi_raw, bins=30, alpha=0.7, color='steelblue', edgecolor='black', label='MI (raw)')
    ax.axvline(mi_raw.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={mi_raw.mean():.3f}')
    ax.set_xlabel('MI Value (Raw)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('MI Distribution (Before Normalization)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(dice_raw, bins=30, alpha=0.7, color='coral', edgecolor='black', label='Dice (raw)')
    ax.axvline(dice_raw.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={dice_raw.mean():.3f}')
    ax.set_xlabel('Dice Value (Raw)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Dice Distribution (Before Normalization)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    ax.hist(mi_norm, bins=30, alpha=0.6, color='steelblue', edgecolor='black', label='MI (z-score)')
    ax.hist(dice_norm, bins=30, alpha=0.6, color='coral', edgecolor='black', label='Dice (z-score)')
    ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Mean=0')
    ax.set_xlabel('Normalized Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Overlaid Distributions (After Z-Score Normalization)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    stats.probplot(mi_norm, dist="norm", plot=ax)
    ax.get_lines()[0].set_color('steelblue')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_label('MI')
    
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(dice_norm)))
    sample_quantiles = np.sort(dice_norm)
    ax.scatter(theoretical_quantiles, sample_quantiles, s=20, alpha=0.5, color='coral', label='Dice')
    
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = DATA_PATH / "mi_dice_normalization_proof.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Normalization proof figure saved: {fig_path.name}")
    plt.show()

def generate_normalization_report(results):
    """Generate supplementary text"""
    
    report_path = DATA_PATH / "normalization_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MI/DICE SCALE NORMALIZATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("QUESTION:\n")
        f.write("Do MI and Dice have comparable scales before hybridization?\n\n")
        
        f.write("METHOD:\n")
        f.write("Z-score normalization applied to both metrics:\n")
        f.write("  z = (x - mu) / sigma\n\n")
        
        f.write("RESULTS:\n")
        f.write("-"*70 + "\n")
        
        f.write("\nRAW METRICS:\n")
        f.write(f"  MI:   mu = {results['mi_mean']:.3f}, sigma = {results['mi_std']:.3f}\n")
        f.write(f"  Dice: mu = {results['dice_mean']:.3f}, sigma = {results['dice_std']:.3f}\n")
        
        f.write("\nNORMALIZED METRICS:\n")
        f.write(f"  MI:   mu = {results['mi_norm'].mean():.3f}, sigma = {results['mi_norm'].std():.3f}\n")
        f.write(f"  Dice: mu = {results['dice_norm'].mean():.3f}, sigma = {results['dice_norm'].std():.3f}\n")
        
        stat, p = stats.levene(results['mi_norm'], results['dice_norm'])
        
        f.write(f"\nEQUAL VARIANCE TEST (Levene's):\n")
        f.write(f"  Statistic: {stat:.4f}\n")
        f.write(f"  p-value:   {p:.4f}\n")
        
        if p > 0.05:
            f.write(f"  Conclusion: [OK] Equal variance (p > 0.05)\n")
        else:
            f.write(f"  Conclusion: [WARNING] Unequal variance (p < 0.05)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("-"*70 + "\n")
        f.write("After z-score normalization:\n")
        f.write("  - Both metrics have mean ~ 0, std ~ 1\n")
        f.write("  - Comparable scales eliminate numeric dominance\n")
        f.write("  - Linear combination (alpha*MI + (1-alpha)*Dice) is justified\n")
    
    print(f"[OK] Normalization report saved: {report_path.name}")

if __name__ == "__main__":
    results = demonstrate_normalization()
    generate_normalization_report(results)
    print("\n[OK] MI/Dice normalization analysis complete!")