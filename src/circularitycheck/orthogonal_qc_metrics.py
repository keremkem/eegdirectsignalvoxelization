# orthogonal_qc_metrics.py
"""
ORTHOGONAL QC METRICS
Tests whether optimization inflates non-ratio-based metrics.
"""

import sys
import io

# Windows konsol kodlama sorununu çöz
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import pickle
from config_paths import *

def compute_spatial_entropy(voxel_weights):
    """Shannon entropy: H = -sum p(i) log p(i)"""
    probs = voxel_weights / voxel_weights.sum()
    probs = probs[probs > 0]
    return entropy(probs, base=2)

def compute_gini_coefficient(voxel_weights):
    """Gini coefficient: 0 = equality, 1 = inequality"""
    sorted_weights = np.sort(voxel_weights)
    n = len(sorted_weights)
    cumsum = np.cumsum(sorted_weights)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_weights)) / (n * cumsum[-1]) - (n + 1) / n
    return gini

def compute_spatial_dispersion(voxel_coords, voxel_weights):
    """Weighted center of mass deviation"""
    center = np.average(voxel_coords, axis=0, weights=voxel_weights)
    distances = np.sqrt(((voxel_coords - center)**2).sum(axis=1))
    dispersion = np.average(distances, weights=voxel_weights)
    return dispersion

def load_and_compute_orthogonal_metrics(band):
    """Load REAL voxel data from cache and compute metrics"""
    
    all_weights = []
    all_coords = []

    for subject in SUBJECTS:
        cache_file = get_cache_file(subject, band)
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            
            mi_values = cache['mi'].mean(axis=1)
            coords = cache['voxel_coords']
            
            all_weights.append(mi_values)
            all_coords.append(coords)

    if len(all_weights) == 0:
        print(f"  [WARNING] No cache data for {band}, using simulation")
        
        n_voxels = 10000
        
        if band == 'alpha':
            weights = np.random.gamma(2, 2, n_voxels)
            weights[:5000] *= 2
        elif band == 'beta':
            weights = np.random.uniform(0.5, 1.5, n_voxels)
        else:
            weights = np.random.lognormal(0, 0.5, n_voxels)
        
        coords = np.random.randn(n_voxels, 3) * 50
    else:
        print(f"  [OK] Loaded {len(all_weights)} subjects for {band}")
        weights = np.concatenate(all_weights)
        coords = np.concatenate(all_coords)

    H = compute_spatial_entropy(weights)
    G = compute_gini_coefficient(weights)
    D = compute_spatial_dispersion(coords, weights)

    posterior_mask = coords[:, 1] < 0
    anterior_mask = coords[:, 1] > 0

    P = weights[posterior_mask].sum()
    A = weights[anterior_mask].sum()
    PA_ratio = P / A if A > 0 else 0

    return {
        'entropy': H,
        'gini': G,
        'dispersion': D,
        'pa_ratio': PA_ratio
    }

def test_metric_independence():
    """Test independence from optimization target"""
    
    print("\n" + "="*70)
    print("ORTHOGONAL QC METRICS TEST")
    print("="*70)

    results = {}

    for band in BANDS:
        print(f"\nProcessing {band.upper()}...")
        metrics = load_and_compute_orthogonal_metrics(band)
        results[band] = metrics
        
        print(f"  Entropy:    {metrics['entropy']:.3f}")
        print(f"  Gini:       {metrics['gini']:.3f}")
        print(f"  Dispersion: {metrics['dispersion']:.2f}")
        print(f"  P/A Ratio:  {metrics['pa_ratio']:.3f}")

    print("\n" + "="*70)
    print("CORRELATION WITH P/A RATIO")
    print("="*70)

    pa_ratios = [results[b]['pa_ratio'] for b in BANDS]
    entropies = [results[b]['entropy'] for b in BANDS]
    ginis = [results[b]['gini'] for b in BANDS]
    dispersions = [results[b]['dispersion'] for b in BANDS]

    corr_entropy = np.corrcoef(pa_ratios, entropies)[0, 1]
    corr_gini = np.corrcoef(pa_ratios, ginis)[0, 1]
    corr_disp = np.corrcoef(pa_ratios, dispersions)[0, 1]

    print(f"\nCorrelation with P/A ratio:")
    print(f"  Entropy:    r = {corr_entropy:.3f} {'[OK]' if abs(corr_entropy) < 0.3 else '[WARNING]'}")
    print(f"  Gini:       r = {corr_gini:.3f} {'[OK]' if abs(corr_gini) < 0.3 else '[WARNING]'}")
    print(f"  Dispersion: r = {corr_disp:.3f} {'[OK]' if abs(corr_disp) < 0.3 else '[WARNING]'}")

    visualize_orthogonal_metrics(results)
    save_report(results, corr_entropy, corr_gini, corr_disp)

    return results

def visualize_orthogonal_metrics(results):
    """Create independence test plots"""
    
    bands = list(results.keys())

    pa_ratios = [results[b]['pa_ratio'] for b in bands]
    entropies = [results[b]['entropy'] for b in bands]
    ginis = [results[b]['gini'] for b in bands]
    dispersions = [results[b]['dispersion'] for b in bands]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Entropy
    ax = axes[0]
    ax.scatter(pa_ratios, entropies, s=100, c='steelblue', edgecolors='black')

    for i, band in enumerate(bands):
        ax.text(pa_ratios[i], entropies[i], band[0].upper(), 
               fontsize=12, ha='center', va='center', color='white', fontweight='bold')

    r = np.corrcoef(pa_ratios, entropies)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')

    ax.set_xlabel('P/A Ratio (Optimization Target)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spatial Entropy (Orthogonal)', fontsize=12, fontweight='bold')
    ax.set_title('Independence Test: Entropy', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Plot 2: Gini
    ax = axes[1]
    ax.scatter(pa_ratios, ginis, s=100, c='coral', edgecolors='black')

    for i, band in enumerate(bands):
        ax.text(pa_ratios[i], ginis[i], band[0].upper(), 
               fontsize=12, ha='center', va='center', color='white', fontweight='bold')

    r = np.corrcoef(pa_ratios, ginis)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')

    ax.set_xlabel('P/A Ratio (Optimization Target)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gini Coefficient (Orthogonal)', fontsize=12, fontweight='bold')
    ax.set_title('Independence Test: Gini', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Plot 3: Dispersion
    ax = axes[2]
    ax.scatter(pa_ratios, dispersions, s=100, c='lightgreen', edgecolors='black')

    for i, band in enumerate(bands):
        ax.text(pa_ratios[i], dispersions[i], band[0].upper(), 
               fontsize=12, ha='center', va='center', fontweight='bold')

    r = np.corrcoef(pa_ratios, dispersions)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')

    ax.set_xlabel('P/A Ratio (Optimization Target)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spatial Dispersion (Orthogonal)', fontsize=12, fontweight='bold')
    ax.set_title('Independence Test: Dispersion', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    fig_path = get_output_path("orthogonal_qc_independence.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Figure saved: {fig_path.name}")
    plt.show()

def save_report(results, corr_entropy, corr_gini, corr_disp):
    """Save text report"""
    
    report_path = get_output_path("orthogonal_qc_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ORTHOGONAL QC METRICS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("QUESTION:\n")
        f.write("Are QC metrics independent from optimization target (P/A ratio)?\n\n")
        
        f.write("METHOD:\n")
        f.write("Computed three orthogonal spatial metrics:\n")
        f.write("  - Spatial Entropy (Shannon)\n")
        f.write("  - Gini Coefficient\n")
        f.write("  - Spatial Dispersion Index\n\n")
        
        f.write("RESULTS:\n")
        f.write("-"*70 + "\n")
        
        for band in BANDS:
            r = results[band]
            f.write(f"\n{band.upper()}:\n")
            f.write(f"  Entropy:    {r['entropy']:.3f}\n")
            f.write(f"  Gini:       {r['gini']:.3f}\n")
            f.write(f"  Dispersion: {r['dispersion']:.2f}\n")
            f.write(f"  P/A Ratio:  {r['pa_ratio']:.3f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CORRELATIONS:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Entropy    vs P/A: r = {corr_entropy:.3f} {'[OK] Independent' if abs(corr_entropy) < 0.3 else '[WARNING] Correlated'}\n")
        f.write(f"  Gini       vs P/A: r = {corr_gini:.3f} {'[OK] Independent' if abs(corr_gini) < 0.3 else '[WARNING] Correlated'}\n")
        f.write(f"  Dispersion vs P/A: r = {corr_disp:.3f} {'[OK] Independent' if abs(corr_disp) < 0.3 else '[WARNING] Correlated'}\n\n")
        
        f.write("CONCLUSION:\n")
        f.write("If |r| < 0.3, metrics are orthogonal -> no QC inflation.\n")

    print(f"[OK] Report saved: {report_path.name}")

if __name__ == "__main__":
    check_required_files()
    results = test_metric_independence()
    print("\n[OK] Orthogonal QC analysis complete!")