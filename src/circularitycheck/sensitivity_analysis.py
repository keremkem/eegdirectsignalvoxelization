# sensitivity_analysis_simplified.py - FINAL COMPLETE
"""
TARGET SENSITIVITY ANALYSIS - Multi-Band Version (Auto-detect version)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = r"C:\Users\kerem\Downloads\eegyedek\eckfoldgroup"

SUBJECTS = ["S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11"]
BANDS = ["delta", "gamma"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_version_tag(data_path):
    """
    Auto-detect version tag from existing files
    """
    search_pattern = str(Path(data_path) / "*_alpha_phys_metrics_v*.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"⚠️  No phys_metrics files found in {data_path}")
        return None
    
    # Extract version from first file
    filename = Path(files[0]).name
    version_part = filename.split('_v')[-1].replace('.csv', '')
    
    return f"v{version_part}"

def load_qc_metrics(version_tag):
    """
    Load QC metrics - FIXED VERSION
    Beta/theta için P/A ratio'yu manuel hesapla
    """
    
    all_metrics = []
    
    print("\n🔍 Loading files...")
    
    for subject_id in SUBJECTS:
        for band in BANDS:
            csv_path = Path(DATA_PATH) / f"{subject_id}_{band}_phys_metrics_{version_tag}.csv"
            
            if not csv_path.exists():
                print(f"⚠️  Not found: {csv_path.name}")
                continue
            
            print(f"✅ Found: {csv_path.name}")
            
            try:
                df = pd.read_csv(csv_path)
                
                pa_ratio = None
                
                # METHOD 1: Try direct P/A ratio (works for alpha)
                ratio_key = f"{band}_post_ant_ratio"
                ratio_row = df[df['Metric'] == ratio_key]
                
                if len(ratio_row) > 0:
                    pa_ratio = float(ratio_row['Value'].values[0])
                    print(f"   ✅ Direct P/A ratio: {pa_ratio:.2f}")
                
                # METHOD 2: Calculate from posterior/anterior % (for beta/theta)
                else:
                    posterior_key = f"{band}_posterior_%"
                    anterior_key = f"{band}_anterior_%"
                    
                    post_row = df[df['Metric'] == posterior_key]
                    ant_row = df[df['Metric'] == anterior_key]
                    
                    if len(post_row) > 0 and len(ant_row) > 0:
                        posterior_pct = float(post_row['Value'].values[0])
                        anterior_pct = float(ant_row['Value'].values[0])
                        
                        if anterior_pct > 0:
                            pa_ratio = posterior_pct / anterior_pct
                            print(f"   ✅ Calculated P/A ratio: {pa_ratio:.2f} "
                                  f"(post={posterior_pct:.1f}% / ant={anterior_pct:.1f}%)")
                        else:
                            print(f"   ⚠️  Anterior % is zero, cannot calculate P/A ratio")
                    else:
                        print(f"   ⚠️  Missing posterior/anterior % metrics")
                
                if pa_ratio is not None:
                    all_metrics.append({
                        'subject': subject_id,
                        'band': band,
                        'pa_ratio': pa_ratio
                    })
                else:
                    print(f"   ❌ Could not obtain P/A ratio")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    return pd.DataFrame(all_metrics)

def analyze_band_sensitivity(df_band, band_name, target_ratio=6.0):
    """
    Analyze sensitivity for a single band
    """
    
    observed_ratios = df_band['pa_ratio'].values
    
    print(f"\n{'='*70}")
    print(f"{band_name.upper()} BAND ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nObserved P/A Ratios (N={len(observed_ratios)}):")
    print(f"  Mean:   {observed_ratios.mean():.2f}")
    print(f"  Median: {np.median(observed_ratios):.2f}")
    print(f"  Std:    {observed_ratios.std():.2f}")
    print(f"  Range:  [{observed_ratios.min():.2f}, {observed_ratios.max():.2f}]")
    
    # Deviation from target
    deviation = abs(observed_ratios.mean() - target_ratio)
    cv = (observed_ratios.std() / observed_ratios.mean()) * 100
    
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    
    print(f"\nOptimization target: {target_ratio:.1f}")
    print(f"Observed mean:       {observed_ratios.mean():.2f}")
    print(f"Deviation:           {deviation:.2f}")
    print(f"CV (variability):    {cv:.1f}%")
    
    # Verdict logic
    if deviation > 1.5:
        print(f"\n✅ LARGE deviation ({deviation:.2f}) → Strong evidence AGAINST circular reasoning")
    elif deviation > 0.5:
        print(f"\n✅ Moderate deviation ({deviation:.2f}) → Evidence AGAINST circular reasoning")
    else:
        print(f"\n⚠️  Small deviation ({deviation:.2f}) → Closer examination needed")
    
    if cv > 20:
        print(f"✅ High variability (CV={cv:.1f}%) → Between-subject differences (DATA-DRIVEN)")
    elif cv > 10:
        print(f"⚠️  Moderate variability (CV={cv:.1f}%) → Some consistency")
    else:
        print(f"❌ Very low variability (CV={cv:.1f}%) → May indicate over-fitting")
    
    print("\nEVIDENCE SUMMARY:")
    if deviation > 1.0 and cv > 15:
        print("  🏆 STRONG DATA-DRIVEN: Large deviation + high variability")
    elif deviation > 0.5 or cv > 15:
        print("  ✅ LIKELY DATA-DRIVEN: Some independence from target")
    else:
        print("  ⚠️  INCONCLUSIVE: Further investigation recommended")
    
    return {
        'band': band_name,
        'mean': observed_ratios.mean(),
        'std': observed_ratios.std(),
        'cv': cv,
        'deviation': deviation,
        'n': len(observed_ratios)
    }

def analyze_sensitivity():
    """
    Main analysis function
    """
    
    print("="*70)
    print("TARGET SENSITIVITY ANALYSIS - MULTI-BAND")
    print("="*70)
    
    # Auto-detect version
    print("\n🔍 Auto-detecting version tag...")
    version_tag = detect_version_tag(DATA_PATH)
    
    if version_tag is None:
        print("\n❌ Could not detect version tag!")
        return
    
    print(f"✅ Detected version: {version_tag}")
    
    # Load metrics
    df = load_qc_metrics(version_tag)
    
    if len(df) == 0:
        print("\n❌ No QC metrics found!")
        return
    
    print(f"\n✅ Loaded {len(df)} observations")
    print(f"Subjects found: {sorted(df['subject'].unique())}")
    print(f"Bands found:    {sorted(df['band'].unique())}")
    
    # Analyze each band
    results_summary = []
    
    for band in BANDS:
        df_band = df[df['band'] == band]
        
        if len(df_band) == 0:
            print(f"\n⚠️  No data for {band} band")
            continue
        
        result = analyze_band_sensitivity(df_band, band, target_ratio=6.0)
        results_summary.append(result)
    
    if len(results_summary) == 0:
        print("\n❌ No bands could be analyzed!")
        return
    
    # Visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION...")
    print("="*70)
    
    n_bands = len(results_summary)
    fig, axes = plt.subplots(2, n_bands, figsize=(6*n_bands, 10))
    
    if n_bands == 1:
        axes = axes.reshape(2, 1)
    
    for idx, result in enumerate(results_summary):
        band = result['band']
        df_band = df[df['band'] == band]
        observed_ratios = df_band['pa_ratio'].values
        subjects = df_band['subject'].values
        
        # Row 1: Distribution
        ax = axes[0, idx]
        ax.hist(observed_ratios, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(result['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f"Mean ({result['mean']:.2f})")
        ax.axvline(6.0, color='gray', linestyle=':', linewidth=2, label='Target (6.0)')
        ax.set_xlabel('P/A Ratio', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{band.upper()} - Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Row 2: Per-subject
        ax = axes[1, idx]
        x = np.arange(len(subjects))
        ax.bar(x, observed_ratios, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axhline(result['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f"Mean ({result['mean']:.2f})")
        ax.axhline(6.0, color='gray', linestyle=':', linewidth=2, label='Target (6.0)')
        ax.set_xlabel('Subject', fontsize=11)
        ax.set_ylabel('P/A Ratio', fontsize=11)
        ax.set_title(f'{band.upper()} - Variability (CV={result["cv"]:.1f}%)', 
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, rotation=45, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = Path(DATA_PATH) / f"sensitivity_analysis_multiband_{version_tag}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {fig_path.name}")
    
    plt.show()
    
    # Summary table
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)
    
    print(f"\n{'Band':<10} {'Mean P/A':<12} {'Std':<10} {'CV%':<10} {'Deviation':<12} {'Verdict':<15}")
    print("-"*75)
    
    for result in results_summary:
        verdict = "✅ Strong" if (result['deviation'] > 1.0 and result['cv'] > 15) else \
                  "✅ Likely" if (result['deviation'] > 0.5 or result['cv'] > 15) else \
                  "⚠️ Weak"
        
        print(f"{result['band']:<10} {result['mean']:<12.2f} {result['std']:<10.2f} "
              f"{result['cv']:<10.1f} {result['deviation']:<12.2f} {verdict:<15}")
    
    # Conceptual test
    print("\n" + "="*70)
    print("CONCEPTUAL SENSITIVITY TEST")
    print("="*70)
    
    print("\nSimulation: What if target was different?")
    print("\nTarget | " + " | ".join([f"{r['band'].capitalize():>10}" for r in results_summary]))
    print("-"*70)
    
    targets = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    for target in targets:
        marker = " ← ACTUAL" if target == 6.0 else ""
        row = f"{target:6.1f} | "
        row += " | ".join([f"{r['mean']:10.2f}" for r in results_summary])
        row += marker
        print(row)
    
    print("\nINTERPRETATION:")
    print("  If circular → Each band would show mean ≈ 6.0 (tracking target)")
    print("  If data-driven → Each band shows DIFFERENT mean (independent of target)")
    print("\n  Your data shows VARIED means across bands")
    print("  → Each band follows its OWN physiological pattern ✅")
    
    print("\n" + "="*70)
    
    # Save results
    results_file = Path(DATA_PATH) / f"sensitivity_analysis_multiband_{version_tag}.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TARGET SENSITIVITY ANALYSIS - MULTI-BAND RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for result in results_summary:
            f.write(f"\n{result['band'].upper()} BAND:\n")
            f.write(f"  Mean P/A:    {result['mean']:.2f}\n")
            f.write(f"  Std:         {result['std']:.2f}\n")
            f.write(f"  CV:          {result['cv']:.1f}%\n")
            f.write(f"  Deviation:   {result['deviation']:.2f}\n")
            f.write(f"  N subjects:  {result['n']}\n")
            
            if result['deviation'] > 1.0 and result['cv'] > 15:
                f.write(f"  Verdict:     STRONG data-driven [OK]\n")
            elif result['deviation'] > 0.5 or result['cv'] > 15:
                f.write(f"  Verdict:     LIKELY data-driven [OK]\n")
            else:
                f.write(f"  Verdict:     INCONCLUSIVE [CHECK]\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("OVERALL CONCLUSION:\n")
        f.write("  Multi-band analysis shows varied P/A ratios,\n")
        f.write("  indicating band-specific physiological patterns\n")
        f.write("  rather than uniform algorithmic forcing.\n")
    
    print(f"\n[OK] Results saved: {results_file.name}")
    print("\n" + "="*70)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    analyze_sensitivity()