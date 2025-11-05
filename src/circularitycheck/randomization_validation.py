# randomization_validation_FINAL.py
"""
FINAL VERSION: Randomization Test with Posterior Fold-Enrichment Metric
========================================================================
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = r"C:\Users\kerem\Downloads\eegyedek\LOSO"
VERSION_TAG = "v34"

SUBJECTS = ["S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11"]
BANDS = ["delta", "gamma"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_volume(subject_id, band_name):
    """Load existing volume"""
    nii_path = Path(DATA_PATH) / f"{subject_id}_{band_name}_voxel_{VERSION_TAG}.nii.gz"
    
    if not nii_path.exists():
        print(f"⚠️  Volume not found: {nii_path.name}")
        return None
    
    img = nib.load(nii_path)
    volume = img.get_fdata()
    affine = img.affine
    
    return volume, affine

def load_gm_mask():
    """
    Load gray matter mask from Harvard-Oxford atlas.
    Must match your grid dimensions.
    """
    from nilearn import datasets, image
    
    # Load first volume to get dimensions
    first_vol, first_affine = load_volume(SUBJECTS[0], BANDS[0])
    target_shape = first_vol.shape[:3]
    
    # Create target image
    target_img = nib.Nifti1Image(np.zeros(target_shape, dtype=np.int16), first_affine)
    
    # Load Harvard-Oxford cortical atlas
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    cort_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
    
    # Resample to match volumes
    cort_resampled = image.resample_to_img(cort_img, target_img, interpolation='nearest')
    gm_mask = cort_resampled.get_fdata() > 0
    
    print(f"✅ GM mask loaded: {gm_mask.sum()} voxels")
    
    return gm_mask

def compute_pa_ratio_from_volume(volume, affine):
    """Compute Posterior/Anterior ratio"""
    if volume.ndim == 3:
        volume = volume[..., np.newaxis]
    
    x, y_dim, z, T = volume.shape
    
    yi = np.arange(y_dim)
    Y_coords = (affine @ np.vstack([
        np.zeros_like(yi), yi, np.zeros_like(yi), np.ones_like(yi)
    ]))[1, :]
    
    Y_grid = Y_coords[np.newaxis, :, np.newaxis]
    Y_grid = np.broadcast_to(Y_grid, (x, y_dim, z))
    
    posterior_mask = (Y_grid < -40)
    anterior_mask = (Y_grid > 0)
    
    V = volume.copy()
    V[V < 0] = 0
    
    if V.ndim == 4:
        V_sum = V.sum(axis=3)
    else:
        V_sum = V.squeeze(-1)
    
    posterior_sum = V_sum[posterior_mask].sum()
    anterior_sum = V_sum[anterior_mask].sum()
    
    ratio = posterior_sum / (anterior_sum + 1e-12)
    
    return ratio

def compute_posterior_enrichment(volume, affine, gm_mask):
    """
    ✅ NEW: Compute posterior fold-enrichment (density-normalized).
    
    This metric accounts for different GM sizes in regions.
    Returns how many TIMES more dense posterior is vs anterior.
    """
    if volume.ndim == 3:
        volume = volume[..., np.newaxis]
    
    x, y_dim, z, T = volume.shape
    
    # Get Y coordinates
    yi = np.arange(y_dim)
    Y_coords = (affine @ np.vstack([
        np.zeros_like(yi), yi, np.zeros_like(yi), np.ones_like(yi)
    ]))[1, :]
    
    Y_grid = Y_coords[np.newaxis, :, np.newaxis]
    Y_grid = np.broadcast_to(Y_grid, (x, y_dim, z))
    
    # Define masks (AND with GM)
    posterior_mask = (Y_grid < -40) & gm_mask
    anterior_mask = (Y_grid > 0) & gm_mask
    
    # Sum activation
    V = volume.copy()
    V[V < 0] = 0
    
    if V.ndim == 4:
        V_sum = V.sum(axis=3)
    else:
        V_sum = V.squeeze(-1)
    
    # Compute sums
    posterior_sum = V_sum[posterior_mask].sum()
    anterior_sum = V_sum[anterior_mask].sum()
    
    # Normalize by GM size (voxel count)
    posterior_gm = posterior_mask.sum()
    anterior_gm = anterior_mask.sum()
    
    # Density = activation per voxel
    posterior_density = posterior_sum / (posterior_gm + 1e-12)
    anterior_density = anterior_sum / (anterior_gm + 1e-12)
    
    # Fold-enrichment
    enrichment = posterior_density / (anterior_density + 1e-12)
    
    return enrichment

def randomize_volume_coordinates(volume, affine):
    """Destroy anatomical structure by spatial shuffling"""
    if volume.ndim == 3:
        volume = volume[..., np.newaxis]
    
    x, y, z, T = volume.shape
    
    volume_flat = volume.reshape(-1, T)
    np.random.shuffle(volume_flat)
    volume_randomized = volume_flat.reshape(x, y, z, T)
    
    if T == 1:
        volume_randomized = volume_randomized.squeeze(-1)
    
    return volume_randomized

# ============================================================================
# MAIN TEST
# ============================================================================

def run_randomization_test():
    """
    ✅ FINAL VERSION: Test with both P/A ratio AND fold-enrichment
    """
    
    print("="*70)
    print("RANDOMIZATION TEST - FINAL VERSION")
    print("="*70)
    print(f"\nData path: {DATA_PATH}")
    print(f"Subjects: {len(SUBJECTS)}")
    print(f"Bands: {BANDS}")
    print()
    
    # Load GM mask (needed for enrichment)
    print("Loading gray matter mask...")
    gm_mask = load_gm_mask()
    
    # Storage - ✅ NOW includes enrichment
    real_results = {
        band: {'pa_ratios': [], 'enrichments': []} 
        for band in BANDS
    }
    random_results = {
        band: {'pa_ratios': [], 'enrichments': []} 
        for band in BANDS
    }
    
    # ========================================================================
    # STEP 1: REAL DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: Computing metrics from REAL volumes...")
    print("="*70)
    
    for subject_id in SUBJECTS:
        print(f"\n  Subject: {subject_id}")
        
        for band in BANDS:
            result = load_volume(subject_id, band)
            
            if result is None:
                print(f"    {band:8s}: SKIP")
                continue
            
            volume, affine = result
            
            # ✅ Compute BOTH metrics
            pa_ratio = compute_pa_ratio_from_volume(volume, affine)
            enrichment = compute_posterior_enrichment(volume, affine, gm_mask)
            
            real_results[band]['pa_ratios'].append(pa_ratio)
            real_results[band]['enrichments'].append(enrichment)
            
            print(f"    {band:8s}: P/A={pa_ratio:.2f}, Enrichment={enrichment:.2f}x")
    
    # ========================================================================
    # STEP 2: RANDOMIZED DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: Computing metrics from RANDOMIZED volumes...")
    print("="*70)
    print("(Destroying anatomical structure)")
    
    n_randomizations = 10
    
    for iteration in range(n_randomizations):
        print(f"\n  Randomization {iteration+1}/{n_randomizations}")
        
        for subject_id in SUBJECTS:
            for band in BANDS:
                result = load_volume(subject_id, band)
                
                if result is None:
                    continue
                
                volume, affine = result
                
                # ✅ RANDOMIZE
                volume_random = randomize_volume_coordinates(volume, affine)
                
                # ✅ Compute BOTH metrics on randomized
                pa_ratio_rand = compute_pa_ratio_from_volume(volume_random, affine)
                enrichment_rand = compute_posterior_enrichment(volume_random, affine, gm_mask)
                
                random_results[band]['pa_ratios'].append(pa_ratio_rand)
                random_results[band]['enrichments'].append(enrichment_rand)
    
    # ========================================================================
    # STEP 3: STATISTICAL COMPARISON
    # ========================================================================
    
    print("\n" + "="*70)
    print("STATISTICAL RESULTS")
    print("="*70)
    
    for band in BANDS:
        print(f"\n{'='*70}")
        print(f"{band.upper()} BAND")
        print(f"{'='*70}")
        
        # ─────────────────────────────────────────────────────────────────
        # P/A Ratio (same as before)
        # ─────────────────────────────────────────────────────────────────
        real_pa = np.array(real_results[band]['pa_ratios'])
        rand_pa = np.array(random_results[band]['pa_ratios'])
        
        if len(real_pa) > 0 and len(rand_pa) > 0:
            U_pa, p_pa = mannwhitneyu(real_pa, rand_pa, alternative='greater')
            cohen_d_pa = (real_pa.mean() - rand_pa.mean()) / np.sqrt(
                (real_pa.std()**2 + rand_pa.std()**2) / 2
            )
            
            print(f"\nPosterior/Anterior Ratio:")
            print(f"  Real Data:      {real_pa.mean():.2f} ± {real_pa.std():.2f}  (N={len(real_pa)})")
            print(f"  Randomized:     {rand_pa.mean():.2f} ± {rand_pa.std():.2f}  (N={len(rand_pa)})")
            print(f"  Mann-Whitney U: {U_pa:.1f}")
            print(f"  p-value:        {p_pa:.6f}  {'***' if p_pa < 0.001 else '**' if p_pa < 0.01 else '*' if p_pa < 0.05 else 'n.s.'}")
            print(f"  Cohen's d:      {cohen_d_pa:.2f}  ({'huge' if abs(cohen_d_pa) > 4 else 'large' if abs(cohen_d_pa) > 0.8 else 'medium' if abs(cohen_d_pa) > 0.5 else 'small'})")
        
        # ─────────────────────────────────────────────────────────────────
        # ✅ FOLD-ENRICHMENT (NEW - replaces occipital %)
        # ─────────────────────────────────────────────────────────────────
        real_enr = np.array(real_results[band]['enrichments'])
        rand_enr = np.array(random_results[band]['enrichments'])
        
        if len(real_enr) > 0 and len(rand_enr) > 0:
            U_enr, p_enr = mannwhitneyu(real_enr, rand_enr, alternative='greater')
            cohen_d_enr = (real_enr.mean() - rand_enr.mean()) / np.sqrt(
                (real_enr.std()**2 + rand_enr.std()**2) / 2
            )
            
            print(f"\nPosterior Fold-Enrichment (Density Ratio):")
            print(f"  Real Data:      {real_enr.mean():.2f}x ± {real_enr.std():.2f}x  (N={len(real_enr)})")
            print(f"  Randomized:     {rand_enr.mean():.2f}x ± {rand_enr.std():.2f}x  (N={len(rand_enr)})")
            print(f"  Mann-Whitney U: {U_enr:.1f}")
            print(f"  p-value:        {p_enr:.6f}  {'***' if p_enr < 0.001 else '**' if p_enr < 0.01 else '*' if p_enr < 0.05 else 'n.s.'}")
            print(f"  Cohen's d:      {cohen_d_enr:.2f}  ({'huge' if abs(cohen_d_enr) > 4 else 'large' if abs(cohen_d_enr) > 0.8 else 'medium' if abs(cohen_d_enr) > 0.5 else 'small'})")
            print(f"\n  Interpretation:")
            if real_enr.mean() > 1.5:
                print(f"    ✅ Posterior is {real_enr.mean():.1f}x MORE DENSE than anterior (anatomically selective)")
            elif real_enr.mean() > 1.1:
                print(f"    ⚠️  Posterior is {real_enr.mean():.1f}x more dense (weak selectivity)")
            else:
                print(f"    ❌ Posterior NOT enriched (uniform distribution)")
    
    # ========================================================================
    # STEP 4: VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION...")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: P/A Ratio
    ax = axes[0]
    x = np.arange(len(BANDS))
    width = 0.35
    
    real_pas = [np.mean(real_results[b]['pa_ratios']) for b in BANDS]
    rand_pas = [np.mean(random_results[b]['pa_ratios']) for b in BANDS]
    real_pas_err = [np.std(real_results[b]['pa_ratios']) for b in BANDS]
    rand_pas_err = [np.std(random_results[b]['pa_ratios']) for b in BANDS]
    
    ax.bar(x - width/2, real_pas, width, yerr=real_pas_err, 
           label='Real Anatomy', color='steelblue', capsize=5)
    ax.bar(x + width/2, rand_pas, width, yerr=rand_pas_err, 
           label='Randomized', color='coral', alpha=0.7, capsize=5)
    
    ax.set_ylabel('Posterior/Anterior Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency Band', fontsize=12)
    ax.set_title('Randomization Test: P/A Ratio', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in BANDS])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add significance stars
    for i, band in enumerate(BANDS):
        real_pa = np.array(real_results[band]['pa_ratios'])
        rand_pa = np.array(random_results[band]['pa_ratios'])
        
        if len(real_pa) > 0 and len(rand_pa) > 0:
            _, p = mannwhitneyu(real_pa, rand_pa, alternative='greater')
            
            y_max = max(real_pas[i] + real_pas_err[i], rand_pas[i] + rand_pas_err[i])
            
            if p < 0.001:
                ax.text(i, y_max * 1.1, '***', ha='center', fontsize=16, fontweight='bold')
            elif p < 0.01:
                ax.text(i, y_max * 1.1, '**', ha='center', fontsize=16)
            elif p < 0.05:
                ax.text(i, y_max * 1.1, '*', ha='center', fontsize=16)
    
    # ✅ Plot 2: FOLD-ENRICHMENT (replaces occipital %)
    ax = axes[1]
    
    real_enrs = [np.mean(real_results[b]['enrichments']) for b in BANDS]
    rand_enrs = [np.mean(random_results[b]['enrichments']) for b in BANDS]
    real_enrs_err = [np.std(real_results[b]['enrichments']) for b in BANDS]
    rand_enrs_err = [np.std(random_results[b]['enrichments']) for b in BANDS]
    
    ax.bar(x - width/2, real_enrs, width, yerr=real_enrs_err, 
           label='Real Anatomy', color='steelblue', capsize=5)
    ax.bar(x + width/2, rand_enrs, width, yerr=rand_enrs_err, 
           label='Randomized', color='coral', alpha=0.7, capsize=5)
    
    ax.set_ylabel('Posterior Fold-Enrichment (×)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency Band', fontsize=12)
    ax.set_title('Randomization Test: Posterior Enrichment', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in BANDS])
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Uniform (1.0x)')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add significance stars
    for i, band in enumerate(BANDS):
        real_enr = np.array(real_results[band]['enrichments'])
        rand_enr = np.array(random_results[band]['enrichments'])
        
        if len(real_enr) > 0 and len(rand_enr) > 0:
            _, p = mannwhitneyu(real_enr, rand_enr, alternative='greater')
            
            y_max = max(real_enrs[i] + real_enrs_err[i], rand_enrs[i] + rand_enrs_err[i])
            
            if p < 0.001:
                ax.text(i, y_max * 1.1, '***', ha='center', fontsize=16, fontweight='bold')
            elif p < 0.01:
                ax.text(i, y_max * 1.1, '**', ha='center', fontsize=16)
            elif p < 0.05:
                ax.text(i, y_max * 1.1, '*', ha='center', fontsize=16)
    
    plt.tight_layout()
    
    fig_path = Path(DATA_PATH) / f"randomization_test_FINAL_{VERSION_TAG}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {fig_path.name}")
    
    plt.show()
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    
    print("\n" + "="*70)
    print("VERDICT: CIRCULAR REASONING TEST")
    print("="*70)
    
    all_significant = True
    verdict_details = []
    
    for band in BANDS:
        real_pa = np.array(real_results[band]['pa_ratios'])
        rand_pa = np.array(random_results[band]['pa_ratios'])
        real_enr = np.array(real_results[band]['enrichments'])
        rand_enr = np.array(random_results[band]['enrichments'])
        
        if len(real_pa) > 0 and len(rand_pa) > 0:
            _, p_pa = mannwhitneyu(real_pa, rand_pa, alternative='greater')
            _, p_enr = mannwhitneyu(real_enr, rand_enr, alternative='greater')
            
            # Both metrics should be significant
            if p_pa >= 0.05 or p_enr >= 0.05:
                all_significant = False
            
            verdict_details.append({
                'band': band,
                'p_pa': p_pa,
                'p_enr': p_enr,
                'pa_pass': p_pa < 0.05,
                'enr_pass': p_enr < 0.05
            })
    
    # Print detailed verdict
    print(f"\nBand-by-band results:")
    print(f"{'Band':<8} {'P/A Ratio p':<15} {'Enrichment p':<15} {'Status':<15}")
    print("-"*60)
    
    for v in verdict_details:
        pa_symbol = '✅' if v['pa_pass'] else '❌'
        enr_symbol = '✅' if v['enr_pass'] else '❌'
        status = 'PASS' if (v['pa_pass'] and v['enr_pass']) else 'FAIL'
        
        print(f"{v['band']:<8} {v['p_pa']:<15.6f} {v['p_enr']:<15.6f} {pa_symbol} {enr_symbol} {status}")
    
    print("\n" + "="*70)
    
    if all_significant:
        print("\n✅ ✅ ✅  PASSED  ✅ ✅ ✅")
        print("\nALL bands show BOTH metrics significant (p < 0.05).")
        print("Physiological patterns COLLAPSE when anatomy is randomized.")
        print("\nCONCLUSION:")
        print("  → Patterns are ANATOMICALLY GROUNDED (data-driven)")
        print("  → NOT circular reasoning (algorithmic artifacts)")
        print("  → Method is VALID for publication")
    else:
        print("\n⚠️  MIXED RESULTS")
        print("\nSome metrics not significant. Review details above.")
        print("\nPOSSIBLE INTERPRETATIONS:")
        print("  1. Pipeline needs optimization (e.g., sparsification too aggressive)")
        print("  2. Some bands naturally less spatially selective")
        print("  3. Sample size (N=10) may be limiting power for some comparisons")
    
    print("\n" + "="*70)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    results_file = Path(DATA_PATH) / f"randomization_test_results_FINAL_{VERSION_TAG}.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RANDOMIZATION TEST RESULTS (FINAL)\n")
        f.write("="*70 + "\n\n")
        
        for band in BANDS:
            f.write(f"\n{band.upper()} BAND\n")
            f.write("-"*70 + "\n")
            
            real_pa = np.array(real_results[band]['pa_ratios'])
            rand_pa = np.array(random_results[band]['pa_ratios'])
            real_enr = np.array(real_results[band]['enrichments'])
            rand_enr = np.array(random_results[band]['enrichments'])
            
            if len(real_pa) > 0 and len(rand_pa) > 0:
                U_pa, p_pa = mannwhitneyu(real_pa, rand_pa, alternative='greater')
                cohen_d_pa = (real_pa.mean() - rand_pa.mean()) / np.sqrt(
                    (real_pa.std()**2 + rand_pa.std()**2) / 2
                )
                
                f.write(f"\nP/A Ratio:\n")
                f.write(f"  Real:       {real_pa.mean():.2f} +/- {real_pa.std():.2f}\n")
                f.write(f"  Randomized: {rand_pa.mean():.2f} +/- {rand_pa.std():.2f}\n")
                f.write(f"  p-value:    {p_pa:.6f}\n")
                f.write(f"  Cohen's d:  {cohen_d_pa:.2f}\n")
            
            if len(real_enr) > 0 and len(rand_enr) > 0:
                U_enr, p_enr = mannwhitneyu(real_enr, rand_enr, alternative='greater')
                cohen_d_enr = (real_enr.mean() - rand_enr.mean()) / np.sqrt(
                    (real_enr.std()**2 + rand_enr.std()**2) / 2
                )
                
                f.write(f"\nFold-Enrichment:\n")
                f.write(f"  Real:       {real_enr.mean():.2f}x +/- {real_enr.std():.2f}x\n")
                f.write(f"  Randomized: {rand_enr.mean():.2f}x +/- {rand_enr.std():.2f}x\n")
                f.write(f"  p-value:    {p_enr:.6f}\n")
                f.write(f"  Cohen's d:  {cohen_d_enr:.2f}\n")
    
    print(f"\n✅ Results saved: {results_file.name}")
    print("\n" + "="*70)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    run_randomization_test()