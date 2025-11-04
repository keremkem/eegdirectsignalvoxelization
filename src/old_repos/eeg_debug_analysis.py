# eeg_debug_analysis.py

import numpy as np
import scipy.io
import nibabel as nib
from scipy.spatial.distance import cdist
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

# === PARAMETRELER === #
TRUE_DIPOLES = np.array([
    [0, -50, 30],    # Parietal/Posterior bölge
    [30, 20, 50]     # Frontal/Motor korteks
])

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

# === YARDIMCI FONKSİYONLAR === #
def log(msg):
    """Zaman damgalı log mesajı"""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def validate_and_transform_coords(coord_file):
    """Kanal koordinatlarını oku ve kontrol et"""
    coords = {}
    with open(coord_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                name, x, y, z = parts
                x, y, z = float(x), float(y), float(z)
                coords[name] = np.array([x, y, z])
    
    positions = np.array(list(coords.values()))
    log(f"Kanal koordinat aralığı:")
    log(f"  X: [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}]")
    log(f"  Y: [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]")
    log(f"  Z: [{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")
    
    radii = np.sqrt(np.sum(positions**2, axis=1))
    log(f"  Ortalama yarıçap: {radii.mean():.1f}mm (min: {radii.min():.1f}, max: {radii.max():.1f})")
    
    return coords

# === 1. VERİ KALİTE KONTROLÜ === #
def validate_simulation_data(eeg_mat_file, coord_file, true_dipoles):
    """Simülasyon verisinin doğruluğunu kontrol et"""
    log("\n🔍 VERİ KALİTE KONTROLÜ")
    log("="*60)
    
    # Mat dosyasını yükle
    mat = scipy.io.loadmat(eeg_mat_file)
    
    # Veri yapısını kontrol et
    log("📋 MAT Dosyası İçeriği:")
    for key in mat.keys():
        if not key.startswith('__'):
            if hasattr(mat[key], 'shape'):
                log(f"  {key}: {type(mat[key]).__name__}, shape: {mat[key].shape}")
            else:
                log(f"  {key}: {type(mat[key]).__name__}")
    
    # EEG verisi
    eeg_data = mat['dataRest'][:64] if 'dataRest' in mat else mat[list(mat.keys())[0]]
    log(f"\n📊 EEG Veri Özeti:")
    log(f"  Shape: {eeg_data.shape}")
    log(f"  Min/Max: {np.min(eeg_data):.2f} / {np.max(eeg_data):.2f}")
    log(f"  Mean/Std: {np.mean(eeg_data):.2f} / {np.std(eeg_data):.2f}")
    
    # Kanal isimleri kontrolü
    if 'chanlocs' in mat:
        loaded_channels = mat['chanlocs'].flatten()
        log(f"\n📍 Yüklenen kanal sayısı: {len(loaded_channels)}")
        log(f"  İlk 5 kanal: {loaded_channels[:5]}")
    
    # Koordinat dosyasını kontrol et
    channel_coords = validate_and_transform_coords(coord_file)
    
    return eeg_data, channel_coords

# === 2. KANAL AKTİVİTE ANALİZİ === #
def analyze_channel_activity(eeg_data, channel_coords, true_dipoles):
    """Hangi kanalların aktif olduğunu analiz et"""
    log("\n📊 KANAL AKTİVİTE ANALİZİ")
    log("="*60)
    
    ch_positions = np.array(list(channel_coords.values()))
    ch_names = list(channel_coords.keys())
    
    # Kanal gücü hesapla
    channel_power = np.mean(eeg_data**2, axis=1)
    
    # En aktif kanallar
    top_10_idx = np.argsort(channel_power)[-10:]
    log("\n🔥 En aktif 10 kanal:")
    for idx in reversed(top_10_idx):
        log(f"  {ch_names[idx]:>4}: Power={channel_power[idx]:>8.2f}, Pos=[{ch_positions[idx,0]:>6.1f}, {ch_positions[idx,1]:>6.1f}, {ch_positions[idx,2]:>6.1f}]")
    
    # Dipollere yakın kanallar
    log("\n🎯 Dipollere en yakın kanalların aktivitesi:")
    for i, dipole in enumerate(true_dipoles):
        distances = np.linalg.norm(ch_positions - dipole, axis=1)
        nearest_5 = np.argsort(distances)[:5]
        
        log(f"\nDipol {i+1} [{dipole[0]:.0f}, {dipole[1]:.0f}, {dipole[2]:.0f}]:")
        for idx in nearest_5:
            power_rank = len(channel_power) - np.where(np.argsort(channel_power) == idx)[0][0]
            log(f"  {ch_names[idx]:>4} ({distances[idx]:>5.1f}mm): Power={channel_power[idx]:>8.2f} (Rank: {power_rank}/{len(channel_power)})")
    
    return channel_power

# === 3. GÖRSELLEŞTIRME === #
def visualize_activity_distribution(eeg_data, channel_coords, true_dipoles, output_prefix="debug"):
    """Aktivite dağılımını görselleştir"""
    log("\n📈 GÖRSELLEŞTIRME")
    log("="*60)
    
    ch_positions = np.array(list(channel_coords.values()))
    ch_names = list(channel_coords.keys())
    channel_power = np.mean(eeg_data**2, axis=1)
    
    # Normalize power for visualization
    power_norm = (channel_power - np.min(channel_power)) / (np.max(channel_power) - np.min(channel_power))
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(ch_positions[:,0], ch_positions[:,1], ch_positions[:,2], 
                         c=power_norm, s=power_norm*300+50, cmap='hot', alpha=0.7)
    ax1.scatter(true_dipoles[:,0], true_dipoles[:,1], true_dipoles[:,2], 
               c='blue', s=500, marker='*', edgecolors='black', linewidth=2, label='True Dipoles')
    
    # En aktif kanalları etiketle
    top_5_idx = np.argsort(channel_power)[-5:]
    for idx in top_5_idx:
        ax1.text(ch_positions[idx,0], ch_positions[idx,1], ch_positions[idx,2], 
                ch_names[idx], fontsize=10, weight='bold')
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Kanal Aktivite Dağılımı')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Normalized Power')
    
    # 2. Üstten görünüm (XY)
    ax2 = fig.add_subplot(222)
    scatter2 = ax2.scatter(ch_positions[:,0], ch_positions[:,1], 
                          c=power_norm, s=300, cmap='hot', alpha=0.7, edgecolors='gray')
    ax2.scatter(true_dipoles[:,0], true_dipoles[:,1], 
               c='blue', s=500, marker='*', edgecolors='black', linewidth=2)
    
    # Kanal isimlerini ekle
    for i, name in enumerate(ch_names):
        ax2.annotate(name, (ch_positions[i,0], ch_positions[i,1]), 
                    fontsize=8, ha='center', va='center')
    
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Üstten Görünüm (XY)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Normalized Power')
    
    # 3. Yandan görünüm (XZ)
    ax3 = fig.add_subplot(223)
    scatter3 = ax3.scatter(ch_positions[:,0], ch_positions[:,2], 
                          c=power_norm, s=300, cmap='hot', alpha=0.7, edgecolors='gray')
    ax3.scatter(true_dipoles[:,0], true_dipoles[:,2], 
               c='blue', s=500, marker='*', edgecolors='black', linewidth=2)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('Yandan Görünüm (XZ)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Normalized Power')
    
    # 4. Mesafe vs Power grafiği
    ax4 = fig.add_subplot(224)
    for i, dipole in enumerate(true_dipoles):
        distances = np.linalg.norm(ch_positions - dipole, axis=1)
        ax4.scatter(distances, channel_power, alpha=0.6, label=f'Dipole {i+1}')
        
        # Trend line
        z = np.polyfit(distances, channel_power, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(min(distances), max(distances), 100)
        ax4.plot(x_trend, p(x_trend), "--", alpha=0.8)
    
    ax4.set_xlabel('Mesafe (mm)')
    ax4.set_ylabel('Kanal Gücü')
    ax4.set_title('Mesafe-Güç İlişkisi')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_activity_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    log(f"Görselleştirme kaydedildi: {output_prefix}_activity_distribution.png")

# === 4. BASİT LOKALİZASYON TESTLERİ === #
def simple_localization_tests(eeg_data, channel_coords, true_dipoles):
    """Basit lokalizasyon yaklaşımlarını test et"""
    log("\n🧪 BASİT LOKALİZASYON TESTLERİ")
    log("="*60)
    
    ch_positions = np.array(list(channel_coords.values()))
    ch_names = list(channel_coords.keys())
    channel_power = np.mean(eeg_data**2, axis=1)
    
    # Test 1: En aktif kanalın konumu
    log("\n📍 Test 1: En Aktif Kanal")
    max_idx = np.argmax(channel_power)
    max_pos = ch_positions[max_idx]
    errors1 = [np.linalg.norm(max_pos - dip) for dip in true_dipoles]
    log(f"  En aktif kanal: {ch_names[max_idx]} {max_pos}")
    log(f"  Min hata: {min(errors1):.1f}mm")
    
    # Test 2: En aktif 5 kanalın ağırlıklı ortalaması
    log("\n📍 Test 2: Top-5 Ağırlıklı Ortalama")
    top_5 = np.argsort(channel_power)[-5:]
    weights = channel_power[top_5] / np.sum(channel_power[top_5])
    weighted_pos = np.sum(ch_positions[top_5] * weights[:, np.newaxis], axis=0)
    errors2 = [np.linalg.norm(weighted_pos - dip) for dip in true_dipoles]
    log(f"  Ağırlıklı pozisyon: {weighted_pos}")
    log(f"  Min hata: {min(errors2):.1f}mm")
    
    # Test 3: Center of Mass
    log("\n📍 Test 3: Center of Mass")
    power_threshold = np.percentile(channel_power, 75)
    high_power_mask = channel_power > power_threshold
    com_weights = channel_power[high_power_mask] / np.sum(channel_power[high_power_mask])
    com_pos = np.sum(ch_positions[high_power_mask] * com_weights[:, np.newaxis], axis=0)
    errors3 = [np.linalg.norm(com_pos - dip) for dip in true_dipoles]
    log(f"  CoM pozisyon: {com_pos}")
    log(f"  Min hata: {min(errors3):.1f}mm")
    
    # Test 4: Triangulation (en aktif 3 kanal)
    log("\n📍 Test 4: Triangulation")
    top_3 = np.argsort(channel_power)[-3:]
    tri_weights = channel_power[top_3] / np.sum(channel_power[top_3])
    tri_pos = np.sum(ch_positions[top_3] * tri_weights[:, np.newaxis], axis=0)
    errors4 = [np.linalg.norm(tri_pos - dip) for dip in true_dipoles]
    log(f"  Triangulation pozisyon: {tri_pos}")
    log(f"  Min hata: {min(errors4):.1f}mm")
    
    return {
        'max_channel': min(errors1),
        'top5_weighted': min(errors2),
        'center_of_mass': min(errors3),
        'triangulation': min(errors4)
    }

# === 5. DİPOL FİTTİNG === #
def dipole_fitting_approach(eeg_data, channel_coords, n_dipoles=2, fs=256):
    """Klasik dipol fitting yaklaşımı"""
    log("\n🎯 DİPOL FİTTİNG")
    log("="*60)
    
    ch_positions = np.array(list(channel_coords.values()))
    
    # Ortalama güç kullan
    snapshot = np.mean(eeg_data**2, axis=1)
    
    def forward_model(dipole_pos, ch_positions):
        """Basit forward model - 1/r decay"""
        distances = np.linalg.norm(ch_positions - dipole_pos, axis=1)
        distances = np.maximum(distances, 10.0)  # Singularity önleme
        return 1.0 / distances
    
    def objective(params):
        """Objective function"""
        n_dip = len(params) // 3
        dipole_positions = params.reshape(n_dip, 3)
        
        # Forward model
        prediction = np.zeros(len(ch_positions))
        for i in range(n_dip):
            leadfield = forward_model(dipole_positions[i], ch_positions)
            prediction += leadfield
        
        # Normalize
        prediction = prediction / np.max(prediction)
        snapshot_norm = snapshot / np.max(snapshot)
        
        # MSE
        return np.sum((snapshot_norm - prediction) ** 2)
    
    # Birden fazla başlangıç noktası dene
    best_result = None
    best_error = np.inf
    
    log("Grid search ile optimizasyon...")
    for trial in range(10):
        if trial == 0:
            # İlk deneme: beklenen konumlar
            initial = np.array([[0, -50, 30], [30, 20, 50]]).flatten()
        else:
            # Random başlangıç
            initial = np.random.uniform(-60, 60, n_dipoles * 3)
            initial[2::3] = np.abs(initial[2::3])  # Z pozitif
        
        result = minimize(
            objective,
            initial,
            method='L-BFGS-B',
            bounds=[(-90, 90), (-90, 90), (0, 80)] * n_dipoles
        )
        
        if result.fun < best_error:
            best_error = result.fun
            best_result = result
    
    # Sonuçları değerlendir
    estimated_dipoles = best_result.x.reshape(n_dipoles, 3)
    
    log(f"\nTahmini dipoller:")
    for i, dip in enumerate(estimated_dipoles):
        log(f"  Dipol {i+1}: [{dip[0]:.1f}, {dip[1]:.1f}, {dip[2]:.1f}]")
    
    # Hata hesapla
    errors = []
    for est_dip in estimated_dipoles:
        min_error = min([np.linalg.norm(est_dip - true_dip) for true_dip in TRUE_DIPOLES])
        errors.append(min_error)
    
    log(f"\nHatalar: {errors}")
    log(f"Ortalama hata: {np.mean(errors):.1f}mm")
    
    return estimated_dipoles, errors

# === 6. SPATIAL CORRELATION ANALİZİ === #
def spatial_correlation_analysis(eeg_data, channel_coords, true_dipoles):
    """Spatial pattern correlation analizi"""
    log("\n📊 SPATIAL CORRELATION ANALİZİ")
    log("="*60)
    
    ch_positions = np.array(list(channel_coords.values()))
    snapshot = np.mean(eeg_data**2, axis=1)
    
    # Her dipol için beklenen pattern
    expected_patterns = []
    for i, dipole in enumerate(true_dipoles):
        distances = np.linalg.norm(ch_positions - dipole, axis=1)
        pattern = 1 / (distances + 20)
        pattern = pattern / np.max(pattern)
        expected_patterns.append(pattern)
        
        # Pattern correlation
        corr = np.corrcoef(snapshot, pattern)[0, 1]
        log(f"Dipol {i+1} pattern correlation: {corr:.3f}")
    
    # Combined pattern
    combined_pattern = np.max(expected_patterns, axis=0)
    combined_corr = np.corrcoef(snapshot, combined_pattern)[0, 1]
    log(f"Combined pattern correlation: {combined_corr:.3f}")
    
    return expected_patterns, combined_corr

# === 7. NIfTI DOSYA ANALİZİ === #
def analyze_nifti_results(band="delta"):
    """Pipeline çıktılarını analiz et"""
    log(f"\n📦 NIfTI DOSYA ANALİZİ - {band.upper()}")
    log("="*60)
    
    files = {
        'voxel': f"{band}_voxel_spm.nii",
        'surface': f"{band}_surface_projection.nii.gz"
    }
    
    results = {}
    
    for method, filename in files.items():
        if os.path.exists(filename):
            log(f"\n{method.upper()} analizi:")
            
            img = nib.load(filename)
            data = img.get_fdata()
            affine = img.affine
            
            if data.ndim == 4:
                data = data[..., 0]
            
            # İstatistikler
            active_voxels = np.sum(data > 0)
            log(f"  Aktif voxel sayısı: {active_voxels}")
            
            if active_voxels > 0:
                log(f"  Max değer: {np.max(data):.4f}")
                log(f"  Mean değer (aktif): {np.mean(data[data > 0]):.4f}")
                
                # En yüksek 5 aktivite
                coords = np.argwhere(data > 0)
                values = data[data > 0]
                top_5_idx = np.argsort(values)[-5:]
                
                log(f"\n  En yüksek 5 aktivite:")
                for i, idx in enumerate(reversed(top_5_idx)):
                    voxel_coord = coords[idx]
                    voxel_hom = np.append(voxel_coord, 1)
                    mni_coord = affine @ voxel_hom
                    mni_coord = mni_coord[:3]
                    
                    # Hata hesapla
                    errors = [np.linalg.norm(mni_coord - dip) for dip in TRUE_DIPOLES]
                    min_error = min(errors)
                    
                    log(f"    {i+1}. MNI: [{mni_coord[0]:>6.1f}, {mni_coord[1]:>6.1f}, {mni_coord[2]:>6.1f}], "
                        f"Değer: {values[idx]:.4f}, Min Hata: {min_error:.1f}mm")
                
                results[method] = {
                    'top_coords': [affine @ np.append(coords[idx], 1) for idx in top_5_idx[-2:]],
                    'active_voxels': active_voxels
                }
        else:
            log(f"\n{method.upper()}: Dosya bulunamadı ({filename})")
    
    return results

# === 8. KAPSAMLI TEST === #
def comprehensive_test(eeg_mat="simulated_eeg.mat", coord_file="kanalkoordinatlarson.txt"):
    """Tüm testleri çalıştır"""
    log("\n🚀 KAPSAMLI EEG LOKALİZASYON TESTİ BAŞLIYOR")
    log("="*80)
    log(f"EEG dosyası: {eeg_mat}")
    log(f"Koordinat dosyası: {coord_file}")
    log(f"Gerçek dipoller: {TRUE_DIPOLES}")
    
    # 1. Veri yükle ve kontrol et
    eeg_data, channel_coords = validate_simulation_data(eeg_mat, coord_file, TRUE_DIPOLES)
    
    # 2. Kanal aktivite analizi
    channel_power = analyze_channel_activity(eeg_data, channel_coords, TRUE_DIPOLES)
    
    # 3. Görselleştirme
    visualize_activity_distribution(eeg_data, channel_coords, TRUE_DIPOLES)
    
    # 4. Basit lokalizasyon testleri
    simple_results = simple_localization_tests(eeg_data, channel_coords, TRUE_DIPOLES)
    
    log("\n📊 BASİT YÖNTEM SONUÇLARI:")
    log("="*40)
    for method, error in simple_results.items():
        log(f"{method:.<30} {error:.1f}mm")
    
    # 5. Dipol fitting
    dipoles, errors = dipole_fitting_approach(eeg_data, channel_coords)
    
    # 6. Spatial correlation
    patterns, corr = spatial_correlation_analysis(eeg_data, channel_coords, TRUE_DIPOLES)
    
    # 7. NIfTI dosya analizi (varsa)
    nifti_results = analyze_nifti_results("delta")
    
    # ÖZET RAPOR
    log("\n" + "="*80)
    log("📋 ÖZET RAPOR")
    log("="*80)
    log(f"En iyi basit yöntem: {min(simple_results.values()):.1f}mm")
    log(f"Dipol fitting: {np.mean(errors):.1f}mm")
    log(f"Spatial correlation: {corr:.3f}")
    
    # Öneriler
    log("\n💡 ÖNERİLER:")
    if min(simple_results.values()) < 25:
        log("✓ Basit yöntemler iyi çalışıyor - veri kalitesi yüksek")
    else:
        log("⚠ Basit yöntemler bile yüksek hata veriyor - muhtemel sorunlar:")
        log("  - Simülasyon verisi yanlış dipoller içeriyor olabilir")
        log("  - Kanal koordinatları hatalı olabilir")
        log("  - SNR çok düşük olabilir")
    
    if corr < 0.5:
        log("⚠ Düşük spatial correlation - beklenen pattern'e uymuyor")
    
    return {
        'simple_results': simple_results,
        'dipole_fitting': np.mean(errors),
        'spatial_correlation': corr,
        'channel_power': channel_power
    }

# === 9. PROBLEM TEŞHİS FONKSİYONU === #
def diagnose_localization_problem(eeg_data, channel_coords, true_dipoles):
    """Lokalizasyon probleminin kaynağını teşhis et"""
    log("\n🏥 PROBLEM TEŞHİSİ")
    log("="*60)
    
    ch_positions = np.array(list(channel_coords.values()))
    ch_names = list(channel_coords.keys())
    
    # Test 1: Dipol bölgelerinde yeterli kanal var mı?
    log("\n📍 Test 1: Dipol Kapsama Analizi")
    for i, dipole in enumerate(true_dipoles):
        distances = np.linalg.norm(ch_positions - dipole, axis=1)
        nearby_30mm = np.sum(distances < 30)
        nearby_50mm = np.sum(distances < 50)
        
        log(f"Dipol {i+1} [{dipole[0]:.0f}, {dipole[1]:.0f}, {dipole[2]:.0f}]:")
        log(f"  30mm içinde: {nearby_30mm} kanal")
        log(f"  50mm içinde: {nearby_50mm} kanal")
        
        if nearby_30mm < 3:
            log("  ⚠️ SORUN: Yetersiz kanal kapsama!")
    
    # Test 2: Sinyal kalitesi
    log("\n📍 Test 2: Sinyal Kalitesi")
    snr = np.mean(eeg_data**2) / np.var(eeg_data)
    log(f"  Ortalama SNR: {snr:.2f}")
    if snr < 1:
        log("  ⚠️ SORUN: Çok düşük SNR!")
    
    # Test 3: Spatial blurring
    log("\n📍 Test 3: Spatial Yayılım")
    channel_power = np.mean(eeg_data**2, axis=1)
    power_std = np.std(channel_power)
    power_mean = np.mean(channel_power)
    cv = power_std / power_mean
    log(f"  Güç dağılım CV: {cv:.2f}")
    if cv < 0.5:
        log("  ⚠️ SORUN: Aşırı spatial smoothing/blurring!")
    
    # Test 4: Dipol derinliği
    log("\n📍 Test 4: Dipol Derinliği")
    for i, dipole in enumerate(true_dipoles):
        depth = np.sqrt(dipole[0]**2 + dipole[1]**2 + dipole[2]**2)
        log(f"  Dipol {i+1} derinlik: {depth:.1f}mm")
        if depth < 60:
            log("    ⚠️ UYARI: Derin dipol - EEG hassasiyeti düşük!")

# === 10. ALTERNATİF SİMÜLASYON TESTİ === #
def test_with_synthetic_dipole(channel_coords, dipole_location, dipole_orientation=[0,0,1]):
    """Tek bir sentetik dipol ile test"""
    log(f"\n🧪 SENTETİK DİPOL TESTİ: {dipole_location}")
    log("="*60)
    
    ch_positions = np.array(list(channel_coords.values()))
    
    # Forward model
    distances = np.linalg.norm(ch_positions - dipole_location, axis=1)
    leadfield = 1 / (distances + 10)  # Simple 1/r model
    
    # Gürültü ekle
    noise = np.random.normal(0, 0.1 * np.max(leadfield), len(leadfield))
    signal = leadfield + noise
    
    # Basit lokalizasyon
    max_idx = np.argmax(signal)
    estimated_pos = ch_positions[max_idx]
    error = np.linalg.norm(estimated_pos - dipole_location)
    
    log(f"  Forward model max kanal: {list(channel_coords.keys())[max_idx]}")
    log(f"  Tahmin hatası: {error:.1f}mm")
    
    # Ağırlıklı ortalama
    top_5 = np.argsort(signal)[-5:]
    weights = signal[top_5] / np.sum(signal[top_5])
    weighted_pos = np.sum(ch_positions[top_5] * weights[:, np.newaxis], axis=0)
    weighted_error = np.linalg.norm(weighted_pos - dipole_location)
    
    log(f"  Ağırlıklı tahmin hatası: {weighted_error:.1f}mm")
    
    return error, weighted_error

# === 11. ÇÖZÜM ÖNERİLERİ === #
def suggest_solutions(test_results):
    """Test sonuçlarına göre çözüm öner"""
    log("\n💡 ÇÖZÜM ÖNERİLERİ")
    log("="*60)
    
    simple_error = min(test_results['simple_results'].values())
    
    if simple_error > 40:
        log("\n🔴 KRİTİK SORUN TESPİT EDİLDİ:")
        log("1. Simülasyon verisini kontrol edin:")
        log("   - Gerçek dipol konumları doğru mu?")
        log("   - Kanal sıralaması uyuyor mu?")
        log("2. Koordinat sistemini doğrulayın:")
        log("   - Tüm koordinatlar aynı sistemde mi (MNI)?")
        log("   - Z ekseni yukarı mı işaret ediyor?")
        
    elif simple_error > 25:
        log("\n🟡 İYİLEŞTİRME ÖNERİLERİ:")
        log("1. Parametreleri optimize edin:")
        log("   - sigma = 15-20")
        log("   - grid_spacing = 1mm")
        log("   - threshold = %50")
        log("2. Post-processing azaltın:")
        log("   - Smoothing sigma < 0.5")
        log("   - Morphological operations kapatın")
        
    else:
        log("\n🟢 PERFORMANS İYİ:")
        log("1. Fine-tuning için:")
        log("   - Multi-resolution grid kullanın")
        log("   - Beamforming yaklaşımı deneyin")
        log("   - Temporal bilgiyi kullanın")

# === ANA FONKSİYON === #
def main():
    """Ana test fonksiyonu"""
    import sys
    
    # Komut satırı argümanları
    eeg_file = sys.argv[1] if len(sys.argv) > 1 else "simulated_eeg.mat"
    coord_file = sys.argv[2] if len(sys.argv) > 2 else "kanalkoordinatlarson.txt"
    
    print("\n" + "="*80)
    print("   EEG LOKALİZASYON DEBUG VE ANALİZ ARACI   ".center(80))
    print("="*80)
    
    try:
        # Ana test
        results = comprehensive_test(eeg_file, coord_file)
        
        # Problem teşhisi
        eeg_data, channel_coords = validate_simulation_data(eeg_file, coord_file, TRUE_DIPOLES)
        diagnose_localization_problem(eeg_data, channel_coords, TRUE_DIPOLES)
        
        # Sentetik dipol testleri
        log("\n🧪 SENTETİK DİPOL VALİDASYON TESTLERİ")
        log("="*60)
        for dipole in TRUE_DIPOLES:
            test_with_synthetic_dipole(channel_coords, dipole)
        
        # Çözüm önerileri
        suggest_solutions(results)
        
        # Sonuç özeti
        log("\n" + "="*80)
        log("TEST TAMAMLANDI".center(80))
        log("="*80)
        log(f"En düşük hata: {min(results['simple_results'].values()):.1f}mm")
        log(f"Çıktı dosyaları:")
        log(f"  - debug_activity_distribution.png")
        log(f"  - test_results.txt")
        
        # Sonuçları dosyaya kaydet
        with open('test_results.txt', 'w') as f:
            f.write("EEG LOCALIZATION TEST RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Test Date: {datetime.now()}\n")
            f.write(f"EEG File: {eeg_file}\n")
            f.write(f"Coordinate File: {coord_file}\n")
            f.write(f"\nTrue Dipoles:\n")
            for i, dip in enumerate(TRUE_DIPOLES):
                f.write(f"  Dipole {i+1}: {dip}\n")
            f.write(f"\nResults:\n")
            for method, error in results['simple_results'].items():
                f.write(f"  {method}: {error:.1f}mm\n")
            f.write(f"  Dipole Fitting: {results['dipole_fitting']:.1f}mm\n")
            f.write(f"  Spatial Correlation: {results['spatial_correlation']:.3f}\n")
        
    except Exception as e:
        log(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
