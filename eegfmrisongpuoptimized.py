# eeg_source_localization_gpu_complete.py

import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch, hilbert
from sklearn.decomposition import FastICA
from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed
import nibabel as nib
import os
from datetime import datetime
from tqdm import tqdm
import gc
import warnings
import time
from multiprocessing import cpu_count
from nilearn import datasets, image, maskers
from nilearn.maskers import NiftiLabelsMasker
from scipy.ndimage import gaussian_filter
from nilearn import surface
from scipy.spatial.distance import cdist
warnings.filterwarnings('ignore')

# GPU imports
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    import torch
    import torch.nn.functional as F
    
    # GPU availability check
    cp.cuda.Device(0).compute_capability
    USE_GPU = True
    print("GPU bulundu ve kullanılacak!")
    
    # RTX 3050 Ti Mobile için özel ayarlar
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
    if '3050' in gpu_name:
        print("RTX 3050 Ti Mobile algılandı!")
        # 4GB GPU için memory limit
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=3.5 * 1024**3)  # 3.5GB limit
except:
    USE_GPU = False
    print("GPU bulunamadı, CPU kullanılacak.")

# === PARAMETRELER === #
data_path = r"C:\Users\kerem\Downloads\eegyedek\GPU"
eeg_mat = os.path.join(data_path, "S03_restingPre_EC.mat")
coord_file = os.path.join(data_path, "kanalkoordinatlarson.txt")
fs = 256
segment_duration = 2.0
segment_samples = int(fs * segment_duration)
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}
n_bins = 6
sigma = 20
MIN_CHANNEL_DISTANCE = 10
MAX_CHANNEL_DISTANCE = 150

# Grid spacing - CPU ile aynı
grid_spacing = 1.5

# RTX 3050 Ti için optimize edilmiş parametreler
if USE_GPU and '3050' in gpu_name:
    batch_size = 4000
    n_jobs = 6
else:
    batch_size = 5000
    n_jobs = min(12, cpu_count())

MI_WEIGHT = 0.2
DICE_WEIGHT = 0.8

# Atlas seçenekleri
USE_BRAIN_MASK = True
USE_HARVARD_OXFORD = False
USE_TISSUE_TYPE = False
USE_GRAY_MATTER_MASK = True
CREATE_CONN_OUTPUT = True
ROI_BASED_ANALYSIS = True
USE_SURFACE_PROJECTION = True
SUPPRESS_DEEP_ACTIVITY = False

# Surface projection için parametreler
SURFACE_PROJECTION_METHOD = "enhanced"
USE_CORTICAL_RIBBON = True
SURFACE_SMOOTHING_FACTOR = 1.5

# Post-processing parametreleri
APPLY_THRESHOLD = True
THRESHOLD_PERCENTILE = 70
APPLY_SMOOTHING = True
SMOOTHING_SIGMA = 0.8
APPLY_ZSCORE = True
ENFORCE_BRAIN_BOUNDS = True
APPLY_MORPHOLOGICAL_OPERATIONS = False

# Hybrid score parametreleri
USE_ADAPTIVE_WEIGHTS = True
SPATIAL_PRIOR_WEIGHT = 0.0
MI_WEIGHT = 0.25
DICE_WEIGHT = 0.75

# === LOGGING === #
def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

# === GPU HELPER FUNCTIONS === #
def check_gpu_memory():
    """GPU bellek durumunu kontrol et"""
    if USE_GPU:
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        log(f"GPU Bellek: Kullanılan={used_bytes/1024**2:.0f}MB, Boş={free_mem/1024**2:.0f}MB, Toplam={total_mem/1024**2:.0f}MB")

def clear_gpu_memory():
    """GPU belleğini temizle"""
    if USE_GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def calculate_dynamic_batch_size(n_elements, element_dim, n_timepoints=1, safety_factor=0.6):
    """Calculate optimal batch size based on available GPU memory"""
    if not USE_GPU:
        return min(5000, n_elements)
    
    try:
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        bytes_per_element = (
            element_dim * 4 +
            n_timepoints * 4 +
            element_dim * n_timepoints * 4 * 0.5
        )
        safe_memory = free_mem * safety_factor
        batch_size = int(safe_memory / bytes_per_element)
        batch_size = max(100, batch_size)
        batch_size = min(n_elements, batch_size)
        
        if '3050' in cp.cuda.runtime.getDeviceProperties(0)['name'].decode():
            batch_size = min(batch_size, 5000)
        
        log(f"Dynamic batch size: {batch_size} (Free mem: {free_mem/1024**3:.1f}GB)")
        return batch_size
        
    except Exception as e:
        log(f"Error calculating dynamic batch size: {e}")
        return 3000

# === CPU'daki tüm fonksiyonları kopyala === #

def validate_and_transform_coords(coord_file):
    """Kanal koordinatlarını kontrol et ve dönüştür - 10-20 sisteminden MNI'ya"""
    coords = {}
    with open(coord_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                name, x, y, z = parts
                x, y, z = float(x), float(y), float(z)
                z += 20
                coords[name] = np.array([x, y, z])
    
    positions = np.array(list(coords.values()))
    log(f"Kanal koordinat aralığı (düzeltilmiş):")
    log(f"  X: [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}]")
    log(f"  Y: [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]")
    log(f"  Z: [{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")
    
    radii = np.sqrt(np.sum(positions**2, axis=1))
    log(f"  Ortalama yarıçap: {radii.mean():.1f}mm (min: {radii.min():.1f}, max: {radii.max():.1f})")
    
    return coords

def bandpass(data, fs, low, high):
    """Bandpass filter"""
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=1)

def notch(data, fs, freq=50.0):
    """Notch filter"""
    b, a = iirnotch(freq / (fs / 2), Q=30)
    return filtfilt(b, a, data, axis=1)

def hilbert_envelope(segment, band, fs):
    """Hilbert envelope hesapla"""
    filtered = bandpass(segment, fs, band[0], band[1])
    analytic = hilbert(filtered, axis=1)
    return np.abs(analytic).mean(axis=1)

def hilbert_envelope_gpu(segment, band, fs):
    """GPU-accelerated Hilbert envelope"""
    if not USE_GPU:
        return hilbert_envelope(segment, band, fs)
    
    try:
        # GPU'ya yükle
        seg_gpu = cp.asarray(segment, dtype=cp.float32)
        
        # Bandpass filter on GPU
        from cupyx.scipy.signal import butter as cp_butter, filtfilt as cp_filtfilt
        b, a = cp_butter(4, [band[0] / (fs/2), band[1] / (fs/2)], btype='band')
        filtered = cp_filtfilt(b, a, seg_gpu, axis=1)
        
        # Hilbert transform
        n = filtered.shape[1]
        fft = cp.fft.fft(filtered, axis=1)
        h = cp.zeros(n)
        h[0] = 1
        h[1:(n+1)//2] = 2
        analytic = cp.fft.ifft(fft * h, axis=1)
        envelope = cp.abs(analytic).mean(axis=1)
        
        return envelope.get()
        
    except Exception as e:
        log(f"GPU Hilbert hatası: {e}")
        return hilbert_envelope(segment, band, fs)

def create_voxel_grid(x_range, y_range, z_range, spacing):
    """MNI uzayında voxel grid oluştur"""
    xs = np.arange(x_range[0], x_range[1], spacing)
    ys = np.arange(y_range[0], y_range[1], spacing)
    zs = np.arange(z_range[0], z_range[1], spacing)
    grid = np.array(np.meshgrid(xs, ys, zs, indexing='ij'))
    coords = grid.reshape(3, -1).T
    shape = (len(xs), len(ys), len(zs))
    log(f"Grid boyutu: {shape}, Toplam voxel: {len(coords)}")
    return coords, shape

def apply_brain_mask(voxel_coords, grid_spacing):
    """MNI152 beyin maskesi"""
    log("Beyin maskesi yükleniyor...")
    
    mni_mask = datasets.load_mni152_brain_mask()
    mask_data = mni_mask.get_fdata()
    mask_affine = mni_mask.affine
    
    brain_voxels = []
    brain_indices = []
    
    for i, coord in enumerate(tqdm(voxel_coords, desc="Beyin maskesi uygulanıyor")):
        voxel_idx = np.round(np.linalg.inv(mask_affine) @ np.append(coord, 1))[:3].astype(int)
        
        if (0 <= voxel_idx[0] < mask_data.shape[0] and
            0 <= voxel_idx[1] < mask_data.shape[1] and
            0 <= voxel_idx[2] < mask_data.shape[2]):
            
            if mask_data[voxel_idx[0], voxel_idx[1], voxel_idx[2]] > 0:
                brain_voxels.append(coord)
                brain_indices.append(i)
    
    brain_voxels = np.array(brain_voxels)
    log(f"Beyin içindeki voxel sayısı: {len(brain_voxels)} / {len(voxel_coords)}")
    
    return brain_voxels, np.array(brain_indices)

def apply_harvard_oxford_mask(voxel_coords):
    """Harvard-Oxford atlas maskesi uygula"""
    log("Harvard-Oxford atlas yükleniyor...")

    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')

    cort_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
    sub_img = ho_sub.maps if isinstance(ho_sub.maps, nib.Nifti1Image) else nib.load(ho_sub.maps)

    cort_data = cort_img.get_fdata()
    sub_data = sub_img.get_fdata()

    all_labels = ['Background'] + ho_cort.labels + ho_sub.labels

    atlas_data = cort_data.copy()
    sub_mask = sub_data > 0
    atlas_data[sub_mask] = sub_data[sub_mask] + len(ho_cort.labels)

    atlas_affine = cort_img.affine
    inv_affine = np.linalg.inv(atlas_affine)

    valid_voxels, valid_indices, voxel_labels, region_ids = [], [], [], []

    for i, coord in enumerate(tqdm(voxel_coords, desc="Harvard-Oxford maskesi uygulanıyor")):
        voxel_idx = np.round(inv_affine @ np.append(coord, 1))[:3].astype(int)
        if (0 <= voxel_idx[0] < atlas_data.shape[0] and
            0 <= voxel_idx[1] < atlas_data.shape[1] and
            0 <= voxel_idx[2] < atlas_data.shape[2]):
            region_id = int(atlas_data[tuple(voxel_idx)])
            if region_id > 0:
                valid_voxels.append(coord)
                valid_indices.append(i)
                voxel_labels.append(all_labels[region_id])
                region_ids.append(region_id)

    valid_voxels = np.array(valid_voxels)
    log(f"Harvard-Oxford içindeki voxel sayısı: {len(valid_voxels)} / {len(voxel_coords)}")

    unique_regions, counts = np.unique(voxel_labels, return_counts=True)
    log("En çok voxel içeren bölgeler:")
    for region, count in sorted(zip(unique_regions, counts), key=lambda x: x[1], reverse=True)[:10]:
        log(f"  {region}: {count} voxel")

    return valid_voxels, np.array(valid_indices), voxel_labels, region_ids, all_labels

def filter_white_matter_from_labels(voxel_labels, region_ids):
    """Harvard-Oxford etiketlerinden white matter ve ilgili yapıları çıkar"""
    exclude_keywords = [
        'White Matter', 'Ventricle', 'Background', 'Brain-Stem',
        'vessel', 'CSF', 'Corpus Callosum', 'Cerebral White Matter',
        'Lateral Ventricle', 'Third Ventricle', 'Fourth Ventricle',
        'Brainstem', 'Cerebellar White Matter'
    ]
    
    keep_indices = []
    excluded_regions = []
    
    for i, label in enumerate(voxel_labels):
        if not any(keyword.lower() in label.lower() for keyword in exclude_keywords):
            keep_indices.append(i)
        else:
            if label not in excluded_regions:
                excluded_regions.append(label)
    
    if excluded_regions:
        log("Çıkarılan bölgeler:")
        for region in excluded_regions:
            log(f"  - {region}")
    
    log(f"Filtreleme sonucu: {len(keep_indices)} / {len(voxel_labels)} voxel tutuldu")
    
    return keep_indices

def compute_directional_voxel_signatures(voxel_coords, channel_coords, sigma=35, batch_size=5000, angle_exponent=0.5):
    """GPU-optimized voxel signature hesaplama"""
    if USE_GPU:
        return compute_voxel_signatures_gpu(voxel_coords, channel_coords, sigma, batch_size)
    else:
        return compute_voxel_signatures_cpu(voxel_coords, channel_coords, sigma, batch_size)

def compute_voxel_signatures_cpu(voxel_coords, channel_coords, sigma, batch_size):
    """CPU version of voxel signature computation"""
    channel_names = list(channel_coords.keys())
    channel_vectors = np.stack([channel_coords[ch] for ch in channel_names])
    
    log(f"\nEEG 10-10 Kanal Analizi:")
    log(f"  Toplam kanal sayısı: {len(channel_vectors)}")
    log(f"  Sigma (spatial spread): {sigma}mm")
    
    n_voxels = len(voxel_coords)
    n_channels = len(channel_vectors)
    voxel_signatures = np.zeros((n_voxels, n_channels), dtype=np.float32)
    
    voxel_radii = np.linalg.norm(voxel_coords, axis=1)
    log(f"\nVoxel istatistikleri:")
    log(f"  Toplam voxel: {n_voxels}")
    log(f"  Ortalama voxel yarıçapı: {np.mean(voxel_radii):.1f}mm")
    
    for i in tqdm(range(0, n_voxels, batch_size), desc="Voxel signatures"):
        end = min(i + batch_size, n_voxels)
        vxs = voxel_coords[i:end]
        
        dists = np.linalg.norm(vxs[:, None, :] - channel_vectors[None, :, :], axis=2)
        
        for j, vox_idx in enumerate(range(i, end)):
            channel_dists = dists[j]
            
            valid_channels = (channel_dists >= MIN_CHANNEL_DISTANCE) & (channel_dists <= MAX_CHANNEL_DISTANCE)
            
            if np.any(valid_channels):
                valid_dists = channel_dists[valid_channels]
                weights = np.exp(-valid_dists**2 / (2 * sigma**2))
                weights /= np.sum(weights)
                voxel_signatures[vox_idx, valid_channels] = weights
            else:
                nearest_3 = np.argpartition(channel_dists, 3)[:3]
                weights = np.exp(-channel_dists[nearest_3]**2 / (2 * sigma**2))
                weights /= np.sum(weights)
                voxel_signatures[vox_idx, nearest_3] = weights
    
    return voxel_signatures

def compute_voxel_signatures_gpu(voxel_coords, channel_coords, sigma, batch_size):
    """GPU-accelerated voxel signature computation"""
    channel_names = list(channel_coords.keys())
    channel_vectors = np.stack([channel_coords[ch] for ch in channel_names])
    
    log(f"\nGPU EEG 10-10 Kanal Analizi:")
    log(f"  Toplam kanal sayısı: {len(channel_vectors)}")
    log(f"  Sigma (spatial spread): {sigma}mm")
    log(f"  Batch size: {batch_size}")
    
    n_voxels = len(voxel_coords)
    n_channels = len(channel_vectors)
    
    # GPU'ya taşı
    channel_vectors_gpu = cp.asarray(channel_vectors, dtype=cp.float32)
    voxel_signatures = cp.zeros((n_voxels, n_channels), dtype=cp.float32)
    
    # Dynamic batch size for RTX 3050 Ti
    actual_batch_size = min(batch_size, calculate_dynamic_batch_size(n_voxels, n_channels * 3))
    
    for i in tqdm(range(0, n_voxels, actual_batch_size), desc="GPU Voxel signatures"):
        end = min(i + actual_batch_size, n_voxels)
        
        # Memory check
        free_mem, _ = cp.cuda.runtime.memGetInfo()
        if free_mem < 500 * 1024 * 1024:  # 500MB'den az
            clear_gpu_memory()
            log("GPU belleği temizlendi (düşük bellek)")
        
        vxs_gpu = cp.asarray(voxel_coords[i:end], dtype=cp.float32)
        
        # GPU'da mesafe hesapla
        dists = cp.linalg.norm(
            vxs_gpu[:, None, :] - channel_vectors_gpu[None, :, :], 
            axis=2
        )
        
        # Valid channels check
        for j in range(end - i):
            channel_dists = dists[j]
            valid_channels = (channel_dists >= MIN_CHANNEL_DISTANCE) & (channel_dists <= MAX_CHANNEL_DISTANCE)
            
            if cp.any(valid_channels):
                valid_dists = channel_dists[valid_channels]
                weights = cp.exp(-valid_dists**2 / (2 * sigma**2))
                weights /= cp.sum(weights)
                voxel_signatures[i+j, valid_channels] = weights
            else:
                # En yakın 3 kanal
                nearest_3 = cp.argpartition(channel_dists, 3)[:3]
                weights = cp.exp(-channel_dists[nearest_3]**2 / (2 * sigma**2))
                weights /= cp.sum(weights)
                voxel_signatures[i+j, nearest_3] = weights
        
        # Belleği hemen temizle
        del vxs_gpu, dists
    
    # CPU'ya geri getir
    result = voxel_signatures.get()
    
    # GPU belleğini tamamen temizle
    del channel_vectors_gpu, voxel_signatures
    clear_gpu_memory()
    
    return result

def compute_mi_dice_for_snapshot(voxel_sigs, snapshot, v_bins, t):
    """Tek bir snapshot için MI ve Dice hesapla"""
    s_bins = np.histogram_bin_edges(snapshot, bins=n_bins)
    sb = np.digitize(snapshot, bins=s_bins)
    
    n_voxels = len(voxel_sigs)
    mi_results = np.zeros(n_voxels, dtype=np.float32)
    
    # MI hesaplama
    for i in range(n_voxels):
        vb = np.digitize(voxel_sigs[i], bins=v_bins[i])
        mi_results[i] = mutual_info_score(vb, sb)
    
    # Dice hesaplama
    numerator = 2 * (voxel_sigs @ snapshot)
    norm_voxels = np.sum(voxel_sigs**2, axis=1)
    norm_snap = np.sum(snapshot**2)
    denominator = norm_voxels + norm_snap + 1e-8
    dice_results = numerator / denominator
    
    # Min-max normalizasyonu
    if np.max(mi_results) > np.min(mi_results):
        mi_norm = (mi_results - np.min(mi_results)) / (np.max(mi_results) - np.min(mi_results))
    else:
        mi_norm = mi_results
        
    if np.max(dice_results) > np.min(dice_results):
        dice_norm = (dice_results - np.min(dice_results)) / (np.max(dice_results) - np.min(dice_results))
    else:
        dice_norm = dice_results
    
    return t, mi_norm, dice_norm

def process_snapshots_parallel(voxel_sigs, snapshots, v_bins, n_jobs=12):
    """Snapshot'ları paralel işle"""
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(compute_mi_dice_for_snapshot)(voxel_sigs, snapshots[t], v_bins, t) 
        for t in range(snapshots.shape[0])
    )
    
    results.sort(key=lambda x: x[0])
    
    mi_scores = np.zeros((len(voxel_sigs), len(snapshots)), dtype=np.float32)
    dice_scores = np.zeros((len(voxel_sigs), len(snapshots)), dtype=np.float32)
    
    for t, mi, dice in results:
        mi_scores[:, t] = mi
        dice_scores[:, t] = dice
    
    return mi_scores, dice_scores

def process_snapshots_gpu(voxel_sigs, snapshots, v_bins, n_jobs=6):
    """GPU-accelerated snapshot processing"""
    n_voxels = len(voxel_sigs)
    n_timepoints = len(snapshots)
    
    mi_scores = np.zeros((n_voxels, n_timepoints), dtype=np.float32)
    dice_scores = np.zeros((n_voxels, n_timepoints), dtype=np.float32)
    
    log("GPU MI/Dice hesaplama başlıyor...")
    
    # GPU'ya bir kere yükle
    if USE_GPU:
        try:
            voxel_sigs_gpu = cp.asarray(voxel_sigs, dtype=cp.float32)
            
            for t in tqdm(range(n_timepoints), desc="GPU MI/Dice"):
                snapshot_gpu = cp.asarray(snapshots[t], dtype=cp.float32)
                
                # Dice hesaplama (GPU'da hızlı)
                numerator = 2 * cp.dot(voxel_sigs_gpu, snapshot_gpu)
                norm_voxels = cp.sum(voxel_sigs_gpu**2, axis=1)
                norm_snap = cp.sum(snapshot_gpu**2)
                denominator = norm_voxels + norm_snap + 1e-8
                dice_results_gpu = numerator / denominator
                
                # GPU'dan CPU'ya transfer
                dice_results = dice_results_gpu.get()
                
                # MI hesaplama (CPU'da)
                mi_results = np.zeros(n_voxels, dtype=np.float32)
                snapshot_cpu = snapshots[t]
                s_bins = np.histogram_bin_edges(snapshot_cpu, bins=n_bins)
                sb = np.digitize(snapshot_cpu, bins=s_bins)
                
                for i in range(n_voxels):
                    vb = np.digitize(voxel_sigs[i], bins=v_bins[i])
                    mi_results[i] = np.sum(vb == sb) / len(sb)
                
                # Normalize
                if mi_results.max() > mi_results.min():
                    mi_norm = (mi_results - mi_results.min()) / (mi_results.max() - mi_results.min() + 1e-8)
                else:
                    mi_norm = mi_results
                    
                if dice_results.max() > dice_results.min():
                    dice_norm = (dice_results - dice_results.min()) / (dice_results.max() - dice_results.min() + 1e-8)
                else:
                    dice_norm = dice_results
                
                mi_scores[:, t] = mi_norm
                dice_scores[:, t] = dice_norm
                
                # Bellek temizleme
                del snapshot_gpu, dice_results_gpu
                
                # Her 10 timepoint'te bellek durumu
                if t % 10 == 0:
                    check_gpu_memory()
                    
            del voxel_sigs_gpu
            clear_gpu_memory()
            
        except Exception as e:
            log(f"GPU MI/Dice hatası: {e}, CPU'ya geçiliyor...")
            return process_snapshots_parallel(voxel_sigs, snapshots, v_bins, n_jobs)
    else:
        return process_snapshots_parallel(voxel_sigs, snapshots, v_bins, n_jobs)
    
    return mi_scores, dice_scores

# === POST-PROCESSING FONKSİYONLARI === #
def apply_threshold(data, percentile=85):
    """Threshold uygula - düşük değerleri sıfırla"""
    threshold = np.percentile(data[data > 0], percentile)
    data_thresholded = data.copy()
    data_thresholded[data_thresholded < threshold] = 0
    return data_thresholded

def apply_spatial_smoothing(data, sigma=0.5):
    """Spatial smoothing - GPU destekli"""
    if USE_GPU and data.shape[3] > 10:
        return apply_spatial_smoothing_gpu(data, sigma)
    
    smoothed = np.zeros_like(data)
    for t in range(data.shape[3]):
        smoothed[:,:,:,t] = gaussian_filter(data[:,:,:,t], sigma=sigma)
    return smoothed

def apply_spatial_smoothing_gpu(data, sigma=0.5):
    """GPU-accelerated Gaussian filtering"""
    log(f"GPU spatial smoothing (sigma={sigma})...")
    
    free_mem, _ = cp.cuda.runtime.memGetInfo()
    data_size = data.nbytes / 1024**2  # MB
    
    if data_size > free_mem / 1024**2 * 0.8:
        log("Data GPU belleğine sığmıyor, CPU'da işleniyor...")
        return apply_spatial_smoothing(data, sigma)
    
    try:
        data_gpu = cp.asarray(data, dtype=cp.float32)
        smoothed_gpu = cp.zeros_like(data_gpu)
        
        for t in range(data.shape[3]):
            smoothed_gpu[:,:,:,t] = cp_ndimage.gaussian_filter(
                data_gpu[:,:,:,t], 
                sigma=sigma
            )
            
            if t % 10 == 0:
                free_mem, _ = cp.cuda.runtime.memGetInfo()
                if free_mem < 200 * 1024 * 1024:
                    log(f"GPU belleği azaldı, slice {t}/{data.shape[3]}")
                    clear_gpu_memory()
        
        result = smoothed_gpu.get()
        del data_gpu, smoothed_gpu
        clear_gpu_memory()
        
        return result
        
    except cp.cuda.memory.OutOfMemoryError:
        log("GPU bellek hatası! CPU'ya geçiliyor...")
        clear_gpu_memory()
        return apply_spatial_smoothing(data, sigma)

def apply_morphological_operations(data, kernel_size=3):
    """Morfolojik işlemlerle gürültü temizleme"""
    from scipy.ndimage import binary_opening, binary_closing, binary_dilation
    
    cleaned_data = np.zeros_like(data)
    
    for t in range(data.shape[3]):
        slice_data = data[:,:,:,t]
        threshold = np.percentile(slice_data[slice_data > 0], 30)
        binary_mask = slice_data > threshold
        
        binary_mask = binary_opening(binary_mask, iterations=1)
        binary_mask = binary_closing(binary_mask, iterations=1)
        binary_mask = binary_dilation(binary_mask, iterations=1)
        
        cleaned_data[:,:,:,t] = slice_data * binary_mask
    
    return cleaned_data

def apply_bilateral_filter(data, spatial_sigma=2.0, intensity_sigma=0.1):
    """Bilateral filter - edge-preserving smoothing"""
    from scipy.ndimage import gaussian_filter
    
    filtered_data = np.zeros_like(data)
    
    for t in range(data.shape[3]):
        slice_data = data[:,:,:,t]
        smoothed = gaussian_filter(slice_data, sigma=spatial_sigma)
        diff = np.abs(slice_data - smoothed)
        weight = np.exp(-diff**2 / (2 * intensity_sigma**2))
        filtered_data[:,:,:,t] = slice_data * (1 - weight) + smoothed * weight
    
    return filtered_data

def apply_zscore_normalization(data):
    """Her zaman noktası için z-score normalizasyonu"""
    normalized = np.zeros_like(data)
    for t in range(data.shape[3]):
        slice_data = data[:,:,:,t]
        mask = slice_data != 0
        if np.any(mask):
            mean_val = np.mean(slice_data[mask])
            std_val = np.std(slice_data[mask])
            if std_val > 0:
                slice_data[mask] = (slice_data[mask] - mean_val) / std_val
            normalized[:,:,:,t] = slice_data
    return normalized

def compute_temporal_stability(data):
    """Her voxel için temporal stabilite hesapla"""
    mask = data[:,:,:,0] != 0
    stability = np.zeros(data.shape[:3])
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if mask[i,j,k]:
                    time_series = data[i,j,k,:]
                    if np.std(time_series) > 0:
                        stability[i,j,k] = 1 / (1 + np.std(time_series) / (np.mean(time_series) + 1e-8))
                    else:
                        stability[i,j,k] = 1
    
    return stability

def adaptive_hybrid_scoring(mi_scores, dice_scores, temporal_stability=None):
    """Adaptive ağırlıklandırma ile hibrit skor hesapla"""
    if temporal_stability is not None and USE_ADAPTIVE_WEIGHTS:
        # Temporal stabiliteye göre ağırlıkları ayarla
        stability_weight = temporal_stability / (temporal_stability + 1)
        mi_weight = MI_WEIGHT * (1 - 0.2 * stability_weight)
        dice_weight = DICE_WEIGHT * (1 - 0.2 * stability_weight)
        stability_contribution = 0.4 * stability_weight
        
        hybrid = (mi_weight[:, np.newaxis] * mi_scores + 
                 dice_weight[:, np.newaxis] * dice_scores + 
                 stability_contribution[:, np.newaxis])
    else:
        # Basit ağırlıklandırma
        hybrid = MI_WEIGHT * mi_scores + DICE_WEIGHT * dice_scores
    
    return hybrid

def suppress_deep_brain_activity(data, affine, suppression_threshold=50):
    """Derin beyin aktivitelerini bastır"""
    log("Derin beyin aktiviteleri bastırılıyor...")
    suppressed = data.copy()
    
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                voxel_idx = np.array([x, y, z, 1])
                mni_coord = affine @ voxel_idx
                
                z_coord = mni_coord[2]
                
                if z_coord < -10:
                    suppressed[x, y, z, :] = 0
                elif z_coord < 10:
                    suppressed[x, y, z, :] *= 0.3
                elif z_coord < 30:
                    suppressed[x, y, z, :] *= 0.6
                
                y_coord = mni_coord[1]
                if abs(y_coord) > 70:
                    suppressed[x, y, z, :] *= 0.8
                
                radius = np.linalg.norm(mni_coord[:3])
                if radius < 50:
                    suppressed[x, y, z, :] *= 0.1
    
    return suppressed

def enforce_brain_boundaries(data, affine):
    """Beyin sınırları dışındaki aktiviteleri sıfırla"""
    log("Beyin sınırları kontrolü...")
    
    x_bounds = (-78, 78)
    y_bounds = (-112, 76)
    z_bounds = (-70, 85)
    
    cleaned_data = data.copy()
    removed_count = 0
    
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                if np.any(data[x, y, z, :] > 0):
                    voxel_idx = np.array([x, y, z, 1])
                    mni_coord = affine @ voxel_idx
                    x_mni, y_mni, z_mni = mni_coord[:3]
                    
                    if (x_mni < x_bounds[0] or x_mni > x_bounds[1] or
                        y_mni < y_bounds[0] or y_mni > y_bounds[1] or
                        z_mni < z_bounds[0] or z_mni > z_bounds[1]):
                        cleaned_data[x, y, z, :] = 0
                        removed_count += 1
    
    if removed_count > 0:
        log(f"  {removed_count} voxel beyin sınırları dışında temizlendi")
    
    return cleaned_data

# === SURFACE PROJECTION FONKSİYONLARI === #
def enhanced_surface_projection_fixed(band_envelope, channel_coords):
    """GPU-optimized enhanced surface projection - DÜZELTME"""
    log("Enhanced surface projection başlıyor...")
    
    # Surface yükle
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    
    # Pial ve white matter yükle
    lh_pial, _ = surface.load_surf_mesh(fsaverage['pial_left'])
    rh_pial, _ = surface.load_surf_mesh(fsaverage['pial_right'])
    lh_white, _ = surface.load_surf_mesh(fsaverage['white_left'])
    rh_white, _ = surface.load_surf_mesh(fsaverage['white_right'])
    
    # Mid-surface hesapla
    lh_mid = (lh_pial + lh_white) / 2
    rh_mid = (rh_pial + rh_white) / 2
    
    # FreeSurfer RAS to MNI transformation
    fs_to_mni = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # Transform vertices to MNI space
    lh_mid_mni = np.array([(fs_to_mni @ np.append(v, 1))[:3] for v in lh_mid])
    rh_mid_mni = np.array([(fs_to_mni @ np.append(v, 1))[:3] for v in rh_mid])
    
    # Transform pial and white surfaces too
    lh_pial_mni = np.array([(fs_to_mni @ np.append(v, 1))[:3] for v in lh_pial])
    rh_pial_mni = np.array([(fs_to_mni @ np.append(v, 1))[:3] for v in rh_pial])
    lh_white_mni = np.array([(fs_to_mni @ np.append(v, 1))[:3] for v in lh_white])
    rh_white_mni = np.array([(fs_to_mni @ np.append(v, 1))[:3] for v in rh_white])
    
    all_vertices = np.vstack([lh_mid_mni, rh_mid_mni])
    
    # *** ÖNEMLİ: GPU VE CPU İÇİN AYNI HESAPLAMA ***
    ch_positions = np.array(list(channel_coords.values()))
    n_vertices = len(all_vertices)
    n_timepoints = band_envelope.shape[1]
    
    if USE_GPU:
        log("GPU Surface projection kullanılıyor...")
        
        # Dynamic batch size
        batch_size = calculate_dynamic_batch_size(n_vertices, ch_positions.shape[0], n_timepoints)
        
        # GPU arrays - float64 kullan
        ch_positions_gpu = cp.asarray(ch_positions, dtype=cp.float64)
        band_envelope_gpu = cp.asarray(band_envelope, dtype=cp.float64)
        surface_data = np.zeros((n_vertices, n_timepoints), dtype=np.float64)
        
        log(f"GPU Surface projection - batch size: {batch_size}")
        
        for v_start in tqdm(range(0, n_vertices, batch_size), desc="GPU Surface projection"):
            v_end = min(v_start + batch_size, n_vertices)
            
            batch_vertices_gpu = cp.asarray(all_vertices[v_start:v_end], dtype=cp.float64)
            
            # Distance computation
            distances = cp.linalg.norm(
                batch_vertices_gpu[:, None, :] - ch_positions_gpu[None, :, :], 
                axis=2
            )
            
            # Gaussian weights
            weights = cp.exp(-distances**2 / (2 * 30**2))
            
            # CPU ile aynı normalizasyon
            row_sums = weights.sum(axis=1, keepdims=True)
            weights = weights / (row_sums + 1e-10)
            
            # Projection
            batch_surface_data = weights @ band_envelope_gpu
            
            # Transfer back
            surface_data[v_start:v_end, :] = batch_surface_data.get()
            
            # Cleanup
            del batch_vertices_gpu, distances, weights, batch_surface_data
            
            if v_start % (batch_size * 5) == 0:
                clear_gpu_memory()
        
        del ch_positions_gpu, band_envelope_gpu
        clear_gpu_memory()
        
    else:
        log("CPU Surface projection kullanılıyor...")
        surface_data = np.zeros((n_vertices, n_timepoints))
        
        for v_start in tqdm(range(0, n_vertices, 1000), desc="CPU Surface projection"):
            v_end = min(v_start + 1000, n_vertices)
            batch_vertices = all_vertices[v_start:v_end]
            
            distances = cdist(batch_vertices, ch_positions)
            weights = np.exp(-distances**2 / (2 * 30**2))
            
            row_sums = weights.sum(axis=1, keepdims=True)
            valid_rows = row_sums.squeeze() > 0
            
            if np.any(valid_rows):
                weights[valid_rows] = weights[valid_rows] / row_sums[valid_rows]
            
            surface_data[v_start:v_end, :] = weights @ band_envelope
    
    # Normalize surface data
    log(f"Surface data range: [{surface_data.min():.6f}, {surface_data.max():.6f}]")
    
    return (surface_data, all_vertices, len(lh_mid), 
            lh_pial_mni, rh_pial_mni, lh_white_mni, rh_white_mni)

def gpu_surface_projection_fixed(band_envelope, channel_coords, all_vertices, 
                          lh_pial_mni, rh_pial_mni, lh_white_mni, rh_white_mni, n_lh):
    """Fixed GPU-accelerated surface projection"""
    ch_positions = np.array(list(channel_coords.values()))
    n_vertices = len(all_vertices)
    n_timepoints = band_envelope.shape[1]
    
    # Dynamic batch size
    batch_size = calculate_dynamic_batch_size(n_vertices, ch_positions.shape[0], n_timepoints)
    
    # Use float64 for better precision
    ch_positions_gpu = cp.asarray(ch_positions, dtype=cp.float64)
    band_envelope_gpu = cp.asarray(band_envelope, dtype=cp.float64)
    surface_data = np.zeros((n_vertices, n_timepoints), dtype=np.float64)
    
    log(f"GPU Surface projection - Dynamic batch size: {batch_size}")
    
    for v_start in tqdm(range(0, n_vertices, batch_size), desc="GPU Surface projection"):
        v_end = min(v_start + batch_size, n_vertices)
        
        # GPU'ya batch yükle
        batch_vertices_gpu = cp.asarray(all_vertices[v_start:v_end], dtype=cp.float64)
        
        # Distance computation
        distances = cp.linalg.norm(
            batch_vertices_gpu[:, None, :] - ch_positions_gpu[None, :, :], 
            axis=2
        )
        
        # Gaussian weights - CPU ile aynı sigma değeri
        weights = cp.exp(-distances**2 / (2 * 30**2))
        
        # CPU ile aynı normalizasyon yöntemi
        row_sums = weights.sum(axis=1, keepdims=True)
        # Sıfır kontrolü
        valid_mask = (row_sums.squeeze() > 1e-10)
        
        # Sadece valid satırları normalize et
        weights_normalized = cp.zeros_like(weights)
        if cp.any(valid_mask):
            valid_indices = cp.where(valid_mask)[0]
            weights_normalized[valid_indices] = weights[valid_indices] / row_sums[valid_indices]
        
        # Matrix multiplication
        batch_surface_data = weights_normalized @ band_envelope_gpu
        
        # Transfer back to CPU
        surface_data[v_start:v_end, :] = batch_surface_data.get()
        
        # Clean up
        del batch_vertices_gpu, distances, weights, weights_normalized, batch_surface_data
        
        # Memory check
        if (v_start // batch_size) % 5 == 0:
            adaptive_memory_management()
    
    # Final cleanup
    del ch_positions_gpu, band_envelope_gpu
    clear_gpu_memory()
    
    # Değerleri kontrol et ve normalize et
    log(f"Surface data range: [{surface_data.min():.6f}, {surface_data.max():.6f}]")
    
    # CPU ile aynı range'e getir
    if surface_data.max() > surface_data.min():
        surface_data = (surface_data - surface_data.min()) / (surface_data.max() - surface_data.min())
    
    return (surface_data, all_vertices, n_lh, lh_pial_mni, rh_pial_mni, lh_white_mni, rh_white_mni)

def cpu_surface_projection(band_envelope, channel_coords, all_vertices,
                          lh_pial_mni, rh_pial_mni, lh_white_mni, rh_white_mni, n_lh):
    """CPU version of surface projection"""
    ch_positions = np.array(list(channel_coords.values()))
    n_vertices = len(all_vertices)
    n_timepoints = band_envelope.shape[1]
    surface_data = np.zeros((n_vertices, n_timepoints))
    
    for v_start in tqdm(range(0, n_vertices, 1000), desc="CPU Surface projection"):
        v_end = min(v_start + 1000, n_vertices)
        batch_vertices = all_vertices[v_start:v_end]
        
        distances = cdist(batch_vertices, ch_positions)
        weights = np.exp(-distances**2 / (2 * 30**2))
        
        row_sums = weights.sum(axis=1, keepdims=True)
        valid_rows = row_sums.squeeze() > 0
        
        if np.any(valid_rows):
            weights[valid_rows] = weights[valid_rows] / row_sums[valid_rows]
        
        surface_data[v_start:v_end, :] = weights @ band_envelope
    
    return (surface_data, all_vertices, n_lh, 
            lh_pial_mni, rh_pial_mni, lh_white_mni, rh_white_mni)

def improved_surface_to_volume_fixed(surface_data, vertices, pial_lh, pial_rh, 
                                   white_lh, white_rh, n_lh,
                                   volume_shape=(91, 109, 91), grid_spacing=2):
    """Düzeltilmiş cortical ribbon filling - GPU/CPU tutarlı"""
    log("Cortical ribbon filling başlıyor...")
    
    volume_4d = np.zeros((*volume_shape, surface_data.shape[1]), dtype=np.float64)
    
    # MNI152 standart affine
    affine = np.array([
        [-2, 0, 0, 90],
        [0, 2, 0, -126], 
        [0, 0, 2, -72],
        [0, 0, 0, 1]
    ])
    
    inv_affine = np.linalg.inv(affine)
    
    # Weight accumulator
    weight_sum = np.zeros(volume_shape, dtype=np.float64)
    
    # Sol hemisphere
    for v_idx in tqdm(range(n_lh), desc="LH ribbon"):
        for alpha in np.linspace(0, 1, 10):
            point = pial_lh[v_idx] * (1 - alpha) + white_lh[v_idx] * alpha
            
            vox_float = inv_affine[:3, :3] @ point + inv_affine[:3, 3]
            vox = np.round(vox_float).astype(int)
            
            if (0 <= vox[0] < volume_shape[0] and
                0 <= vox[1] < volume_shape[1] and
                0 <= vox[2] < volume_shape[2]):
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            x, y, z = vox[0]+dx, vox[1]+dy, vox[2]+dz
                            if (0 <= x < volume_shape[0] and
                                0 <= y < volume_shape[1] and
                                0 <= z < volume_shape[2]):
                                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                                weight = np.exp(-dist/1.5)
                                
                                # Accumulate instead of maximum
                                volume_4d[x, y, z, :] += surface_data[v_idx, :] * weight
                                weight_sum[x, y, z] += weight
    
    # Sağ hemisphere
    for v_idx in tqdm(range(len(pial_rh)), desc="RH ribbon"):
        for alpha in np.linspace(0, 1, 10):
            point = pial_rh[v_idx] * (1 - alpha) + white_rh[v_idx] * alpha
            
            vox_float = inv_affine[:3, :3] @ point + inv_affine[:3, 3]
            vox = np.round(vox_float).astype(int)
            
            if (0 <= vox[0] < volume_shape[0] and
                0 <= vox[1] < volume_shape[1] and
                0 <= vox[2] < volume_shape[2]):
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            x, y, z = vox[0]+dx, vox[1]+dy, vox[2]+dz
                            if (0 <= x < volume_shape[0] and
                                0 <= y < volume_shape[1] and
                                0 <= z < volume_shape[2]):
                                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                                weight = np.exp(-dist/1.5)
                                
                                volume_4d[x, y, z, :] += surface_data[n_lh + v_idx, :] * weight
                                weight_sum[x, y, z] += weight
    
    # Normalize by weight sum
    for t in range(volume_4d.shape[3]):
        mask = weight_sum > 0
        volume_4d[mask, t] /= weight_sum[mask]
    
    log(f"Volume range: [{volume_4d.min():.6f}, {volume_4d.max():.6f}]")
    
    return volume_4d.astype(np.float32), affine

def save_surface_maps(surface_data, band_name, n_lh_vertices, output_path):
    """Surface map'leri kaydet"""
    lh_data = surface_data[:n_lh_vertices]
    rh_data = surface_data[n_lh_vertices:]
    
    np.save(os.path.join(output_path, f'{band_name}_surface_lh.npy'), lh_data)
    np.save(os.path.join(output_path, f'{band_name}_surface_rh.npy'), rh_data)
    
    log(f"Surface maps kaydedildi: {band_name}_surface_[lh/rh].npy")

# === ROI ANALİZ FONKSİYONLARI === #
def compute_and_save_roi_centers(voxel_coords, region_ids, all_labels, output_path, band_name):
    """ROI merkez koordinatlarını hesaplar ve numpy dosyasına kaydeder"""
    
    log("ROI merkez koordinatları hesaplanıyor...")
    
    unique_regions = np.unique(region_ids)
    roi_centers_dict = {}
    roi_centers_list = []
    
    deep_structures = ['Thalamus', 'Putamen', 'Pallidum', 'Caudate', 'Accumbens', 
                      'Hippocampus', 'Amygdala', 'Brain-Stem']
    
    for region_id in unique_regions:
        if region_id > 0:
            region_name = all_labels[region_id] if all_labels else f"Region_{region_id}"
            
            mask = np.array(region_ids) == region_id
            region_voxels = np.array(voxel_coords)[mask]
            
            if len(region_voxels) > 0:
                roi_center = np.mean(region_voxels, axis=0)
                
                roi_centers_dict[f"r{region_id:03d}"] = {
                    'region_name': region_name,
                    'center_mni': roi_center,
                    'n_voxels': len(region_voxels)
                }
                
                roi_centers_list.append([
                    region_id,
                    region_name,
                    roi_center[0],
                    roi_center[1],
                    roi_center[2],
                    len(region_voxels)
                ])
    
    roi_centers_array = np.array(roi_centers_list, dtype=object)
    
    roi_centers_file = os.path.join(output_path, f"{band_name}_roi_centers.npy")
    np.save(roi_centers_file, roi_centers_array)
    log(f"ROI merkez koordinatları kaydedildi: {roi_centers_file}")
    log(f"  Toplam ROI sayısı: {len(roi_centers_list)}")
    
    centers_only = np.array([[item[2], item[3], item[4]] for item in roi_centers_list])
    centers_only_file = os.path.join(output_path, f"{band_name}_roi_centers_coords_only.npy")
    np.save(centers_only_file, centers_only)
    
    mat_data = {
        'roi_centers': roi_centers_array,
        'roi_names': [item[1] for item in roi_centers_list],
        'roi_ids': [item[0] for item in roi_centers_list],
        'centers_mni': centers_only,
        'n_voxels': [item[5] for item in roi_centers_list]
    }
    mat_file = os.path.join(output_path, f"{band_name}_roi_centers.mat")
    scipy.io.savemat(mat_file, mat_data)
    log(f"ROI merkezleri MATLAB formatında da kaydedildi: {mat_file}")
    
    sorted_rois = sorted(roi_centers_list, key=lambda x: x[5], reverse=True)[:10]
    log("\nEn büyük 10 ROI:")
    for roi in sorted_rois:
        log(f"  {roi[1]}: {roi[5]} voxel, merkez=({roi[2]:.1f}, {roi[3]:.1f}, {roi[4]:.1f})")
    
    return roi_centers_dict, centers_only

def create_conn_compatible_output(nii_img, band_name, region_labels, output_path):
    """CONN toolbox için ROI tabanlı time series oluştur"""
    log("CONN toolbox uyumlu çıktı oluşturuluyor...")
    
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    atlas_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
    
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=False,
        detrend=False,
        verbose=0
    )
    
    roi_timeseries = masker.fit_transform(nii_img)
    
    all_roi_names = ['Background'] + ho_cort.labels + ho_sub.labels
    
    active_rois = []
    active_timeseries = []
    for i in range(roi_timeseries.shape[1]):
        if np.any(roi_timeseries[:, i] != 0):
            active_rois.append(all_roi_names[i+1])
            active_timeseries.append(roi_timeseries[:, i])
    
    if active_timeseries:
        active_timeseries = np.column_stack(active_timeseries)
    else:
        log("UYARI: Hiç aktif ROI bulunamadı!")
        active_timeseries = roi_timeseries
        active_rois = all_roi_names[1:roi_timeseries.shape[1]+1]
    
    conn_data = {
        'roi_timeseries': active_timeseries,
        'roi_names': active_rois,
        'band': band_name,
        'n_timepoints': active_timeseries.shape[0],
        'n_rois': active_timeseries.shape[1],
        'TR': segment_duration
    }
    
    mat_filename = os.path.join(output_path, f'{band_name}_conn_roi_timeseries.mat')
    scipy.io.savemat(mat_filename, conn_data)
    log(f"CONN uyumlu dosya kaydedildi: {mat_filename}")
    log(f"  Aktif ROI sayısı: {len(active_rois)}")
    
    return active_timeseries

def compute_roi_statistics_from_nifti(nii_img, band_name, output_path):
    """NIfTI dosyasından direkt ROI istatistikleri hesapla"""
    log("NIfTI dosyasından ROI istatistikleri hesaplanıyor...")
    
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    atlas_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
    all_labels = ['Background'] + ho_cort.labels + ho_sub.labels
    
    from nilearn.image import resample_to_img
    nii_resampled = resample_to_img(nii_img, atlas_img, interpolation='nearest')
    
    nii_data = nii_resampled.get_fdata()
    atlas_data = atlas_img.get_fdata()
    
    roi_stats = {}
    
    unique_regions = np.unique(atlas_data[atlas_data > 0]).astype(int)
    
    for region_id in unique_regions:
        mask = atlas_data == region_id
        region_name = all_labels[region_id]
        
        roi_timeseries = nii_data[mask, :]
        
        if roi_timeseries.size > 0 and np.any(roi_timeseries > 0):
            mean_activity = np.mean(roi_timeseries)
            std_activity = np.std(roi_timeseries)
            max_activity = np.max(roi_timeseries)
            
            temporal_mean = np.mean(roi_timeseries, axis=0)
            peak_time = np.argmax(temporal_mean)
            
            roi_stats[f"r{region_id:03}"] = {
                'region_name': region_name,
                'mean': float(mean_activity),
                'std': float(std_activity),
                'max': float(max_activity),
                'peak_time': float(peak_time * segment_duration),
                'n_voxels': int(np.sum(mask))
            }
    
    log("\nEn aktif 10 beyin bölgesi:")
    sorted_rois = sorted(roi_stats.items(), key=lambda x: x[1]['mean'], reverse=True)[:10]
    for roi, stats in sorted_rois:
        log(f"  {stats['region_name']}: mean={stats['mean']:.3f}, peak_time={stats['peak_time']:.1f}s")
    
    return roi_stats

def compute_surface_roi_centers(surface_img, output_path, band_name):
    """Surface projection'dan ROI merkezlerini hesapla"""
    
    log("Surface-based ROI merkezleri hesaplanıyor...")
    
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
    all_labels = ['Background'] + ho_cort.labels
    
    from nilearn.image import resample_to_img
    surface_resampled = resample_to_img(surface_img, atlas_img, interpolation='nearest')
    
    surface_data = surface_resampled.get_fdata()
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    
    roi_centers_list = []
    unique_regions = np.unique(atlas_data[atlas_data > 0]).astype(int)
    
    for region_id in unique_regions:
        mask = atlas_data == region_id
        region_name = all_labels[region_id]
        
        active_mask = mask & (np.mean(surface_data, axis=3) > 0)
        
        if np.any(active_mask):
            active_indices = np.where(active_mask)
            
            mni_coords = []
            for i in range(len(active_indices[0])):
                vox_idx = [active_indices[0][i], active_indices[1][i], active_indices[2][i], 1]
                mni_coord = affine @ vox_idx
                mni_coords.append(mni_coord[:3])
            
            roi_center = np.mean(mni_coords, axis=0)
            
            roi_centers_list.append([
                region_id,
                region_name,
                roi_center[0],
                roi_center[1],
                roi_center[2],
                len(mni_coords)
            ])
    
    roi_centers_array = np.array(roi_centers_list, dtype=object)
    surface_centers_file = os.path.join(output_path, f"{band_name}_surface_roi_centers.npy")
    np.save(surface_centers_file, roi_centers_array)
    log(f"Surface ROI merkezleri kaydedildi: {surface_centers_file}")
    
    return roi_centers_array

# === ANA FONKSİYON === #
def main():
    start_time = datetime.now()
    log("="*60)
    log("EEG SOURCE LOCALIZATION - GPU ACCELERATED")
    log("="*60)
    
    # Sistem bilgisi
    if USE_GPU:
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = gpu_props['name'].decode('utf-8')
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        log(f"GPU: {gpu_name}")
        log(f"GPU Belleği: {total_mem/1024**3:.1f} GB (Boş: {free_mem/1024**3:.1f} GB)")
        
        if '3050' in gpu_name and total_mem/1024**3 < 6:
            log("RTX 3050 Ti Mobile (4GB) için özel optimizasyonlar aktif")
            log(f"  - Batch size: {batch_size}")
            log(f"  - Grid spacing: {grid_spacing}")
            log(f"  - CPU threads: {n_jobs}")
    else:
        log("CPU modunda çalışılıyor")
        import platform
        log(f"İşlemci: {platform.processor()}")
        log(f"CPU threads: {n_jobs}")

    try:
        # 1. EEG VERİSİNİ YÜKLE VE TEMİZLE
        log("\n1. EEG verisi yükleniyor...")
        mat = scipy.io.loadmat(eeg_mat)
        eeg = mat['dataRest'][:64].astype(np.float32)
        
        # Filtreleme
        if USE_GPU:
            log("GPU'da filtreleme yapılıyor...")
            try:
                eeg_gpu = cp.asarray(eeg)
                
                # Bandpass filter
                from cupyx.scipy.signal import butter as cp_butter, filtfilt as cp_filtfilt
                b, a = cp_butter(4, [1 / (fs / 2), 45 / (fs / 2)], btype='band')
                eeg_filtered = cp_filtfilt(b, a, eeg_gpu, axis=1)
                
                # Notch filter
                from cupyx.scipy.signal import iirnotch as cp_iirnotch
                b, a = cp_iirnotch(50 / (fs / 2), Q=30)
                eeg_filtered = cp_filtfilt(b, a, eeg_filtered, axis=1)
                
                eeg = eeg_filtered.get()
                del eeg_gpu, eeg_filtered
                clear_gpu_memory()
                
            except Exception as e:
                log(f"GPU filtreleme hatası: {e}. CPU'da devam ediliyor...")
                eeg = bandpass(eeg, fs, 1, 45)
                eeg = notch(eeg, fs)
        else:
            eeg = bandpass(eeg, fs, 1, 45)
            eeg = notch(eeg, fs)

        # ICA ile artefakt temizleme
        log("ICA uygulanıyor...")
        ica = FastICA(n_components=64, random_state=0, max_iter=1000, tol=1e-2)
        sources = ica.fit_transform(eeg.T).T
        mixing = ica.mixing_
        artifact_idx = np.argsort(np.std(sources, axis=1))[-2:]
        sources[artifact_idx, :] = 0
        eeg_clean = (mixing @ sources).T.T
        
        del eeg, sources, mixing
        gc.collect()

        # Segmentlere ayır
        log("Segmentlere ayrılıyor...")
        n_segments = eeg_clean.shape[1] // segment_samples
        segments = np.stack([
            eeg_clean[:, i*segment_samples:(i+1)*segment_samples]
            for i in range(n_segments)
        ])
        log(f"Toplam segment sayısı: {n_segments}")

        # 2. KOORDİNAT SİSTEMİNİ HAZIRLA
        log("\n2. Koordinat sistemi hazırlanıyor...")
        chan_coords = validate_and_transform_coords(coord_file)
        
        # Grid oluştur
        voxel_coords_all, grid_shape = create_voxel_grid(
            [-90, 91], [-126, 91], [-72, 109], grid_spacing
        )

        # 3. MASKELEME
        if USE_HARVARD_OXFORD:
            log("\n3. Harvard-Oxford atlas maskesi uygulanıyor...")
            voxel_coords, brain_indices, voxel_labels, region_ids, all_labels = apply_harvard_oxford_mask(voxel_coords_all)
            
            if USE_GRAY_MATTER_MASK:
                keep_indices = filter_white_matter_from_labels(voxel_labels, region_ids)
                voxel_coords = voxel_coords[keep_indices]
                brain_indices = brain_indices[keep_indices]
                voxel_labels = [voxel_labels[i] for i in keep_indices]
                region_ids = [region_ids[i] for i in keep_indices]
                log(f"Gray matter filtreleme sonrası: {len(voxel_coords)} voxel")
        
        elif USE_BRAIN_MASK:
            log("\n3. Beyin maskesi uygulanıyor...")
            voxel_coords, brain_indices = apply_brain_mask(voxel_coords_all, grid_spacing)
            voxel_labels = None
            region_ids = None
            all_labels = None
        else:
            voxel_coords = voxel_coords_all
            brain_indices = np.arange(len(voxel_coords_all))
            voxel_labels = None
            region_ids = None
            all_labels = None

        # 4. VOXEL SIGNATURES HESAPLA
        log("\n4. Voxel signatures hesaplanıyor...")
        if USE_GPU:
            dynamic_batch = calculate_dynamic_batch_size(len(voxel_coords), len(chan_coords) * 3)
            actual_batch_size = min(batch_size, dynamic_batch)
        else:
            actual_batch_size = batch_size

        voxel_sigs = compute_directional_voxel_signatures(
            voxel_coords, chan_coords, sigma=sigma, batch_size=actual_batch_size
        )
        
        if USE_GPU:
            check_gpu_memory()

        # 5. HER FREKANS BANDI İÇİN İŞLEM
        for band_name, band_range in bands.items():
            log(f"\n{'='*60}")
            log(f"{band_name.upper()} BANDI İŞLENİYOR ({band_range[0]}-{band_range[1]} Hz)")
            log(f"{'='*60}")
            
            band_start_time = time.time()
            
            # Hilbert envelope hesapla
            log("Hilbert envelope hesaplanıyor...")
            
            if USE_GPU:
                try:
                    snapshots = []
                    for seg_idx, seg in enumerate(tqdm(segments, desc=f"GPU Hilbert {band_name}")):
                        envelope = hilbert_envelope_gpu(seg, band_range, fs)
                        snapshots.append(envelope)
                        
                        if seg_idx % 50 == 0:
                            clear_gpu_memory()
                    
                    snapshots = np.stack(snapshots)
                    
                except Exception as e:
                    log(f"GPU Hilbert hatası: {e}. CPU'da devam ediliyor...")
                    snapshots = np.stack([hilbert_envelope(seg, band_range, fs) for seg in segments])
            else:
                snapshots = np.stack([hilbert_envelope(seg, band_range, fs) for seg in segments])

            # 6. SURFACE PROJECTION
            surface_img = None
            if USE_SURFACE_PROJECTION:
                log("\nSurface-based projection başlıyor...")
                
                if SURFACE_PROJECTION_METHOD == "enhanced" and USE_CORTICAL_RIBBON:
                    surface_data, vertices, n_lh, lh_p, rh_p, lh_w, rh_w = enhanced_surface_projection_fixed(
                        snapshots.T, chan_coords
                    )
                    
                    surface_volume, surface_affine = improved_surface_to_volume_fixed(
                        surface_data, vertices, lh_p, rh_p, lh_w, rh_w,
                        n_lh, grid_shape, grid_spacing
                    )
                else:
                    # Simplified version
                    surface_data, vertices, n_lh, lh_p, rh_p, lh_w, rh_w = enhanced_surface_projection_fixed(
                        snapshots.T, chan_coords
                    )
                    
                    surface_volume, surface_affine = improved_surface_to_volume_fixed(
                        surface_data, vertices, lh_p, rh_p, lh_w, rh_w,
                        n_lh, grid_shape, grid_spacing
                    )
                
                # Surface maps kaydet
                save_surface_maps(surface_data, band_name, n_lh, data_path)
                
                # Surface smoothing
                if APPLY_SMOOTHING:
                    log(f"Surface volume smoothing (sigma={SMOOTHING_SIGMA * SURFACE_SMOOTHING_FACTOR})...")
                    surface_volume = apply_spatial_smoothing(
                        surface_volume, 
                        SMOOTHING_SIGMA * SURFACE_SMOOTHING_FACTOR
                    )
                
                # Surface NIfTI kaydet
                surface_img = nib.Nifti1Image(surface_volume, surface_affine)
                surface_img.header['pixdim'][4] = segment_duration
                surface_img.header.set_qform(surface_affine, code=1)
                surface_img.header.set_sform(surface_affine, code=1)
                
                surface_output = os.path.join(data_path, f"{band_name}_surface_projection.nii.gz")
                nib.save(surface_img, surface_output)
                log(f"Surface projection kaydedildi: {surface_output}")

            # 7. VOXEL-BASED PROJECTION
            log("\nVoxel-based projection başlıyor...")
            
            # Bin edges hesapla
            log("Histogram bin sınırları hesaplanıyor...")
            v_bins = [np.histogram_bin_edges(voxel_sigs[i], bins=n_bins) 
                     for i in tqdm(range(len(voxel_sigs)), desc="Bin hesaplama")]
            
            # MI ve Dice skorları hesapla
            log("MI ve Dice skorları hesaplanıyor...")
            
            if USE_GPU:
                mi_scores, dice_scores = process_snapshots_gpu(voxel_sigs, snapshots, v_bins, n_jobs)
            else:
                mi_scores, dice_scores = process_snapshots_parallel(voxel_sigs, snapshots, v_bins, n_jobs)

            # Hybrid skorlar
            log("Hybrid skorlar hesaplanıyor...")
            x, y, z = grid_shape
            
            if USE_ADAPTIVE_WEIGHTS:
                # Temporal stability hesapla
                temp_nii_data = np.zeros((x, y, z, snapshots.shape[0]), dtype=np.float32)
                temp_hybrid = MI_WEIGHT * mi_scores + DICE_WEIGHT * dice_scores
                
                # Geçici volume oluştur
                inv_affine = np.linalg.inv(np.array([
                    [-grid_spacing, 0, 0, 90],
                    [0, grid_spacing, 0, -126], 
                    [0, 0, grid_spacing, -72],
                    [0, 0, 0, 1]
                ]))
                
                for idx in range(len(voxel_coords)):
                    mni_coord = voxel_coords[idx]
                    vox_idx = inv_affine @ np.append(mni_coord, 1)
                    xi, yi, zi = np.round(vox_idx[:3]).astype(int)
                    if 0 <= xi < x and 0 <= yi < y and 0 <= zi < z:
                        temp_nii_data[xi, yi, zi, :] = temp_hybrid[idx, :]
                
                temporal_stability = compute_temporal_stability(temp_nii_data)
                
                # Voxel stability map et
                voxel_stability = []
                for idx in range(len(voxel_coords)):
                    mni_coord = voxel_coords[idx]
                    vox_idx = inv_affine @ np.append(mni_coord, 1)
                    xi, yi, zi = np.round(vox_idx[:3]).astype(int)
                    if 0 <= xi < x and 0 <= yi < y and 0 <= zi < z:
                        voxel_stability.append(temporal_stability[xi, yi, zi])
                    else:
                        voxel_stability.append(0.5)
                
                voxel_stability = np.array(voxel_stability)
                hybrid_scores = adaptive_hybrid_scoring(mi_scores, dice_scores, voxel_stability)
            else:
                hybrid_scores = MI_WEIGHT * mi_scores + DICE_WEIGHT * dice_scores

            # Skorları optimize et
            log("Hybrid skorlar optimize ediliyor...")
            active_voxels = np.sum(np.max(hybrid_scores, axis=1) > 0)
            log(f"Aktif voxel sayısı: {active_voxels} / {len(hybrid_scores)}")
            
            # Normalize
            min_score = np.min(hybrid_scores)
            max_score = np.max(hybrid_scores)
            if max_score > min_score:
                hybrid_scores = (hybrid_scores - min_score) / (max_score - min_score)
            
            # Power transform
            hybrid_scores = np.power(hybrid_scores, 0.5)
            
            # Temporal consistency
            temporal_diff = np.diff(hybrid_scores, axis=1)
            temporal_smoothness = np.mean(np.abs(temporal_diff), axis=1)
            stability_weight = 1 / (1 + temporal_smoothness)
            hybrid_scores = hybrid_scores * stability_weight[:, np.newaxis]

            # 8. 3D VOLUME OLUŞTUR
            log("3D volume oluşturuluyor...")
            nii_data = np.zeros((x, y, z, snapshots.shape[0]), dtype=np.float32)
            
            # Affine matrix
            affine = np.array([
                [-grid_spacing, 0, 0, 90],
                [0, grid_spacing, 0, -126], 
                [0, 0, grid_spacing, -72],
                [0, 0, 0, 1]
            ])
            
            # Voxelleri yerleştir
            inv_affine = np.linalg.inv(affine)
            placed_count = 0
            
            for idx in tqdm(range(len(voxel_coords)), desc="Voxel yerleştirme"):
                mni_coord = voxel_coords[idx]
                vox_idx = inv_affine @ np.append(mni_coord, 1)
                xi, yi, zi = np.round(vox_idx[:3]).astype(int)
                
                if 0 <= xi < x and 0 <= yi < y and 0 <= zi < z:
                    nii_data[xi, yi, zi, :] = hybrid_scores[idx, :]
                    placed_count += 1
            
            log(f"Yerleştirilen voxel sayısı: {placed_count}/{len(voxel_coords)}")

            # 9. POST-PROCESSING
            log("\nPost-processing başlıyor...")
            
            # Komşu voxel interpolasyonu
            if grid_spacing <= 2:
                log("Komşu voxel interpolasyonu...")
                from scipy.ndimage import maximum_filter
                for t in range(nii_data.shape[3]):
                    nii_data[:,:,:,t] = maximum_filter(nii_data[:,:,:,t], size=3)
            
            # Z-score normalizasyon
            if APPLY_ZSCORE:
                log("Z-score normalizasyonu...")
                nii_data = apply_zscore_normalization(nii_data)
            
            # Threshold
            if APPLY_THRESHOLD:
                log(f"Threshold uygulanıyor (percentile={THRESHOLD_PERCENTILE})...")
                nii_data = apply_threshold(nii_data, THRESHOLD_PERCENTILE)
            
            # Spatial smoothing
            if APPLY_SMOOTHING:
                log(f"Spatial smoothing (sigma={SMOOTHING_SIGMA})...")
                nii_data = apply_spatial_smoothing(nii_data, SMOOTHING_SIGMA)
            
            # Bilateral filter
            if grid_spacing <= 2:
                log("Bilateral filter...")
                nii_data = apply_bilateral_filter(nii_data, spatial_sigma=1.5, intensity_sigma=0.15)
            
            # Morfolojik temizleme
            if APPLY_MORPHOLOGICAL_OPERATIONS:
                log("Morfolojik temizleme...")
                nii_data = apply_morphological_operations(nii_data, kernel_size=3)
            
            # Final smoothing
            log("Final smoothing...")
            nii_data = apply_spatial_smoothing(nii_data, 1.0)
            
            # Derin aktivite bastırma
            if SUPPRESS_DEEP_ACTIVITY:
                log("Derin beyin aktiviteleri bastırılıyor...")
                nii_data = suppress_deep_brain_activity(nii_data, affine, suppression_threshold=50)
            
            # Beyin sınırları kontrolü
            if ENFORCE_BRAIN_BOUNDS:
                nii_data = enforce_brain_boundaries(nii_data, affine)

            # 10. NIfTI KAYDET
            log("\nNIfTI dosyası kaydediliyor...")
            img = nib.Nifti1Image(nii_data, affine=affine)
            img.header['pixdim'][4] = segment_duration
            img.header.set_qform(affine, code=1)
            img.header.set_sform(affine, code=1)
            
            # Dosya adı oluştur
            atlas_suffix = "_harvard_oxford" if USE_HARVARD_OXFORD else "_brain_masked"
            post_suffix = ""
            if APPLY_ZSCORE: post_suffix += "_zscore"
            if APPLY_THRESHOLD: post_suffix += f"_thr{THRESHOLD_PERCENTILE}"
            if APPLY_SMOOTHING: post_suffix += f"_smooth{SMOOTHING_SIGMA}"
            if SUPPRESS_DEEP_ACTIVITY: post_suffix += "_cortical"
            
            voxel_output = os.path.join(data_path, 
                f"{band_name}_voxel{atlas_suffix}{post_suffix}.nii.gz")
            nib.save(img, voxel_output)
            log(f"Voxel-based NIfTI kaydedildi: {voxel_output}")
            
            # SPM uyumlu kaydet
            spm_output = os.path.join(data_path, f"{band_name}_voxel_spm.nii")
            nib.save(img, spm_output)
            log(f"SPM uyumlu dosya kaydedildi: {spm_output}")

            # 11. ROI ANALİZİ
            if ROI_BASED_ANALYSIS:
                log("\nROI analizi yapılıyor...")
                
                # ROI MERKEZLERİNİ HESAPLA VE KAYDET
                if voxel_labels is not None and region_ids is not None:
                    roi_centers_dict, roi_centers_coords = compute_and_save_roi_centers(
                        voxel_coords, 
                        region_ids, 
                        all_labels, 
                        data_path, 
                        band_name
                    )
                    
                    # Global ROI centers dosyası
                    global_roi_centers_file = os.path.join(data_path, "all_roi_centers.npy")
                    if not os.path.exists(global_roi_centers_file):
                        np.save(global_roi_centers_file, roi_centers_coords)
                        log(f"Global ROI merkezleri kaydedildi: {global_roi_centers_file}")
                
                # Surface projection ROI analizi
                if USE_SURFACE_PROJECTION and surface_img is not None:
                    surface_roi_stats = compute_roi_statistics_from_nifti(
                        surface_img, f"{band_name}_surface", data_path
                    )
                    
                    # Surface ROI centers
                    surface_roi_centers = compute_surface_roi_centers(
                        surface_img, data_path, band_name
                    )
                    
                    surface_roi_file = os.path.join(data_path, 
                        f'{band_name}_surface_roi_statistics.mat')
                    scipy.io.savemat(surface_roi_file, {
                        'roi_stats': surface_roi_stats,
                        'band': band_name,
                        'type': 'surface'
                    })
                    log(f"Surface ROI istatistikleri kaydedildi")
                
                # Voxel-based ROI analizi
                voxel_roi_stats = compute_roi_statistics_from_nifti(
                    img, f"{band_name}_voxel", data_path
                )
                
                voxel_roi_file = os.path.join(data_path, 
                    f'{band_name}_voxel_roi_statistics.mat')
                scipy.io.savemat(voxel_roi_file, {
                    'roi_stats': voxel_roi_stats,
                    'band': band_name,
                    'type': 'voxel'
                })
                log(f"Voxel ROI istatistikleri kaydedildi")

            # 12. CONN TOOLBOX OUTPUT
            if CREATE_CONN_OUTPUT:
                log("\nCONN toolbox output oluşturuluyor...")
                
                # Surface CONN output
                if USE_SURFACE_PROJECTION and surface_img is not None:
                    surface_roi_ts = create_conn_compatible_output(
                        surface_img,
                        f"{band_name}_surface",
                        None,
                        data_path
                    )
                    
                    if surface_roi_ts is not None:
                        connectivity = np.corrcoef(surface_roi_ts.T)
                        conn_file = os.path.join(data_path, 
                            f'{band_name}_surface_connectivity.mat')
                        scipy.io.savemat(conn_file, {
                            'connectivity': connectivity,
                            'band': band_name,
                            'type': 'surface'
                        })
                        log("Surface connectivity matrix kaydedildi")
                
                # Voxel CONN output
                voxel_roi_ts = create_conn_compatible_output(
                    img,
                    f"{band_name}_voxel",
                    None,
                    data_path
                )
                
                if voxel_roi_ts is not None:
                    connectivity = np.corrcoef(voxel_roi_ts.T)
                    conn_file = os.path.join(data_path, 
                        f'{band_name}_voxel_connectivity.mat')
                    scipy.io.savemat(conn_file, {
                        'connectivity': connectivity,
                        'band': band_name,
                        'type': 'voxel'
                    })
                    log("Voxel connectivity matrix kaydedildi")

            # 13. İSTATİSTİKLER
            log("\n=== Band İstatistikleri ===")
            active_mask = nii_data != 0
            if np.any(active_mask):
                log(f"Min değer: {np.min(nii_data[active_mask]):.4f}")
                log(f"Max değer: {np.max(nii_data[active_mask]):.4f}")
                log(f"Ortalama: {np.mean(nii_data[active_mask]):.4f}")
                log(f"Std sapma: {np.std(nii_data[active_mask]):.4f}")
                log(f"Aktif voxel sayısı: {np.sum(nii_data[:,:,:,0] != 0)}")
                
                # Kortikal aktivite oranı
                total_active = np.sum(nii_data[:,:,:,0] != 0)
                cortical_count = 0
                for xx in range(nii_data.shape[0]):
                    for yy in range(nii_data.shape[1]):
                        for zz in range(nii_data.shape[2]):
                            if nii_data[xx, yy, zz, 0] > 0:
                                voxel_idx = np.array([xx, yy, zz, 1])
                                mni_coord = affine @ voxel_idx
                                if mni_coord[2] > 0:  # Z > 0 (kortikal)
                                    cortical_count += 1
                
                cortical_percentage = (cortical_count / total_active * 100) if total_active > 0 else 0
                log(f"Kortikal aktivite oranı: {cortical_percentage:.1f}%")
            
            # Band işlem süresi
            band_duration = time.time() - band_start_time
            log(f"\n{band_name} bandı işlem süresi: {band_duration:.1f} saniye")
            
            # Belleği temizle
            del nii_data, mi_scores, dice_scores, hybrid_scores, snapshots
            gc.collect()
            
            if USE_GPU:
                clear_gpu_memory()
                check_gpu_memory()
            
            log(f"\n{band_name} bandı tamamlandı.")
            log("="*60)
            
            # RTX 3050 Ti için termal bekleme
            if USE_GPU and '3050' in gpu_name:
                log("GPU cooldown için 5 saniye bekleniyor...")
                time.sleep(5)

        # 14. TAMAMLANDI
        log("\n" + "="*60)
        log("TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
        log(f"Çıktı dizini: {data_path}")
        log("="*60)
        
        # Çıktı dosyalarını listele
        log("\nOluşturulan dosyalar:")
        output_files = []
        for band in bands.keys():
            # NIfTI dosyaları
            voxel_file = f"{band}_voxel_brain_masked_zscore_thr{THRESHOLD_PERCENTILE}_smooth{SMOOTHING_SIGMA}.nii.gz"
            if os.path.exists(os.path.join(data_path, voxel_file)):
                output_files.append(voxel_file)
                log(f"  ✓ {voxel_file}")
            
            # Surface dosyaları
            if USE_SURFACE_PROJECTION:
                surface_file = f"{band}_surface_projection.nii.gz"
                if os.path.exists(os.path.join(data_path, surface_file)):
                    output_files.append(surface_file)
                    log(f"  ✓ {surface_file}")
                
                # Surface maps
                for hemi in ['lh', 'rh']:
                    surface_map = f"{band}_surface_{hemi}.npy"
                    if os.path.exists(os.path.join(data_path, surface_map)):
                        output_files.append(surface_map)
                        log(f"  ✓ {surface_map}")
            
            # ROI dosyaları
            if ROI_BASED_ANALYSIS:
                roi_files = [
                    f"{band}_roi_centers.npy",
                    f"{band}_roi_centers.mat",
                    f"{band}_voxel_roi_statistics.mat",
                    f"{band}_surface_roi_statistics.mat"
                ]
                for roi_file in roi_files:
                    if os.path.exists(os.path.join(data_path, roi_file)):
                        output_files.append(roi_file)
                        log(f"  ✓ {roi_file}")
            
            # CONN dosyaları
            if CREATE_CONN_OUTPUT:
                conn_files = [
                    f"{band}_voxel_conn_roi_timeseries.mat",
                    f"{band}_voxel_connectivity.mat",
                    f"{band}_surface_conn_roi_timeseries.mat",
                    f"{band}_surface_connectivity.mat"
                ]
                for conn_file in conn_files:
                    if os.path.exists(os.path.join(data_path, conn_file)):
                        output_files.append(conn_file)
                        log(f"  ✓ {conn_file}")
        
        log(f"\nToplam {len(output_files)} dosya oluşturuldu.")
        
    except Exception as e:
        log(f"\n*** HATA OLUŞTU ***")
        log(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log(f"\nToplam işlem süresi: {int(duration // 60)} dakika {int(duration % 60)} saniye")
        
        # GPU belleğini tamamen temizle
        if USE_GPU:
            clear_gpu_memory()
            final_free, final_total = cp.cuda.runtime.memGetInfo()
            log(f"\nFinal GPU bellek durumu: {final_free/1024**2:.0f}/{final_total/1024**2:.0f} MB")
            
            # GPU sıcaklık kontrolü
            try:
                import nvidia_ml_py as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                log(f"GPU sıcaklığı: {temp}°C")
                
                if temp > 80:
                    log("UYARI: GPU çok ısındı! Bir süre bekleyin.")
                
                nvml.nvmlShutdown()
            except:
                pass

# === YARDIMCI FONKSİYONLAR === #
def benchmark_gpu():
    """GPU performans testi"""
    if not USE_GPU:
        log("GPU mevcut değil, benchmark yapılamıyor.")
        return
    
    log("\n=== GPU Benchmark ===")
    
    # Test parametreleri
    test_sizes = {
        'small': (1000, 64),
        'medium': (5000, 64),
        'large': (10000, 64),
        'xlarge': (20000, 64)
    }
    
    results = {}
    
    for size_name, (n_voxels, n_channels) in test_sizes.items():
        log(f"\nTest: {size_name} ({n_voxels} voxels, {n_channels} channels)")
        
        # Test verisi oluştur
        test_data = np.random.randn(n_voxels, n_channels).astype(np.float32)
        
        try:
            # Bellek kontrolü
            free_mem_before, total_mem = cp.cuda.runtime.memGetInfo()
            
            # GPU'ya yükle ve işlem yap
            start_time = time.time()
            
            data_gpu = cp.asarray(test_data)
            result_gpu = cp.sum(data_gpu, axis=1)
            cp.cuda.Device().synchronize()
            
            gpu_time = time.time() - start_time
            
            # Bellek kullanımı
            free_mem_after, _ = cp.cuda.runtime.memGetInfo()
            memory_used = (free_mem_before - free_mem_after) / 1024**2
            
            results[size_name] = {
                'success': True,
                'time': gpu_time,
                'memory_mb': memory_used
            }
            
            log(f"  ✓ Başarılı - Süre: {gpu_time:.3f}s, Bellek: {memory_used:.1f}MB")
            
            # Temizle
            del data_gpu, result_gpu
            clear_gpu_memory()
            
        except cp.cuda.memory.OutOfMemoryError:
            results[size_name] = {'success': False}
            log(f"  ✗ Bellek yetersiz!")
            clear_gpu_memory()
            break
    
    # Öneriler
    log("\n=== Öneriler ===")
    successful = [name for name, res in results.items() if res.get('success', False)]
    
    if 'xlarge' in successful:
        log("GPU büyük veri setlerini işleyebilir. batch_size = 10000 kullanabilirsiniz.")
    elif 'large' in successful:
        log("GPU orta-büyük veri setlerini işleyebilir. batch_size = 7500 önerilir.")
    elif 'medium' in successful:
        log("GPU orta veri setlerini işleyebilir. batch_size = 5000 önerilir.")
    else:
        log("GPU küçük veri setleriyle çalışmalı. batch_size = 3000 önerilir.")
    
    return results

# === ANA PROGRAM GİRİŞ NOKTASI === #
if __name__ == "__main__":
    log("="*60)
    log("EEG SOURCE LOCALIZATION - GPU ACCELERATED")
    log("Complete Edition with All Features")
    log("="*60)
    
    # Sistem kontrolü
    if USE_GPU:
        try:
            # GPU özelliklerini göster
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = props['name'].decode('utf-8')
            
            log(f"GPU bulundu: {gpu_name}")
            
            # RTX 3050 Ti kontrolü
            if '3050' in gpu_name:
                log("RTX 3050 Ti Mobile algılandı!")
                
                # Benchmark çalıştır (opsiyonel)
                run_benchmark = input("\nGPU benchmark çalıştırılsın mı? (y/n): ").lower() == 'y'
                if run_benchmark:
                    benchmark_gpu()
            
            # Bellek durumu
            check_gpu_memory()
            
        except Exception as e:
            log(f"GPU kontrolünde hata: {e}")
            USE_GPU = False
    
    # İşlem uyarısı
    log("\n=== İŞLEM BAŞLIYOR ===")
    if USE_GPU:
        log("GPU hızlandırması AKTİF")
        log(f"Batch size: {batch_size}")
        log(f"Grid spacing: {grid_spacing}mm")
    else:
        log("CPU modunda çalışılıyor")
    
    log("\nTahmini işlem süresi:")
    if USE_GPU:
        log("- GPU ile: 10-20 dakika")
    else:
        log("- CPU ile: 30-45 dakika")
    
    log("\nÇıktı dosyaları:")
    log("- NIfTI dosyaları (.nii.gz)")
    log("- Surface maps (.npy)")
    log("- ROI merkezleri (.npy, .mat)")
    log("- ROI istatistikleri (.mat)")
    log("- CONN toolbox dosyaları (.mat)")
    log("- Connectivity matrisleri (.mat)")
    
    # Onay al
    proceed = input("\nDevam etmek istiyor musunuz? (y/n): ").lower() == 'y'
    
    if proceed:
        # Ana işlemi başlat
        main()
        
        # Final kontroller
        if USE_GPU:
            log("\n=== GPU KULLANIM ÖZETİ ===")
            check_gpu_memory()
            
            # Sıcaklık kontrolü
            try:
                import nvidia_ml_py as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                log(f"GPU sıcaklığı: {temp}°C")
                
                if temp > 80:
                    log("UYARI: GPU çok ısındı! Bir süre bekleyin.")
                elif temp > 70:
                    log("GPU sıcaklığı yüksek, soğuması için bekleyin.")
                else:
                    log("GPU sıcaklığı normal.")
                
                # GPU kullanım istatistikleri
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                log(f"GPU kullanımı: {utilization.gpu}%")
                log(f"Bellek kullanımı: {utilization.memory}%")
                
                nvml.nvmlShutdown()
            except:
                pass
        
        # Çıktı özeti
        log("\n=== ÇIKTI ÖZETİ ===")
        
        # Her band için çıktıları kontrol et
        all_bands_complete = True
        for band in bands.keys():
            band_files = []
            
            # Voxel NIfTI
            voxel_file = f"{band}_voxel_brain_masked_zscore_thr{THRESHOLD_PERCENTILE}_smooth{SMOOTHING_SIGMA}.nii.gz"
            if os.path.exists(os.path.join(data_path, voxel_file)):
                band_files.append("Voxel NIfTI ✓")
            else:
                band_files.append("Voxel NIfTI ✗")
                all_bands_complete = False
            
            # Surface NIfTI
            if USE_SURFACE_PROJECTION:
                surface_file = f"{band}_surface_projection.nii.gz"
                if os.path.exists(os.path.join(data_path, surface_file)):
                    band_files.append("Surface NIfTI ✓")
                else:
                    band_files.append("Surface NIfTI ✗")
                    all_bands_complete = False
            
            # ROI analizi
            if ROI_BASED_ANALYSIS:
                roi_file = f"{band}_roi_centers.npy"
                if os.path.exists(os.path.join(data_path, roi_file)):
                    band_files.append("ROI Centers ✓")
                else:
                    band_files.append("ROI Centers ✗")
                    all_bands_complete = False
            
            # CONN output
            if CREATE_CONN_OUTPUT:
                conn_file = f"{band}_voxel_connectivity.mat"
                if os.path.exists(os.path.join(data_path, conn_file)):
                    band_files.append("Connectivity ✓")
                else:
                    band_files.append("Connectivity ✗")
                    all_bands_complete = False
            
            log(f"\n{band.upper()} band: {', '.join(band_files)}")
        
        if all_bands_complete:
            log("\n✓ TÜM ÇIKTILAR BAŞARIYLA OLUŞTURULDU!")
        else:
            log("\n⚠ BAZI ÇIKTILAR EKSİK!")
        
        # Disk kullanımı
        total_size = 0
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if any(band in file for band in bands.keys()):
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        
        log(f"\nToplam disk kullanımı: {total_size / 1024**2:.1f} MB")
        
        # İşlem önerileri
        log("\n=== SONRAKI ADIMLAR ===")
        log("1. NIfTI dosyalarını FSLeyes veya MRIcron ile görüntüleyin")
        log("2. Surface maps'leri Python'da matplotlib ile görselleştirin")
        log("3. CONN toolbox dosyalarını MATLAB'da yükleyin")
        log("4. ROI istatistiklerini analiz edin")
        
        # MATLAB komutları
        log("\nMATLAB'da yüklemek için:")
        log(">> data = load('delta_voxel_connectivity.mat');")
        log(">> imagesc(data.connectivity); colorbar;")
        
    else:
        log("\nİşlem iptal edildi.")
    
    log("\nProgram sonlandı.")
    
    # Temizlik
    if USE_GPU:
        clear_gpu_memory()
        log("GPU belleği temizlendi.")