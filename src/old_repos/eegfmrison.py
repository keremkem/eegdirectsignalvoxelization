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
from multiprocessing import cpu_count
from nilearn import datasets, image, maskers
from nilearn.maskers import NiftiLabelsMasker
from scipy.ndimage import gaussian_filter
from nilearn import surface, plotting
from scipy.spatial.distance import cdist
warnings.filterwarnings('ignore')

# === PARAMETRELER === #
data_path = r"Set Path Here"
eeg_mat = os.path.join(data_path, "Set EEG file as *.mat")
coord_file = os.path.join(data_path, "channel coordinations table fulfilled txt file with chanell name and x,y,z coordinates suitable with mni")
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

MI_WEIGHT = 0.2  
DICE_WEIGHT = 0.8  
grid_spacing = 1.5  
batch_size = 5000
n_jobs = min(12, cpu_count())

# Atlas Options
USE_BRAIN_MASK = True
USE_HARVARD_OXFORD = False
USE_TISSUE_TYPE = False
USE_GRAY_MATTER_MASK = True
CREATE_CONN_OUTPUT = True
ROI_BASED_ANALYSIS = True
USE_SURFACE_PROJECTION = True

# Parameters for Surface Projection 
SURFACE_PROJECTION_METHOD = "enhanced"
USE_CORTICAL_RIBBON = True
SURFACE_SMOOTHING_FACTOR = 1.5

# Post-processing parameters
APPLY_THRESHOLD = True
THRESHOLD_PERCENTILE = 70 
APPLY_SMOOTHING = True
SMOOTHING_SIGMA = 0.8  
APPLY_ZSCORE = True
ENFORCE_BRAIN_BOUNDS = True 
APPLY_MORPHOLOGICAL_OPERATIONS = False 


# Hybrid score parameters
USE_ADAPTIVE_WEIGHTS = True
SPATIAL_PRIOR_WEIGHT = 0.0
MI_WEIGHT = 0.25  
DICE_WEIGHT = 0.75 
SUPPRESS_DEEP_ACTIVITY = False 

# === LOGGING === #
def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def validate_and_transform_coords(coord_file):
    coords = {}
    with open(coord_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                name, x, y, z = parts
                x, y, z = float(x), float(y), float(z)

                z += 20  # -3 olan kanallar +17'ye gelir
                
                coords[name] = np.array([x, y, z])
				
    positions = np.array(list(coords.values()))
    log(f"Kanal koordinat aralığı (düzeltilmiş):")
    log(f"  X: [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}]")
    log(f"  Y: [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]")
    log(f"  Z: [{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")
    
    radii = np.sqrt(np.sum(positions**2, axis=1))
    log(f"  Ortalama yarıçap: {radii.mean():.1f}mm (min: {radii.min():.1f}, max: {radii.max():.1f})")
    
    if 'Cz' in coords:
        log(f"  Cz: {coords['Cz']}")
    if 'Fpz' in coords:
        log(f"  Fpz: {coords['Fpz']}")
    if 'Oz' in coords:
        log(f"  Oz: {coords['Oz']}")
    
    return coords

def apply_threshold(data, percentile=85):
    threshold = np.percentile(data[data > 0], percentile)
    data_thresholded = data.copy()
    data_thresholded[data_thresholded < threshold] = 0
    return data_thresholded

def apply_spatial_smoothing(data, sigma=0.5):
    smoothed = np.zeros_like(data)
    for t in range(data.shape[3]):
        smoothed[:,:,:,t] = gaussian_filter(data[:,:,:,t], sigma=sigma)
    return smoothed
    
def apply_morphological_operations(data, kernel_size=3):
    from scipy.ndimage import binary_opening, binary_closing, binary_dilation
    
    cleaned_data = np.zeros_like(data)
    
    for t in range(data.shape[3]):
        slice_data = data[:,:,:,t]
        threshold = np.percentile(slice_data[slice_data > 0], 30)
        binary_mask = slice_data > threshold
        
        # Morphological operations
        binary_mask = binary_opening(binary_mask, iterations=1)
        binary_mask = binary_closing(binary_mask, iterations=1)
        binary_mask = binary_dilation(binary_mask, iterations=1)
        
        # Apply Mask
        cleaned_data[:,:,:,t] = slice_data * binary_mask
    
    return cleaned_data

def apply_bilateral_filter(data, spatial_sigma=2.0, intensity_sigma=0.1):
    """Bilateral filter - edge-preserving smoothing"""
    from scipy.ndimage import gaussian_filter
    
    filtered_data = np.zeros_like(data)
    
    for t in range(data.shape[3]):
        slice_data = data[:,:,:,t]
        
        # Simple Bilateral Approach
        smoothed = gaussian_filter(slice_data, sigma=spatial_sigma)
        
        # Weight by Intensity difference
        diff = np.abs(slice_data - smoothed)
        weight = np.exp(-diff**2 / (2 * intensity_sigma**2))
        
        filtered_data[:,:,:,t] = slice_data * (1 - weight) + smoothed * weight
    
    return filtered_data

def apply_zscore_normalization(data):
    """z-score normalization for every segment"""
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
    """Calculate temporal stability for every voxel"""
    mask = data[:,:,:,0] != 0
    stability = np.zeros(data.shape[:3])
    
    # Temporal variation coefficient for active voxels
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
def compute_and_save_roi_centers(voxel_coords, region_ids, all_labels, output_path, band_name):
  
    
    log("Calculating ROI center coordinates...")
    
    unique_regions = np.unique(region_ids)
    roi_centers_dict = {}
    roi_centers_list = []
    
    # Filter deep structures (opsiyonel)
    deep_structures = ['Thalamus', 'Putamen', 'Pallidum', 'Caudate', 'Accumbens', 
                      'Hippocampus', 'Amygdala', 'Brain-Stem']
    
    for region_id in unique_regions:
        if region_id > 0:  # Background'u atla
            region_name = all_labels[region_id] if all_labels else f"Region_{region_id}"
            
            # Derin yapıları atlamak isterseniz (opsiyonel)
            # if any(deep in region_name for deep in deep_structures):
            #     continue
            
            mask = np.array(region_ids) == region_id
            region_voxels = np.array(voxel_coords)[mask]
            
            if len(region_voxels) > 0:
                # Calculate center coordinates (MNI coordinates)
                roi_center = np.mean(region_voxels, axis=0)
                
                # Add to Dictionary
                roi_centers_dict[f"r{region_id:03d}"] = {
                    'region_name': region_name,
                    'center_mni': roi_center,
                    'n_voxels': len(region_voxels)
                }
                
                # Save as a list
                roi_centers_list.append([
                    region_id,
                    region_name,
                    roi_center[0],  # X
                    roi_center[1],  # Y
                    roi_center[2],  # Z
                    len(region_voxels)
                ])
    
    # Save as a nump array
    roi_centers_array = np.array(roi_centers_list, dtype=object)
    
    # Main ROI centers file
    roi_centers_file = os.path.join(output_path, f"{band_name}_roi_centers.npy")
    np.save(roi_centers_file, roi_centers_array)
    log(f"ROI merkez koordinatları kaydedildi: {roi_centers_file}")
    log(f"  Toplam ROI sayısı: {len(roi_centers_list)}")
    
    # A simple array for only coordinates 
    centers_only = np.array([[item[2], item[3], item[4]] for item in roi_centers_list])
    centers_only_file = os.path.join(output_path, f"{band_name}_roi_centers_coords_only.npy")
    np.save(centers_only_file, centers_only)
    
    # MATLAB uyumlu format
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
    
    # Show biggest 10 ROI's
    sorted_rois = sorted(roi_centers_list, key=lambda x: x[5], reverse=True)[:10]
    log("\nEn büyük 10 ROI:")
    for roi in sorted_rois:
        log(f"  {roi[1]}: {roi[5]} voxel, merkez=({roi[2]:.1f}, {roi[3]:.1f}, {roi[4]:.1f})")
    
    return roi_centers_dict, centers_only

def adaptive_hybrid_scoring(mi_scores, dice_scores, temporal_stability=None):
    """Calculate hybrid score with adaptive weighting """
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
        # Simple Weighting
        hybrid = MI_WEIGHT * mi_scores + DICE_WEIGHT * dice_scores
    
    return hybrid
def suppress_deep_brain_activity(data, affine, suppression_threshold=50):
    """Supress deep brain activities more agressive"""
    log("Supressing Deep Brain Activity...")
    suppressed = data.copy()
    
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                # Calculate voxel's MNI coordinates
                voxel_idx = np.array([x, y, z, 1])
                mni_coord = affine @ voxel_idx
                
                # Supress by z coordinates
                z_coord = mni_coord[2]
                
                # Multiple layers z coordinates supress
                if z_coord < -10:  # under ear level
                    suppressed[x, y, z, :] = 0  # Totally zero
                elif z_coord < 10:  # low level
                    suppressed[x, y, z, :] *= 0.3
                elif z_coord < 30:  # Mid level
                    suppressed[x, y, z, :] *= 0.6
                
                # control with y coordinates ( too front or too rear)
                y_coord = mni_coord[1]
                if abs(y_coord) > 70:  
                    suppressed[x, y, z, :] *= 0.8
                
                # control for distnace from center
                radius = np.linalg.norm(mni_coord[:3])
                if radius < 50:  
                    suppressed[x, y, z, :] *= 0.1
    
    return suppressed
# === Helper Functions === #
def bandpass(data, fs, low, high):
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=1)

def notch(data, fs, freq=50.0):
    b, a = iirnotch(freq / (fs / 2), Q=30)
    return filtfilt(b, a, data, axis=1)

def hilbert_envelope(segment, band, fs):
    filtered = bandpass(segment, fs, band[0], band[1])
    analytic = hilbert(filtered, axis=1)
    return np.abs(analytic).mean(axis=1)

def create_voxel_grid(x_range, y_range, z_range, spacing):
    """Create Voxel Grid in MNI Space"""
    xs = np.arange(x_range[0], x_range[1], spacing)
    ys = np.arange(y_range[0], y_range[1], spacing)
    zs = np.arange(z_range[0], z_range[1], spacing)
    grid = np.array(np.meshgrid(xs, ys, zs, indexing='ij'))
    coords = grid.reshape(3, -1).T
    shape = (len(xs), len(ys), len(zs))
    log(f"Grid boyutu: {shape}, Total voxel: {len(coords)}")
    return coords, shape

def filter_white_matter_from_labels(voxel_labels, region_ids):
    """Discard white matter and related parts from Harvard-Oxford labels """
    exclude_keywords = [
        'White Matter',
        'Ventricle', 
        'Background',
        'Brain-Stem',
        'vessel',
        'CSF',
        'Corpus Callosum',
        'Cerebral White Matter',
        'Lateral Ventricle',
        'Third Ventricle',
        'Fourth Ventricle',
        'Brainstem',
        'Cerebellar White Matter'
    ]
    
    keep_indices = []
    excluded_regions = []
    
    for i, label in enumerate(voxel_labels):
        # Case-insensitive compairment
        if not any(keyword.lower() in label.lower() for keyword in exclude_keywords):
            keep_indices.append(i)
        else:
            if label not in excluded_regions:
                excluded_regions.append(label)
    
    # Log which areas excluded
    if excluded_regions:
        log("Excluded Regions:")
        for region in excluded_regions:
            log(f"  - {region}")
    
    log(f"Filtreleme sonucu: {len(keep_indices)} / {len(voxel_labels)} voxel tutuldu")
    
    return keep_indices
    
def apply_gray_matter_mask(voxel_coords, voxel_labels, valid_indices=None):
    """White matter filtreleme wrapper"""
    log("Gray matter filtreleme uygulanıyor...")
    
    # Filter White matter and ventricles
    keep_indices = filter_white_matter_from_labels(voxel_labels, None)
    
    # Filter Coordinates
    gray_voxels = voxel_coords[keep_indices]
    
    # Update if Valid indices occur 
    if valid_indices is not None:
        gray_indices = valid_indices[keep_indices]
    else:
        gray_indices = np.array(keep_indices)
    
    log(f"Gray matter filtreleme sonrası: {len(gray_voxels)} / {len(voxel_coords)} voxel")
    
    return gray_voxels, gray_indices
    
def fix_affine_for_harvard_oxford(atlas_img):
    """fix affine for Harvard-Oxford atlas"""
    affine = atlas_img.affine.copy()
    if np.any(np.abs(affine[:3, :3] - np.diag([2, 2, 2])) > 1e-3):
        fixed_affine = np.array([
            [2, 0, 0, -90],
            [0, 2, 0, -126],
            [0, 0, 2, -72],
            [0, 0, 0, 1]
        ])
        fixed_img = nib.Nifti1Image(atlas_img.get_fdata(), affine=fixed_affine)
        return fixed_img
    return atlas_img
# === Corrected SURFACE PROJECTION Functions === #
def enhanced_surface_projection_fixed(band_envelope, channel_coords):
    """Corrected enhanced surface projection"""
    log("Starting Enhanced surface projection ...")
    
    # Install Surface 
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    
    # Install Pial and white matter 
    lh_pial, _ = surface.load_surf_mesh(fsaverage['pial_left'])
    rh_pial, _ = surface.load_surf_mesh(fsaverage['pial_right'])
    lh_white, _ = surface.load_surf_mesh(fsaverage['white_left'])
    rh_white, _ = surface.load_surf_mesh(fsaverage['white_right'])
    
    # Calculate Mid-surface 
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
    
    # Channel positions
    ch_positions = np.array(list(channel_coords.values()))
    
    # Projeksiyon
    n_vertices = len(all_vertices)
    n_timepoints = band_envelope.shape[1]
    surface_data = np.zeros((n_vertices, n_timepoints))
    
    # Batch processing with wider sigma
    for v_start in tqdm(range(0, n_vertices, 1000), desc="Surface projection"):
        v_end = min(v_start + 1000, n_vertices)
        batch_vertices = all_vertices[v_start:v_end]
        
        # Distance hesapla
        distances = cdist(batch_vertices, ch_positions)
        
        # Gaussian weights - daha geniş sigma
        weights = np.exp(-distances**2 / (2 * 30**2))  # 60mm sigma
        
        # Normalize weights - DÜZELTİLMİŞ
        row_sums = weights.sum(axis=1, keepdims=True)
        valid_rows = row_sums.squeeze() > 0
        
        if np.any(valid_rows):
            weights[valid_rows] = weights[valid_rows] / row_sums[valid_rows]
        
        # Projection
        surface_data[v_start:v_end, :] = weights @ band_envelope
    
    return (surface_data, all_vertices, len(lh_mid), 
            lh_pial_mni, rh_pial_mni, lh_white_mni, rh_white_mni)

def improved_surface_to_volume_fixed(surface_data, vertices, pial_lh, pial_rh, 
                                   white_lh, white_rh, n_lh,
                                   volume_shape=(91, 109, 91), grid_spacing=2):
    """Corrected cortical ribbon filling"""
    log("Starting Cortical ribbon filling...")
    
    volume_4d = np.zeros((*volume_shape, surface_data.shape[1]))
    
    # MNI152 standart affine 
    affine = np.array([
        [-2, 0, 0, 90],      # X flip for neurological convention
        [0, 2, 0, -126], 
        [0, 0, 2, -72],
        [0, 0, 0, 1]
    ])
    
    inv_affine = np.linalg.inv(affine)
    
    # ribbon filling for left hemisphere
    for v_idx in tqdm(range(n_lh), desc="LH ribbon"):
        for alpha in np.linspace(0, 1, 10):
            # Interpolation from pial to white matter
            point = pial_lh[v_idx] * (1 - alpha) + white_lh[v_idx] * alpha
            
            # Turn voxel index to MNI coordinates
            vox_float = inv_affine[:3, :3] @ point + inv_affine[:3, 3]
            vox = np.round(vox_float).astype(int)
            
            if (0 <= vox[0] < volume_shape[0] and
                0 <= vox[1] < volume_shape[1] and
                0 <= vox[2] < volume_shape[2]):
                
                # (3x3x3 kernel)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            x, y, z = vox[0]+dx, vox[1]+dy, vox[2]+dz
                            if (0 <= x < volume_shape[0] and
                                0 <= y < volume_shape[1] and
                                0 <= z < volume_shape[2]):
                                # Distance-based weighting
                                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                                weight = np.exp(-dist/1.5)
                                # Get Maximum (not overwrite )
                                volume_4d[x, y, z, :] = np.maximum(
                                    volume_4d[x, y, z, :],
                                    surface_data[v_idx, :] * weight
                                )
    
    #  ribbon filling for right hemisphere
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
                                volume_4d[x, y, z, :] = np.maximum(
                                    volume_4d[x, y, z, :],
                                    surface_data[n_lh + v_idx, :] * weight
                                )
    
    return volume_4d, affine

def windows_friendly_surface_projection(band_envelope, channel_coords):
    log("Surface template yükleniyor...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    
    # Get Surface vertices 
    lh_coords, lh_faces = surface.load_surf_mesh(fsaverage['pial_left'])
    rh_coords, rh_faces = surface.load_surf_mesh(fsaverage['pial_right'])
    
    # FreeSurfer to MNI transformation
    fs_to_mni = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # Transform to MNI
    lh_coords_mni = np.array([(fs_to_mni @ np.append(v, 1))[:3] for v in lh_coords])
    rh_coords_mni = np.array([(fs_to_mni @ np.append(v, 1))[:3] for v in rh_coords])
    all_vertices = np.vstack([lh_coords_mni, rh_coords_mni])
    
    log(f"Total vertex count: {len(all_vertices)}")
    
    # Channel positions
    ch_positions = np.array(list(channel_coords.values()))
    
    # Projection
    n_vertices = len(all_vertices)
    n_timepoints = band_envelope.shape[1]
    surface_data = np.zeros((n_vertices, n_timepoints))
    
    for v_start in tqdm(range(0, n_vertices, 1000), desc="Surface projection"):
        v_end = min(v_start + 1000, n_vertices)
        batch_vertices = all_vertices[v_start:v_end]
        
        # Distance matrix
        distances = cdist(batch_vertices, ch_positions)
        
        # Gaussian weights
        weights = np.exp(-distances**2 / (2 * 30**2))  # 60mm sigma
        weights /= weights.sum(axis=1, keepdims=True)
        
        # Weighted projection
        surface_data[v_start:v_end, :] = weights @ band_envelope
    
    return surface_data, all_vertices, len(lh_coords)

def surface_to_volume_simple(surface_data, vertices, volume_shape=(91, 109, 91), grid_spacing=2):

    volume_4d = np.zeros((*volume_shape, surface_data.shape[1]))
    
    affine = np.array([
        [-2, 0, 0, 90],
        [0, 2, 0, -126], 
        [0, 0, 2, -72],
        [0, 0, 0, 1]
    ])
    
    inv_affine = np.linalg.inv(affine)
    
    for v_idx, vertex in enumerate(tqdm(vertices, desc="Surface to volume")):
        # MNI to voxel
        vox_float = inv_affine[:3, :3] @ vertex + inv_affine[:3, 3]
        vox = np.round(vox_float).astype(int)
        
        if (0 <= vox[0] < volume_shape[0] and
            0 <= vox[1] < volume_shape[1] and
            0 <= vox[2] < volume_shape[2]):
            
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        x, y, z = vox[0]+dx, vox[1]+dy, vox[2]+dz
                        if (0 <= x < volume_shape[0] and
                            0 <= y < volume_shape[1] and
                            0 <= z < volume_shape[2]):
                            dist = np.sqrt(dx**2 + dy**2 + dz**2)
                            weight = np.exp(-dist/1.5)
                            volume_4d[x, y, z, :] = np.maximum(
                                volume_4d[x, y, z, :],
                                surface_data[v_idx, :] * weight
                            )
    
    return volume_4d, affine

def save_surface_maps(surface_data, band_name, n_lh_vertices, output_path):
    lh_data = surface_data[:n_lh_vertices]
    rh_data = surface_data[n_lh_vertices:]
    
    np.save(os.path.join(output_path, f'{band_name}_surface_lh.npy'), lh_data)
    np.save(os.path.join(output_path, f'{band_name}_surface_rh.npy'), rh_data)
    
    log(f"Saved Surface Maps: {band_name}_surface_[lh/rh].npy")
									
# === Atlas and Mask functions === #
def apply_harvard_oxford_mask(voxel_coords):
    log("Installing Harvard-Oxford atlas ...")

    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')

    cort_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
    sub_img = ho_sub.maps if isinstance(ho_sub.maps, nib.Nifti1Image) else nib.load(ho_sub.maps)

    cort_img = fix_affine_for_harvard_oxford(cort_img)
    sub_img = fix_affine_for_harvard_oxford(sub_img)

    cort_data = cort_img.get_fdata()
    sub_data = sub_img.get_fdata()

    all_labels = ['Background'] + ho_cort.labels + ho_sub.labels

    atlas_data = cort_data.copy()
    sub_mask = sub_data > 0
    atlas_data[sub_mask] = sub_data[sub_mask] + len(ho_cort.labels)

    atlas_affine = cort_img.affine
    inv_affine = np.linalg.inv(atlas_affine)

    valid_voxels, valid_indices, voxel_labels, region_ids = [], [], [], []

    for i, coord in enumerate(tqdm(voxel_coords, desc="Applying Harvard-Oxford maskesi")):
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
    log(f"Voxel Count in Harvard-Oxford: {len(valid_voxels)} / {len(voxel_coords)}")

    unique_regions, counts = np.unique(voxel_labels, return_counts=True)
    log("Areas has most voxel count:")
    for region, count in sorted(zip(unique_regions, counts), key=lambda x: x[1], reverse=True)[:10]:
        log(f"  {region}: {count} voxel")

    return valid_voxels, np.array(valid_indices), voxel_labels, region_ids, all_labels

def apply_brain_mask(voxel_coords, grid_spacing):
    """MNI152 Brain mask"""
    log("Applying Brain Mask  ...")
    
    mni_mask = datasets.load_mni152_brain_mask()
    mask_data = mni_mask.get_fdata()
    mask_affine = mni_mask.affine
    
    brain_voxels = []
    brain_indices = []
    
    for i, coord in enumerate(tqdm(voxel_coords, desc="Applying Brain Mask")):
        voxel_idx = np.round(np.linalg.inv(mask_affine) @ np.append(coord, 1))[:3].astype(int)
        
        if (0 <= voxel_idx[0] < mask_data.shape[0] and
            0 <= voxel_idx[1] < mask_data.shape[1] and
            0 <= voxel_idx[2] < mask_data.shape[2]):
            
            if mask_data[voxel_idx[0], voxel_idx[1], voxel_idx[2]] > 0:
                brain_voxels.append(coord)
                brain_indices.append(i)
    
    brain_voxels = np.array(brain_voxels)
    log(f"Voxel Count in Brain: {len(brain_voxels)} / {len(voxel_coords)}")
    
    return brain_voxels, np.array(brain_indices)
# === VOXEL SIGNATURE and MI/DICE Calculation === #
def compute_directional_voxel_signatures(voxel_coords, channel_coords, sigma=35, batch_size=5000, angle_exponent=0.5):
    channel_names = list(channel_coords.keys())
    channel_vectors = np.stack([channel_coords[ch] for ch in channel_names])
    
    log(f"\nEEG 10-10 Channel Analyse:")
    log(f"  Total Channel Count: {len(channel_vectors)}")
    log(f"  Sigma (spatial spread): {sigma}mm")
    
    n_voxels = len(voxel_coords)
    n_channels = len(channel_vectors)
    voxel_signatures = np.zeros((n_voxels, n_channels), dtype=np.float32)
    
    # Voxel Statistics
    voxel_radii = np.linalg.norm(voxel_coords, axis=1)
    log(f"\nVoxel statistics:")
    log(f"  Toplam voxel: {n_voxels}")
    log(f"  Average voxel radius: {np.mean(voxel_radii):.1f}mm")
    
    # Calculate for every voxel
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
    
    signature_sums = np.sum(voxel_signatures, axis=1)
    zero_signatures = np.sum(signature_sums == 0)
    if zero_signatures > 0:
        log(f"UYARI: {zero_signatures} can not calculate signature for voxel!")
    
    return voxel_signatures

def compute_mi_dice_for_snapshot(voxel_sigs, snapshot, v_bins, t):
    """Calculate MI and Dice for one snapshot"""
    s_bins = np.histogram_bin_edges(snapshot, bins=n_bins)
    sb = np.digitize(snapshot, bins=s_bins)
    
    n_voxels = len(voxel_sigs)
    mi_results = np.zeros(n_voxels, dtype=np.float32)
    
    # MI calculation
    for i in range(n_voxels):
        vb = np.digitize(voxel_sigs[i], bins=v_bins[i])
        mi_results[i] = mutual_info_score(vb, sb)
    
    # Dice calculation
    numerator = 2 * (voxel_sigs @ snapshot)
    norm_voxels = np.sum(voxel_sigs**2, axis=1)
    norm_snap = np.sum(snapshot**2)
    denominator = norm_voxels + norm_snap + 1e-8
    dice_results = numerator / denominator
    
    # Min-max normalization (0-1 aralığına çek)
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
# === CONN TOOLBOX and ROI ANALİZ functions === #
def create_conn_compatible_output(nii_img, band_name, region_labels, output_path):
    log("CONN toolbox uyumlu çıktı oluşturuluyor...")
    
    # Harvard-Oxford atlas ile ROI time series çıkar
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    # Atlas yükleme
    atlas_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
    
    # SORUN BURADA: region_labels voxel-based projection'dan geliyor
    # ama nii_img surface'den veya farklı bir projection'dan gelebilir
    
    # Masker oluştur
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=False,
        detrend=False,
        verbose=0
    )
    
    # ROI time series çıkar
    roi_timeseries = masker.fit_transform(nii_img)
    
    # ROI isimlerini atlas'tan al, region_labels'dan değil
    all_roi_names = ['Background'] + ho_cort.labels + ho_sub.labels
    
    # Sadece sıfır olmayan ROI'leri tut
    active_rois = []
    active_timeseries = []
    for i in range(roi_timeseries.shape[1]):
        if np.any(roi_timeseries[:, i] != 0):
            active_rois.append(all_roi_names[i+1])  # +1 çünkü Background'u atlıyoruz
            active_timeseries.append(roi_timeseries[:, i])
    
    if active_timeseries:
        active_timeseries = np.column_stack(active_timeseries)
    else:
        log("UYARI: Hiç aktif ROI bulunamadı!")
        active_timeseries = roi_timeseries
        active_rois = all_roi_names[1:roi_timeseries.shape[1]+1]
    
    # CONN formatında kaydet
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
    
    # Harvard-Oxford atlas
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    atlas_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
    all_labels = ['Background'] + ho_cort.labels + ho_sub.labels
    
    # Resample NIfTI to atlas space if needed
    from nilearn.image import resample_to_img
    nii_resampled = resample_to_img(nii_img, atlas_img, interpolation='nearest')
    
    # Get data
    nii_data = nii_resampled.get_fdata()
    atlas_data = atlas_img.get_fdata()
    
    roi_stats = {}
    
    # Her ROI için istatistik
    unique_regions = np.unique(atlas_data[atlas_data > 0]).astype(int)
    
    for region_id in unique_regions:
        mask = atlas_data == region_id
        region_name = all_labels[region_id]
        
        # ROI içindeki voxellerin time series'i
        roi_timeseries = nii_data[mask, :]
        
        if roi_timeseries.size > 0 and np.any(roi_timeseries > 0):
            # Temporal ortalama
            mean_activity = np.mean(roi_timeseries)
            std_activity = np.std(roi_timeseries)
            max_activity = np.max(roi_timeseries)
            
            # En aktif zaman noktası
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
    
    # En aktif 10 bölgeyi yazdır
    log("\nEn aktif 10 beyin bölgesi:")
    sorted_rois = sorted(roi_stats.items(), key=lambda x: x[1]['mean'], reverse=True)[:10]
    for roi, stats in sorted_rois:
        log(f"  {stats['region_name']}: mean={stats['mean']:.3f}, peak_time={stats['peak_time']:.1f}s")
    
    return roi_stats
def enforce_brain_boundaries(data, affine):
    """Beyin sınırları dışındaki aktiviteleri sıfırla"""
    log("Beyin sınırları kontrolü...")
    
    # MNI152 beyin sınırları (yaklaşık)
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
def compute_surface_roi_centers(surface_img, output_path, band_name):
    """Surface projection'dan ROI merkezlerini hesapla"""
    
    log("Surface-based ROI merkezleri hesaplanıyor...")
    
    # Harvard-Oxford atlas
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = ho_cort.maps if isinstance(ho_cort.maps, nib.Nifti1Image) else nib.load(ho_cort.maps)
    all_labels = ['Background'] + ho_cort.labels
    
    # Surface image'ı atlas space'e resample et
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
        
        # Bu ROI'deki aktif voxelleri bul
        active_mask = mask & (np.mean(surface_data, axis=3) > 0)
        
        if np.any(active_mask):
            # Aktif voxellerin indekslerini al
            active_indices = np.where(active_mask)
            
            # Her indeksi MNI koordinatına çevir
            mni_coords = []
            for i in range(len(active_indices[0])):
                vox_idx = [active_indices[0][i], active_indices[1][i], active_indices[2][i], 1]
                mni_coord = affine @ vox_idx
                mni_coords.append(mni_coord[:3])
            
            # Merkezi hesapla
            roi_center = np.mean(mni_coords, axis=0)
            
            roi_centers_list.append([
                region_id,
                region_name,
                roi_center[0],
                roi_center[1],
                roi_center[2],
                len(mni_coords)
            ])
    
    # Kaydet
    roi_centers_array = np.array(roi_centers_list, dtype=object)
    surface_centers_file = os.path.join(output_path, f"{band_name}_surface_roi_centers.npy")
    np.save(surface_centers_file, roi_centers_array)
    log(f"Surface ROI merkezleri kaydedildi: {surface_centers_file}")
    
    return roi_centers_array
def compute_roi_statistics(hybrid_scores, region_ids, region_labels, all_labels):
    """Her ROI için istatistikler hesapla"""
    unique_regions = np.unique(region_ids)
    roi_stats = {}
    
    # Derin yapıları filtrele - YENİ
    deep_structures = ['Thalamus', 'Putamen', 'Pallidum', 'Caudate', 'Accumbens', 
                      'Hippocampus', 'Amygdala', 'Brain-Stem']

    for region_id in unique_regions:
        if region_id > 0:  # 0 = background
            mask = np.array(region_ids) == region_id
            region_name = all_labels[region_id]
            region_key = f"r{region_id:03}"
            
            # Derin yapıları atla - YENİ
            if any(deep in region_name for deep in deep_structures):
                continue

            # Temporal ortalama
            mean_activity = np.mean(hybrid_scores[mask, :])
            std_activity = np.std(hybrid_scores[mask, :])
            max_activity = np.max(hybrid_scores[mask, :])

            # En aktif zaman noktası
            temporal_mean = np.mean(hybrid_scores[mask, :], axis=0)
            peak_time = np.argmax(temporal_mean)

            roi_stats[region_key] = {
                'region_name': region_name,
                'mean': float(mean_activity),
                'std': float(std_activity),
                'max': float(max_activity),
                'peak_time': float(peak_time * segment_duration),
                'n_voxels': int(np.sum(mask))
            }

    # En aktif 10 bölgeyi yazdır
    log("\nEn aktif 10 beyin bölgesi:")
    sorted_rois = sorted(roi_stats.items(), key=lambda x: x[1]['mean'], reverse=True)[:10]
    for roi, stats in sorted_rois:
        log(f"  {stats['region_name']}: mean={stats['mean']:.3f}, peak_time={stats['peak_time']:.1f}s")

    return roi_stats
def main():
    start_time = datetime.now()
    log("İşlem başladı.")

    try:
        # 1. EEG VERİSİNİ YÜKLE VE TEMİZLE
        log("EEG dosyası yükleniyor ve filtreleniyor...")
        mat = scipy.io.loadmat(eeg_mat)
        eeg = mat['dataRest'][:64].astype(np.float32)
        eeg = bandpass(eeg, fs, 1, 45)
        eeg = notch(eeg, fs)

        # ICA ile artefakt temizleme
        log("ICA uygulanıyor ve artefaktlar temizleniyor...")
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
        log("Kanal koordinatları ve voxel grid oluşturuluyor...")
        chan_coords = validate_and_transform_coords(coord_file)
        
        # Grid oluştur
        voxel_coords_all, grid_shape = create_voxel_grid(
            [-90, 91], [-126, 91], [-72, 109], grid_spacing
        )

        # 3. GRAY MATTER MASKELEME
        if USE_HARVARD_OXFORD:
            log("\n=== HARVARD-OXFORD MASKELEME ===")
            voxel_coords, brain_indices, voxel_labels, region_ids, all_labels = apply_harvard_oxford_mask(voxel_coords_all)
            
            # Gray matter filtreleme
            if USE_GRAY_MATTER_MASK:
                keep_indices = filter_white_matter_from_labels(voxel_labels, region_ids)
                
                voxel_coords = voxel_coords[keep_indices]
                brain_indices = brain_indices[keep_indices]
                voxel_labels = [voxel_labels[i] for i in keep_indices]
                region_ids = [region_ids[i] for i in keep_indices]
                
                log(f"Gray matter filtreleme sonrası voxel sayısı: {len(voxel_coords)}")
                
                # Filtreleme sonrası istatistikleri göster
                unique_regions, counts = np.unique(voxel_labels, return_counts=True)
                log("\nGray matter voxel dağılımı:")
                for region, count in sorted(zip(unique_regions, counts), key=lambda x: x[1], reverse=True)[:10]:
                    log(f"  {region}: {count} voxel")
        
        elif USE_BRAIN_MASK:
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
        log("\nVoxel signatures hesaplanıyor...")
        voxel_sigs = compute_directional_voxel_signatures(
            voxel_coords, chan_coords, sigma=sigma, batch_size=batch_size
        )
        # 5. HER FREKANS BANDI İÇİN İŞLEM
        for band_name, band_range in bands.items():
            log(f"\n{'='*60}")
            log(f"{band_name.upper()} BANDI İÇİN İŞLEM BAŞLIYOR")
            log(f"{'='*60}")
            
            # Hilbert envelope hesapla
            snapshots = np.stack([hilbert_envelope(seg, band_range, fs) for seg in segments])
            
            # 6. SURFACE PROJECTION (OPSIYONEL)
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
                    surface_data, vertices, n_lh = windows_friendly_surface_projection(
                        snapshots.T, chan_coords
                    )
                    
                    surface_volume, surface_affine = surface_to_volume_simple(
                        surface_data, vertices, grid_shape, grid_spacing
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
            log("Bin sınırları hesaplanıyor...")
            v_bins = [np.histogram_bin_edges(voxel_sigs[i], bins=n_bins) 
                     for i in tqdm(range(len(voxel_sigs)), desc="Bin hesaplama")]
            
            # MI ve Dice skorları hesapla
            log(f"Segmentler paralel işleniyor ({n_jobs} çekirdek)...")
            mi_scores, dice_scores = process_snapshots_parallel(voxel_sigs, snapshots, v_bins, n_jobs)
            
            # Hybrid skorlar
            log("Hibrit skorlar hesaplanıyor...")
            x, y, z = grid_shape
            
            if USE_ADAPTIVE_WEIGHTS:
                # Temporal stability hesapla
                temp_nii_data = np.zeros((x, y, z, snapshots.shape[0]), dtype=np.float32)
                temp_hybrid = MI_WEIGHT * mi_scores + DICE_WEIGHT * dice_scores
                
                # Geçici volume oluştur
                for idx in range(len(voxel_coords)):
                    mni_coord = voxel_coords[idx]
                    xi = int((mni_coord[0] + 90) / grid_spacing)
                    yi = int((mni_coord[1] + 126) / grid_spacing)
                    zi = int((mni_coord[2] + 72) / grid_spacing)
                    if 0 <= xi < x and 0 <= yi < y and 0 <= zi < z:
                        temp_nii_data[xi, yi, zi, :] = temp_hybrid[idx, :]
                
                temporal_stability = compute_temporal_stability(temp_nii_data)
                
                # Voxel stability map et
                voxel_stability = []
                for idx in range(len(voxel_coords)):
                    mni_coord = voxel_coords[idx]
                    xi = int((mni_coord[0] + 90) / grid_spacing)
                    yi = int((mni_coord[1] + 126) / grid_spacing)
                    zi = int((mni_coord[2] + 72) / grid_spacing)
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
                
                # MNI'dan voxel indeksine dönüştür
                vox_idx = inv_affine @ np.append(mni_coord, 1)
                xi, yi, zi = np.round(vox_idx[:3]).astype(int)
                
                if 0 <= xi < x and 0 <= yi < y and 0 <= zi < z:
                    nii_data[xi, yi, zi, :] = hybrid_scores[idx, :]
                    placed_count += 1
            
            log(f"Yerleştirilen voxel sayısı: {placed_count}/{len(voxel_coords)}")
            
            # 9. POST-PROCESSING
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
                    
                    # Global ROI centers dosyası (tüm bantlar için tek bir dosya)
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
            log("\n=== İstatistikler ===")
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
            
            # Belleği temizle
            del nii_data, mi_scores, dice_scores, hybrid_scores
            gc.collect()
            
            log(f"\n{band_name} bandı tamamlandı.")
            log("="*60)
        # 14. TAMAMLANDI (for döngüsü dışında)
        log("\n" + "="*60)
        log("TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
        log(f"Çıktı dizini: {data_path}")
        log("="*60)
        
    except Exception as e:
        log(f"\n*** HATA OLUŞTU ***")
        log(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log(f"\nToplam işlem süresi: {int(duration // 60)} dakika {int(duration % 60)} saniye")


# Giriş noktası
if __name__ == "__main__":
    main()
