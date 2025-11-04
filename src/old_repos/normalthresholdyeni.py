# filename: apply_threshold_and_normalize.py
import os
import nibabel as nib
import numpy as np

# === AYARLAR === #
BASE_DIR = r"C:\Users\kerem\Downloads\eegyedek"
BANDS = ["alpha", "beta", "gamma", "theta", "delta"]
SUBJECTS = [f"ec{str(i).zfill(2)}" for i in range(2, 12)]
THRESHOLD_PERCENTILE = 70  # <- Burayı ihtiyaca göre değiştir
OUTPUT_PREFIX = f"zthr{THRESHOLD_PERCENTILE}"

def normalize_and_threshold(file_path, output_path, percentile):
    img = nib.load(file_path)
    data = img.get_fdata()
    mask = data != 0
    values = data[mask]

    # Global mean çıkar ve z-score uygula
    demeaned = values - np.mean(values)
    zscored = (demeaned - np.mean(demeaned)) / np.std(demeaned)

    # Eşikleme
    threshold = np.percentile(np.abs(zscored), percentile)
    zscored[np.abs(zscored) < threshold] = 0

    # Geri yaz
    new_data = np.zeros_like(data)
    new_data[mask] = zscored

    new_img = nib.Nifti1Image(new_data, affine=img.affine, header=img.header)
    nib.save(new_img, output_path)
    print(f"✔ Saved: {output_path}")

# === Ana döngü === #
for subj in SUBJECTS:
    subj_path = os.path.join(BASE_DIR, subj)
    if not os.path.exists(subj_path):
        continue

    for band in BANDS:
        in_name = f"{band}_voxel_spm.nii"
        in_path = os.path.join(subj_path, in_name)
        out_name = f"{OUTPUT_PREFIX}{band}_voxel_spm.nii"
        out_path = os.path.join(subj_path, out_name)

        if os.path.exists(in_path):
            try:
                normalize_and_threshold(in_path, out_path, THRESHOLD_PERCENTILE)
            except Exception as e:
                print(f"✖ Error in {in_path}: {e}")
        else:
            print(f"⚠ Not found: {in_path}")
