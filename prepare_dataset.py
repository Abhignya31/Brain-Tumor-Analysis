import os
import numpy as np
import cv2
import nibabel as nib

# ===============================
# PATHS
# ===============================
input_folder = "brats_raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
output_image_dir = "segmentation_dataset/images"
output_mask_dir = "segmentation_dataset/masks"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

count = 0

# ===============================
# PROCESS EACH PATIENT
# ===============================
for patient in os.listdir(input_folder):

    patient_path = os.path.join(input_folder, patient)

    flair_path = os.path.join(patient_path, f"{patient}_flair.nii")
    seg_path = os.path.join(patient_path, f"{patient}_seg.nii")

    if not os.path.exists(flair_path) or not os.path.exists(seg_path):
        continue

    print("Processing:", patient)

    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    # ===============================
    # SLICE LOOP
    # ===============================
    for i in range(flair.shape[2]):

        image = flair[:, :, i]
        mask = seg[:, :, i]

        # skip empty slices
        if np.sum(mask) < 50:
            continue

        # normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        image = (image * 255).astype(np.uint8)

        # convert mask to binary
        mask = (mask > 0).astype(np.uint8) * 255

        # resize
        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

        # save
        cv2.imwrite(os.path.join(output_image_dir, f"{count}.png"), image)
        cv2.imwrite(os.path.join(output_mask_dir, f"{count}.png"), mask)

        count += 1

print("✅ Segmentation dataset created:", count)