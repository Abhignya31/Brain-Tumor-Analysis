import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from vnet_model import VNet

# ==========================
# DEVICE
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================
# DATASET CLASS
# ==========================
class BrainTumorDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))[:1000]
        print("Total images:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128))

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)


# ==========================
# LOAD DATA
# ==========================
image_dir = "segmentation_dataset/images"
mask_dir = "segmentation_dataset/masks"

dataset = BrainTumorDataset(image_dir, mask_dir)

if len(dataset) == 0:
    raise ValueError("No data found. Run prepare_dataset.py first.")

loader = DataLoader(dataset, batch_size=2, shuffle=True)


# ==========================
# MODEL
# ==========================
model = VNet().to(device)

bce_loss = nn.BCELoss()


def dice_loss(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))


optimizer = optim.Adam(model.parameters(), lr=1e-4)


# ==========================
# TRAINING
# ==========================
epochs = 5
best_loss = float("inf")

print("Starting training...")

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        bce = bce_loss(outputs, masks)
        dice = dice_loss(outputs, masks)

        loss = bce + dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_vnet_trained.pth")
        print("Best model saved")

# FINAL SAVE
torch.save(model.state_dict(), "vnet_trained.pth")

print("Training complete!")