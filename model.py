import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import time
import os

# ==============================
# SETTINGS
# ==============================
data_dir = "dataset"
num_classes = 4
batch_size = 8
num_epochs = 25
learning_rate = 0.00005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# DATA TRANSFORMS (UPDATED)
# ==============================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),

    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
}

# ==============================
# DATASET LOADING
# ==============================
image_datasets = {
    x: datasets.ImageFolder(
        root=os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in ['train', 'val']
}

dataloaders = {
    x: DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

print("Dataset sizes:", dataset_sizes)
print("Classes:", image_datasets['train'].classes)

# ==============================
# MODEL
# ==============================
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# ==============================
# LOSS FUNCTION (CLASS WEIGHTS)
# ==============================
# Order MUST match folder names:
# ['Glioma','Meningioma','No Tumor','Pituitary']

class_weights = torch.tensor([2.5, 1.0, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==============================
# TRAINING LOOP
# ==============================
since = time.time()
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 20)

    for phase in ['train', 'val']:

        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        # Save best model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "best_efficientnet_brain_tumor_classification.pth")
            print("✅ Best model saved!")

# ==============================
# FINAL SAVE
# ==============================
torch.save(model.state_dict(), "efficientnet_brain_tumor_classification.pth")

time_elapsed = time.time() - since
print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
print(f"Best Validation Accuracy: {best_acc:.4f}")