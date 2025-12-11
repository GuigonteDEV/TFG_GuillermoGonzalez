'''Organización del dataset en train y validation, en primera instancia se hace en script a parte para
organizar mejor. Se probaran dos métodos, con probabilidad de leakage y sin probabilidad de leakage.

Paso importante previo al desarrollo de la red neuronal.'''


import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------
# Configuración general
# ---------------------------
ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')
PT_DIR = ROOT / 'processed' 
BATCH_SIZE = 32
IMAGE_SIZE = 256  
RANDOM_SEED = 42


# ---------------------------
# Dataset PyTorch
# ---------------------------
class PatchesDataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, transform=None):
        self.images = images_tensor  # [N, C, H, W]
        self.labels = labels_tensor  # [N]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # torchvision transforma PIL, así que convertimos temporalmente
            image = to_pil_image(image)
            image = self.transform(image)
        return image, label


# ---------------------------
# Definir augmentations on-the-fly
# ---------------------------

train_transforms = transforms.Compose([
    transforms.RandomRotation([0, 90, 180, 270]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3)
    ),
    transforms.ToTensor(),
    transforms.GaussianNoise(sigma=0.1),  
    transforms.GaussianBlur(kernel_size=3, sigma=0.1),  
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# ---------------------------
# Dividir train / val (estratificado)
# ---------------------------
Posible_Leakage = True

if Posible_Leakage:
    
    pt_files = list(PT_DIR.glob("*.pt"))
    all_images = []
    all_labels = []

    for f in pt_files:
        data = torch.load(f)
        all_images.append(data['images'])  # [N, C, H, W]
        all_labels.append(data['labels'])  # [N]

    # Concatenar todos los patches
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    train_idx, val_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=0.15,
        stratify=all_labels.numpy(),
        random_state=RANDOM_SEED
    )

    train_dataset = PatchesDataset(all_images[train_idx], all_labels[train_idx], transform=train_transforms)
    val_dataset = PatchesDataset(all_images[val_idx], all_labels[val_idx], transform=val_transforms)

# ---------------------------
# WeightedRandomSampler para train
# ---------------------------
train_labels = all_labels[train_idx].numpy()
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]

train_sampler = WeightedRandomSampler(weights=sample_weights,
num_samples=len(sample_weights),
replacement=True)

# ---------------------------
# DataLoaders
# ---------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ---------------------------
# Ejemplo de iteración
# ---------------------------
if __name__ == "__main__":
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx} - images: {images.shape}, labels: {labels.shape}")
        if batch_idx == 1:
            break
