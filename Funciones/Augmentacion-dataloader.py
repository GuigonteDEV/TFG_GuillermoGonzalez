'''Organización del dataset en train y validation, en primera instancia se hace en script a parte para
organizar mejor. Se probaran dos métodos, con probabilidad de leakage y sin probabilidad de leakage.

Paso importante previo al desarrollo de la red neuronal.'''


import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------
# Configuración general
# ---------------------------
ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')
CSV_PATH = ROOT / 'Statistics' / 'WSI_stats.csv'
PT_DIR = ROOT / 'processed' 
BATCH_SIZE = 32
IMAGE_SIZE = 256  
RANDOM_SEED = 42
VAL_FRACTION = 0.15

# ---------------------------
# Division Dataset No Leakage
# ---------------------------

#Leer el CSV con info de WSI

def Read_CSV(CSV_PATH):

    df = pd.read_csv(CSV_PATH)

    has_tumor = df['has_tumor']
    tumor_patches = df['tumor_patches']
    no_tumor_patches = df['no_tumor_patches']
    tot_patches = df['total_patches']
    paths_WSI = df['wsi_path']


    strat_labels = np.array(has_tumor.to_numpy())
    
    return strat_labels, tumor_patches, no_tumor_patches, tot_patches, paths_WSI

strat_labels, tumor_patches, no_tumor_patches, tot_patches, paths_WSI = Read_CSV(CSV_PATH)

#División stratificada de val y train

sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
train_idx, val_idx = next(sss.split(paths_WSI, strat_labels))

# calcular patch-level distribuciones iniciales

def summarize_file_list(idx, tot_patches = tot_patches, tumor_patches = tumor_patches, no_tumor_patches = no_tumor_patches):
    tot_patches = np.array(tot_patches.to_numpy())
    tumor_patches = np.array(tumor_patches.to_numpy())
    no_tumor_patches = np.array(no_tumor_patches.to_numpy())
    patches = int(tot_patches[idx].sum())
    tumors = int(tumor_patches[idx].sum())
    notumors = int(no_tumor_patches[idx].sum())
    
    return patches, tumors, notumors

train_imgs, train_tumors, train_notumors = summarize_file_list(train_idx)
val_imgs, val_tumors, val_notumors = summarize_file_list(val_idx)

print("Split inicial:")
print(f" Train WSI: {len(train_idx)}, patches: {train_imgs}, tumors: {train_tumors}")
print(f" Val   WSI: {len(val_idx)}, patches: {val_imgs}, tumors: {val_tumors}")


# ---------------------------
# Definir augmentations on-the-fly
# ---------------------------

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomVerticalFlip(p = 0.5),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size = 3, sigma = (0.1, 0.5)), 
    ], p = 0.2),
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness = (0.9, 1.1),
            contrast = (0.9, 1.1)
            ), 
    ], p = 0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
])

# ---------------------------
# Dataset PyTorch
# ---------------------------

class PatchesDataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, transform=None):
        self.images = images_tensor
        self.labels = labels_tensor
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].clone()
        label = self.labels[idx]

        if self.transform:
            image = to_pil_image(image)
            image = self.transform(image)

        return image, label


# ---------------------------
# Generación Dataset train / val
# ---------------------------

pt_files = list(PT_DIR.glob("*.pt"))

pt_files = np.array(pt_files)

print(pt_files)

train_files = pt_files[train_idx]
val_files = pt_files[val_idx]

def generate_dataset(files):
    all_images = []
    all_labels = []
    for f in files:
        data = torch.load(f)
        all_images.append(data['images'])  # [N, C, H, W]
        all_labels.append(data['labels'])  # [N]
    
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_images, all_labels

train_images, train_labels = generate_dataset(train_files)
val_images, val_labels = generate_dataset(val_files)
        
train_dataset = PatchesDataset(train_images, train_labels, transform=train_transforms)
val_dataset = PatchesDataset(val_images, val_labels, transform=val_transforms)


'''

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
'''