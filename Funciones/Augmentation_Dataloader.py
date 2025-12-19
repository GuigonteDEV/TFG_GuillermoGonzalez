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
import random
from tqdm import tqdm


# ---------------------------
# Configuración general
# ---------------------------

ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')
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

def Dataset_Division(CSV_PATH):

    strat_labels, tumor_patches, no_tumor_patches, tot_patches, paths_WSI = Read_CSV(CSV_PATH)

    #División stratificada de val y train

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sss.split(paths_WSI, strat_labels))
    
    return train_idx, val_idx

# calcular patch-level distribuciones iniciales

def summarize_file_list(idx, CSV_PATH):
    strat_labels, tumor_patches, no_tumor_patches, tot_patches, paths_WSI = Read_CSV(CSV_PATH)
    tot_patches = np.array(tot_patches.to_numpy())
    tumor_patches = np.array(tumor_patches.to_numpy())
    no_tumor_patches = np.array(no_tumor_patches.to_numpy())
    patches = int(tot_patches[idx].sum())
    tumors = int(tumor_patches[idx].sum())
    notumors = int(no_tumor_patches[idx].sum())
    
    return patches, tumors, notumors


# ---------------------------
# Definir augmentations on-the-fly
# ---------------------------

def rotate_90(img):
    angle = random.choice([0, 90, 180, 270])
    return transforms.functional.rotate(img, angle)

def Transforms(Image_SIZE):
    
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(rotate_90),
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
    
    return train_transforms, val_transforms

# ---------------------------
# Dataset PyTorch
# ---------------------------
'''
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
'''

# ---------------------------
# Dataset PyTorch Lazy
# ---------------------------
class LazyPatchDataset(Dataset):
    def __init__(self, pt_files, transform=None):
        self.pt_files = list(pt_files)
        self.transform = transform

        self.index = []   # [(pt_path, patch_idx), ...]
        self.labels = []

        for pt_path in self.pt_files:
            data = torch.load(pt_path, map_location="cpu")
            n_patches = data["labels"].shape[0]

            for i in range(n_patches):
                self.index.append((pt_path, i))
                self.labels.append(int(data["labels"][i]))

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pt_path, patch_idx = self.index[idx]

        data = torch.load(pt_path, map_location="cpu")
        image = data["images"][patch_idx]
        label = data["labels"][patch_idx]

        if self.transform:
            image = to_pil_image(image)
            image = self.transform(image)
        
        return image, label



# ---------------------------
# Generación Dataset train / val
# ---------------------------

#No lazy
'''
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

def WeightedSampler(train_dataset):

    class_counts = np.bincount(train_dataset)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_dataset]

    train_sampler = WeightedRandomSampler(weights = sample_weights, num_samples = len(sample_weights), replacement = True)
    
    return train_sampler



