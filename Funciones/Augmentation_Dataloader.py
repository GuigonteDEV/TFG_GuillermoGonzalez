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
import h5py
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, average_precision_score


# ---------------------------
# Configuración general
# ---------------------------

ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')
IMAGE_SIZE = 256  
RANDOM_SEED = 42
VAL_FRACTION = 0.15
CLASS_NAMES = [
    'normal',
    'lowgrade_dysplasia',
    'inflammation',
    'highgrade_dysplasia',
    'tumor_necrosis',
    'suspicious_for_invasion',
    'adenocarcinoma',
]

# =============================================================================
# DISTRIBUCIONES INICIALES
# =============================================================================

def summarize_h5_files(h5_files) -> dict:
    
    per_class     = {name: 0 for name in CLASS_NAMES}
    total_patches = 0
 
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as h5:
            labels = h5["labels"][:]
            total_patches += labels.shape[0]
            for i, name in enumerate(CLASS_NAMES):
                per_class[name] += int(labels[:,i].sum())
 
    return {"total_patches": total_patches, "per_class": per_class}
 
 
def print_split_summary(name: str, stats: dict) -> None:
    total = stats["total_patches"]
    print(f"\n========== {name} ==========")
    print(f"  Patches totales : {total:,}")
    for class_name, count in stats["per_class"].items():
        pct = 100 * count / total if total > 0 else 0.0
        print(f"  [{CLASS_NAMES.index(class_name)}] {class_name:<30} {count:>8,}  ({pct:.1f}%)")
        
        
# =============================================================================
# MÉTRICAS MULTILABEL
# =============================================================================

def compute_multilabel_metrics(
    all_probs:  np.ndarray,   # (N, NUM_CLASSES)  float
    all_labels: np.ndarray,   # (N, NUM_CLASSES)  int/float
) -> tuple[float, float, dict, dict]:
    roc_per_class = {}
    pr_per_class  = {}
    if np.isnan(all_probs).any():
        print("  [WARN] NaN detectado en probabilidades — métricas no disponibles esta época")
        nan_dict = {name: float('nan') for name in CLASS_NAMES}
        return float('nan'), float('nan'), nan_dict, nan_dict
 
    for i, name in enumerate(CLASS_NAMES):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]
 
        # Si no hay positivos en este split, la métrica no está definida
        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            roc_per_class[name] = float('nan')
            pr_per_class[name]  = float('nan')
            continue
 
        roc_per_class[name] = roc_auc_score(y_true, y_prob)
        pr_per_class[name]  = average_precision_score(y_true, y_prob)
 
    valid_rocs = [v for v in roc_per_class.values() if not np.isnan(v)]
    valid_prs  = [v for v in pr_per_class.values()  if not np.isnan(v)]
 
    macro_roc = float(np.mean(valid_rocs)) if valid_rocs else float('nan')
    macro_pr  = float(np.mean(valid_prs))  if valid_prs  else float('nan')
 
    return macro_roc, macro_pr, roc_per_class, pr_per_class
 
 
def print_metrics(split: str, loss: float, macro_roc: float, macro_pr: float,
                  roc_per_class: dict, pr_per_class: dict) -> None:
    print(f"\n  [{split}]  loss={loss:.4f}  ROC-macro={macro_roc:.4f}  PR-macro={macro_pr:.4f}")
    for name in CLASS_NAMES:
        roc = roc_per_class[name]
        pr  = pr_per_class[name]
        roc_str = f"{roc:.4f}" if not np.isnan(roc) else "  n/a "
        pr_str  = f"{pr:.4f}"  if not np.isnan(pr)  else "  n/a "
        print(f"    [{CLASS_NAMES.index(name)}] {name:<30}  ROC={roc_str}  PR={pr_str}")
        

# ---------------------------
# Definir augmentations on-the-fly
# ---------------------------

def rotate_90(img):
    angle = random.choice([0, 90, 180, 270])
    return transforms.functional.rotate(img, angle)

def Transforms(Image_SIZE):
    
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomApply([
            transforms.Lambda(rotate_90)
        ], p=0.5),
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
        ], p = 0.25),
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
# Dataset H5 
# ---------------------------    
class H5DatasetMultilabel(Dataset):
    def __init__(self, h5_files, transform=None):
        self.transform = transform
        self.index     = []   # lista de (h5_path, local_idx)
 
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as h5:
                n = h5["images"].shape[0]
            for i in range(n):
                self.index.append((str(h5_path), i))
 
    def __len__(self) -> int:
        return len(self.index)
 
    def __getitem__(self, idx):
        h5_path, local_idx = self.index[idx]
 
        with h5py.File(h5_path, "r") as h5:
            img    = h5["images"][local_idx]          
            labels = h5["labels"][local_idx]
 
        # Imagen: uint8 numpy → PIL → transforms → FloatTensor (3, H, W)
        if self.transform:
            img = to_pil_image(img)
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
 
        # Labels: uint8 numpy → FloatTensor (8,)
        labels = torch.from_numpy(labels.copy()).float()
 
        return img, labels


# ---------------------------
# Inferencia
# ---------------------------

@torch.no_grad()
def extract_predictions(dataloader, model, device):
    all_probs = []
    all_labels = []
    for X, y in dataloader:
        X = X.to(device)
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.vstack(all_labels), np.vstack(all_probs)



