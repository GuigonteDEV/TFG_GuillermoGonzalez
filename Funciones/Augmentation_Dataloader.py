'''Organización del dataset en train y validation, en primera instancia se hace en script a parte para
organizar mejor. Se probaran dos métodos, con probabilidad de leakage y sin probabilidad de leakage.

Paso importante previo al desarrollo de la red neuronal.'''


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
from pathlib import Path
import random
import h5py
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, average_precision_score, auc


# ---------------------------
# Configuración general
# ---------------------------

ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')
IMAGE_SIZE = 256  
CLASS_NAMES = [
    'normal',
    'lowgrade_dysplasia',
    'inflammation',
    'highgrade_dysplasia',
    'tumor_necrosis',
    'suspicious_for_invasion',
    'adenocarcinoma',
]

SEVERITY_CLASSES = [
    'normal',                  # 0 — Sano / Ausencia de patología
    'inflammation',            # 1 — menor riesgo patológico
    'lowgrade_dysplasia',      # 2
    'highgrade_dysplasia',     # 3
    'tumor_necrosis',          # 4
    'suspicious_for_invasion', # 5
    'adenocarcinoma',          # 6 — mayor riesgo
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
        

def compute_binary_metrics(
    all_probs: np.ndarray,  
    all_labels: np.ndarray, 
) -> tuple[float, float, float]:
    """
    Calcula las métricas para el enfoque binario.
    Devuelve: (ROC AUC, Average Precision, PR AUC)
    """
    
    if np.isnan(all_probs).any():
        print("  [WARN] NaN detectado en probabilidades — métricas no disponibles esta época")
        return float('nan'), float('nan'), float('nan')
 
    if all_labels.sum() == 0 or (1 - all_labels).sum() == 0:
        print("  [WARN] El split no contiene ambas clases (0 y 1) — métricas no definidas")
        return float('nan'), float('nan'), float('nan')
 
    roc_auc = float(roc_auc_score(all_labels, all_probs))
    
    precisions, recalls, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = float(auc(recalls, precisions))
 
    return roc_auc, pr_auc

def compute_multiclass_metrics(
    all_probs:  np.ndarray,   # Matriz Softmax:  (N, NUM_CLASSES) float
    all_labels: np.ndarray, # Vector plano:     (N,) con enteros (0, 1, 2...)
) -> tuple[float, dict]:
    """
    Calcula de forma eficiente el F1-Macro global y el F1-Score desglosado
    por cada una de las patologías.
    """
    if np.isnan(all_probs).any():
        print("  [WARN] NaN detectado en probabilidades — métricas no disponibles esta época")
        nan_dict = {name: float('nan') for name in CLASS_NAMES}
        return float('nan'), nan_dict
 
    all_preds = np.argmax(all_probs, axis=1)

    class_indices = list(range(len(SEVERITY_CLASSES)))

    f1_values = f1_score(
        all_labels, 
        all_preds, 
        average=None, 
        labels=class_indices, 
        zero_division=0
    )
    
    f1_per_class = {name: float(f1_values[i]) for i, name in enumerate(SEVERITY_CLASSES)}
 
    macro_f1 = float(f1_score(
        all_labels, 
        all_preds, 
        average='macro', 
        labels=class_indices, 
        zero_division=0
    ))
 
    return macro_f1, f1_per_class

def print_metrics_multiclass(split: str, loss: float, macro_f1: float, f1_per_class: dict) -> None:
    # Imprime el resumen global del split (Train o Val)
    print(f"\n  [{split}]  loss={loss:.4f}  F1-macro={macro_f1:.4f}")
    
    # Desglose por cada una de las patologías
    for name in SEVERITY_CLASSES:
        f1 = f1_per_class[name]
        f1_str = f"{f1:.4f}" if not np.isnan(f1) else "  n/a "
        print(f"    [{SEVERITY_CLASSES.index(name)}] {name:<30}  F1={f1_str}")

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

class H5DatasetBinary(Dataset):
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
        binary_value = 1.0 if labels > 0 else 0.0
        
        # Lo convertimos a un FloatTensor escalar
        label_tensor = torch.tensor(binary_value, dtype=torch.float32)
 
        return img, label_tensor
    
class H5DatasetMulticlass(Dataset):
    def __init__(self, h5_files, transform=None):
        self.transform = transform
        self.index     = []   # lista de (h5_path, local_idx)
 
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as h5:
                labels = h5["labels"][:]
                for i, label in enumerate(labels):
                    if label > 0:
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
        
        labels = labels - 1
        
        labels = torch.tensor(labels, dtype=torch.long)
 
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



