import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
from pathlib import Path
import random
import h5py
import timm
from huggingface_hub import login
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, average_precision_score, auc
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import re
import os
import pandas as pd
from tqdm import tqdm


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
        
        
# =========================
# CREACIÓN FOLDS
# =========================

def folds_creation(H5_FILES, NUM_FOLDS, seed):
    
    wsi_data = []

    for h5_path in H5_FILES:
        with h5py.File(h5_path, "r") as f:
            labels = f["labels"][:]
            
            # Matriz binaria de presencia
            
            presence_amp = np.max(labels, axis=0) 
            
            # Patches por clase
            per_class_counts = labels.sum(axis=0).astype(int)
            
            patch_count = len(labels)
            
            
            # Guardamos los datos de presencia y desglose de patches
            wsi_entry = {
                "archivo": os.path.basename(h5_path),
                "total patches": patch_count,
                "clases_presencia": presence_amp
            }
            
            # Añadimos de forma dinámica una columna por cada clase con su número de patches
            for idx_class, name_class in enumerate(CLASS_NAMES):
                wsi_entry[name_class] = per_class_counts[idx_class]
                
            wsi_data.append(wsi_entry)

    # Convertir a matrices limpias para el algoritmo de estratificación
    X = np.array([d["archivo"] for d in wsi_data])
    Y = np.array([d["clases_presencia"] for d in wsi_data])  

    # Aplicar la estratificación multilabel WSI wise
    mskf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    map_folds = {}

    for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(X, Y)):
        for idx in val_idx:
            map_folds[X[idx]] = fold_idx
            

    folds_files = [[] for _ in range(NUM_FOLDS)]
    
    # Extraemos el índice en lugar del nombre del archivo
    for file, fold_idx in map_folds.items():
        
        match = re.search(r'\d+', file)
        
        if match:
            numero_wsi = int(match.group())
            idx_original = numero_wsi - 1
            folds_files[fold_idx].append(idx_original)
        else:
            raise ValueError(f"No se pudo encontrar un número identificador en el archivo: {file}")

    # Convertimos a arrays de NumPy para que la función de división funcione idéntica
    folds_files = [np.array(f) for f in folds_files]
            
    return folds_files, wsi_data

def folds_statistics(H5_FILES, folds_files, NUM_FOLDS):
    # Estadisticas de patches por clase por fold
    statistics_folds = np.zeros((5, 7), dtype=int)

    # Numero patches por fold
    patches_per_fold = np.zeros(5, dtype=int)

    for fold in range(NUM_FOLDS):
        for idx in folds_files[fold]:
            with h5py.File(H5_FILES[idx], "r") as f:
                labels = f["labels"][:]
                
                classes_per_wsi = np.sum(labels, axis=0)
        
                # Acumular en Fold correspondiente
                statistics_folds[fold] += classes_per_wsi.astype(int)
                patches_per_fold[fold] += len(labels)
                
    df_resultados = pd.DataFrame(statistics_folds, columns=CLASS_NAMES)
    df_resultados.insert(0, "Total Patches", patches_per_fold)
    df_resultados.index.name = "Fold ID"


    print(df_resultados.to_string())


def get_dataset_split(FOLD_CONFIG, folds_list, NUM_FOLDS):
    
    if FOLD_CONFIG < 1 or FOLD_CONFIG > 5:
        raise ValueError("El parámetro del fold debe estar entre 1 y 5.")
        
    # Convertimos el parámetro (1-5) a índice de Python (0-4)
    val_fold_idx = FOLD_CONFIG - 1
    
    # Asignamos Test al siguiente fold de forma circular para evaluar siempre en datos "ciegos"
    test_fold_idx = (val_fold_idx + 1) % NUM_FOLDS
    
    # Los 3 folds restantes van para entrenamiento
    train_folds_indices = [i for i in range(NUM_FOLDS) if i != val_fold_idx and i != test_fold_idx]
    
    # Construcción de los sets de datos
    val_files = folds_list[val_fold_idx]
    test_files = folds_list[test_fold_idx]
    train_files = np.concatenate([folds_list[i] for i in train_folds_indices])
    
    return train_files, val_files, test_files
        
        
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
    all_probs:  np.ndarray,  
    all_labels: np.ndarray, 
) -> tuple[float, dict]:
    """
    Calcula el F1-Macro global y el F1-Score desglosado
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

class H5Dataset(Dataset):
    def __init__(self, h5_path, transform):
        self.h5_path   = h5_path
        self.transform = transform
        # Abrimos solo para leer el tamaño — se cierra enseguida
        with h5py.File(h5_path, 'r') as f:
            self.n_samples = f['images'].shape[0]
                
 
    def __len__(self) -> int:
        return self.n_samples
 
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as h5:
            img    = h5["images"][idx]
            coords_x = h5["topleft_x"][idx]
            coords_y = h5["topleft_y"][idx]
            
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = self.transform(img)
        
        coords_x = torch.tensor(coords_x, dtype=torch.float32)
        coords_y = torch.tensor(coords_y, dtype=torch.float32)
 
        return img, coords_x, coords_y
    
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
        self.index     = []
 
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
    
class H5DatasetInferencia(Dataset):
    def __init__(self, h5_files, transform=None):
        self.transform = transform
        self.index     = []
 
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
        
        labels = torch.tensor(labels, dtype=torch.long)
 
        return img, labels
    
class NPZDatasetBinaryFeatures(Dataset):
    def __init__(self, npz_files):
        """
        npz_files: lista de strings con las rutas a los archivos .npz
        """
        all_features = []
        all_labels = []
        self.index = []
        
        # Cargamos todos los archivos .npz en memoria
        for npz_path in npz_files:
            data = np.load(npz_path)
            all_features.append(data['features'])
            all_labels.append(data['labels'])
            
            file_stem = Path(npz_path).stem
            indices = data['features'].shape[0]
            for local_idx in range(indices):
                self.index.append((file_stem, local_idx))
            
        # Concatenamos las listas en un solo mega-array de NumPy
        features_np = np.concatenate(all_features, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        
        # Convertimos a Tensores de PyTorch de una sola vez
        self.features = torch.from_numpy(features_np).float()
        self.labels = torch.from_numpy(labels_np)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        # La lectura es instantánea porque ya está en la RAM
        feature_tensor = self.features[idx]
        label_val = self.labels[idx].item()
        
        # Lógica binaria: 1.0 si es patología (>0), 0.0 si es normal (0)
        binary_value = 1.0 if label_val > 0 else 0.0
        
        # Convertimos a FloatTensor escalar
        label_tensor = torch.tensor(binary_value, dtype=torch.float32)

        return feature_tensor, label_tensor
    
class NPZDatasetMulticlassFeatures(Dataset):
    def __init__(self, npz_files):
        """
        npz_files: lista de rutas a los archivos .npz
        """
        all_features = []
        all_labels = []
        self.index = [] # Mantenemos el índice para inferencia/trazabilidad
        
        for npz_path in npz_files:
            data = np.load(npz_path)
            features = data['features']
            labels = data['labels']
            
            # Filtramos solo los parches que tienen etiqueta > 0
            mask = labels > 0
            
            # Guardamos las features y labels que pasan el filtro
            all_features.append(features[mask])
            all_labels.append(labels[mask])
            
            # Guardamos los índices originales de los parches que SÍ se usan
            file_stem = Path(npz_path).stem
            valid_indices = np.where(mask)[0]
            for local_idx in valid_indices:
                self.index.append((file_stem, local_idx))
            
        # Concatenamos todo en tensores en RAM
        self.features = torch.from_numpy(np.concatenate(all_features, axis=0)).float()
        self.labels = torch.from_numpy(np.concatenate(all_labels, axis=0))
        
        # Ajuste de etiquetas
        self.labels = self.labels - 1

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        feature_tensor = self.features[idx]
        label_tensor = self.labels[idx].long() 
        
        return feature_tensor, label_tensor

class NPZDatasetInferenceFeatures(Dataset):
    def __init__(self, npz_files):
        """
        npz_files: lista de rutas a los archivos .npz
        """
        all_features = []
        all_labels = []
        self.index = [] # Mantenemos el índice para inferencia/trazabilidad
        
        for npz_path in npz_files:
            data = np.load(npz_path)
            features = data['features']
            labels = data['labels']
            
            # Guardamos las features y labels que pasan el filtro
            all_features.append(features)
            all_labels.append(labels)
            
            file_stem = Path(npz_path).stem
            indices = data['features'].shape[0]
            for local_idx in range(indices):
                self.index.append((file_stem, local_idx))
            
        # Concatenamos todo en tensores en RAM
        self.features = torch.from_numpy(np.concatenate(all_features, axis=0)).float()
        self.labels = torch.from_numpy(np.concatenate(all_labels, axis=0))
        

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        feature_tensor = self.features[idx]
        label_tensor = self.labels[idx].long() 
        
        return feature_tensor, label_tensor

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

@torch.no_grad()
def extract_predictions_binary(dataloader, model, device):
    all_probs = []
    all_labels = []
    for X, y in dataloader:
        X = X.to(device)
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze(1)
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)

@torch.no_grad()
def extract_predictions_multiclass(dataloader, model, device):
    all_probs = []
    all_labels = []
    for X, y in dataloader:
        X = X.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


# ---------------------------
# UNI
# ---------------------------

def load_uni_model(ckpt_dir, device, TOKEN):
    """
    Carga UNI desde checkpoint local o lo descarga de HuggingFace.
    Devuelve el modelo en modo eval con gradientes desactivados.
    """
    ckpt_path = Path(ckpt_dir) / 'pytorch_model.bin'

    if not ckpt_path.exists():
        print("Checkpoint no encontrado localmente. Descargando de HuggingFace...")
        if TOKEN is not None:
            login(token=TOKEN)
        else:
            login()  # pide token interactivamente si no está cacheado
        from huggingface_hub import hf_hub_download
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            "MahmoodLab/UNI",
            filename="pytorch_model.bin",
            local_dir=str(ckpt_dir),
            force_download=False
        )
    else:
        print(f"Checkpoint encontrado en {ckpt_path}")

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,        # sin cabeza de clasificación → devuelve embedding
        dynamic_img_size=True
    )
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)

    # Congelamos todos los parámetros — no se calculan gradientes
    for param in model.parameters():
        param.requires_grad = False

    print(f"UNI cargado en {device} | Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    return model


def extract_features(model, dataloader, device):
    """
    Pasa todos los batches por UNI y acumula features y labels.
    Devuelve arrays numpy: features (N, 1024), labels (N,1)
    """
    all_features = []
    all_coords_x = []
    all_coords_y = []

    with torch.no_grad():
        for imgs, coords_x, coords_y in tqdm(dataloader, desc="  Extrayendo", leave=False, disable=False):
            imgs = imgs.to(device)

            # Forward pass — num_classes=0 devuelve el CLS token: (B, 1024)
            feats = model(imgs)

            all_features.append(feats.cpu().numpy())
            all_coords_x.append(coords_x.numpy())
            all_coords_y.append(coords_y.numpy())

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    features = torch.from_numpy(features)
    topleft_x = np.concatenate(all_coords_x, axis=0).astype(np.float32)
    topleft_y = np.concatenate(all_coords_y, axis=0).astype(np.float32)

    return features, topleft_x, topleft_y