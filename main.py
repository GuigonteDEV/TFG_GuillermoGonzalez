from pathlib import Path
import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import re
import h5py
import time as time
import matplotlib.pyplot as plt
import random
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Funciones.Tensor_Images import build_tensors, load_csv
from Funciones.Build_WSI import reconstruct, load_pt
from Funciones.Augmentation_Dataloader import summarize_h5_files, print_split_summary, Transforms, H5DatasetMultilabel, compute_multilabel_metrics, print_metrics
from Funciones.KFCV_Create import folds_creation, get_dataset_split, folds_statistics
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, average_precision_score


# ---------------------------
# Configuración general universal
# ---------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

CLASS_NAMES = [
    'normal',
    'lowgrade_dysplasia',
    'inflammation',
    'highgrade_dysplasia',
    'tumor_necrosis',
    'suspicious_for_invasion',
    'adenocarcinoma',
]

ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto') 

Create_WSI = False

CSV_PATH = ROOT / 'Statistics' / 'WSI_stats.csv'
H5_FILES = ROOT / 'h5_multilabel' 
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_CLASSES = len(CLASS_NAMES)
LR         = 1e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 15

# ---------------------------
# Configuración de Folds
# ---------------------------
NUM_FOLDS = 5
FOLD_CONFIG = 1

# ---------------------------
# Configuración de Estrategias
# ---------------------------
DATA_STRATEGY = 'None'  # Opciones: 'None', 'CUS', 'ROS', 'WS'
ALGO_STRATEGY = 'None'  # Opciones: 'None', 'Focal', 'PW'


################################
# ---------------------------
# Reconstrucción WSI
# ---------------------------
################################

if Create_WSI:
    for n_slide in range(1,201):
        n_slide_str = str(n_slide).zfill(3)    
        INPUT_PATH = ROOT / 'processed' / f'{n_slide_str}_tensor.pt'
        FILL_COLOR = 255

        print("Cargando:", INPUT_PATH)
        data = load_pt(INPUT_PATH)
        WSI_Image, WSI_Map, grid_shape, placed = reconstruct(data)

        out_dir = ROOT / 'WSI_Images'
        out_dir.mkdir(exist_ok=True)

        WSI_Image.save(out_dir / f'{n_slide_str}_WSI.png')
        WSI_Map.save(out_dir / f'{n_slide_str}_WSI_Map.png')

        print(f"Reconstrucción guardada en: {out_dir / f'{n_slide_str}_WSI.png'}")
        print(f"Rejilla (cols, rows): {grid_shape}, patches colocados: {placed}")
        
        
################################
# ---------------------------
# Creación Dataloader
# ---------------------------
################################

#Inicio cronómetro
start_time = time.time()

# ---------------------------
# Creación índices Dataset
# ---------------------------

pt_files = list(H5_FILES.glob("*.h5"))
pt_files = sorted(pt_files, key=lambda f: int(f.stem.split('_')[0]))
pt_files = np.array(pt_files)

folds_files, wsi_data = folds_creation(pt_files, NUM_FOLDS)

folds_statistics(pt_files, folds_files, NUM_FOLDS)

train_idx, val_idx, test_idx = get_dataset_split(FOLD_CONFIG, folds_files, NUM_FOLDS)

train_files = pt_files[train_idx]
val_files = pt_files[val_idx]

# --- Resumen de distribución de clases ---
train_stats = summarize_h5_files(train_files)
val_stats   = summarize_h5_files(val_files)
print_split_summary("TRAIN",      train_stats)
print_split_summary("VALIDATION", val_stats)


# ---------------------------
# Inicialización Transforms Augmentation
# ---------------------------

train_transforms, val_transforms = Transforms(IMAGE_SIZE)


# ---------------------------
# Creación Dataset
# ---------------------------

train_dataset = H5DatasetMultilabel(train_files, transform = train_transforms)
val_dataset = H5DatasetMultilabel(val_files, transform = val_transforms)


# ---------------------------
# WeightedRandomSampler para train
# ---------------------------

#train_sampler = WeightedSampler(train_dataset.labels)


# ---------------------------
# DataLoaders
# ---------------------------

train_sampler = None
shuffle_train = True

if DATA_STRATEGY == 'WS':
    print("Aplicando: Weighted Sampler")
    shuffle_train = False 

elif DATA_STRATEGY == 'ROS':
    print("Aplicando: Random Over-Sampling")
    
elif DATA_STRATEGY == 'CUS':
    print("Aplicando: Cluster Under-Sampling")

elif DATA_STRATEGY == 'None':
    print("Aplicando: Baseline (Datos originales)")

else:
    raise ValueError(f"Estrategia de datos desconocida: {DATA_STRATEGY}")


#¡¡IMPORTANTE!! añadir num_workers si se usa GPU
#Las especificaciones de num_workers puede variar segun ordenador

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = shuffle_train, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)


################################
# ---------------------------
# Implementación Modelo + Entrenamiento
# ---------------------------
################################


# ---------------------------
# Modelo CNN
# ---------------------------
class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")

        # --- Freeze backbone completo ---
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_last_fc(self):
        for param in self.backbone.features[-1].parameters():
                param.requires_grad = True
    

# =============================================================================
# LOOPS DE ENTRENAMIENTO Y VALIDACIÓN
# =============================================================================


def train_loop(
    dataloader, model, loss_fn, optimizer, scaler, device
) -> tuple[float, float, float, dict, dict]:
    model.train()
    losses     = []
    all_probs  = []
    all_labels = []
 
    for X, y in tqdm(dataloader, desc="  train", leave=False):
        # X : (B, 3, H, W) float   y : (B, 7) float
        X, y = X.to(device), y.to(device)
 
        optimizer.zero_grad()
 
        with torch.amp.autocast("cuda"):
            logits = model(X)          # (B, 7)
            loss   = loss_fn(logits, y)
 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
 
        losses.append(loss.item())
 
        # sigmoid aquí solo para métricas, no afecta el entrenamiento
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())
 
    all_probs  = np.concatenate(all_probs,  axis=0)   # (N, 7)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, 7)
 
    macro_roc, macro_pr, roc_per_class, pr_per_class = compute_multilabel_metrics(
        all_probs, all_labels
    )
    return float(np.mean(losses)), macro_roc, macro_pr, roc_per_class, pr_per_class


@torch.no_grad()
def val_loop(
    dataloader, model, loss_fn, device
) -> tuple[float, float, float, dict, dict]:
    model.eval()
    losses     = []
    all_probs  = []
    all_labels = []
 
    for X, y in tqdm(dataloader, desc="  val  ", leave=False):
        X, y = X.to(device), y.to(device)
 
        logits = model(X)              # (B, 7)
        loss   = loss_fn(logits, y)
        losses.append(loss.item())
 
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())
 
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
 
    macro_roc, macro_pr, roc_per_class, pr_per_class = compute_multilabel_metrics(
        all_probs, all_labels
    )
    return float(np.mean(losses)), macro_roc, macro_pr, roc_per_class, pr_per_class
 

# =============================================================================
# INICIALIZACIÓN
# =============================================================================
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo: {device}")
 
scaler  = torch.amp.GradScaler("cuda")
model   = EfficientNet(num_classes=NUM_CLASSES).to(device)
#loss_fn = build_loss(train_files, device)

if ALGO_STRATEGY == 'PW':
    print("Aplicando Loss: BCE con Pos Weights (PW)")


elif ALGO_STRATEGY == 'Focal':
    print("Aplicando Loss: Focal Loss")


elif ALGO_STRATEGY == 'None':
    print("Aplicando Loss: Baseline (BCE pura)")
    loss_fn = nn.BCEWithLogitsLoss()

else:
    raise ValueError(f"Estrategia de algoritmo desconocida: {ALGO_STRATEGY}")

 
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)
 
 
# =============================================================================
# HISTÓRICO DE MÉTRICAS
# =============================================================================
 
history = {
    'train_loss': [], 'val_loss':    [],
    'train_roc':  [], 'val_roc':     [],
    'train_pr':   [], 'val_pr':      [],
}
 
best_val_pr = 0.0
 
# =============================================================================
# BUCLE DE ENTRENAMIENTO
# =============================================================================

CKPT_DIR = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')
CKPT_DIR.mkdir(exist_ok=True)
 
for epoch in range(1, EPOCHS + 1):
    if epoch == 11:
        model.unfreeze_last_fc()
        print("Unfreeze")
    print(f"\n{'='*60}")
    print(f"  Epoch {epoch}/{EPOCHS}")
    print(f"{'='*60}")
 
    train_loss, train_roc, train_pr, train_roc_cls, train_pr_cls = train_loop(
        train_loader, model, loss_fn, optimizer, scaler, device
    )
    val_loss, val_roc, val_pr, val_roc_cls, val_pr_cls = val_loop(
        val_loader, model, loss_fn, device
    )
 
    scheduler.step()
 
    # --- Guardar histórico ---
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_roc'].append(train_roc)
    history['val_roc'].append(val_roc)
    history['train_pr'].append(train_pr)
    history['val_pr'].append(val_pr)
 
    # --- Imprimir métricas ---
    print_metrics("TRAIN", train_loss, train_roc, train_pr, train_roc_cls, train_pr_cls)
    print_metrics("VAL  ", val_loss,   val_roc,   val_pr,   val_roc_cls,   val_pr_cls)
    print(f"\n  LR actual: {optimizer.param_groups[0]['lr']:.2e}")
 
    # --- Checkpoint si mejora PR-AUC macro en validación ---
    if val_pr > best_val_pr:
        best_val_pr = val_pr
        checkpoint = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss':             val_loss,
            'val_pr_macro':         val_pr,
            'val_roc_macro':        val_roc,
            'val_pr_per_class':     val_pr_cls,
            'val_roc_per_class':    val_roc_cls,
            'class_names':          CLASS_NAMES,
        }
        ckpt_path = CKPT_DIR / "best_model.pth"
        torch.save(checkpoint, ckpt_path)
        print(f"\n Checkpoint guardado (PR-macro={val_pr:.4f})  →  {ckpt_path}")
 
 
# =============================================================================
# RESUMEN FINAL
# =============================================================================
 
elapsed = time.time() - start_time
print(f"\n{'#'*60}")
print(f"  Entrenamiento completado en {elapsed/60:.1f} min")
print(f"  Mejor val PR-macro: {best_val_pr:.4f}")
print(f"  Checkpoint en:      {CKPT_DIR / 'best_model.pth'}")
print(f"{'#'*60}")


