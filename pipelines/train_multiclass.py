from pathlib import Path
import os
import numpy as np
from torch.utils.data import DataLoader
import json
import pandas as pd
import argparse
import time as time
import random
from torchvision import models
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils import Transforms, H5DatasetMulticlass, compute_multiclass_metrics, print_metrics_multiclass, extract_predictions, folds_creation, get_dataset_split, folds_statistics
from src.models import EfficientNet
from src.balancing_methods import ASLSingleLabel



# ---------------------------
# Configuración general universal
# ---------------------------

parser = argparse.ArgumentParser(description="Entrenamiento Biopsias HTCondor")
parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria para reproducibilidad')
parser.add_argument('--fold', type=int, required=True, help="Numero de fold (1-5)")
args = parser.parse_args()

ROOT = Path('.') 

SEVERITY_CLASSES = [
    'inflammation',            # 1 — menor riesgo patológico
    'lowgrade_dysplasia',      # 2
    'highgrade_dysplasia',     # 3
    'tumor_necrosis',          # 4
    'suspicious_for_invasion', # 5
    'adenocarcinoma',          # 6 — mayor riesgo
]

H5_FILES = ROOT / 'Dataset'
H5_FILES_MULTI = ROOT / 'Dataset_multiclass'  
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_CLASSES = len(SEVERITY_CLASSES)
LR         = 1e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 100
SEED = args.seed

# Fijamos la semilla

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

set_seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------------------------
# Configuración de Folds
# ---------------------------
NUM_FOLDS = 5
FOLD_CONFIG = args.fold
        
        
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

binary_files = list(H5_FILES_MULTI.glob("*.h5"))
binary_files = sorted(binary_files, key=lambda f: int(f.stem.split('_')[0]))
binary_files = np.array(binary_files)

folds_files, wsi_data = folds_creation(pt_files, NUM_FOLDS, SEED)

folds_statistics(pt_files, folds_files, NUM_FOLDS)

train_idx, val_idx, test_idx = get_dataset_split(FOLD_CONFIG, folds_files, NUM_FOLDS)

train_files = binary_files[train_idx]
val_files = binary_files[val_idx]


# ---------------------------
# Inicialización Transforms Augmentation
# ---------------------------

train_transforms, val_transforms = Transforms(IMAGE_SIZE)

# ---------------------------
# Creación Dataset
# ---------------------------

train_dataset = H5DatasetMulticlass(train_files, transform = train_transforms)
val_dataset = H5DatasetMulticlass(val_files, transform = val_transforms)


# ---------------------------
# DataLoaders
# ---------------------------

train_sampler = None
shuffle_train = True

#¡¡IMPORTANTE!! añadir num_workers si se usa GPU
#Las especificaciones de num_workers puede variar segun ordenador

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = shuffle_train, sampler = train_sampler, num_workers = 4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4, worker_init_fn=seed_worker, generator=g)


################################
# ---------------------------
# Implementación Modelo + Entrenamiento
# ---------------------------
################################

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
 
    for X, y in tqdm(dataloader, desc="  train", leave=False, disable=True):
        X = X.to(device)
        y = y.long().to(device)
 
        optimizer.zero_grad()
 
        with torch.amp.autocast("cuda"):
            logits = model(X)         
            loss   = loss_fn(logits.float(), y)
 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
 
        losses.append(loss.item())
 
        # sigmoid aquí solo para métricas, no afecta el entrenamiento
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())
 
    all_probs  = np.concatenate(all_probs,  axis=0)  
    all_labels = np.concatenate(all_labels, axis=0)   
 
    macro_f1, f1_per_class = compute_multiclass_metrics(
        all_probs, all_labels
    )
    return float(np.mean(losses)), macro_f1, f1_per_class


@torch.no_grad()
def val_loop(
    dataloader, model, loss_fn, device
) -> tuple[float, float, float, dict, dict]:
    model.eval()
    losses     = []
    all_probs  = []
    all_labels = []
 
    for X, y in tqdm(dataloader, desc="  val  ", leave=False, disable=True):
        X = X.to(device)
        y = y.long().to(device)
 
        logits = model(X)             
        loss   = loss_fn(logits, y)
        losses.append(loss.item())
 
        probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())
 
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
 
    macro_f1, f1_per_class = compute_multiclass_metrics(
        all_probs, all_labels
    )
    return float(np.mean(losses)), macro_f1, f1_per_class
 

# =============================================================================
# INICIALIZACIÓN
# =============================================================================
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo: {device}")
 
scaler  = torch.amp.GradScaler("cuda")
model   = EfficientNet(num_classes=NUM_CLASSES).to(device)

loss_fn = ASLSingleLabel()

 
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)


# =============================================================================
# HISTÓRICO DE MÉTRICAS
# =============================================================================

history = {
    'time': [],
    'fold': FOLD_CONFIG,
    'seed': SEED,
    'config': {
        'EPOCHS': EPOCHS,
        'lr': LR,
    },
    'train': {
        'lr': [],
        'loss': [],
        'f1_macro': [],
        'f1_per_class': [] 
    },
    'val': {
        'loss': [],
        'f1_macro': [],
        'f1_per_class': []   
    }
}


best_val_f1 = 0.0

# =============================================================================
# BUCLE DE ENTRENAMIENTO
# =============================================================================

CKPT_DIR = ROOT / 'Model_output_multiclass'
CKPT_DIR.mkdir(exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    if epoch == 61:
        model.unfreeze_last_fc()
        print("Unfreeze")
    print(f"\n{'='*60}")
    print(f"  Epoch {epoch}/{EPOCHS}")
    print(f"{'='*60}")
 
    train_loss, train_f1, train_f1_cls = train_loop(
        train_loader, model, loss_fn, optimizer, scaler, device
    )
    val_loss, val_f1, val_f1_cls = val_loop(
        val_loader, model, loss_fn, device
    )
 
    scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
 
    # Guardar histórico
    history['train']['lr'].append(float(current_lr))
    history['train']['loss'].append(float(train_loss))
    history['train']['f1_macro'].append(float(train_f1))
    history['train']['f1_per_class'].append({k: float(v) for k, v in train_f1_cls.items()})

    history['val']['loss'].append(float(val_loss))
    history['val']['f1_macro'].append(float(val_f1))
    history['val']['f1_per_class'].append({k: float(v) for k, v in val_f1_cls.items()})
    
    # Guardar JSON
    with open(CKPT_DIR / f'multiclass_seed_{SEED}_fold_{FOLD_CONFIG}_history.json', 'w') as f:
        json.dump(history, f, indent=4)
 
    # Imprimir métricas
    print_metrics_multiclass("TRAIN", train_loss, train_f1, train_f1_cls)
    print_metrics_multiclass("VAL  ", val_loss, val_f1, val_f1_cls)
    print(f"\n  LR actual: {optimizer.param_groups[0]['lr']:.2e}")
 
    # Checkpoint si mejora F1 macro en validación 
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1_macro': val_f1,
                'val_loss': val_loss,
                'config': {
                    'lr': LR,
                    'weight_decay': WEIGHT_DECAY,
                    'batch_size': BATCH_SIZE,
                    'image_size': IMAGE_SIZE
                },
            }
        ckpt_path = CKPT_DIR / f"best_model_multiclass_seed_{SEED}_fold_{FOLD_CONFIG}.pth"
        torch.save(checkpoint, ckpt_path)
        print(f"\n Checkpoint guardado (F1-macro={val_f1:.4f})  →  {ckpt_path}")
 
 
# =============================================================================
# RESUMEN FINAL
# =============================================================================
 
elapsed = time.time() - start_time
history['time'].append(float(elapsed))
with open(CKPT_DIR / f'multiclass_seed_{SEED}_fold_{FOLD_CONFIG}_history.json', 'w') as f:
    json.dump(history, f, indent=4)
print(f"\n{'#'*60}")
print(f"  Entrenamiento completado en {elapsed/60:.1f} min")
print(f"  Mejor val F1-macro: {best_val_f1:.4f}")
print(f"  Checkpoint en:      {CKPT_DIR / 'best_model_multiclass_seed_{SEED}_fold_{FOLD_CONFIG}.pth'}")
print(f"{'#'*60}")

# =============================================================================
# INFERENCIA
# =============================================================================

best_checkpoint = torch.load(CKPT_DIR / f"best_model_multiclass_seed_{SEED}_fold_{FOLD_CONFIG}.pth", map_location=device)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()


def save_to_csv(y_true, y_prob, split_name):
    data = {}
    for i, cls in enumerate(SEVERITY_CLASSES):
        data[f"{cls}_true"] = y_true[:, i]
        data[f"{cls}_prob"] = y_prob[:, i]
    df = pd.DataFrame(data)
    df.to_csv(CKPT_DIR / f"predictions_multiclass_seed_{SEED}_fold{FOLD_CONFIG}_{split_name}.csv", index=False)

# Extraer y guardar Validación
print("Guardando predicciones de Validación...")
y_val_true, y_val_prob = extract_predictions(val_loader, model, device)
save_to_csv(y_val_true, y_val_prob, "val")

print("\n¡Todo listo! Historial, Checkpoint y Predicciones exportadas correctamente.")

