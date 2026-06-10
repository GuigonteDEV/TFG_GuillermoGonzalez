from pathlib import Path
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import json
import argparse
import time as time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils import summarize_h5_files, print_split_summary, Transforms, H5DatasetMultilabel, compute_multilabel_metrics, print_metrics, extract_predictions, folds_creation, get_dataset_split, folds_statistics
from src.balancing_methods import compute_pos_weight_efective, AsymmetricLoss, compute_sample_weights
from src.models import EfficientNet


# ---------------------------
# Configuración general universal
# ---------------------------

parser = argparse.ArgumentParser(description="Entrenamiento Biopsias HTCondor")
parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria para reproducibilidad')
parser.add_argument('--fold', type=int, required=True, help="Numero de fold (1-5)")
parser.add_argument('--data', type=str, required=True, choices=['None', 'WS'])
parser.add_argument('--algo', type=str, required=True, choices=['None', 'ASL', 'PW'])
args = parser.parse_args()

CLASS_NAMES = [
    'normal',
    'lowgrade_dysplasia',
    'inflammation',
    'highgrade_dysplasia',
    'tumor_necrosis',
    'suspicious_for_invasion',
    'adenocarcinoma',
]

ROOT = Path('.') 


H5_FILES = ROOT / 'Dataset' 
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_CLASSES = len(CLASS_NAMES)
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

# ---------------------------
# Configuración de Estrategias
# ---------------------------
DATA_STRATEGY = args.data  # Opciones: 'None', 'WS'
ALGO_STRATEGY = args.algo  # Opciones: 'None', 'Focal', 'PW'
        
        
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

folds_files, wsi_data = folds_creation(pt_files, NUM_FOLDS, SEED)

folds_statistics(pt_files, folds_files, NUM_FOLDS)

train_idx, val_idx, test_idx = get_dataset_split(FOLD_CONFIG, folds_files, NUM_FOLDS)

train_files = pt_files[train_idx]
val_files = pt_files[val_idx]

# Resumen de distribución de clases
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
# DataLoaders
# ---------------------------

train_sampler = None
shuffle_train = True

if DATA_STRATEGY == 'WS':
    print("Aplicando: Weighted Sampler")
    shuffle_train = False 
    train_sampler = compute_sample_weights(train_files)

elif DATA_STRATEGY == 'None':
    print("Aplicando: Baseline (Datos originales)")

else:
    raise ValueError(f"Estrategia de datos desconocida: {DATA_STRATEGY}")


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
 
        probs = torch.sigmoid(logits).detach().cpu().numpy()
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

if ALGO_STRATEGY == 'PW':
    print("Aplicando Loss: BCE con Pos Weights (PW)")
    pos_weight = compute_pos_weight_efective(train_files).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = pos_weight)


elif ALGO_STRATEGY == 'ASL':
    print("Aplicando Loss: Asymmetric Loss")
    loss_fn = AsymmetricLoss()


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
    'time': [],
    'fold': FOLD_CONFIG,
    'seed': SEED,
    'config': {
        'EPOCHS': EPOCHS,
        'lr': LR,
        'strategy_data': DATA_STRATEGY,
        'strategy_algo': ALGO_STRATEGY
    },
    'train': {
        'lr': [],
        'loss': [],
        'roc_macro': [],
        'pr_macro': [],
        'roc_per_class': [], 
        'pr_per_class': [] 
    },
    'val': {
        'loss': [],
        'roc_macro': [],
        'pr_macro': [],
        'roc_per_class': [], 
        'pr_per_class': []   
    }
}

best_val_pr = 0.0

# =============================================================================
# BUCLE DE ENTRENAMIENTO
# =============================================================================

CKPT_DIR = ROOT / 'Model_output'
CKPT_DIR.mkdir(exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    if epoch == 61:
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
    
    current_lr = optimizer.param_groups[0]['lr']
 
    # Guardar histórico
    history['train']['lr'].append(float(current_lr))
    history['train']['loss'].append(float(train_loss))
    history['train']['roc_macro'].append(float(train_roc))
    history['train']['pr_macro'].append(float(train_pr))
    history['train']['roc_per_class'].append({k: float(v) for k, v in train_roc_cls.items()})
    history['train']['pr_per_class'].append({k: float(v) for k, v in train_pr_cls.items()})

    history['val']['loss'].append(float(val_loss))
    history['val']['roc_macro'].append(float(val_roc))
    history['val']['pr_macro'].append(float(val_pr))
    history['val']['roc_per_class'].append({k: float(v) for k, v in val_roc_cls.items()})
    history['val']['pr_per_class'].append({k: float(v) for k, v in val_pr_cls.items()})
    
    # Guardar JSON
    with open(CKPT_DIR / f'seed_{SEED}_fold_{FOLD_CONFIG}_strategy_{DATA_STRATEGY}_{ALGO_STRATEGY}_history.json', 'w') as f:
        json.dump(history, f, indent=4)
 
    # Imprimir métricas
    print_metrics("TRAIN", train_loss, train_roc, train_pr, train_roc_cls, train_pr_cls)
    print_metrics("VAL  ", val_loss,   val_roc,   val_pr,   val_roc_cls,   val_pr_cls)
    print(f"\n  LR actual: {optimizer.param_groups[0]['lr']:.2e}")
 
    # Checkpoint si mejora PR-AUC macro en validación 
    if val_pr > best_val_pr:
        best_val_pr = val_pr
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_pr_macro': val_pr,
                'val_loss': val_loss,
                'config': {
                    'strategy_data': DATA_STRATEGY,
                    'strategy_algo': ALGO_STRATEGY,
                    'lr': LR,
                    'weight_decay': WEIGHT_DECAY,
                    'batch_size': BATCH_SIZE,
                    'image_size': IMAGE_SIZE
                },
                'class_names': CLASS_NAMES,
            }
        ckpt_path = CKPT_DIR / f"best_model_seed_{SEED}_fold_{FOLD_CONFIG}_strategy_{DATA_STRATEGY}_{ALGO_STRATEGY}.pth"
        torch.save(checkpoint, ckpt_path)
        print(f"\n Checkpoint guardado (PR-macro={val_pr:.4f})  →  {ckpt_path}")
 
 
# =============================================================================
# RESUMEN FINAL
# =============================================================================
 
elapsed = time.time() - start_time
history['time'].append(float(elapsed))
with open(CKPT_DIR / f'seed_{SEED}_fold_{FOLD_CONFIG}_strategy_{DATA_STRATEGY}_{ALGO_STRATEGY}_history.json', 'w') as f:
    json.dump(history, f, indent=4)
print(f"\n{'#'*60}")
print(f"  Entrenamiento completado en {elapsed/60:.1f} min")
print(f"  Mejor val PR-macro: {best_val_pr:.4f}")
print(f"  Checkpoint en:      {CKPT_DIR / 'best_model.pth'}")
print(f"{'#'*60}")

# =============================================================================
# INFERENCIA
# =============================================================================

best_checkpoint = torch.load(CKPT_DIR / f"best_model_seed_{SEED}_fold_{FOLD_CONFIG}_strategy_{DATA_STRATEGY}_{ALGO_STRATEGY}.pth", map_location=device)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

# Definimos Dataloader de test (que no estaba en tu main.py)
test_files = pt_files[test_idx]
test_dataset = H5DatasetMultilabel(test_files, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4, worker_init_fn=seed_worker, generator=g)

def save_to_csv(y_true, y_prob, split_name):
    data = {}
    for i, cls in enumerate(CLASS_NAMES):
        data[f"{cls}_true"] = y_true[:, i]
        data[f"{cls}_prob"] = y_prob[:, i]
    df = pd.DataFrame(data)
    df.to_csv(CKPT_DIR / f"predictions_seed_{SEED}_fold{FOLD_CONFIG}_{DATA_STRATEGY}_{ALGO_STRATEGY}_{split_name}.csv", index=False)

# Extraer y guardar Validación
print("Guardando predicciones de Validación...")
y_val_true, y_val_prob = extract_predictions(val_loader, model, device)
save_to_csv(y_val_true, y_val_prob, "val")

# Extraer y guardar Test
print("Guardando predicciones de Test...")
y_test_true, y_test_prob = extract_predictions(test_loader, model, device)
save_to_csv(y_test_true, y_test_prob, "test")

print("\n¡Todo listo! Historial, Checkpoint y Predicciones exportadas correctamente.")


