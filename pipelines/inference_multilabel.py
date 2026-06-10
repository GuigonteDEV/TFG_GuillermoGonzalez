from pathlib import Path
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import time as time
import random
import torch
from src.utils import Transforms, H5DatasetMultilabel, extract_predictions, folds_creation, get_dataset_split
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


# ---------------------------
# Creación índices Dataset
# ---------------------------

pt_files = list(H5_FILES.glob("*.h5"))
pt_files = sorted(pt_files, key=lambda f: int(f.stem.split('_')[0]))
pt_files = np.array(pt_files)

folds_files, wsi_data = folds_creation(pt_files, NUM_FOLDS, SEED)

train_idx, val_idx, test_idx = get_dataset_split(FOLD_CONFIG, folds_files, NUM_FOLDS)

val_files = pt_files[val_idx]
test_files = pt_files[test_idx]


# ---------------------------
# Inicialización Transforms Augmentation
# ---------------------------

train_transforms, val_transforms = Transforms(IMAGE_SIZE)

# ---------------------------
# Creación Dataset
# ---------------------------

# Definimos Dataloader de test (que no estaba en tu main.py)

val_dataset = H5DatasetMultilabel(val_files, transform = val_transforms)
test_dataset = H5DatasetMultilabel(test_files, transform=val_transforms)

val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = 0, worker_init_fn=seed_worker, generator=g)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo: {device}")

model   = EfficientNet(num_classes=NUM_CLASSES).to(device)

CKPT_DIR = ROOT / 'Model_output'
CKPT_DIR.mkdir(exist_ok=True)

best_checkpoint = torch.load(CKPT_DIR / f"best_model_seed_{SEED}_fold_{FOLD_CONFIG}_strategy_{DATA_STRATEGY}_{ALGO_STRATEGY}.pth", map_location=device)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()



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