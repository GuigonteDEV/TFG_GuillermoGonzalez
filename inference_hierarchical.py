import torch
import numpy as np
from pathlib import Path
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import json
import argparse
import time as time
import random
from torchvision import models
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from Funciones.Augmentation_Dataloader import Transforms, H5DatasetInferencia
from Funciones.KFCV_Create import folds_creation, get_dataset_split, folds_statistics

# ---------------------------
# Configuración general universal
# ---------------------------

parser = argparse.ArgumentParser(description="Entrenamiento Biopsias HTCondor")
parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria para reproducibilidad')
parser.add_argument('--fold', type=int, required=True, help="Numero de fold (1-5)")
args = parser.parse_args()

ROOT = Path('.') 


H5_FILES = ROOT / 'Dataset'
H5_FILES_BINARY = ROOT / 'Dataset_multiclass'
CKPT_DIR_BINARY = ROOT / 'Model_output_binary'
CKPT_DIR_MULTICLASS = ROOT / 'Model_output_multiclass'
CKPT_DIR = ROOT / 'Model_output_hierarchical'
CKPT_DIR.mkdir(exist_ok=True)  
IMAGE_SIZE = 256
BATCH_SIZE = 32
LR         = 1e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 100
SEED = args.seed


SEVERITY_CLASSES = [
    'inflammation',            # 1 — menor riesgo patológico
    'lowgrade_dysplasia',      # 2
    'highgrade_dysplasia',     # 3
    'tumor_necrosis',          # 4
    'suspicious_for_invasion', # 5
    'adenocarcinoma',          # 6 — mayor riesgo
]
NUM_CLASSES_MULTI = len(SEVERITY_CLASSES)

NUM_CLASSES_B = 1

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
# Inicialización Transforms Augmentation
# ---------------------------

train_transforms, val_transforms = Transforms(IMAGE_SIZE)

# ---------------------------
# Creación índices Dataset
# ---------------------------

pt_files = list(H5_FILES.glob("*.h5"))
pt_files = sorted(pt_files, key=lambda f: int(f.stem.split('_')[0]))
pt_files = np.array(pt_files)

binary_files = list(H5_FILES_BINARY.glob("*.h5"))
binary_files = sorted(binary_files, key=lambda f: int(f.stem.split('_')[0]))
binary_files = np.array(binary_files)

folds_files, wsi_data = folds_creation(pt_files, NUM_FOLDS, SEED)

folds_statistics(pt_files, folds_files, NUM_FOLDS)

train_idx, val_idx, test_idx = get_dataset_split(FOLD_CONFIG, folds_files, NUM_FOLDS)

test_files = binary_files[test_idx]


test_dataset = H5DatasetInferencia(test_files, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = 4, worker_init_fn=seed_worker, generator=g)

class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")

        # Freeze backbone completo
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
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_binary = EfficientNet(num_classes=NUM_CLASSES_B).to(device)
best_checkpoint_binary = torch.load(CKPT_DIR_BINARY / f"best_model_binary_seed_{SEED}_fold_{FOLD_CONFIG}.pth", map_location=device)
model_binary.load_state_dict(best_checkpoint_binary['model_state_dict'])
best_thres = best_checkpoint_binary['threshold']
model_binary.eval()

model_multiclass = EfficientNet(num_classes=NUM_CLASSES_MULTI).to(device)
best_checkpoint_multiclass = torch.load(CKPT_DIR_MULTICLASS / f"best_model_multiclass_seed_{SEED}_fold_{FOLD_CONFIG}.pth", map_location=device)
model_multiclass.load_state_dict(best_checkpoint_multiclass['model_state_dict'])
model_multiclass.eval()


list_global_preds = []
list_global_trues = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        batch_size = images.size(0)
        
        batch_preds_global = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        logits_bin = model_binary(images)
        probs_bin = torch.sigmoid(logits_bin).squeeze(1)
        preds_bin = (probs_bin >= best_thres).long()            
        
        indices_positivos = torch.where(preds_bin == 1)[0]
        
        if len(indices_positivos) > 0:
            patches_patologicos = images[indices_positivos]
            
            logits_multi = model_multiclass(patches_patologicos)
            preds_multi = torch.argmax(logits_multi, dim=1)      
            
            preds_multi_mapeadas = preds_multi + 1
            
            batch_preds_global[indices_positivos] = preds_multi_mapeadas
            
        list_global_preds.append(batch_preds_global.cpu().numpy())
        list_global_trues.append(labels.numpy())

y_pred_final = np.concatenate(list_global_preds, axis=0)
y_true_final = np.concatenate(list_global_trues, axis=0)

class_names = ['normal'] + SEVERITY_CLASSES

def save_to_csv(y_true, y_pred, split_name, dataset_object):
    image_ids = [
        f"{Path(h5_path).stem}_patch_{local_idx}" 
        for h5_path, local_idx in dataset_object.index
    ]
    
    df = pd.DataFrame({
        "image_id": image_ids,
        "true_label": y_true,
        "true_class": [class_names[t] for t in y_true],
        "pred_label": y_pred,
        "pred_class": [class_names[p] for p in y_pred],
        "correct": y_true == y_pred
    })
    
    df.to_csv(
        CKPT_DIR / f"predictions_hierarchical_seed_{SEED}_fold_{FOLD_CONFIG}_{split_name}.csv",
        index=False
    )

save_to_csv(y_true_final, y_pred_final, "test", test_dataset)