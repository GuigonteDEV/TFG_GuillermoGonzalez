from pathlib import Path
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
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
from Funciones.Augmentation_Dataloader import Dataset_Division, summarize_file_h5, Transforms, H5DatasetSoft, WeightedSampler, compute_pos_weight
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, average_precision_score


# ---------------------------
# Configuración general universal
# ---------------------------

ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto') 

Create_Tensor = False

Create_WSI = False


################################
# ---------------------------
# Creación de Tensores WSI
# ---------------------------
################################

if Create_Tensor:
    excl_art_resection_tot = 0
    excl_conflict_tot = 0
    excl_no_label_tot = 0
    tumor_tot = 0
    no_tumor_tot = 0
    patch_tot = 0
    for n_slide in range(1, 201):
        n_slide_str = str(n_slide).zfill(3) 
        DATA_DIR = ROOT / 'Dataset_Publico' / f'zoom_2_{n_slide_str}'
        CSV_PATH = DATA_DIR / f'{n_slide_str}_labels.csv'
        TARGET_SIZE = (256, 256)   
        FORCE_CHANNELS = 3  
        
        df = load_csv(CSV_PATH)
        images_tensor, labels_tensor, meta_tensor, continuous_cols, missing, excl_art_resection, excl_conflict, excl_no_label, patch_tumor, patch_no_tumor = build_tensors(
            df,
            DATA_DIR,
            n_slide = n_slide_str,
            target_size=TARGET_SIZE,
            force_channels=FORCE_CHANNELS,
            use_torch=True
        )
        print("Tamaño imágenes tensor:", images_tensor.shape)
        if missing:
            print(f"{len(missing)} imágenes faltantes / errores (primeros 10):", missing[:10])
        print(f'Número exclusiones normales:', excl_art_resection)
        print(f'Número exclusiones conflictos:', excl_conflict)
        print(f'Número patches:', len(labels_tensor))
        print(f'Número patches tumor:', patch_tumor)
        print(f'Número patches no tumor:', patch_no_tumor)
        
        out_dir = ROOT / 'processed'
        out_dir.mkdir(exist_ok=True)
        torch.save({
            'images': images_tensor,
            'labels': labels_tensor,
            'meta_continuous': meta_tensor,
            'continuous_cols': continuous_cols
        }, out_dir / f'{n_slide_str}_tensor.pt')
        print("Guardado en:", out_dir / f'{n_slide_str}_tensor.pt')
        
        excl_art_resection_tot += excl_art_resection
        excl_conflict_tot += excl_conflict
        excl_no_label_tot += excl_no_label
        patch_tot += len(labels_tensor)
        tumor_tot += patch_tumor
        no_tumor_tot += patch_no_tumor
        
    
    print(f'Número total exclusiones normales', excl_art_resection_tot)
    print(f'Número total exclusiones conflictos:', excl_conflict_tot)
    print(f'Número total exclusiones sin etiqueta:', excl_no_label_tot)
    print(f'Número total patches', patch_tot)
    print(f'Número total patches tumor:', tumor_tot)
    print(f'Número total patches no tumor', no_tumor_tot)
    


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

# ---------------------------
# Configuración general
# ---------------------------

CSV_PATH = ROOT / 'Statistics' / 'WSI_stats.csv'
H5_SOFT_DIR = ROOT / 'processed_h5_soft' 
IMAGE_SIZE = 256
BATCH_SIZE = 32 

#Inicio cronómetro
start_time = time.time()

# ---------------------------
#Creación índices Dataset
# ---------------------------

train_idx, val_idx = Dataset_Division(CSV_PATH)

pt_files = list(H5_SOFT_DIR.glob("*.h5"))
pt_files = sorted(pt_files, key=lambda f: int(f.stem.split('_')[0]))
pt_files = np.array(pt_files)

train_files = pt_files[train_idx]
val_files = pt_files[val_idx]


train_imgs, train_tumors, train_notumors, t_adenocarcinoma, t_suspicious_for_invasion, t_highgrade_dysplasia, t_tumor_necrosis, t_lowgrade_dysplasia, t_inflammation, t_normal = summarize_file_h5(train_files)
print('###################')
val_imgs, val_tumors, val_notumors, v_adenocarcinoma, v_suspicious_for_invasion, v_highgrade_dysplasia, v_tumor_necrosis, v_lowgrade_dysplasia, v_inflammation, v_normal = summarize_file_h5(val_files)

print("========== TRAIN ==========")
print(f"Imágenes totales: {train_imgs}")
print(f"Tumor: {train_tumors}")
print(f"No tumor: {train_notumors}")
print(f"Adenocarcinoma: {t_adenocarcinoma}")
print(f"Suspicious for invasion: {t_suspicious_for_invasion}")
print(f"High-grade dysplasia: {t_highgrade_dysplasia}")
print(f"Tumor necrosis: {t_tumor_necrosis}")
print(f"Low-grade dysplasia: {t_lowgrade_dysplasia}")
print(f"Inflammation: {t_inflammation}")
print(f"Normal: {t_normal}")

print("\n========== VALIDATION ==========")
print(f"Imágenes totales: {val_imgs}")
print(f"Tumor: {val_tumors}")
print(f"No tumor: {val_notumors}")
print(f"Adenocarcinoma: {v_adenocarcinoma}")
print(f"Suspicious for invasion: {v_suspicious_for_invasion}")
print(f"High-grade dysplasia: {v_highgrade_dysplasia}")
print(f"Tumor necrosis: {v_tumor_necrosis}")
print(f"Low-grade dysplasia: {v_lowgrade_dysplasia}")
print(f"Inflammation: {v_inflammation}")
print(f"Normal: {v_normal}")


# ---------------------------
#Inicialización Transforms Augmentation
# ---------------------------

train_transforms, val_transforms = Transforms(IMAGE_SIZE)


# ---------------------------
# Creación subset
# ---------------------------

subset_train_idx = train_idx[:int(len(train_idx) * 0.1)]
subset_val_idx = val_idx[:int(len(val_idx) * 0.1)]


# ---------------------------
# Creación subset
# ---------------------------

train_dataset = H5DatasetSoft(train_files, transform = train_transforms)
val_dataset = H5DatasetSoft(val_files, transform = val_transforms)


# ---------------------------
# WeightedRandomSampler para train
# ---------------------------

#train_sampler = WeightedSampler(train_dataset.labels)


# ---------------------------
# DataLoaders
# ---------------------------
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, shuffle = False, num_workers = 4)

#Las especificaciones de num_workers puede variar segun ordenador



################################
# ---------------------------
# Implementación Modelo + Entrenamiento
# ---------------------------
################################

epochs = 4


# ---------------------------
# Modelo ResNet18
# ---------------------------
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 1)
        )  # salida escalar

    def forward(self, x):
        return self.backbone(x) 

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b2(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)
    

def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    losses = []
    all_probs, all_labels = [], []

    for X, y_soft, y_hard in tqdm(dataloader):
        X, y_soft, y_hard = X.to(device), y_soft.float().to(device), y_hard.float().to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(X).squeeze(1)
            loss = loss_fn(logits, y_soft)

        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        probs = torch.sigmoid(logits)

        # Guardar para métricas globales
        all_probs.append(probs.detach().cpu())
        all_labels.append(y_hard.detach().cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    roc = roc_auc_score(all_labels, all_probs)
    pr  = average_precision_score(all_labels, all_probs)

    return np.mean(losses), roc, pr


def val_loop(dataloader, model, loss_fn, device):
    model.eval()
    losses = []
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X, y_soft, y_hard in tqdm(dataloader):
            X, y_soft, y_hard = X.to(device), y_soft.float().to(device), y_hard.float().to(device)
            logits = model(X).squeeze(1)
            loss = loss_fn(logits, y_soft)
            losses.append(loss.item())

            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(y_hard.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    roc = roc_auc_score(all_labels, all_probs)
    pr  = average_precision_score(all_labels, all_probs)

    return np.mean(losses), roc, pr

# -----------------------------
# DEVICE Y SCALER
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler("cuda")
model = EfficientNet().to(device)

# -----------------------------
# POS WEIGHT
# -----------------------------
pos_weight = compute_pos_weight(train_files).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# -----------------------------
# FASE 2: descongelar último bloque del backbone
# -----------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

train_losses = []
test_losses = []
train_prs = []
val_prs = []
train_rocs = []
val_rocs = []

output_dir = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto\Codigo\Model_Checkpoints')
output_dir.mkdir(exist_ok=True)

best_pr = 0

# -----------------------------
# TRAIN LOOP 
# -----------------------------
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    
    train_loss, train_roc, train_pr = train_loop(train_loader, model, loss_fn, optimizer, device)
    val_loss, val_roc, val_pr = val_loop(val_loader, model, loss_fn, device)

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    test_losses.append(val_loss)
    train_prs.append(train_pr)
    val_prs.append(val_pr)
    train_rocs.append(train_roc)
    val_rocs.append(val_roc)

    if val_pr > best_pr:
        best_pr = val_pr
        checkpoint = {
            'epoch': t+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'metrics': val_pr
        }
        torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))

    print(f"Train Loss: {train_loss:.4f}, ROC: {train_roc:.4f}, PR: {train_pr:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, ROC: {val_roc:.4f}, PR: {val_pr:.4f}")
    print("Current learning rate:", scheduler.get_last_lr())

end_time = time.time()
dense_elapsed_time = end_time - start_time