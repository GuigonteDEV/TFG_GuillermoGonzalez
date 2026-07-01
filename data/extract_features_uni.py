import os
from pathlib import Path
import argparse
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from huggingface_hub import login
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CONFIGURACIÓN 
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Entrenamiento Biopsias HTCondor")
parser.add_argument('--token', type=str, required=True, help="TOKEN UNI")
args = parser.parse_args()

ROOT         = Path('.')
H5_DIR       = ROOT / 'h5_multiclass_UNI'       
OUT_DIR      = ROOT / 'features_UNI'             
CKPT_DIR     = ROOT / 'assets' / 'ckpts' / 'vit_large_patch16_224.dinov2.uni_mass100k'

BATCH_SIZE   = 32       
NUM_WORKERS  = 4        # workers del DataLoader
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKEN = args.token


# ---------------------------------------------------------------------------
# DATASET 
# ---------------------------------------------------------------------------
    
class H5Dataset(Dataset):
    def __init__(self, h5_files, transform):
        self.h5_path   = h5_path
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            self.n_samples = f['images'].shape[0]
                
 
    def __len__(self) -> int:
        return self.n_samples
 
    def __getitem__(self, idx):
        with h5py.File(h5_path, "r") as h5:
            img    = h5["images"][idx]          
            labels = h5["labels"][idx]
            
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = self.transform(img)
        
        labels = torch.tensor(labels, dtype=torch.long)
 
        return img, labels


# ---------------------------------------------------------------------------
# CARGA DEL MODELO UNI
# ---------------------------------------------------------------------------

def load_uni_model(ckpt_dir, device):
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
        num_classes=0,        
        dynamic_img_size=True
    )
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)

    # Congelamos todos los parámetros
    for param in model.parameters():
        param.requires_grad = False

    print(f"UNI cargado en {device} | Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ---------------------------------------------------------------------------
# EXTRACCIÓN DE FEATURES
# ---------------------------------------------------------------------------

def extract_features(model, dataloader, device):
    """
    Pasa todos los batches por UNI y acumula features y labels.
    Devuelve arrays numpy: features (N, 1024), labels (N,1)
    """
    all_features = []
    all_labels   = []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="  Extrayendo", leave=False, disable=True):
            imgs = imgs.to(device)

            # Forward pass — num_classes=0 devuelve el CLS token: (B, 1024)
            feats = model(imgs)

            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels   = np.concatenate(all_labels,   axis=0)

    return features, labels


# ---------------------------------------------------------------------------
# SCRIPT PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    uni_normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    # Carga del modelo
    print("="*55)
    print("Cargando modelo UNI...")
    print("="*55)
    model = load_uni_model(CKPT_DIR, DEVICE)

    # Acumuladores globales para resumen final
    total_slides    = 0
    total_patches   = 0
    total_skipped   = 0

    # Bucle por slide
    for n_slide in range(1, 201):
        n_slide_str = str(n_slide).zfill(3)
        h5_path  = H5_DIR  / f'{n_slide_str}_multiclass_UNI.h5'
        npz_path = OUT_DIR / f'{n_slide_str}_features.npz'

        # Skip si el h5 no existe
        if not h5_path.exists():
            continue

        if npz_path.exists():
            print(f"[SKIP] Slide {n_slide_str} — ya procesado")
            total_skipped += 1
            continue

        print(f"\nSlide {n_slide_str} | {h5_path.name}")

        dataset = H5Dataset(h5_path, transform=uni_normalize)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,         
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE == 'cuda')
        )

        features, labels = extract_features(model, dataloader, DEVICE)

        # Guardamos features y labels juntos — correspondencia garantizada
        np.savez_compressed(npz_path, features=features, labels=labels)

        print(f"  Patches: {features.shape[0]:,} | Features shape: {features.shape} | Guardado: {npz_path.name}")

        total_slides  += 1
        total_patches += features.shape[0]

    # Resumen final
    print(f"\n{'#'*55}")
    print("  RESUMEN FINAL")
    print(f"{'#'*55}")
    print(f"  Slides procesados : {total_slides}")
    print(f"  Slides saltados   : {total_skipped}")
    print(f"  Total patches     : {total_patches:,}")
    print(f"  Features por patch: 1024 (CLS token UNI)")
    print(f"  Guardado en       : {OUT_DIR}")
