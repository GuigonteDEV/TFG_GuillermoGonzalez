# archivo: build_tensors_cnn_v2.py
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import torch

ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')  
DATA_DIR = ROOT / 'Dataset_Publico' / 'zoom_2_001'
CSV_PATH = DATA_DIR / '001_labels.csv'  
LOCAL_FOLDER_NAME = 'zoom_2_001'  
IMAGE_FOLDER_INSIDE = '001'  
TARGET_SIZE = (256, 256)   
FORCE_CHANNELS = 3         

IGNORED_COLUMNS = ['burn_out_pct', 'low_saturation_pct', 'n_masks_for_slide']  
FNAME_COL = 'fname'

# Columnas de etiquetas binarias para CNN multilabel
LABEL_COLUMNS = [
    'highgrade_dysplasia', 'adenocarcinoma', 'suspicious_for_invasion',
    'lymphovascular_invasion', 'inflammation', 'resection_edge', 
    'tumor_necrosis', 'artifact', 'normal', 'lowgrade_dysplasia'
]

# Columnas que queremos mantener aunque no sean etiquetas, ya se dropearan en el entrenamiento
KEEP_COLUMNS = ['topleft_x', 'topleft_y']

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    for c in IGNORED_COLUMNS:
        if c in df.columns:
            df = df.drop(columns=c)
    if FNAME_COL not in df.columns:
        raise ValueError(f"No existe columna '{FNAME_COL}' en {csv_path}")
    return df

def Local_folder():
    return LOCAL_FOLDER_NAME

def load_image_as_array(path, target_size=None, force_channels=3):
    img = Image.open(path)
    if force_channels == 3:
        img = img.convert('RGB')
    elif force_channels == 1:
        img = img.convert('L')
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img)
    if force_channels == 3 and arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

def build_tensors(df, dataset_publico_dir, target_size=None, force_channels=3, use_torch=True):
    images = []
    labels = []
    meta_rows = []
    missing_images = []

    # columnas continuas a mantener (solo KEEP_COLUMNS)
    continuous_cols = [c for c in df.columns if c in KEEP_COLUMNS]

    for idx, row in df.iterrows():
        fname = str(row[FNAME_COL]).strip()
        parts = fname.split('/')
        if len(parts) >= 2:
            tail = os.path.join(*parts[1:])
        else:
            tail = fname
        img_path = dataset_publico_dir.parent / Local_folder() / tail
        if not img_path.exists():
            missing_images.append((idx, str(img_path)))
            continue
        try:
            arr = load_image_as_array(img_path, target_size, force_channels)
        except Exception as e:
            print(f"Error cargando imagen {img_path}: {e}")
            missing_images.append((idx, str(img_path)))
            continue
        images.append(arr)

        # etiquetas multilabel: vector de 0/1
        label_vector = []
        for c in LABEL_COLUMNS:
            val = row[c]
            try:
                val_int = int(val)
                val_int = 1 if val_int != 0 else 0
            except:
                val_int = 0
            label_vector.append(val_int)
        labels.append(label_vector)

        # columnas continuas (KEEP_COLUMNS)
        meta_vector = []
        for c in continuous_cols:
            val = row[c]
            try:
                meta_vector.append(float(val))
            except:
                meta_vector.append(np.nan)
        meta_rows.append(meta_vector)

    if len(images) == 0:
        raise RuntimeError("No se cargó ninguna imagen válida. Revisa paths y CSV.")

    images_np = np.stack(images, axis=0)  
    labels_np = np.array(labels, dtype=np.float32)  
    meta_np = np.array(meta_rows, dtype=np.float32) if continuous_cols else None

    images_torch = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
    labels_torch = torch.from_numpy(labels_np)
    meta_torch = torch.from_numpy(meta_np) if meta_np is not None else None
    return images_torch, labels_torch, meta_torch, LABEL_COLUMNS, continuous_cols, missing_images


