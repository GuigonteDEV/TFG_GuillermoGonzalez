# archivo: build_tensors_cnn_v2.py
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

USE_TORCH = True
if USE_TORCH:
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

    if use_torch:
        images_torch = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
        labels_torch = torch.from_numpy(labels_np)
        meta_torch = torch.from_numpy(meta_np) if meta_np is not None else None
        return images_torch, labels_torch, meta_torch, LABEL_COLUMNS, continuous_cols, missing_images
    else:
        return images_np, labels_np, meta_np, LABEL_COLUMNS, continuous_cols, missing_images

if __name__ == '__main__':
    df = load_csv(CSV_PATH)
    images_tensor, labels_tensor, meta_tensor, label_cols, continuous_cols, missing = build_tensors(
        df,
        DATA_DIR,
        target_size=TARGET_SIZE,
        force_channels=FORCE_CHANNELS,
        use_torch=USE_TORCH
    )
    print("Tamaño imágenes tensor:", images_tensor.shape)
    print("Tamaño etiquetas tensor:", labels_tensor.shape)
    if meta_tensor is not None:
        print("Tamaño metadata continua:", meta_tensor.shape)
    print("Columnas etiquetas:", label_cols)
    if continuous_cols:
        print("Columnas continuas:", continuous_cols)
    if missing:
        print(f"{len(missing)} imágenes faltantes / errores (primeros 10):", missing[:10])

    out_dir = ROOT / 'processed'
    out_dir.mkdir(exist_ok=True)
    if USE_TORCH:
        torch.save({
            'images': images_tensor,
            'labels': labels_tensor,
            'meta_continuous': meta_tensor,
            'label_cols': label_cols,
            'continuous_cols': continuous_cols
        }, out_dir / '001_tensor.pt')
        print("Guardado en:", out_dir / '001_tensor.pt')
    else:
        np.save(out_dir / 'images.npy', images_tensor)
        np.save(out_dir / 'labels.npy', labels_tensor)
        if meta_tensor is not None:
            np.save(out_dir / 'meta.npy', meta_tensor)
        with open(out_dir / 'label_cols.txt', 'w') as f:
            f.write('\n'.join(label_cols))
        if continuous_cols:
            with open(out_dir / 'continuous_cols.txt', 'w') as f:
                f.write('\n'.join(continuous_cols))
        print("Guardado en:", out_dir)


#Comprobacion de la transformacion de la imagen, y su no destrucción

from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

img_tensor = images_tensor[1]  # (3, H, W) con valores entre 0 y 1

# Convertir a imagen PIL
img = to_pil_image(img_tensor)  # esto reescala 0–1 → 0–255 automáticamente

# Mostrar
plt.imshow(img)
plt.axis('off')
plt.show()