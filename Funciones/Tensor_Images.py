# archivo: build_tensors_cnn_v2.py
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import torch


IGNORED_COLUMNS = ['burn_out_pct', 'low_saturation_pct', 'n_masks_for_slide']  
FNAME_COL = 'fname'

# Columnas de etiquetas de distinción entre TUMOR vs NO TUMOR
TUMOR_COLUMNS = ['highgrade_dysplasia', 'adenocarcinoma', 'suspicious_for_invasion',
            'lymphovascular_invasion', 'tumor_necrosis']

NOTUMOR_COLUMNS = ['normal', 'lowgrade_dysplasia', 'inflammation']

# Columnas de imagenes que se excluiran en el entrenamiento, no favorecen aprendizaje (no utiles)
EXCLUDE_COLUMNS = ['artifact', 'resection_edge']

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

def build_tensors(df, dataset_publico_dir, n_slide,  target_size=None, force_channels=3, use_torch=True):
    
    images = []
    binary_labels = []
    meta_rows = []
    missing_images = []
    
    excl_art_resection = 0
    excl_conflict = 0
    excl_no_label = 0
    
    # columnas continuas a mantener (solo KEEP_COLUMNS)
    continuous_cols = [c for c in df.columns if c in KEEP_COLUMNS]
    
    for idx, row in df.iterrows():
        fname = str(row[FNAME_COL]).strip()
        parts = fname.split('/')
        if len(parts) >= 2:
            tail = os.path.join(*parts[1:])
        else:
            tail = fname
        img_path = dataset_publico_dir.parent / f'zoom_2_{n_slide}' / tail
        if not img_path.exists():
            missing_images.append((idx, str(img_path)))
            continue
        
        excluded_flag = False
        for c in EXCLUDE_COLUMNS:
            try:
                if int(row.get(c, 0)) != 0:
                    excl_art_resection += 1
                    excluded_flag = True
                    break
            except:
                # si valor no convertible, tratamos como 0
                pass
        if excluded_flag:
            continue
        
        # etiquetas multilabel: vector de 0/1
        
        tumor_present = any(int(row.get(c, 0)) != 0 if str(row.get(c, 0)).strip() != '' else False for c in TUMOR_COLUMNS)
        notumor_present = any(int(row.get(c, 0)) != 0 if str(row.get(c, 0)).strip() != '' else False for c in NOTUMOR_COLUMNS)
        
        # Las siguientes lineas es de comprobación de que esta todo OK
        if tumor_present and notumor_present:
            excl_conflict += 1
            
            #continue
        
        if not (tumor_present or notumor_present):
            excl_no_label += 1
            continue
        
        try:
            arr = load_image_as_array(img_path, target_size, force_channels)
        except Exception as e:
            print(f"Error cargando imagen {img_path}: {e}")
            missing_images.append((idx, str(img_path)))
            continue
        images.append(arr)
        
        # etiqueta binaria: 1 si tumor_present, 0 si notumor_present
        binary_labels.append(1 if tumor_present else 0)
        
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
    labels_np = np.array(binary_labels, dtype=np.float32)  
    meta_np = np.array(meta_rows, dtype=np.float32) if continuous_cols else None
    
    images_torch = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
    labels_torch = torch.from_numpy(labels_np)
    meta_torch = torch.from_numpy(meta_np) if meta_np is not None else None
    
    return images_torch, labels_torch, meta_torch, continuous_cols, missing_images, excl_art_resection, excl_conflict, excl_no_label


