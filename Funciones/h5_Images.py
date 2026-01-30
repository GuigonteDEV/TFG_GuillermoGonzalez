'''
Docstring for Tensor_Images

Construye los tensores con los que se alimentara la red neuronal a partir de las imagenes y archivo CSV del dataset.

Los tensores son de tipo Torch y de la forma:

· images_tensor: imagenes de los patch

· labels_tensor: etiqueta de los patch

· meta_continuous: informacion espacial de la imagen dentro de WSI global

· continuous_cols: indica cual que columna es cual dentro de la informacion de continuas
'''


import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import torch
import h5py


IGNORED_COLUMNS = ['burn_out_pct', 'low_saturation_pct', 'n_masks_for_slide']  
FNAME_COL = 'fname'

# Columnas de etiquetas de distinción entre TUMOR vs NO TUMOR
TUMOR_COLUMNS = ['highgrade_dysplasia', 'adenocarcinoma', 'suspicious_for_invasion',
            'lymphovascular_invasion', 'tumor_necrosis']

NOTUMOR_COLUMNS = ['normal', 'lowgrade_dysplasia', 'inflammation']

# Columnas de imagenes que se excluiran en el entrenamiento, no favorecen aprendizaje (no utiles)
EXCLUDE_COLUMNS = ['artifact', 'resection_edge']


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

def build_h5(df, dataset_publico_dir, n_slide, h5_out_path,  target_size=None, force_channels=3, use_torch=True):
    
    images_written = 0
    missing_images = []

    tumor_tot = 0
    no_tumor_tot = 0
    excl_art_resection = 0
    excl_conflict = 0
    excl_no_label = 0

    # Estimar número máximo (para prealocar)
    max_samples = len(df)

    with h5py.File(h5_out_path, "w") as h5:
        img_ds = h5.create_dataset(
            "images",
            shape=(max_samples, target_size[1], target_size[0], force_channels),
            maxshape=(None, target_size[1], target_size[0], force_channels),
            dtype="uint8",
            compression="gzip",
            compression_opts=4,
            chunks=(1, target_size[1], target_size[0], force_channels)
        )

        label_ds = h5.create_dataset(
            "labels",
            shape=(max_samples,),
            maxshape=(None,),
            dtype="uint8"
        )

    
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
            
            img_ds[images_written] = arr
            label_ds[images_written] = 1 if tumor_present else 0
            
            if tumor_present:
                tumor_tot += 1
            else:
                no_tumor_tot += 1
                
            images_written += 1
        
    
        # Recortar datasets al tamaño real
        #img_ds.resize((images_written, target_size[1], target_size[0], force_channels))
        #label_ds.resize((images_written,))
    
    return {
        "n_images": images_written,
        "tumor": tumor_tot,
        "no_tumor": no_tumor_tot,
        "excluded_artifact": excl_art_resection,
        "excluded_conflict": excl_conflict,
        "excluded_no_label": excl_no_label,
        "missing_images": missing_images
    }


ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto') 
out_dir = ROOT / 'processed_h5'
out_dir.mkdir(exist_ok=True)
    


for n_slide in range(1,201):
        n_slide_str = str(n_slide).zfill(3) 
        DATA_DIR = ROOT / 'Dataset_Publico' / f'zoom_2_{n_slide_str}'
        CSV_PATH = DATA_DIR / f'{n_slide_str}_labels.csv'
        out_dir = ROOT / 'processed_h5' / f'{n_slide_str}_h5.h5'
        TARGET_SIZE = (256, 256)   
        FORCE_CHANNELS = 3  
        
        df = load_csv(CSV_PATH)
        x = build_h5(df, DATA_DIR, n_slide_str, out_dir,  target_size=TARGET_SIZE, force_channels=3, use_torch=True)
        print("Guardado en:", out_dir)

        