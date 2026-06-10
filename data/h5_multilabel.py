"""
h5_Images.py
============
Construye archivos HDF5 a partir de las imágenes y CSVs del dataset de biopsias.
 
Formato de salida por archivo .h5
----------------------------------
  images       : uint8  (N, H, W, 3)   — imágenes en píxeles [0,255]
  labels       : uint8  (N, NUM_CLASSES) — vector multilabel binario por patch
  class_names  : atributo del dataset   — lista ordenada de nombres de clase
 
Clases (orden fijo, índice = posición en el vector):
  0  normal
  1  lowgrade_dysplasia
  2  inflammation
  3  highgrade_dysplasia
  4  tumor_necrosis
  5  suspicious_for_invasion
  6  lymphovascular_invasion
  7  adenocarcinoma
 
Exclusiones (no forman parte del vector, filtran el patch):
  artifact, resection_edge  →  patch descartado completamente
 
Un patch se descarta también si ninguna de las 8 clases tiene valor 1.
Un patch con múltiples etiquetas activas es completamente válido:
  el vector tendrá varios 1. Ej: adenocarcinoma=1, inflammation=1 → [0,0,1,0,0,0,0,1]
 
Notas de diseño
---------------
- Las imágenes se guardan en su tamaño nativo (sin resize) por defecto.
  El resize se delega al DataLoader para no comprometer los H5 si se
  quiere experimentar con distintas resoluciones de entrada.
- Si target_size se especifica, se aplica resize antes de guardar.
- Compresión gzip level 4: buen equilibrio velocidad/espacio para uint8.
- chunks=(1, H, W, 3): lectura aleatoria óptima para el DataLoader.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import torch
import h5py


# ---------------------------------------------------------------------------
# Definición de clases — ORDEN FIJO Y DOCUMENTADO
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    'normal',
    'lowgrade_dysplasia',
    'inflammation',
    'highgrade_dysplasia',
    'tumor_necrosis',
    'suspicious_for_invasion',
    'adenocarcinoma',
]

NUM_CLASSES = len(CLASS_NAMES)  # 7

# Columnas que se usan solo para filtrar — el patch se descarta si alguna es != 0
EXCLUDE_COLUMNS = ['artifact', 'resection_edge', 'lymphovascular_invasion']

# Columnas del CSV que no son etiquetas ni fname
IGNORED_COLUMNS = ['burn_out_pct', 'low_saturation_pct', 'n_masks_for_slide']

FNAME_COL = 'fname'


# ---------------------------------------------------------------------------
# Creación de tensores
# ---------------------------------------------------------------------------

def load_csv(csv_path):
    """Carga el CSV de etiquetas, eliminando columnas auxiliares."""
    df = pd.read_csv(csv_path)
    for c in IGNORED_COLUMNS:
        if c in df.columns:
            df = df.drop(columns=c)
    if FNAME_COL not in df.columns:
        raise ValueError(f"No existe columna '{FNAME_COL}' en {csv_path}")
    return df

def load_image_as_array(path, target_size=None, force_channels=3):
    """Carga una imagen como array uint8 (H, W, C)."""
    
    img = Image.open(path)
    if force_channels == 3:
        img = img.convert('RGB')
    elif force_channels == 1:
        img = img.convert('L')
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    if force_channels == 3 and arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

def build_h5(df, dataset_publico_dir, n_slide, h5_out_path,  target_size=None, force_channels=3, use_torch=True):
    
    images_written = 0
    tumor_tot = 0
    no_tumor_tot = 0
    excl_art_resection = 0
    excl_no_label = 0
    missing_images = []
    per_class = {name: 0 for name in CLASS_NAMES}

    # Estimar número máximo 
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
            shape=(max_samples, NUM_CLASSES),
            maxshape=(None,NUM_CLASSES),
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
            
            label_vector = np.zeros(NUM_CLASSES, dtype=np.uint8)
            
            for i, class_name in enumerate(CLASS_NAMES):
                raw = row.get(class_name, 0)
                try:
                    if int(float(str(raw).strip())) != 0:
                        label_vector[i] = 1
                except (ValueError, TypeError):
                    pass
        
            if label_vector.sum() == 0:
                excl_no_label += 1
                continue
            
            #if label_vector.sum() > 1:
            #    continue
            
            try:
                arr = load_image_as_array(img_path, target_size, force_channels)
            except Exception as e:
                print(f"Error cargando imagen {img_path}: {e}")
                missing_images.append((idx, str(img_path)))
                continue
            
            img_ds[images_written] = arr
            label_ds[images_written] = label_vector
            
            # Conteo de pathes por clase
            for i, name in enumerate(CLASS_NAMES):
                if label_vector[i] == 1:
                    per_class[name] += 1
                
            images_written += 1
        
        # Recortar datasets al tamaño real
        img_ds.resize((images_written, target_size[1], target_size[0], force_channels))
        label_ds.resize((images_written, NUM_CLASSES))
        
    return {
        "n_images": images_written,
        "per_class": per_class,
        "excluded_artifact": excl_art_resection,
        "excluded_no_label": excl_no_label,
        "missing_images": missing_images
    }


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    ROOT    = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')
    OUT_DIR = ROOT / 'h5_multilabel'
    OUT_DIR.mkdir(exist_ok=True)
    TARGET_SIZE = (256,256)
    FORCE_CHANNELS = 3

    # -----------------------------------------------------------------------
    # Acumuladores globales para resumen final
    # -----------------------------------------------------------------------
    total_written     = 0
    total_excl_art    = 0
    total_excl_nolbl  = 0
    total_missing     = 0
    total_per_class   = {name: 0 for name in CLASS_NAMES}
 
    # -----------------------------------------------------------------------
    # Bucle por slide
    # -----------------------------------------------------------------------
    for n_slide in range(1,201):
        n_slide_str = str(n_slide).zfill(3)
 
        DATA_DIR  = ROOT / 'Dataset_Publico' / f'zoom_2_{n_slide_str}'
        CSV_PATH  = DATA_DIR / f'{n_slide_str}_labels.csv'
        H5_PATH   = OUT_DIR / f'{n_slide_str}_multilabel.h5'
 
        if not CSV_PATH.exists():
            print(f"[WARN] CSV no encontrado: {CSV_PATH}, saltando.")
            continue
 
        df    = load_csv(CSV_PATH)
        stats = build_h5(df, DATA_DIR, n_slide_str, H5_PATH, target_size=TARGET_SIZE, force_channels=FORCE_CHANNELS)
 
        print(f"\n{'='*50}")
        print(f"  Slide {n_slide_str}")
        print(f"{'='*50}")
        print(f"  Patches escritos : {stats['n_images']}")
        print(f"  Excl. artefactos : {stats['excluded_artifact']}")
        print(f"  Excl. sin label  : {stats['excluded_no_label']}")
        print(f"  Imágenes faltantes: {len(stats['missing_images'])}")
        print(f"  Distribución de clases:")
        for name, count in stats['per_class'].items():
            pct = 100 * count / stats['n_images'] if stats['n_images'] > 0 else 0
            print(f"    [{CLASS_NAMES.index(name)}] {name:<30} {count:>7}  ({pct:.1f}%)")

        total_written    += stats['n_images']
        total_excl_art   += stats['excluded_artifact']
        total_excl_nolbl += stats['excluded_no_label']
        total_missing    += len(stats['missing_images'])
        for name in CLASS_NAMES:
            total_per_class[name] += stats['per_class'][name]
 
    # -----------------------------------------------------------------------
    # Resumen global
    # -----------------------------------------------------------------------
    print(f"\n{'#'*50}")
    print("  RESUMEN GLOBAL")
    print(f"{'#'*50}")
    print(f"  Total patches escritos  : {total_written:,}")
    print(f"  Total excl. artefactos  : {total_excl_art:,}")
    print(f"  Total excl. sin label   : {total_excl_nolbl:,}")
    print(f"  Total imágenes faltantes: {total_missing:,}")
    print(f"  Distribución global de clases:")
    for name, count in total_per_class.items():
        pct = 100 * count / total_written if total_written > 0 else 0
        print(f"    [{CLASS_NAMES.index(name)}] {name:<30} {count:>9,}  ({pct:.1f}%)")
    print(f"\n  Archivos H5 guardados en: {OUT_DIR}")

        