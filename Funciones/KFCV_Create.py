from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import os
import glob
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import re


CLASS_NAMES = [
    'normal',
    'lowgrade_dysplasia',
    'inflammation',
    'highgrade_dysplasia',
    'tumor_necrosis',
    'suspicious_for_invasion',
    'adenocarcinoma',
]

# =========================
# CREACIÓN FOLDS
# =========================

def folds_creation(H5_FILES, NUM_FOLDS, seed):
    
    wsi_data = []

    for h5_path in H5_FILES:
        with h5py.File(h5_path, "r") as f:
            labels = f["labels"][:]
            
            # Matriz binaria de presencia
            
            #presence_classes = np.max(labels, axis=0) 
            
            # Patches por clase
            per_class_counts = labels.sum(axis=0).astype(int)
            
            patch_count = len(labels)
            
            presence_classes = (per_class_counts >= 10).astype(int)
            wsi_big = int(patch_count > 1000)
            presence_amp = np.append(presence_classes, wsi_big)
            
            # Guardamos los datos de presencia y desglose de patches
            wsi_entry = {
                "archivo": os.path.basename(h5_path),
                "total patches": patch_count,
                "clases_presencia": presence_amp
            }
            
            # Añadimos de forma dinámica una columna por cada clase con su número de patches
            for idx_class, name_class in enumerate(CLASS_NAMES):
                wsi_entry[name_class] = per_class_counts[idx_class]
                
            wsi_data.append(wsi_entry)

    # Convertir a matrices limpias para el algoritmo de estratificación
    X = np.array([d["archivo"] for d in wsi_data])
    Y = np.array([d["clases_presencia"] for d in wsi_data])  

    # Aplicar la estratificación multilabel WSI wise
    mskf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    map_folds = {}

    for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(X, Y)):
        for idx in val_idx:
            map_folds[X[idx]] = fold_idx
            

    folds_files = [[] for _ in range(NUM_FOLDS)]
    
    # Extraemos el índice en lugar del nombre del archivo
    for file, fold_idx in map_folds.items():
        
        match = re.search(r'\d+', file)
        
        if match:
            numero_wsi = int(match.group())
            idx_original = numero_wsi - 1
            folds_files[fold_idx].append(idx_original)
        else:
            raise ValueError(f"No se pudo encontrar un número identificador en el archivo: {file}")

    # Convertimos a arrays de NumPy para que la función de división funcione idéntica
    folds_files = [np.array(f) for f in folds_files]
            
    return folds_files, wsi_data

def folds_statistics(H5_FILES, folds_files, NUM_FOLDS):
    # Estadisticas de patches por clase por fold
    statistics_folds = np.zeros((5, 7), dtype=int)

    # Numero patches por fold
    patches_per_fold = np.zeros(5, dtype=int)

    for fold in range(NUM_FOLDS):
        for idx in folds_files[fold]:
            with h5py.File(H5_FILES[idx], "r") as f:
                labels = f["labels"][:]
                
                classes_per_wsi = np.sum(labels, axis=0)
        
                # Acumular en Fold correspondiente
                statistics_folds[fold] += classes_per_wsi.astype(int)
                patches_per_fold[fold] += len(labels)
                
    df_resultados = pd.DataFrame(statistics_folds, columns=CLASS_NAMES)
    df_resultados.insert(0, "Total Patches", patches_per_fold)
    df_resultados.index.name = "Fold ID"


    print(df_resultados.to_string())


def get_dataset_split(FOLD_CONFIG, folds_list, NUM_FOLDS):
    
    if FOLD_CONFIG < 1 or FOLD_CONFIG > 5:
        raise ValueError("El parámetro del fold debe estar entre 1 y 5.")
        
    # Convertimos el parámetro (1-5) a índice de Python (0-4)
    val_fold_idx = FOLD_CONFIG - 1
    
    # Asignamos Test al siguiente fold de forma circular para evaluar siempre en datos "ciegos"
    test_fold_idx = (val_fold_idx + 1) % NUM_FOLDS
    
    # Los 3 folds restantes van para entrenamiento
    train_folds_indices = [i for i in range(NUM_FOLDS) if i != val_fold_idx and i != test_fold_idx]
    
    # Construcción de los sets de datos
    val_files = folds_list[val_fold_idx]
    test_files = folds_list[test_fold_idx]
    train_files = np.concatenate([folds_list[i] for i in train_folds_indices])
    
    return train_files, val_files, test_files

