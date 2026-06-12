import pandas as pd
import numpy as np
from pathlib import Path 
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')

SEEDS = [22, 32, 42, 52, 62]
NUM_FOLDS = 5

thresholds = np.linspace(0.01, 0.99, 99)

for fold in range(1, NUM_FOLDS + 1):
    for seed in SEEDS:
        
        csv_path = ROOT  / 'UNI_output_binary' / f'predictions_UNI_binary_seed_{seed}_fold_{fold}_val.csv'
        CKPT_DIR = ROOT / 'UNI_output_binary' / f'best_UNI_binary_seed_{seed}_fold_{fold}.pth'

        df = pd.read_csv(csv_path)

        y_true = df[f'y_true'].values
        y_prob = df[f'y_prob'].values
        
        best_thres = 0.5
        mejor_f1 = 0.0
        
        for thresh in thresholds:
            y_pred_temporal = (y_prob >= thresh).astype(int)
            score_f1 = f1_score(y_true, y_pred_temporal, zero_division=0)
            
            if score_f1 > mejor_f1:
                mejor_f1 = score_f1
                best_thres = thresh
                p_opt = precision_score(y_true, y_pred_temporal, zero_division=0)
                r_opt = recall_score(y_true, y_pred_temporal, zero_division=0)
        
        best_thres = round(float(best_thres), 2)
        
        try:
            checkpoint = torch.load(CKPT_DIR, map_location='cpu')
            
            checkpoint['threshold'] = best_thres
            
            torch.save(checkpoint, CKPT_DIR)
            print(f"¡Archivo de PyTorch actualizado con éxito en: {CKPT_DIR}!")
            print(f"Umbral inyectado: {best_thres} | F1-Val máximo: {mejor_f1:.4f} | Precision: {p_opt:.4f} | Recall: {r_opt:.4f}\n")
        except FileNotFoundError:
            print(f"No se encontró '{CKPT_DIR}'.")
    
    

