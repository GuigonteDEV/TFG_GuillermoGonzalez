import pandas as pd
import numpy as np
from pathlib import Path 
import torch
from sklearn.metrics import f1_score

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')

SEEDS = [22, 32, 42, 52, 62]
NUM_FOLDS = 5
ALGO_STRATEGY = 'None'
DATA_STRATEGY = 'WS'

thresholds = np.linspace(0.01, 0.99, 99)

for fold in range(1, NUM_FOLDS + 1):
    for seed in SEEDS:
        
        csv_path = ROOT  / 'Model_output' / f'predictions_seed_{seed}_fold{fold}_{DATA_STRATEGY}_{ALGO_STRATEGY}_val.csv'
        CKPT_DIR = ROOT / 'Model_output' / f'best_model_seed_{seed}_fold_{fold}_strategy_{DATA_STRATEGY}_{ALGO_STRATEGY}.pth'

        df = pd.read_csv(csv_path)
        classes = [col.replace('_true', '') for col in df.columns if col.endswith('_true')]

        best_thres_list = []

        for cls in classes:
            y_true = df[f'{cls}_true'].values
            y_prob = df[f'{cls}_prob'].values
            
            best_thres = 0.5
            mejor_f1 = 0.0
            
            for thresh in thresholds:
                y_pred_temporal = (y_prob >= thresh).astype(int)
                score_f1 = f1_score(y_true, y_pred_temporal, zero_division=0)
                
                if score_f1 > mejor_f1:
                    mejor_f1 = score_f1
                    best_thres = thresh
            
            best_thres_list.append(round(float(best_thres), 2))
            
        try:
            checkpoint = torch.load(CKPT_DIR, map_location='cpu')
            
            checkpoint['thresholds'] = best_thres_list
            checkpoint['classes_order'] = classes 
            
            torch.save(checkpoint, CKPT_DIR)
            print(f"¡Archivo de PyTorch actualizado con éxito en: {CKPT_DIR}!")
            print(f"Umbrales inyectados: {best_thres_list}")
        except FileNotFoundError:
            print(f"No se encontró '{CKPT_DIR}'.")
    
    

