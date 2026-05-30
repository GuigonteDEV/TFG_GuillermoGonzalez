import pandas as pd
import numpy as np
from pathlib import Path 
import json
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')

SEEDS = [22, 32, 42, 52, 62]
NUM_FOLDS = 5
ALGO_STRATEGY = 'None'
DATA_STRATEGY = 'WS'

JSON_DIR = ROOT / 'Metrics_multilabel'
JSON_DIR.mkdir(exist_ok=True)

metrics_accumulator = {}

for fold in range(1, NUM_FOLDS + 1):
    for seed in SEEDS:
        
        csv_path = ROOT  / 'Model_output' / f'predictions_seed_{seed}_fold{fold}_{DATA_STRATEGY}_{ALGO_STRATEGY}_test.csv'
        CKPT_DIR = ROOT / 'Model_output' / f'best_model_seed_{seed}_fold_{fold}_strategy_{DATA_STRATEGY}_{ALGO_STRATEGY}.pth'         

        df = pd.read_csv(csv_path)
        
        checkpoint = torch.load(CKPT_DIR, map_location='cpu')

        classes = checkpoint.get('classes_order', [col.replace('_true', '') for col in df.columns if col.endswith('_true')])

        final_report = {}

        for i, cls in enumerate(classes):
            y_true = df[f'{cls}_true'].values
            y_prob = df[f'{cls}_prob'].values
            
                
            roc_auc = float(roc_auc_score(y_true, y_prob))
            
            precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recalls, precisions))
            
            y_pred_05 = (y_prob >= 0.5).astype(int)
            p_05 = precision_score(y_true, y_pred_05, zero_division=0)
            r_05 = recall_score(y_true, y_pred_05, zero_division=0)
            f1_05 = f1_score(y_true, y_pred_05, zero_division=0)
            
            best_thres = checkpoint['thresholds'][i]
                    
            y_pred_opt = (y_prob >= best_thres).astype(int)
            p_opt = precision_score(y_true, y_pred_opt, zero_division=0)
            r_opt = recall_score(y_true, y_pred_opt, zero_division=0)
            f1_opt = f1_score(y_true, y_pred_opt, zero_division=0)
            
            
            metrics_dict = {
                "independent": {
                    "PR-AUC": pr_auc,
                    "ROC-AUC": roc_auc
                },
                "default_0.5": {
                    "precision": p_05,
                    "recall": r_05,
                    "f1_score": f1_05
                },
                "optimized": {
                    "best_threshold": best_thres,
                    "precision": p_opt,
                    "recall": r_opt,
                    "f1_score": f1_opt
                }
            }
            
            # Redondeamos para el JSON individual por comodidad visual
            final_report[cls] = {
                grupo: {metrica: round(float(valor), 4) for metrica, valor in subdict.items()}
                for grupo, subdict in metrics_dict.items()
            }
            
            # --- ACUMULADOR PARA EL REPORT GLOBAL ---
            if cls not in metrics_accumulator:
                metrics_accumulator[cls] = {grupo: {metrica: [] for metrica in metrics_dict[grupo]} for grupo in metrics_dict}
            
            for grupo in metrics_dict:
                for metrica, valor in metrics_dict[grupo].items():
                    metrics_accumulator[cls][grupo][metrica].append(valor)
            
        with open(JSON_DIR / f'report_seed_{seed}_fold_{fold}_strategy_{DATA_STRATEGY}_{ALGO_STRATEGY}.json', 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=4, ensure_ascii=False)
        print(f"Reporte guardado con éxito en: {JSON_DIR}")
        
# ==========================================
# GENERACIÓN DEL REPORTE GLOBAL (MEDIA ± DESVIACIÓN)
# ==========================================
global_report = {}

for cls, grupos in metrics_accumulator.items():
    global_report[cls] = {}
    for grupo, metricas in grupos.items():
        global_report[cls][grupo] = {}
        for metrica, lista_valores in metricas.items():
            
            # Calculamos la media y la desviación estándar de los 25 valores
            mean_val = np.mean(lista_valores)
            std_val = np.std(lista_valores)
            
            # Guardamos ambos datos estructurados
            global_report[cls][grupo][metrica] = {
                "mean": round(float(mean_val), 4),
                "std": round(float(std_val), 4)
            }

# Guardar el JSON global definitivo
json_global_path = JSON_DIR / f'global_report_strategy_{DATA_STRATEGY}_{ALGO_STRATEGY}.json'
with open(json_global_path, 'w', encoding='utf-8') as f:
    json.dump(global_report, f, indent=4, ensure_ascii=False)

print(f"Reporte GLOBAL (con Medias y STD) guardado con éxito en: {json_global_path}")