import os
import glob
from pathlib import Path
import re
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')

# 1. Configuración de rutas
input_dir = ROOT / 'UNI_output_hierarchical' 
output_dir = ROOT / 'Metrics_hierarchical_UNI'
os.makedirs(output_dir, exist_ok=True)

# Buscar todos los archivos de predicciones
pattern = os.path.join(input_dir, "predictions_hierarchical_seed_*_fold_*_test.csv")
csv_files = glob.glob(pattern)

all_flat_results = []
individual_reports = []

print(f"Se encontraron {len(csv_files)} archivos para procesar.\n")

# 2. Iterar por cada archivo para extraer métricas detalladas
for file_path in csv_files:
    file_name = os.path.basename(file_path)
    
    # Extraer seed y fold
    match = re.search(r"seed_(\d+)_fold_(\d+)", file_name)
    if not match:
        continue
    
    seed = int(match.group(1))
    fold = int(match.group(2))
    
    # Leer CSV
    df = pd.read_csv(file_path)
    
    # Generar el reporte de clasificación completo como diccionario
    # (Ignoramos el warning si alguna clase no tiene predicciones en un fold específico)
    report_dict = classification_report(
        df['true_class'], 
        df['pred_class'], 
        output_dict=True,
        zero_division=0
    )
    
    # Guardar archivo JSON individual con toda la estructura por clase
    individual_metrics = {
        "file": file_name,
        "seed": seed,
        "fold": fold,
        "metrics": report_dict
    }
    
    json_filename = f"metrics_hierarchical_UNI_seed_{seed}_fold_{fold}.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(individual_metrics, f, indent=4, ensure_ascii=False)
    
    # Aplanar el diccionario para poder meterlo en un DataFrame de Pandas fácilmente
    flat_entry = {"seed": seed, "fold": fold}
    for key, value in report_dict.items():
        if isinstance(value, dict):
            for metric, val in value.items():
                if metric != 'support':  # El soporte (nº de muestras) no se promedia con std
                    flat_entry[f"{key}__{metric}"] = val
        else:
            # Para el "accuracy" que viene como un float directo en el primer nivel
            flat_entry[key] = value
            
    all_flat_results.append(flat_entry)

# 3. Crear el Reporte Global (Cálculo de Media y STD)
if all_flat_results:
    df_results = pd.DataFrame(all_flat_results)
    
    # Calcular media y std de todas las columnas de métricas
    means = df_results.mean()
    stds = df_results.std()
    
    # Reconstruir la estructura jerárquica para el JSON global
    global_metrics = {}
    
    for col in df_results.columns:
        if col in ['seed', 'fold']:
            continue
            
        if '__' in col:
            clase, metric_name = col.split('__')
            if clase not in global_metrics:
                global_metrics[clase] = {}
            
            global_metrics[clase][metric_name] = {
                "mean": float(means[col]),
                "std": float(stds[col]) if not pd.isna(stds[col]) else 0.0
            }
        else:
            # Caso del accuracy
            global_metrics[col] = {
                "mean": float(means[col]),
                "std": float(stds[col]) if not pd.isna(stds[col]) else 0.0
            }
            
    global_report = {
        "total_files_processed": len(all_flat_results),
        "global_metrics": global_metrics
    }
    
    # Guardar reporte global en JSON
    global_report_path = os.path.join(output_dir, "global_report_hierarchical.json")
    with open(global_report_path, 'w', encoding='utf-8') as f:
        json.dump(global_report, f, indent=4, ensure_ascii=False)
        
    # 4. Mostrar un resumen bonito en consola
    print("¡Proceso completado con éxito!")
    print(f"Resultados guardados en la carpeta: {output_dir}\n")
    print("====================== REPORTE GLOBAL (RESUMEN) ======================")
    
    # Imprimir primero las clases individuales
    for clase, metrics in global_metrics.items():
        if clase in ['macro avg', 'weighted avg', 'accuracy']:
            continue
        print(f"Clase: {clase}")
        print(f"  - Precision : {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
        print(f"  - Recall    : {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")
        print(f"  - F1-Score  : {metrics['f1-score']['mean']:.4f} ± {metrics['f1-score']['std']:.4f}")
    
    print("-" * 70)
    # Imprimir el Macro global
    macro = global_metrics['macro avg']
    print(f"MACRO AVERAGE (Global):")
    print(f"  - Precision : {macro['precision']['mean']:.4f} ± {macro['precision']['std']:.4f}")
    print(f"  - Recall    : {macro['recall']['mean']:.4f} ± {macro['recall']['std']:.4f}")
    print(f"  - F1-Score  : {macro['f1-score']['mean']:.4f} ± {macro['f1-score']['std']:.4f}")
    print("======================================================================")
else:
    print("No se encontraron archivos válidos.")