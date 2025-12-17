from pathlib import Path
import pandas as pd
import torch
import os
import csv

# Ruta donde están todos tus .pt
ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto') 

out_path = Path(r"C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto\Statistics\WSI_stats.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
csv_output = str(out_path)

def load_pt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    data = torch.load(path, map_location="cpu")
    return data

'''
total_labels = 0
total_tumor = 0
total_no_tumor = 0



for n_slide in range(1,201):
    n_slide_str = str(n_slide).zfill(3)
    TENSOR_PATH = ROOT / 'processed' / f'{n_slide_str}_tensor.pt'
    data = load_pt(TENSOR_PATH)
    
    labels = data["labels"]

    total_labels += len(labels)
    total_tumor += (labels == 1).sum().item()
    total_no_tumor += (labels == 0).sum().item()

print("===== ESTADÍSTICAS GLOBALES =====")
print(f"Total patches: {total_labels}")
print(f"Tumor (1): {total_tumor} ({total_tumor/total_labels*100:.2f}%)")
print(f"No tumor (0): {total_no_tumor} ({total_no_tumor/total_labels*100:.2f}%)")'''

n_slide_tumor = 0
n_slide_both = 0

rows = []

for n_slide in range(1,201):
    HAS_TUMOR = False
    n_slide_str = str(n_slide).zfill(3)
    TENSOR_PATH = ROOT / 'processed' / f'{n_slide_str}_tensor.pt'
    data = load_pt(TENSOR_PATH)
    labels = data["labels"]
    
    total_patches = len(labels)
    
    no_tumor_WSI = (labels == 0).sum().item()
    tumor_per_WSI = (labels == 1).sum().item()
    
    if any(labels == 1):
        HAS_TUMOR = True
        print(f'{n_slide_str}_tensor.pt')
        print(f'Número de patches tumor: {tumor_per_WSI}')
        n_slide_tumor += 1
        if any(labels == 0):
            print('#############')
            n_slide_both += 1
            
    rows.append([f'{n_slide_str}_tensor.pt', total_patches, tumor_per_WSI, no_tumor_WSI, HAS_TUMOR])
        
        
print(f'Número de WSI tumor: {n_slide_tumor}')
print(f'Número de WSI ambos: {n_slide_both}')

with open(csv_output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["wsi_path", "total_patches", "tumor_patches", "no_tumor_patches", "has_tumor"])
    writer.writerows(rows)

print("CSV creado en:", csv_output)