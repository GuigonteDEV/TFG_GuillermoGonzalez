from pathlib import Path
import pandas as pd
import torch
import os

# Ruta donde están todos tus .pt
ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto') 

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

for n_slide in range(1,201):
    n_slide_str = str(n_slide).zfill(3)
    TENSOR_PATH = ROOT / 'processed' / f'{n_slide_str}_tensor.pt'
    data = load_pt(TENSOR_PATH)
    labels = data["labels"]
    if any(labels == 1):
        print(f'{n_slide_str}_tensor.pt')
        tumor_per_WSI = (labels == 1).sum().item()
        print(f'Número de patches tumor: {tumor_per_WSI}')
        n_slide_tumor += 1
        if any(labels == 0):
            print('#############')
            n_slide_both += 1
        
        
print(f'Número de WSI tumor: {n_slide_tumor}')
print(f'Número de WSI ambos: {n_slide_both}')