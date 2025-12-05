#!/usr/bin/env python3
"""
Reconstruye una WSI desde un .pt (images en [0,1], canales = 3).
- Fija la ruta INPUT_PATH dentro del script (para usar directamente en VSCode).
- Asume que el .pt contiene:
    'images' (torch tensor o numpy) shape (N,3,H,W) o (N,H,W,3)
    'meta_continuous' (N,K) con columnas continuas
    'continuous_cols' lista con nombres (p.ej. ['topleft_y','topleft_x',...])
- Guarda la imagen resultante en la misma carpeta que el .pt con sufijo "_reconstruida.png".
- Si alguna celda de la rejilla no tiene parche, queda en blanco (color configurable).
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# ----------------------- CAMBIA AQUÍ la ruta a tu .pt -----------------------
INPUT_PATH = r"C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto\processed\001_tensor.pt"
# ---------------------------------------------------------------------------

# Color de fondo para parches faltantes (0 = negro, 255 = blanco)
FILL_COLOR = 255

def load_pt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    data = torch.load(path, map_location="cpu")
    return data

def images_to_numpy(images):
    """Convierte images a numpy (N,H,W,3) floats (presumiblemente en [0,1])."""
    arr = images.detach().cpu().numpy()
    # Normalizar dimensión
    if arr.ndim == 3:
        # (3,H,W) -> (1,H,W,3)
        arr = np.expand_dims(arr, 0)
    if arr.ndim != 4:
        raise ValueError(f"images deben ser 4D. Shape detectada: {arr.shape}")
    # Si es (N,3,H,W) -> pasar a (N,H,W,3)
    if arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 3, 1)
    elif arr.shape[-1] == 3:
        pass
    return arr  # (N,H,W,3)

def float01_to_uint8(patch):
    a = np.array(patch, dtype=np.float32)
    a = np.clip(a, 0.0, 1.0)
    a = (a * 255.0).round().astype(np.uint8)
    return a

def reconstruct(data, out_path=None, fill_color=255):
    if 'images' not in data:
        raise KeyError("El .pt no contiene la clave 'images'")
    if 'meta_continuous' not in data or 'continuous_cols' not in data:
        raise KeyError("El .pt debe contener 'meta_continuous' y 'continuous_cols' con las coordenadas topleft_x/topleft_y")
    imgs = images_to_numpy(data['images'])  # (N,H,W,3)
    meta = data['meta_continuous']
    continuous_cols = data['continuous_cols']
    # coords
    if isinstance(meta, torch.Tensor):
        coords = meta.detach().cpu().numpy()
    else:
        coords = np.array(meta)
    xs = coords[:, 0].astype(int) #topleft_x es la primera columna, indice 0
    ys = coords[:, 1].astype(int) #topleft_y es la segunda columna, indice 1
    
    N, H, W, _ = imgs.shape

    # Rejilla: usar valores únicos ordenados
    unique_x = np.unique(xs)
    unique_y = np.unique(ys)
    unique_x_sorted = np.sort(unique_x)
    unique_y_sorted = np.sort(unique_y)
    n_cols = len(unique_x_sorted)
    n_rows = len(unique_y_sorted)
    canvas_w = n_cols * W
    canvas_h = n_rows * H

    # Crear lienzo RGB
    canvas = np.full((canvas_h, canvas_w, 3), fill_color, dtype=np.uint8)

    x_to_col = {v: i for i, v in enumerate(unique_x_sorted)}
    y_to_row = {v: i for i, v in enumerate(unique_y_sorted)}

    placed = 0
    for i in range(N):
        px = xs[i]
        py = ys[i]
        col = x_to_col.get(px, None)
        row = y_to_row.get(py, None)
        if col is None:
            col = int(np.argmin(np.abs(unique_x_sorted - px)))
        if row is None:
            row = int(np.argmin(np.abs(unique_y_sorted - py)))
        top = row * H
        left = col * W
        patch = imgs[i]
        patch_u8 = float01_to_uint8(patch)
        h0 = min(H, canvas_h - top)
        w0 = min(W, canvas_w - left)
        canvas[top:top+h0, left:left+w0, :] = patch_u8[:h0, :w0, :]
        placed += 1

    # Guardar
    if out_path is None:
        base = os.path.splitext(os.path.basename(INPUT_PATH))[0]
        out_path = os.path.join(os.path.dirname(INPUT_PATH), base + "_reconstruida.png")
    Image.fromarray(canvas).save(out_path)
    return out_path, (n_cols, n_rows), placed

def main():
    try:
        print("Cargando:", INPUT_PATH)
        data = load_pt(INPUT_PATH)
        out_file, grid_shape, placed = reconstruct(data, out_path=None, fill_color=FILL_COLOR)
        print(f"Reconstrucción guardada en: {out_file}")
        print(f"Rejilla (cols, rows): {grid_shape}, patches colocados: {placed}")
    except Exception as e:
        print("Error:", str(e))
        # mostrar traceback para debugging en VSCode
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
