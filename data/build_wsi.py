import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image


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

def reconstruct(data):
    if 'images' not in data:
        raise KeyError("El .pt no contiene la clave 'images'")
    if 'meta_continuous' not in data:
        raise KeyError("El .pt debe contener 'meta_continuous' con las coordenadas topleft_x/topleft_y")
    imgs = images_to_numpy(data['images'])  # (N,H,W,3)
    meta = data['meta_continuous']
    label = data['labels']
    
    # coords
    coords = meta.detach().cpu().numpy()
    
    xs = coords[:, 0].astype(int) #topleft_x es la primera columna, indice 0
    ys = coords[:, 1].astype(int) #topleft_y es la segunda columna, indice 1
    
    N, H, W, _ = imgs.shape
    
    # Rejilla: usar valores únicos ordenados
    unique_x_sorted = np.sort(np.unique(xs))
    unique_y_sorted = np.sort(np.unique(ys))
    n_cols = len(unique_x_sorted)
    n_rows = len(unique_y_sorted)
    canvas_w = n_cols * W
    canvas_h = n_rows * H

    canvas = Image.new('RGB', (canvas_w, canvas_h), color= (255,255,255)) #Fill que sea Blanco
    canvas_map = Image.new('RGB', (canvas_w, canvas_h), color= (255,255,255))
    x_to_col = {v: i for i, v in enumerate(unique_x_sorted)}
    y_to_row = {v: i for i, v in enumerate(unique_y_sorted)}
    
    pixels = canvas_map.load()
    placed = 0
    for i in range(N):
        px, py = xs[i], ys[i]
        col = x_to_col[px]
        row = y_to_row[py]
        
        left = col * W
        top = row * H
        
        patch_img = to_pil_image(imgs[i])
        canvas.paste(patch_img, (left, top))
        
        color = (255, 0, 0) if label[i].item() == 1.0 else (0, 255, 0)
        # Dibujar un rectángulo con el color, rellenando todo el patch
        ImageDraw.Draw(canvas_map).rectangle([left, top, left + W - 1, top + H - 1], fill=color, outline=None)
        
        placed += 1
    
    canvas = canvas.convert("RGBA")
    canvas_map = canvas_map.convert("RGBA")
    
    alpha = 100  # 0-255
    canvas_map.putalpha(alpha)
    
    canvas_combined = Image.alpha_composite(canvas, canvas_map)
    
    return canvas, canvas_combined, (n_cols, n_rows), placed




