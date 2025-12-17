from pathlib import Path
import torch
from Funciones.Tensor_Images import build_tensors, load_csv
from Funciones.Build_WSI import reconstruct, load_pt


ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto') 

Create_Tensor = False

Create_WSI = False



if Create_Tensor:
    excl_art_resection_tot = 0
    excl_conflict_tot = 0
    excl_no_label_tot = 0
    tumor_tot = 0
    no_tumor_tot = 0
    patch_tot = 0
    for n_slide in range(1, 201):
        n_slide_str = str(n_slide).zfill(3) 
        DATA_DIR = ROOT / 'Dataset_Publico' / f'zoom_2_{n_slide_str}'
        CSV_PATH = DATA_DIR / f'{n_slide_str}_labels.csv'
        TARGET_SIZE = (256, 256)   
        FORCE_CHANNELS = 3  
        
        df = load_csv(CSV_PATH)
        images_tensor, labels_tensor, meta_tensor, continuous_cols, missing, excl_art_resection, excl_conflict, excl_no_label, patch_tumor, patch_no_tumor = build_tensors(
            df,
            DATA_DIR,
            n_slide = n_slide_str,
            target_size=TARGET_SIZE,
            force_channels=FORCE_CHANNELS,
            use_torch=True
        )
        print("Tamaño imágenes tensor:", images_tensor.shape)
        if missing:
            print(f"{len(missing)} imágenes faltantes / errores (primeros 10):", missing[:10])
        print(f'Número exclusiones normales:', excl_art_resection)
        print(f'Número exclusiones conflictos:', excl_conflict)
        print(f'Número patches:', len(labels_tensor))
        print(f'Número patches tumor:', patch_tumor)
        print(f'Número patches no tumor:', patch_no_tumor)
        
        out_dir = ROOT / 'processed'
        out_dir.mkdir(exist_ok=True)
        torch.save({
            'images': images_tensor,
            'labels': labels_tensor,
            'meta_continuous': meta_tensor,
            'continuous_cols': continuous_cols
        }, out_dir / f'{n_slide_str}_tensor.pt')
        print("Guardado en:", out_dir / f'{n_slide_str}_tensor.pt')
        
        excl_art_resection_tot += excl_art_resection
        excl_conflict_tot += excl_conflict
        excl_no_label_tot += excl_no_label
        patch_tot += len(labels_tensor)
        tumor_tot += patch_tumor
        no_tumor_tot += patch_no_tumor
        
    
    print(f'Número total exclusiones normales', excl_art_resection_tot)
    print(f'Número total exclusiones conflictos:', excl_conflict_tot)
    print(f'Número total exclusiones sin etiqueta:', excl_no_label_tot)
    print(f'Número total patches', patch_tot)
    print(f'Número total patches tumor:', tumor_tot)
    print(f'Número total patches no tumor', no_tumor_tot)
    

if Create_WSI:
    for n_slide in range(1,201):
        n_slide_str = str(n_slide).zfill(3)    
        INPUT_PATH = ROOT / 'processed' / f'{n_slide_str}_tensor.pt'
        FILL_COLOR = 255

        print("Cargando:", INPUT_PATH)
        data = load_pt(INPUT_PATH)
        WSI_Image, WSI_Map, grid_shape, placed = reconstruct(data)

        out_dir = ROOT / 'WSI_Images'
        out_dir.mkdir(exist_ok=True)

        WSI_Image.save(out_dir / f'{n_slide_str}_WSI.png')
        WSI_Map.save(out_dir / f'{n_slide_str}_WSI_Map.png')

        print(f"Reconstrucción guardada en: {out_dir / f'{n_slide_str}_WSI.png'}")
        print(f"Rejilla (cols, rows): {grid_shape}, patches colocados: {placed}")

