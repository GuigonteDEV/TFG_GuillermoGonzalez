from pathlib import Path
import torch
from Funciones.Tensor_Images import build_tensors, load_csv

Create_Tensor = True




if Create_Tensor:
    excl_art_resection_tot = 0
    excl_conflict_tot = 0
    excl_no_label_tot = 0
    for n_slide in range(1, 201):
        ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')  
        DATA_DIR = ROOT / 'Dataset_Publico' / f'zoom_2_00{n_slide}'
        CSV_PATH = DATA_DIR / f'00{n_slide}_labels.csv'
        TARGET_SIZE = (256, 256)   
        FORCE_CHANNELS = 3  
        
        df = load_csv(CSV_PATH)
        images_tensor, labels_tensor, meta_tensor, continuous_cols, missing, excl_art_resection, excl_conflict, excl_no_label = build_tensors(
            df,
            DATA_DIR,
            n_slide = n_slide,
            target_size=TARGET_SIZE,
            force_channels=FORCE_CHANNELS,
            use_torch=True
        )
        print("Tamaño imágenes tensor:", images_tensor.shape)
        print("Tamaño etiquetas tensor:", labels_tensor.shape)
        if meta_tensor is not None:
            print("Tamaño metadata continua:", meta_tensor.shape)
        if missing:
            print(f"{len(missing)} imágenes faltantes / errores (primeros 10):", missing[:10])
        print(f'Número exclusiones normales:', excl_art_resection)
        print(f'Número exclusiones conflictos:', excl_conflict)
        print(f'Número exclusiones sin etiqueta:', excl_no_label)

        out_dir = ROOT / 'processed'
        out_dir.mkdir(exist_ok=True)
        torch.save({
            'images': images_tensor,
            'labels': labels_tensor,
            'meta_continuous': meta_tensor,
            'continuous_cols': continuous_cols
        }, out_dir / '00{n_slide}_tensor.pt')
        print("Guardado en:", out_dir / '00{n_slide}_tensor.pt')
        
        excl_art_resection_tot += excl_art_resection
        excl_conflict_tot += excl_conflict
        excl_no_label_tot += excl_no_label
    
    print(f'Número total exclusiones normales', excl_art_resection_tot)
    print(f'Número total exclusiones conflictos:', excl_conflict_tot)
    print(f'Número total exclusiones sin etiqueta:', excl_no_label_tot)
    
    
#Comprobacion de la transformacion de la imagen, y su no destrucción


'''
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

img_tensor = images_tensor[1]  # (3, H, W) con valores entre 0 y 1

# Convertir a imagen PIL
img = to_pil_image(img_tensor)  # esto reescala 0–1 → 0–255 automáticamente

# Mostrar
plt.imshow(img)
plt.axis('off')
plt.show()

camino = out_dir / '001_tensor.pt'

data = torch.load(camino)
print(type(data))

imgs = data["images"]
print(imgs[0].shape)         # ejemplo
print(imgs[0].min(), imgs[0].max())
'''