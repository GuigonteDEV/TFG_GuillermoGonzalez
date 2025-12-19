from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import time as time
import matplotlib.pyplot as plt
import random
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Funciones.Tensor_Images import build_tensors, load_csv
from Funciones.Build_WSI import reconstruct, load_pt
from Funciones.Augmentation_Dataloader import Dataset_Division, summarize_file_list, Transforms, LazyPatchDataset, WeightedSampler


# ---------------------------
# Configuración general universal
# ---------------------------

ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto') 

Create_Tensor = False

Create_WSI = False


################################
# ---------------------------
# Creación de Tensores WSI
# ---------------------------
################################

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
    


################################
# ---------------------------
# Reconstrucción WSI
# ---------------------------
################################

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
        
        
################################
# ---------------------------
# Creación Dataloader
# ---------------------------
################################

# ---------------------------
# Configuración general
# ---------------------------

CSV_PATH = ROOT / 'Statistics' / 'WSI_stats.csv'
PT_DIR = ROOT / 'processed' 
IMAGE_SIZE = 256
BATCH_SIZE = 32 

#Inicio cronómetro
start_time = time.time()

# ---------------------------
#Creación índices Dataset
# ---------------------------

train_idx, val_idx = Dataset_Division(CSV_PATH)

train_imgs, train_tumors, train_notumors = summarize_file_list(train_idx,CSV_PATH)
val_imgs, val_tumors, val_notumors = summarize_file_list(val_idx, CSV_PATH)


# ---------------------------
#Inicialización Transforms Augmentation
# ---------------------------

train_transforms, val_transforms = Transforms(IMAGE_SIZE)


# ---------------------------
# Creación subset
# ---------------------------

subset_train_idx = train_idx[:int(len(train_idx) * 0.1)]
subset_val_idx = val_idx[:int(len(val_idx) * 0.1)]

train_imgs_sub, train_tumors_sub, train_notumors_sub = summarize_file_list(subset_train_idx, CSV_PATH)
val_imgs_sub, val_tumors_sub, val_notumors_sub = summarize_file_list(subset_val_idx, CSV_PATH)


# ---------------------------
# Generación Dataset train / val
# ---------------------------

#Separación de files

pt_files = list(PT_DIR.glob("*.pt"))
pt_files = np.array(pt_files)

train_files = pt_files[subset_train_idx]
val_files = pt_files[subset_val_idx]


train_dataset = LazyPatchDataset(train_files, transform = train_transforms)
val_dataset = LazyPatchDataset(val_files, transform = val_transforms)


# ---------------------------
# WeightedRandomSampler para train
# ---------------------------

train_sampler = WeightedSampler(train_dataset.labels)


# ---------------------------
# DataLoaders
# ---------------------------
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, sampler = train_sampler)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)


# ---------------------------
# Ejemplo de iteración
# ---------------------------
'''
if __name__ == "__main__":
    
    print("Split inicial:")
    print(f" Train WSI: {len(train_idx)}, patches: {train_imgs}, tumors: {train_tumors}")
    print(f" Val   WSI: {len(val_idx)}, patches: {val_imgs}, tumors: {val_tumors}")
    
    print("Split subset:")
    print(f" Train WSI: {len(subset_train_idx)}, patches: {train_imgs_sub}, tumors: {train_tumors_sub}")
    print(f" Val   WSI: {len(subset_val_idx)}, patches: {val_imgs_sub}, tumors: {val_tumors_sub}")
    
    #Creacion Imagenes por comprobacion
    # coger un batch
    images, labels = next(iter(train_loader))

    # seleccionar 10 índices aleatorios del batch
    idxs = random.sample(range(images.size(0)), 10)

    # desnormalizar (para que se vean bien)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    plt.figure(figsize=(15, 6))

    for i, idx in enumerate(idxs):
        img = images[idx].cpu() * std + mean
        img = img.clamp(0, 1)

        plt.subplot(2, 5, i + 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Label: {labels[idx].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx} - images: {images.shape}, labels: {labels.shape}")
        print(f"Número de patches tumor: {(labels == 1).sum().item()}")
        if batch_idx == 1:
            break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTiempo de entrenamiento: {elapsed_time:.4f} segundos")
    
'''

################################
# ---------------------------
# Implementación Modelo + Entrenamiento
# ---------------------------
################################

epochs = 2

# ---------------------------
# Modelo ResNet18
# ---------------------------
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)  # salida escalar

    def forward(self, x):
        return self.backbone(x) 
    

def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    losses, accs = [], []

    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.float().to(device)

        optimizer.zero_grad()
        logits = model(X).squeeze(1)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean()
        accs.append(acc.item())
        
    scheduler.step(np.mean(losses))

    return np.mean(losses), np.mean(accs)


def val_loop(dataloader, model, loss_fn, device):
    model.eval()
    losses, accs = [], []

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.float().to(device)
            logits = model(X).squeeze(1)
            loss = loss_fn(logits, y)
            losses.append(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds == y).float().mean()
            accs.append(acc.item())

    return np.mean(losses), np.mean(accs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18()

loss_fn = nn.BCEWithLogitsLoss()  # logits → sigmoid implícito
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

#Creamos listas donde guardaremos los resultados
train_losses=[]
test_losses=[]
train_accuracies=[]
test_accuracies=[]

#Iniciamos un contador par cronometrar el tiempo de ejecución
start_time = time.time()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_accuracy = train_loop(train_loader, model, loss_fn, optimizer, device)
    test_loss, test_accuracy = val_loop(val_loader, model, loss_fn, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print("Avg train loss", train_loss, ", Avg test loss", test_loss, "Current learning rate", scheduler.get_last_lr())
print("Done!")

end_time = time.time()
dense_elapsed_time = end_time - start_time