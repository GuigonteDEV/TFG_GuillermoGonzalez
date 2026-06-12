from pathlib import Path
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models import MLPBinary, MLP
from src.utils import H5Dataset, load_uni_model, extract_features

# ---------------------------
# Configuración general universal
# ---------------------------

ROOT  = Path('.')
h5_path = ROOT / 'biopsies' / '148_multiclass_UNI.h5'
CKPT_DIR_UNI     = ROOT / 'assets' / 'ckpts' / 'vit_large_patch16_224.dinov2.uni_mass100k'
CKPT_DIR = ROOT / 'checkpoints'


BATCH_SIZE   = 32       
NUM_WORKERS  = 0
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKEN = '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'


SEEDS = [22, 32, 42, 52, 62]
FOLDS = [1, 2, 3, 4, 5]
N_MODELS = len(SEEDS) * len(FOLDS)

SEVERITY_CLASSES = [
    'inflammation',            # 1 — menor riesgo patológico
    'lowgrade_dysplasia',      # 2
    'highgrade_dysplasia',     # 3
    'tumor_necrosis',          # 4
    'suspicious_for_invasion', # 5
    'adenocarcinoma',          # 6 — mayor riesgo
]
MAP_IDX = [
    'Inflamación',
    'LGD',
    'HGD',
    'Necrosis',
    'Sospechoso de invasión',
    'Adenocarcinoma'
]
NUM_CLASSES_MULTI = len(SEVERITY_CLASSES)
NUM_CLASSES_B = 1

uni_normalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Inicio inferencia
# ---------------------------

dataset = H5Dataset(h5_path, transform=uni_normalize)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == 'cuda')
)

# Cargar UNI
model = load_uni_model(CKPT_DIR_UNI, DEVICE, TOKEN)

# Extraer features con UNI
features, topleft_x, topleft_y = extract_features(model, dataloader, DEVICE)

N_PATCHES = features.size(0)

# Cargar MLP binario y multiclase
model_binary = MLPBinary(input_dim=1024, num_classes=NUM_CLASSES_B).to(device)
model_multiclass = MLP(input_dim=1024, num_classes=NUM_CLASSES_MULTI).to(device)

all_binary_probs = np.zeros((N_PATCHES, N_MODELS))
all_thresholds = np.zeros(N_MODELS)

print("\nInferencia Nivel 1...")

# Inferencia binaria
model_idx = 0
for s in SEEDS:
    for f_config in FOLDS:
        bin_path = CKPT_DIR / f"best_UNI_binary_seed_{s}_fold_{f_config}.pth"
        if not bin_path.exists(): continue
            
        chk_bin = torch.load(bin_path, map_location=device)
        model_binary.load_state_dict(chk_bin['model_state_dict'])
        all_thresholds[model_idx] = chk_bin['threshold']
        model_binary.eval()
        
        probs_list = []
        with torch.no_grad():
            for i in range(0, N_PATCHES, BATCH_SIZE):
                batch_feat = features[i:i+BATCH_SIZE].to(device)
                logits = model_binary(batch_feat)
                probs = torch.sigmoid(logits).squeeze(1)
                probs_list.append(probs.cpu().numpy())
                
        all_binary_probs[:, model_idx] = np.concatenate(probs_list)
        model_idx += 1

# Media de probabilidades de los modelos del ensemble
mean_binary_probs = np.mean(all_binary_probs, axis=1)
ensemble_threshold = np.mean(all_thresholds)
is_pathological = mean_binary_probs >= ensemble_threshold

# Filtramos entre patologicas y normales para el Nivel 2
indices_normales = np.where(~is_pathological)[0]
indices_patologicos = np.where(is_pathological)[0]

y_pred_final = np.zeros(N_PATCHES, dtype=int)
y_uncertainty = np.zeros(N_PATCHES, dtype=float)

# Asignación de Tejido Normal
y_pred_final[indices_normales] = 0
y_uncertainty[indices_normales] = np.std(all_binary_probs[indices_normales, :], axis=1)

print("\nInferencia Nivel 2...")

# Procesamiento selectivo para parches patológicos
if len(indices_patologicos) > 0:
    pathological_features = features[indices_patologicos]
    N_PATHOLOGICAL = pathological_features.size(0)
    all_multiclass_probs = np.zeros((N_PATHOLOGICAL, N_MODELS, NUM_CLASSES_MULTI))
    
    model_idx = 0
    for s in SEEDS:
        for f_config in FOLDS:
            multi_path = CKPT_DIR / f"best_UNI_multiclass_seed_{s}_fold_{f_config}.pth"
            if not multi_path.exists(): continue
                
            chk_multi = torch.load(multi_path, map_location=device)
            model_multiclass.load_state_dict(chk_multi['model_state_dict'])
            model_multiclass.eval()
            
            probs_list = []
            with torch.no_grad():
                for i in range(0, N_PATHOLOGICAL, BATCH_SIZE):
                    batch_feat = pathological_features[i:i+BATCH_SIZE].to(device)
                    logits = model_multiclass(batch_feat)
                    probs = torch.softmax(logits, dim=1)
                    probs_list.append(probs.cpu().numpy())
                    
            all_multiclass_probs[:, model_idx, :] = np.concatenate(probs_list, axis=0)
            model_idx += 1
            
    mean_multi_probs = np.mean(all_multiclass_probs, axis=1)
    preds_multi = np.argmax(mean_multi_probs, axis=1)
    
    y_pred_final[indices_patologicos] = preds_multi + 1
    
    std_multi_probs = np.std(all_multiclass_probs, axis=1)
    winning_class_std = np.take_along_axis(std_multi_probs, preds_multi[:, None], axis=1).squeeze(1)
    y_uncertainty[indices_patologicos] = winning_class_std

# =====================================================================
# GENERACIÓN DE MAPAS ESPACIALES
# =====================================================================
print("\nGenerando mapas visuales...")

unique_x = np.sort(np.unique(topleft_x))
unique_y = np.sort(np.unique(topleft_y))
if len(unique_x) > 1 and len(unique_y) > 1:
    patch_size_x = np.min(np.diff(unique_x))
    patch_size_y = np.min(np.diff(unique_y))
else:
    patch_size_x = patch_size_y = 224

# Definir las dimensiones del Grid
min_x, max_x = unique_x[0], unique_x[-1]
min_y, max_y = unique_y[0], unique_y[-1]
num_cols = int(round((max_x - min_x) / patch_size_x)) + 1
num_rows = int(round((max_y - min_y) / patch_size_y)) + 1

# Inicializar matrices
grid_preds = np.full((num_rows, num_cols), np.nan)
grid_unc = np.full((num_rows, num_cols), np.nan)
PATCH_RES = 32  
grid_background = np.full((num_rows * PATCH_RES, num_cols * PATCH_RES, 3), 1.0) 

# Rellenar matrices con datos del H5
with h5py.File(h5_path, "r") as h5:
    for i in range(len(topleft_x)):
        col = int(round((topleft_x[i] - min_x) / patch_size_x))
        row = int(round((topleft_y[i] - min_y) / patch_size_y))
        
        # Solo registramos en el mapa si es una patología
        if y_pred_final[i] > 0:
            grid_preds[row, col] = y_pred_final[i]
            
        grid_unc[row, col] = y_uncertainty[i]
        
        # Reconstrucción de la textura de fondo de la biopsia
        patch = h5["images"][i]
        patch_t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()
        patch_resized = torch.nn.functional.interpolate(patch_t, size=(PATCH_RES, PATCH_RES), mode='area')
        patch_small = patch_resized.squeeze(0).permute(1, 2, 0).numpy() / 255.0
        grid_background[row*PATCH_RES : (row+1)*PATCH_RES, col*PATCH_RES : (col+1)*PATCH_RES] = patch_small

# Configuración colores personalizados
colores_patologias_rgba = [
    [0.00, 0.80, 1.00, 0.45], # 1: inflammation
    [0.15, 0.85, 0.00, 0.45], # 2: lowgrade_dysplasia
    [1.00, 0.80, 0.00, 0.50], # 3: highgrade_dysplasia
    [0.35, 0.35, 0.35, 0.65], # 4: tumor_necrosis
    [1.00, 0.40, 0.00, 0.50], # 5: suspicious_for_invasion
    [0.90, 0.00, 0.10, 0.55]  # 6: adenocarcinoma
]
cmap_diagnostico = ListedColormap(colores_patologias_rgba)

# Paleta Incertidumbre: Azul Oscuro (Certeza) a Amarillo (Duda)
colores_incertidumbre_list = [
    "#071120", "#112d55", "#1d5b90", "#3596b5", "#7ccca5", "#eed64d", "#fff200"
]
cmap_incertidumbre = LinearSegmentedColormap.from_list("blue_to_yellow", colores_incertidumbre_list)


fig, axes = plt.subplots(1, 2, figsize=(22, 9), facecolor='white')
wsi_extent = [min_x, max_x + patch_size_x, max_y + patch_size_y, min_y]

# MAPA DE DIAGNÓSTICO
# Capa de fondo: Tejido real
axes[0].imshow(grid_background, extent=wsi_extent, interpolation='bilinear', aspect='equal')
# Capa superior: Predicciones patológicas
sc1 = axes[0].imshow(grid_preds, 
                    cmap=cmap_diagnostico, 
                    vmin=1, vmax=6,
                    extent=wsi_extent, 
                    interpolation='nearest', 
                    aspect='equal')

axes[0].set_title('Mapa de Patologías sobre Biopsia\n', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Coordenada X', fontsize=14)
axes[0].set_ylabel('Coordenada Y', fontsize=14)
axes[0].grid(False)

# Configurar barra de colores
cbar1 = fig.colorbar(sc1, ax=axes[0], ticks=range(1, 7), shrink=0.75, pad=0.03)
cbar1.ax.set_yticklabels(MAP_IDX, fontsize=14)
cbar1.outline.set_visible(False)


# MAPA DE INCERTIDUMBRE
# Capa de fondo: Tejido real
axes[1].imshow(grid_background, extent=wsi_extent, interpolation='bilinear', aspect='equal')
# Capa superior: Incertidumbre
sc2 = axes[1].imshow(grid_unc, 
                    cmap=cmap_incertidumbre, 
                    extent=wsi_extent, 
                    interpolation='nearest', 
                    alpha=0.50, 
                    aspect='equal')

axes[1].set_title('Distribución de la Incertidumbre del Ensemble\n', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Coordenada X', fontsize=14)
axes[1].set_ylabel('Coordenada Y', fontsize=14)
axes[1].grid(False)

# Configurar barra de colores
cbar2 = fig.colorbar(sc2, ax=axes[1], shrink=0.75, pad=0.03)
cbar2.set_label('Incertidumbre (Desviación Estándar entre Modelos)', fontsize=14)
cbar2.outline.set_visible(False)

# Guardar y cerrar
plt.tight_layout()
output_map_path = f"map_biopsy_148_UNI.png"
plt.savefig(output_map_path, dpi=300, bbox_inches='tight')
plt.close()

