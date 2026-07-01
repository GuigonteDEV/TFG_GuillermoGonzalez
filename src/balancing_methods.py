import torch
import torch.nn as nn
import numpy as np
import time as time
import h5py
from torch.utils.data import WeightedRandomSampler

CLASS_NAMES = [
    'normal',
    'lowgrade_dysplasia',
    'inflammation',
    'highgrade_dysplasia',
    'tumor_necrosis',
    'suspicious_for_invasion',
    'adenocarcinoma',
]

NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------
# Positive Weight
# ---------------------------

def compute_pos_weight_efective(h5_files, beta=0.9999) -> torch.Tensor:

    n_pos = np.zeros(NUM_CLASSES, dtype=np.float64)
    n_neg = np.zeros(NUM_CLASSES, dtype=np.float64)
 
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as h5:
            labels = h5["labels"][:]
            
            n_pos += labels.sum(axis=0)
            n_neg += (1 - labels).sum(axis=0)
 
    pos_weight = np.zeros(NUM_CLASSES, dtype=np.float64)
    
    for i in range(NUM_CLASSES):
        if n_pos[i] > 0:
            e_neg = 1.0 - (beta ** n_neg[i])
            e_pos = 1.0 - (beta ** n_pos[i])
            pos_weight[i] = e_neg / e_pos
        else:
            pos_weight[i] = 1.0
 
    print(f"\n  pos_weight por clase (Número Efectivo con beta={beta}):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    [{i}] {name:<30}  pos={int(n_pos[i]):>8,}  neg={int(n_neg[i]):>8,}  w={pos_weight[i]:.2f}")
 
    return torch.tensor(pos_weight, dtype=torch.float32)

# ---------------------------
# Asymmetric Loss
# ---------------------------

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, cost_matrix, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.cost_matrix = cost_matrix
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        # loss calculation
        if self.cost_matrix is not None:
            costs = self.cost_matrix[target.long()]

            # Multiplicamos la matriz target (suavizada) por log_preds y luego por los costes
            loss = - self.targets_classes.mul(log_preds) * costs

        else:
            loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

# ---------------------------
# Weighted Sampler
# ---------------------------

def compute_sample_weights(h5_files, beta=0.9999):

    n_pos = np.zeros(NUM_CLASSES, dtype=np.float64)
    all_labels = []

    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as h5:
            labels = h5["labels"][:]
            n_pos += labels.sum(axis=0)
            all_labels.append(labels)

    all_labels = np.concatenate(all_labels, axis=0)  # (N, C)

    e_pos = 1.0 - (beta ** n_pos)   # shape: (C,)

    class_weights = np.where(n_pos > 0, 1.0 / e_pos, 0.0)  # shape: (C,)

    class_weights /= class_weights.max()

    sample_weights = np.zeros(len(all_labels), dtype=np.float64)

    for idx, label_vec in enumerate(all_labels):
        pos_indices = np.where(label_vec == 1)[0]
        if len(pos_indices) > 0:
            sample_weights[idx] = class_weights[pos_indices].max()
        else:
            # Muestra sin ningún positivo: peso mínimo
            sample_weights[idx] = class_weights[class_weights > 0].min()

    print(f"\n  =========================================================================")
    print(f"  PESOS DEL SAMPLER MULTILABEL (Número Efectivo | beta={beta})")
    print(f"  =========================================================================")
    print(f"  Total de parches cargados en el Sampler: {len(sample_weights):,}\n")
    print(f"  Importancia relativa asignada a la presencia de cada clase (Escala 0-1):")
    print(f"  {'-'*73}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    [{i}] {name:<28} -> Conteo Pos: {int(n_pos[i]):>8,} | Peso Relativo: {class_weights[i]:.6f}")
    print(f"  =========================================================================\n")

    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
    
    train_sampler = WeightedRandomSampler(weights = sample_weights, num_samples = len(sample_weights), replacement = True)

    return train_sampler