"""
h5_levels.py
=========================
Genera un único conjunto unificado de archivos HDF5 para un pipeline jerárquico.
Reduce a la mitad el uso de almacenamiento y el tiempo de procesamiento.

Formato de salida por archivo .h5 unificado
----------------------------------------------------
  images     : uint8  (N, H, W, 3)
  labels     : uint8  (N,)            — índice en CLASS_NAMES (0-6)
  topleft_x  : int32  (N,)
  topleft_y  : int32  (N,)
  Atributos  : class_names = CLASS_NAMES

Exclusiones (se descartan por completo):
  artifact, resection_edge, lymphovascular_invasion → patch descartado
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import h5py


# ---------------------------------------------------------------------------
# DEFINICIÓN DE CLASES UNIFICADAS
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    'normal',                  # 0 — Sano / Ausencia de patología
    'inflammation',            # 1 — menor riesgo patológico
    'lowgrade_dysplasia',      # 2
    'highgrade_dysplasia',     # 3
    'tumor_necrosis',          # 4
    'suspicious_for_invasion', # 5
    'adenocarcinoma',          # 6 — mayor riesgo
]

# Mapa clase → índice unificado (búsqueda O(1))
UNIFIED_INDEX: dict[str, int] = {cls: i for i, cls in enumerate(CLASS_NAMES)}

# Columnas que provocan el descarte TOTAL del patch si son != 0
EXCLUDE_COLUMNS = ['artifact', 'resection_edge', 'lymphovascular_invasion']

# Columnas auxiliares a ignorar
IGNORED_COLUMNS = ['burn_out_pct', 'low_saturation_pct', 'n_masks_for_slide']

FNAME_COL      = 'fname'
TOPLEFT_X_COL  = 'topleft_x'
TOPLEFT_Y_COL  = 'topleft_y'


# ---------------------------------------------------------------------------
# UTILIDADES DE DATOS
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV de etiquetas, eliminando columnas puramente auxiliares."""
    df = pd.read_csv(csv_path)
    for col in IGNORED_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)
    if FNAME_COL not in df.columns:
        raise ValueError(f"No existe columna '{FNAME_COL}' en {csv_path}")
    return df


def load_image_as_array(
    path: Path,
    target_size: tuple[int, int] | None = None,
    force_channels: int = 3,
) -> np.ndarray:
    """Carga una imagen como array uint8 (H, W, C)."""
    img = Image.open(path)
    img = img.convert('RGB') if force_channels == 3 else img.convert('L')
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)
    arr = np.array(img, dtype=np.uint8)
    if force_channels == 3 and arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def _safe_int(value, default: int = 0) -> int:
    """Convierte un valor a int de forma segura; devuelve default si falla."""
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# RESOLUCIÓN UNIFICADA
# ---------------------------------------------------------------------------

def resolve_unified_label(row: pd.Series) -> int | None:
    """
    Devuelve el índice unificado (0-6) según la jerarquía de gravedad.
    Prioriza cualquier patología activa sobre la clase normal.
    
    Returns
    -------
    int  → índice en CLASS_NAMES (0-6)
    None → sin ninguna clase activa (patch a descartar)
    """
    
    for cls in reversed(CLASS_NAMES):
        if _safe_int(row.get(cls, 0)) != 0:
            return UNIFIED_INDEX[cls]
        
    return None  # Sin ninguna etiqueta activa -> descartar


# ---------------------------------------------------------------------------
# CONSTRUCCIÓN DE ARCHIVOS HDF5
# ---------------------------------------------------------------------------

def _create_datasets(
    h5: h5py.File,
    max_samples: int,
    h: int,
    w: int,
    c: int,
    label_shape: tuple,
    label_dtype: str,
) -> tuple:
    """Crea los datasets vacíos (redimensionables) dentro de un archivo HDF5."""
    img_ds = h5.create_dataset(
        "images",
        shape=(max_samples, h, w, c),
        maxshape=(None, h, w, c),
        dtype="uint8",
        compression="gzip",
        compression_opts=4,
        chunks=(1, h, w, c),
    )
    lbl_ds = h5.create_dataset(
        "labels",
        shape=(max_samples,) + label_shape,
        maxshape=(None,) + label_shape,
        dtype=label_dtype,
    )
    tx_ds = h5.create_dataset(
        "topleft_x",
        shape=(max_samples,),
        maxshape=(None,),
        dtype="int32",
    )
    ty_ds = h5.create_dataset(
        "topleft_y",
        shape=(max_samples,),
        maxshape=(None,),
        dtype="int32",
    )
    return img_ds, lbl_ds, tx_ds, ty_ds


def build_h5_unified(
    df: pd.DataFrame,
    dataset_dir: Path,
    n_slide: str,
    h5_out_path: Path,
    target_size: tuple[int, int] = (256, 256),
    force_channels: int = 3,
) -> dict:
    """
    Construye un único HDF5 que contiene todas las muestras procesadas de un slide
    siguiendo la jerarquía unificada de 7 clases (0=normal, 1-6=patologías).
    """
    W, H = target_size
    C = force_channels

    written = 0
    excl_excluded  = 0   # descartados por artifact / resection_edge / lvi
    excl_no_label  = 0   # descartados por ausencia de etiquetas activas
    missing        = []
    per_class      = {cls: 0 for cls in CLASS_NAMES}

    with h5py.File(h5_out_path, "w") as h5:
        img_ds, lbl_ds, tx_ds, ty_ds = _create_datasets(
            h5, len(df), H, W, C, label_shape=(), label_dtype="uint8"
        )
        h5["labels"].attrs["class_names"] = CLASS_NAMES

        for _, row in df.iterrows():

            # ── Construir ruta de imagen ──────────────────────────────────
            fname = str(row[FNAME_COL]).strip()
            parts = fname.split('/')
            tail  = os.path.join(*parts[1:]) if len(parts) >= 2 else fname
            img_path = dataset_dir.parent / f'zoom_2_{n_slide}' / tail

            if not img_path.exists():
                missing.append(str(img_path))
                continue

            # ── Filtro de exclusión (Artefactos, etc.) ────────────────────
            if any(_safe_int(row.get(c, 0)) != 0 for c in EXCLUDE_COLUMNS):
                excl_excluded += 1
                continue

            # ── Resolución de label unificado (0 a 6) ─────────────────────
            label = resolve_unified_label(row)
            if label is None:
                excl_no_label += 1
                continue

            # ── Carga física de imagen ────────────────────────────────────
            try:
                arr = load_image_as_array(img_path, target_size, force_channels)
            except Exception as e:
                print(f"  [WARN] Error cargando {img_path}: {e}")
                missing.append(str(img_path))
                continue

            # ── Escritura directa en HDF5 ─────────────────────────────────
            img_ds[written] = arr
            lbl_ds[written] = label
            tx_ds[written]  = _safe_int(row.get(TOPLEFT_X_COL, 0))
            ty_ds[written]  = _safe_int(row.get(TOPLEFT_Y_COL, 0))

            per_class[CLASS_NAMES[label]] += 1
            written += 1

        # ── Limpieza/Ajuste de tamaño final ───────────────────────────────
        if written == 0:
            if os.path.exists(h5_out_path):
                os.remove(h5_out_path)
                print(f"El archivo {h5_out_path} fue eliminado porque no contenía parches válidos.")
        else:
            for ds in (img_ds, lbl_ds, tx_ds, ty_ds):
                ds.resize((written,) + ds.shape[1:])

    return {
        "n_images":           written,
        "per_class":          per_class,
        "excluded_filter":    excl_excluded,
        "excluded_no_label":  excl_no_label,
        "missing_images":     missing,
    }


# ---------------------------------------------------------------------------
# HELPER DE IMPRESIÓN
# ---------------------------------------------------------------------------

def _print_unified_stats(n_slide: str, stats: dict) -> None:
    n = stats["n_images"]
    print(f"\n{'='*55}")
    print(f"  [UNIFICADO] Slide {n_slide}")
    print(f"{'='*55}")
    print(f"  Patches escritos     : {n:,}")
    print(f"  Excl. filtro         : {stats['excluded_filter']:,}")
    print(f"  Excl. sin label      : {stats['excluded_no_label']:,}")
    print(f"  Imágenes faltantes   : {len(stats['missing_images']):,}")
    for cls in CLASS_NAMES:
        cnt = stats['per_class'][cls]
        idx = UNIFIED_INDEX[cls]
        pct = 100 * cnt / n if n > 0 else 0
        print(f"    [{idx}] {cls:<28} {cnt:>8,}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# SCRIPT PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    ROOT = Path(r'C:\Users\guigo\OneDrive\Escritorio\TFG_Biopsias\Proyecto')

    # Única carpeta de destino unificada
    OUT_DIR = ROOT / 'h5_multiclass_UNI'
    OUT_DIR.mkdir(exist_ok=True)

    TARGET_SIZE    = (224, 224)
    FORCE_CHANNELS = 3

    # Acumuladores globales para el reporte final del dataset completo
    g_stats = {
        "written": 0, "excl_filt": 0, "excl_lbl": 0, "missing": 0,
        "per_class": {cls: 0 for cls in CLASS_NAMES}
    }

    # ── Bucle por slide ──────────────────────────────────────────────────────
    for n_slide in range(1, 201):
        n_slide_str = str(n_slide).zfill(3)

        DATA_DIR = ROOT / 'Dataset_Publico' / f'zoom_2_{n_slide_str}'
        CSV_PATH = DATA_DIR / f'{n_slide_str}_labels.csv'
        H5_OUT   = OUT_DIR / f'{n_slide_str}_multiclass_UNI.h5'

        if not CSV_PATH.exists():
            print(f"[WARN] CSV no encontrado: {CSV_PATH}, saltando.")
            continue

        df = load_csv(CSV_PATH)

        # Procesa el slide una única vez
        stats = build_h5_unified(
            df, DATA_DIR, n_slide_str, H5_OUT,
            target_size=TARGET_SIZE, force_channels=FORCE_CHANNELS,
        )
        _print_unified_stats(n_slide_str, stats)

        # Acumular métricas globales
        g_stats["written"]   += stats["n_images"]
        g_stats["excl_filt"] += stats["excluded_filter"]
        g_stats["excl_lbl"]  += stats["excluded_no_label"]
        g_stats["missing"]   += len(stats["missing_images"])
        for cls in CLASS_NAMES:
            g_stats["per_class"][cls] += stats["per_class"][cls]


    # ── Resumen global definitivo ────────────────────────────────────────────
    print(f"\n{'#'*55}")
    print("  RESUMEN GLOBAL FINAL — DATASET UNIFICADO")
    print(f"{'#'*55}")
    print(f"  Total patches escritos : {g_stats['written']:,}")
    print(f"  Total excl. filtro     : {g_stats['excl_filt']:,}")
    print(f"  Total excl. sin label  : {g_stats['excl_lbl']:,}")
    print(f"  Total faltantes        : {g_stats['missing']:,}")
    print(f"{'-'*55}")
    for cls in CLASS_NAMES:
        cnt = g_stats["per_class"][cls]
        idx = UNIFIED_INDEX[cls]
        pct = 100 * cnt / g_stats["written"] if g_stats["written"] > 0 else 0
        print(f"    [{idx}] {cls:<28} {cnt:>10,}  ({pct:.1f}%)")
    print(f"{'#'*55}")
    print(f"  Archivos unificados generados con éxito en: {OUT_DIR}")