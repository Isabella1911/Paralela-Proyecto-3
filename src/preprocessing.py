"""
src/preprocessing.py
Laboratorio 3 - Deep Learning: Reconocimiento de Lenguaje de Señas (ASL)
Persona 2: EDA (cierre del ejercicio 2) + Preprocesamiento (ejercicio 3)
CC3084 - Data Science, UVG

Este módulo:
1. Construye un manifiesto (dataframe) con la ruta y clase de cada imagen,
   tomando una submuestra por clase (evita cargar las ~87,000 imágenes originales).
2. Crea el split train/val/test de forma ESTRATIFICADA, sobre las rutas
   (antes de cargar píxeles a memoria y antes de cualquier augmentation).
3. Aplica el preprocesamiento (resize + normalización) y guarda arrays .npy
   listos para que Persona 3 los consuma en sus modelos.
4. Da utilidades para convertir a tensores de TensorFlow o de PyTorch
   (usar solo la que corresponda al framework que eligió el equipo).
5. Detecta programáticamente qué letras se parecen visualmente entre sí,
   como apoyo a la pregunta del ejercicio 2 (M/N/S, U/V/R, etc.).

Se asume la estructura del repo:
    data/asl_alphabet_train/asl_alphabet_train/<clase>/*.jpg  <- dataset de Kaggle ya descomprimido
    data/processed/               <- aquí se guardan los splits ya procesados
    images/                       <- aquí se guardan las figuras del EDA
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataset import ensure_dataset

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/asl_alphabet_train/asl_alphabet_train")   # carpeta anidada tal cual la descomprime Kaggle
PROCESSED_DIR = Path("data/processed")
IMAGES_DIR = Path("images")

IMG_SIZE = 64          # justificado en el informe (ver build_processed_splits)
N_PER_CLASS = 700      # submuestra por clase, dentro del rango 500-800 sugerido en el enunciado
SEED = 42


def get_classes(raw_dir=RAW_DIR):
    ensure_dataset(raw_dir)
    return sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])


# ---------------------------------------------------------------------------
# 1. Manifiesto de imágenes (rutas + etiqueta, sin cargar píxeles todavía)
# ---------------------------------------------------------------------------
def build_manifest(raw_dir=RAW_DIR, n_per_class=N_PER_CLASS, seed=SEED):
    """
    Recorre cada carpeta de clase y arma un DataFrame [filepath, label],
    muestreando como máximo n_per_class imágenes por clase de forma
    aleatoria pero reproducible (mismo seed que usa Persona 1 en el EDA).
    """
    ensure_dataset(raw_dir)   # descarga y ordena el dataset si todavía no está completo

    rng = np.random.RandomState(seed)
    rows = []
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if len(files) > n_per_class:
            idx = rng.choice(len(files), size=n_per_class, replace=False)
            files = [files[i] for i in idx]
        rows.extend({"filepath": str(f), "label": class_dir.name} for f in files)

    manifest = pd.DataFrame(rows)
    print(f"Manifiesto: {len(manifest)} imágenes, {manifest['label'].nunique()} clases, "
          f"~{len(manifest) // manifest['label'].nunique()} img/clase")
    return manifest


# ---------------------------------------------------------------------------
# 2. Split estratificado train / val / test (ejercicio 2)
# ---------------------------------------------------------------------------
def stratified_split(manifest, train_size=0.70, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Divide el manifiesto en train/val/test manteniendo la proporción de
    clases en cada subconjunto (stratify=label). Se hace en dos pasos porque
    train_test_split solo separa en dos partes a la vez.

    Nota: el set de prueba oficial de Kaggle trae muy pocas imágenes por
    clase (a propósito, según el enunciado), así que NO se usa para evaluar;
    en su lugar generamos aquí nuestro propio conjunto de prueba.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "las proporciones deben sumar 1"

    train_df, temp_df = train_test_split(
        manifest,
        train_size=train_size,
        stratify=manifest["label"],
        random_state=seed,
    )
    relative_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val,
        stratify=temp_df["label"],
        random_state=seed,
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# 3. Carga + preprocesamiento: resize y normalización (ejercicio 3)
# ---------------------------------------------------------------------------
def load_and_preprocess(df, img_size=IMG_SIZE, classes=None):
    """
    Carga cada imagen del subconjunto, la redimensiona a img_size x img_size
    (interpolación bilineal, default de PIL) y normaliza los píxeles a [0, 1].
    Devuelve X (float32, NHWC) y y (enteros codificados según `classes`).
    """
    if classes is None:
        classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X = np.zeros((len(df), img_size, img_size, 3), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int64)

    for i, row in enumerate(df.itertuples()):
        img = Image.open(row.filepath).convert("RGB").resize((img_size, img_size))
        X[i] = np.asarray(img, dtype=np.float32) / 255.0   # normalización min-max
        y[i] = class_to_idx[row.label]

    return X, y


def build_processed_splits(save=True):
    """
    Pipeline completo: manifiesto -> split estratificado -> preprocesamiento
    -> guardado en .npy dentro de data/processed/.
    """
    classes = get_classes()
    manifest = build_manifest()
    train_df, val_df, test_df = stratified_split(manifest)

    X_train, y_train = load_and_preprocess(train_df, classes=classes)
    X_val, y_val = load_and_preprocess(val_df, classes=classes)
    X_test, y_test = load_and_preprocess(test_df, classes=classes)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(PROCESSED_DIR / "X_train.npy", X_train)
        np.save(PROCESSED_DIR / "y_train.npy", y_train)
        np.save(PROCESSED_DIR / "X_val.npy", X_val)
        np.save(PROCESSED_DIR / "y_val.npy", y_val)
        np.save(PROCESSED_DIR / "X_test.npy", X_test)
        np.save(PROCESSED_DIR / "y_test.npy", y_test)
        pd.Series(classes).to_csv(PROCESSED_DIR / "classes.csv", index=False, header=["label"])
        print(f"Splits procesados guardados en {PROCESSED_DIR}/")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ---------------------------------------------------------------------------
# 4. Conversión a tensores — usar SOLO la función del framework que eligió el equipo
# ---------------------------------------------------------------------------
def to_tensorflow_dataset(X, y, batch_size=32, shuffle=False):
    """Envuelve X, y en un tf.data.Dataset listo para model.fit()."""
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def to_pytorch_dataloader(X, y, batch_size=32, shuffle=False):
    """Envuelve X, y en un DataLoader de PyTorch (formato NCHW)."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    X_t = torch.from_numpy(X).permute(0, 3, 1, 2)  # NHWC -> NCHW
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# 5. Apoyo al ejercicio 2: ¿qué letras se confunden visualmente?
# ---------------------------------------------------------------------------
def average_images_by_class(manifest, img_size=IMG_SIZE, classes=None):
    """
    Calcula la imagen promedio (escala de grises) de cada clase. Comparar
    estos promedios entre clases es una forma objetiva de detectar qué
    letras tienen una forma de mano parecida.
    """
    if classes is None:
        classes = sorted(manifest["label"].unique())

    averages = {}
    for c in classes:
        subset = manifest[manifest["label"] == c]
        imgs = [
            np.asarray(Image.open(f).convert("L").resize((img_size, img_size)), dtype=np.float32) / 255.0
            for f in subset["filepath"]
        ]
        averages[c] = np.mean(imgs, axis=0)
    return averages


def remove_common_background(averages):
    """
    Le resta a cada promedio por clase el promedio global de todas las clases.

    Es solo para visualizar: como todas las fotos se tomaron en el mismo escenario,
    el fondo sale casi idéntico en las 29 clases y tapa la mano al graficar (la
    desviación entre clases es de ~0.05 sobre un rango de 0.14-0.73). Al restar lo
    común queda resaltado dónde difiere cada seña.

    OJO: no cambia el orden de most_similar_pairs. La distancia euclidiana es
    invariante a restar la misma constante a ambos términos: (A-G)-(B-G) = A-B.
    """
    global_mean = np.mean(list(averages.values()), axis=0)
    return {c: avg - global_mean for c, avg in averages.items()}


def most_similar_pairs(averages, top_n=10):
    """
    Ordena los pares de clases por similitud: menor distancia euclidiana
    entre sus imágenes promedio = más parecidas visualmente.
    """
    classes = list(averages.keys())
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dist = np.linalg.norm(averages[classes[i]] - averages[classes[j]])
            pairs.append((classes[i], classes[j], dist))
    pairs.sort(key=lambda x: x[2])
    return pairs[:top_n]


def plot_similar_pairs(averages, pairs, out_path=IMAGES_DIR / "letras_similares.png"):
    """
    Grafica los pares más parecidos. Le quita el fondo común a los promedios para
    que se note dónde está la mano; las distancias del título son las originales,
    calculadas sobre los promedios sin modificar.
    """
    # sin el fondo los valores quedan centrados en cero (positivos y negativos),
    # asi que se usa una escala divergente simetrica en vez de grises
    diffs = remove_common_background(averages)
    limit = max(np.abs(v).max() for v in diffs.values())
    style = {"cmap": "bwr", "vmin": -limit, "vmax": limit}

    fig, axes = plt.subplots(len(pairs), 2, figsize=(4, 2 * len(pairs)))
    for row, (a, b, dist) in enumerate(pairs):
        axes[row, 0].imshow(diffs[a], **style)
        axes[row, 0].set_title(f"{a} (vs. fondo)")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(diffs[b], **style)
        axes[row, 1].set_title(f"{b} (dist={dist:.2f})")
        axes[row, 1].axis("off")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    # 1) Split + preprocesamiento -> guarda los .npy que usará Persona 3
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_processed_splits()

    # 2) Análisis de letras visualmente similares (cierra el ejercicio 2)
    manifest = build_manifest()
    averages = average_images_by_class(manifest)
    top_pairs = most_similar_pairs(averages, top_n=8)
    print("\nPares de letras más parecidas visualmente (menor distancia = más parecidas):")
    for a, b, d in top_pairs:
        print(f"  {a} - {b}: distancia={d:.3f}")
    plot_similar_pairs(averages, top_pairs)