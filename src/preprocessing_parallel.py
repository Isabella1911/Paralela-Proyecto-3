"""
src/preprocessing_parallel.py
Proyecto 3 - Mejorando algoritmos con CUDA
CC3069 - Computación Paralela y Distribuida, UVG

FASE 1 DEL PIPELINE: PREPARACIÓN DE DATOS (CPU, VARIOS NÚCLEOS)

Versión paralela de la carga de imágenes de src/preprocessing.py. El manifiesto y
el split estratificado se reutilizan tal cual (son baratos: solo trabajan con rutas);
lo que se paraleliza es load_and_preprocess, que es donde se va el tiempo:
por cada imagen hay que leer el archivo del disco, decodificar el JPEG, redimensionar
a 64x64 y normalizar a [0, 1].

Este trabajo se queda en la CPU a propósito:
- leer archivos y decodificar JPEG son operaciones de E/S y de lógica con muchos
  saltos (Huffman, cabeceras), que la GPU no acelera;
- cada imagen es independiente de las demás, así que se reparte por bloques entre
  procesos (no hilos: el GIL de Python serializaría parte del trabajo).

El resultado es idéntico bit a bit al de la versión secuencial (se valida con
verify_same_output), así que los dos pipelines entregan exactamente los mismos datos
al entrenamiento.
"""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from preprocessing import (
    IMG_SIZE,
    PROCESSED_DIR,
    build_manifest,
    get_classes,
    load_and_preprocess,
    stratified_split,
)

DEFAULT_WORKERS = os.cpu_count() or 1


def make_pool(workers=DEFAULT_WORKERS):
    """
    Pool de procesos con "spawn" en todos los sistemas operativos. En Linux el default
    es "fork", que copia el proceso padre tal cual; si el padre ya inicializó CUDA (el
    benchmark consulta la GPU al arrancar) los hijos pueden colgarse. "spawn" arranca
    procesos limpios, igual que en Windows.
    """
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def _load_chunk(args):
    """
    Trabajo de un proceso: carga y preprocesa un bloque contiguo de imágenes.
    Es la misma operación que hace el ciclo de load_and_preprocess, pero sobre una
    parte de las rutas.
    """
    paths, img_size = args
    X = np.zeros((len(paths), img_size, img_size, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB").resize((img_size, img_size))
        X[i] = np.asarray(img, dtype=np.float32) / 255.0
    return X


def load_and_preprocess_parallel(df, img_size=IMG_SIZE, classes=None, workers=DEFAULT_WORKERS, pool=None):
    """
    Igual que preprocessing.load_and_preprocess, pero repartiendo las imágenes entre
    `workers` procesos. Se hacen 4 bloques por proceso para balancear la carga (no
    todas las imágenes tardan lo mismo en decodificarse) sin pagar demasiado en
    comunicación entre procesos.

    Si se pasa un `pool` ya creado se reutiliza; así, al procesar train/val/test, el
    costo de arrancar los procesos se paga una sola vez.
    """
    if classes is None:
        classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    paths = df["filepath"].tolist()
    y = df["label"].map(class_to_idx).to_numpy(dtype=np.int64)

    n_chunks = max(1, min(len(paths), workers * 4))
    bounds = np.linspace(0, len(paths), n_chunks + 1, dtype=int)
    chunks = [(paths[a:b], img_size) for a, b in zip(bounds[:-1], bounds[1:])]

    if pool is None:
        with make_pool(workers) as own_pool:
            parts = list(own_pool.map(_load_chunk, chunks))
    else:
        parts = list(pool.map(_load_chunk, chunks))

    # map respeta el orden de entrada, así que X queda alineado con y
    return np.concatenate(parts, axis=0), y


def build_processed_splits_sequential(save=True):
    """
    La versión ORIGINAL (secuencial) del Lab 3, con cronómetro por sub-etapa para
    poder compararla con la paralela.
    """
    times = {}
    t0 = time.perf_counter()
    classes = get_classes()
    train_df, val_df, test_df = stratified_split(build_manifest())
    times["manifiesto_split"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    splits = tuple(load_and_preprocess(df, classes=classes) for df in (train_df, val_df, test_df))
    times["carga_imagenes"] = time.perf_counter() - t0

    if save:
        t0 = time.perf_counter()
        save_splits(classes, *splits)
        times["guardado"] = time.perf_counter() - t0
    return splits, times


def build_processed_splits_parallel(workers=DEFAULT_WORKERS, save=True):
    """Mismo pipeline que el secuencial, con la carga de píxeles en paralelo."""
    times = {}
    t0 = time.perf_counter()
    classes = get_classes()
    train_df, val_df, test_df = stratified_split(build_manifest())
    times["manifiesto_split"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with make_pool(workers) as pool:
        splits = tuple(load_and_preprocess_parallel(df, classes=classes, workers=workers, pool=pool)
                       for df in (train_df, val_df, test_df))
    times["carga_imagenes"] = time.perf_counter() - t0

    if save:
        t0 = time.perf_counter()
        save_splits(classes, *splits)
        times["guardado"] = time.perf_counter() - t0
    return splits, times


def save_splits(classes, train, val, test, out_dir=PROCESSED_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (X, y) in (("train", train), ("val", val), ("test", test)):
        np.save(out_dir / f"X_{name}.npy", X)
        np.save(out_dir / f"y_{name}.npy", y)
    pd.Series(classes).to_csv(out_dir / "classes.csv", index=False, header=["label"])


def verify_same_output(splits_a, splits_b):
    """True si ambos pipelines produjeron exactamente los mismos arrays."""
    return all(np.array_equal(Xa, Xb) and np.array_equal(ya, yb)
               for (Xa, ya), (Xb, yb) in zip(splits_a, splits_b))


if __name__ == "__main__":
    _, times = build_processed_splits_parallel()
    print(f"Preprocesamiento paralelo con {DEFAULT_WORKERS} procesos: "
          + ", ".join(f"{k}={v:.2f}s" for k, v in times.items()))
