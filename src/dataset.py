"""
src/dataset.py
Laboratorio 3 - Deep Learning: Reconocimiento de Lenguaje de Señas (ASL)
CC3084 - Data Science, UVG

Descarga y organiza el dataset crudo de Kaggle. Su única responsabilidad es dejar
data/asl_alphabet_train/asl_alphabet_train/<clase>/*.jpg listo para que
src/preprocessing.py lo consuma; no hace split ni normalización.

Es idempotente: si las 29 clases ya están completas, no descarga ni copia nada.
"""

import shutil
from pathlib import Path

KAGGLE_DATASET = "grassknoted/asl-alphabet"
RAW_DIR = Path("data/asl_alphabet_train/asl_alphabet_train")

# 26 letras + del + nothing + space. se valida contra los nombres, no contra la cantidad:
# una carpeta de sobra no debe tapar la ausencia de una clase real
EXPECTED_CLASSES = frozenset(
    [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]
)


def _populated_classes(raw_dir):
    """
    Nombres de las carpetas de clase que existen y tienen al menos una imagen.
    Una carpeta vacía no cuenta: es justo el caso que dejó el dataset a medias antes.
    """
    if not raw_dir.is_dir():
        return set()
    return {
        d.name
        for d in raw_dir.iterdir()
        if d.is_dir() and next(d.glob("*.jpg"), None) is not None
    }


def _find_class_root(download_path):
    """
    Ubica, dentro de lo que descargó kagglehub, la carpeta que contiene directamente
    las subcarpetas de clase. El zip de Kaggle anida asl_alphabet_train/asl_alphabet_train/,
    pero no damos por sentada esa profundidad: se busca la carpeta que tenga todas
    las clases esperadas.
    """
    candidates = [download_path] + [d for d in download_path.rglob("*") if d.is_dir()]
    for candidate in candidates:
        if EXPECTED_CLASSES <= _populated_classes(candidate):
            return candidate
    raise RuntimeError(
        f"No se encontraron las {len(EXPECTED_CLASSES)} clases esperadas dentro de {download_path}. "
        "Revisar manualmente qué estructura trajo la descarga."
    )


def ensure_dataset(raw_dir=RAW_DIR):
    """
    Garantiza que el dataset crudo esté completo en raw_dir.

    Si ya están las 29 clases pobladas, no hace nada. Si falta alguna, descarga de
    Kaggle vía kagglehub y copia únicamente las clases faltantes (copia, no mueve,
    para no dejar el cache de kagglehub a medias si algo falla).
    """
    missing = EXPECTED_CLASSES - _populated_classes(raw_dir)
    if not missing:
        print(f"Dataset completo en {raw_dir} ({len(EXPECTED_CLASSES)} clases), no se descarga nada.")
        return raw_dir

    print(f"Faltan {len(missing)} clases en {raw_dir}: {sorted(missing)}. Descargando de Kaggle...")
    import kagglehub
    download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    class_root = _find_class_root(download_path)

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(missing):
        shutil.copytree(class_root / name, raw_dir / name, dirs_exist_ok=True)
        print(f"  copiada: {name}")

    still_missing = EXPECTED_CLASSES - _populated_classes(raw_dir)
    if still_missing:
        raise RuntimeError(
            f"El dataset quedó incompleto en {raw_dir}. Faltan: {sorted(still_missing)}"
        )

    print(f"Dataset listo en {raw_dir} ({len(EXPECTED_CLASSES)} clases).")
    return raw_dir


if __name__ == "__main__":
    ensure_dataset()
