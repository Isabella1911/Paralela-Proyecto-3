"""
src/train.py
Proyecto 3 - Mejorando algoritmos con CUDA

FASE 2 DEL PIPELINE: ENTRENAMIENTO E INFERENCIA (GPU CON CUDA)

Entrena la CNN 2 del Lab 3 con exactamente el mismo código en CPU o en GPU; solo
cambia el `device`. Así el speedup mide únicamente el efecto del hardware.

Separación de responsabilidades:
- CPU: lee los .npy que dejó la fase 1 (src/preprocessing*.py). Aquí no se abre
  ningún archivo de imagen.
- GPU: toda la matemática de la red (convoluciones, multiplicaciones de matrices,
  activaciones, batch norm, backpropagation y Adam) corre en kernels CUDA/cuDNN.

Decisiones para la GPU:
- Los tres splits se copian UNA sola vez a la memoria de la GPU (~1 GB en float32)
  en lugar de copiar cada batch. La transferencia CPU -> GPU se paga una vez y se
  mide aparte ("h2d").
- El barajado de cada época se hace con índices dentro de la GPU (torch.randperm en
  el device), sin volver a la CPU.
- torch.backends.cudnn.benchmark = True: cuDNN prueba varios algoritmos de
  convolución al inicio y se queda con el más rápido para estos tamaños.
- Antes de leer cada cronómetro se llama torch.cuda.synchronize(): los kernels CUDA
  se lanzan de forma asíncrona y sin sincronizar se mediría solo el lanzamiento.
- La pérdida y los aciertos se acumulan en la GPU; leerlos con .item() en cada batch
  obligaría a sincronizar CPU y GPU 445 veces por época.

Uso (desde la raíz del repo):
    python src/train.py --device cuda --epochs 15 --patience 3 --save models/cnn2_torch.pt
    python src/train.py --device cpu --threads 1 --epochs 3
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score

from model import CNN2, count_parameters

PROCESSED_DIR = Path("data/processed")
SEED = 42


def load_splits(processed_dir=PROCESSED_DIR):
    """Fase de CPU: lectura de los arrays ya preprocesados."""
    processed_dir = Path(processed_dir)
    splits = {name: (np.load(processed_dir / f"X_{name}.npy"), np.load(processed_dir / f"y_{name}.npy"))
              for name in ("train", "val", "test")}
    classes = pd.read_csv(processed_dir / "classes.csv")["label"].tolist()
    return splits, classes


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _predict(model, X, batch_size, device, amp):
    """Predice en batches (igual que model.predict de Keras con batch_size=32)."""
    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast(device.type, enabled=amp):
        for start in range(0, X.shape[0], batch_size):
            preds.append(model(X[start:start + batch_size]).argmax(dim=1))
    return torch.cat(preds)


def train_model(splits, device="cuda", epochs=15, batch_size=32, lr=5e-4,
                threads=None, amp=False, patience=None, seed=SEED, verbose=True):
    """
    Entrena la CNN 2 y devuelve (resultado, historial, modelo, y_pred). Cada etapa se
    cronometra por separado:

        h2d             copia CPU -> GPU de los datos (0 en CPU)
        entrenamiento   todas las épocas (forward + backward + Adam + validación)
        inferencia      predicción del conjunto de prueba
        d2h             copia GPU -> CPU de las predicciones (0 en CPU)

    patience=None entrena todas las épocas (lo que usa el benchmark, para que CPU y GPU
    hagan exactamente el mismo trabajo); con un entero se usa early stopping sobre
    val_loss restaurando los mejores pesos, como en el Lab 3.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió device=cuda pero PyTorch no detecta una GPU con CUDA.")
    if threads is not None:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    amp = amp and device.type == "cuda"

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits["train"], splits["val"], splits["test"]

    # --- transferencia CPU -> GPU (una sola vez) ---------------------------------
    _sync(device)
    t0 = time.perf_counter()
    Xtr, ytr = torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device)
    Xva, yva = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)
    Xte = torch.from_numpy(X_test).to(device)
    _sync(device)
    t_h2d = time.perf_counter() - t0 if device.type == "cuda" else 0.0

    model = CNN2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)   # eps por defecto de Keras
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    generator = torch.Generator(device=device).manual_seed(seed)

    history = []
    best_val, best_state, bad_epochs = float("inf"), None, 0
    n = Xtr.shape[0]

    # --- entrenamiento -----------------------------------------------------------
    t_train0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        perm = torch.randperm(n, device=device, generator=generator)
        running_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.detach() * yb.shape[0]
            correct += (logits.argmax(1) == yb).sum()

        # validación: parte del costo de cada época, como validation_data en Keras
        model.eval()
        val_loss = torch.zeros((), device=device)
        val_correct = torch.zeros((), device=device, dtype=torch.long)
        with torch.no_grad(), torch.autocast(device.type, enabled=amp):
            for start in range(0, Xva.shape[0], batch_size):
                logits = model(Xva[start:start + batch_size])
                yb = yva[start:start + batch_size]
                val_loss += loss_fn(logits.float(), yb) * yb.shape[0]
                val_correct += (logits.argmax(1) == yb).sum()
        _sync(device)

        row = {
            "epoch": epoch,
            "loss": running_loss.item() / n,
            "accuracy": correct.item() / n,
            "val_loss": val_loss.item() / Xva.shape[0],
            "val_accuracy": val_correct.item() / Xva.shape[0],
            "tiempo_s": time.perf_counter() - t_ep,
        }
        history.append(row)
        if verbose:
            print(f"[{device.type}] epoch {epoch:2d}/{epochs} - {row['tiempo_s']:6.2f}s - "
                  f"loss {row['loss']:.4f} acc {row['accuracy']:.4f} - "
                  f"val_loss {row['val_loss']:.4f} val_acc {row['val_accuracy']:.4f}", flush=True)

        if patience is not None:
            if row["val_loss"] < best_val:
                best_val, bad_epochs = row["val_loss"], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if verbose:
                        print(f"Early stopping en la época {epoch}")
                    break
    t_train = time.perf_counter() - t_train0
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- inferencia sobre test ---------------------------------------------------
    _sync(device)
    t0 = time.perf_counter()
    preds_dev = _predict(model, Xte, batch_size, device, amp)
    _sync(device)
    t_inf = time.perf_counter() - t0

    # --- transferencia GPU -> CPU de las predicciones ----------------------------
    t0 = time.perf_counter()
    y_pred = preds_dev.cpu().numpy()
    t_d2h = time.perf_counter() - t0 if device.type == "cuda" else 0.0

    result = {
        "device": "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device),
        "threads": torch.get_num_threads() if device.type == "cpu" else None,
        "amp": amp,
        "epochs": len(history),
        "batch_size": batch_size,
        "parametros": count_parameters(model),
        "h2d_s": t_h2d,
        "entrenamiento_s": t_train,
        "s_por_epoca": t_train / len(history),
        "inferencia_s": t_inf,
        "d2h_s": t_d2h,
        "total_s": t_h2d + t_train + t_inf + t_d2h,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "test_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }
    if device.type == "cuda":
        result["gpu_mem_max_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return result, history, model, y_pred


def main():
    parser = argparse.ArgumentParser(description="Entrena la CNN 2 del Lab 3 en CPU o GPU")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--threads", type=int, default=None, help="hilos de CPU para PyTorch")
    parser.add_argument("--amp", action="store_true", help="precisión mixta (solo GPU)")
    parser.add_argument("--patience", type=int, default=None, help="early stopping")
    parser.add_argument("--save", type=Path, default=None, help="ruta para guardar los pesos")
    args = parser.parse_args()

    t0 = time.perf_counter()
    splits, _ = load_splits()
    print(f"Datos cargados en CPU en {time.perf_counter() - t0:.2f}s "
          f"(train={len(splits['train'][1])}, val={len(splits['val'][1])}, test={len(splits['test'][1])})")

    result, _, model, _ = train_model(
        splits, device=args.device, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, threads=args.threads, amp=args.amp, patience=args.patience)

    print("\nResumen:")
    for k, v in result.items():
        print(f"  {k:16s} {v:.4f}" if isinstance(v, float) else f"  {k:16s} {v}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Pesos guardados en {args.save}")


if __name__ == "__main__":
    main()
