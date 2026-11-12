"""
src/benchmark.py
Proyecto 3 - Mejorando algoritmos con CUDA

Mide cada fase del pipeline en su versión secuencial (la original del Lab 3) y en su
versión optimizada, repitiendo varias veces en las mismas condiciones:

  1. preprocesamiento   secuencial (1 proceso)  vs  paralelo (todos los núcleos de la CPU)
  2. entrenamiento      CPU 1 hilo  vs  CPU todos los hilos  vs  GPU CUDA  vs  GPU + AMP
  3. modelo_final       entrena 15 épocas en GPU con early stopping (para el accuracy
                        y para tener pesos con los que probar los kernels)
  4. kernels            inferencia con kernels propios: CPU secuencial vs CPU paralelo vs
                        GPU CUDA (Numba), con PyTorch CPU/GPU como referencia

Cada repetición queda en results/tiempos_<fase>.csv y el uso de CPU/GPU durante cada
medición en results/logs/<fase>_<config>_rep<i>.csv. Al final se calcula el resumen con
promedio, desviación estándar y speedup en results/resumen_speedup.csv y se generan
las gráficas en images/p3_*.png.

Uso (desde la raíz del repo):
    python src/benchmark.py                                  # todo
    python src/benchmark.py --fases entrenamiento --reps 3 --epochs 3
    python src/benchmark.py --sin-cpu-1hilo                  # salta la configuración más lenta
"""

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from monitor import ResourceMonitor

RESULTS_DIR = Path("results")
MODEL_PATH = Path("models/cnn2_torch.pt")
PHASES = ["preprocesamiento", "entrenamiento", "modelo_final", "kernels"]

# configuración de referencia (la secuencial) y columna de tiempo usada para el speedup
BASELINES = {"preprocesamiento": "secuencial", "entrenamiento": "cpu_1hilo", "kernels": "numba_cpu_secuencial"}
TIME_COLUMN = {"preprocesamiento": "total_s", "entrenamiento": "entrenamiento_s", "kernels": "total_s"}


def system_info():
    import psutil
    import torch
    info = {
        "sistema": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_torch": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": props.name,
            "gpu_mem_gb": round(props.total_memory / 2**30, 1),
            "gpu_multiprocesadores": props.multi_processor_count,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
        })
    return info


# ---------------------------------------------------------------------------
# 1. Preprocesamiento (CPU)
# ---------------------------------------------------------------------------
def bench_preprocessing(reps, workers):
    from preprocessing import get_classes
    from preprocessing_parallel import (build_processed_splits_parallel, build_processed_splits_sequential,
                                        save_splits, verify_same_output)

    # calentamiento: la primera lectura del dataset viene del disco y las siguientes
    # del caché de archivos del sistema operativo; sin esto la primera configuración
    # medida saldría castigada
    print("\n== Preprocesamiento: calentamiento del caché de disco ==", flush=True)
    build_processed_splits_parallel(workers=workers, save=False)

    configs = {
        "secuencial": lambda: build_processed_splits_sequential(save=False),
        "paralelo": lambda: build_processed_splits_parallel(workers=workers, save=False),
    }
    rows, reference = [], None
    for rep in range(1, reps + 1):
        for config, fn in configs.items():      # alternadas, para repartir cualquier ruido
            print(f"\n== Preprocesamiento {config} - repetición {rep}/{reps} ==", flush=True)
            with ResourceMonitor(f"preprocesamiento_{config}_rep{rep}") as mon:
                t0 = time.perf_counter()
                splits, times = fn()
                total = time.perf_counter() - t0
            if reference is None:
                reference = splits
            elif not verify_same_output(reference, splits):
                raise RuntimeError("El preprocesamiento paralelo no produjo los mismos datos que el secuencial")
            rows.append({"fase": "preprocesamiento", "config": config, "rep": rep,
                         "procesos": 1 if config == "secuencial" else workers, "total_s": total,
                         **{f"{k}_s": v for k, v in times.items()}, **mon.summary()})
            print(f"   total {total:.2f}s  " + ", ".join(f"{k}={v:.2f}s" for k, v in times.items()), flush=True)

    # los .npy para el entrenamiento (idénticos en ambas versiones)
    save_splits(get_classes(), *reference)
    print("\nDatos idénticos entre la versión secuencial y la paralela: OK")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Entrenamiento (GPU)
# ---------------------------------------------------------------------------
def bench_training(reps, epochs, include_cpu1, include_amp):
    import torch
    from train import load_splits, train_model

    t0 = time.perf_counter()
    splits, _ = load_splits()
    print(f"\nCarga de .npy en CPU: {time.perf_counter() - t0:.2f}s", flush=True)

    configs = []
    if include_cpu1:
        configs.append(("cpu_1hilo", dict(device="cpu", threads=1)))
    configs.append(("cpu_multinucleo", dict(device="cpu", threads=os.cpu_count())))
    if torch.cuda.is_available():
        configs.append(("gpu_cuda", dict(device="cuda")))
        if include_amp:
            configs.append(("gpu_cuda_amp", dict(device="cuda", amp=True)))
        # calentamiento: crear el contexto CUDA y cargar cuDNN tarda unos segundos
        # la primera vez y no es parte del entrenamiento
        print("\n== Calentamiento de la GPU ==", flush=True)
        small = {k: (X[:512], y[:512]) for k, (X, y) in splits.items()}
        train_model(small, device="cuda", epochs=1, verbose=False)
    else:
        print("AVISO: PyTorch no detecta CUDA; solo se medirá la CPU.")

    rows, histories = [], []
    for rep in range(1, reps + 1):
        for config, kwargs in configs:
            print(f"\n== Entrenamiento {config} - repetición {rep}/{reps} ({epochs} épocas) ==", flush=True)
            with ResourceMonitor(f"entrenamiento_{config}_rep{rep}") as mon:
                result, history, _, _ = train_model(splits, epochs=epochs, **kwargs)
            rows.append({"fase": "entrenamiento", "config": config, "rep": rep, **result, **mon.summary()})
            histories += [{"config": config, "rep": rep, **h} for h in history]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    pd.DataFrame(histories).to_csv(RESULTS_DIR / "historial_epocas.csv", index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Modelo final (GPU, igual que el Lab 3: 15 épocas con early stopping)
# ---------------------------------------------------------------------------
def train_final_model(epochs):
    import torch
    from train import load_splits, train_model

    splits, _ = load_splits()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n== Entrenamiento del modelo final en {device} ({epochs} épocas, early stopping) ==", flush=True)
    with ResourceMonitor("modelo_final") as mon:
        result, history, model, _ = train_model(splits, device=device, epochs=epochs, patience=3)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    pd.DataFrame([{**result, **mon.summary()}]).to_csv(RESULTS_DIR / "metricas_modelo_final.csv", index=False)
    pd.DataFrame(history).to_csv(RESULTS_DIR / "historial_modelo_final.csv", index=False)
    print(f"Modelo guardado en {MODEL_PATH}: test accuracy = {result['test_accuracy']:.4f}")


# ---------------------------------------------------------------------------
# 4. Kernels CUDA propios (inferencia sobre test)
# ---------------------------------------------------------------------------
def bench_kernels(reps):
    import torch
    import cuda_kernels as ck
    from model import CNN2
    from train import load_splits

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Falta {MODEL_PATH}: correr antes la fase modelo_final")

    splits, _ = load_splits()
    X_test, y_test = splits["test"]

    model = CNN2()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    weights = ck.extract_weights(model)
    torch.set_num_threads(os.cpu_count())

    with torch.no_grad():
        reference = model(torch.from_numpy(X_test)).numpy()   # logits de PyTorch como referencia

    has_gpu = torch.cuda.is_available()

    # compilación JIT de Numba y calentamiento de cuDNN (no se miden)
    print("\n== Compilando kernels (Numba JIT) ==", flush=True)
    ck.forward_cpu(X_test[:2], weights, "cpu_secuencial")
    ck.forward_cpu(X_test[:2], weights, "cpu_paralelo")
    if has_gpu:
        ck.forward_gpu(X_test[:2], weights)
        model_gpu = CNN2()
        model_gpu.load_state_dict(model.state_dict())
        model_gpu = model_gpu.cuda().eval()
        with torch.no_grad():
            model_gpu(torch.from_numpy(X_test[:32]).cuda())

    def run_torch_cpu():
        with torch.no_grad():
            return model(torch.from_numpy(X_test)).numpy(), {}

    def run_torch_gpu():
        t0 = time.perf_counter()
        x = torch.from_numpy(X_test).cuda()
        torch.cuda.synchronize()
        t_h2d = time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            out = torch.cat([model_gpu(x[i:i + ck.BATCH]) for i in range(0, x.shape[0], ck.BATCH)])
        torch.cuda.synchronize()
        t_k = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = out.cpu().numpy()
        return out, {"h2d": t_h2d, "kernels": t_k, "d2h": time.perf_counter() - t0}

    configs = [
        ("numba_cpu_secuencial", lambda: (ck.forward_cpu(X_test, weights, "cpu_secuencial"), {})),
        ("numba_cpu_paralelo", lambda: (ck.forward_cpu(X_test, weights, "cpu_paralelo"), {})),
        ("pytorch_cpu", run_torch_cpu),
    ]
    if has_gpu:
        configs += [("numba_cuda_propio", lambda: ck.forward_gpu(X_test, weights)),
                    ("pytorch_cuda_cudnn", run_torch_gpu)]
    else:
        print("AVISO: no hay GPU; solo se medirán las versiones de CPU.")

    rows = []
    for rep in range(1, reps + 1):
        for config, fn in configs:
            with ResourceMonitor(f"kernels_{config}_rep{rep}") as mon:
                t0 = time.perf_counter()
                logits, parts = fn()
                total = time.perf_counter() - t0
            err = float(np.abs(logits - reference).max())
            acc = float((logits.argmax(1) == y_test).mean())
            rows.append({"fase": "kernels", "config": config, "rep": rep, "total_s": total,
                         **{f"{k}_s": v for k, v in parts.items()},
                         "error_max_vs_pytorch": err, "test_accuracy": acc, **mon.summary()})
            print(f"   {config:22s} rep {rep}: {total:8.3f}s  error máx={err:.2e}  acc={acc:.4f}", flush=True)

    pd.DataFrame(ck.launch_config_table()).to_csv(RESULTS_DIR / "configuracion_kernels.csv", index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Resumen: promedio, desviación y speedup frente a la versión secuencial
# ---------------------------------------------------------------------------
def summarize():
    frames = []
    for fase, base in BASELINES.items():
        path = RESULTS_DIR / f"tiempos_{fase}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        agg = (df.groupby("config", sort=False)[TIME_COLUMN[fase]]
                 .agg(promedio_s="mean", desv_s="std", min_s="min", reps="count").reset_index())
        if base not in set(agg["config"]):        # p. ej. si se usó --sin-cpu-1hilo
            base = agg["config"].iloc[0]
        agg.insert(0, "fase", fase)
        agg["base"] = base
        agg["speedup"] = agg.loc[agg["config"] == base, "promedio_s"].iloc[0] / agg["promedio_s"]
        if fase == "entrenamiento":
            # speedup contando también las transferencias CPU<->GPU y la inferencia
            total = df.groupby("config", sort=False)["total_s"].mean()
            agg["total_con_transferencias_s"] = agg["config"].map(total)
            agg["speedup_total"] = total[base] / agg["total_con_transferencias_s"]
        frames.append(agg)
    if not frames:
        return None
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(RESULTS_DIR / "resumen_speedup.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark secuencial vs paralelo (CPU/GPU)")
    parser.add_argument("--fases", nargs="+", default=PHASES, choices=PHASES)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3, help="épocas por repetición del benchmark")
    parser.add_argument("--epochs-final", type=int, default=15)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--sin-cpu-1hilo", action="store_true")
    parser.add_argument("--sin-amp", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    info = system_info()
    pd.Series(info).to_csv(RESULTS_DIR / "sistema.csv", header=["valor"])
    print("Sistema:", info, flush=True)

    if "preprocesamiento" in args.fases:
        bench_preprocessing(args.reps, args.workers).to_csv(RESULTS_DIR / "tiempos_preprocesamiento.csv", index=False)
    if "entrenamiento" in args.fases:
        bench_training(args.reps, args.epochs, not args.sin_cpu_1hilo, not args.sin_amp).to_csv(
            RESULTS_DIR / "tiempos_entrenamiento.csv", index=False)
    if "modelo_final" in args.fases:
        train_final_model(args.epochs_final)
    if "kernels" in args.fases:
        bench_kernels(args.reps).to_csv(RESULTS_DIR / "tiempos_kernels.csv", index=False)

    summary = summarize()
    if summary is not None:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print("\n", summary)

    from plots import make_all_plots
    make_all_plots()


if __name__ == "__main__":
    main()
