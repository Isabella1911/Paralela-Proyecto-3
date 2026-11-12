"""
src/monitor.py
Proyecto 3 - Mejorando algoritmos con CUDA

Registro del uso de recursos durante cada fase del benchmark, como evidencia de qué
hardware trabajó en cada una:
- CPU: porcentaje de uso de cada núcleo lógico (psutil).
- GPU: uso de los núcleos, memoria usada, temperatura y potencia (nvidia-smi).

Se usa como context manager y deja un CSV por medición en results/logs/:

    with ResourceMonitor("entrenamiento_gpu_cuda_rep1") as mon:
        train_model(...)
    mon.summary()

Complementa (no reemplaza) las capturas del Administrador de Tareas y de nvidia-smi
que pide el enunciado.
"""

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil

LOGS_DIR = Path("results/logs")
GPU_QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
GPU_COLUMNS = ["gpu_util_%", "gpu_mem_usada_mb", "gpu_mem_total_mb", "gpu_temp_c", "gpu_potencia_w"]


def _to_float(value):
    try:
        return float(value)
    except ValueError:      # nvidia-smi devuelve "[N/A]" en algunos campos
        return None


def query_gpu():
    """Una lectura de nvidia-smi (primera GPU), o None si no hay GPU NVIDIA."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        line = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
    except (subprocess.SubprocessError, OSError, IndexError):
        return None
    return [_to_float(v.strip()) for v in line.split(",")]


class ResourceMonitor:
    def __init__(self, name, interval=0.5, logs_dir=LOGS_DIR):
        self.name = name
        self.interval = interval
        self.path = Path(logs_dir) / f"{name}.csv"
        self.n_cores = psutil.cpu_count(logical=True)
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        psutil.cpu_percent(percpu=True)   # la primera lectura de psutil siempre es 0
        t0 = time.perf_counter()
        while not self._stop.wait(self.interval):
            cores = psutil.cpu_percent(percpu=True)
            gpu = query_gpu() or [None] * len(GPU_COLUMNS)
            self.samples.append([round(time.perf_counter() - t0, 2), *cores, *gpu])

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_s", *[f"cpu{i}" for i in range(self.n_cores)], *GPU_COLUMNS])
            writer.writerows(self.samples)

    def summary(self):
        """Promedios de la medición: uso medio de CPU, núcleos ocupados y uso de GPU."""
        if not self.samples:
            return {}
        k = self.n_cores
        mean_per_core = [sum(s[1 + i] for s in self.samples) / len(self.samples) for i in range(k)]
        gpu_util = [s[1 + k] for s in self.samples if s[1 + k] is not None]
        gpu_mem = [s[2 + k] for s in self.samples if s[2 + k] is not None]
        return {
            "cpu_promedio_%": sum(mean_per_core) / k,
            "nucleos_sobre_50%": sum(v > 50 for v in mean_per_core),
            "gpu_util_promedio_%": sum(gpu_util) / len(gpu_util) if gpu_util else None,
            "nvidia_smi_mem_max_mb": max(gpu_mem) if gpu_mem else None,
        }
