"""
src/plots.py
Proyecto 3 - Mejorando algoritmos con CUDA

Gráficas del informe a partir de los CSV que deja src/benchmark.py:
- tiempo promedio por configuración (con desviación estándar) y speedup, por fase;
- uso de cada núcleo de la CPU y de la GPU a lo largo del tiempo, por configuración.
Se guardan en images/p3_*.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
LOGS_DIR = RESULTS_DIR / "logs"
IMAGES_DIR = Path("images")

CPU_COLOR = "#4C72B0"
GPU_COLOR = "#55A868"


def _color(config):
    return GPU_COLOR if ("gpu" in config or "cuda" in config) else CPU_COLOR


def plot_times_and_speedup(summary, fase, out_path):
    df = summary[summary["fase"] == fase].reset_index(drop=True)
    if df.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = [_color(c) for c in df["config"]]
    ax1.bar(df["config"], df["promedio_s"], yerr=df["desv_s"].fillna(0), color=colors, capsize=4)
    ax1.set_ylabel("tiempo promedio (s)")
    ax1.set_title(f"{fase}: tiempo")
    for i, v in enumerate(df["promedio_s"]):
        ax1.text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=8)
    ax2.bar(df["config"], df["speedup"], color=colors)
    ax2.axhline(1, color="gray", lw=0.8, ls="--")
    ax2.set_ylabel(f"speedup vs {df['base'].iloc[0]}")
    ax2.set_title(f"{fase}: speedup")
    for i, v in enumerate(df["speedup"]):
        ax2.text(i, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=8)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_resource_log(log_path, out_path):
    """Mapa de calor del uso por núcleo de CPU + curva de uso y memoria de la GPU."""
    df = pd.read_csv(log_path)
    if len(df) < 2:
        return
    cpu_cols = [c for c in df.columns if c.startswith("cpu")]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    im = ax1.imshow(df[cpu_cols].T, aspect="auto", cmap="viridis", vmin=0, vmax=100,
                    extent=[df["t_s"].iloc[0], df["t_s"].iloc[-1], len(cpu_cols) - 0.5, -0.5])
    ax1.set_ylabel("núcleo lógico")
    ax1.set_title(f"Uso de recursos: {log_path.stem}")
    fig.colorbar(im, ax=ax1, label="% CPU", pad=0.01)
    if df["gpu_util_%"].notna().any():
        ax2.plot(df["t_s"], df["gpu_util_%"], color=GPU_COLOR, label="uso GPU (%)")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("% GPU")
        ax3 = ax2.twinx()
        ax3.plot(df["t_s"], df["gpu_mem_usada_mb"], color="#C44E52", ls="--", label="memoria GPU (MB)")
        ax3.set_ylabel("MB")
        ax2.legend(loc="upper left", fontsize=8)
        ax3.legend(loc="upper right", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "sin GPU NVIDIA", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("tiempo (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_all_plots():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "resumen_speedup.csv"
    if path.exists():
        summary = pd.read_csv(path)
        for fase in summary["fase"].unique():
            plot_times_and_speedup(summary, fase, IMAGES_DIR / f"p3_speedup_{fase}.png")
    if LOGS_DIR.exists():
        # un log por configuración: la repetición 1 es representativa
        for log in sorted(LOGS_DIR.glob("*_rep1.csv")):
            plot_resource_log(log, IMAGES_DIR / f"p3_recursos_{log.stem.removesuffix('_rep1')}.png")
        final = LOGS_DIR / "modelo_final.csv"
        if final.exists():
            plot_resource_log(final, IMAGES_DIR / "p3_recursos_modelo_final.png")
    print(f"Gráficas guardadas en {IMAGES_DIR}/p3_*.png")


if __name__ == "__main__":
    make_all_plots()
