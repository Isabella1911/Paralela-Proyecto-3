"""
src/cuda_kernels.py
Proyecto 3 - Mejorando algoritmos con CUDA

Kernels CUDA escritos a mano con Numba (CUDA Python) para la parte matemática de la
CNN 2: la pasada hacia adelante (inferencia) capa por capa.

PyTorch/cuDNN (src/train.py) esconde cómo se reparte el trabajo en la GPU. Este módulo
lo hace explícito: cada kernel define qué calcula UN hilo y cómo se organizan los
hilos en bloques y los bloques en el grid. Las mismas operaciones se implementan en
CPU (compiladas con Numba) en versión secuencial y en versión multinúcleo, para
comparar el mismo algoritmo en los tres escenarios.

Diseño de la distribución del trabajo
-------------------------------------
conv3x3 + ReLU + BatchNorm (fusionados en un solo kernel)
    Un hilo calcula UN píxel de salida (n, f, oy, ox): 9*C multiplicaciones-suma.
    Bloque: 16x16 hilos = un parche de 16x16 píxeles de un mismo mapa de salida.
    Grid:   (ceil(Wo/16), ceil(Ho/16), N*F); el eje z selecciona imagen y filtro.
    Como todos los hilos de un bloque usan el mismo filtro f, sus 9*C pesos se copian
    una vez a memoria compartida (shared memory) y se leen de ahí, en vez de que cada
    uno de los 256 hilos los lea de la memoria global.
    Fusionar ReLU y BN evita escribir y volver a leer el mapa completo dos veces más.

max pooling 2x2
    Un hilo por píxel de salida; mismo esquema de bloques 16x16 y grid con N*C en z.

capa densa (multiplicación de matrices + bias + ReLU opcional)
    Multiplicación por bloques (tiling) con memoria compartida: cada bloque de 16x16
    hilos calcula un bloque 16x16 de la salida; en cada paso carga un mosaico de A y uno
    de B a shared memory y cada hilo acumula 16 productos. Cada dato de memoria global
    se lee 16 veces menos que en la versión ingenua.
    Bloque: (16, 16). Grid: (ceil(N_out/16), ceil(M/16)).

Las imágenes se procesan por lotes (BATCH) para que los mapas intermedios quepan en
GPUs de 4 GB: con las 3045 imágenes de test de una vez, solo la salida de la primera
convolución ocuparía ~1.5 GB.
"""

import math
import time

import numpy as np
from numba import cuda, float32, njit, prange

TPB = 16                 # hilos por bloque en cada dimensión (16x16 = 256 hilos)
TILE = 16                # tamaño del mosaico de la multiplicación de matrices
MAX_W = 128 * 9          # pesos de un filtro: 9*C, con C <= 128
BATCH = 256              # imágenes por lote en la GPU


# ============================================================================
# Kernels CUDA
# ============================================================================
@cuda.jit
def conv3x3_relu_bn_kernel(x, w, b, scale, shift, out):
    """x: (N,C,H,W)  w: (F,C,3,3)  out: (N,F,H-2,W-2)"""
    C = x.shape[1]
    F, Ho, Wo = out.shape[1], out.shape[2], out.shape[3]

    ox = cuda.blockIdx.x * TPB + cuda.threadIdx.x
    oy = cuda.blockIdx.y * TPB + cuda.threadIdx.y
    nf = cuda.blockIdx.z
    n = nf // F
    f = nf % F

    # los 256 hilos del bloque cooperan para copiar los 9*C pesos del filtro f
    w_sh = cuda.shared.array(MAX_W, float32)
    tid = cuda.threadIdx.y * TPB + cuda.threadIdx.x
    for i in range(tid, C * 9, TPB * TPB):
        k = i % 9
        w_sh[i] = w[f, i // 9, k // 3, k % 3]
    cuda.syncthreads()

    if ox < Wo and oy < Ho:
        acc = b[f]
        for c in range(C):
            base = c * 9
            for ky in range(3):
                for kx in range(3):
                    acc += x[n, c, oy + ky, ox + kx] * w_sh[base + ky * 3 + kx]
        if acc < 0.0:
            acc = 0.0
        out[n, f, oy, ox] = acc * scale[f] + shift[f]


@cuda.jit
def maxpool2x2_kernel(x, out):
    """x: (N,C,H,W)  out: (N,C,H//2,W//2)"""
    C, Ho, Wo = out.shape[1], out.shape[2], out.shape[3]
    ox = cuda.blockIdx.x * TPB + cuda.threadIdx.x
    oy = cuda.blockIdx.y * TPB + cuda.threadIdx.y
    nc = cuda.blockIdx.z
    n = nc // C
    c = nc % C
    if ox < Wo and oy < Ho:
        iy, ix = 2 * oy, 2 * ox
        m = x[n, c, iy, ix]
        m = max(m, x[n, c, iy, ix + 1])
        m = max(m, x[n, c, iy + 1, ix])
        m = max(m, x[n, c, iy + 1, ix + 1])
        out[n, c, oy, ox] = m


@cuda.jit
def dense_tiled_kernel(A, B, bias, out, relu):
    """out = A @ B + bias, con ReLU opcional.  A: (M,K)  B: (K,Nout)"""
    sA = cuda.shared.array((TILE, TILE), float32)
    sB = cuda.shared.array((TILE, TILE), float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE + tx
    row = cuda.blockIdx.y * TILE + ty
    M, K = A.shape
    Nout = B.shape[1]

    acc = float32(0.0)
    for t in range((K + TILE - 1) // TILE):
        k_a = t * TILE + tx
        k_b = t * TILE + ty
        sA[ty, tx] = A[row, k_a] if (row < M and k_a < K) else 0.0
        sB[ty, tx] = B[k_b, col] if (k_b < K and col < Nout) else 0.0
        cuda.syncthreads()
        for k in range(TILE):
            acc += sA[ty, k] * sB[k, tx]
        cuda.syncthreads()

    if row < M and col < Nout:
        acc += bias[col]
        if relu and acc < 0.0:
            acc = 0.0
        out[row, col] = acc


def _grid_maps(n_maps, h, w):
    """Grid y bloque para kernels de "un hilo por píxel de salida"."""
    return (math.ceil(w / TPB), math.ceil(h / TPB), n_maps), (TPB, TPB)


# ============================================================================
# Mismos algoritmos en CPU (Numba). Se escriben una vez con prange y se compilan
# dos veces: sin parallel, prange se comporta como range (1 núcleo); con
# parallel=True, Numba reparte las iteraciones externas entre todos los núcleos.
# (Sin cache=True: las dos versiones compartirían el mismo archivo de caché.)
# ============================================================================
def _conv3x3_relu_bn_cpu(x, w, b, scale, shift, out):
    C = x.shape[1]
    F, Ho, Wo = out.shape[1], out.shape[2], out.shape[3]
    for nf in prange(out.shape[0] * F):
        n = nf // F
        f = nf % F
        for oy in range(Ho):
            for ox in range(Wo):
                acc = b[f]
                for c in range(C):
                    for ky in range(3):
                        for kx in range(3):
                            acc += x[n, c, oy + ky, ox + kx] * w[f, c, ky, kx]
                if acc < 0.0:
                    acc = 0.0
                out[n, f, oy, ox] = acc * scale[f] + shift[f]


def _maxpool2x2_cpu(x, out):
    C, Ho, Wo = out.shape[1], out.shape[2], out.shape[3]
    for nc in prange(out.shape[0] * C):
        n = nc // C
        c = nc % C
        for oy in range(Ho):
            for ox in range(Wo):
                iy, ix = 2 * oy, 2 * ox
                m = x[n, c, iy, ix]
                m = max(m, x[n, c, iy, ix + 1])
                m = max(m, x[n, c, iy + 1, ix])
                m = max(m, x[n, c, iy + 1, ix + 1])
                out[n, c, oy, ox] = m


def _dense_cpu(A, B, bias, out, relu):
    M, K = A.shape
    Nout = B.shape[1]
    for i in prange(M):
        for j in range(Nout):
            acc = bias[j]
            for k in range(K):
                acc += A[i, k] * B[k, j]
            if relu and acc < 0.0:
                acc = 0.0
            out[i, j] = acc


CPU_IMPLS = {
    "cpu_secuencial": tuple(njit(fn) for fn in (_conv3x3_relu_bn_cpu, _maxpool2x2_cpu, _dense_cpu)),
    "cpu_paralelo": tuple(njit(parallel=True)(fn) for fn in (_conv3x3_relu_bn_cpu, _maxpool2x2_cpu, _dense_cpu)),
}


# ============================================================================
# Pesos: del modelo de PyTorch entrenado a arrays de NumPy
# ============================================================================
def extract_weights(model):
    """
    Extrae los pesos de la CNN2 (en modo evaluación) como arrays float32 contiguos.
    BatchNorm en inferencia es una transformación afín por canal:
        y = gamma * (x - media) / sqrt(var + eps) + beta = x * scale + shift
    así que se reduce a dos vectores, scale y shift.
    """
    model = model.eval().cpu()
    to_np = lambda t: np.ascontiguousarray(t.detach().numpy(), dtype=np.float32)
    convs = []
    for block in model.features:
        conv, bn = block[0], block[2]
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        shift = bn.bias - bn.running_mean * scale
        convs.append((to_np(conv.weight), to_np(conv.bias), to_np(scale), to_np(shift)))
    denses = [(to_np(layer.weight.T), to_np(layer.bias))
              for layer in model.classifier if hasattr(layer, "weight")]
    return convs, denses


# ============================================================================
# Pasada hacia adelante completa
# ============================================================================
def forward_cpu(X_nhwc, weights, mode="cpu_paralelo"):
    """Logits de la CNN 2 calculados en CPU con las funciones de Numba."""
    conv_fn, pool_fn, dense_fn = CPU_IMPLS[mode]
    convs, denses = weights
    x = np.ascontiguousarray(X_nhwc.transpose(0, 3, 1, 2), dtype=np.float32)
    for w, b, scale, shift in convs:
        n, _, h, wd = x.shape
        y = np.empty((n, w.shape[0], h - 2, wd - 2), dtype=np.float32)
        conv_fn(x, w, b, scale, shift, y)
        x = np.empty((n, w.shape[0], (h - 2) // 2, (wd - 2) // 2), dtype=np.float32)
        pool_fn(y, x)
    a = x.reshape(x.shape[0], -1)
    for i, (B, bias) in enumerate(denses):
        out = np.empty((a.shape[0], B.shape[1]), dtype=np.float32)
        dense_fn(a, B, bias, out, i < len(denses) - 1)
        a = out
    return a


def forward_gpu(X_nhwc, weights, batch=BATCH):
    """
    Logits de la CNN 2 calculados con los kernels CUDA propios.
    Devuelve (logits, tiempos) separando transferencias y cómputo:
        h2d      copia de pesos e imágenes CPU -> GPU
        kernels  ejecución de todos los kernels
        d2h      copia de logits GPU -> CPU
    """
    times = {"h2d": 0.0, "kernels": 0.0, "d2h": 0.0}
    convs, denses = weights

    t0 = time.perf_counter()
    d_convs = [tuple(cuda.to_device(a) for a in layer) for layer in convs]
    d_denses = [tuple(cuda.to_device(a) for a in layer) for layer in denses]
    cuda.synchronize()
    times["h2d"] += time.perf_counter() - t0

    X = np.ascontiguousarray(X_nhwc.transpose(0, 3, 1, 2), dtype=np.float32)
    results = []
    for start in range(0, X.shape[0], batch):
        t0 = time.perf_counter()
        d_x = cuda.to_device(X[start:start + batch])
        cuda.synchronize()
        times["h2d"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        n = d_x.shape[0]
        for d_w, d_b, d_scale, d_shift in d_convs:
            h, wd = d_x.shape[2], d_x.shape[3]
            f = d_w.shape[0]
            d_y = cuda.device_array((n, f, h - 2, wd - 2), dtype=np.float32)
            grid, block = _grid_maps(n * f, h - 2, wd - 2)
            conv3x3_relu_bn_kernel[grid, block](d_x, d_w, d_b, d_scale, d_shift, d_y)
            d_x = cuda.device_array((n, f, (h - 2) // 2, (wd - 2) // 2), dtype=np.float32)
            grid, block = _grid_maps(n * f, (h - 2) // 2, (wd - 2) // 2)
            maxpool2x2_kernel[grid, block](d_y, d_x)
        d_a = d_x.reshape(n, d_x.shape[1] * d_x.shape[2] * d_x.shape[3])
        for i, (d_B, d_bias) in enumerate(d_denses):
            n_out = d_B.shape[1]
            d_out = cuda.device_array((n, n_out), dtype=np.float32)
            grid = (math.ceil(n_out / TILE), math.ceil(n / TILE))
            dense_tiled_kernel[grid, (TILE, TILE)](d_a, d_B, d_bias, d_out, i < len(d_denses) - 1)
            d_a = d_out
        cuda.synchronize()
        times["kernels"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        results.append(d_a.copy_to_host())
        times["d2h"] += time.perf_counter() - t0

    return np.concatenate(results), times


def launch_config_table(n_images=BATCH):
    """Configuración de lanzamiento de cada kernel para un lote (tabla del informe)."""
    rows = []
    h, c = 64, 3
    for f in (32, 64, 128):
        ho, hp = h - 2, (h - 2) // 2
        for name, size, work in ((f"conv3x3+ReLU+BN ({c}->{f})", ho, f"{9 * c} MAC"),
                                 (f"maxpool 2x2 ({f})", hp, "max de 4")):
            grid, block = _grid_maps(n_images * f, size, size)
            rows.append({"kernel": name, "salida": f"{n_images}x{f}x{size}x{size}",
                         "bloque": f"{block[0]}x{block[1]}", "grid": "x".join(map(str, grid)),
                         "hilos": math.prod(grid) * TPB * TPB, "trabajo_por_hilo": work})
        h, c = hp, f
    for k, n_out in ((6 * 6 * 128, 256), (256, 128), (128, 29)):
        grid = (math.ceil(n_out / TILE), math.ceil(n_images / TILE))
        rows.append({"kernel": f"dense tiled ({k}->{n_out})", "salida": f"{n_images}x{n_out}",
                     "bloque": f"{TILE}x{TILE}", "grid": "x".join(map(str, grid)),
                     "hilos": math.prod(grid) * TILE * TILE, "trabajo_por_hilo": f"{k} MAC"})
    return rows
