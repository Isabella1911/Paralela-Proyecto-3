# Proyecto 3 - Mejorando algoritmos con CUDA

Computación Paralela y Distribuida, UVG

## Integrantes

- Nelson Escalante
- Isabella Obando
- Anthony Lou

## Descripción

Este proyecto toma una solución que ya existía, el **Laboratorio 3 de Data Science: Deep Learning para
reconocer el alfabeto del Lenguaje de Señas Americano (ASL)**, y la acelera usando la GPU con **CUDA**.
La idea es medir de forma práctica cuánto mejora el rendimiento al pasar la carga computacional pesada
a la GPU.

Reconocer señas ASL es una aplicación de tecnología inclusiva: el objetivo es facilitar la comunicación
a partir de imágenes de señas hechas frente a una cámara. Para eso hay que procesar muchas imágenes y
hacer una gran cantidad de operaciones matemáticas para clasificarlas. Ese trabajo se puede hacer de
forma secuencial en la CPU o en paralelo en la GPU.

El pipeline se divide en dos fases con responsabilidades bien separadas:

| Fase | Hardware | Qué hace |
|------|----------|----------|
| Preparación de datos | CPU (varios núcleos) | Lectura de archivos, decodificación, redimensionamiento y normalización de imágenes, split train/val/test |
| Entrenamiento e inferencia del modelo | GPU (CUDA) | Operaciones de la red neuronal: convoluciones, multiplicaciones de matrices, activaciones, backpropagation |

## Etapas del proyecto

1. **Análisis de la solución secuencial.** Revisar el código del Lab 3, identificar sus partes principales
   y descomponerlo en operaciones, desde que se reciben las imágenes hasta que se obtiene la predicción.
   Identificar qué operaciones se repiten, qué datos procesan y cuántas veces se ejecutan.
2. **Medición del tiempo secuencial.** Ejecutar la solución original varias veces en las mismas
   condiciones, calcular el tiempo promedio y analizar qué factores influyen en el rendimiento.
3. **Oportunidades de paralelización.** Elegir qué operaciones pueden ejecutarse al mismo tiempo en la
   GPU y justificar esa elección.
4. **Diseño del algoritmo paralelo.** Definir cómo se reparte el trabajo entre hilos, bloques y grids
   de la GPU.
5. **Implementación con CUDA Python.** Implementar el diseño y medir los tiempos con los mismos datos y
   condiciones que la versión secuencial, incluyendo las transferencias CPU ↔ GPU.
6. **Análisis y conclusiones.** Comparar ambas versiones con tablas y gráficas, calcular el speedup de
   cada fase y explicar las diferencias de rendimiento.

## Entregables

- **Código** (`.py` / `.ipynb`) documentado, con la versión secuencial original y la versión optimizada.
  Debe quedar clara la separación entre la preparación de datos (CPU) y la configuración de CUDA para el
  entrenamiento del modelo (GPU).
- **Informe de rendimiento (PDF)** con:
  - Tabla comparativa de tiempos.
  - Speedup de cada fase.
  - Capturas del uso de recursos (Administrador de Tareas, `nvidia-smi` o similar) durante las
    ejecuciones, que muestren la carga en los núcleos de la CPU y el uso de memoria y núcleos de la GPU.
  - Discusión técnica:
    - ¿Por qué es más eficiente procesar las imágenes con varios núcleos de la CPU y delegar a la GPU
      solo la matemática de la red neuronal?
    - ¿Cómo permitió CUDA reducir o cambiar el tiempo de procesamiento?
    - ¿Qué factores justifican o limitan el uso de CUDA?

## Estructura del repositorio

```
data/fotos_propias/   Fotos propias de señas, para probar el modelo con imágenes reales
docs/report/          Documentación y reporte
images/               Figuras generadas
models/               Modelos entrenados (.keras)
notebooks/            Notebooks del Lab 3 (01-06) y del Proyecto 3 (07_proyecto3_cuda.ipynb)
results/              Métricas y tiempos del benchmark (tiempos_*.csv, resumen_speedup.csv)
results/logs/         Uso de cada núcleo de CPU y de la GPU durante cada medición
src/dataset.py        Descarga del dataset de Kaggle con kagglehub
src/preprocessing.py  Preprocesamiento ORIGINAL (secuencial) y split train/val/test
src/preprocessing_parallel.py  Fase 1 en CPU multiproceso (mismo resultado, bit a bit)
src/model.py          CNN 2 del Lab 3 en PyTorch
src/train.py          Fase 2: entrenamiento e inferencia en CPU o GPU (CUDA)
src/cuda_kernels.py   Kernels CUDA propios con Numba (conv, pooling, densa) y su versión en CPU
src/monitor.py        Registro del uso de CPU/GPU (psutil + nvidia-smi)
src/benchmark.py      Mide todas las fases varias veces y calcula el speedup
src/plots.py          Gráficas del informe (images/p3_*.png)
docs/report/informe_proyecto3.md  Informe (se exporta a PDF para la entrega)
requirements.txt      Dependencias
```

## Datos y ejecución

El dataset es el [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) de Kaggle,
con 29 clases (A-Z, `del`, `nothing` y `space`). Los datos crudos (~1 GB) y los arrays procesados no se
versionan en git por su tamaño. Se generan localmente desde la raíz del repo:

```bash
pip install -r requirements.txt
python src/preprocessing.py
```

`src/dataset.py` descarga el dataset (solo si hace falta) en
`data/asl_alphabet_train/asl_alphabet_train/<clase>/*.jpg`. Después, `src/preprocessing.py` genera
`data/processed/{X,y}_{train,val,test}.npy` y `classes.csv`. Se usa `SEED=42` fijo, así que el split y la
submuestra (700 imágenes por clase) salen iguales en cualquier máquina. Esto permite comparar los
tiempos de la versión secuencial y la paralela con exactamente los mismos datos.

Para la fase en GPU se necesita una tarjeta NVIDIA con su driver instalado (el runtime de CUDA viene
dentro de los paquetes de PyTorch y numba-cuda). Se puede verificar con:

```bash
nvidia-smi
```

TensorFlow no tiene soporte de GPU en Windows nativo desde la versión 2.11, por eso la CNN 2 del Lab 3
se reescribió en PyTorch (`src/model.py`, misma arquitectura capa por capa). Así CPU y GPU se comparan
con el mismo framework.

## Ejecución del Proyecto 3

Instalación (en un entorno virtual, con ~12 GB libres):

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Comprobar que PyTorch y Numba ven la GPU:

```bash
python -c "import torch; from numba import cuda; print(torch.cuda.is_available(), cuda.is_available())"
```

Benchmark completo, desde la raíz del repo (descarga el dataset si hace falta):

```bash
python src/benchmark.py --reps 3 --epochs 3
```

Corre, en orden:

1. **Preprocesamiento**: secuencial vs paralelo (CPU), 3 repeticiones, verificando que ambos den los
   mismos arrays. Deja los `.npy` en `data/processed/`.
2. **Entrenamiento**: CNN 2 con 3 épocas en CPU con 1 hilo, CPU con todos los hilos, GPU y GPU con
   precisión mixta, 3 repeticiones cada una.
3. **Modelo final**: 15 épocas en GPU con early stopping; guarda `models/cnn2_torch.pt`.
4. **Kernels propios**: inferencia del test con `src/cuda_kernels.py` en CPU secuencial, CPU paralelo y
   GPU, comparada con PyTorch.

Se puede correr una sola fase (`--fases entrenamiento`) y saltar la configuración más lenta
(`--sin-cpu-1hilo`). Los resultados quedan en `results/`, las gráficas en `images/p3_*.png`, y el
notebook `notebooks/07_proyecto3_cuda.ipynb` los presenta con el análisis.

Mientras corre, abrir el Administrador de Tareas (Rendimiento → CPU con "procesadores lógicos" y
GPU con la gráfica "Cuda") y una terminal con `nvidia-smi -l 1` para tomar las capturas del informe.
Guardarlas en `docs/report/capturas/`.

## Origen del código base

El código de partida viene del repositorio del Laboratorio 3 de Data Science:
[EscasanN/Lab3-DataScience](https://github.com/EscasanN/Lab3-DataScience).
