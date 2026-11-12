# Proyecto 3: Mejorando algoritmos con CUDA

**Informe de rendimiento y análisis.** Computación Paralela y Distribuida, UVG.
Integrantes: Nelson Escalante, Isabella Obando, Anthony Lou.

> Las partes marcadas con **[COMPLETAR]** se llenan con los resultados de `python src/benchmark.py`
> (archivos de `results/` e `images/`) y con las capturas de `docs/report/capturas/`.

## 1. Objetivo y entorno

Se tomó el Laboratorio 3 de Data Science (clasificación de las 29 señas del alfabeto ASL con una CNN) y
se separó su pipeline en dos fases con hardware distinto:

| Fase | Hardware | Implementación |
|------|----------|----------------|
| Preparación de datos: lectura, decodificación JPEG, resize, normalización, split | CPU, varios procesos | `src/preprocessing_parallel.py` (`ProcessPoolExecutor`) |
| Matemática de la red: entrenamiento e inferencia | GPU con CUDA | `src/train.py` (PyTorch + cuDNN) y `src/cuda_kernels.py` (kernels propios con Numba) |

Entorno de pruebas **[COMPLETAR con `results/sistema.csv`]**:

| Componente | Valor |
|------------|-------|
| CPU / núcleos lógicos | |
| RAM | |
| GPU / memoria / multiprocesadores | |
| Sistema operativo | |
| Python / PyTorch / CUDA / cuDNN | |

Datos: 20 300 imágenes (700 por clase, seed 42), split 14 210 / 3 045 / 3 045, 64×64×3 en float32.

## 2. Metodología de medición

- 3 repeticiones de cada configuración, alternadas, después de una pasada de calentamiento (caché de
  disco del sistema operativo, contexto CUDA, compilación JIT de Numba).
- Cronómetro `time.perf_counter()`; en GPU, `torch.cuda.synchronize()` / `cuda.synchronize()` antes de
  cada lectura, porque los kernels se lanzan de forma asíncrona.
- Entrenamiento del benchmark con 3 épocas fijas y batch 32 (el del Lab 3) en todas las
  configuraciones; el modelo final se entrena aparte con 15 épocas y early stopping.
- Se verifica que la versión paralela produzca exactamente los mismos datos (preprocesamiento) y los
  mismos logits que PyTorch (kernels propios, error máximo reportado).
- `src/monitor.py` registra cada 0.5 s el uso de cada núcleo de la CPU y de la GPU (`results/logs/`).

## 3. Tabla comparativa de tiempos

**[COMPLETAR con `results/resumen_speedup.csv`]**

### 3.1 Fase de preparación de datos (CPU)

| Configuración | Procesos | Tiempo promedio (s) | Desv. (s) | Speedup |
|---------------|---------:|--------------------:|----------:|--------:|
| Secuencial (original) | 1 | | | 1.00× |
| Paralelo | | | | |

### 3.2 Fase de entrenamiento (3 épocas)

| Configuración | Tiempo entrenamiento (s) | s/época | H2D (s) | Inferencia (s) | D2H (s) | Speedup | Speedup con transferencias |
|---------------|-------------------------:|--------:|--------:|---------------:|--------:|--------:|---------------------------:|
| CPU 1 hilo (secuencial) | | | 0 | | 0 | 1.00× | 1.00× |
| CPU todos los hilos | | | 0 | | 0 | | |
| GPU CUDA | | | | | | | |
| GPU CUDA + AMP | | | | | | | |

### 3.3 Kernels propios: inferencia de las 3 045 imágenes de test

| Configuración | Tiempo total (s) | Kernels (s) | H2D + D2H (s) | Error máx. vs PyTorch | Speedup |
|---------------|-----------------:|------------:|--------------:|----------------------:|--------:|
| Numba CPU secuencial | | – | – | | 1.00× |
| Numba CPU paralelo | | – | – | | |
| PyTorch CPU | | – | – | 0 | |
| Numba CUDA (kernels propios) | | | | | |
| PyTorch CUDA (cuDNN) | | | | | |

Gráficas: `images/p3_speedup_preprocesamiento.png`, `images/p3_speedup_entrenamiento.png`,
`images/p3_speedup_kernels.png`.

## 4. Speedup por fase

Speedup = tiempo promedio de la versión secuencial / tiempo promedio de la versión optimizada.

**[COMPLETAR]**

- Preparación de datos: S = T_secuencial / T_paralelo = ___ / ___ = ___×. Con *p* procesos el máximo
  teórico es *p*; la diferencia se explica por el arranque de los procesos, el envío de los arrays de
  vuelta al proceso principal y la parte secuencial (manifiesto y split).
- Entrenamiento: S = T_CPU 1 hilo / T_GPU = ___×, y frente a la CPU multinúcleo ___×.
- Pipeline completo (preparación + entrenamiento de 15 épocas): ___ s → ___ s, S = ___×.

## 5. Evidencia de uso de recursos

**[COMPLETAR con capturas]**

| Captura | Qué debe mostrar |
|---------|------------------|
| `capturas/preprocesamiento_secuencial.png` | Administrador de Tareas: un solo núcleo ocupado, GPU en ~0 %. |
| `capturas/preprocesamiento_paralelo.png` | Todos los núcleos lógicos ocupados, GPU en ~0 %. |
| `capturas/entrenamiento_cpu.png` | CPU al 100 % (1 o todos los núcleos según la configuración), GPU en ~0 %. |
| `capturas/entrenamiento_gpu.png` | `nvidia-smi`: proceso `python`, memoria ocupada y uso de GPU alto; CPU con un solo núcleo activo (el que lanza los kernels). |
| `capturas/kernels_numba_cuda.png` | `nvidia-smi` durante la inferencia con los kernels propios. |

Además, `images/p3_recursos_*.png` muestran el registro de `src/monitor.py`: mapa de calor del uso por
núcleo de la CPU y la curva de uso y memoria de la GPU en cada configuración.

## 6. Discusión técnica

### ¿Por qué es más eficiente procesar las imágenes con los múltiples núcleos del CPU y delegar exclusivamente la matemática de la red neuronal al GPU?

Porque cada fase tiene un tipo de trabajo distinto y cada procesador está diseñado para uno de ellos.

**La preparación de datos es trabajo de CPU.** Para cada imagen hay que pedirle el archivo al sistema
operativo, leer la cabecera del JPEG, decodificar Huffman (un proceso con mucha lógica condicional y
lectura bit a bit) y aplicar la IDCT. Después hay que redimensionar. Esto tiene muchas ramas y acceso a
disco, y la GPU no puede leer archivos: todo tendría que pasar por la CPU de todas formas. Además, el
trabajo por imagen es pequeño (unos pocos milisegundos) pero las imágenes son independientes, así que el
paralelismo natural es de grano grueso: una imagen o un bloque de imágenes por núcleo. Por eso se usan
procesos (`ProcessPoolExecutor`) y no hilos: el GIL de Python impediría que varios hilos ejecuten código
Python al mismo tiempo.

**La red neuronal es trabajo de GPU.** Una pasada hacia adelante de la CNN 2 son ~31 millones de
multiplicaciones-suma por imagen (96 % en las convoluciones). En el entrenamiento, con backpropagation,
son ~1.3 billones por época. Todas tienen la misma forma (sumas de productos), no tienen ramas y son
independientes entre sí: cada píxel de salida de una convolución, cada neurona de una capa densa y cada
imagen del batch se pueden calcular a la vez. Una GPU tiene miles de núcleos simples y un ancho de
banda de memoria muy alto justo para esto. La CPU, con pocos núcleos complejos, tarda mucho más aunque
use todos sus hilos.

**Separarlas evita el peor caso.** Si la GPU tuviera que esperar a que se decodifiquen las imágenes,
estaría ociosa; si la CPU hiciera la matemática, sería el cuello de botella. Al preparar todo una vez en
la CPU (en paralelo) y copiar los datos ya listos a la memoria de la GPU una sola vez, cada procesador
hace lo que mejor sabe hacer y la transferencia entre ambos se paga una sola vez.

### ¿Cómo CUDA permitió reducir o cambiar el tiempo de procesamiento?

**[COMPLETAR con los números]** El entrenamiento pasó de ___ s/época en la CPU con 1 hilo (___ s con
todos los hilos) a ___ s/época en la GPU, un speedup de ___×. Las razones:

1. **Paralelismo masivo.** Cada operación de la red se ejecuta como un kernel con miles de hilos. En los
   kernels propios (`src/cuda_kernels.py`) esto se ve directamente: la primera convolución de un lote de
   256 imágenes lanza un grid de 4×4×8192 bloques de 16×16 hilos (33.5 millones de hilos), uno por píxel
   de salida.
2. **Uso de la jerarquía de memoria.** Los pesos de cada filtro se copian una vez a memoria compartida
   y la capa densa usa multiplicación por mosaicos de 16×16, así que cada dato de la memoria global se
   lee muchas menos veces.
3. **Librerías optimizadas.** cuDNN elige el algoritmo de convolución más rápido para estos tamaños
   (`cudnn.benchmark=True`) y cuBLAS hace las multiplicaciones de matrices. La comparación entre
   nuestros kernels (___ s) y cuDNN (___ s) muestra cuánto suman esas optimizaciones.
4. **Menos transferencias.** Los datos se copian una sola vez a la GPU (___ s) y el barajado se hace en
   la GPU; las transferencias representan solo el ___ % del tiempo total.
5. **Precisión mixta (AMP).** Con FP16 se procesa el doble de datos por ciclo y se ocupa la mitad de
   memoria; resultado: ___×.

Lo que no cambió: el número de épocas y de pasos de gradiente sigue siendo el mismo, porque cada paso
depende de los pesos del anterior. CUDA acelera cada paso, no elimina la dependencia secuencial entre
pasos.

### ¿Qué factores justifican o limitan la utilización de CUDA?

**La justifican:**
- Mucho cómputo con la misma forma y sin dependencias (convoluciones, productos de matrices).
- Datos que caben en la memoria de la GPU y se reutilizan muchas veces (15 épocas sobre el mismo
  conjunto), de modo que la transferencia inicial se amortiza.
- Librerías maduras (cuDNN, cuBLAS) que ya están optimizadas para cada arquitectura de GPU.

**La limitan:**
- **Transferencias CPU ↔ GPU** por el bus PCIe: si el cómputo por dato es poco, copiar cuesta más que
  calcular. Por eso el preprocesamiento no se llevó a la GPU.
- **Tamaño del problema.** Con batches de 32 imágenes de 64×64 hay pocos hilos por kernel y la GPU no
  se llena; el costo fijo de lanzar cada kernel (microsegundos) pesa más. **[COMPLETAR: uso promedio
  de GPU medido = ___ %]**
- **Memoria de la GPU.** **[COMPLETAR: memoria de la GPU usada]**; obliga a procesar por lotes (los
  kernels propios usan lotes de 256 imágenes) y limita el tamaño del modelo y del batch.
- **Ley de Amdahl.** Las partes que siguen en la CPU (lectura de archivos, Python que lanza los
  kernels, métricas de sklearn) ponen un techo al speedup del pipeline completo.
- **Costo y portabilidad.** Requiere hardware NVIDIA y drivers; TensorFlow ya no soporta GPU en Windows
  nativo, lo que obligó a portar el modelo a PyTorch.
- **Costo de arranque.** Crear el contexto CUDA, cargar cuDNN y compilar los kernels de Numba tarda
  algunos segundos; para tareas muy cortas no compensa.

## 7. Librerías y funciones usadas para trabajar con la GPU

| Librería / función | Uso |
|--------------------|-----|
| `torch.cuda.is_available()`, `torch.device("cuda")`, `.to(device)` | Detectar la GPU y mover modelo y datos a su memoria. |
| `torch.backends.cudnn.benchmark` | Que cuDNN elija el algoritmo de convolución más rápido. |
| `torch.cuda.synchronize()` | Esperar a que terminen los kernels antes de medir el tiempo. |
| `torch.autocast`, `torch.amp.GradScaler` | Precisión mixta (FP16) en GPU. |
| `torch.cuda.max_memory_allocated()` | Memoria de GPU usada por el entrenamiento. |
| `numba.cuda.jit`, `cuda.blockIdx`, `cuda.threadIdx` | Kernels propios y cálculo del índice global de cada hilo. |
| `cuda.shared.array`, `cuda.syncthreads()` | Memoria compartida por bloque y sincronización de sus hilos. |
| `cuda.to_device`, `cuda.device_array`, `copy_to_host` | Transferencias explícitas CPU ↔ GPU. |
| `numba.njit(parallel=True)`, `prange` | Versión multinúcleo en CPU del mismo algoritmo, para comparar. |
| `concurrent.futures.ProcessPoolExecutor` | Preprocesamiento de imágenes en varios procesos de CPU. |
| `nvidia-smi`, `psutil` | Registro del uso de GPU y de cada núcleo de la CPU. |

## 8. Conclusiones

**[COMPLETAR]**

- La preparación de datos en varios procesos de CPU logró un speedup de ___× con ___ procesos, con
  resultados idénticos a la versión original.
- Entrenar la CNN 2 en la GPU con CUDA logró un speedup de ___× sobre la CPU secuencial y de ___× sobre
  la CPU con todos sus hilos, con el mismo accuracy (___ en test).
- Los kernels CUDA propios fueron ___× más rápidos que el mismo algoritmo secuencial en CPU, pero ___×
  más lentos que cuDNN: escribir el kernel es fácil, optimizarlo al nivel de una librería no lo es.
- La separación CPU para archivos / GPU para matemática es la que da el mejor tiempo total, porque cada
  procesador recibe el tipo de paralelismo para el que fue diseñado y la transferencia entre ambos se
  hace una sola vez.
