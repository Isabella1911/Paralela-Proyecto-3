# Selección de modelos y plan de procesamiento

Documento de planeación para el avance. Los modelos se implementan en la entrega final; aquí se
fija arquitectura e hiperparámetros iniciales.

Framework: TensorFlow / Keras. El módulo de preprocesamiento ya incluye el conversor
`to_tensorflow_dataset`.

## CNN 1: arquitectura sencilla

Dos bloques `Conv2D + MaxPooling2D`, seguidos de `Flatten` y `Dense` de salida.

```text
Conv2D(32, 3x3, activation='relu') -> MaxPooling2D(2x2)
Conv2D(64, 3x3, activation='relu') -> MaxPooling2D(2x2)
Flatten() -> Dense(128, activation='relu') -> Dropout(0.3) -> Dense(29, activation='softmax')
```

Activación ReLU en capas ocultas, softmax en la salida para las 29 clases. Optimizador Adam,
tasa de aprendizaje 0.001. Loss: `sparse_categorical_crossentropy`.

## CNN 2: arquitectura más profunda

Tres bloques `Conv2D + BatchNormalization + MaxPooling2D`, seguidos de dos capas `Dense`.

```text
Conv2D(32, 3x3) -> BatchNorm -> MaxPooling2D
Conv2D(64, 3x3) -> BatchNorm -> MaxPooling2D
Conv2D(128, 3x3) -> BatchNorm -> MaxPooling2D
Flatten() -> Dense(256, relu) -> Dropout(0.4) -> Dense(128, relu) -> Dropout(0.3) -> Dense(29, softmax)
```

Misma activación ReLU/softmax. Optimizador Adam, tasa de aprendizaje 0.0005. Incluye
`BatchNormalization` y mayor dropout que CNN 1.

## Red fully connected

`Flatten()` de la imagen 64x64x3 a un vector de 12288 posiciones, seguido de capas densas:

```text
Dense(512, relu) -> Dropout(0.3) -> Dense(128, relu) -> Dense(29, softmax)
```

Sin capas convolucionales. Sirve como punto de comparación frente a las dos CNN.

## Modelo clásico: Random Forest

Random Forest sobre las imágenes aplanadas (mismo input que la red fully connected). Se comparan
`n_estimators` en 100 y 300, y `max_depth` en None, 20 y 40. Métrica de selección: accuracy en
validación, con matriz de confusión.

## Plan de procesamiento de imágenes

Resolución final: 64x64 píxeles, ya fijada en el preprocesamiento.

Normalización: división por 255, rango [0, 1] en float32, aplicada igual en los cuatro modelos.

Image augmentation: rotaciones de hasta 15 grados, zoom leve y variación de brillo/contraste.
Flip horizontal y vertical quedan excluidos: invierten la lateralidad de la seña y pueden convertir
una letra en otra distinta. Se implementa con `RandomRotation`, `RandomZoom` y `RandomBrightness`
de Keras, o alternativamente con `ImageDataGenerator`.

Métricas de evaluación: accuracy, precision y recall en promedio macro, y matriz de confusión.

Estrategia de comparación: mismo split train/val/test y mismas métricas para los cuatro modelos
(CNN 1, CNN 2, red fully connected, Random Forest). Selección del mejor modelo por accuracy en
test, usando la matriz de confusión como criterio adicional para las letras visualmente similares
(M/N/S, U/V/R).
