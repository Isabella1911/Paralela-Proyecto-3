"""
src/model.py
Proyecto 3 - Mejorando algoritmos con CUDA

CNN 2 del Lab 3 (el mejor modelo, guardado como models/best_cnn.keras) reescrita en
PyTorch con la misma arquitectura capa por capa.

Por qué PyTorch y no TensorFlow: desde TensorFlow 2.11 no hay soporte de GPU en
Windows nativo (solo vía WSL2). PyTorch sí trae CUDA + cuDNN en sus wheels para
Windows y Linux, así que permite medir el mismo modelo en CPU y en GPU dentro del
mismo framework, que es la única forma de que la comparación sea justa.

Equivalencias Keras -> PyTorch:
- Conv2D(f, 3x3, activation="relu")  -> Conv2d(c, f, 3) + ReLU   (padding "valid")
- BatchNormalization()               -> BatchNorm2d(eps=1e-3, momentum=0.01)
  (Keras usa momentum=0.99 sobre el promedio viejo; PyTorch lo define al revés)
- Dense(n, activation="relu")        -> Linear + ReLU
- softmax + sparse_categorical_crossentropy -> logits + CrossEntropyLoss
Las imágenes llegan en NHWC (como en el Lab 3) y se pasan a NCHW dentro del modelo.
"""

from torch import nn

NUM_CLASSES = 29


def _conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
        nn.MaxPool2d(2),
    )


class CNN2(nn.Module):
    """
    Conv(32)-BN-Pool -> Conv(64)-BN-Pool -> Conv(128)-BN-Pool
    -> Flatten -> Dense(256) -> Dropout(0.4) -> Dense(128) -> Dropout(0.3) -> Dense(29)

    Tamaños: 64 -> 62 -> 31 -> 29 -> 14 -> 12 -> 6, así que Flatten da 6*6*128 = 4608.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_nhwc):
        x = x_nhwc.permute(0, 3, 1, 2)   # NHWC -> NCHW
        return self.classifier(self.features(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
