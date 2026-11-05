# Laboratorio 3 - Deep Learning

## Autores
Nelson Escalante
Roberto Najera
Diego Lopez

## Datos

Los datos (dataset crudo y arrays procesados) no se versionan en git por peso: el dataset de Kaggle
pesa ~1GB y los `.npy` procesados otro tanto. Cada integrante los genera localmente con dos
comandos, desde la raíz del repo:

```bash
pip install -r requirements.txt
python src/preprocessing.py
```

`src/dataset.py` se encarga de descargar el dataset de Kaggle (vía `kagglehub`) y dejarlo en
`data/asl_alphabet_train/asl_alphabet_train/<clase>/*.jpg` con las 29 clases (A-Z, del, nothing,
space). Es idempotente: si el dataset ya está completo no descarga nada, y si falta alguna clase
solo copia esa. Después, `src/preprocessing.py` genera
`data/processed/{X,y}_{train,val,test}.npy` y `classes.csv`.

El script usa `SEED=42` fijo, así que el split y la submuestra (700 img/clase) salen igual para
todos sin necesidad de compartir los archivos generados.
