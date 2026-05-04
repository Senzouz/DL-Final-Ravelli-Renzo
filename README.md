# DL-Final-Ravelli-Renzo

Proyecto final del curso **Advanced Deep Learning**.
Detección de **neumonía pediátrica** en radiografías de tórax mediante una **CNN propia** comparada contra dos arquitecturas de **transfer learning** (ResNet50 y EfficientNetB0) en PyTorch.

---

## Problema y objetivo

La neumonía es una de las principales causas de mortalidad infantil. La interpretación de radiografías es operador-dependiente y en zonas con escasez de radiólogos retrasa el diagnóstico. El objetivo es entrenar un clasificador binario (NORMAL vs PNEUMONIA) con **alta sensibilidad (recall)** que pueda servir como segunda opinión y triage automático.

## Dataset

- **Fuente:** Kaggle — `paultimothymooney/chest-xray-pneumonia`
- **Tamaño:** ~5,863 radiografías de tórax pediátricas
- **Clases:** `NORMAL` vs `PNEUMONIA`
- **Splits:** train / val / test (la val original tiene solo 16 imágenes → re-split estratificado 85/15 desde train)
- **Desbalance:** ~3:1 a favor de PNEUMONIA → uso de pesos por clase en la pérdida

> El dataset **NO se incluye** en el repositorio. Se descarga vía Kaggle API al ejecutar el notebook.

## Modelos comparados

| Modelo            | Tipo                       | Parámetros aprox. |
|-------------------|----------------------------|-------------------|
| CNN custom        | Baseline desde cero        | ~1 M              |
| ResNet50          | Transfer learning ImageNet | ~25 M             |
| EfficientNet-B0   | Transfer learning ImageNet | ~5 M              |

Los modelos de TL se entrenan en dos fases: *feature extraction* (backbone congelado) + *fine-tuning* de las últimas capas con learning rate bajo.

## Cómo ejecutar (Colab)

1. Abrir [`notebooks/final_project.ipynb`](notebooks/final_project.ipynb) en Google Colab.
2. `Runtime → Change runtime type → GPU (T4)`.
3. `Runtime → Run all`. Cuando se pida, subir tu `kaggle.json` (Kaggle → Account → Create New API Token).
4. Tiempo total estimado: 35–55 min en T4.

## Estructura del repositorio

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── final_project.ipynb     # notebook principal ejecutable end-to-end
├── src/                        # código auxiliar reutilizable
│   ├── data.py                 # descarga, splits, DataLoaders
│   ├── models.py               # build_cnn_custom / build_resnet50 / build_efficientnet
│   ├── train.py                # bucle de entrenamiento + early stopping
│   └── evaluate.py             # métricas, ROC, matriz de confusión, Grad-CAM
├── data/                       # SOLO instrucciones; los datos no se versionan
├── results/                    # tabla de métricas y artefactos
├── figures/                    # gráficos generados
└── report/                     # reporte final (PDF o markdown)
```

## Métricas reportadas

Sobre el test set: **Accuracy, Precision, Recall, F1, AUC-ROC y AUC-PR**, más **matriz de confusión** y **curva ROC superpuesta** de los tres modelos. Se prioriza recall por el contexto clínico.

## Interpretabilidad

Se aplica **Grad-CAM** sobre el mejor modelo para visualizar qué regiones del pulmón motivan la predicción, evidenciando que el modelo atiende a regiones plausibles desde el punto de vista clínico.

## Video

🎥 *Pendiente de subir a YouTube (≤ 5 min).*

## Autor

**Renzo Ravelli** — Especialización en Deep Learning, 2026.
