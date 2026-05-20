# 🧠 Pesos del modelo entrenado

Los pesos del modelo **no están incluidos en el repositorio** (archivo ~100 MB).

## Descarga

Los pesos están disponibles en Google Drive:

**[→ Descargar best_resnet50.pth](https://drive.google.com/drive/folders/1agbyv68Jck9VaQZouOh22eTBIsENsVY1?usp=drive_link)**

## Archivos necesarios

Descargar y colocar en `models/checkpoints/`:

| Archivo | Tamaño aprox. | Descripción |
|---|---|---|
| `best_resnet50.pth` | ~100 MB | Pesos del mejor modelo (mejor val_acc) |
| `config_modelo.json` | <1 KB | Configuración: clases, img_size, normalización |

## Estructura del config_modelo.json

```json
{
  "arquitectura": "resnet50",
  "n_classes": 8,
  "class_names": ["Corn_(maize)___Common_rust_", "..."],
  "img_size": 224,
  "model_path": "models/checkpoints/best_resnet50.pth",
  "target_layer": "layer4",
  "imagenet_mean": [0.485, 0.456, 0.406],
  "imagenet_std": [0.229, 0.224, 0.225]
}
```

## Reproducir el entrenamiento desde cero

1. Descargar dataset (ver `data/README_data.md`)
2. Ejecutar `notebooks/01_eda.ipynb` en Google Colab
3. Ejecutar `notebooks/02_cnn_training.ipynb` — genera `best_resnet50.pth` automáticamente
4. Los pesos quedan guardados en `Google Drive/IA_EAFIT/artifacts/`

## Métricas del modelo guardado

*(Obtenidas tras completar fase2_cnn_training.ipynb)*

- **Accuracy (test):** 0.9981 (99.81%)
- **F1-macro (test):** 0.9952  
- **AUC-ROC (test):** 1.0000
- **Épocas entrenadas:** 20/20 (early stopping con paciencia=5)
- **Mejor val_acc:** 0.9983
- **Hardware:** Google Colab GPU T4
