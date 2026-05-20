# 📦 Dataset — PlantVillage

El dataset **no está incluido en el repositorio** por su tamaño (~1.5 GB).

## Descarga

### Opción A — Kaggle (recomendado)
1. Crear cuenta en [kaggle.com](https://www.kaggle.com)
2. Ir a: https://www.kaggle.com/datasets/emmarex/plantdisease
3. Clic en **Download** → descomprimir en `data/raw/`

### Opción B — Desde los notebooks (automático)
Los notebooks de Colab descargan y preparan el dataset automáticamente.
Solo necesitas subir tu `kaggle.json` (API token) cuando se solicite.

## Estructura esperada tras la descarga

```
data/
├── raw/
│   └── PlantVillage/
│       ├── Tomato___Late_blight/
│       ├── Tomato___healthy/
│       ├── Potato___Late_blight/
│       ├── Potato___healthy/
│       ├── Corn_(maize)___Common_rust_/
│       ├── Corn_(maize)___healthy/
│       ├── Pepper,_bell___Bacterial_spot/
│       └── Pepper,_bell___healthy/
└── processed/
    └── dataset_split/          # generado por 01_eda.ipynb
        ├── train/
        ├── val/
        └── test/
```

## Estadísticas del subconjunto usado

| Clase | Train | Val | Test | Total |
|---|---|---|---|---|
| Corn_(maize)___Common_rust_ | 373 | 80 | 80 | 533 |
| Corn_(maize)___healthy | 397 | 85 | 85 | 567 |
| Pepper__bell___Bacterial_spot | 333 | 71 | 71 | 475 |
| Pepper__bell___healthy | 410 | 88 | 88 | 586 |
| Potato___Late_blight | 404 | 86 | 86 | 576 |
| Potato___healthy | 409 | 88 | 88 | 585 |
| Tomato_Late_blight | 415 | 89 | 89 | 593 |
| Tomato_healthy | 409 | 88 | 88 | 585 |
| **Total** | **3,550** | **775** | **775** | **5,100** |

**Porcentaje split:** 70% train / 15% val / 15% test (random_state=42)

## Cita

Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health
to enable the development of mobile disease diagnostics. arXiv:1511.08060.
