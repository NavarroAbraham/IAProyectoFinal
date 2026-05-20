# 🌿 PlantAI — Clasificación de enfermedades en plantas

> **Proyecto Final · Inteligencia Artificial · EAFIT 2026-1**  
> CNN ResNet-50 + Grad-CAM++ + LLM (Groq) para diagnóstico automático de enfermedades en cultivos

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Descripción

Sistema end-to-end que clasifica enfermedades en hojas de plantas usando visión computacional e inteligencia artificial explicable:

1. **CNN (ResNet-50)** — Fine-tuning sobre PlantVillage para clasificar 8 clases (4 cultivos × sana/enferma)
2. **Grad-CAM++** — Mapa de calor que muestra qué zonas de la hoja activaron la predicción
3. **LLM (LLaMA-3.1-8b vía Groq)** — Explicación en lenguaje natural en modo técnico o para agricultores

**Pregunta de investigación:** ¿Puede un modelo CNN + Grad-CAM + LLM clasificar enfermedades en plantas con alta precisión y generar explicaciones comprensibles para agricultores no técnicos?

---

## 🏗️ Arquitectura del sistema

```
Imagen de hoja
      ↓
ResNet-50 (clasificación)
      ↓
Grad-CAM++ (mapa de activación) → Descripción textual de zona
      ↓
LLM - LLaMA-3.1-8b (Groq) → Explicación en lenguaje natural
      ↓
Streamlit App (interfaz web)
```

---

## 📁 Estructura del repositorio

```
proyecto-ia-eafit/
├── README.md                        # Este archivo
├── requirements.txt                 # Dependencias del proyecto
│
├── docs/
│   └── informe_final.pdf            # Informe LaTeX compilado (entrega)
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Fase 1: EDA y preparación del dataset
│   ├── 02_cnn_training.ipynb        # Fase 2: ResNet-50 fine-tuning y evaluación
│   ├── 03_gradcam.ipynb             # Fase 3: Grad-CAM++ y visualizaciones
│   └── 04_llm_groq.ipynb            # Fase 4: Integración LLM con Groq
│
├── src/
│   ├── models/
│   │   └── resnet50.py              # Definición de la arquitectura CNN
│   ├── gradcam/
│   │   └── explain.py               # Funciones de Grad-CAM++ y descripción
│   └── llm/
│       └── prompts.py               # Prompts del sistema (agrónomo / agricultor)
│
├── app/
│   └── main.py                      # Aplicación Streamlit (demo interactiva)
│
├── data/
│   └── README_data.md               # Instrucciones para descargar el dataset
│
└── models/
    └── checkpoints/
        └── README_model.md          # Link a los pesos del modelo en Google Drive
```

---

## ⚙️ Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/usuario/proyecto-ia-eafit.git
cd proyecto-ia-eafit
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

> **GPU:** Para usar GPU instala PyTorch con CUDA desde [pytorch.org](https://pytorch.org/get-started/locally/).  
> La app funciona con CPU también, pero la inferencia es más lenta (~3-5s por imagen).

### 3. Descargar el dataset

Ver instrucciones en [`data/README_data.md`](data/README_data.md).  
O descargar directamente: [kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

### 4. Descargar el modelo entrenado

Ver instrucciones en [`models/checkpoints/README_model.md`](models/checkpoints/README_model.md).  
Los pesos (`best_resnet50.pth` y `config_modelo.json`) deben quedar en `models/checkpoints/`.

### 5. Configurar la API Key de Groq

```bash
# Opción A: variable de entorno
export GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx

# Opción B: archivo .env en la raíz del proyecto
echo "GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx" > .env
```

Obtener API key gratuita en [console.groq.com](https://console.groq.com).

### 6. Ejecutar la aplicación Streamlit

```bash
streamlit run app/main.py
```

Abrir en el navegador: **http://localhost:8501**

---

## 🚀 Demo en Streamlit Cloud

La app está desplegada en:  
**[→ plantai-eafit.streamlit.app](https://plantai-eafit.streamlit.app)** *(reemplazar con URL real)*

> El modelo se carga desde Google Drive automáticamente en el deploy.  
> Requiere ingresar la Groq API Key en el panel lateral de la app.

---

## 📓 Ejecutar los notebooks

Todos los notebooks están diseñados para **Google Colab** con Google Drive como almacenamiento persistente.

| Notebook | Descripción | Tiempo aprox. |
|---|---|---|
| `01_eda.ipynb` | Descarga PlantVillage, EDA completo, split 70/15/15 | ~15 min |
| `02_cnn_training.ipynb` | Fine-tuning ResNet-50 (20 épocas, GPU T4) | ~45-60 min |
| `03_gradcam.ipynb` | Grad-CAM++ sobre el modelo entrenado | ~10 min |
| `04_llm_groq.ipynb` | Explicaciones LLM con Groq API | ~10 min |

**Orden de ejecución:** 01 → 02 → 03 → 04  
Cada notebook carga automáticamente los artefactos del anterior desde `Google Drive/IA_EAFIT/`.

---

## 📊 Resultados

*(Completar con valores reales tras el entrenamiento)*

| Modelo | Accuracy | F1-macro | AUC-ROC |
|---|---|---|---|
| Baseline (clase mayoritaria) | X.XX | X.XX | 0.500 |
| ResNet-50 Fase A (solo FC) | X.XX | X.XX | X.XX |
| **ResNet-50 Fase A+B (mejor)** | **X.XX** | **X.XX** | **X.XX** |

**Clases utilizadas:** 8 (Tomato, Potato, Corn, Pepper × sana/enferma)  
**Dataset:** ~[N] imágenes tras split 70/15/15

---

## 🛠️ Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Deep Learning | PyTorch 2.1 + torchvision |
| Grad-CAM | pytorch-grad-cam |
| LLM | Groq API — LLaMA-3.1-8b-instant |
| App web | Streamlit 1.35+ |
| Entrenamiento | Google Colab GPU T4 |
| Persistencia | Google Drive |
| Dataset | Kaggle PlantVillage |

---

## 👥 Equipo y contribuciones

| Integrante | Correo | Contribución principal |
|---|---|---|
| Nombre Completo 1 | correo1@eafit.edu.co | EDA, preprocesamiento, visualizaciones |
| Nombre Completo 2 | correo2@eafit.edu.co | CNN, Grad-CAM, LLM, app Streamlit |

---

## 📚 Referencias

- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV. https://arxiv.org/abs/1610.02391  
- Chattopadhay et al. (2018). *Grad-CAM++: Generalized Gradient-based Visual Explanations.* WACV. https://arxiv.org/abs/1710.11063  
- He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.  
- Hughes & Salathé (2015). *An open access repository of images on plant health.* arXiv:1511.08060  
- Chase, H. (2022). LangChain. https://github.com/langchain-ai/langchain  

---

## 📄 Licencia

MIT License — ver [LICENSE](LICENSE)
