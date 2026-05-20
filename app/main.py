"""
app/main.py
Aplicación Streamlit — Clasificación de enfermedades en plantas
CNN ResNet-50 + Grad-CAM++ + LLM (Groq)

Proyecto Final IA · EAFIT 2026-1
"""

import io
import json
import time
import urllib.request
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantAI — Diagnóstico de enfermedades",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes y rutas
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
CKPT_DIR    = BASE_DIR / "models" / "checkpoints"
MODEL_PATH  = CKPT_DIR / "best_resnet50.pth"
CONFIG_PATH = CKPT_DIR / "config_modelo.json"

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Hugging Face Hub — repositorio con los pesos del modelo ──────────────────
# Después de entrenar, subir los archivos con:
#   pip install huggingface_hub
#   python -c "
#     from huggingface_hub import HfApi
#     api = HfApi()
#     api.upload_file(path_or_fileobj='models/checkpoints/best_resnet50.pth',
#                     path_in_repo='best_resnet50.pth',
#                     repo_id='TU_USUARIO/plantai-eafit',
#                     repo_type='model')
#     api.upload_file(path_or_fileobj='models/checkpoints/config_modelo.json',
#                     path_in_repo='config_modelo.json',
#                     repo_id='TU_USUARIO/plantai-eafit',
#                     repo_type='model')
#   "
HF_REPO_ID = "https://huggingface.co/Aenavarro/plantai-eafit/tree/main" 
HF_BASE    = f"https://huggingface.co/{HF_REPO_ID}/resolve/main"
HF_FILES   = {
    MODEL_PATH : f"{HF_BASE}/best_resnet50.pth",
    CONFIG_PATH: f"{HF_BASE}/config_modelo.json",
}

# Clases por defecto (se sobreescriben con config_modelo.json)
DEFAULT_CLASSES = [
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Late_blight",
    "Tomato___healthy",
]


# ─────────────────────────────────────────────────────────────────────────────
# Descarga automática desde Hugging Face
# ─────────────────────────────────────────────────────────────────────────────
def _download_file(url: str, dest: Path, label: str) -> bool:
    """
    Descarga un archivo desde `url` a `dest` con barra de progreso en Streamlit.
    Retorna True si fue exitoso, False si hubo error.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        progress_text = f"Descargando {label}…"
        bar = st.progress(0, text=progress_text)

        def _reporthook(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(block_num * block_size / total_size, 1.0)
                bar.progress(pct, text=f"{progress_text} ({pct*100:.0f}%)")

        urllib.request.urlretrieve(url, dest, _reporthook)
        bar.progress(1.0, text=f"✅ {label} descargado")
        time.sleep(0.4)
        bar.empty()
        return True
    except Exception as e:
        st.error(f"❌ Error descargando {label}: {e}")
        return False


@st.cache_resource(show_spinner=False)
def ensure_model_files() -> bool:
    """
    Verifica que los archivos del modelo existan localmente.
    Si no, los descarga desde Hugging Face Hub.
    Retorna True si ambos archivos están disponibles al final.
    """
    hf_configured = HF_REPO_ID != "TU_USUARIO/plantai-eafit"
    all_ok = True

    for local_path, url in HF_FILES.items():
        if local_path.exists():
            continue  # ya está en disco — no descargar

        label = local_path.name
        if not hf_configured:
            # HF no configurado aún — mostrar instrucciones en lugar de error
            st.warning(
                f"**{label} no encontrado.** "
                f"Para el deploy en Streamlit Cloud:\n"
                f"1. Entrena el modelo con `notebooks/02_cnn_training.ipynb`\n"
                f"2. Sube los pesos a Hugging Face Hub (ver instrucciones en `app/main.py`)\n"
                f"3. Actualiza `HF_REPO_ID` en `app/main.py` con tu usuario/repo\n\n"
                f"Mientras tanto la app corre en **modo demo** con pesos aleatorios."
            )
            all_ok = False
            continue

        with st.spinner(f"Primera carga: descargando {label} desde Hugging Face…"):
            ok = _download_file(url, local_path, label)
        if not ok:
            all_ok = False

    return all_ok

# ─────────────────────────────────────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Reducir padding top del main */
.block-container { padding-top: 1.5rem; }

/* Tarjetas de métricas */
div[data-testid="metric-container"] {
    background: var(--background-color, #fafafa);
    border: 1px solid rgba(49,51,63,0.1);
    border-radius: 10px;
    padding: 12px 16px;
}

/* Badge de estado */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.badge-healthy  { background: #d4edda; color: #155724; }
.badge-diseased { background: #f8d7da; color: #721c24; }
.badge-warning  { background: #fff3cd; color: #856404; }

/* Caja de explicación LLM */
.llm-box {
    background: linear-gradient(135deg, #f0f9f4 0%, #e8f5e9 100%);
    border-left: 4px solid #2e7d32;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.7;
}
.llm-box-farmer {
    background: linear-gradient(135deg, #fffde7 0%, #fff9c4 100%);
    border-left: 4px solid #f9a825;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.7;
}
.section-header {
    font-size: 16px;
    font-weight: 600;
    margin: 16px 0 8px 0;
    color: #1a1a2e;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de etiquetas
# ─────────────────────────────────────────────────────────────────────────────
def label_legible(class_name: str) -> tuple[str, str]:
    """'Tomato___Late_blight' → ('Tomato', 'Late blight')"""
    parts = class_name.split("___")
    planta = parts[0].replace("_", " ").strip()
    estado = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Desconocido"
    return planta, estado


def is_healthy(class_name: str) -> bool:
    return "healthy" in class_name.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Carga del modelo (cacheado)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo CNN…")
def load_model_and_config():
    """
    Carga ResNet-50 y la configuración de clases.
    Si los archivos no existen localmente los descarga desde Hugging Face Hub.
    Cacheado entre sesiones con st.cache_resource.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Asegurar que los archivos existen (descarga si hace falta)
    files_ok = ensure_model_files()

    # 2. Cargar config de clases
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        class_names = cfg["class_names"]
    else:
        class_names = DEFAULT_CLASSES

    n_classes = len(class_names)

    # 3. Construir arquitectura idéntica a la de Fase 2
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, n_classes),
    )

    # 4. Cargar pesos
    if MODEL_PATH.exists():
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        model_loaded = True
    else:
        model_loaded = False   # modo demo — pesos aleatorios

    model = model.to(device)
    model.eval()

    # 5. Grad-CAM++ sobre layer4
    cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])

    return model, cam, class_names, device, model_loaded


# ─────────────────────────────────────────────────────────────────────────────
# Transform de inferencia
# ─────────────────────────────────────────────────────────────────────────────
transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia + Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
def predict_with_gradcam(
    img_pil: Image.Image,
    model,
    cam,
    class_names: list,
    device,
    alpha: float = 0.45,
) -> dict:
    """Inferencia completa: predicción + Grad-CAM++ + descripción de zona."""
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_np = np.array(img_resized) / 255.0

    tensor = transform_eval(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    targets = [ClassifierOutputTarget(pred_idx)]

    cam_mask = cam(input_tensor=tensor, targets=targets)[0]
    cam_mask = np.clip(cam_mask, 0, 1)

    overlay_np = show_cam_on_image(
        img_np.astype(np.float32),
        cam_mask,
        use_rgb=True,
        colormap=cv2.COLORMAP_JET,
        image_weight=1 - alpha,
    )
    overlay_pil = Image.fromarray(overlay_np)

    # Descripción textual de la zona activa (para el LLM)
    zone_desc = describe_activation_zone(cam_mask)

    top3 = [
        {"class": class_names[i], "prob": float(probs[i])}
        for i in probs.argsort()[::-1][:3]
    ]

    return {
        "pred_idx": pred_idx,
        "class_name": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": probs,
        "top3": top3,
        "cam_mask": cam_mask,
        "overlay": overlay_pil,
        "original": img_resized,
        "zone_description": zone_desc,
    }


def describe_activation_zone(cam_mask: np.ndarray, threshold: float = 0.5) -> str:
    """Traduce el mapa Grad-CAM a descripción textual para el LLM."""
    h, w = cam_mask.shape
    rows = ["superior", "central", "inferior"]
    cols = ["izquierda", "centro", "derecha"]

    scores = {}
    for r_i, rn in enumerate(rows):
        for c_i, cn in enumerate(cols):
            r0, r1 = r_i * h // 3, (r_i + 1) * h // 3
            c0, c1 = c_i * w // 3, (c_i + 1) * w // 3
            scores[f"{rn}-{cn}"] = float(cam_mask[r0:r1, c0:c1].mean())

    active = sorted(
        [(k, v) for k, v in scores.items() if v >= threshold],
        key=lambda x: -x[1],
    )
    if not active:
        active = sorted(scores.items(), key=lambda x: -x[1])[:2]

    max_act = cam_mask.max()
    intensidad = "alta" if max_act > 0.75 else ("moderada" if max_act > 0.45 else "difusa")
    zonas = ", ".join(z.replace("-", " ") for z, _ in active[:3])
    pct = 100 * (cam_mask >= threshold).mean()

    return (
        f"Las regiones de mayor activación (Grad-CAM++) se concentran en la zona {zonas} "
        f"de la imagen, con una intensidad {intensidad}. "
        f"Aproximadamente el {pct:.0f}% del área de la hoja supera el umbral de activación."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM con Groq
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_AGRONOMO = """Eres un experto en fitopatología y agricultura de precisión.
Analiza predicciones de un modelo CNN (ResNet-50) combinadas con mapas Grad-CAM++.
Responde SIEMPRE en español, en exactamente 3 párrafos cortos:
1. Diagnóstico: qué indica la predicción
2. Evidencia visual: qué sugieren las zonas de activación
3. Implicaciones: riesgo agronómico y próximos pasos
Máximo 120 palabras."""

SYSTEM_AGRICULTOR = """Eres un asistente agrícola que explica diagnósticos de IA a agricultores.
Usa lenguaje sencillo, sin tecnicismos. Responde en español con:
- Una oración sobre qué encontró la IA
- Una oración sobre dónde se ven los síntomas
- Dos recomendaciones prácticas concretas
Máximo 80 palabras. Tono amable y directo."""


@st.cache_data(show_spinner=False, ttl=3600)
def call_llm(
    pred_label: str,
    true_label: str,
    confidence: float,
    zone_desc: str,
    top3_str: str,
    mode: str,
    api_key: str,
) -> str:
    """Llama a Groq LLM. Cacheado por parámetros para no repetir llamadas."""
    try:
        from groq import Groq, RateLimitError

        client = Groq(api_key=api_key)
        system = SYSTEM_AGRONOMO if mode == "agronomo" else SYSTEM_AGRICULTOR
        user_msg = (
            f"Cultivo analizado: {true_label}\n"
            f"Predicción del modelo: {pred_label} ({confidence*100:.1f}%)\n"
            f"Top 3:\n{top3_str}\n\n"
            f"Análisis Grad-CAM++:\n{zone_desc}\n\n"
            "Por favor analiza este caso."
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=300,
                    temperature=0.4,
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError:
                time.sleep(2 ** attempt)
        return "⚠️ Rate limit alcanzado. Intenta de nuevo en unos segundos."
    except ImportError:
        return "⚠️ groq no instalado. Ejecuta: pip install groq"
    except Exception as e:
        return f"⚠️ Error al contactar Groq: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Figura de probabilidades (barras horizontales)
# ─────────────────────────────────────────────────────────────────────────────
def plot_probabilities(all_probs: np.ndarray, class_names: list, top_n: int = 8) -> plt.Figure:
    indices = all_probs.argsort()[::-1][:top_n]
    labels  = [label_legible(class_names[i])[1][:22] for i in indices]
    values  = [all_probs[i] * 100 for i in indices]
    colors  = ["#2e7d32" if is_healthy(class_names[i]) else "#c62828" for i in indices]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Probabilidad (%)", fontsize=9)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    from matplotlib.patches import Patch
    legend = [Patch(color="#2e7d32", label="Sana"),
              Patch(color="#c62828", label="Enferma")]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figura Grad-CAM (original | heatmap | overlay)
# ─────────────────────────────────────────────────────────────────────────────
def plot_gradcam_trio(original: Image.Image, cam_mask: np.ndarray, overlay: Image.Image) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(original);  axes[0].set_title("Original",    fontsize=10); axes[0].axis("off")
    im = axes[1].imshow(cam_mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Mapa Grad-CAM++", fontsize=10); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay);   axes[2].set_title("Superposición", fontsize=10); axes[2].axis("off")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/EAFIT_Logo.svg/320px-EAFIT_Logo.svg.png",
             width=160)
    st.markdown("## 🌿 PlantAI")
    st.markdown("**Diagnóstico de enfermedades en plantas**  \nCNN + Grad-CAM++ + LLM")
    st.divider()

    st.markdown("### ⚙️ Configuración")

    # API Key de Groq
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_xxxxxxxxxxxxxxxx",
        help="Obtén tu clave gratuita en console.groq.com",
    )

    # Parámetro de transparencia del heatmap
    alpha = st.slider(
        "Transparencia Grad-CAM",
        min_value=0.1, max_value=0.9, value=0.45, step=0.05,
        help="Controla qué tanto se ve el heatmap sobre la imagen original",
    )

    # Modo de explicación LLM
    llm_mode = st.radio(
        "Modo de explicación",
        options=["agronomo", "agricultor"],
        format_func=lambda x: "🔬 Técnico (agrónomo)" if x == "agronomo" else "🌱 Simple (agricultor)",
        help="Cambia el nivel técnico de la explicación del LLM",
    )

    st.divider()
    st.markdown("### 📋 Sobre el proyecto")
    st.markdown("""
    **Dataset:** PlantVillage (Kaggle)  
    **Modelo:** ResNet-50 fine-tuning  
    **Explicabilidad:** Grad-CAM++  
    **LLM:** LLaMA-3.1-8b (Groq)  
    **Curso:** IA 2026-1 · EAFIT
    """)
    st.markdown("[📁 Repositorio GitHub](https://github.com/usuario/proyecto-ia-eafit)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — cargar modelo
# ─────────────────────────────────────────────────────────────────────────────
model, cam_obj, class_names, device, model_loaded = load_model_and_config()

if not model_loaded:
    st.info(
        "**🧪 Modo demo** — pesos no encontrados. Las predicciones son aleatorias. "
        "Sube `best_resnet50.pth` a Hugging Face y actualiza `HF_REPO_ID` en `app/main.py`.",
        icon="ℹ️",
    )

# ─────────────────────────────────────────────────────────────────────────────
# TABS principales
# ─────────────────────────────────────────────────────────────────────────────
tab_demo, tab_batch, tab_about = st.tabs([
    "🔍 Diagnóstico individual",
    "📊 Análisis por lotes",
    "ℹ️ Acerca del sistema",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGNÓSTICO INDIVIDUAL
# ════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("## Diagnóstico individual de planta")
    st.markdown("Sube una imagen de una hoja de planta y el sistema clasificará la enfermedad, "
                "visualizará las zonas de activación y generará una explicación en lenguaje natural.")

    # Upload
    uploaded = st.file_uploader(
        "Subir imagen de hoja",
        type=["jpg", "jpeg", "png"],
        help="Formatos: JPG, JPEG, PNG. Tamaño recomendado: 256×256 px o mayor.",
    )

    if uploaded is not None:
        img_pil = Image.open(uploaded).convert("RGB")

        col_img, col_meta = st.columns([1, 1])
        with col_img:
            st.image(img_pil, caption="Imagen cargada", use_column_width=True)
        with col_meta:
            w, h = img_pil.size
            st.markdown(f"**Archivo:** `{uploaded.name}`")
            st.markdown(f"**Resolución:** {w} × {h} px")
            st.markdown(f"**Tamaño:** {uploaded.size / 1024:.1f} KB")

        st.divider()

        # ── Inferencia ────────────────────────────────────────────────────────
        with st.spinner("Analizando imagen con CNN + Grad-CAM++…"):
            result = predict_with_gradcam(
                img_pil, model, cam_obj, class_names, device, alpha
            )

        planta, estado = label_legible(result["class_name"])
        healthy = is_healthy(result["class_name"])
        conf    = result["confidence"]

        # ── Banner de resultado ───────────────────────────────────────────────
        badge_class = "badge-healthy" if healthy else "badge-diseased"
        badge_text  = "✅ SANA" if healthy else "⚠️ ENFERMA"
        st.markdown(
            f'<div style="font-size:22px;font-weight:700;margin:8px 0">'
            f'{planta} &nbsp;<span class="status-badge {badge_class}">{badge_text}</span>'
            f'</div>'
            f'<div style="font-size:16px;color:#555;margin-bottom:12px">'
            f'Condición detectada: <strong>{estado}</strong></div>',
            unsafe_allow_html=True,
        )

        # ── Métricas ──────────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Confianza", f"{conf*100:.1f}%")
        m2.metric("Cultivo", planta)
        m3.metric("Estado", estado[:20])

        st.divider()

        # ── Grad-CAM ──────────────────────────────────────────────────────────
        st.markdown("### 🔥 Análisis Grad-CAM++")
        st.markdown("Las zonas en **rojo/naranja** son las regiones que el modelo consideró "
                    "más importantes para tomar la decisión.")

        fig_cam = plot_gradcam_trio(result["original"], result["cam_mask"], result["overlay"])
        st.pyplot(fig_cam, use_container_width=True)

        with st.expander("Ver descripción de zonas activas"):
            st.info(result["zone_description"])

        st.divider()

        # ── Probabilidades ────────────────────────────────────────────────────
        st.markdown("### 📊 Probabilidades por clase")
        col_bar, col_top3 = st.columns([2, 1])
        with col_bar:
            fig_probs = plot_probabilities(result["all_probs"], class_names)
            st.pyplot(fig_probs, use_container_width=True)
        with col_top3:
            st.markdown("**Top 3 predicciones**")
            for i, p in enumerate(result["top3"]):
                p_name, p_estado = label_legible(p["class"])
                color = "#2e7d32" if is_healthy(p["class"]) else "#c62828"
                st.markdown(
                    f'**{i+1}.** {p_name} — {p_estado}  \n'
                    f'<span style="color:{color};font-weight:600">{p["prob"]*100:.1f}%</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

        st.divider()

        # ── Explicación LLM ───────────────────────────────────────────────────
        st.markdown("### 🤖 Explicación en lenguaje natural")

        if not groq_key:
            st.warning("🔑 Ingresa tu Groq API Key en el panel lateral para obtener la explicación del LLM.")
        else:
            top3_str = "\n".join(
                f"  {i+1}. {label_legible(p['class'])[1]} ({p['prob']*100:.1f}%)"
                for i, p in enumerate(result["top3"])
            )
            with st.spinner("Consultando LLM (Groq / LLaMA-3)…"):
                explanation = call_llm(
                    pred_label=f"{planta} / {estado}",
                    true_label=planta,
                    confidence=conf,
                    zone_desc=result["zone_description"],
                    top3_str=top3_str,
                    mode=llm_mode,
                    api_key=groq_key,
                )

            icon  = "🔬" if llm_mode == "agronomo" else "🌱"
            label = "Modo técnico (agrónomo)" if llm_mode == "agronomo" else "Modo simple (agricultor)"
            box_class = "llm-box" if llm_mode == "agronomo" else "llm-box-farmer"

            st.markdown(f"**{icon} {label}**")
            st.markdown(
                f'<div class="{box_class}">{explanation}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Descarga de resultados ────────────────────────────────────────────
        st.markdown("### 💾 Descargar resultados")
        col_d1, col_d2, col_d3 = st.columns(3)

        # Overlay PNG
        buf_overlay = io.BytesIO()
        result["overlay"].save(buf_overlay, format="PNG")
        col_d1.download_button(
            "⬇️ Grad-CAM overlay",
            data=buf_overlay.getvalue(),
            file_name=f"gradcam_{uploaded.name}",
            mime="image/png",
        )

        # JSON con resultado
        result_json = {
            "archivo": uploaded.name,
            "prediccion": result["class_name"],
            "confianza": round(conf, 4),
            "top3": [{"clase": p["class"], "prob": round(p["prob"], 4)} for p in result["top3"]],
            "zona_gradcam": result["zone_description"],
        }
        col_d2.download_button(
            "⬇️ Resultado JSON",
            data=json.dumps(result_json, indent=2, ensure_ascii=False),
            file_name=f"resultado_{uploaded.name.split('.')[0]}.json",
            mime="application/json",
        )

    else:
        # Pantalla de bienvenida cuando no hay imagen
        st.markdown("")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("""
            <div style="text-align:center;padding:40px;background:#f8f9fa;border-radius:16px;border:2px dashed #dee2e6">
                <div style="font-size:56px;margin-bottom:16px">🌿</div>
                <div style="font-size:18px;font-weight:600;margin-bottom:8px;color:#1a1a2e">
                    Sube una imagen para comenzar
                </div>
                <div style="font-size:14px;color:#6c757d">
                    Formatos aceptados: JPG · JPEG · PNG<br>
                    Cultivos disponibles: Tomate · Papa · Maíz · Pimiento
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISIS POR LOTES
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("## Análisis por lotes")
    st.markdown("Sube múltiples imágenes para analizarlas todas a la vez y obtener un resumen.")

    uploaded_batch = st.file_uploader(
        "Subir múltiples imágenes",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if uploaded_batch:
        st.markdown(f"**{len(uploaded_batch)} imágenes cargadas.** Procesando…")
        progress = st.progress(0)
        results_batch = []

        for i, f in enumerate(uploaded_batch):
            img = Image.open(f).convert("RGB")
            res = predict_with_gradcam(img, model, cam_obj, class_names, device, alpha)
            planta, estado = label_legible(res["class_name"])
            results_batch.append({
                "Archivo"   : f.name,
                "Cultivo"   : planta,
                "Condición" : estado,
                "Confianza" : f"{res['confidence']*100:.1f}%",
                "Estado"    : "✅ Sana" if is_healthy(res["class_name"]) else "⚠️ Enferma",
            })
            progress.progress((i + 1) / len(uploaded_batch))

        import pandas as pd
        df = pd.DataFrame(results_batch)
        st.dataframe(df, use_container_width=True)

        n_enfermas = sum(1 for r in results_batch if "Enferma" in r["Estado"])
        n_sanas    = len(results_batch) - n_enfermas

        col_s, col_e, col_r = st.columns(3)
        col_s.metric("Plantas sanas",    n_sanas)
        col_e.metric("Plantas enfermas", n_enfermas)
        col_r.metric("Tasa de enfermedad", f"{100*n_enfermas/len(results_batch):.0f}%")

        # Descargar CSV
        csv_data = df.to_csv(index=False, encoding="utf-8")
        st.download_button(
            "⬇️ Descargar resultados CSV",
            data=csv_data,
            file_name="resultados_batch.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ACERCA DEL SISTEMA
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("## Acerca del sistema")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🧠 Arquitectura del sistema
        El sistema integra tres componentes de IA en un pipeline end-to-end:

        1. **CNN (ResNet-50)** — Clasificación de la imagen.
           Fine-tuning en 2 fases sobre el dataset PlantVillage (38 clases).
           Entrenado con PyTorch en Google Colab GPU T4.

        2. **Grad-CAM++** — Explicabilidad visual.
           Genera mapas de calor sobre `layer4` de ResNet-50 que muestran
           qué regiones de la hoja activaron la predicción.

        3. **LLM (LLaMA-3.1-8b vía Groq)** — Explicación en lenguaje natural.
           Traduce la predicción + zonas activas a texto comprensible,
           en modo técnico (agrónomo) o simple (agricultor).

        ### 📊 Dataset
        - **Fuente:** PlantVillage (Kaggle — emmarex/plantdisease)
        - **Total:** ~54,000 imágenes, 38 clases
        - **Subconjunto usado:** 8 clases (4 cultivos × sana/enferma)
        - **Split:** 70% train / 15% val / 15% test (random_state=42)
        """)

    with col2:
        st.markdown("""
        ### ⚡ Métricas del modelo
        *(completar con valores reales tras entrenamiento)*

        | Métrica | Baseline (LR) | ResNet-50 Fase A | **ResNet-50 Fase A+B** |
        |---|---|---|---|
        | Accuracy | 0.XX | 0.XX | **0.XX** |
        | F1-macro | 0.XX | 0.XX | **0.XX** |
        | AUC-ROC  | 0.XX | 0.XX | **0.XX** |

        ### 🔧 Stack tecnológico
        | Componente | Tecnología |
        |---|---|
        | Lenguaje | Python 3.10 |
        | Deep Learning | PyTorch + torchvision |
        | Grad-CAM | pytorch-grad-cam |
        | LLM | Groq API (LLaMA-3.1-8b) |
        | App | Streamlit |
        | Entrenamiento | Google Colab GPU T4 |
        | Dataset | Kaggle PlantVillage |

        ### 👥 Equipo
        | Integrante | Contribución |
        |---|---|
        | Nombre 1 | EDA, preprocesamiento, visualizaciones |
        | Nombre 2 | CNN, Grad-CAM, LLM, integración |
        """)

    st.divider()
    st.markdown("""
    ### 📚 Referencias
    - Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV.
    - Chattopadhay et al. (2018). *Grad-CAM++: Generalized Gradient-based Visual Explanations.* WACV.
    - He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
    - Hughes & Salathé (2015). *An open access repository of images on plant health.* arXiv.
    """)
