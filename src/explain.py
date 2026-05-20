"""
src/gradcam/explain.py
Grad-CAM++ sobre ResNet-50: inferencia, mapa de activación y descripción
textual de zonas activas para el LLM.

Extraído de: notebooks/03_gradcam.ipynb
Proyecto Final IA · EAFIT 2026-1

Referencias:
    Chattopadhay et al. (2018). Grad-CAM++: Generalized Gradient-based Visual
    Explanations for Deep Convolutional Networks. WACV 2018.
    https://arxiv.org/abs/1710.11063
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# Constantes por defecto (deben coincidir con el entrenamiento)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE:      int   = 224
IMAGENET_MEAN: list  = [0.485, 0.456, 0.406]
IMAGENET_STD:  list  = [0.229, 0.224, 0.225]
DEFAULT_ALPHA: float = 0.45   # transparencia del heatmap sobre la imagen

# Transform estándar de inferencia (sin augmentation)
_transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Métodos CAM disponibles (para comparación en el informe)
CAM_METHODS = {
    "gradcam"   : GradCAM,
    "gradcam++" : GradCAMPlusPlus,   # método principal usado en el proyecto
    "eigencam"  : EigenCAM,
}


# ─────────────────────────────────────────────────────────────────────────────
# Inicialización de Grad-CAM++
# ─────────────────────────────────────────────────────────────────────────────
def build_cam(model: torch.nn.Module, method: str = "gradcam++") -> GradCAMPlusPlus:
    """
    Crea el objeto CAM sobre la capa layer4 de ResNet-50.

    ¿Por qué layer4?
        Es la última capa convolucional de ResNet-50. Combina la máxima
        semántica (qué detecta) con algo de resolución espacial (dónde).
        Su mapa de activación de 7×7 se interpola bilinealmente a 224×224.

    Args:
        model:  Modelo ResNet-50 (en eval mode, en el device correcto).
        method: 'gradcam', 'gradcam++' (default) o 'eigencam'.

    Returns:
        Objeto CAM listo para llamar con input_tensor y targets.

    Ejemplo:
        >>> cam = build_cam(model, method='gradcam++')
    """
    if method not in CAM_METHODS:
        raise ValueError(f"method debe ser uno de {list(CAM_METHODS)}. Recibido: '{method}'")
    target_layers = [model.layer4[-1]]
    return CAM_METHODS[method](model=model, target_layers=target_layers)


# ─────────────────────────────────────────────────────────────────────────────
# Descripción textual de la zona activa (input para el LLM)
# ─────────────────────────────────────────────────────────────────────────────
def describe_activation_zone(cam_mask: np.ndarray, threshold: float = 0.5) -> str:
    """
    Traduce un mapa de activación Grad-CAM a una descripción textual en español.

    Divide la imagen en una cuadrícula 3×3 (superior/central/inferior ×
    izquierda/centro/derecha) y describe las regiones con activación superior
    al umbral. Esta descripción es el contexto que recibe el LLM en la Fase 4.

    Args:
        cam_mask:  Array 2D float32 normalizado en [0, 1] (H × W).
        threshold: Umbral mínimo de activación para considerar una zona activa.
                   Por defecto 0.5 (50% de la activación máxima).

    Returns:
        String en español describiendo zonas activas, intensidad y cobertura.

    Ejemplo:
        >>> desc = describe_activation_zone(result['cam_mask'])
        >>> print(desc)
        'Las regiones de mayor activación se concentran en la zona central-centro
         de la imagen, con una intensidad alta. Aproximadamente el 38% del área...'
    """
    h, w   = cam_mask.shape
    rows   = ["superior", "central",   "inferior"]
    cols   = ["izquierda", "centro",   "derecha"]

    region_scores: dict[str, float] = {}
    for r_i, rn in enumerate(rows):
        for c_i, cn in enumerate(cols):
            r0, r1 = r_i * h // 3, (r_i + 1) * h // 3
            c0, c1 = c_i * w // 3, (c_i + 1) * w // 3
            region_scores[f"{rn}-{cn}"] = float(cam_mask[r0:r1, c0:c1].mean())

    active = sorted(
        [(k, v) for k, v in region_scores.items() if v >= threshold],
        key=lambda x: -x[1],
    )
    if not active:
        active = sorted(region_scores.items(), key=lambda x: -x[1])[:2]

    max_act = float(cam_mask.max())
    if max_act > 0.75:
        intensidad = "alta"
    elif max_act > 0.45:
        intensidad = "moderada"
    else:
        intensidad = "difusa y distribuida"

    zonas   = ", ".join(z.replace("-", " ") for z, _ in active[:3])
    pct_act = 100.0 * (cam_mask >= threshold).mean()

    return (
        f"Las regiones de mayor activación (Grad-CAM++) se concentran en la zona {zonas} "
        f"de la imagen, con una intensidad {intensidad}. "
        f"Aproximadamente el {pct_act:.0f}% del área de la hoja supera el umbral de activación."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de inferencia + Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
def predict_with_gradcam(
    image_path:   str | Path | Image.Image,
    model:        torch.nn.Module,
    cam:          GradCAMPlusPlus,
    class_names:  list[str],
    device:       torch.device,
    target_class: Optional[int] = None,
    alpha:        float = DEFAULT_ALPHA,
    colormap:     int   = cv2.COLORMAP_JET,
) -> dict:
    """
    Pipeline completo de inferencia + Grad-CAM++ sobre una imagen.

    Combina:
      1. Preprocesamiento y forward pass del modelo (predicción + probabilidades)
      2. Cálculo del mapa de activación Grad-CAM++ sobre layer4
      3. Superposición del heatmap sobre la imagen original
      4. Descripción textual de la zona activa (contexto para el LLM)

    Args:
        image_path:   Ruta al archivo de imagen, o directamente un PIL.Image.
        model:        Modelo ResNet-50 en modo eval.
        cam:          Objeto Grad-CAM++ creado con build_cam().
        class_names:  Lista de nombres de clases en orden de índice.
        device:       Dispositivo del modelo (cpu / cuda).
        target_class: Si se especifica, genera el mapa para esa clase
                      en lugar de la predicción del modelo.
        alpha:        Transparencia del heatmap (0=invisible, 1=opaco).
        colormap:     Colormap OpenCV para el heatmap (por defecto COLORMAP_JET).

    Returns:
        Dict con:
          - 'class_idx'        : int — índice de la clase predicha
          - 'class_name'       : str — nombre interno (ej. 'Tomato___Late_blight')
          - 'class_label'      : str — etiqueta legible (ej. 'Tomato / Late blight')
          - 'confidence'       : float — confianza de la predicción [0, 1]
          - 'all_probs'        : np.ndarray — probabilidades para todas las clases
          - 'top3'             : list[dict] — top-3 predicciones con clase y prob
          - 'cam_mask'         : np.ndarray — mapa de calor normalizado (H × W)
          - 'overlay'          : PIL.Image — imagen con heatmap superpuesto
          - 'original'         : PIL.Image — imagen original redimensionada
          - 'zone_description' : str — descripción textual para el LLM

    Ejemplo:
        >>> result = predict_with_gradcam('hoja.jpg', model, cam, class_names, device)
        >>> print(result['class_label'], f"{result['confidence']*100:.1f}%")
        >>> result['overlay'].save('gradcam_output.png')
    """
    # 1. Cargar imagen
    if isinstance(image_path, (str, Path)):
        img_pil = Image.open(image_path).convert("RGB")
    else:
        img_pil = image_path.convert("RGB")

    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_np      = np.array(img_resized) / 255.0   # float [0,1] para overlay

    # 2. Tensor para el modelo
    tensor = _transform_eval(img_resized).unsqueeze(0).to(device)

    # 3. Inferencia (sin gradientes para velocidad)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx   = int(probs.argmax())
    target_idx = target_class if target_class is not None else pred_idx
    targets    = [ClassifierOutputTarget(target_idx)]

    # 4. Grad-CAM++ (sí necesita gradientes internamente)
    cam_mask = cam(input_tensor=tensor, targets=targets)[0]   # (H, W)
    cam_mask = np.clip(cam_mask, 0, 1).astype(np.float32)

    # 5. Superposición del heatmap
    overlay_np  = show_cam_on_image(
        img_np.astype(np.float32),
        cam_mask,
        use_rgb=True,
        colormap=colormap,
        image_weight=1.0 - alpha,
    )
    overlay_pil = Image.fromarray(overlay_np)

    # 6. Descripción textual
    zone_desc = describe_activation_zone(cam_mask)

    # 7. Top-3 predicciones
    top3 = [
        {"class": class_names[i], "label": _label_legible(class_names[i]),
         "prob": float(probs[i])}
        for i in probs.argsort()[::-1][:3]
    ]

    return {
        "class_idx"       : pred_idx,
        "class_name"      : class_names[pred_idx],
        "class_label"     : _label_legible(class_names[pred_idx]),
        "confidence"      : float(probs[pred_idx]),
        "all_probs"       : probs,
        "top3"            : top3,
        "cam_mask"        : cam_mask,
        "overlay"         : overlay_pil,
        "original"        : img_resized,
        "zone_description": zone_desc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparación de métodos CAM (para el informe — fig10)
# ─────────────────────────────────────────────────────────────────────────────
def compare_cam_methods(
    image_path:  str | Path | Image.Image,
    model:       torch.nn.Module,
    class_names: list[str],
    device:      torch.device,
    alpha:       float = DEFAULT_ALPHA,
) -> dict[str, dict]:
    """
    Aplica GradCAM, GradCAM++ y EigenCAM sobre la misma imagen y devuelve
    los tres overlays para comparación.

    Usado para generar la figura fig10_cam_variants.png del informe.

    Args:
        image_path:  Ruta a la imagen o PIL.Image.
        model:       Modelo ResNet-50 en eval mode.
        class_names: Lista de nombres de clases.
        device:      Dispositivo.
        alpha:       Transparencia del heatmap.

    Returns:
        Dict {nombre_metodo: resultado_predict_with_gradcam}.

    Ejemplo:
        >>> comparacion = compare_cam_methods('hoja.jpg', model, class_names, device)
        >>> for metodo, res in comparacion.items():
        ...     res['overlay'].save(f'overlay_{metodo}.png')
    """
    resultados = {}
    for name, CAMClass in CAM_METHODS.items():
        cam_obj = CAMClass(model=model, target_layers=[model.layer4[-1]])
        resultados[name] = predict_with_gradcam(
            image_path, model, cam_obj, class_names, device, alpha=alpha
        )
    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────
def _label_legible(class_name: str) -> str:
    """'Tomato___Late_blight' → 'Tomato / Late blight'"""
    parts  = class_name.split("___")
    planta = parts[0].replace("_", " ").strip()
    estado = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
    return f"{planta} / {estado}" if estado else planta
