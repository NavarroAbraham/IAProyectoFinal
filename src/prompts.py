"""
src/llm/prompts.py
Prompts del sistema, cliente Groq y función de explicación en lenguaje natural
basada en la predicción CNN + análisis Grad-CAM++.

Extraído de: notebooks/04_llm_groq.ipynb
Proyecto Final IA · EAFIT 2026-1

Modelo: llama-3.1-8b-instant vía Groq Cloud API (tier gratuito)
Docs:   https://console.groq.com/docs/openai
"""

import os
import time
from typing import Literal

# ─────────────────────────────────────────────────────────────────────────────
# Constantes del LLM
# ─────────────────────────────────────────────────────────────────────────────
LLM_MODEL:   str   = "llama-3.1-8b-instant"
MAX_TOKENS:  int   = 300
TEMPERATURE: float = 0.4    # bajo para respuestas consistentes y factuales
MAX_RETRIES: int   = 3
RETRY_BASE:  float = 2.0    # backoff exponencial: 2s, 4s, 8s

ExplainMode = Literal["agronomo", "agricultor"]


# ─────────────────────────────────────────────────────────────────────────────
# Prompts del sistema
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_AGRONOMO: str = """\
Eres un experto en fitopatología y agricultura de precisión.
Tu función es analizar predicciones de un modelo de visión computacional (CNN ResNet-50)
entrenado para clasificar enfermedades en plantas, combinadas con mapas de activación
Grad-CAM++ que indican qué zonas de la imagen activaron la predicción.

Proporciona explicaciones técnicas pero comprensibles. Responde SIEMPRE en español.
Estructura tu respuesta en exactamente 3 párrafos cortos:
1. Diagnóstico: qué indica la predicción del modelo
2. Evidencia visual: qué sugieren las zonas de activación Grad-CAM sobre los síntomas
3. Implicaciones: riesgo agronómico y próximos pasos recomendados

Sé conciso. Máximo 120 palabras en total.\
"""

SYSTEM_AGRICULTOR: str = """\
Eres un asistente agrícola que ayuda a agricultores a entender
el diagnóstico de enfermedades en sus cultivos explicado por una aplicación de inteligencia artificial.

Usa lenguaje sencillo y cotidiano, sin tecnicismos. Responde SIEMPRE en español.
Estructura tu respuesta así:
- Una oración que diga qué encontró la IA en la planta
- Una oración sobre dónde se ven los síntomas en la hoja
- Dos recomendaciones prácticas y concretas que el agricultor puede hacer hoy

Máximo 80 palabras. Tono amable y directo.\
"""

# Mapa modo → system prompt
_SYSTEM_MAP: dict[ExplainMode, str] = {
    "agronomo"  : SYSTEM_AGRONOMO,
    "agricultor": SYSTEM_AGRICULTOR,
}


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del mensaje de usuario
# ─────────────────────────────────────────────────────────────────────────────
def build_user_message(
    pred_label:   str,
    true_label:   str,
    confidence:   float,
    zone_desc:    str,
    top3:         list[dict],
    correct:      bool = True,
) -> str:
    """
    Construye el mensaje de usuario con toda la información del resultado
    CNN + Grad-CAM para enviarlo al LLM.

    Args:
        pred_label:  Etiqueta predicha legible (ej. 'Tomato / Late blight').
        true_label:  Cultivo real (ej. 'Tomato'). Puede ser igual a pred_label.
        confidence:  Confianza de la predicción [0, 1].
        zone_desc:   Descripción textual de la zona activa (de describe_activation_zone()).
        top3:        Lista de dicts con keys 'label' y 'prob' (top-3 predicciones).
        correct:     Si la predicción coincide con la clase real.

    Returns:
        String formateado listo para enviarse como mensaje 'user' al LLM.

    Ejemplo:
        >>> msg = build_user_message(
        ...     pred_label='Tomato / Late blight',
        ...     true_label='Tomato',
        ...     confidence=0.923,
        ...     zone_desc='Activación alta en zona superior-izquierda...',
        ...     top3=[{'label': 'Tomato / Late blight', 'prob': 0.923}, ...],
        ... )
    """
    estado  = "correctamente" if correct else "incorrectamente"
    top3_str = "\n".join(
        f"  {i+1}. {p['label']} ({p['prob']*100:.1f}%)"
        for i, p in enumerate(top3)
    )
    return (
        f"Información del modelo:\n"
        f"- Cultivo analizado: {true_label}\n"
        f"- Predicción del modelo: {pred_label} (clasificado {estado})\n"
        f"- Confianza: {confidence*100:.1f}%\n"
        f"- Top 3 predicciones:\n{top3_str}\n\n"
        f"Análisis Grad-CAM++:\n{zone_desc}\n\n"
        f"Por favor analiza este caso."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cliente Groq
# ─────────────────────────────────────────────────────────────────────────────
def get_groq_client(api_key: str | None = None):
    """
    Crea y devuelve un cliente Groq.

    Orden de búsqueda de la API key:
        1. Parámetro api_key (si se pasa)
        2. Variable de entorno GROQ_API_KEY
        3. Archivo .env en el directorio raíz del proyecto

    Args:
        api_key: API key de Groq. Si None, la busca en el entorno.

    Returns:
        Instancia de groq.Groq lista para usar.

    Raises:
        ValueError: Si no se encuentra ninguna API key.
        ImportError: Si el paquete 'groq' no está instalado.

    Ejemplo:
        >>> client = get_groq_client()          # desde variable de entorno
        >>> client = get_groq_client('gsk_xx')  # explícita
    """
    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError(
            "El paquete 'groq' no está instalado. "
            "Ejecuta: pip install groq"
        ) from e

    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY")

    if api_key is None:
        # Intentar cargar desde .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GROQ_API_KEY")
        except ImportError:
            pass

    if not api_key:
        raise ValueError(
            "No se encontró GROQ_API_KEY. "
            "Opciones:\n"
            "  1. export GROQ_API_KEY=gsk_xxx\n"
            "  2. Añadir GROQ_API_KEY=gsk_xxx en archivo .env\n"
            "  3. Pasar api_key='gsk_xxx' como argumento"
        )

    return Groq(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de explicación
# ─────────────────────────────────────────────────────────────────────────────
def explain_prediction(
    result:      dict,
    mode:        ExplainMode = "agronomo",
    api_key:     str | None = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Genera una explicación en lenguaje natural de la predicción CNN + Grad-CAM
    usando el LLM (LLaMA-3.1-8b vía Groq API).

    Maneja automáticamente el rate limiting con backoff exponencial
    (límite gratuito de Groq: 30 req/min, 6000 tokens/min).

    Args:
        result:      Dict de salida de predict_with_gradcam() con al menos:
                       'pred_label' o 'class_label', 'class_name', 'confidence',
                       'zone_description', 'top3', 'correct' (opcional).
        mode:        'agronomo' → explicación técnica (3 párrafos, máx 120 palabras)
                     'agricultor' → lenguaje simple + 2 recomendaciones (máx 80 palabras)
        api_key:     API key de Groq. Si None, usa GROQ_API_KEY del entorno.
        max_retries: Intentos máximos ante errores 429 (rate limit).

    Returns:
        String con la explicación generada por el LLM.
        En caso de error devuelve un mensaje descriptivo (no lanza excepción).

    Ejemplo:
        >>> result = predict_with_gradcam(...)
        >>> exp = explain_prediction(result, mode='agronomo')
        >>> print(exp)

        >>> exp_farmer = explain_prediction(result, mode='agricultor')
        >>> print(exp_farmer)
    """
    from groq import RateLimitError

    # Extraer campos del resultado (compatibilidad con app/main.py y notebooks)
    pred_label  = result.get("class_label") or result.get("pred_label", "Desconocido")
    true_label  = result.get("true_label", pred_label.split(" / ")[0] if " / " in pred_label else pred_label)
    confidence  = result.get("confidence", 0.0)
    zone_desc   = result.get("zone_description", "")
    top3        = result.get("top3", [])
    correct     = result.get("correct", True)

    system_prompt = _SYSTEM_MAP[mode]
    user_message  = build_user_message(
        pred_label=pred_label,
        true_label=true_label,
        confidence=confidence,
        zone_desc=zone_desc,
        top3=top3,
        correct=correct,
    )

    try:
        client = get_groq_client(api_key)
    except (ValueError, ImportError) as e:
        return f"⚠️ Error de configuración: {e}"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            wait_secs = RETRY_BASE ** attempt
            print(f"  ⏳ Rate limit — esperando {wait_secs:.0f}s (intento {attempt}/{max_retries})")
            time.sleep(wait_secs)

        except Exception as e:
            return f"⚠️ Error inesperado al contactar Groq: {e}"

    return "⚠️ Rate limit agotado. Espera 1 minuto e intenta de nuevo."


# ─────────────────────────────────────────────────────────────────────────────
# Procesar un lote de resultados
# ─────────────────────────────────────────────────────────────────────────────
def explain_batch(
    results:       list[dict],
    mode:          ExplainMode = "agronomo",
    api_key:       str | None = None,
    sleep_between: float = 2.2,
) -> list[dict]:
    """
    Genera explicaciones LLM para una lista de resultados Grad-CAM.

    Respeta el rate limit gratuito de Groq (30 req/min) con un sleep
    configurable entre llamadas.

    Args:
        results:       Lista de dicts de salida de predict_with_gradcam().
        mode:          Modo de explicación ('agronomo' o 'agricultor').
        api_key:       API key de Groq.
        sleep_between: Segundos entre llamadas (2.2s es seguro para 30 req/min).

    Returns:
        Lista de dicts enriquecidos con key 'llm_explanation'.

    Ejemplo:
        >>> enriched = explain_batch(gradcam_results, mode='agronomo')
        >>> for r in enriched:
        ...     print(r['class_label'], '→', r['llm_explanation'][:80])
    """
    enriched = []
    for i, result in enumerate(results):
        explanation = explain_prediction(result, mode=mode, api_key=api_key)
        enriched.append({**result, "llm_explanation": explanation})
        if i < len(results) - 1:
            time.sleep(sleep_between)
    return enriched
