import json
import os
from typing import Any

import fitz
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


MODEL_FAST = os.getenv("MODEL_FAST", "gpt-4.1-mini")
MODEL_SMART = os.getenv("MODEL_SMART", "gpt-4.1")


def procesar_texto(prompt: str, smart: bool = False) -> str:
    response = client.responses.create(
        model=MODEL_SMART if smart else MODEL_FAST,
        input=prompt,
    )
    return response.output_text


def responder_con_contexto(
    system_prompt: str,
    user_prompt: str,
    smart: bool = True,
    max_output_tokens: int = 260,
) -> str:
    response = client.responses.create(
        model=MODEL_SMART if smart else MODEL_FAST,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        max_output_tokens=max_output_tokens,
    )
    return response.output_text


def analizar_turno_conversacional(contexto: str, texto_usuario: str) -> dict[str, Any]:
    prompt = f"""
Analiza el turno del usuario para un asistente turistico de Yucatan.
Devuelve JSON estricto con esta forma:
{{
  "mode": "onboarding|recommend|logistics|destination_detail|itinerary|general",
  "destination_name": "string o vacio",
  "wants_weather": true/false,
  "wants_route": true/false,
  "wants_eta": true/false,
  "wants_budget": true/false,
  "wants_companion_advice": true/false,
  "needs_clarification": true/false,
  "clarification_question": "string corto o vacio"
}}

Reglas:
- "recommend" solo cuando claramente pide sugerencias/lista de lugares.
- "logistics" cuando pide como llegar, distancia, tiempo, ruta, trafico, clima para llegar.
- "destination_detail" cuando ya parece tener destino y pide detalles practicos (presupuesto, horarios, con quien va, tips).
- No inventes datos.

Contexto conversacion:
{contexto}

Texto usuario:
{texto_usuario}
"""
    response = client.responses.create(
        model=MODEL_SMART,
        input=prompt,
        text={"format": {"type": "json_object"}},
    )
    raw = response.output_text
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {
        "mode": "general",
        "destination_name": "",
        "wants_weather": False,
        "wants_route": False,
        "wants_eta": False,
        "wants_budget": False,
        "wants_companion_advice": False,
        "needs_clarification": False,
        "clarification_question": "",
    }


def extraer_memoria_importante(texto_usuario: str) -> list[dict[str, Any]]:
    prompt = f"""
Extrae solo hechos estables y utiles para personalizar un asistente turistico de Yucatan.
Devuelve JSON estricto con esta forma:
{{"items":[{{"key":"...","value":"...","importance":1-5}}]}}

Reglas:
- keys sugeridas: nombre_usuario, idioma, municipio_preferido, intereses, presupuesto, movilidad, restricciones, tono_preferido, detalle_preferido.
- No inventes datos.
- Si no hay nada estable, devuelve {{"items":[]}}.

Texto usuario:
{texto_usuario}
"""
    response = client.responses.create(
        model=MODEL_FAST,
        input=prompt,
        text={"format": {"type": "json_object"}},
    )
    raw = response.output_text
    try:
        data = json.loads(raw)
        return data.get("items", []) if isinstance(data, dict) else []
    except Exception:
        return []


def extraer_texto_pdf(path: str) -> str:
    doc = fitz.open(path)
    texto = ""
    for page in doc:
        texto += page.get_text()
    return texto


def procesar_pdf(path: str) -> str:
    texto = extraer_texto_pdf(path)

    response = client.responses.create(
        model=MODEL_FAST,
        input=f"Resume el documento para un contexto turistico de Yucatan en maximo 12 lineas:\n\n{texto}",
        max_output_tokens=450,
    )

    return response.output_text
