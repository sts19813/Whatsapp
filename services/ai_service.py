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


def analizar_intencion_taquilla(contexto: str, texto_usuario: str) -> dict[str, Any]:
    prompt = f"""
Analiza el mensaje de un admin para un asistente de taquilla/eventos.
Devuelve JSON ESTRICTO con esta forma:
{{
  "intent": "help|events|events_upcoming|event_detail|availability|sales_overview|sales_search|sales_latest|pdf_lookup|sell_cash|unknown",
  "event_id": "",
  "event_query": "",
  "scope": "",
  "limit": 10,
  "sale_type": "all",
  "q": "",
  "name": "",
  "email": "",
  "reference": "",
  "date": "",
  "from": "",
  "to": "",
  "buyer_name": "",
  "buyer_email": "",
  "buyer_phone": "",
  "cart": [],
  "needs_clarification": false,
  "clarification_question": ""
}}

Reglas:
- Usa "events" para resumen de eventos, y "events_upcoming" para proximos eventos.
- Usa "event_detail" cuando pidan desglose de un evento especifico.
- Usa "availability" para disponibilidad y boletos vendibles.
- Usa "sales_overview" para resumen global de ventas por fecha/evento.
- Usa "sales_search" para busquedas de tickets/registros por filtros.
- Usa "sales_latest" para ultimas ventas.
- Usa "pdf_lookup" cuando pidan reimpresion/reenvio PDF o URL PDF.
- Usa "sell_cash" cuando pidan venta de taquilla en efectivo.
- "scope" permitido: all|upcoming|past|today.
- "sale_type" permitido: all|ticket|registration.
- Si detectas fechas, usa formato YYYY-MM-DD.
- Para "sell_cash", intenta poblar buyer_name/buyer_email/buyer_phone y cart.
- Cada item de cart debe ser:
  - ticket: {{"type":"ticket","id":"UUID","qty":2}}
  - registration: {{"type":"registration","qty":1,"price":1500}}
- Si falta info critica para ejecutar, usa needs_clarification=true con una pregunta corta.
- No inventes UUIDs.

Contexto:
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
            intent = (data.get("intent") or "").strip().lower()
            if not intent:
                data["intent"] = "unknown"

            limit = data.get("limit")
            try:
                limit_int = int(limit) if limit is not None else 10
            except Exception:
                limit_int = 10
            data["limit"] = max(1, min(limit_int, 200))

            cart = data.get("cart")
            if isinstance(cart, dict):
                data["cart"] = [cart]
            elif not isinstance(cart, list):
                data["cart"] = []

            return data
    except Exception:
        pass

    return {
        "intent": "unknown",
        "event_id": "",
        "event_query": "",
        "scope": "",
        "limit": 10,
        "sale_type": "all",
        "q": "",
        "name": "",
        "email": "",
        "reference": "",
        "date": "",
        "from": "",
        "to": "",
        "buyer_name": "",
        "buyer_email": "",
        "buyer_phone": "",
        "cart": [],
        "needs_clarification": False,
        "clarification_question": "",
    }
