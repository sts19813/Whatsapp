import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from fastapi import FastAPI, Request

from services.ai_service import (
    extraer_memoria_importante,
    procesar_pdf,
    responder_con_contexto,
)
from services.audio_service import transcribir_audio
from services.db_service import (
    get_latest_location,
    get_memory_value,
    get_memories,
    get_recent_messages,
    get_user,
    init_db,
    save_location,
    save_message,
    set_user_municipality,
    upsert_memory,
    upsert_user,
)
from services.image_service import procesar_imagen
from services.tourism_service import (
    estimate_drive_route,
    enrich_places_with_distance,
    filter_places,
    find_best_place_match,
    format_places_for_whatsapp,
    geocode_place,
    haversine_km,
    infer_category,
    infer_interest_topic,
    is_exploratory_query,
    maps_directions_link,
    normalize_text,
    travel_advice,
    weather_label,
    weather_snapshot,
)

app = FastAPI()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "mi_webhook_verificacion_123")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

BASE_PATH = Path("base_turismo_yucatan_inventur.json")
with BASE_PATH.open(encoding="utf-8") as f:
    BASE_TURISMO = json.load(f)

BOT_NAME = "Itzamna AI Turismo Yucatan"
MUNICIPIOS_CLAVE = [
    "merida",
    "progreso",
    "valladolid",
    "celestun",
    "tekax",
    "tinum",
    "muna",
    "homun",
    "izamal",
]

SYSTEM_PROMPT = """
Eres un asesor conversacional de alto nivel, especializado unicamente en turismo en Yucatan.
Habla como una IA natural (estilo ChatGPT), no como bot de respuestas rigidas.
Tu prioridad es la mejor experiencia del viajero: claridad, utilidad, logistica, seguridad y contexto real.
Siempre responde en espanol.

Reglas:
1) Mantente estrictamente en turismo de Yucatan. Si preguntan algo fuera de eso, redirige con elegancia al contexto turistico yucateco.
2) Responde fluido y humano. No listes lugares automaticamente salvo que realmente aporte valor o el usuario lo pida.
3) Si hay destino concreto, prioriza: como llegar, tiempo estimado, clima, precauciones, costos orientativos y siguiente mejor accion.
4) Si faltan datos exactos, dilo claramente y propone la forma mas rapida de validarlo.
5) Usa la fecha local proporcionada para interpretar "hoy", "estas fechas", "manana", etc.
""".strip()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def detectar_municipio(texto: str) -> str | None:
    t = normalize_text(texto)
    for m in MUNICIPIOS_CLAVE:
        if m in t:
            return m.capitalize()
    return None


def contexto_fecha_actual() -> str:
    now = datetime.now(ZoneInfo("America/Mexico_City"))
    dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
    meses = [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ]
    dia_semana = dias[now.weekday()]
    mes = meses[now.month - 1]
    return (
        f"Fecha local actual confirmada: {dia_semana} {now.day} de {mes} de {now.year}. "
        f"Hora local aproximada: {now.strftime('%H:%M')} (America/Mexico_City). "
        "Si el usuario dice 'hoy', 'estas fechas' o similares, usa esta fecha exacta."
    )


def construir_contexto_usuario(sender: str) -> str:
    user = get_user(sender) or {}
    memories = get_memories(sender, limit=10)
    recent = get_recent_messages(sender, limit=12)
    location = get_latest_location(sender)

    memoria_txt = "\n".join([f"- {m['key']}: {m['value']}" for m in memories]) or "- (sin memoria)"
    mensajes_txt = "\n".join([f"{m['role']}: {m['content']}" for m in recent]) or "(sin historial)"
    loc_txt = f"lat={location['lat']}, lon={location['lon']}" if location else "sin ubicacion"

    return (
        f"Usuario: {sender}\n"
        f"Municipio guardado: {user.get('municipality', 'sin definir')}\n"
        f"Ubicacion actual: {loc_txt}\n"
        f"Memorias:\n{memoria_txt}\n\n"
        f"Historial reciente:\n{mensajes_txt}"
    )


def guardar_memoria_desde_texto(sender: str, text: str) -> None:
    items = extraer_memoria_importante(text)
    for item in items:
        key = (item.get("key") or "").strip()
        value = (item.get("value") or "").strip()
        importance = int(item.get("importance") or 2)
        if key and value:
            upsert_memory(sender, key, value, max(1, min(importance, 5)))


def obtener_lugares_relevantes(sender: str, consulta: str, limit: int = 5) -> list[dict[str, Any]]:
    user = get_user(sender) or {}
    municipality = user.get("municipality")
    category_hint = infer_category(consulta)

    lugares = filter_places(
        places=BASE_TURISMO,
        query=consulta,
        municipality=municipality,
        category_hint=category_hint,
        limit=10,
    )

    if not lugares and municipality:
        lugares = filter_places(
            places=BASE_TURISMO,
            query=consulta,
            municipality=None,
            category_hint=category_hint,
            limit=max(10, limit + 3),
        )

    if not lugares:
        return []

    user_location = get_latest_location(sender)
    lugares = enrich_places_with_distance(lugares, user_location)
    return lugares[:limit]


def guardar_ultimas_recomendaciones(sender: str, lugares: list[dict[str, Any]]) -> None:
    resumen = []
    for p in lugares[:8]:
        resumen.append(
            {
                "nombre": p.get("nombre", ""),
                "categoria": p.get("categoria", ""),
                "municipio": p.get("municipio", ""),
                "direccion": p.get("direccion", ""),
                "telefono": p.get("telefono", ""),
                "web": p.get("web", ""),
                "fuente_url": p.get("fuente_url", ""),
            }
        )
    upsert_memory(sender, "last_recommendations", json.dumps(resumen, ensure_ascii=False), 4)


def cargar_ultimas_recomendaciones(sender: str) -> list[dict[str, Any]]:
    raw = get_memory_value(sender, "last_recommendations")
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def responder_mas_cercano(sender: str, consulta: str = "") -> str | None:
    user_location = get_latest_location(sender)
    if not user_location:
        upsert_memory(sender, "pending_action", "resolve_nearest", 5)
        return "Para decirte exacto cual te queda mas cerca, comparteme tu ubicacion actual en WhatsApp."

    candidatos = cargar_ultimas_recomendaciones(sender)
    if not candidatos:
        if consulta:
            candidatos = obtener_lugares_relevantes(sender, consulta, limit=5)
        if not candidatos:
            return "No tengo candidatos recientes. Dime que tipo de lugar buscas y te doy opciones cercanas."

    mejores = []
    for lugar in candidatos:
        coords = geocode_place(lugar)
        if not coords:
            continue
        dist = haversine_km(user_location["lat"], user_location["lon"], coords["lat"], coords["lon"])
        item = dict(lugar)
        item["distance_km"] = round(dist, 2)
        mejores.append(item)

    if not mejores:
        return "No pude calcular distancia exacta con esos lugares. Te paso nuevas opciones si me dices el tipo de lugar."

    mejores.sort(key=lambda x: x.get("distance_km", 999999.0))
    top = mejores[0]
    upsert_memory(sender, "destino_actual_nombre", top.get("nombre", ""), 5)
    upsert_memory(sender, "destino_actual_municipio", top.get("municipio", ""), 4)
    upsert_memory(sender, "pending_action", "none", 3)

    mapas = (
        f"https://www.google.com/maps/search/?api=1&query="
        f"{top.get('nombre', '').replace(' ', '+')}+{top.get('direccion', '').replace(' ', '+')}"
    )

    extra = ""
    if len(mejores) > 1:
        alt = mejores[1]
        extra = (
            f"\nAlternativa cercana: {alt.get('nombre', 'N/D')} "
            f"({alt.get('distance_km', 'N/D')} km aprox)."
        )

    return (
        f"El que te queda mas cerca es *{top.get('nombre', 'N/D')}* "
        f"({top.get('distance_km', 'N/D')} km aprox).\n"
        f"Direccion: {top.get('direccion', 'N/D')}\n"
        f"Maps: {mapas}{extra}\n\n"
        "Si quieres, te digo cuanto tiempo haces en auto desde donde estas."
    )


def recomendar_lugares(sender: str, consulta: str) -> str | None:
    lugares = obtener_lugares_relevantes(sender, consulta, limit=5)
    if not lugares:
        return None

    user = get_user(sender) or {}
    municipality = user.get("municipality")
    user_location = get_latest_location(sender)
    guardar_ultimas_recomendaciones(sender, lugares)
    texto_lugares = format_places_for_whatsapp(lugares, limit=5)

    if not texto_lugares:
        return None

    prefijo = "Estas son recomendaciones en Yucatan"
    if municipality:
        prefijo = f"Estas son recomendaciones en/para {municipality}"
    if user_location:
        prefijo += " ordenadas por cercania cuando fue posible"

    return f"{prefijo}:\n\n{texto_lugares}"


def resolver_destino(sender: str, consulta: str) -> dict[str, Any] | None:
    user = get_user(sender) or {}
    municipality = user.get("municipality")

    place = find_best_place_match(BASE_TURISMO, consulta, municipality=municipality)
    if not place:
        place = find_best_place_match(BASE_TURISMO, consulta, municipality=None)

    if place:
        upsert_memory(sender, "destino_actual_nombre", place.get("nombre", ""), 5)
        upsert_memory(sender, "destino_actual_municipio", place.get("municipio", ""), 4)
        return place

    saved_name = get_memory_value(sender, "destino_actual_nombre")
    if saved_name:
        remembered = find_best_place_match(BASE_TURISMO, saved_name, municipality=None)
        if remembered:
            return remembered

    return None


def construir_info_destino_y_ruta(sender: str, destino: dict[str, Any]) -> dict[str, Any]:
    user_location = get_latest_location(sender)
    destino_coords = geocode_place(destino)
    weather = None
    route = None
    directions = ""

    if destino_coords:
        weather = weather_snapshot(destino_coords["lat"], destino_coords["lon"])
        if user_location:
            route = estimate_drive_route(user_location, destino_coords)
            directions = maps_directions_link(user_location, destino_coords)

    return {
        "user_location": user_location,
        "destino_coords": destino_coords,
        "weather": weather,
        "route": route,
        "directions": directions,
    }


def responder_logistica(sender: str, consulta: str, destino_hint: str = "") -> str | None:
    destino = resolver_destino(sender, destino_hint or consulta)
    if not destino:
        return (
            "Entiendo que quieres ruta/logistica. Dime el nombre exacto del lugar en Yucatan "
            "(ejemplo: Marina Paraiso Progreso) para darte tiempo, ruta, clima y precauciones."
        )

    info = construir_info_destino_y_ruta(sender, destino)
    user_location = info["user_location"]
    if not user_location:
        return (
            f"Ya identifique tu destino: {destino.get('nombre', 'lugar seleccionado')}.\n"
            "Comparte tu ubicacion en WhatsApp y te doy tiempo estimado en auto, distancia, "
            "ruta de Google Maps y recomendaciones de clima/seguridad."
        )

    destino_coords = info["destino_coords"]
    if not destino_coords:
        maps_search = destino.get("fuente_url") or destino.get("web") or "https://www.google.com/maps"
        return (
            f"Ubique el destino {destino.get('nombre', '')}, pero no pude geocodificar coordenadas exactas.\n"
            f"Te dejo referencia: {maps_search}\n"
            "Si me compartes una direccion mas exacta, te calculo tiempo y distancia."
        )

    route = info["route"]
    directions = info["directions"]
    weather = info["weather"]

    clima_txt = "Sin datos de clima en este momento."
    if weather:
        code = int(weather.get("weather_code", -1))
        clima_txt = (
            f"{weather_label(code)} | {weather.get('temperature_2m', 'N/D')} C | "
            f"precipitacion {weather.get('precipitation', 'N/D')} mm | "
            f"viento {weather.get('wind_speed_10m', 'N/D')} km/h"
        )

    consejo = travel_advice(weather)
    origen_txt = f"{round(user_location['lat'], 5)}, {round(user_location['lon'], 5)}"

    return (
        f"Ruta hacia *{destino.get('nombre', 'Destino')}* ({destino.get('municipio', 'Yucatan')}):\n\n"
        f"- Distancia estimada: {route['distance_km']} km\n"
        f"- Tiempo en auto: {route['duration_min']} min aprox\n"
        f"- Origen detectado: {origen_txt}\n"
        f"- Navegacion: {directions}\n"
        f"- Clima actual destino: {clima_txt}\n"
        f"- Precauciones: {consejo}\n\n"
        "Si quieres, te calculo tambien una hora recomendada de salida para evitar calor o lluvia."
    )


def responder_detalle_destino(sender: str, consulta: str, destino_hint: str = "") -> str | None:
    destino = resolver_destino(sender, destino_hint or consulta)
    if not destino:
        return None

    info = construir_info_destino_y_ruta(sender, destino)
    weather = info["weather"]
    route = info["route"]
    user = get_user(sender) or {}
    memories = get_memories(sender, limit=8)
    memory_lines = "\n".join([f"- {m['key']}: {m['value']}" for m in memories]) or "- (sin datos)"

    clima_txt = "No disponible"
    if weather:
        code = int(weather.get("weather_code", -1))
        clima_txt = (
            f"{weather_label(code)} | {weather.get('temperature_2m', 'N/D')} C | "
            f"precipitacion {weather.get('precipitation', 'N/D')} mm | "
            f"viento {weather.get('wind_speed_10m', 'N/D')} km/h"
        )

    route_txt = "No disponible"
    if route:
        route_txt = f"{route['distance_km']} km y {route['duration_min']} min aprox"

    consejo = travel_advice(weather)

    style_inst = obtener_instrucciones_estilo(sender)
    prompt_usuario = f"""
{contexto_fecha_actual()}

Consulta actual:
{consulta}

Perfil usuario:
- municipio_preferido: {user.get('municipality', 'sin definir')}
{memory_lines}

Destino seleccionado:
- nombre: {destino.get('nombre', '')}
- categoria: {destino.get('categoria', '')}
- municipio: {destino.get('municipio', '')}
- direccion: {destino.get('direccion', '')}
- telefono: {destino.get('telefono', '')}
- web: {destino.get('web', '')}
- fuente: {destino.get('fuente_url', '')}

Logistica:
- ruta estimada: {route_txt}
- navegacion: {info.get('directions', '') or 'no disponible'}
- clima actual destino: {clima_txt}
- precauciones base: {consejo}

Responde en tono natural, como asistente personal.
No regreses una lista de recomendaciones.
Enfocate en valor practico para este destino: costos estimados, mejores horarios, tips para familia/ninos si aplica, seguridad, que llevar, y siguiente mejor accion.
Si un dato no existe en la base, dilo y sugiere como validarlo rapido.
{style_inst}
"""
    return responder_con_contexto(SYSTEM_PROMPT, prompt_usuario, smart=True)


def responder_tema_turistico(sender: str, consulta: str) -> str | None:
    topic = infer_interest_topic(consulta)
    if not topic:
        return None

    user = get_user(sender) or {}
    municipality = user.get("municipality")
    muestras = obtener_lugares_relevantes(sender, topic, limit=3)

    ejemplos = "\n".join(
        [f"- {p.get('nombre', '')} ({p.get('municipio', '')})" for p in muestras]
    ) or "- Sin ejemplos cercanos en base"

    memories = get_memories(sender, limit=8)
    memory_lines = "\n".join([f"- {m['key']}: {m['value']}" for m in memories]) or "- (sin datos)"

    style_inst = obtener_instrucciones_estilo(sender)
    prompt = f"""
{contexto_fecha_actual()}

Usuario pregunta:
{consulta}

Tema principal detectado: {topic}
Municipio preferido: {municipality or "sin definir"}
Memorias de contexto:
{memory_lines}

Ejemplos en la base local:
{ejemplos}

Responde como asesor turistico conversacional (no como bot de listado).
Da informacion precisa y practica del tema en Yucatan: mejores epocas, costos orientativos, seguridad, logistica, que llevar, errores comunes y como aprovechar mejor la visita.
Si no hay dato exacto (precio/horario), dilo y propone como validarlo rapido.
Cierra con una sola pregunta de seguimiento util.
{style_inst}
"""
    return responder_con_contexto(SYSTEM_PROMPT, prompt, smart=True)


def respuesta_primer_mensaje(sender: str, display_name: str | None = None) -> str:
    upsert_user(sender, display_name=display_name)
    upsert_memory(sender, "primer_contacto", "Usuario inicio conversacion", 3)
    return (
        "Hola, soy Itzamna AI, tu asistente premium de turismo en Yucatan.\n\n"
        "Ya guarde tu perfil inicial para personalizar recomendaciones.\n"
        "Comparte tu ubicacion de WhatsApp para recomendar lugares cercanos con distancia y links de Maps.\n\n"
        "Tambien puedes decirme: presupuesto, intereses (cenotes, gastronomia, arqueologia),"
        " movilidad y si viajas con familia/pareja."
    )


def es_peticion_explicita_de_recomendacion(text: str) -> bool:
    t = normalize_text(text)
    triggers = [
        "recomiend",
        "lugares para",
        "donde comer",
        "quiero opciones",
        "sugerencias",
        "que lugares",
        "que me recomiendas",
        "cerca de mi",
    ]
    return any(k in t for k in triggers)


def responder_capacidades(sender: str) -> str:
    user = get_user(sender) or {}
    municipality = user.get("municipality") or "Yucatan"
    return (
        f"Te puedo ayudar de forma personalizada para {municipality}: "
        "planear rutas, calcular tiempos y distancias, sugerirte horarios ideales, "
        "explicarte clima/precauciones, estimar presupuesto por tipo de plan, "
        "armarte itinerarios (1 dia o fin de semana), y afinar opciones si viajas con ninos, pareja o adultos mayores.\n\n"
        "Si quieres, empezamos con tu plan de hoy: dime zona, presupuesto aproximado y con quien viajas."
    )


def es_pregunta_logistica(text: str) -> bool:
    t = normalize_text(text)
    keywords = [
        "como llegar",
        "como llego",
        "tiempo",
        "cuanto tardo",
        "cuanto tiempo",
        "distancia",
        "ruta",
        "trafico",
        "clima",
        "precauciones",
        "seguridad",
    ]
    return any(k in t for k in keywords)


def actualizar_estilo_desde_texto(sender: str, text: str) -> None:
    t = normalize_text(text)

    if "formal" in t:
        upsert_memory(sender, "tono_preferido", "formal", 4)
    elif "casual" in t or "relajado" in t or "amigable" in t:
        upsert_memory(sender, "tono_preferido", "casual", 4)

    if "corto" in t or "breve" in t or "resumen" in t:
        upsert_memory(sender, "detalle_preferido", "corto", 4)
    elif "detallado" in t or "detalle" in t or "amplio" in t:
        upsert_memory(sender, "detalle_preferido", "detallado", 4)


def obtener_instrucciones_estilo(sender: str) -> str:
    tono = (get_memory_value(sender, "tono_preferido") or "").strip().lower()
    detalle = (get_memory_value(sender, "detalle_preferido") or "").strip().lower()

    tono_inst = "tono profesional y cercano"
    if tono == "formal":
        tono_inst = "tono formal, claro y respetuoso"
    elif tono == "casual":
        tono_inst = "tono casual, fluido y natural"

    detalle_inst = "nivel de detalle medio, muy practico"
    if detalle == "corto":
        detalle_inst = "respuesta corta (4-7 lineas) con lo esencial"
    elif detalle == "detallado":
        detalle_inst = "respuesta detallada con pasos claros y contexto util"

    return (
        f"Adapta la respuesta al usuario con {tono_inst} y {detalle_inst}. "
        "Mantente estrictamente en turismo de Yucatan."
    )


def responder_chat_turismo(sender: str, text: str) -> str:
    user = get_user(sender) or {}
    municipality = user.get("municipality") or "sin definir"
    location = get_latest_location(sender)

    style_inst = obtener_instrucciones_estilo(sender)
    memories = get_memories(sender, limit=10)
    memory_lines = "\n".join([f"- {m['key']}: {m['value']}" for m in memories]) or "- (sin datos)"
    recent = get_recent_messages(sender, limit=10)
    recent_lines = "\n".join([f"{m['role']}: {m['content']}" for m in recent]) or "(sin historial)"

    # Contexto de lugares/tema segun la consulta.
    candidates = obtener_lugares_relevantes(sender, text, limit=5)
    if candidates and es_peticion_explicita_de_recomendacion(text):
        guardar_ultimas_recomendaciones(sender, candidates)

    places_lines = "\n".join(
        [
            f"- {p.get('nombre', '')} | {p.get('categoria', '')} | {p.get('municipio', '')} | {p.get('direccion', '')}"
            for p in candidates[:5]
        ]
    ) or "- (sin lugares relevantes directos)"

    # Si existe destino en contexto y el usuario pide logistica, inyecta datos duros.
    destination = resolver_destino(sender, text)
    logistics_block = "No aplica."
    if destination and es_pregunta_logistica(text):
        info = construir_info_destino_y_ruta(sender, destination)
        weather = info.get("weather")
        route = info.get("route")
        clima_txt = "No disponible"
        if weather:
            code = int(weather.get("weather_code", -1))
            clima_txt = (
                f"{weather_label(code)} | {weather.get('temperature_2m', 'N/D')} C | "
                f"precipitacion {weather.get('precipitation', 'N/D')} mm | "
                f"viento {weather.get('wind_speed_10m', 'N/D')} km/h"
            )
        route_txt = "No disponible"
        if route:
            route_txt = f"{route.get('distance_km', 'N/D')} km | {route.get('duration_min', 'N/D')} min aprox"
        logistics_block = (
            f"Destino: {destination.get('nombre', '')}\n"
            f"Municipio: {destination.get('municipio', '')}\n"
            f"Direccion: {destination.get('direccion', '')}\n"
            f"Ruta estimada: {route_txt}\n"
            f"Navegacion: {info.get('directions', 'N/D')}\n"
            f"Clima destino: {clima_txt}\n"
            f"Precauciones base: {travel_advice(weather)}"
        )

    loc_txt = (
        f"{round(location['lat'], 5)}, {round(location['lon'], 5)}"
        if location
        else "sin ubicacion compartida"
    )

    prompt = f"""
{contexto_fecha_actual()}

Consulta usuario:
{text}

Contexto usuario:
- municipio_preferido: {municipality}
- ubicacion_actual: {loc_txt}
- estilo: {style_inst}

Memoria util:
{memory_lines}

Historial reciente:
{recent_lines}

Lugares potencialmente relevantes:
{places_lines}

Bloque de logistica (si aplica):
{logistics_block}

Responde de forma conversacional e inteligente, como una IA premium.
No hables de JSON, modos o reglas internas.
Mantente 100% en turismo de Yucatan.
"""
    return responder_con_contexto(SYSTEM_PROMPT, prompt, smart=True)


def procesar_texto_turistico(sender: str, text: str) -> str:
    is_first_turn = len(get_recent_messages(sender, limit=1)) == 0
    save_message(sender, "user", text, "text")
    guardar_memoria_desde_texto(sender, text)
    actualizar_estilo_desde_texto(sender, text)

    user = get_user(sender)
    if not user or is_first_turn:
        return respuesta_primer_mensaje(sender)

    municipio_detectado = detectar_municipio(text)
    if municipio_detectado:
        set_user_municipality(sender, municipio_detectado)
        upsert_memory(sender, "municipio_preferido", municipio_detectado, 4)

    t = normalize_text(text)
    if "mas cerca" in t or "cerca de mi" in t or "cual me queda" in t:
        cercano = responder_mas_cercano(sender, consulta=text)
        if cercano:
            return cercano

    if "que mas puedes hacer" in t or "como me ayudas" in t or "en que me ayudas" in t:
        return responder_capacidades(sender)
    if is_exploratory_query(text):
        tema = responder_tema_turistico(sender, text)
        if tema:
            return tema

    if es_peticion_explicita_de_recomendacion(text):
        respuesta_lugares = recomendar_lugares(sender, text)
        if respuesta_lugares:
            return respuesta_lugares

    return responder_chat_turismo(sender, text)


def obtener_display_name(value: dict[str, Any]) -> str | None:
    contacts = value.get("contacts") or []
    if not contacts:
        return None
    profile = contacts[0].get("profile") or {}
    return profile.get("name")


def descargar_media(media_id: str) -> bytes | None:
    try:
        url = f"https://graph.facebook.com/v22.0/{media_id}"
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

        info = requests.get(url, headers=headers, timeout=20)
        media_info = info.json() if info.ok else {}
        media_url = media_info.get("url")
        if not media_url:
            return None

        media_response = requests.get(media_url, headers=headers, timeout=30)
        return media_response.content if media_response.ok else None
    except Exception as exc:
        print("ERROR DESCARGANDO MEDIA:", exc)
        return None


def enviar_mensaje_whatsapp(numero: str, mensaje: str) -> None:
    try:
        url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": numero,
            "type": "text",
            "text": {"body": mensaje[:4000]},
        }
        requests.post(url, headers=headers, json=payload, timeout=20)
    except Exception as exc:
        print("ERROR ENVIANDO MENSAJE:", exc)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "running", "bot": BOT_NAME}


@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return int(challenge)
    return {"error": "Verification failed"}


@app.post("/webhook")
async def receive_message(request: Request):
    try:
        data = await request.json()

        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                if "messages" not in value:
                    continue

                display_name = obtener_display_name(value)

                for message in value.get("messages", []):
                    sender = message.get("from")
                    msg_type = message.get("type")
                    if not sender or not msg_type:
                        continue

                    upsert_user(sender, display_name=display_name)
                    respuesta_final = None

                    if msg_type == "text":
                        text = message.get("text", {}).get("body", "")
                        respuesta_final = procesar_texto_turistico(sender, text)

                    elif msg_type == "image":
                        media_id = message.get("image", {}).get("id")
                        image_bytes = descargar_media(media_id) if media_id else None
                        if image_bytes:
                            descripcion = procesar_imagen(image_bytes)
                            save_message(sender, "user", "[imagen] " + descripcion, "image")
                            respuesta_final = procesar_texto_turistico(
                                sender,
                                f"El usuario compartio una imagen. Descripcion: {descripcion}",
                            )
                        else:
                            respuesta_final = "No pude descargar la imagen. Intenta de nuevo."

                    elif msg_type == "audio":
                        media_id = message.get("audio", {}).get("id")
                        audio_bytes = descargar_media(media_id) if media_id else None
                        if audio_bytes:
                            texto_audio = transcribir_audio(audio_bytes)
                            save_message(sender, "user", "[audio] " + texto_audio, "audio")
                            respuesta_final = procesar_texto_turistico(sender, texto_audio)
                        else:
                            respuesta_final = "No pude descargar el audio. Intenta de nuevo."

                    elif msg_type == "location":
                        loc = message.get("location", {})
                        lat = loc.get("latitude")
                        lon = loc.get("longitude")
                        if lat is not None and lon is not None:
                            save_location(sender, float(lat), float(lon), "whatsapp_location")
                            upsert_memory(sender, "ultima_ubicacion", f"{lat},{lon}", 5)
                            pending_action = get_memory_value(sender, "pending_action") or ""
                            if pending_action == "resolve_nearest":
                                respuesta_final = responder_mas_cercano(sender) or (
                                    "Ubicacion guardada. Que tipo de lugar te interesa para calcular cercania exacta?"
                                )
                            else:
                                respuesta_final = (
                                    "Ubicacion guardada. Ahora puedo recomendarte lugares cercanos con distancia, "
                                    "link de Maps y telefono directo.\n\n"
                                    "Dime que buscas: cenotes, restaurantes, haciendas, museos o playas."
                                )
                        else:
                            respuesta_final = "No recibi coordenadas validas de ubicacion."

                    elif msg_type == "document":
                        document = message.get("document", {})
                        mime_type = document.get("mime_type")
                        media_id = document.get("id")
                        if mime_type == "application/pdf" and media_id:
                            pdf_bytes = descargar_media(media_id)
                            if pdf_bytes:
                                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                                    tmp.write(pdf_bytes)
                                    temp_path = tmp.name
                                try:
                                    analisis = procesar_pdf(temp_path)
                                finally:
                                    try:
                                        os.remove(temp_path)
                                    except OSError:
                                        pass
                                save_message(sender, "user", "[documento pdf]", "document")
                                respuesta_final = procesar_texto_turistico(
                                    sender,
                                    f"El usuario compartio un PDF. Resumen: {analisis}",
                                )
                            else:
                                respuesta_final = "No pude descargar el PDF. Intenta nuevamente."
                        else:
                            respuesta_final = "Solo se soportan documentos PDF."

                    else:
                        respuesta_final = (
                            "Tipo de mensaje no soportado aun. Puedes enviarme texto, imagen, audio, ubicacion o PDF."
                        )

                    if respuesta_final:
                        save_message(sender, "assistant", respuesta_final, "text")
                        enviar_mensaje_whatsapp(sender, respuesta_final)

        return {"status": "ok"}
    except Exception as exc:
        print("ERROR:", exc)
        return {"error": str(exc)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
