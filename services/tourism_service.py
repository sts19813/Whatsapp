import math
import re
import unicodedata
import urllib.parse
from typing import Any, Optional

import requests

from services.db_service import get_cached_geocode, save_geocode_cache


STOPWORDS_ES = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "u",
    "a", "en", "por", "para", "con", "sin", "del", "al", "que", "como", "me",
    "te", "le", "les", "mi", "tu", "su", "sus", "es", "son", "hay", "puedo",
    "puedes", "interesa", "interesan", "decir", "sobre", "quiero", "busco",
}

CATEGORY_ALIASES = {
    "cenote": ["cenote", "cenotes", "parador", "balneario"],
    "muse": ["museo", "museos", "galeria", "galerias"],
    "arque": ["arqueologica", "arqueologico", "ruinas", "zona arqueologica"],
    "restaur": ["restaurante", "restaurantes", "comida", "gastronomia", "comer"],
    "bar": ["bar", "bares", "cantina"],
    "hosped": ["hotel", "hoteles", "hospedaje"],
    "playa": ["playa", "playas", "costa", "marina"],
    "hacienda": ["hacienda", "haciendas"],
    "consulado": ["consulado", "consulados"],
}


def normalize_text(value: str) -> str:
    text = (value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text)


def tokenize_text(value: str) -> list[str]:
    raw = re.split(r"[^a-z0-9]+", normalize_text(value))
    return [tok for tok in raw if len(tok) >= 3 and tok not in STOPWORDS_ES]


def build_search_text(place: dict[str, Any]) -> str:
    parts = [
        place.get("categoria", ""),
        place.get("nombre", ""),
        place.get("region", ""),
        place.get("municipio", ""),
        place.get("direccion", ""),
    ]
    return normalize_text(" ".join(parts))


def filter_places(
    places: list[dict[str, Any]],
    query: str,
    municipality: Optional[str] = None,
    category_hint: Optional[str] = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    q = normalize_text(query)
    muni = normalize_text(municipality or "")
    cat = normalize_text(category_hint or "")

    ranked: list[tuple[int, dict[str, Any]]] = []
    query_tokens = tokenize_text(q)

    for place in places:
        search_blob = build_search_text(place)
        search_tokens = set(tokenize_text(search_blob))
        place_muni = normalize_text(place.get("municipio", ""))
        place_cat = normalize_text(place.get("categoria", ""))
        place_name = normalize_text(place.get("nombre", ""))

        if muni and muni not in place_muni:
            continue

        # Si ya hay pista clara de categoria, evita ruido fuera del tema.
        if cat and not category_matches(place, cat):
            if any(alias in q for alias in CATEGORY_ALIASES.get(cat, [cat])):
                continue

        score = 0
        if cat and category_matches(place, cat):
            score += 8

        if not q or q in STOPWORDS_ES:
            score += 1
        elif q in search_blob or q in place_name:
            score += 6

        for token in query_tokens:
            if token in search_tokens:
                score += 3
            elif token in place_name:
                score += 2

        completitud = normalize_text(place.get("completitud", ""))
        if completitud == "alta":
            score += 1

        if score > 0:
            ranked.append((score, place))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [place for _, place in ranked[:limit]]


def infer_category(text: str) -> Optional[str]:
    t = normalize_text(text)
    mapping = {
        "muse": "muse",
        "cenote": "cenote",
        "zona arque": "arque",
        "ruinas": "arque",
        "restaur": "restaur",
        "comer": "restaur",
        "bar": "bar",
        "hotel": "hosped",
        "hosped": "hosped",
        "playa": "playa",
        "hacienda": "hacienda",
        "consulado": "consulado",
    }
    for key, value in mapping.items():
        if key in t:
            return value
    return None


def category_matches(place: dict[str, Any], category_hint: str) -> bool:
    hint = normalize_text(category_hint)
    aliases = CATEGORY_ALIASES.get(hint, [hint])
    blob = build_search_text(place)
    return any(alias in blob for alias in aliases)


def infer_interest_topic(text: str) -> Optional[str]:
    t = normalize_text(text)
    for key, aliases in CATEGORY_ALIASES.items():
        if any(alias in t for alias in aliases):
            if key == "muse":
                return "museos"
            if key == "arque":
                return "zonas arqueologicas"
            if key == "restaur":
                return "gastronomia"
            if key == "hosped":
                return "hospedaje"
            if key == "playa":
                return "playas"
            if key == "hacienda":
                return "haciendas"
            if key == "cenote":
                return "cenotes"
            return key
    return None


def is_exploratory_query(text: str) -> bool:
    t = normalize_text(text)
    starters = [
        "que me puedes decir",
        "que me dices",
        "informacion",
        "tips",
        "consejos",
        "me interesa",
        "me interesan",
        "como funciona",
        "que hay",
        "cuando conviene",
    ]
    if any(s in t for s in starters):
        return True
    return infer_interest_topic(t) is not None and "recom" not in t


def infer_intent(text: str) -> str:
    t = normalize_text(text)

    logistics_terms = [
        "como llegar",
        "como llego",
        "como ir",
        "ruta",
        "ir a",
        "llego en",
        "tiempo",
        "cuanto tardo",
        "cuanto tiempo",
        "distancia",
        "trafico",
        "clima",
        "lluvia",
        "precauciones",
        "seguridad",
    ]
    if any(term in t for term in logistics_terms):
        return "logistics"

    recommendation_terms = [
        "recomiend",
        "sugier",
        "que hacer",
        "lugares",
        "donde ir",
        "restaurante",
        "cenote",
        "playa",
        "museo",
        "hotel",
        "hacienda",
    ]
    if any(term in t for term in recommendation_terms):
        return "recommendation"

    return "general"


def find_best_place_match(
    places: list[dict[str, Any]],
    text: str,
    municipality: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    q = normalize_text(text)
    muni = normalize_text(municipality or "")
    q_tokens = [tok for tok in re.split(r"[^a-z0-9]+", q) if len(tok) > 2]
    if not q_tokens:
        return None

    best_place = None
    best_score = 0

    for place in places:
        place_muni = normalize_text(place.get("municipio", ""))
        if muni and muni not in place_muni:
            continue

        name = normalize_text(place.get("nombre", ""))
        if not name:
            continue

        score = 0
        if name in q:
            score += 15
        if q in name:
            score += 12

        name_tokens = [tok for tok in re.split(r"[^a-z0-9]+", name) if len(tok) > 2]
        overlap = len(set(name_tokens) & set(q_tokens))
        score += overlap * 3

        if len(name_tokens) > 0 and overlap >= max(1, len(name_tokens) // 2):
            score += 4

        if score > best_score:
            best_score = score
            best_place = place

    return best_place if best_score >= 5 else None


def maps_link_from_text(address: str, name: str = "") -> str:
    q = urllib.parse.quote_plus(f"{name} {address}".strip())
    return f"https://www.google.com/maps/search/?api=1&query={q}"


def tel_link(phone: str) -> Optional[str]:
    if not phone:
        return None
    digits = re.sub(r"[^0-9+]", "", phone)
    if not digits:
        return None
    return f"tel:{digits}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def geocode_place(place: dict[str, Any]) -> Optional[dict[str, float]]:
    address = (place.get("direccion") or "").strip()
    municipality = (place.get("municipio") or "").strip()
    name = (place.get("nombre") or "").strip()
    query = ", ".join([x for x in [name, address, municipality, "Yucatan", "Mexico"] if x])
    return geocode_text(query)


def geocode_text(query: str) -> Optional[dict[str, float]]:
    clean_query = (query or "").strip()
    if not clean_query:
        return None

    cached = get_cached_geocode(clean_query)
    if cached:
        return cached

    params = {"q": clean_query, "format": "json", "limit": 1}
    headers = {"User-Agent": "YucatanTurismoBot/1.0"}

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
            timeout=8,
        )
        data = response.json() if response.ok else []
        if not data:
            return None

        first = data[0]
        lat = float(first["lat"])
        lon = float(first["lon"])
        save_geocode_cache(clean_query, lat, lon, "nominatim")
        return {"lat": lat, "lon": lon}
    except Exception:
        return None


def estimate_drive_route(
    origin: dict[str, float],
    destination: dict[str, float],
) -> dict[str, float]:
    try:
        url = (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{origin['lon']},{origin['lat']};{destination['lon']},{destination['lat']}"
        )
        response = requests.get(url, params={"overview": "false"}, timeout=8)
        data = response.json() if response.ok else {}
        routes = data.get("routes") or []
        if routes:
            route = routes[0]
            return {
                "distance_km": round(float(route["distance"]) / 1000.0, 2),
                "duration_min": round(float(route["duration"]) / 60.0),
                "source": "osrm",
            }
    except Exception:
        pass

    distance_km = haversine_km(origin["lat"], origin["lon"], destination["lat"], destination["lon"])
    # Fallback conservador para carretera/ciudad en Yucatan.
    duration_min = max(5, round((distance_km / 55.0) * 60))
    return {"distance_km": round(distance_km, 2), "duration_min": duration_min, "source": "fallback"}


def maps_directions_link(origin: dict[str, float], destination: dict[str, float]) -> str:
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={origin['lat']},{origin['lon']}"
        f"&destination={destination['lat']},{destination['lon']}"
        "&travelmode=driving"
    )


def weather_snapshot(lat: float, lon: float) -> Optional[dict[str, Any]]:
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
                "timezone": "auto",
            },
            timeout=8,
        )
        data = response.json() if response.ok else {}
        current = data.get("current")
        return current if isinstance(current, dict) else None
    except Exception:
        return None


def weather_label(code: int) -> str:
    mapping = {
        0: "despejado",
        1: "mayormente despejado",
        2: "parcialmente nublado",
        3: "nublado",
        45: "niebla",
        48: "niebla con escarcha",
        51: "llovizna ligera",
        53: "llovizna",
        55: "llovizna intensa",
        61: "lluvia ligera",
        63: "lluvia",
        65: "lluvia fuerte",
        80: "chubascos",
        81: "chubascos moderados",
        82: "chubascos fuertes",
        95: "tormenta",
    }
    return mapping.get(code, "condicion variable")


def travel_advice(weather: Optional[dict[str, Any]]) -> str:
    tips = []
    if not weather:
        return "Lleva agua, bloqueador, efectivo y confirma horario antes de salir."

    precipitation = float(weather.get("precipitation", 0) or 0)
    wind = float(weather.get("wind_speed_10m", 0) or 0)
    temp = float(weather.get("temperature_2m", 0) or 0)

    if precipitation > 0:
        tips.append("Puede llover, lleva impermeable y maneja con mayor distancia de frenado.")
    if wind >= 25:
        tips.append("Hay viento notable; asegura objetos sueltos y evita maniobras bruscas.")
    if temp >= 33:
        tips.append("Hace calor alto; hidratarse y usar bloqueador cada 2-3 horas.")
    if not tips:
        tips.append("Condiciones razonables para conducir; revisa gasolina y horario de regreso.")

    return " ".join(tips)


def enrich_places_with_distance(
    places: list[dict[str, Any]],
    user_location: Optional[dict[str, float]],
) -> list[dict[str, Any]]:
    if not user_location:
        return places

    output: list[dict[str, Any]] = []
    for place in places:
        address = (place.get("direccion") or "").strip()
        municipality = (place.get("municipio") or "").strip()
        name = (place.get("nombre") or "").strip()
        query = ", ".join([x for x in [name, address, municipality, "Yucatan", "Mexico"] if x])

        coords = geocode_text(query)
        if coords:
            distance = haversine_km(
                user_location["lat"],
                user_location["lon"],
                coords["lat"],
                coords["lon"],
            )
            new_place = dict(place)
            new_place["distance_km"] = round(distance, 2)
            output.append(new_place)
        else:
            output.append(place)

    output.sort(key=lambda p: p.get("distance_km", 999999.0))
    return output


def format_places_for_whatsapp(places: list[dict[str, Any]], limit: int = 5) -> str:
    if not places:
        return ""

    chunks = []
    for place in places[:limit]:
        name = place.get("nombre", "Sin nombre")
        category = place.get("categoria", "")
        address = place.get("direccion", "")
        municipality = place.get("municipio", "")
        phone = place.get("telefono", "")
        web = place.get("web", "")
        source = place.get("fuente_url", "")
        maps = maps_link_from_text(address, name)
        phone_link = tel_link(phone)

        lines = [f"*{name}*", f"Categoria: {category}"]
        if municipality:
            lines.append(f"Municipio: {municipality}")
        if address:
            lines.append(f"Direccion: {address}")
        if "distance_km" in place:
            lines.append(f"Distancia aprox: {place['distance_km']} km")
        lines.append(f"Maps: {maps}")
        if phone:
            lines.append(f"Tel: {phone}")
        if phone_link:
            lines.append(f"Llamar: {phone_link}")
        if web:
            lines.append(f"Web: {web}")
        if source:
            lines.append(f"Fuente: {source}")

        chunks.append("\n".join(lines))

    return "\n\n".join(chunks)
