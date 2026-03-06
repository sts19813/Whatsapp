from fastapi import FastAPI, Request
from services.ai_service import procesar_texto
from services.image_service import procesar_imagen
from services.audio_service import transcribir_audio

import requests
import os
import json

app = FastAPI()

VERIFY_TOKEN = "mi_webhook_verificacion_123"
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

# -------------------------------
# CARGAR BASE TURISMO
# -------------------------------

with open("base_turismo_yucatan_inventur.json", encoding="utf-8") as f:
    BASE_TURISMO = json.load(f)

# -------------------------------
# MEMORIA
# -------------------------------

USER_STATE = {}
CHAT_MEMORY = {}

def guardar_memoria(user, mensaje):
    if user not in CHAT_MEMORY:
        CHAT_MEMORY[user] = []
    CHAT_MEMORY[user].append(mensaje)
    if len(CHAT_MEMORY[user]) > 12:
        CHAT_MEMORY[user] = CHAT_MEMORY[user][-12:]

def obtener_contexto(user):
    return "\n".join(CHAT_MEMORY.get(user, []))

# -------------------------------
# CONFIG BOT
# -------------------------------

BOT_NAME = "Itzamná AI"

MENSAJE_BIENVENIDA = f"""
¡Hola! 👋

Puedo ayudarte a encontrar:

🏛 Museos  
🍽 Restaurantes  
🌴 Playas  
🏺 Zonas arqueológicas  
🍹 Bares  
🏨 Hoteles  
🌿 Cenotes  

Primero dime:

📍 ¿En qué ciudad o municipio estás?
(Ejemplo: Mérida, Progreso, Valladolid)
"""

# -------------------------------
# BUSCADOR MEJORADO
# -------------------------------

def buscar_lugares(texto, municipio=None):

    texto = texto.lower()

    resultados = []

    for lugar in BASE_TURISMO:

        categoria = (lugar.get("categoria") or "").lower()
        nombre = (lugar.get("nombre") or "").lower()
        region = (lugar.get("region") or "").lower()
        muni = (lugar.get("municipio") or "").lower()

        if municipio and municipio.lower() not in muni:
            continue

        if (
            texto in categoria
            or texto in nombre
            or texto in region
        ):
            resultados.append(lugar)

    return resultados[:5]

# -------------------------------
# BUSQUEDA POR CATEGORIA
# -------------------------------

def buscar_por_categoria(categoria, municipio=None):

    resultados = []

    for lugar in BASE_TURISMO:

        cat = (lugar.get("categoria") or "").lower()
        muni = (lugar.get("municipio") or "").lower()

        if categoria.lower() in cat:

            if municipio and municipio.lower() not in muni:
                continue

            resultados.append(lugar)

    return resultados[:5]

# -------------------------------
# FORMATEAR RESPUESTA
# -------------------------------

def formatear_lugares(lugares):

    if not lugares:
        return None

    texto = ""

    for lugar in lugares:

        nombre = lugar.get("nombre", "")
        direccion = lugar.get("direccion", "")
        telefono = lugar.get("telefono", "")

        texto += f"📍 *{nombre}*\n"

        if direccion:
            texto += f"Dirección: {direccion}\n"

        if telefono:
            texto += f"Tel: {telefono}\n"

        texto += "\n"

    return texto

# -------------------------------
# DETECTAR INTENCION
# -------------------------------

def detectar_categoria(texto):

    t = texto.lower()

    if "museo" in t:
        return "Museos"

    if "cenote" in t:
        return "Parador"

    if "zona arqueologica" in t or "ruinas" in t:
        return "Zonas"

    if "restaurante" in t or "comer" in t:
        return "Restaurante"

    if "bar" in t or "tomar" in t:
        return "Bar"

    if "hotel" in t:
        return "Hospedaje"

    if "playa" in t:
        return "Playa"

    return None

# -------------------------------
# PROCESAR MENSAJE
# -------------------------------

def procesar_mensaje_turistico(sender, texto):

    texto_lower = texto.lower()

    if sender not in USER_STATE:
        USER_STATE[sender] = {}

    user = USER_STATE[sender]

    guardar_memoria(sender, f"Usuario: {texto}")

    if "hola" in texto_lower:
        return MENSAJE_BIENVENIDA

    # detectar municipio
    municipios = ["merida","progreso","valladolid","celestun","tekax","tinum","muna"]

    for m in municipios:
        if m in texto_lower:
            user["municipio"] = m.capitalize()

            lugares = buscar_lugares("", m)

            texto_lugares = formatear_lugares(lugares)

            return f"""
Perfecto 👍

Entonces estás en *{m.capitalize()}*.

Aquí algunos lugares interesantes:

{texto_lugares}

¿Qué te gustaría buscar?

🍽 Restaurantes  
🏛 Museos  
🌴 Playas  
🏺 Zonas arqueológicas  
🍹 Bares  
🏨 Hoteles  
🌿 Cenotes
"""

    municipio = user.get("municipio")

    categoria = detectar_categoria(texto)

    if categoria:

        lugares = buscar_por_categoria(categoria, municipio)

        texto_lugares = formatear_lugares(lugares)

        if texto_lugares:

            return f"""
Te recomiendo estos lugares en {municipio}:

{texto_lugares}

¿Quieres más recomendaciones?
"""

    # búsqueda semántica
    lugares = buscar_lugares(texto, municipio)

    texto_lugares = formatear_lugares(lugares)

    if texto_lugares:

        return f"""
Encontré estos lugares en {municipio}:

{texto_lugares}

¿Te interesa alguno?
"""

    return None

# -------------------------------
# VERIFY WEBHOOK
# -------------------------------

@app.get("/webhook")
async def verify_webhook(request: Request):

    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return int(challenge)

    return {"error": "Verification failed"}

# -------------------------------
# RECEIVE MESSAGE
# -------------------------------

@app.post("/webhook")
async def receive_message(request: Request):

    try:

        data = await request.json()

        for entry in data.get("entry", []):
            for change in entry.get("changes", []):

                value = change.get("value", {})

                if "messages" not in value:
                    continue

                for message in value["messages"]:

                    sender = message.get("from")
                    msg_type = message.get("type")

                    respuesta_final = None

                    if msg_type == "text":

                        text = message["text"]["body"]

                        respuesta_final = procesar_mensaje_turistico(sender, text)

                        if not respuesta_final:

                            contexto = obtener_contexto(sender)

                            respuesta_final = procesar_texto(f"""
Eres un asistente turístico experto en Yucatán.

Contexto conversación:
{contexto}

Usuario dice:
{text}

Responde corto y útil.
""")

                        guardar_memoria(sender, f"Bot: {respuesta_final}")

                    elif msg_type == "image":

                        media_id = message["image"]["id"]
                        image_bytes = descargar_media(media_id)

                        descripcion = procesar_imagen(image_bytes)

                        contexto = obtener_contexto(sender)

                        respuesta_final = procesar_texto(f"""
Descripción de imagen:
{descripcion}

Contexto:
{contexto}

Responde útil para turismo en Yucatán.
""")

                    elif msg_type == "audio":

                        media_id = message["audio"]["id"]
                        audio_bytes = descargar_media(media_id)

                        texto_audio = transcribir_audio(audio_bytes)

                        respuesta_final = procesar_mensaje_turistico(sender, texto_audio)

                        if not respuesta_final:

                            contexto = obtener_contexto(sender)

                            respuesta_final = procesar_texto(f"""
Contexto conversación:
{contexto}

Usuario dijo por audio:
{texto_audio}

Responde útil.
""")

                    elif msg_type == "location":

                        lat = message["location"]["latitude"]
                        lon = message["location"]["longitude"]

                        USER_STATE.setdefault(sender, {})
                        USER_STATE[sender]["location"] = {"lat": lat, "lon": lon}

                        respuesta_final = f"""
Ubicación recibida 📍

Latitud: {lat}
Longitud: {lon}

Puedes preguntarme por:

🍽 Restaurantes
🏛 Museos
🏺 Zonas arqueológicas
🌴 Playas
🍹 Bares
"""

                    elif msg_type == "document":

                        document = message.get("document", {})
                        mime_type = document.get("mime_type")

                        if mime_type == "application/pdf":

                            media_id = document.get("id")
                            pdf_bytes = descargar_media(media_id)

                            temp_path = "/tmp/documento.pdf"

                            with open(temp_path, "wb") as f:
                                f.write(pdf_bytes)

                            from services.ai_service import procesar_pdf
                            respuesta_final = procesar_pdf(temp_path)

                        else:
                            respuesta_final = "Solo se soportan documentos PDF."

                    else:
                        respuesta_final = "Tipo de mensaje no soportado."

                    if respuesta_final:
                        enviar_mensaje_whatsapp(sender, respuesta_final)

        return {"status": "ok"}

    except Exception as e:

        print("ERROR:", e)
        return {"error": str(e)}

# -------------------------------
# DOWNLOAD MEDIA
# -------------------------------

def descargar_media(media_id):

    try:

        url = f"https://graph.facebook.com/v22.0/{media_id}"

        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

        r = requests.get(url, headers=headers)
        media_info = r.json()

        media_url = media_info.get("url")

        media_response = requests.get(media_url, headers=headers)

        return media_response.content

    except Exception as e:

        print("ERROR DESCARGANDO MEDIA:", e)
        return None

# -------------------------------
# SEND MESSAGE
# -------------------------------

def enviar_mensaje_whatsapp(numero, mensaje):

    try:

        url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"

        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "messaging_product": "whatsapp",
            "to": numero,
            "type": "text",
            "text": {"body": mensaje[:4000]}
        }

        requests.post(url, headers=headers, json=payload)

    except Exception as e:

        print("ERROR ENVIANDO MENSAJE:", e)

# -------------------------------
# HEALTH
# -------------------------------

@app.get("/")
def health():
    return {"status": "running"}

# -------------------------------
# START
# -------------------------------

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(app, host="0.0.0.0", port=port)