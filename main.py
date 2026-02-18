from fastapi import FastAPI, Request
from services.ai_service import procesar_texto
from services.image_service import procesar_imagen
from services.audio_service import transcribir_audio

import requests
import os

app = FastAPI()

VERIFY_TOKEN = "mi_webhook_verificacion_123"
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")


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
# RECEIVE MESSAGES
# -------------------------------
@app.post("/webhook")
async def receive_message(request: Request):
    try:
        body = await request.body()
        if not body:
            return {"status": "empty body"}

        data = await request.json()

        if "entry" not in data:
            return {"status": "no entry"}

        for entry in data["entry"]:
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
                        respuesta_final = procesar_texto(text)

                    elif msg_type == "image":
                        media_id = message["image"]["id"]
                        image_bytes = descargar_media(media_id)
                        respuesta_final = procesar_imagen(image_bytes)

                    elif msg_type == "audio":
                        media_id = message["audio"]["id"]
                        audio_bytes = descargar_media(media_id)
                        respuesta_final = transcribir_audio(audio_bytes)

                    else:
                        respuesta_final = "Tipo de mensaje no soportado aún."

                    if respuesta_final:
                        enviar_mensaje_whatsapp(sender, respuesta_final)

        return {"status": "ok"}

    except Exception as e:
        print("ERROR EN WEBHOOK:", e)
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
            "text": {
                "body": mensaje[:4000]
            }
        }

        resp = requests.post(url, headers=headers, json=payload)

        if resp.status_code != 200:
            print("ERROR WHATSAPP:", resp.status_code, resp.text)

    except Exception as e:
        print("ERROR ENVIANDO MENSAJE:", e)


# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/")
def health():
    return {"status": "running"}


# -------------------------------
# CLOUD RUN START
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
