import os
import io
import subprocess
from openai import OpenAI
from services.ai_service import procesar_texto

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribir_audio(audio_bytes: bytes) -> str:

    # Guardar archivo original
    with open("/tmp/input_audio.ogg", "wb") as f:
        f.write(audio_bytes)

    # Convertir a WAV
    subprocess.run([
        "ffmpeg",
        "-i", "/tmp/input_audio.ogg",
        "-ar", "16000",
        "-ac", "1",
        "/tmp/output_audio.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Transcribir
    with open("/tmp/output_audio.wav", "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )

    texto_transcrito = transcript.text

    # 🔥 Ahora procesamos con IA
    respuesta = procesar_texto(texto_transcrito)

    return respuesta
