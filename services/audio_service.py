import os
import subprocess
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribir_audio(audio_bytes: bytes) -> str:

    with open("/tmp/input_audio.ogg", "wb") as f:
        f.write(audio_bytes)

    subprocess.run([
        "ffmpeg",
        "-i", "/tmp/input_audio.ogg",
        "-ar", "16000",
        "-ac", "1",
        "/tmp/output_audio.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open("/tmp/output_audio.wav", "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )

    texto_transcrito = transcript.text

    return texto_transcrito