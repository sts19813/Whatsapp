import os
import io
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribir_audio(audio_bytes: bytes) -> str:

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.webm"

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file
    )

    return transcript.text
