import os
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribir_audio(audio_bytes: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input_audio.ogg"
        output_path = Path(tmpdir) / "output_audio.wav"

        input_path.write_bytes(audio_bytes)

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(input_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        with output_path.open("rb") as f:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f,
            )

    return transcript.text
