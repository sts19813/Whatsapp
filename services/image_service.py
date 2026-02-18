import os
import base64
from openai import OpenAI
from PIL import Image
import io

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def procesar_imagen(image_bytes: bytes) -> str:
    # Convertir bytes a imagen PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Convertir a RGB por seguridad (evita errores con PNG/alpha)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Redimensionar manteniendo proporción (máximo 700x700)
    image.thumbnail((700, 700))

    # Comprimir a JPEG con calidad reducida
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70, optimize=True)
    compressed_bytes = buffer.getvalue()

    # Convertir a base64
    image_base64 = base64.b64encode(compressed_bytes).decode("utf-8")

    # Enviar a OpenAI
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe la imagen y extrae cualquier texto visible."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                }
            ]
        }]
    )

    return response.output_text
