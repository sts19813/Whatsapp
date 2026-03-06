import os
import base64
from openai import OpenAI
from PIL import Image
import io

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def procesar_imagen(image_bytes: bytes) -> str:

    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != "RGB":
        image = image.convert("RGB")

    image.thumbnail((700, 700))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70, optimize=True)
    compressed_bytes = buffer.getvalue()

    image_base64 = base64.b64encode(compressed_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": """
Describe la imagen en detalle.

Si hay texto visible, extráelo.

Si aparece ropa, artesanía, comida o algo relacionado con turismo,
menciónalo claramente.
"""
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                }
            ]
        }]
    )

    return response.output_text