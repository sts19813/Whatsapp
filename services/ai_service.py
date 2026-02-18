# services/ai_service.py

import os
from openai import OpenAI
import fitz  # PyMuPDF

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def procesar_texto(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )
    return response.output_text


def extraer_texto_pdf(path: str) -> str:
    doc = fitz.open(path)
    texto = ""
    for page in doc:
        texto += page.get_text()
    return texto


def procesar_pdf(path: str) -> str:
    texto = extraer_texto_pdf(path)

    response = client.responses.create(
        model="gpt-4.1",
        input=f"Resume o analiza este documento:\n\n{texto}"
    )

    return response.output_text
