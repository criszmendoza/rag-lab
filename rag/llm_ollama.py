"""
LLM con Ollama: generación de respuesta a partir del contexto RAG.

- ask_ollama: envía un prompt a la API /api/generate de Ollama y devuelve el texto generado.
- rag_ask: obtiene contexto con get_context_for_prompt, arma el prompt y llama a Ollama.
"""
import json
import logging
import urllib.error
import urllib.request

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from query import get_context_for_prompt

log = logging.getLogger(__name__)


def ask_ollama(prompt: str, model: str | None = None, base_url: str | None = None) -> str:
    """
    Envía un prompt a Ollama y devuelve la respuesta generada.

    Pasos:
      1. Resolver modelo y URL base (config o argumentos).
      2. Construir el body JSON: model, prompt, stream=false (respuesta completa de una vez).
      3. Hacer POST a /api/generate; timeout 120 s (modelos grandes pueden tardar).
      4. Leer la respuesta JSON y extraer el campo "response" (texto generado).
      5. Si falla la conexión, lanzar RuntimeError con mensaje claro.

    Returns:
      El texto generado por el modelo (sin el prompt).
    """
    model = model or OLLAMA_MODEL
    base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
    url = f"{base_url}/api/generate"

    log.info("Paso 2/2 LLM: enviando prompt a Ollama (modelo=%s, prompt=%s caracteres).", model, len(prompt))

    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"No se pudo conectar con Ollama en {base_url}. ¿Está corriendo? (ollama serve)"
        ) from e

    response = data.get("response", "").strip()
    log.info("Ollama respondió: %s caracteres.", len(response))
    return response


def rag_ask(
    question: str,
    model: str | None = None,
    top_k: int = 5,
) -> str:
    """
    RAG completo: recuperar contexto + generar respuesta con Ollama.

    Pasos:
      1. Obtener el contexto: get_context_for_prompt(question, top_k) → texto de los chunks similares.
      2. Construir el prompt: instrucción + contexto + pregunta (el modelo debe responder solo con el contexto).
      3. Llamar a ask_ollama con ese prompt y devolver la respuesta del LLM.
    """
    log.info("Paso 1/2 Recuperación: obteniendo contexto desde la base vectorial (top_k=%s).", top_k)
    context = get_context_for_prompt(question, top_k=top_k)
    log.info("Contexto listo: %s caracteres para el prompt.", len(context))

    prompt = (
        "Usa únicamente el siguiente contexto para responder la pregunta. "
        "Si la respuesta no está en el contexto, di que no tienes esa información.\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n"
        "Respuesta:"
    )
    return ask_ollama(prompt, model=model)
