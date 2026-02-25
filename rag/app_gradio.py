"""
Interfaz web con Gradio para el RAG.

Pensado como base para quien se adentra en LLMs: probar el RAG desde el navegador
sin usar la terminal, ver la respuesta y los chunks recuperados.
"""
import logging
from pathlib import Path

import gradio as gr

from config import DOCS_DIR
from index import build_index
from query import retrieve
from llm_ollama import rag_ask
from visualize import run_visualize, VISUALIZE_OUTPUT

log = logging.getLogger(__name__)


def _ask_rag(question: str) -> tuple[str, str]:
    """Ejecuta RAG y devuelve (respuesta, chunks_usados)."""
    if not question or not question.strip():
        return "Escribe una pregunta.", ""
    try:
        chunks = retrieve(question.strip())
        answer = rag_ask(question.strip())
        chunks_text = "\n\n---\n\n".join(
            f"[{c['source']}] (dist={c['distance']:.3f})\n{c['text'][:300]}{'...' if len(c['text']) > 300 else ''}"
            for c in chunks
        )
        return answer, chunks_text or "(sin chunks)"
    except RuntimeError as e:
        return str(e), ""
    except Exception as e:
        log.exception("Error en RAG")
        return f"Error: {e}", ""


def _reindex() -> str:
    """Reindexa los PDFs en data/docs."""
    try:
        if not DOCS_DIR.exists():
            return f"Crea la carpeta {DOCS_DIR} y pon ahí tus PDFs, luego presiona **Reindexar**."
        build_index(force_recreate=True)
        return "Índice actualizado correctamente."
    except Exception as e:
        log.exception("Error al indexar")
        return f"Error: {e}"


def _generate_viz() -> str:
    """Genera el HTML 2D/3D y devuelve mensaje con la ruta."""
    try:
        run_visualize(show=False)
        return f"Listo. Abre el archivo en tu navegador:\n{VISUALIZE_OUTPUT}"
    except Exception as e:
        log.exception("Error al visualizar")
        return f"Error: {e}"


def build_ui() -> gr.Blocks:
    """Construye la interfaz Gradio: Preguntar, Indexar, Visualizar."""
    with gr.Blocks(title="RAG — Pregunta sobre tus PDFs", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# RAG sobre tus documentos\n"
            "Pregunta en lenguaje natural; el modelo responde usando solo el contenido de los PDFs indexados.\n\n"
            "**Ollama** ejecuta el LLM en tu máquina: puedes probar modelos **sin coste ni cargos** de API."
        )

        with gr.Tabs():
            with gr.Tab("Preguntar"):
                q = gr.Textbox(
                    label="Pregunta",
                    placeholder="Ej: ¿Qué dice el documento sobre UX?",
                    lines=2,
                )
                ask_btn = gr.Button("Enviar (RAG con Ollama)")
                answer_out = gr.Textbox(label="Respuesta", lines=8, interactive=False)
                chunks_out = gr.Textbox(label="Chunks usados como contexto", lines=6, interactive=False)
                ask_btn.click(fn=_ask_rag, inputs=q, outputs=[answer_out, chunks_out])

            with gr.Tab("Indexar"):
                gr.Markdown(f"Los PDFs se leen desde `{DOCS_DIR}`. Pon ahí tus archivos y presiona **Reindexar**.")
                reindex_btn = gr.Button("Reindexar")
                reindex_out = gr.Textbox(label="Estado", lines=2, interactive=False)
                reindex_btn.click(fn=_reindex, inputs=[], outputs=reindex_out)

            with gr.Tab("Visualizar"):
                gr.Markdown("Genera el gráfico 2D/3D de los embeddings (primero indexa). Se guarda un HTML que puedes abrir en el navegador.")
                viz_btn = gr.Button("Generar visualización 2D/3D")
                viz_out = gr.Textbox(label="Ruta del archivo", lines=2, interactive=False)
                viz_btn.click(fn=_generate_viz, inputs=[], outputs=viz_out)

        gr.Markdown("---\n*Proyecto base RAG — LangChain, Chroma, **Ollama** (modelos locales, sin cargos). Puedes extenderlo con más documentos, otros LLMs o evaluación.*")

    return demo


def launch_ui(share: bool = False, server_port: int | None = None):
    """Lanza la app Gradio."""
    demo = build_ui()
    demo.launch(share=share, server_port=server_port)
