"""
CLI del miniproyecto RAG.

Uso:
  python main.py index         -> indexar PDFs en data/docs y guardar en Chroma
  python main.py query "pregunta" -> recuperar chunks similares a la pregunta
  python main.py context "pregunta" -> mismo que query pero imprime solo el texto para el prompt
  python main.py ask "pregunta"    -> RAG con Ollama: recupera contexto y responde con el LLM local
  python main.py visualize         -> visualizar embeddings en 2D y 3D (una página), guarda HTML
  python main.py visualize --method tsne  -> t-SNE en 2D, PCA en 3D
  python main.py visualize --no-show      -> solo guardar HTML sin abrir navegador
  python main.py ui                       -> interfaz web Gradio (preguntar, indexar, visualizar)
"""
import logging
import sys
from pathlib import Path

# Permitir importar módulos de este paquete al ejecutar como script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DOCS_DIR
from index import build_index
from query import get_context_for_prompt, retrieve
from llm_ollama import rag_ask
from visualize import run_visualize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    """
    Punto de entrada del CLI. Enruta a index, query, context o ask.

    Pasos:
      1. Leer comando (argv[1]) y validar que haya argumentos suficientes.
      2. index: comprobar que exista DOCS_DIR (crear si no); llamar build_index(force_recreate=True).
      3. query: recuperar chunks con retrieve(), imprimir cada chunk con source y distancia.
      4. context: recuperar y imprimir solo el texto concatenado (para copiar al prompt).
      5. ask: llamar rag_ask(question) (recuperación + Ollama) e imprimir la respuesta; capturar errores de conexión.
      6. visualize: cargar vectores de Chroma, reducir a 2D y 3D, graficar con plotly y guardar HTML.
      7. ui: lanzar la interfaz Gradio (preguntar, indexar, visualizar desde el navegador).
      8. Si el comando no es reconocido, mostrar ayuda y salir con código 1.
    """
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "index":
        # Paso 2: asegurar directorio de documentos antes de indexar
        if not DOCS_DIR.exists():
            print(f"Creando directorio de documentos: {DOCS_DIR}")
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            print("Coloca PDFs en ese directorio y vuelve a ejecutar 'python main.py index'.")
            sys.exit(0)
        build_index(force_recreate=True)
        return

    if cmd in ("query", "context", "ask"):
        if len(sys.argv) < 3:
            print("Uso: python main.py query \"tu pregunta aquí\"")
            sys.exit(1)
        question = " ".join(sys.argv[2:])

        if cmd == "query":
            # Paso 3: mostrar chunks con metadatos
            chunks = retrieve(question)
            for i, c in enumerate(chunks, 1):
                print(f"\n--- Chunk {i} (source={c['source']}, dist={c['distance']:.4f}) ---")
                print(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
        elif cmd == "context":
            # Paso 4: solo el texto para inyectar en un prompt
            print(get_context_for_prompt(question))
        else:
            # Paso 5: RAG completo (recuperación + LLM)
            try:
                log = logging.getLogger(__name__)
                log.info("Iniciando RAG: pregunta=%s", repr(question[:60] + ("..." if len(question) > 60 else "")))
                answer = rag_ask(question)
                log.info("RAG finalizado correctamente.")
                print(answer)
            except RuntimeError as e:
                print(e)
                sys.exit(1)
        return

    if cmd == "visualize":
        # Paso 6: visualización 2D de embeddings (sklearn + plotly)
        method = "pca"
        if "--method" in sys.argv:
            i = sys.argv.index("--method")
            if i + 1 < len(sys.argv) and sys.argv[i + 1].lower() == "tsne":
                method = "tsne"
        show = "--no-show" not in sys.argv
        try:
            run_visualize(method=method, show=show)
        except ValueError as e:
            print(e)
            sys.exit(1)
        return

    if cmd == "ui":
        # Paso 7: interfaz web Gradio
        from app_gradio import launch_ui
        launch_ui()
        return

    # Paso 8: comando desconocido
    print(__doc__)
    sys.exit(1)


if __name__ == "__main__":
    main()
