# Miniproyecto RAG: base de datos vectorial

Pipeline RAG local: PDFs → chunking → embeddings (sentence-transformers) → Chroma → consulta por similitud. Usa **Ollama** para el LLM: puedes probar modelos en tu máquina sin coste ni cargos de API.

## Requisitos

- [uv](https://docs.astral.sh/uv/) (o Python 3.11+ con dependencias en `pyproject.toml`)

## Instalación

Desde la raíz del proyecto:

```bash
uv sync
```

Las dependencias del RAG (pypdf, sentence-transformers, chromadb) están en `pyproject.toml`.

## Uso

1. **Coloca los PDFs** en `rag/data/docs/`. Si la carpeta no existe, créala y pon ahí los PDFs.

2. **Indexar** (primera vez o tras añadir/cambiar documentos):

   Desde la raíz del proyecto:

   ```bash
   cd rag
   uv run python main.py index
   ```

   La primera vez descargará el modelo de embeddings (`all-MiniLM-L6-v2`). La base Chroma se guarda en `rag/data/chroma_db/`.

3. **Consultar** (recuperar chunks similares a una pregunta):

   ```bash
   uv run python main.py query "¿Qué dice el documento sobre X?"
   ```

4. **Obtener contexto para un LLM** (texto concatenado de los chunks):

   ```bash
   uv run python main.py context "¿Qué dice el documento sobre X?"
   ```

5. **Preguntar con Ollama** (RAG completo: recupera contexto y responde con el LLM local):

   Este proyecto usa [Ollama](https://ollama.com) para generar las respuestas: los modelos se ejecutan en tu computadora y **no generan cargos** (no hay uso de APIs de pago). Instala Ollama y descarga un modelo (`ollama run llama3.2` o el que prefieras). Luego:

   ```bash
   uv run python main.py ask "¿Qué dice el documento sobre X?"
   ```

   El modelo usará solo el contexto recuperado de tus PDFs para responder.

6. **Visualizar embeddings en 2D y 3D** (sklearn + plotly):

   Tras indexar, se genera una sola página con vista 2D (izq) y 3D (der). Mismo color por documento en ambas; hover con fragmento de texto. Se guarda en `rag/data/embeddings_2d_3d.html`.

   ```bash
   uv run python main.py visualize              # PCA en 2D y 3D (rápido)
   uv run python main.py visualize --method tsne  # t-SNE en 2D, PCA en 3D
   uv run python main.py visualize --no-show   # solo guardar HTML, no abrir navegador
   ```

7. **Interfaz web con Gradio** (para probar sin usar la terminal):

   ```bash
   uv run python main.py ui
   ```

   Se abre en el navegador una app con pestañas: **Preguntar** (RAG con Ollama y ver los chunks usados), **Indexar** (reindexar PDFs de `data/docs`), **Visualizar** (generar el HTML 2D/3D). Útil como base para demos o para quien prefiere una UI.

## Ideas para extender el proyecto

Este proyecto está pensado como base para quien se adentra en LLMs. Puedes sumar:

- **Otros LLMs**: además de Ollama, conectar OpenAI, Anthropic o modelos locales vía LangChain/LiteLLM.
- **Streaming**: que la respuesta del LLM se vaya mostrando token a token (Ollama con `stream=true` lo permite).
- **Subida de PDFs desde la UI**: en Gradio, `gr.File` o `gr.UploadButton` para subir archivos a `data/docs` y reindexar.
- **Evaluación**: métricas de recuperación (precisión/recall) o de respuesta (similitud respuesta–contexto).
- **Tests**: pytest con mocks de Chroma y del modelo de embeddings para CI.
- **Chunking avanzado**: otros splitters (por párrafo, semántico) o distintos tamaños desde config/UI.
- **Múltiples colecciones**: indexar por “proyecto” o carpeta y elegir en la UI sobre qué corpus preguntar.

## Configuración

En `config.py` puedes ajustar:

- `CHUNK_SIZE`, `CHUNK_OVERLAP`: tamaño y solapamiento de los chunks.
- `TOP_K`: cuántos chunks recuperar por consulta.
- `EMBEDDING_MODEL`: otro modelo de [sentence-transformers](https://www.sbt.sbert.net/docs/pretrained_models.html) si lo prefieres.
- `OLLAMA_BASE_URL`: URL de Ollama (por defecto `http://localhost:11434`).
- `OLLAMA_MODEL`: modelo a usar (por defecto `llama3.2`; cámbialo a `llama2`, `mistral`, etc. según lo que tengas). Ollama corre en local y no genera cargos.

## Estructura

- `load_docs_lc.py`: carga de PDFs y splitting con LangChain (PyPDFLoader + RecursiveCharacterTextSplitter).
- `index.py`: embeddings y guardado en Chroma.
- `query.py`: recuperación por similitud y función para contexto.
- `llm_ollama.py`: llamada a Ollama y RAG (prompt con contexto).
- `visualize.py`: visualización 2D y 3D de embeddings (sklearn + plotly, una página con ambas vistas).
- `app_gradio.py`: interfaz web con Gradio (preguntar, indexar, visualizar).
- `main.py`: CLI (index / query / context / ask / visualize / ui).
