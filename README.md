# rag-lab / RAG base

Proyecto base para experimentar con **RAG** (Retrieval-Augmented Generation) y LLMs. Pensado para quien se adentra en el mundo de los LLMs y quiere tener una base clara para construir herramientas más robustas.

## Contenido principal: `rag/`

En la carpeta **rag** está el miniproyecto RAG completo:

- **Carga de PDFs** con LangChain (PyPDFLoader + RecursiveCharacterTextSplitter).
- **Embeddings** con sentence-transformers y **base vectorial** Chroma.
- **Consulta** por similitud y **respuesta** con **Ollama** (LLM local; permite probar modelos sin coste ni cargos).
- **Visualización** 2D/3D de embeddings (sklearn + plotly).
- **Interfaz web** con Gradio (preguntar, indexar, visualizar desde el navegador).

Documentación detallada y uso: **[rag/README.md](rag/README.md)**.

## Requisitos

- **Python** 3.11 o 3.12 (ver `pyproject.toml`).
- **[uv](https://docs.astral.sh/uv/)** recomendado para instalar dependencias.
- **Ollama** (opcional, para las respuestas del LLM): [ollama.com](https://ollama.com). Los modelos se ejecutan en tu máquina; no genera cargos ni uso de APIs de pago.

## Inicio rápido

```bash
# Clonar (o descargar) y entrar al repo
cd rag-lab

# Instalar dependencias
uv sync

# Ir al RAG e indexar tus PDFs
cd rag
uv run python main.py index    # pon tus PDFs en rag/data/docs/ antes

# Preguntar por terminal
uv run python main.py ask "Tu pregunta"

# O abrir la interfaz web
uv run python main.py ui
```

## Estructura del repositorio

```
rag-lab/
├── README.md           # Este archivo
├── LICENSE             # MIT
├── pyproject.toml      # Dependencias y configuración (uv)
├── uv.lock             # Lock de dependencias
├── .gitignore
├── .env.example        # Plantilla de variables de entorno (copiar a .env)
└── rag/           # Miniproyecto RAG
    ├── README.md       # Documentación del RAG
    ├── main.py         # CLI (index, query, ask, visualize, ui)
    ├── config.py
    ├── load_docs_lc.py
    ├── index.py
    ├── query.py
    ├── llm_ollama.py
    ├── visualize.py
    ├── app_gradio.py
    └── data/
        ├── docs/       # PDFs a indexar
        └── chroma_db/  # Índice vectorial (generado, no versionar)
```

## Licencia

MIT. Ver [LICENSE](LICENSE).
