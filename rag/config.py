"""
Configuración del pipeline RAG.

Rutas:
  BASE_DIR, DATA_DIR, DOCS_DIR: dónde están los PDFs de entrada.
  CHROMA_DIR: dónde Chroma persiste la base vectorial (no versionar).

Chunking:
  CHUNK_SIZE / CHUNK_OVERLAP: tamaño y solapamiento al dividir documentos (LangChain).

Retrieval:
  TOP_K: cuántos chunks recuperar por cada consulta (búsqueda semántica).

Embeddings:
  EMBEDDING_MODEL: modelo sentence-transformers para convertir texto → vector.

Ollama:
  OLLAMA_BASE_URL, OLLAMA_MODEL: LLM local; los modelos se ejecutan en tu máquina sin coste.
"""
from pathlib import Path

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"  # Pon aquí los PDFs a indexar
CHROMA_DIR = DATA_DIR / "chroma_db"  # Persistencia del índice vectorial

# Chunking (LangChain RecursiveCharacterTextSplitter)
CHUNK_SIZE = 500   # caracteres por fragmento
CHUNK_OVERLAP = 50 # caracteres de solapamiento entre fragmentos (evita cortes bruscos)

# Retrieval
TOP_K = 5  # número de chunks a devolver por cada pregunta (los más similares)

# Embeddings (modelo local, sin API)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ollama (LLM local; modelos en tu máquina, sin cargos ni APIs de pago)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"  # Nombre del modelo en ollama list (ej. llama3.2, mistral)
