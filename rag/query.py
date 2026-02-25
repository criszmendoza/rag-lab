"""
Recuperación: búsqueda por similitud en Chroma.

- Abre la colección indexada (vectores + metadatos).
- Convierte la pregunta a vector con el mismo modelo que se usó al indexar.
- Devuelve los top_k chunks más cercanos (menor distancia = más similares).
"""
import logging

import chromadb

from config import CHROMA_DIR, TOP_K
from index import get_embedding_model

log = logging.getLogger(__name__)


def get_collection(collection_name: str = "rag_docs"):
    """
    Abre la colección Chroma donde están los vectores indexados.

    Pasos:
      1. Conectar al mismo directorio persistente que en index (CHROMA_DIR).
      2. Obtener la colección por nombre (debe existir; crear antes con build_index).
    """
    log.debug("Abriendo colección Chroma: %s", CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=collection_name)


def retrieve(
    question: str,
    top_k: int = TOP_K,
    collection_name: str = "rag_docs",
) -> list[dict]:
    """
    Busca los chunks más similares a la pregunta (búsqueda semántica).

    Pasos:
      1. Abrir la colección Chroma (donde están los vectores de los chunks).
      2. Cargar el mismo modelo de embeddings y convertir la pregunta a vector.
      3. Llamar a collection.query con ese vector y n_results=top_k.
      4. Chroma devuelve documentos, metadatos y distancias (menor = más similar).
      5. Formatear como lista de dicts con "text", "source", "chunk_index", "distance".

    Returns:
      Lista de hasta top_k chunks, ordenados por similitud (más similar primero).
    """
    log.info("Paso 1/3 Recuperación: abriendo base vectorial (Chroma).")
    collection = get_collection(collection_name)

    log.info("Paso 2/3 Recuperación: generando embedding de la pregunta.")
    model = get_embedding_model()
    query_embedding = model.encode([question]).tolist()

    log.info("Paso 3/3 Recuperación: buscando %s chunks más similares.", top_k)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma devuelve listas por query; aquí solo hay una pregunta
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    chunks = [
        {
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]
    log.info("Recuperados %s chunks (fuentes: %s).", len(chunks), [c["source"] for c in chunks])
    return chunks


def get_context_for_prompt(question: str, top_k: int = TOP_K) -> str:
    """
    Obtiene el texto concatenado de los chunks recuperados para el prompt del LLM.

    Pasos:
      1. Llamar a retrieve(question, top_k) para obtener los chunks más relevantes.
      2. Unir solo el texto de cada chunk con un separador (---) para inyectar en el prompt.

    Returns:
      Un único string con todo el contexto a pasar al LLM.
    """
    chunks = retrieve(question, top_k=top_k)
    return "\n\n---\n\n".join(c["text"] for c in chunks)
