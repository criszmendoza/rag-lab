"""
Indexación: de texto a vectores en Chroma.

- Carga chunks con LangChain (load_docs_lc).
- Convierte cada chunk a vector con sentence-transformers.
- Persiste vectores + metadatos + texto en Chroma para búsqueda por similitud.
"""
from sentence_transformers import SentenceTransformer
import chromadb

from config import CHROMA_DIR, EMBEDDING_MODEL
from load_docs_lc import load_and_chunk_docs_lc


def get_embedding_model() -> SentenceTransformer:
    """
    Devuelve el modelo de embeddings (carga bajo demanda).

    Pasos:
      1. Instanciar SentenceTransformer con el modelo de config (ej. all-MiniLM-L6-v2).
      2. La primera vez descarga pesos; luego usa caché local.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def build_index(collection_name: str = "rag_docs", force_recreate: bool = False):
    """
    Construye (o reconstruye) el índice vectorial en Chroma.

    Pasos:
      1. Conectar a Chroma en modo persistente (ruta en disco).
      2. Si force_recreate y la colección existe, borrarla para empezar de cero.
      3. Obtener o crear la colección donde se guardarán los vectores.
      4. Cargar y trocear documentos con LangChain → lista de chunks.
      5. Extraer textos, metadatos e ids de cada chunk (ids únicos: source + índice).
      6. Cargar el modelo de embeddings y convertir todos los textos a vectores.
      7. Añadir a Chroma: ids, embeddings, metadatas y documents (texto plano para mostrar).

    Returns:
      La colección Chroma ya poblada (para uso en query).
    """
    # Paso 1: cliente persistente (datos en CHROMA_DIR)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = [c.name for c in chroma_client.list_collections()]

    # Paso 2 y 3: preparar colección
    if collection_name in existing and force_recreate:
        chroma_client.delete_collection(collection_name)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Chunks para RAG"},
    )

    # Paso 4: chunks desde LangChain
    chunks = load_and_chunk_docs_lc()
    if not chunks:
        print("No se encontraron chunks (¿hay PDFs en data/docs?).")
        return collection

    # Paso 5: estructuras para Chroma (ids únicos por chunk)
    texts = [c["text"] for c in chunks]
    metadatas = [
        {"source": c["source"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]
    ids = [f"{c['source']}_{c['chunk_index']}" for c in chunks]

    # Paso 6: texto → vector (misma dimensión para todos)
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Paso 7: escribir en la base vectorial
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts,
    )
    print(f"Indexados {len(chunks)} chunks en la colección '{collection_name}'.")
    return collection
