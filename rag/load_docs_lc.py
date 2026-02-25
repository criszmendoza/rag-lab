"""
Carga de PDFs y splitting con LangChain.

- Loader: PyPDFLoader (langchain_community) — un Document por página.
- Splitter: RecursiveCharacterTextSplitter — divide por tamaño respetando separadores.

Salida: lista de dicts {"text", "source", "chunk_index"} para alimentar el índice.
"""
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR

log = logging.getLogger(__name__)


def load_and_chunk_docs_lc(docs_dir: Path | None = None) -> list[dict]:
    """
    Carga todos los PDFs con LangChain y los divide en chunks.

    Pasos:
      1. Resolver directorio de documentos y comprobar que existe.
      2. Listar todos los *.pdf y crear el splitter (tamaño y overlap desde config).
      3. Por cada PDF: cargar con PyPDFLoader → lista de Document (una por página).
      4. Pasar esa lista a split_documents() → chunks con metadata (source, etc.).
      5. Convertir cada chunk a dict con "text", "source", "chunk_index" para el índice.
      6. Devolver la lista de chunks de todos los PDFs.

    Returns:
      Lista de dicts con "text", "source", "chunk_index".
    """
    # Paso 1: directorio de entrada
    docs_dir = docs_dir or DOCS_DIR
    if not docs_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de documentos: {docs_dir}")

    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    if not pdf_paths:
        return []

    # Paso 2: splitter con separadores para no cortar en medio de palabra/frase
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for path in pdf_paths:
        # Paso 3: cargar PDF → N Document (uno por página)
        log.info("LangChain: cargando %s con PyPDFLoader.", path.name)
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        if not documents:
            continue

        # Paso 4: dividir en chunks; LangChain mantiene metadata (source, page)
        split_docs = splitter.split_documents(documents)

        # Paso 5: adaptar a nuestro formato para el índice
        for i, doc in enumerate(split_docs):
            text = doc.page_content.strip()
            if not text:
                continue
            source = doc.metadata.get("source", path.name)
            if isinstance(source, Path):
                source = source.name
            elif "/" in source or "\\" in source:
                source = Path(source).name
            all_chunks.append({
                "text": text,
                "source": source,
                "chunk_index": i,
            })

    # Paso 6
    log.info("LangChain: %s chunks generados desde %s PDFs.", len(all_chunks), len(pdf_paths))
    return all_chunks
