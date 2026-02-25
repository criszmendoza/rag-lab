"""
Visualización del espacio de embeddings del RAG.

- Obtiene todos los vectores y metadatos desde Chroma.
- Reduce a 2D y 3D con sklearn (PCA o t-SNE solo 2D).
- Una sola página con vista 2D (izq) y 3D (der), mismo color por documento, estilo claro.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import BASE_DIR
from query import get_collection

log = logging.getLogger(__name__)

# Ruta por defecto para guardar el HTML (una página con 2D + 3D)
VISUALIZE_OUTPUT = BASE_DIR / "data" / "embeddings_2d_3d.html"

# Paleta de colores por documento (estilo claro, distinguibles)
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def get_all_embeddings(collection_name: str = "rag_docs") -> tuple[list, list[dict], list[str]]:
    """
    Obtiene de Chroma todos los embeddings, metadatos y documentos.

    Pasos:
      1. Abrir la colección con get_collection().
      2. Llamar collection.get(include=["embeddings", "metadatas", "documents"]) sin ids → todo.
      3. Devolver listas de embeddings, metadatas y textos (snippet para hover).

    Returns:
      (embeddings, metadatas, documents) — listas alineadas por índice.
    """
    log.info("Cargando vectores y metadatos desde Chroma.")
    collection = get_collection(collection_name)
    data = collection.get(include=["embeddings", "metadatas", "documents"])

    embeddings = data["embeddings"]
    metadatas = data.get("metadatas") or []
    documents = data.get("documents") or []

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("La colección está vacía. Ejecuta antes: python main.py index")

    # Snippet para hover (máximo N caracteres); doc puede ser str o array en algunas versiones de Chroma
    def _snippet(doc) -> str:
        s = str(doc) if doc is not None else ""
        return s[:120] + ("..." if len(s) > 120 else "")

    n = len(embeddings)
    # Alinear longitud: metadatas y documents pueden venir con menos elementos
    metadatas = list(metadatas) if metadatas else [{}] * n
    if len(metadatas) < n:
        metadatas = metadatas + [{}] * (n - len(metadatas))
    docs_snippet = [_snippet(documents[i]) if i < len(documents) else "" for i in range(n)]
    return embeddings, metadatas, docs_snippet


def reduce_to_2d(embeddings: list, method: str = "pca", random_state: int = 42) -> np.ndarray:
    """
    Reduce los vectores de alta dimensión a 2D para poder graficar.

    Pasos:
      1. Convertir a array numpy.
      2. Si method=="pca": PCA(n_components=2); si method=="tsne": TSNE(n_components=2, perplexity adecuado).
      3. Ajustar y transformar; devolver array de forma (n_samples, 2).
    """
    X = np.array(embeddings, dtype=np.float32)
    log.info("Reduciendo %s vectores a 2D con %s.", X.shape[0], method.upper())

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(X)
        log.info("PCA: varianza explicada por componente: %s", reducer.explained_variance_ratio_.round(4).tolist())
    elif method == "tsne":
        perplexity = min(30, max(5, X.shape[0] // 4))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        coords = reducer.fit_transform(X)
    else:
        raise ValueError("method debe ser 'pca' o 'tsne'")

    return coords


def reduce_to_3d(embeddings: list, random_state: int = 42) -> np.ndarray:
    """
    Reduce los vectores a 3D con PCA (estable y rápido para 3D).

    Pasos:
      1. Convertir a array numpy.
      2. PCA(n_components=3); ajustar y transformar.
      3. Devolver array (n_samples, 3).
    """
    X = np.array(embeddings, dtype=np.float32)
    log.info("Reduciendo %s vectores a 3D con PCA.", X.shape[0])
    reducer = PCA(n_components=3, random_state=random_state)
    coords = reducer.fit_transform(X)
    log.info("PCA 3D: varianza explicada %s", reducer.explained_variance_ratio_.round(4).tolist())
    return coords


def _build_figure_2d_3d(
    coords_2d: np.ndarray,
    coords_3d: np.ndarray,
    metadatas: list[dict],
    documents_snippet: list[str],
    method_label: str,
) -> go.Figure:
    """
    Construye una figura con dos subplots: scatter 2D (izq) y scatter 3D (der).
    Un color por documento, mismo en ambas vistas. Hover con documento, chunk y texto.
    """
    sources = [str(m.get("source", "?")) for m in metadatas]
    chunk_indices = [int(m.get("chunk_index", i)) for i, m in enumerate(metadatas)]
    doc_uniq = list(dict.fromkeys(sources))
    color_map = {d: COLORS[i % len(COLORS)] for i, d in enumerate(doc_uniq)}

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Vista 2D — cada punto es un fragmento de texto", "Vista 3D — misma agrupación en 3 ejes"),
        specs=[[{"type": "scatter"}, {"type": "scatter3d"}]],
        horizontal_spacing=0.08,
    )

    for doc in doc_uniq:
        mask = [s == doc for s in sources]
        x2 = coords_2d[mask, 0].tolist()
        y2 = coords_2d[mask, 1].tolist()
        x3 = coords_3d[mask, 0].tolist()
        y3 = coords_3d[mask, 1].tolist()
        z3 = coords_3d[mask, 2].tolist()
        texts = [documents_snippet[i] for i, m in enumerate(mask) if m]
        chunks = [chunk_indices[i] for i, m in enumerate(mask) if m]
        hover = [f"<b>{doc}</b><br>Chunk {c}<br>{t[:100]}..." if len(t) > 100 else f"<b>{doc}</b><br>Chunk {c}<br>{t}" for c, t in zip(chunks, texts)]

        fig.add_trace(
            go.Scatter(
                x=x2, y=y2, name=doc, mode="markers",
                marker=dict(size=8, color=color_map[doc], line=dict(width=0.5, color="white")),
                text=hover, hovertemplate="%{text}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=x3, y=y3, z=z3, name=doc, mode="markers",
                marker=dict(size=5, color=color_map[doc], line=dict(width=0.5, color="white")),
                text=hover, hovertemplate="%{text}<extra></extra>",
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Componente 1", row=1, col=1, gridcolor="lightgray")
    fig.update_yaxes(title_text="Componente 2", row=1, col=1, gridcolor="lightgray")
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"Embeddings RAG — 2D y 3D ({method_label})", font=dict(size=20)),
        height=550,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="center", x=0.5, title_text="Documento"),
        font=dict(size=12),
        margin=dict(t=100, b=60, l=60, r=40),
        scene2=dict(
            xaxis_title="Comp. 1",
            yaxis_title="Comp. 2",
            zaxis_title="Comp. 3",
        ),
    )
    return fig


def run_visualize(
    collection_name: str = "rag_docs",
    method: str = "pca",
    output_path: Path | None = None,
    show: bool = True,
) -> go.Figure:
    """
    Pipeline completo: cargar desde Chroma → reducir a 2D y 3D → una página con ambas vistas.

    Pasos:
      1. get_all_embeddings() → embeddings, metadatas, documents_snippet.
      2. reduce_to_2d(embeddings, method) → coords_2d (vista izquierda).
      3. reduce_to_3d(embeddings) → coords_3d (vista derecha, siempre PCA).
      4. _build_figure_2d_3d() → figura con dos subplots y estilo claro.
      5. Guardar HTML y, si show, abrir en navegador.
    """
    output_path = output_path or VISUALIZE_OUTPUT
    embeddings, metadatas, docs_snippet = get_all_embeddings(collection_name)
    coords_2d = reduce_to_2d(embeddings, method=method)
    coords_3d = reduce_to_3d(embeddings)
    method_label = "PCA" if method == "pca" else "t-SNE (2D) / PCA (3D)"
    fig = _build_figure_2d_3d(
        coords_2d,
        coords_3d,
        metadatas,
        docs_snippet,
        method_label=method_label,
    )
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        log.info("Gráfico guardado en: %s", output_path)
    if show:
        fig.show()
    return fig
