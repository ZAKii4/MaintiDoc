"""Pipeline d'indexation : PDF -> LlamaIndex -> embeddings -> PGVector.

Usage : python -m maintidoc.indexing chemin/vers/document.pdf
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding

from . import config, store, vision
from .extraction import MultimodalPDFReader, file_doc_id


@lru_cache(maxsize=1)
def embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(model_name=config.EMBED_MODEL, base_url=config.OLLAMA_URL)


def _node(node_id: str, content: str, metadata: dict) -> TextNode:
    node = TextNode(id_=node_id, text=content, metadata=dict(metadata))
    # Les chemins et identifiants ne doivent pas influencer les embeddings.
    node.excluded_embed_metadata_keys = list(metadata)
    node.excluded_llm_metadata_keys = list(metadata)
    return node


def build_nodes(documents, describe: Callable[[str, str], str],
                progress: Callable[[str], None]) -> tuple[list[TextNode], list[TextNode]]:
    """Transforme les `Document` en nœuds textuels et en nœuds CLIP (sans vecteurs)."""
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    text_nodes: list[TextNode] = []
    image_nodes: list[TextNode] = []
    counters: dict[tuple, int] = {}

    for doc in documents:
        meta = doc.metadata
        key = (meta["page"], meta["type"])
        if meta["type"] == "text":
            chunks = splitter.split_text(doc.text)
        elif meta["type"] == "table":
            # Un tableau reste entier : le découper séparerait les valeurs des en-têtes.
            chunks = [f"Tableau (page {meta['page']}). {meta['caption']}\n\n{doc.text}"]
        else:
            progress(f"Description de la figure, page {meta['page']}…")
            description = describe(meta["image_path"], meta["caption"])
            chunks = [f"Figure (page {meta['page']}). Légende : {meta['caption'] or 'aucune'}\n\n"
                      f"Description générée automatiquement : {description}"]

        for chunk in chunks:
            counters[key] = counters.get(key, 0) + 1
            ref = f"{meta['file_id']}-p{meta['page']}-{meta['type']}-{counters[key]}"
            text_nodes.append(_node(ref, chunk, {**meta, "ref": ref}))
            if meta["type"] == "image":
                image_nodes.append(_node(f"{ref}-clip", chunk, {**meta, "ref": ref}))
    return text_nodes, image_nodes


def index_pdf(path: str | Path, source_name: str | None = None,
              progress: Callable[[str], None] = print) -> dict:
    """Indexe un PDF. Réindexer le même fichier remplace ses entrées."""
    path = Path(path)
    name = source_name or path.name
    doc_id = file_doc_id(path)

    progress("Analyse du PDF avec LlamaIndex…")
    documents = MultimodalPDFReader().load_data(path, source_name=name)
    if not any(d.metadata["type"] == "text" for d in documents):
        raise ValueError("Aucun texte extractible : ce PDF semble scanné (OCR non pris en charge).")

    text_nodes, image_nodes = build_nodes(documents, vision.describe_image, progress)

    progress(f"Embeddings {config.EMBED_MODEL} de {len(text_nodes)} éléments…")
    vectors = embed_model().get_text_embedding_batch(
        [n.get_content(metadata_mode=MetadataMode.NONE) for n in text_nodes])
    for node, vector in zip(text_nodes, vectors):
        node.embedding = vector

    progress(f"Embeddings CLIP de {len(image_nodes)} images…")
    for node in image_nodes:
        node.embedding = vision.clip_image_embedding(node.metadata["image_path"])

    # Tous les calculs sont terminés : on remplace alors l'ancienne version.
    progress("Écriture dans PGVector…")
    store.delete_document(doc_id)
    store.text_store().add(text_nodes)
    if image_nodes:
        store.clip_store().add(image_nodes)

    stats = {
        "pages": len({d.metadata["page"] for d in documents}),
        "passages": sum(n.metadata["type"] == "text" for n in text_nodes),
        "tableaux": sum(n.metadata["type"] == "table" for n in text_nodes),
        "figures": len(image_nodes),
    }
    store.register_document(doc_id, name, stats)
    progress(f"Terminé : {stats}")
    return {"doc_id": doc_id, "name": name, **stats}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    index_pdf(sys.argv[1])
