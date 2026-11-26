"""Recherche dans PGVector, toujours limitée au document sélectionné."""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from . import config, store, vision
from .indexing import embed_model


@dataclass
class Passage:
    ref: str
    kind: str          # "text" | "table" | "image"
    page: int
    source: str
    text: str
    caption: str
    image_path: str
    page_image: str
    score: float
    table_markdown: str = ""   # grille affichée dans l'interface


def _query(vs, embedding: list[float], doc_id: str, k: int, **equals) -> list[Passage]:
    filters = [MetadataFilter(key="file_id", value=doc_id)]
    filters += [MetadataFilter(key=key, value=value) for key, value in equals.items()]
    result = vs.query(VectorStoreQuery(query_embedding=embedding, similarity_top_k=k,
                                       filters=MetadataFilters(filters=filters)))
    passages = []
    for node, score in zip(result.nodes or [], result.similarities or []):
        m = node.metadata
        passages.append(Passage(
            ref=m["ref"], kind=m["type"], page=int(m["page"]), source=m["source"],
            text=node.get_content(), caption=m.get("caption", ""),
            image_path=m.get("image_path", ""), page_image=m.get("page_image", ""),
            score=float(score), table_markdown=m.get("table_markdown", "")))
    return passages


def search_text(query: str, doc_id: str, k: int = config.TOP_K_TEXT) -> list[Passage]:
    return _query(store.text_store(), embed_model().get_query_embedding(query), doc_id, k,
                  type="text")


def search_tables(query: str, doc_id: str, k: int = config.TOP_K_TABLE) -> list[Passage]:
    return _query(store.text_store(), embed_model().get_query_embedding(query), doc_id, k,
                  type="table")


def reciprocal_rank_fusion(rankings: list[list[Passage]], k: int, constant: int = 60) -> list[Passage]:
    """Fusionne des classements dont les scores ne sont pas comparables."""
    scores: dict[str, float] = {}
    best: dict[str, Passage] = {}
    for ranking in rankings:
        for rank, passage in enumerate(ranking, start=1):
            scores[passage.ref] = scores.get(passage.ref, 0.0) + 1.0 / (constant + rank)
            best.setdefault(passage.ref, passage)
    return [best[ref] for ref in sorted(scores, key=scores.get, reverse=True)[:k]]


def search_images(query: str, doc_id: str, query_en: str = "",
                  k: int = config.TOP_K_IMAGE) -> list[Passage]:
    """Recherche hybride : descriptions (bge-m3) et image elle-même (CLIP, texte-image)."""
    by_description = _query(store.text_store(), embed_model().get_query_embedding(query),
                            doc_id, k * 2, type="image")
    if not by_description:
        return []
    by_clip = _query(store.clip_store(), vision.clip_text_embedding(query_en or query),
                     doc_id, k * 2)
    # Le classement par description passe en premier : il départage les égalités.
    return reciprocal_rank_fusion([by_description, by_clip], k)


def read_page(page: int, doc_id: str) -> list[Passage]:
    """Tous les éléments indexés d'une page donnée."""
    # Le vecteur n'a pas d'importance ici : le filtre de page sélectionne tout.
    found = _query(store.text_store(), embed_model().get_query_embedding(f"page {page}"),
                   doc_id, 20, page=page)
    return sorted(found, key=lambda p: p.ref)


# Recherches élargies, utilisées par l'agent. Le choix d'outil d'un petit LLM est irrégulier :
# chaque outil joint un contexte complémentaire pour qu'un mauvais choix ne suffise pas à
# écarter la bonne source.

def text_with_context(query: str, doc_id: str) -> list[Passage]:
    return search_text(query, doc_id, 4) + search_tables(query, doc_id, 1)


def tables_with_context(query: str, doc_id: str) -> list[Passage]:
    return search_tables(query, doc_id) + search_text(query, doc_id, 2)


def figures_with_context(query: str, doc_id: str, query_en: str = "") -> list[Passage]:
    """Ajoute le texte des pages des figures : les descriptions automatiques sont faillibles."""
    figures = search_images(query, doc_id, query_en=query_en)
    if not figures:
        return []
    embedding = embed_model().get_query_embedding(query)
    around = []
    for page in dict.fromkeys(f.page for f in figures):
        around += _query(store.text_store(), embedding, doc_id, 2, type="text", page=page)
    return figures + around


def retrieve_all(query: str, doc_id: str) -> list[Passage]:
    """Récupération fixe du pipeline RAG simple (sans agent)."""
    return search_text(query, doc_id) + search_tables(query, doc_id) + search_images(query, doc_id)
