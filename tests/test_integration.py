"""Tests avec PGVector, bge-m3 et CLIP réels. Sautés si la base n'est pas indexée.
Le LLM n'est pas appelé : voir scripts/evaluate.py."""

import pytest

from maintidoc import retrieval, store


@pytest.fixture(scope="module")
def doc_id():
    try:
        docs = store.list_documents()
    except Exception as error:
        pytest.skip(f"PostgreSQL indisponible : {error}")
    found = next((d for d in docs if "attention" in d["name"].lower()), None)
    if found is None:
        pytest.skip("Document de démonstration non indexé")
    return found["doc_id"]


def test_recherche_texte_en_francais(doc_id):
    pages = {p.page for p in retrieval.search_text(
        "Sur quels GPU les modèles ont-ils été entraînés ?", doc_id)}
    assert 7 in pages


def test_recherche_tableaux(doc_id):
    results = retrieval.search_tables("BLEU English-German comparison of models", doc_id)
    assert 8 in {p.page for p in results} and all(p.kind == "table" for p in results)


def test_recherche_figures_hybride(doc_id):
    results = retrieval.search_images("architecture du Transformer", doc_id,
                                      query_en="Transformer encoder decoder architecture diagram")
    assert results and results[0].page == 3 and results[0].kind == "image"


def test_cloisonnement_entre_documents(doc_id):
    assert retrieval.search_text("attention", "document-inexistant") == []
    assert retrieval.search_images("schéma", "document-inexistant") == []


def test_lecture_de_page(doc_id):
    results = retrieval.read_page(4, doc_id)
    assert {p.page for p in results} == {4} and {"text", "image"} <= {p.kind for p in results}
