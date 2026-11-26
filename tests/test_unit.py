"""Tests rapides : extraction réelle du PDF de démonstration et logique pure.
Aucun modèle ni base requis."""

from pathlib import Path

import pytest

from maintidoc.extraction import MultimodalPDFReader
from maintidoc.indexing import build_nodes
from maintidoc.rag import REFUSAL, SourceRegistry, finalize, history_messages, search_query
from maintidoc.retrieval import Passage, reciprocal_rank_fusion

PDF = Path(__file__).resolve().parent.parent / "data" / "attention_is_all_you_need.pdf"


@pytest.fixture(scope="module")
def documents(tmp_path_factory):
    if not PDF.exists():
        pytest.skip("PDF de démonstration absent")
    return MultimodalPDFReader(artifacts_dir=tmp_path_factory.mktemp("artifacts")).load_data(PDF)


def of_type(documents, kind):
    return [d for d in documents if d.metadata["type"] == kind]


def row(table, label, nth=0):
    lines = [line for line in table.metadata["table_markdown"].splitlines()
             if line.startswith(f"| {label} |")]
    return [v.strip() for v in lines[nth].split("|")[1:-1]]


def test_contenu_multimodal(documents):
    assert [d.metadata["page"] for d in of_type(documents, "table")] == [6, 8, 9]
    # La figure 2 est faite de deux images : elle doit rester une seule figure.
    assert [d.metadata["page"] for d in of_type(documents, "image")] == [3, 4]
    assert all(Path(d.metadata["image_path"]).exists() for d in of_type(documents, "image"))
    assert all(d.metadata["caption"] for d in of_type(documents, "image"))
    assert {d.metadata["page"] for d in of_type(documents, "text")} == set(range(1, 12))


def test_tableau_1_exposants(documents):
    table = of_type(documents, "table")[0]
    assert row(table, "Self-Attention") == ["Self-Attention", "O(n^2 · d)", "O(1)", "O(1)"]
    assert row(table, "Recurrent")[1] == "O(n · d^2)"


def test_tableau_2_puissances_et_cases_vides(documents):
    table = of_type(documents, "table")[1]
    assert row(table, "GNMT + RL [31]")[1:] == ["24.6", "39.92", "2.3 · 10^19", "1.4 · 10^20"]
    assert row(table, "ByteNet [15]")[1:] == ["23.75", "", "", ""]
    assert "Transformer (big) : BLEU EN-DE = 28.4 ; BLEU EN-FR = 41.0" in table.text


def test_tableau_3_alignement_et_groupes(documents):
    table = of_type(documents, "table")[2]
    assert row(table, "base")[1:] == ["6", "512", "2048", "8", "64", "64", "0.1", "0.1",
                                      "100K", "4.92", "25.8", "65"]
    first_a = row(table, "(A)")
    assert (first_a[4], first_a[11]) == ("1", "24.9")
    assert row(table, "(C)")[1] == "2"          # libellé de groupe propagé
    assert "h = 8 ; d_k = 64" in table.text


def test_noeuds_metadonnees_et_separation(documents):
    text_nodes, image_nodes = build_nodes(documents, lambda *_: "description", lambda _: None)
    assert len(image_nodes) == 2
    assert len({n.node_id for n in text_nodes}) == len(text_nodes)
    for node in text_nodes:
        assert {"file_id", "source", "page", "type", "ref"} <= set(node.metadata)
        assert "doc_id" not in node.metadata   # clé réservée par LlamaIndex
    assert sum(n.metadata["type"] == "table" for n in text_nodes) == 3   # jamais découpés


def passage(ref, page=1):
    return Passage(ref, "text", page, "doc.pdf", f"contenu {ref}", "", "", "", 1.0)


def test_registre_sans_doublon():
    registry = SourceRegistry()
    first = registry.format([passage("a")])
    second = registry.format([passage("a"), passage("b")])
    assert 'id="S1"' in first and "[S1] déjà fournie" in second and 'id="S2"' in second
    assert list(registry.sources) == ["S1", "S2"]


def test_controle_des_citations():
    registry = SourceRegistry()
    registry.format([passage("a"), passage("b")])
    assert finalize("Fait [S1, S2].", registry).cited == ["S1", "S2"]
    invented = finalize("Fait [S1][S9].", registry)
    assert invented.cited == ["S1"] and "S9" in invented.warnings[0]
    assert "sans aucune citation" in finalize("Fait.", registry).warnings[0]
    assert finalize(REFUSAL, registry).warnings == []
    assert "http" not in finalize("![x](http://exemple.test/a.png) [S1]", registry).text


def test_historique_borne_sans_anciennes_citations():
    messages = history_messages([(f"q{i}", f"r{i} [S1]") for i in range(10)])
    assert len(messages) == 6 and all("[S1]" not in m.content for m in messages)


def test_fusion_des_classements():
    a, b, c = passage("a"), passage("b"), passage("c")
    assert [p.ref for p in reciprocal_rank_fusion([[a, b, c], [b, c, a]], k=2)] == ["b", "a"]


def test_budget_de_contexte(monkeypatch):
    monkeypatch.setattr("maintidoc.config.MAX_CONTEXT_CHARS", 10)
    registry = SourceRegistry()
    output = registry.format([passage("a"), passage("b"), passage("c")])
    assert "Budget de contexte atteint" in output
    assert list(registry.sources) == ["S1"]   # ce qui n'est pas envoyé n'est pas citable


def test_question_de_suivi_recherchee_avec_la_precedente():
    assert search_query("Et en français ?", []) == "Et en français ?"
    assert search_query("Et en français ?", [("Score BLEU ?", "28.4")]) == "Score BLEU ?\nEt en français ?"
