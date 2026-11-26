"""Interface Streamlit de MaintiDoc.   Lancement : streamlit run app.py"""

import tempfile
from pathlib import Path

import streamlit as st

from maintidoc import config, store
from maintidoc.agent import answer_with_agent
from maintidoc.indexing import index_pdf
from maintidoc.rag import Answer, answer_with_rag

st.set_page_config(page_title="MaintiDoc", page_icon="🔧", layout="wide")

LABELS = {"text": "Texte", "table": "Tableau", "image": "Figure"}


def show_sources(answer: Answer, key: str) -> None:
    for warning in answer.warnings:
        st.warning(warning)

    cited = [(s, answer.sources[s]) for s in answer.cited]
    # Les figures et tableaux cités sont montrés directement sous la réponse.
    for sid, source in cited:
        if source.kind in ("image", "table") and Path(source.image_path).exists():
            st.image(source.image_path, width=650,
                     caption=f"[{sid}] p. {source.page} — {source.caption}")

    others = [(s, p) for s, p in answer.sources.items() if s not in answer.cited]
    for title, items in (("Sources citées", cited), ("Consultées mais non citées", others)):
        if not items:
            continue
        with st.expander(f"{title} ({len(items)})"):
            for sid, source in items:
                st.markdown(f"**[{sid}] {LABELS[source.kind]} — page {source.page}**")
                st.caption(source.source)
                if source.kind == "table" and source.table_markdown:
                    st.caption(source.caption)
                    st.markdown(source.table_markdown)
                else:
                    st.text(source.text)
                if Path(source.page_image).exists() and st.toggle(
                        f"Vérifier sur la page {source.page}", key=f"{key}-{title}-{sid}"):
                    st.image(source.page_image, width=700)

    if answer.trace:
        with st.expander("Démarche de l'orchestrateur"):
            for number, step in enumerate(answer.trace, start=1):
                st.markdown(f"{number}. `{step['outil']}`")
                st.json(step["arguments"])


def sidebar() -> tuple[dict | None, bool]:
    with st.sidebar:
        st.title("🔧 MaintiDoc")
        st.caption("Assistant documentaire de maintenance")

        uploaded = st.file_uploader("Charger un PDF", type="pdf")
        if uploaded and st.button("Indexer le document", type="primary", width="stretch"):
            with st.status("Indexation en cours…", expanded=True) as status:
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "document.pdf"   # nom fourni jamais utilisé comme chemin
                        path.write_bytes(uploaded.getvalue())
                        result = index_pdf(path, source_name=Path(uploaded.name).name,
                                           progress=status.write)
                    st.session_state.selected = result["doc_id"]
                    status.update(label="Document indexé", state="complete")
                except Exception as error:   # affiché à l'utilisateur
                    status.update(label="Échec de l'indexation", state="error")
                    st.error(str(error))

        documents = {d["doc_id"]: d for d in store.list_documents()}
        if not documents:
            return None, True
        if st.session_state.get("selected") not in documents:
            st.session_state.selected = next(iter(documents))
        doc_id = st.selectbox("Document interrogé", list(documents), key="selected",
                              format_func=lambda d: documents[d]["name"])
        stats = documents[doc_id]["stats"]
        st.caption(" · ".join(f"{v} {k}" for k, v in stats.items()))

        agentic = st.radio("Mode", ["Agentique (LangChain)", "RAG simple"]).startswith("Agent")
        if st.button("Nouvelle conversation", width="stretch"):
            st.session_state.chats[doc_id] = []
        st.divider()
        st.caption(f"LLM : {config.LLM_MODEL}\n\nVision : {config.VLM_MODEL}\n\n"
                   f"Embeddings : {config.EMBED_MODEL} · CLIP")
        st.caption("MaintiDoc ne pilote aucun équipement et ne remplace pas un technicien "
                   "habilité. Vérifiez les sources.")
        return documents[doc_id], agentic


def main() -> None:
    st.session_state.setdefault("chats", {})
    try:
        document, agentic = sidebar()
    except Exception as error:
        st.error(f"Base PGVector inaccessible. Lancez `scripts/db.sh start`.\n\n{error}")
        return

    st.header("Posez votre question")
    if document is None:
        st.info("Chargez puis indexez un PDF pour commencer.")
        return

    # Une conversation par document : le contexte ne passe pas de l'un à l'autre.
    chat = st.session_state.chats.setdefault(document["doc_id"], [])
    for number, (question, answer) in enumerate(chat):
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            st.markdown(answer.text or "_Aucune réponse._")
            show_sources(answer, f"{document['doc_id']}-{number}")

    question = st.chat_input("Ex. : Que montre le schéma de la page 3 ?")
    if not question:
        return
    st.chat_message("user").write(question)
    with st.chat_message("assistant"), st.spinner("Recherche et génération en local…"):
        try:
            history = [(q, a.text) for q, a in chat]
            run = answer_with_agent if agentic else answer_with_rag
            answer = run(question, document["doc_id"], history)
        except Exception as error:
            st.error(f"Erreur : {error}\n\nVérifiez qu'Ollama et PostgreSQL sont démarrés.")
            return
    chat.append((question, answer))
    st.rerun()


main()
