# MaintiDoc — Partie I : RAG multimodal agentique

Assistant de questions-réponses sur un PDF contenant du texte, des tableaux et des images,
avec réponses sourcées par page. Tous les modèles s'exécutent en local.
Il ne pilote aucun équipement et ne remplace pas un technicien habilité.

## Installation (macOS)

Prérequis : [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/),
PostgreSQL 17 et pgvector (`brew install postgresql@17 pgvector`).

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements.txt
ollama pull qwen3:8b && ollama pull qwen2.5vl:3b && ollama pull bge-m3
```

CLIP (`openai/clip-vit-base-patch32`) est téléchargé depuis Hugging Face à la première indexation.

## Exécution

```bash
scripts/db.sh start                      # PostgreSQL du projet, port 5433
source .venv/bin/activate
python -m maintidoc.indexing data/attention_is_all_you_need.pdf   # ou depuis l'interface
streamlit run app.py                     # http://127.0.0.1:8501
scripts/db.sh stop                       # à la fin
```

Sans Homebrew : `docker compose up -d` (voir `docker-compose.yml`, non testé).

Le PDF de démonstration n'est pas à redistribuer sans vérifier ses droits ; placez votre copie
dans `data/attention_is_all_you_need.pdf`.

## Tests

```bash
pytest                                   # unitaires + intégration PGVector (base indexée)
python scripts/evaluate.py les-deux      # questions de référence avec le LLM (≈ 15 à 20 min)
python scripts/inspect_extraction.py data/attention_is_all_you_need.pdf
```

## Organisation

| Fichier | Rôle |
|---|---|
| `maintidoc/extraction.py` | Lecteur LlamaIndex : texte, tableaux, figures |
| `maintidoc/vision.py` | Embeddings CLIP et descriptions d'images |
| `maintidoc/indexing.py` | Découpage, embeddings, écriture PGVector |
| `maintidoc/store.py` | Index PGVector et registre des documents |
| `maintidoc/retrieval.py` | Recherche texte, tableaux, figures (hybride), pages |
| `maintidoc/rag.py` | Contexte sourcé, génération, contrôle des citations |
| `maintidoc/agent.py` | Agent LangChain : outils, contexte, repli |
| `app.py` | Interface Streamlit |

La note technique est dans [`docs/note_technique.md`](docs/note_technique.md), les réponses
observées dans [`docs/resultats_evaluation.md`](docs/resultats_evaluation.md).
