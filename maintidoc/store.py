"""Accès à PostgreSQL / PGVector.

Deux index vectoriels, car les dimensions et les espaces diffèrent :
* maintidoc_text (bge-m3, 1024) : passages, tableaux et descriptions d'images ;
* maintidoc_clip (CLIP, 512)    : les images elles-mêmes.
La table maintidoc_documents tient le registre des documents indexés.
"""

import json
from functools import lru_cache

from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from . import config


@lru_cache(maxsize=None)
def vector_store(table: str, dim: int) -> PGVectorStore:
    return PGVectorStore.from_params(
        **config.PG, table_name=table, embed_dim=dim,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64,
                     "hnsw_ef_search": 100, "hnsw_dist_method": "vector_cosine_ops"},
    )


def text_store() -> PGVectorStore:
    return vector_store(config.TEXT_TABLE, config.EMBED_DIM)


def clip_store() -> PGVectorStore:
    return vector_store(config.CLIP_TABLE, config.CLIP_DIM)


@lru_cache(maxsize=1)
def engine():
    url = URL.create("postgresql+psycopg2", username=config.PG["user"],
                     password=config.PG["password"] or None, host=config.PG["host"],
                     port=int(config.PG["port"]), database=config.PG["database"])
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS maintidoc_documents (
                doc_id     text PRIMARY KEY,
                name       text NOT NULL,
                stats      jsonb NOT NULL,
                indexed_at timestamptz NOT NULL DEFAULT now()
            )"""))
    return eng


def register_document(doc_id: str, name: str, stats: dict) -> None:
    with engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO maintidoc_documents (doc_id, name, stats)
            VALUES (:d, :n, CAST(:s AS jsonb))
            ON CONFLICT (doc_id) DO UPDATE
            SET name = :n, stats = CAST(:s AS jsonb), indexed_at = now()
        """), {"d": doc_id, "n": name, "s": json.dumps(stats)})


def list_documents() -> list[dict]:
    with engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT doc_id, name, stats FROM maintidoc_documents ORDER BY indexed_at DESC"))
        return [dict(r._mapping) for r in rows]


def delete_document(doc_id: str) -> None:
    """Supprime toutes les entrées d'un document (réindexation sans doublons)."""
    with engine().begin() as conn:
        for table in (config.TEXT_TABLE, config.CLIP_TABLE):
            name = f"data_{table}"
            if conn.execute(text("SELECT to_regclass(:t)"), {"t": f"public.{name}"}).scalar():
                conn.execute(text(f"DELETE FROM {name} WHERE metadata_->>'file_id' = :d"),
                             {"d": doc_id})
        conn.execute(text("DELETE FROM maintidoc_documents WHERE doc_id = :d"), {"d": doc_id})
