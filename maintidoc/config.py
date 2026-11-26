"""Configuration centrale. Chaque valeur peut être surchargée par variable d'environnement (.env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


# --- Modèles (tous à poids ouverts, exécutés localement) ---
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = _env("LLM_MODEL", "qwen3:8b")
VLM_MODEL = _env("VLM_MODEL", "qwen2.5vl:3b")
EMBED_MODEL = _env("EMBED_MODEL", "bge-m3")
EMBED_DIM = int(_env("EMBED_DIM", "1024"))
CLIP_MODEL = _env("CLIP_MODEL", "openai/clip-vit-base-patch32")
CLIP_DIM = int(_env("CLIP_DIM", "512"))
LLM_NUM_CTX = int(_env("LLM_NUM_CTX", "12288"))

# --- PostgreSQL / PGVector ---
PG = {
    "host": _env("PG_HOST", "127.0.0.1"),
    "port": _env("PG_PORT", "5433"),
    "database": _env("PG_DATABASE", "maintidoc"),
    "user": _env("PG_USER", "maintidoc"),
    "password": _env("PG_PASSWORD", ""),
}
TEXT_TABLE = "maintidoc_text"   # table réelle : data_maintidoc_text
CLIP_TABLE = "maintidoc_clip"   # table réelle : data_maintidoc_clip

# --- Indexation et recherche ---
CHUNK_SIZE = int(_env("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "60"))
TOP_K_TEXT = int(_env("TOP_K_TEXT", "5"))
TOP_K_TABLE = int(_env("TOP_K_TABLE", "2"))
TOP_K_IMAGE = int(_env("TOP_K_IMAGE", "2"))
MAX_TOOL_CALLS = int(_env("MAX_TOOL_CALLS", "5"))
MAX_SOURCE_CHARS = int(_env("MAX_SOURCE_CHARS", "3600"))
# Plafond cumulé des sources par question : au-delà de la fenêtre du modèle, Ollama tronque
# sans prévenir. 24 000 caractères représentent environ 6 000 à 8 000 tokens.
MAX_CONTEXT_CHARS = int(_env("MAX_CONTEXT_CHARS", "24000"))
MAX_HISTORY_TURNS = int(_env("MAX_HISTORY_TURNS", "3"))

ARTIFACTS_DIR = ROOT / "data" / "artifacts"
UPLOADS_DIR = ROOT / "data" / "uploads"
