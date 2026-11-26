"""Affiche ce que le lecteur extrait d'un PDF (contrôle visuel, sans modèle ni base)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maintidoc.extraction import MultimodalPDFReader  # noqa: E402

for d in MultimodalPDFReader().load_data(sys.argv[1]):
    m = d.metadata
    if m["type"] == "table":
        print(f"\n== TABLEAU p.{m['page']} — {m['caption'][:80]}\n{m['table_markdown']}")
        print(f"\n-- Forme indexée --\n{d.text}")
    elif m["type"] == "image":
        print(f"\n== FIGURE p.{m['page']} — {m['caption'][:80]}\n{m['image_path']}")
    else:
        print(f"\n== TEXTE p.{m['page']} : {len(d.text)} caractères")
