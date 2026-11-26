"""Analyse d'un PDF en objets LlamaIndex : texte, tableaux et figures.

LlamaIndex n'extrait pas à lui seul les tableaux et les figures d'un PDF en local (LlamaParse
est un service distant). On implémente donc un `BaseReader` LlamaIndex qui s'appuie sur
PyMuPDF et renvoie des `Document` avec leurs métadonnées (page, type, légende, image).

Deux traitements spécifiques :
* tableaux : PyMuPDF localise bien la zone mais fusionne les lignes et aplatit les exposants
  (10^19 devient 1019). La grille est reconstruite à partir de la position des caractères ;
* figures : les images partageant une même légende « Figure N » sont regroupées et la zone
  est rendue depuis la page, pour conserver aussi les libellés vectoriels.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

from . import config

CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.|Table|Tableau)\s*(\d+)\s*[:.\-–]", re.IGNORECASE)
MIN_IMAGE_SIDE = 40      # points : ignore puces, filets et petits logos
CELL_GAP = 7.0           # points : écart horizontal séparant deux cellules


def file_doc_id(path: str | Path) -> str:
    """Identifiant stable d'un document : empreinte de son contenu."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- tableaux

@dataclass
class _Cell:
    x0: float
    x1: float
    text: str


def _chars_in(page: pymupdf.Page, clip: pymupdf.Rect) -> list[dict]:
    out = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    x0, y0, x1, y1 = ch["bbox"]
                    if clip.contains(pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)):
                        out.append({"c": ch["c"], "x0": x0, "x1": x1,
                                    "base": ch["origin"][1], "size": span["size"]})
    return out


def _rows_from_chars(chars: list[dict]) -> list[tuple[float, list[_Cell]]]:
    """Regroupe les caractères en lignes puis en cellules, en marquant exposants et indices."""
    if not chars:
        return []
    sizes = sorted(c["size"] for c in chars if c["c"].strip())
    main = sizes[len(sizes) // 2] if sizes else 10.0
    full = [c for c in chars if c["size"] >= 0.85 * main and c["c"].strip()]

    baselines: list[float] = []
    for c in sorted(full, key=lambda c: c["base"]):
        if not baselines or c["base"] - baselines[-1] > 3.0:
            baselines.append(c["base"])
    if not baselines:
        return []

    rows: dict[float, list[dict]] = {b: [] for b in baselines}
    for c in chars:
        if not c["c"].strip():
            continue
        base = min(baselines, key=lambda b: abs(b - c["base"]))
        script = ""
        if c["size"] < 0.85 * main:
            # Un exposant appartient à la ligne du caractère qu'il suit.
            host = [f for f in full if abs(f["base"] - c["base"]) < 7 and -1 <= c["x0"] - f["x1"] < 4]
            if host:
                near = min(host, key=lambda f: c["x0"] - f["x1"])
                base = min(baselines, key=lambda b: abs(b - near["base"]))
            delta = c["base"] - base
            script = "^" if delta < -1.5 else "_" if delta > 0.8 else ""
        rows[base].append({**c, "script": script})

    result = []
    for base in baselines:
        cells: list[_Cell] = []
        prev = None
        for c in sorted(rows[base], key=lambda c: c["x0"]):
            gap = 0.0 if prev is None else c["x0"] - prev["x1"]
            marker = c["script"] if c["script"] and (prev is None or prev["script"] != c["script"]) else ""
            if prev is None or gap > CELL_GAP:
                cells.append(_Cell(c["x0"], c["x1"], marker + c["c"]))
            else:
                space = " " if gap > 1.6 and not c["script"] else ""
                cells[-1].text += space + marker + c["c"]
                cells[-1].x1 = max(cells[-1].x1, c["x1"])
            prev = c
        if cells:
            result.append((base, cells))
    return result


def _columns(rows: list[tuple[float, list[_Cell]]]) -> list[list[float]]:
    """Déduit les colonnes en traitant d'abord les cellules étroites ;
    une cellule couvrant plusieurs colonnes (fusionnée) n'en crée pas."""
    cols: list[list[float]] = []
    for cell in sorted((c for _, cells in rows for c in cells), key=lambda c: c.x1 - c.x0):
        hits = [col for col in cols if cell.x0 < col[1] and cell.x1 > col[0]]
        if not hits:
            cols.append([cell.x0, cell.x1])
        elif len(hits) == 1:
            hits[0][0] = min(hits[0][0], cell.x0)
            hits[0][1] = max(hits[0][1], cell.x1)
    return sorted(cols)


def _horizontal_rules(page: pymupdf.Page, clip: pymupdf.Rect) -> list[float]:
    ys = sorted(d["rect"].y0 for d in page.get_drawings()
                if d["rect"].height < 1.5 and d["rect"].width > 0.6 * (clip.width - 24)
                and clip.y0 - 4 <= d["rect"].y0 <= clip.y1 + 4)
    return ys


def table_to_markdown(page: pymupdf.Page, bbox: pymupdf.Rect) -> str:
    clip = pymupdf.Rect(bbox) + (-12, -3, 12, 3)
    rows = _rows_from_chars(_chars_in(page, clip))
    cols = _columns(rows)
    if len(cols) < 2:
        return ""

    rules = _horizontal_rules(page, clip)
    # En-tête : tout ce qui précède le deuxième filet ; sinon la première ligne.
    header_end = rules[1] if len(rules) >= 3 else rows[0][0]
    header: list[list[str]] = [[] for _ in cols]
    grid: list[tuple[float, list[str]]] = []
    for base, cells in rows:
        in_header = base <= header_end + 1
        line = [""] * len(cols)
        for cell in cells:
            hits = [i for i, col in enumerate(cols) if cell.x0 < col[1] and cell.x1 > col[0]]
            hits = hits or [0]
            if in_header:
                # Un titre couvrant plusieurs colonnes est répété sur chacune d'elles.
                for i in hits:
                    if cell.text not in header[i]:
                        header[i].append(cell.text)
                continue
            text = cell.text
            if len(hits) > 1:
                text += f" [fusionnée sur {len(hits)} colonnes]"
            line[hits[0]] = f"{line[hits[0]]} {text}".strip()
        if not in_header:
            grid.append((base, line))

    # Libellé de groupe centré verticalement, ex. « (A) » : s'il est le seul de son groupe
    # (délimité par deux filets), on le recopie sur chaque ligne du groupe.
    for top, bottom in zip(rules, rules[1:]):
        group = [r for r in grid if top < r[0] <= bottom + 1]
        labels = [r for r in group if r[1][0]]
        if len(group) > 1 and len(labels) == 1:
            label = labels[0][1][0]
            for r in group:
                r[1][0] = label
            if not any(labels[0][1][1:]):
                grid.remove(labels[0])

    def fmt(line: list[str]) -> str:
        return "| " + " | ".join(v.replace("|", "\\|") for v in line) + " |"

    lines = [fmt([" ".join(h) for h in header]), "|" + "---|" * len(cols)]
    lines += [fmt(line) for _, line in grid]
    return "\n".join(lines)


def markdown_to_records(markdown: str) -> str:
    """Réécrit un tableau en « libellé : en-tête = valeur ». Un petit LLM se trompe de colonne
    dans une grille creuse ; ici chaque valeur porte le nom de la sienne."""
    def split(line: str) -> list[str]:
        return [v.strip().replace("\\|", "|") for v in re.split(r"(?<!\\)\|", line.strip())[1:-1]]

    lines = markdown.splitlines()
    headers = split(lines[0])
    records = []
    for line in lines[2:]:
        values = split(line)
        pairs = [f"{headers[i] or f'colonne {i + 1}'} = {v}"
                 for i, v in enumerate(values) if i > 0 and v]
        label = values[0] or "(sans libellé)"
        records.append(f"- {headers[0] or 'ligne'} {label} : " + " ; ".join(pairs))
    return "\n".join(records)


# --------------------------------------------------------------------------- lecteur

def _captions(page: pymupdf.Page) -> list[dict]:
    found = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        m = CAPTION_RE.match(text)
        if m:
            kind = "table" if m.group(1).lower().startswith("tab") else "figure"
            found.append({"kind": kind, "num": int(m.group(2)),
                          "rect": pymupdf.Rect(x0, y0, x1, y1),
                          "text": " ".join(text.split())})
    return found


def _nearest_caption(captions: list[dict], kind: str, bbox: pymupdf.Rect) -> dict | None:
    def distance(c):
        return max(c["rect"].y0 - bbox.y1, bbox.y0 - c["rect"].y1, 0)
    near = [c for c in captions if c["kind"] == kind and distance(c) < 90]
    return min(near, key=distance) if near else None


def _figure_regions(page: pymupdf.Page, captions: list[dict]) -> list[tuple[pymupdf.Rect, dict | None]]:
    """Une région par figure : images regroupées par la première légende située en dessous."""
    rects = []
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            r = r & page.rect
            if r.width >= MIN_IMAGE_SIDE and r.height >= MIN_IMAGE_SIDE:
                rects.append(r)

    groups: dict[int, list] = {}
    captions_fig = sorted((c for c in captions if c["kind"] == "figure"), key=lambda c: c["rect"].y0)
    orphans = []
    for r in rects:
        below = next((i for i, c in enumerate(captions_fig) if c["rect"].y0 >= r.y1 - 5), None)
        if below is None:
            orphans.append(r)
        else:
            groups.setdefault(below, []).append(r)

    regions: list[tuple[pymupdf.Rect, dict | None]] = []
    for i, members in groups.items():
        area = pymupdf.Rect(members[0])
        for r in members[1:]:
            area |= r
        # Inclure les titres vectoriels courts placés juste au-dessus des images.
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            if area.y0 - 30 <= y0 < area.y0 and len(text.strip()) < 80:
                area |= pymupdf.Rect(x0, y0, x1, y1)
        regions.append(((area + (-8, -6, 8, 6)) & page.rect, captions_fig[i]))
    regions += [((r + (-4, -4, 4, 4)) & page.rect, None) for r in orphans]
    return sorted(regions, key=lambda x: x[0].y0)


def _page_text(page: pymupdf.Page, excluded: list[pymupdf.Rect]) -> str:
    """Texte courant de la page, hors zones de tableaux et de figures."""
    parts = []
    for x0, y0, x1, y1, text, _, block_type in page.get_text("blocks"):
        if block_type != 0:
            continue
        rect = pymupdf.Rect(x0, y0, x1, y1)
        centre = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any(z.contains(centre) for z in excluded) and not CAPTION_RE.match(text):
            continue
        if rect.y0 > page.rect.height * 0.9 and text.strip().isdigit():
            continue  # numéro de page
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        parts.append(" ".join(text.split()))
    return "\n\n".join(p for p in parts if p)


class MultimodalPDFReader(BaseReader):
    """Renvoie des `Document` LlamaIndex de type « text », « table » ou « image »."""

    def __init__(self, artifacts_dir: Path = config.ARTIFACTS_DIR, dpi: int = 200):
        self.artifacts_dir = Path(artifacts_dir)
        self.dpi = dpi

    def load_data(self, file: str | Path, source_name: str | None = None) -> list[Document]:
        path = Path(file)
        doc_id = file_doc_id(path)
        out_dir = self.artifacts_dir / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        name = source_name or path.name
        documents: list[Document] = []

        with pymupdf.open(path) as pdf:
            for number, page in enumerate(pdf, start=1):
                # « doc_id » est réservé par LlamaIndex (écrasé à l'écriture) : on utilise « file_id ».
                base = {"file_id": doc_id, "source": name, "page": number}
                page_png = out_dir / f"page_{number}.png"
                page.get_pixmap(dpi=110).save(page_png)
                base["page_image"] = str(page_png)

                captions = _captions(page)
                excluded: list[pymupdf.Rect] = []

                for i, table in enumerate(page.find_tables().tables, start=1):
                    bbox = pymupdf.Rect(table.bbox)
                    markdown = table_to_markdown(page, bbox)
                    if not markdown:
                        continue
                    excluded.append(bbox + (-12, -3, 12, 3))
                    caption = _nearest_caption(captions, "table", bbox)
                    crop = out_dir / f"table_p{number}_{i}.png"
                    page.get_pixmap(clip=(bbox + (-12, -6, 12, 6)) & page.rect, dpi=self.dpi).save(crop)
                    documents.append(Document(text=markdown_to_records(markdown), metadata={
                        **base, "type": "table", "table_markdown": markdown,
                        "image_path": str(crop),
                        "caption": caption["text"] if caption else ""}))

                for i, (region, caption) in enumerate(_figure_regions(page, captions), start=1):
                    excluded.append(region)
                    crop = out_dir / f"figure_p{number}_{i}.png"
                    page.get_pixmap(clip=region, dpi=self.dpi).save(crop)
                    documents.append(Document(text="", metadata={
                        **base, "type": "image", "image_path": str(crop),
                        "caption": caption["text"] if caption else ""}))

                text = _page_text(page, excluded)
                if text:
                    documents.append(Document(text=text, metadata={
                        **base, "type": "text", "image_path": "", "caption": ""}))
        return documents
