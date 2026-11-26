"""Traitement des images : embeddings CLIP et descriptions par un modèle vision-langage."""

from functools import lru_cache

from ollama import Client

from . import config

DESCRIPTION_PROMPT = """Tu analyses une figure issue d'un document technique.
Légende d'origine : {caption}
Décris en français, en 120 à 200 mots, uniquement ce qui est visible : type de schéma,
éléments et libellés (recopiés tels quels, dans leur langue d'origine), et liaisons entre
les éléments en suivant le sens des flèches. N'invente rien. Si un détail est illisible,
dis-le."""


def describe_image(image_path: str, caption: str) -> str:
    """Description textuelle générée localement. Elle peut contenir des erreurs."""
    response = Client(host=config.OLLAMA_URL).chat(
        model=config.VLM_MODEL,
        messages=[{"role": "user",
                   "content": DESCRIPTION_PROMPT.format(caption=caption or "aucune"),
                   "images": [image_path]}],
        options={"temperature": 0.1, "num_predict": 600},
    )
    return response["message"]["content"].strip()


@lru_cache(maxsize=1)
def _clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(device).eval()
    return model, CLIPProcessor.from_pretrained(config.CLIP_MODEL), device


def _normalise(output) -> list[float]:
    # transformers 5 renvoie un objet de sortie, transformers 4 un tenseur.
    vector = getattr(output, "pooler_output", output)[0]
    return (vector / vector.norm()).float().cpu().tolist()


def clip_image_embedding(image_path: str) -> list[float]:
    import torch
    from PIL import Image

    model, processor, device = _clip()
    inputs = processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        return _normalise(model.get_image_features(**inputs))


def clip_text_embedding(text: str) -> list[float]:
    """CLIP (OpenAI) est entraîné en anglais et limité à 77 tokens."""
    import torch

    model, processor, device = _clip()
    inputs = processor(text=[text], return_tensors="pt", truncation=True, max_length=77).to(device)
    with torch.no_grad():
        return _normalise(model.get_text_features(**inputs))
