"""Joue des questions de référence avec les vrais modèles et enregistre les résultats observés.

Usage : python scripts/evaluate.py [agent|rag|les-deux]
Le « contrôle » vérifie la présence des valeurs attendues et la page citée. Il ne remplace
pas une relecture humaine des réponses (docs/resultats_evaluation.md).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maintidoc import store  # noqa: E402
from maintidoc.agent import answer_with_agent  # noqa: E402
from maintidoc.rag import REFUSAL, answer_with_rag  # noqa: E402

# (question, groupes de variantes attendues, pages acceptables)
CASES = [
    ("Combien de têtes d'attention le modèle de base utilise-t-il, et quelle est la dimension d_k ?",
     [["8", "huit"], ["64"]], {4, 5, 9}),
    ("Quel score BLEU le Transformer (big) obtient-il en anglais-allemand et en anglais-français ?",
     [["28.4", "28,4"], ["41.0", "41,0"]], {8}),
    ("Dans le tableau des variantes, quel BLEU obtient-on avec une seule tête d'attention ?",
     [["24.9", "24,9"]], {9}),
    ("Quelle est la complexité par couche de la self-attention comparée à une couche récurrente ?",
     [["n^2", "n²", "n2"], ["d^2", "d²", "d2"]], {6}),
    ("Montre-moi le schéma de l'architecture du Transformer et décris ses deux parties principales.",
     [["encod", "Encod"], ["décod", "Décod", "decod", "Decod"]], {2, 3}),
    ("Sur quel matériel et pendant combien de temps les modèles de base ont-ils été entraînés ?",
     [["P100"], ["12"]], {7}),
    ("Quelle est la procédure de remplacement du roulement d'une pompe centrifuge ?",
     [], set()),   # hors document : un refus est attendu
]

# Questions de contrôle : ajoutées après le réglage de l'agent, elles n'ont pas servi à l'ajuster.
CONTROL_CASES = [
    ("Quel optimiseur est utilisé et combien d'étapes de warmup sont prévues ?",
     [["Adam"], ["4000", "4 000"]], {7}),
    ("Quelle valeur de label smoothing est employée pendant l'entraînement ?",
     [["0.1", "0,1"]], {8, 9}),
    ("Combien de paramètres, en millions, compte le modèle « big » ?",
     [["213"]], {9}),
    ("Quelle est la dimension de la couche interne du réseau feed-forward du modèle de base ?",
     [["2048", "2 048"]], {5, 9}),
    ("Quel couple de serrage faut-il appliquer aux boulons de fixation du moteur ?",
     [], set()),   # hors document
]


def check(answer, expected, pages) -> bool:
    if not expected:
        return REFUSAL.rstrip(".") in answer.text
    values = all(any(v in answer.text for v in group) for group in expected)
    cited_pages = {answer.sources[s].page for s in answer.cited}
    return values and bool(cited_pages & pages)


def main(mode: str) -> None:
    doc = next((d for d in store.list_documents() if "attention" in d["name"].lower()), None)
    if doc is None:
        sys.exit("Indexez d'abord data/attention_is_all_you_need.pdf.")
    modes = ["agent", "rag"] if mode == "les-deux" else [mode]
    report = ["# Résultats observés", "",
              "Produits par `python scripts/evaluate.py`. Ce sont des mesures sur "
              f"{len(CASES) + len(CONTROL_CASES)} questions, pas une garantie de qualité.", ""]
    for m in modes:
        run = answer_with_agent if m == "agent" else answer_with_rag
        for title, cases in (("réglage", CASES), ("contrôle", CONTROL_CASES)):
            passed, body = 0, []
            for number, (question, expected, pages) in enumerate(cases, start=1):
                start = time.time()
                answer = run(question, doc["doc_id"])
                elapsed = time.time() - start
                ok = check(answer, expected, pages)
                passed += ok
                cites = ", ".join(f"{s} = p.{answer.sources[s].page} ({answer.sources[s].kind})"
                                  for s in answer.cited) or "aucune"
                tools = " → ".join(t["outil"] for t in answer.trace)
                block = [f"### {number}. {question}", "",
                         f"- Contrôle automatique : {'réussi' if ok else '**échec**'} — {elapsed:.0f} s",
                         f"- Outils : {tools}", f"- Citations : {cites}"]
                block += [f"- Avertissement : {w}" for w in answer.warnings]
                block += ["", *(f"> {line}" for line in answer.text.splitlines()), ""]
                body += block
                print(f"[{m} / {title}]", "\n".join(block), flush=True)
            report += [f"## Mode {m}, jeu de {title} : {passed}/{len(cases)} contrôles réussis",
                       "", *body]
    out = ROOT / "docs" / "resultats_evaluation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Rapport : {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "les-deux")
