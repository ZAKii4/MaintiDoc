"""Orchestration agentique avec LangChain.

Le LLM choisit les outils de recherche, peut reformuler et enchaîner plusieurs recherches,
puis rédige la réponse. Trois garde-fous : nombre d'appels limité, repli sur le RAG linéaire
si aucune recherche n'a été faite, et contrôle des citations.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from . import config, retrieval
from .rag import (
    ANSWER_RULES,
    Answer,
    SourceRegistry,
    answer_with_rag,
    chat_model,
    finalize,
    history_messages,
)

AGENT_PROMPT = ANSWER_RULES + """

Tu disposes d'outils de recherche dans le document sélectionné.
- Appelle toujours au moins un outil avant de répondre.
- Choisis l'outil adapté : explications -> rechercher_texte ; chiffres, comparaisons,
  performances, paramètres -> rechercher_tableaux ; schéma, figure, architecture
  représentée -> rechercher_figures ; page citée par l'utilisateur -> lire_page.
- Le document peut être en anglais : formule les requêtes avec les termes techniques anglais.
- Si les résultats sont insuffisants, reformule ou change d'outil, sans répéter la même requête.
- Avant d'écrire qu'une information, même partielle, est absente, tu dois avoir essayé
  rechercher_texte.
- Pour une question de suivi, reformule la requête pour qu'elle soit autonome.
- Les sources des réponses précédentes ne sont plus valables : relance une recherche."""


def build_tools(doc_id: str, registry: SourceRegistry):
    """Les outils sont liés au document : le modèle ne peut pas en interroger un autre."""

    @tool
    def rechercher_texte(requete: str) -> str:
        """Cherche des passages de texte (explications, définitions, procédures)."""
        return registry.format(retrieval.text_with_context(requete, doc_id))

    @tool
    def rechercher_tableaux(requete: str) -> str:
        """Cherche des tableaux (valeurs numériques, comparaisons, paramètres, résultats)."""
        return registry.format(retrieval.tables_with_context(requete, doc_id))

    @tool
    def rechercher_figures(requete: str, requete_en: str) -> str:
        """Cherche des figures et schémas. requete : besoin à trouver ; requete_en : courte
        description en anglais de l'image recherchée (utilisée par CLIP)."""
        return registry.format(retrieval.figures_with_context(requete, doc_id, requete_en))

    @tool
    def lire_page(page: int) -> str:
        """Lit le contenu indexé d'une page précise (numérotation à partir de 1)."""
        return registry.format(retrieval.read_page(page, doc_id)[:6])

    return [rechercher_texte, rechercher_tableaux, rechercher_figures, lire_page]


def answer_with_agent(question: str, doc_id: str, history: list[tuple[str, str]] | None = None,
                      llm=None) -> Answer:
    registry = SourceRegistry()
    agent = create_agent(
        model=llm or chat_model(),
        tools=build_tools(doc_id, registry),
        system_prompt=AGENT_PROMPT,
        middleware=[ToolCallLimitMiddleware(run_limit=config.MAX_TOOL_CALLS,
                                            exit_behavior="continue")],
    )
    messages = history_messages(history or []) + [HumanMessage(content=question)]
    try:
        result = agent.invoke({"messages": messages},
                              config={"recursion_limit": 4 * config.MAX_TOOL_CALLS + 6})
    except GraphRecursionError:
        fallback = answer_with_rag(question, doc_id, history, llm=llm)
        fallback.warnings.insert(0, "Agent interrompu (trop d'étapes) : repli sur le RAG simple.")
        return fallback

    new_messages = result["messages"][len(messages):]
    trace = [{"outil": call["name"], "arguments": call["args"]}
             for m in new_messages if isinstance(m, AIMessage) for call in m.tool_calls]

    if not registry.sources:
        fallback = answer_with_rag(question, doc_id, history, llm=llm)
        fallback.warnings.insert(0, "L'agent n'a récupéré aucune source : repli sur le RAG simple.")
        fallback.trace = trace + fallback.trace
        return fallback

    final = next((m.content for m in reversed(new_messages)
                  if isinstance(m, AIMessage) and not m.tool_calls), "")
    return finalize(final, registry, trace=trace)
