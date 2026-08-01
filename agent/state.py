import logging
from typing import Annotated, Optional, TypedDict

from langchain_core.tools import tool
from langgraph.graph.message import add_messages

from models.model_store import ModelStore
from rag.retrieval_tool import (
    retrieve_player_context,
    retrieve_injury_context,
    retrieve_news_context,
    retrieve_college_scouting,
    on_demand_ingest,
)

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, reused across every agent turn.
# Mirrors the lru_cache pattern used in rag/retrieval_tool.py for the vector store.
_model_store: Optional[ModelStore] = None


def get_model_store() -> ModelStore:
    """Lazy-load the ModelStore singleton. Models must already be trained on disk."""
    global _model_store
    if _model_store is None:
        _model_store = ModelStore()
        _model_store.load_all()
    return _model_store


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class DynastyState(TypedDict):
    """
    Shared state passed between every node in the graph.

    messages uses the add_messages reducer — LangGraph appends rather than
    overwrites on every node transition, which is what gives the supervisor
    the full conversational + tool-call history when deciding what to do next.

    add_messages (not a bare operator.add) is required so that plain dict
    messages passed into graph.invoke({"messages": [{"role": "user", ...}]})
    are coerced into LangChain BaseMessage objects. With operator.add the dicts
    would be appended verbatim and node code doing `msg.content` would raise
    `AttributeError: 'dict' object has no attribute 'content'`.
    """
    messages: Annotated[list, add_messages]

    # Routing
    query_type: str              # "nfl_analysis" | "college_scouting" | "dynasty_advice" | "injury_check"
    player_name: str
    season: int

    # Populated by sub-agent nodes, read by report_writer
    ml_prediction: dict           # PredictionResult.to_dict() output
    rag_context: str              # formatted scouting text
    final_report: str


# ---------------------------------------------------------------------------
# ML prediction tools — thin wrappers around ModelStore
# ---------------------------------------------------------------------------

@tool
def predict_nfl_performance(player_name: str, season: int) -> str:
    """
    Predict next-season fantasy PPG for an established NFL player using the
    LightGBM forecaster, including injury risk and SHAP-based explanations.

    Args:
        player_name: Full player name (e.g. "Justin Jefferson")
        season: The upcoming season to project (e.g. 2025)

    Returns:
        Formatted prediction string with floor/ceiling, confidence, and
        the factors driving the projection up or down.
    """
    store = get_model_store()
    player_id = store.get_player_id_by_name(player_name)
    if not player_id:
        return f"Could not find player '{player_name}' in the database."

    features = store.get_player_features(player_id, season - 1)
    if features is None:
        return (
            f"No feature data found for {player_name} in season {season - 1}. "
            f"This player may be a rookie — try predict_college_prospect instead."
        )

    result = store.predict_player(player_id, season, features, player_name)
    if result is None:
        return f"No model available for {player_name}'s position."

    return result.to_prose()


@tool
def predict_injury_risk(player_name: str, season: int) -> str:
    """
    Predict the probability that a player misses 4+ games next season,
    using the calibrated logistic regression injury model.

    Args:
        player_name: Full player name
        season: The upcoming season to assess risk for

    Returns:
        Formatted risk assessment with probability, risk tier, and
        the injury history factors driving the score.
    """
    store = get_model_store()
    player_id = store.get_player_id_by_name(player_name)
    if not player_id:
        return f"Could not find player '{player_name}' in the database."

    features = store.get_player_features(player_id, season - 1)
    if features is None:
        return f"No feature data found for {player_name} in season {season - 1}."

    result = store.predict_injury(player_id, season, features, player_name)
    if result is None:
        return "Injury model not available."

    return result.to_prose()


@tool
def predict_college_prospect(player_name: str, draft_season: int) -> str:
    """
    Predict year-3 NFL fantasy PPG for an incoming college prospect using
    college production, combine athleticism, and draft capital — no NFL
    history required.

    Args:
        player_name: Player's name as drafted
        draft_season: The NFL season they were/will be drafted into

    Returns:
        Formatted prediction with historical comps and prospect archetype.
    """
    store = get_model_store()
    player_id = store.get_player_id_by_name(player_name)
    if not player_id:
        return (
            f"Could not find '{player_name}' in the database. "
            f"College data may not yet be ingested for this prospect."
        )

    features = store.get_player_features(player_id, draft_season)
    if features is None:
        return f"No college feature data found for {player_name}."

    result = store.predict_prospect(player_id, draft_season, features, player_name)
    if result is None:
        return f"No college translator available for {player_name}'s position."

    return result.to_prose()


# ---------------------------------------------------------------------------
# Tool groups — bound to different sub-agents in graph.py
# ---------------------------------------------------------------------------

ML_TOOLS = [
    predict_nfl_performance,
    predict_injury_risk,
    predict_college_prospect,
]

RAG_TOOLS = [
    retrieve_player_context,
    retrieve_injury_context,
    retrieve_news_context,
    retrieve_college_scouting,
    on_demand_ingest,
]

ALL_TOOLS = ML_TOOLS + RAG_TOOLS
