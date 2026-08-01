import logging
from functools import lru_cache
from typing import Optional

from langchain_core.tools import tool

from rag.vector_store import ScoutingVectorStore
from data.schema.database import SourceType

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, reused across all agent tool calls
# lru_cache(maxsize=1) ensures only one ScoutingVectorStore instance exists
@lru_cache(maxsize=1)
def _get_vector_store() -> ScoutingVectorStore:
    return ScoutingVectorStore()


# ---------------------------------------------------------------------------
# LangChain tools — decorated with @tool so LangGraph can bind them to agents
# ---------------------------------------------------------------------------

@tool
def retrieve_player_context(player_name: str, query: str = "", season: int = 0) -> str:
    """
    Retrieve scouting context for an NFL player from the vector store.
    Returns a formatted string containing relevant chunks from bio pages,
    beat reporter articles, game logs, and depth chart notes.

    Args:
        player_name: Full player name (e.g. "Justin Jefferson")
        query: Specific aspect to focus on (e.g. "route running separation")
               Leave empty for a broad overview.
        season: Filter to a specific season year (0 = all seasons)

    Returns:
        Formatted scouting context string ready to inject into LLM prompt.
    """
    store = _get_vector_store()

    # Resolve player_id from name for precise metadata filtering
    player_id = _resolve_player_id(player_name)

    context = store.retrieve_for_player(
        player_id=player_id or "",
        player_name=player_name,
        query=query or f"{player_name} NFL performance scouting",
        k=8,
    )

    # Cold start: if nothing found, trigger on-demand ingestion
    if "No scouting context available" in context and player_id:
        logger.info(f"No context for {player_name} — triggering on-demand ingest")
        _on_demand_ingest_sync(player_name)
        # Retry after ingest
        context = store.retrieve_for_player(
            player_id=player_id,
            player_name=player_name,
            query=query or f"{player_name} NFL performance",
            k=8,
        )

    return context


@tool
def retrieve_injury_context(player_name: str, seasons: str = "") -> str:
    """
    Retrieve injury history and recent injury reports for a player.
    Filters specifically to injury report source types for precision.

    Args:
        player_name: Full player name
        seasons: Comma-separated season years to filter (e.g. "2022,2023,2024")
                 Leave empty for all available seasons.

    Returns:
        Formatted injury history string.
    """
    store = _get_vector_store()
    player_id = _resolve_player_id(player_name)

    injury_sources = [
        SourceType.INJURY_REPORT,
        SourceType.BEAT_REPORTER,
        SourceType.SLEEPER_NOTE,
    ]

    query = f"{player_name} injury report missed games practice limited status"

    results = store.retrieve(
        query=query,
        player_id=player_id or None,
        player_name=player_name,
        source_types=injury_sources,
        k=6,
    )

    if not results:
        return f"No injury history indexed for {player_name}. Check the ingestion pipeline."

    # Format with dates and sources for temporal context
    lines = [f"Injury context for {player_name}:\n"]
    for r in results:
        season_str = f" ({r['season']})" if r.get("season") else ""
        src = r["source_type"].replace("_", " ").title()
        lines.append(f"[{src}{season_str}] {r['text']}")
        lines.append("")

    return "\n".join(lines).strip()


@tool
def retrieve_news_context(player_name: str) -> str:
    """
    Retrieve the most recent news, transactions, and depth chart notes for a player.
    Prioritizes recency — useful for the dynasty advisor agent to check for
    recent FA signings, injuries, or role changes before making recommendations.

    Args:
        player_name: Full player name

    Returns:
        Formatted recent news string.
    """
    store = _get_vector_store()
    player_id = _resolve_player_id(player_name)

    news_sources = [
        SourceType.SLEEPER_NOTE,
        SourceType.TRANSACTION_NOTE,
        SourceType.INJURY_REPORT,
        SourceType.DEPTH_CHART,
    ]

    results = store.retrieve(
        query=f"{player_name} recent news transaction depth chart injury",
        player_id=player_id or None,
        player_name=player_name,
        source_types=news_sources,
        k=5,
    )

    if not results:
        return f"No recent news indexed for {player_name}."

    lines = [f"Recent news for {player_name}:\n"]
    for r in results:
        lines.append(f"• {r['text']}")

    return "\n".join(lines)


@tool
def retrieve_college_scouting(player_name: str) -> str:
    """
    Retrieve college scouting report and draft profile context for a prospect.
    Used by the college scout agent for rookie evaluations.

    Args:
        player_name: Player name (college or NFL name)

    Returns:
        Draft profile and college scouting text.
    """
    store = _get_vector_store()
    player_id = _resolve_player_id(player_name)

    prospect_sources = [
        SourceType.DRAFT_PROFILE,
        SourceType.COLLEGE_PROFILE,
        SourceType.NFL_COM_BIO,
    ]

    results = store.retrieve(
        query=f"{player_name} draft prospect college scouting report strengths weaknesses",
        player_id=player_id or None,
        player_name=player_name,
        source_types=prospect_sources,
        k=6,
    )

    if not results:
        return f"No draft profile indexed for {player_name}. Run ingestion with ESPN draft profile URL."

    lines = [f"Draft scouting context for {player_name}:\n"]
    for r in results:
        src = r["source_type"].replace("_", " ").title()
        lines.append(f"[{src}]\n{r['text']}\n")

    return "\n".join(lines).strip()


@tool
def on_demand_ingest(player_name: str) -> str:
    """
    Trigger on-demand scraping and indexing for a player not yet in the vector store.
    The LangGraph agent calls this when retrieve_player_context returns empty.

    Args:
        player_name: Player to ingest

    Returns:
        Status message indicating success or failure.
    """
    return _on_demand_ingest_sync(player_name)


# ---------------------------------------------------------------------------
# Non-tool retrieval functions (called directly by model_store, not agent tools)
# ---------------------------------------------------------------------------

def get_player_context_for_prompt(
    player_id: str,
    player_name: str,
    query: str = "",
    k: int = 6,
) -> str:
    """
    Direct retrieval for use in model_store.py and Streamlit UI —
    not decorated as a LangChain tool since it's called programmatically.

    Returns formatted context string for injecting into LLM system prompts.
    """
    store = _get_vector_store()
    return store.retrieve_for_player(
        player_id=player_id,
        player_name=player_name,
        query=query,
        k=k,
    )


def get_vector_store_stats() -> dict:
    """Return stats about what's currently indexed — used in Streamlit dashboard."""
    return _get_vector_store().get_collection_stats()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_player_id(player_name: str) -> Optional[str]:
    """
    Look up player_id from name via SQLite.
    Returns None if not found — retrieval falls back to name-based filtering.
    """
    try:
        from data.schema.database import get_engine
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            # Try exact match first
            result = conn.execute(
                text("SELECT player_id FROM players WHERE LOWER(name) = LOWER(:name) LIMIT 1"),
                {"name": player_name}
            ).fetchone()
            if result:
                return result[0]
            # Fuzzy: last name match
            last = player_name.strip().split()[-1] if player_name.strip() else ""
            result = conn.execute(
                text("SELECT player_id FROM players WHERE LOWER(name) LIKE LOWER(:pat) LIMIT 1"),
                {"pat": f"%{last}%"}
            ).fetchone()
            return result[0] if result else None
    except Exception as e:
        logger.debug(f"player_id lookup failed for {player_name}: {e}")
        return None


def _on_demand_ingest_sync(player_name: str) -> str:
    """
    Synchronous on-demand ingestion — scrapes and indexes a single player.
    Kept synchronous so it works cleanly inside LangGraph's sync tool execution.
    """
    try:
        from rag.ingestion_pipeline import RAGIngestionPipeline
        pipeline = RAGIngestionPipeline()
        stats = pipeline.run_for_player(player_name, seasons=[2024, 2023])
        chunks = stats.get("chunks_added", 0)
        if chunks > 0:
            # Invalidate the lru_cache so new chunks are visible immediately
            _get_vector_store.cache_clear()
            return f"Successfully indexed {chunks} chunks for {player_name}."
        else:
            return f"Ingestion ran for {player_name} but no content found. Player may lack public sources."
    except Exception as e:
        logger.error(f"On-demand ingest failed for {player_name}: {e}")
        return f"Ingestion failed for {player_name}: {e}"


# ---------------------------------------------------------------------------
# All tools as a list — for binding to LangGraph agents
# ---------------------------------------------------------------------------

ALL_RAG_TOOLS = [
    retrieve_player_context,
    retrieve_injury_context,
    retrieve_news_context,
    retrieve_college_scouting,
    on_demand_ingest,
]
