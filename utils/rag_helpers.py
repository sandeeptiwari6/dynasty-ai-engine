from sqlalchemy.orm import Session
from data.schema.database import Base, ScoutingDocument, RagIngestionLog
from typing import Optional
import re


def init_rag_schema(engine) -> None:
    """
    Create RAG tables in the existing dynasty_scout.db.
    Safe to call after init_db() — adds only new tables.
    """
    Base.metadata.create_all(engine, tables=[
        ScoutingDocument.__table__,
        RagIngestionLog.__table__,
    ])


def get_unembedded_docs(session: Session, limit: int = 500) -> list[ScoutingDocument]:
    """Return documents scraped but not yet embedded into Chroma."""
    return (
        session.query(ScoutingDocument)
        .filter(
            ScoutingDocument.is_embedded == False,
            ScoutingDocument.is_usable == True,
        )
        .order_by(ScoutingDocument.scraped_at.desc())
        .limit(limit)
        .all()
    )


def get_stale_docs(session: Session, player_id: str, season: int) -> list[ScoutingDocument]:
    """Return all docs for a player-season (for re-indexing after news update)."""
    return (
        session.query(ScoutingDocument)
        .filter(
            ScoutingDocument.player_id == player_id,
            ScoutingDocument.season == season,
        )
        .all()
    )

def doc_exists(session: Session, url: str, content_hash: str) -> Optional[ScoutingDocument]:
    """
    Check if this URL has already been scraped with identical content.
    Returns the existing doc if hash matches (skip re-embedding),
    None if new or content has changed.
    """
    doc = session.query(ScoutingDocument).filter(
        ScoutingDocument.url == url
    ).first()
    if doc is None:
        return None
    if doc.content_hash == content_hash:
        return doc          # unchanged — skip
    return None             # content changed — re-scrape and re-embed


######################################################
################# WEB SCRAPE HELPERS #################
######################################################

def _clean_text(text: str) -> str:
    """Normalize whitespace and remove common scraping artifacts."""
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Remove lines that are clearly nav/button text (< 4 words, no period)
    lines = text.split('\n')
    cleaned = [
        line for line in lines
        if len(line.split()) >= 4 or '.' in line or line.strip() == ""
    ]
    return '\n'.join(cleaned).strip()


def build_player_url_manifest(
    player_name: str,
    player_id: str,
    sleeper_id: str = "",
    pfr_id: str = "",
    espn_id: str = "",
    nfl_slug: str = "",
    seasons: Optional[list[int]] = None,
) -> list[dict]:
    """
    Build the full list of URLs to scrape for a single player.
    Returns a list of dicts that ScrapeOrchestrator iterates over.

    Sources:
      - nfl_bio: NFL.com player page (bio, physical info, roster status)
      - sleeper_news: Sleeper status + ESPN rotowire/news (Sleeper /news deprecated)
      - game_log: nfl_data_py weekly stats per season (PFR is Cloudflare-blocked)
      - espn_draft_profile: ESPN JSON API (only when espn_id is provided)
    """
    season_list = seasons or [2024, 2023, 2022]
    manifest = []

    if nfl_slug:
        manifest.append({
            "type": "nfl_bio",
            "player_name": player_name,
            "player_id": player_id,
            "nfl_slug": nfl_slug,
        })

    if sleeper_id:
        manifest.append({
            "type": "sleeper_news",
            "player_name": player_name,
            "player_id": player_id,
            "sleeper_id": sleeper_id,
        })

    # Game logs via nfl_data_py (primary) with PFR/Playwright as fallback.
    # pfr_id is passed for the Playwright fallback path; gsis_id (=player_id)
    # is the preferred key for nfl_data_py lookup.
    for season in season_list:
        manifest.append({
            "type": "game_log",
            "player_name": player_name,
            "player_id": player_id,   # GSIS ID — used by nfl_data_py
            "pfr_id": pfr_id or "",
            "season": season,
        })

    if espn_id:
        manifest.append({
            "type": "espn_draft_profile",
            "player_name": player_name,
            "player_id": player_id,
            "espn_id": espn_id,
        })

    return manifest
