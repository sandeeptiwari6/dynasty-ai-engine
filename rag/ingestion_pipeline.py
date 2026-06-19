import argparse
import json
import logging
import time
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from data.schema.database import Player, get_engine, get_session, init_db
from data.schema.database import (
    ScoutingDocument, RagIngestionLog, SourceType,
)
from utils.rag_helpers import (
    get_unembedded_docs, doc_exists, init_rag_schema, build_player_url_manifest
)
from rag.web_scraper import (
    RawDocument,
    scrape_nfl_player_bio,
    scrape_sleeper_news,
    scrape_pfr_game_log,
    scrape_espn_draft_profile,
    scrape_generic_article
)
from rag.vector_store import ScoutingVectorStore

logger = logging.getLogger(__name__)

# Seasons to scrape game logs for — most recent 3 years is sufficient
GAME_LOG_SEASONS = [2024, 2023, 2022]

# Minimum text length to bother embedding
MIN_EMBED_CHARS = 150


class RAGIngestionPipeline:
    """
    End-to-end RAG ingestion: scrape → deduplicate → store → embed.

    SQLite is the source of truth for what's been ingested and what's current.
    Chroma is the search index. They're kept in sync by this orchestrator.

    Design:
      - Idempotent: safe to re-run; already-current docs are skipped
      - Incremental: only processes docs where content has changed
      - Auditable: every run logged to rag_ingestion_log table
    """

    def __init__(self, engine=None):
        self.engine = engine or get_engine()
        init_db(self.engine)
        init_rag_schema(self.engine)
        self.vector_store = ScoutingVectorStore()
        self._run_stats = {
            "docs_scraped": 0,
            "docs_skipped": 0,
            "docs_embedded": 0,
            "chunks_added": 0,
            "chunks_deleted": 0,
            "errors": [],
        }

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run_for_all_players(self, seasons: Optional[list[int]] = None) -> dict:
        """
        Build/refresh RAG content for every skill-position player in the DB.
        Pulls player IDs from the players table — same source the ML models use.
        """
        seasons = seasons or GAME_LOG_SEASONS
        start = time.time()
        logger.info("Starting full RAG ingestion for all players...")

        with get_session(self.engine) as session:
            players = (
                session.query(Player)
                .filter(Player.position.in_(["QB", "RB", "WR", "TE"]))
                .filter(Player.status.in_(["ACT", "PUP", "IR", None]))
                .filter(Player.draft_year >= 2005)
                .all()
            )
            logger.info(f"  {len(players)} players to process")
            for player in players:
                self._process_player(session, player, seasons)

        return self._finalize_run(start)

    def run_for_player(self, player_name: str, seasons: Optional[list[int]] = None) -> dict:
        """
        Run ingestion for a single named player. Useful for testing and
        for the agent to trigger on-demand refresh when a user asks about
        a player who isn't yet indexed.
        """
        seasons = seasons or GAME_LOG_SEASONS
        start = time.time()

        with get_session(self.engine) as session:
            player = (
                session.query(Player)
                .filter(Player.name.ilike(f"%{player_name}%"))
                .first()
            )
            if not player:
                logger.warning(f"Player '{player_name}' not found in DB.")
                return {"error": f"Player '{player_name}' not found"}

            self._process_player(session, player, seasons)

        return self._finalize_run(start)

    def run_incremental(self) -> dict:
        """
        Embed all documents that have been scraped but not yet indexed in Chroma.
        Fast path — no scraping, just picks up where a previous run left off.
        """
        start = time.time()
        logger.info("Running incremental RAG embedding (unembedded docs only)...")

        with get_session(self.engine) as session:
            unembedded = get_unembedded_docs(session, limit=1000)
            logger.info(f"  {len(unembedded)} documents to embed")
            self._embed_documents(session, unembedded)

        return self._finalize_run(start)

    def ingest_custom_urls(
        self,
        urls: list[dict],
    ) -> dict:
        """
        Ingest a manually curated list of article URLs.
        Each dict: {url, player_name, player_id, season, source_type}

        Use this for beat reporter articles you find and want to add manually.
        """
        start = time.time()

        with get_session(self.engine) as session:
            for item in urls:
                doc = scrape_generic_article(
                    url=item["url"],
                    player_name=item.get("player_name", ""),
                    player_id=item.get("player_id", ""),
                    season=item.get("season"),
                    source_type=item.get("source_type", SourceType.BEAT_REPORTER),
                )
                if doc:
                    self._store_and_embed_raw_doc(session, doc)

        return self._finalize_run(start)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _process_player(
        self, session: Session, player: Player, seasons: list[int]
    ) -> None:
        """
        Scrape + index all sources for a single player.
        Catches per-player errors so one bad player doesn't stop the whole run.
        """
        try:
            manifest = build_player_url_manifest(
                player_name=player.name,
                player_id=player.player_id,
                sleeper_id=player.sleeper_id or "",
                pfr_id=player.pfr_id,
                espn_id="",          # populate from a player ID mapping table if available
                nfl_slug=self._get_nfl_slug(player),
                seasons=seasons,
            )

            for item in manifest:
                raw_doc = self._execute_manifest_item(item)
                if raw_doc and raw_doc.is_usable:
                    self._store_and_embed_raw_doc(session, raw_doc)
                    self._run_stats["docs_scraped"] += 1
                elif raw_doc:
                    logger.debug(f"Skipping short doc: {raw_doc.url} ({raw_doc.char_count} chars)")

        except Exception as e:
            msg = f"Failed processing {player.name}: {e}"
            logger.error(msg)
            self._run_stats["errors"].append(msg)

    def _execute_manifest_item(self, item: dict) -> Optional[RawDocument]:
        """Dispatch a manifest item to the correct scraper function."""
        t = item["type"]
        try:
            if t == "nfl_bio":
                return scrape_nfl_player_bio(item["player_name"], item["nfl_slug"])
            elif t == "sleeper_news":
                # Returns multiple docs; we return None here and handle separately
                docs = scrape_sleeper_news(item["sleeper_id"], item["player_name"])
                # Store all inline since there are multiple
                return docs[0] if docs else None
            elif t == "pfr_game_log":
                return scrape_pfr_game_log(item["player_name"], item["pfr_id"], item["season"])
            elif t == "espn_draft_profile":
                return scrape_espn_draft_profile(item["player_name"], item["espn_id"])
            else:
                logger.warning(f"Unknown manifest item type: {t}")
                return None
        except Exception as e:
            self._run_stats["errors"].append(f"{t} for {item.get('player_name')}: {e}")
            print(e)
            return None

    def _store_and_embed_raw_doc(
        self, session: Session, raw_doc: RawDocument
    ) -> Optional[ScoutingDocument]:
        """
        Dedup check → store in SQLite → chunk + embed into Chroma → mark embedded.

        Dedup logic:
          - If URL exists with same content hash: skip entirely
          - If URL exists with different hash: delete old Chroma chunks, re-embed
          - If URL is new: store and embed
        """
        content_hash = ScoutingDocument.compute_hash(raw_doc.raw_text)

        # Check if doc already exists and is current
        existing = doc_exists(session, raw_doc.url, content_hash)
        if existing and existing.is_embedded:
            self._run_stats["docs_skipped"] += 1
            return existing

        # Content changed or new — handle old chunks
        existing_by_url = (
            session.query(ScoutingDocument)
            .filter(ScoutingDocument.url == raw_doc.url)
            .first()
        )
        if existing_by_url and existing_by_url.is_embedded:
            # Delete stale Chroma chunks before re-embedding
            deleted = self.vector_store.delete_document_chunks(existing_by_url.id)
            self._run_stats["chunks_deleted"] += deleted
            doc = existing_by_url
            doc.raw_text = raw_doc.raw_text
            doc.content_hash = content_hash
            doc.char_count = raw_doc.char_count
            doc.scraped_at = raw_doc.scraped_at
            doc.is_embedded = False
            doc.embedded_at = None
        else:
            # New document
            doc = ScoutingDocument(
                url=raw_doc.url,
                source_type=raw_doc.source_type,
                player_id=raw_doc.player_id or "",
                player_name=raw_doc.player_name,
                season=raw_doc.season,
                team=raw_doc.team,
                title=raw_doc.title,
                raw_text=raw_doc.raw_text,
                content_hash=content_hash,
                char_count=raw_doc.char_count,
                is_usable=raw_doc.is_usable,
                scraped_at=raw_doc.scraped_at,
            )
            session.add(doc)
            session.flush()  # get doc.id before embedding

        # Embed into Chroma
        chunks_added = self.vector_store.add_document(doc)

        # Mark embedded in SQLite
        doc.is_embedded = True
        doc.embedded_at = datetime.utcnow()
        doc.chunk_count = chunks_added
        doc.embedding_model = self.vector_store._embedding_model
        session.commit()

        self._run_stats["docs_embedded"] += 1
        self._run_stats["chunks_added"] += chunks_added
        return doc

    def _embed_documents(self, session: Session, docs: list[ScoutingDocument]) -> None:
        """Embed a list of already-stored ScoutingDocuments into Chroma."""
        chunk_counts = self.vector_store.add_documents_batch(docs)
        for doc in docs:
            count = chunk_counts.get(doc.id, 0)
            doc.is_embedded = True
            doc.embedded_at = datetime.utcnow()
            doc.chunk_count = count
            doc.embedding_model = self.vector_store._embedding_model
            self._run_stats["docs_embedded"] += 1
            self._run_stats["chunks_added"] += count
        session.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # def _get_pfr_id(self, player: Player) -> str:
    #     """
    #     Derive PFR player ID from GSIS ID or name.
    #     PFR IDs are formatted like "JeffJu00", "BradTo00".
    #     This is a heuristic — a real implementation would use a mapping table.
    #     """
    #     if not player.last_name or not player.first_name:
    #         return ""
    #     last = (player.last_name or "")[:4].title()
    #     first = (player.first_name or "")[:2].title()
    #     return f"{last}{first}00"

    def _get_nfl_slug(self, player: Player) -> str:
        """Build NFL.com URL slug from player name."""
        if not player.name:
            return ""
        return player.name.lower().replace(" ", "-").replace("'", "").replace(".", "")

    def _finalize_run(self, start_time: float) -> dict:
        """Log the run to SQLite and return stats."""
        duration = round(time.time() - start_time, 1)
        stats = {**self._run_stats, "duration_seconds": duration}

        with get_session(self.engine) as session:
            log = RagIngestionLog(
                run_at=datetime.now(),
                trigger="manual",
                docs_scraped=stats["docs_scraped"],
                docs_skipped=stats["docs_skipped"],
                docs_embedded=stats["docs_embedded"],
                chunks_added=stats["chunks_added"],
                chunks_deleted=stats["chunks_deleted"],
                duration_seconds=duration,
                errors=json.dumps(stats["errors"][:20]) if stats["errors"] else None,
            )
            session.add(log)
            session.commit()

        logger.info(
            f"RAG ingestion complete in {duration}s: "
            f"{stats['docs_scraped']} scraped, "
            f"{stats['docs_skipped']} skipped, "
            f"{stats['docs_embedded']} embedded, "
            f"{stats['chunks_added']} chunks added"
        )
        if stats["errors"]:
            logger.warning(f"  {len(stats['errors'])} errors — check rag_ingestion_log table")

        return stats


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="dynasty-scout RAG ingestion")
    parser.add_argument("--all", action="store_true", help="Ingest all players")
    parser.add_argument("--player-name", type=str, help="Ingest single player by name")
    parser.add_argument("--incremental", action="store_true", help="Embed unembedded docs only")
    parser.add_argument("--seasons", nargs="+", type=int, default=GAME_LOG_SEASONS)
    args = parser.parse_args()

    pipeline = RAGIngestionPipeline()

    if args.all:
        pipeline.run_for_all_players(seasons=args.seasons)
    elif args.player_name:
        pipeline.run_for_player(args.player_name, seasons=args.seasons)
    elif args.incremental:
        pipeline.run_incremental()
    else:
        parser.print_help()
