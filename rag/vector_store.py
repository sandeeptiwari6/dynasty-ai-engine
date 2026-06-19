import hashlib
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions

from data.schema.database import ScoutingDocument, SourceType

logger = logging.getLogger(__name__)

# Chroma persists here — sits alongside dynasty_scout.db
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Collection name — one collection for all scouting content
# (metadata filtering handles player/season scoping at query time)
COLLECTION_NAME = "dynasty_scouting"

# Chunking parameters
DEFAULT_CHUNK_SIZE = 500        # characters
DEFAULT_CHUNK_OVERLAP = 100

# Sources that should NOT be chunked (already small, self-contained)
NO_CHUNK_SOURCES = {
    SourceType.SLEEPER_NOTE,
    SourceType.TRANSACTION_NOTE,
    SourceType.INJURY_REPORT,
}

# Embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ScoutingVectorStore:
    """
    ChromaDB-backed vector store for dynasty scouting content.

    Wraps all Chroma operations so the rest of the codebase never
    imports chromadb directly — makes it easy to swap the backend later.

    Usage:
        store = ScoutingVectorStore()
        store.add_document(scouting_doc)
        results = store.retrieve("Justin Jefferson injury history", player_id="...", k=5)
    """

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = COLLECTION_NAME,
    ):
        chroma_dir.mkdir(parents=True, exist_ok=True)

        # Persistent client — data survives process restarts
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Embedding function — sentence-transformers runs locally
        # Swap to OpenAIEmbeddingFunction for higher quality (requires API key):
        #   from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        #   self._embed_fn = OpenAIEmbeddingFunction(api_key=..., model_name="text-embedding-3-small")
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._embedding_model = embedding_model

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},   # cosine similarity for text
        )

        logger.info(
            f"ScoutingVectorStore initialized: {self._collection.count()} chunks in '{collection_name}'"
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_document(self, doc: ScoutingDocument) -> int:
        """
        Chunk a ScoutingDocument and upsert all chunks into Chroma.
        Returns the number of chunks added.

        Uses upsert (not add) so re-running after content changes replaces
        stale chunks rather than duplicating them.
        """
        chunks = self._chunk_document(doc)
        if not chunks:
            logger.warning(f"No chunks produced for doc {doc.id} ({doc.url})")
            return 0

        ids = []
        texts = []
        metadatas = []

        for i, chunk_text in enumerate(chunks):
            # Chunk ID: deterministic from doc_id + chunk index
            # Deterministic IDs enable upsert to replace old chunks correctly
            chunk_id = f"doc_{doc.id}_chunk_{i}"
            ids.append(chunk_id)
            texts.append(chunk_text)
            metadatas.append(self._build_metadata(doc, i, len(chunks)))

        # Upsert: insert if new, replace if chunk_id already exists
        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        logger.debug(f"Upserted {len(chunks)} chunks for '{doc.player_name}' ({doc.source_type})")
        return len(chunks)

    def delete_document_chunks(self, doc_id: int) -> int:
        """
        Delete all chunks belonging to a document (before re-indexing updated content).
        Returns number of chunks deleted.
        """
        # Chroma metadata filtering to find all chunks for this doc
        results = self._collection.get(
            where={"doc_id": doc_id},
            include=[],   # IDs only — no need to fetch text
        )
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def add_documents_batch(self, docs: list[ScoutingDocument]) -> dict[str, int]:
        """
        Batch add multiple documents. More efficient than calling add_document
        in a loop because it amortizes the embedding model's overhead.
        Returns {doc_id: chunk_count} mapping.
        """
        all_ids, all_texts, all_metadatas = [], [], []
        doc_chunk_counts = {}

        for doc in docs:
            chunks = self._chunk_document(doc)
            if not chunks:
                doc_chunk_counts[doc.id] = 0
                continue

            for i, chunk_text in enumerate(chunks):
                chunk_id = f"doc_{doc.id}_chunk_{i}"
                all_ids.append(chunk_id)
                all_texts.append(chunk_text)
                all_metadatas.append(self._build_metadata(doc, i, len(chunks)))

            doc_chunk_counts[doc.id] = len(chunks)

        if all_ids:
            # Chroma handles batching internally — safe to upsert thousands at once
            self._collection.upsert(
                ids=all_ids,
                documents=all_texts,
                metadatas=all_metadatas,
            )
            logger.info(f"Batch upserted {len(all_ids)} chunks from {len(docs)} documents")

        return doc_chunk_counts

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        player_id: Optional[str] = None,
        player_name: Optional[str] = None,
        season: Optional[int] = None,
        source_types: Optional[list[str]] = None,
        k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the k most relevant chunks for a query.

        Metadata filtering happens INSIDE Chroma (not in Python after retrieval)
        so the ANN search only considers matching documents.

        Args:
            query:        Natural language query (e.g. "injury history 2023")
            player_id:    Filter to one player (most common case)
            player_name:  Alternative player filter (used when player_id unknown)
            season:       Filter to a specific season
            source_types: Filter to specific source types (e.g. [INJURY_REPORT])
            k:            Number of chunks to return

        Returns:
            List of dicts with keys: text, player_name, source_type, season,
            url, doc_id, relevance_score, chunk_index
        """
        where = self._build_where_filter(player_id, player_name, season, source_types)

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self._collection.count() or 1),
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []

        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        output = []
        for text, meta, dist in zip(docs_list, metas_list, dists_list):
            # Convert cosine distance → similarity score (0-1, higher = more relevant)
            similarity = max(0.0, 1.0 - dist)
            output.append({
                "text": text,
                "player_name": meta.get("player_name", ""),
                "player_id": meta.get("player_id", ""),
                "source_type": meta.get("source_type", ""),
                "season": meta.get("season"),
                "url": meta.get("url", ""),
                "doc_id": meta.get("doc_id"),
                "chunk_index": meta.get("chunk_index", 0),
                "relevance_score": round(similarity, 4),
            })

        return output

    def retrieve_for_player(
        self,
        player_id: str,
        player_name: str,
        query: str = "",
        k: int = 8,
    ) -> str:
        """
        Retrieve and format all relevant context for a player as a single
        string block suitable for injecting into an LLM prompt.

        If query is empty, retrieves the most recent/relevant chunks across
        all source types. If query is provided, retrieves semantically closest.

        Returns formatted string:
          [NFL.com Bio]
          Justin Jefferson, WR, Minnesota Vikings...

          [Injury Report — 2024]
          Jefferson was limited in practice Wednesday with a hamstring...
        """
        if not query:
            # No query — fetch broadly with a player-description query
            query = f"{player_name} NFL fantasy football scouting report performance"

        chunks = self.retrieve(player_id=player_id, query=query, k=k)

        if not chunks:
            return f"No scouting context available for {player_name}."

        # Group by source_type for readable formatting
        by_source: dict[str, list[str]] = {}
        for chunk in chunks:
            src = chunk["source_type"].replace("_", " ").title()
            season = chunk.get("season")
            key = f"{src}" + (f" — {season}" if season else "")
            by_source.setdefault(key, []).append(chunk["text"])

        lines = []
        for source_label, texts in by_source.items():
            lines.append(f"[{source_label}]")
            lines.append("\n".join(texts))
            lines.append("")

        return "\n".join(lines).strip()

    def get_collection_stats(self) -> dict:
        """Return summary stats about what's indexed."""
        count = self._collection.count()
        # Sample metadata to understand coverage
        if count == 0:
            return {"total_chunks": 0}

        sample = self._collection.get(limit=min(500, count), include=["metadatas"])
        metas = sample.get("metadatas", [])

        players = set(m.get("player_name", "") for m in metas)
        sources = {}
        for m in metas:
            src = m.get("source_type", "unknown")
            sources[src] = sources.get(src, 0) + 1

        return {
            "total_chunks": count,
            "unique_players_sampled": len(players),
            "source_type_distribution": sources,
        }

    # ------------------------------------------------------------------
    # Internal: chunking
    # ------------------------------------------------------------------

    def _chunk_document(self, doc: ScoutingDocument) -> list[str]:
        """
        Split document text into chunks based on source type.

        Strategy:
          - Short sources (Sleeper notes, injury reports): no chunking
          - Game logs: split on week boundaries ("\nWeek " markers)
          - Everything else: fixed-size with overlap
        """
        text = doc.raw_text.strip()
        if not text:
            return []

        # No-chunk sources — return as single chunk
        if doc.source_type in NO_CHUNK_SOURCES or len(text) <= DEFAULT_CHUNK_SIZE:
            return [text]

        # Game logs: split on week boundaries to keep game lines together
        if doc.source_type == SourceType.GAME_RECAP:
            return self._chunk_by_week(text)

        # Default: fixed-size with overlap
        return self._chunk_fixed_size(text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

    def _chunk_fixed_size(self, text: str, size: int, overlap: int) -> list[str]:
        """
        Split text into overlapping fixed-size chunks.
        Tries to split on sentence boundaries (". ") within the window
        to avoid cutting mid-sentence.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            if end < len(text):
                # Try to end at a sentence boundary within the last 100 chars
                boundary = text.rfind(". ", start + size - 100, end)
                if boundary != -1:
                    end = boundary + 1  # include the period

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap  # step back by overlap for next chunk

        return chunks

    def _chunk_by_week(self, text: str) -> list[str]:
        """
        Split game log text on "Week N" boundaries.
        Groups 3 weeks per chunk so each chunk has enough context.
        """
        lines = text.split("\n")
        header_lines = []
        week_lines = []

        for line in lines:
            if line.startswith("Week ") or (len(line) < 10 and line.strip()):
                week_lines.append(line)
            else:
                if not week_lines:
                    header_lines.append(line)
                else:
                    week_lines.append(line)

        # Group into chunks of 3 weeks
        chunks = []
        header = "\n".join(header_lines)
        for i in range(0, len(week_lines), 3):
            group = week_lines[i:i+3]
            chunk = (header + "\n" + "\n".join(group)).strip()
            if chunk:
                chunks.append(chunk)

        return chunks or [text]

    def _build_metadata(self, doc: ScoutingDocument, chunk_idx: int, total_chunks: int) -> dict:
        """
        Build the Chroma metadata dict for a chunk.
        All values must be str, int, float, or bool — Chroma doesn't support None.
        """
        return {
            "doc_id": doc.id or 0,
            "player_id": doc.player_id or "",
            "player_name": doc.player_name or "",
            "source_type": doc.source_type or "",
            "season": doc.season or 0,
            "team": doc.team or "",
            "url": doc.url or "",
            "chunk_index": chunk_idx,
            "total_chunks": total_chunks,
            "embedding_model": self._embedding_model,
        }

    def _build_where_filter(
        self,
        player_id: Optional[str],
        player_name: Optional[str],
        season: Optional[int],
        source_types: Optional[list[str]],
    ) -> Optional[dict]:
        """
        Build a Chroma metadata filter dict.
        Uses $and operator when multiple filters are needed.
        Returns None if no filters specified (retrieve from all chunks).
        """
        conditions = []

        if player_id:
            conditions.append({"player_id": {"$eq": player_id}})
        elif player_name:
            # Name matching is less reliable — use only when no player_id
            conditions.append({"player_name": {"$eq": player_name}})

        if season:
            conditions.append({"season": {"$eq": season}})

        if source_types and len(source_types) == 1:
            conditions.append({"source_type": {"$eq": source_types[0]}})
        elif source_types:
            conditions.append({"source_type": {"$in": source_types}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
