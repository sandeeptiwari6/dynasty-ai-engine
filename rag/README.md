# Layer 4: RAG (Retrieval Augmented Generation) pipeline.

## Architecture:
  SQLite (dynasty_scout.db)           — ingestion bookkeeping, dedup, raw text
      ScoutingDocument table          — one row per scraped URL
      RagIngestionLog table           — audit log per pipeline run

  ChromaDB (chroma_db/)              — vector index for semantic search
      "dynasty_scouting" collection  — all chunks across all players/sources

  Sources indexed:
      NFL.com player bios             → SourceType.NFL_COM_BIO
      Sleeper news feed               → SourceType.SLEEPER_NOTE / INJURY_REPORT / TRANSACTION_NOTE
      ESPN draft profiles             → SourceType.DRAFT_PROFILE
      PFR game logs (→ prose)         → SourceType.GAME_RECAP
      Beat reporter articles          → SourceType.BEAT_REPORTER

Public API (used by LangGraph agent tools):
    from rag.retrieval_tool import ALL_RAG_TOOLS
    # bind to agent: agent = create_react_agent(llm, ALL_RAG_TOOLS)

    from rag.retrieval_tool import get_player_context_for_prompt
    # direct use in model_store or Streamlit

Pipeline runner:

    python -m rag.ingestion_pipeline --all          # full build
    python -m rag.ingestion_pipeline --incremental  # embed unembedded docs
    python -m rag.ingestion_pipeline --player-name "Justin Jefferson"