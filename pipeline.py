import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.schema.database import get_engine, init_db
from data.ingestion.nfl_ingestion import NFLIngestion
from data.ingestion.college_ingestion import CollegeIngestion
from data.ingestion.sleeper_ingestion import SleeperIngestion
from data.features.engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

def run_full_backfill(
    start_year: int = 2015,
    end_year: Optional[int] = None,
    cfbd_api_key: Optional[str] = None,
    sleeper_league_id: Optional[str] = None,
):
    """
    Complete backfill from start_year to present.
    Run this once to build the full historical dataset.
    Subsequent runs are incremental (safe to re-run).
    """
    import datetime
    end_year = end_year or datetime.datetime.now().year
    years = list(range(start_year, end_year + 1))

    logger.info("=" * 60)
    logger.info(f"dynasty-scout full backfill: {start_year}–{end_year}")
    logger.info("=" * 60)

    engine = get_engine()
    init_db(engine)

    # 1. NFL data
    logger.info("\n[1/5] NFL Player & Stats Ingestion")
    nfl_pipeline = NFLIngestion(engine)
    nfl_pipeline.run_full_backfill(start_year=start_year, end_year=end_year)

    # 2. College data
    logger.info("\n[2/5] College Stats & Combine Ingestion")
    cfbd_key = cfbd_api_key or os.getenv("CFBD_API_KEY")
    if cfbd_key:
        college_pipeline = CollegeIngestion(engine, api_key=cfbd_key)
        college_pipeline.run_full_backfill(start_year=max(2010, start_year - 5))
    else:
        logger.warning("  CFBD_API_KEY not set — skipping college stats. "
                       "Get a free key at https://collegefootballdata.com/key")

    # 3. Sleeper data
    logger.info("\n[3/5] Sleeper API Ingestion")
    sleeper_pipeline = SleeperIngestion(engine)
    sleeper_pipeline.ingest_players()    # maps sleeper IDs → GSIS IDs
    # sleeper_pipeline.ingest_trending_snapshot()  # today's trending adds/drops

    # 4. Feature engineering
    logger.info("\n[4/5] Feature Engineering")
    feature_pipeline = FeatureEngineer(engine)
    feature_pipeline.run(seasons=years)

    logger.info("\n[5/5] Done! Summary:")
    _print_summary(engine)

    return engine

def run_annual_refresh(year: int, cfbd_api_key: Optional[str] = None):
    """
    Pull just the latest year of data and recompute features.
    Run this at the start of each NFL season (late August/September).
    """
    logger.info(f"dynasty-scout annual refresh for {year}")
    engine = get_engine()

    # Pull current year
    NFLIngestion(engine).run_full_backfill(start_year=year, end_year=year)

    cfbd_key = cfbd_api_key or os.getenv("CFBD_API_KEY")
    if cfbd_key:
        CollegeIngestion(engine, api_key=cfbd_key).run_full_backfill(
            start_year=year - 1, end_year=year - 1  # last college season
        )

    SleeperIngestion(engine).ingest_trending_snapshot()

    # Recompute features for last 3 years (rolling features need context)
    FeatureEngineer(engine).run(seasons=[year - 2, year - 1, year])

    logger.info("Annual refresh complete.")
    return engine

# ------------------------------------------------------------------
# ML data loading interface
# ------------------------------------------------------------------

def load_features_for_ml(
    position: Optional[str] = None,
    min_season: int = 2015,
    max_season: int = 2023,
    require_target: bool = True,
    min_games: int = 6,
    player_type: str = "nfl",
) -> pd.DataFrame:
    """
    Load the final feature store as a pandas DataFrame for ML training.
    This is the ONLY function ML models need to call.

    Args:
        position: Filter to one position ("QB", "RB", "WR", "TE") or None for all
        min_season: Earliest season to include (avoid old data for modern metrics)
        max_season: Latest season to include (hold out recent for evaluation)
        require_target: If True, drop rows with no next-season label (training only)
        min_games: Minimum games played to include (filter out IR stints)
        player_type: "nfl" for veterans, "college" for prospects

    Returns:
        DataFrame with one row per player-season, ready for sklearn/LightGBM.
    """
    engine = get_engine()

    query = """
        SELECT ef.*, p.name as player_name, p.nfl_team, p.sleeper_id
        FROM engineered_features ef
        JOIN players p ON ef.player_id = p.player_id
        WHERE ef.season BETWEEN :min_season AND :max_season
          AND ef.player_type = :player_type
    """
    params = {"min_season": min_season, "max_season": max_season, "player_type": player_type}

    if position:
        query += " AND ef.position = :position"
        params["position"] = position

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    # Apply filters
    if min_games > 0 and "games_played" in df.columns:
        df = df[df["games_played"] >= min_games]

    if require_target:
        df = df[df["fantasy_ppg_next_season"].notna()]

    logger.info(
        f"Loaded {len(df)} rows for ML training "
        f"(position={position or 'all'}, seasons={min_season}–{max_season})"
    )
    return df


def get_feature_columns(position: Optional[str] = None) -> dict[str, list[str]]:
    """
    Returns feature column groups for use in model training.
    Organized so you can easily ablate feature groups.
    """
    performance = [
        "fantasy_ppg_last_season", "fantasy_ppg_2yr_avg", "fantasy_ppg_3yr_avg",
        "fantasy_ppg_trend", "games_played", "career_games",
    ]
    role = [
        "target_share", "air_yards_share", "wopr", "snap_pct", "snap_pct_trend",
        "carries_per_game", "targets_per_game",
    ]
    efficiency = [
        "yards_per_target", "yards_per_carry", "yards_after_catch_per_rec",
        "racr", "epa_per_play", "cpoe", "ryoe_per_att",
        "separation_avg", "catch_pct_above_expected",
    ]
    injury = [
        "games_missed_last_season", "games_missed_2yr_total", "injury_risk_score",
        "soft_tissue_injury_flag", "acl_history_flag", "concussion_history_count",
    ]
    age = [
        "age", "age_at_nfl_entry", "years_experience", "age_vs_position_peak",
    ]
    team_context = [
        "team_pass_rate", "team_pass_rate_neutral", "team_plays_per_game",
        "team_pass_attempts", "team_points_per_game", "offensive_line_rank",
        "new_team_flag", "new_oc_flag", "scheme_fit_score",
    ]
    college_prospect = [
        "dominator_rating", "breakout_age", "college_yards_per_game",
        "college_tds_per_game", "college_conference_tier", "draft_round",
        "draft_pick_normalized", "sparq_score", "relative_athletic_score",
        "speed_score", "forty_yard", "vertical_jump",
    ]

    base_features = performance + role + efficiency + injury + age + team_context

    if position == "QB":
        # QBs don't need receiving features; add passing-specific ones
        base_features = [
            f for f in base_features
            if f not in ("target_share", "air_yards_share", "wopr", "targets_per_game",
                         "yards_per_target", "racr", "separation_avg", "catch_pct_above_expected")
        ]
    elif position == "RB":
        # RBs receiving role matters less; rushing efficiency more
        pass
    elif position in ("WR", "TE"):
        # Remove rushing-specific features
        base_features = [f for f in base_features if f not in ("carries_per_game", "ryoe_per_att")]

    return {
        "all_features": base_features + college_prospect,
        "nfl_features": base_features,
        "college_features": college_prospect + age,
        "performance": performance,
        "role": role,
        "efficiency": efficiency,
        "injury": injury,
        "age_curve": age,
        "team_context": team_context,
        "prospect": college_prospect,
        "target": "fantasy_ppg_next_season",
    }

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def _print_summary(engine):
    """Print row counts for all tables."""
    tables = [
        "players", "nfl_season_stats", "nfl_advanced_stats",
        "nfl_weekly_snaps", "injury_records",
        "college_season_stats", "combine_measurements",
        "engineered_features", "sleeper_league_snapshots"
    ]
    with engine.connect() as conn:
        for table in tables:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                logger.info(f"  {table:<35} {count:>8,} rows")
            except Exception:
                pass


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    """
    pipeline.py
    -----------
    Master orchestration script for the dynasty-scout data layer.
    Run this script to build the entire feature store from scratch.

    Also exposes load_features_for_ml() — the single function called by
    the ML model training notebooks and LangGraph agent tools.

    Usage (first time):
        python pipeline.py --backfill --start-year 2015 --end-year 2024 --cfbd-key $CFBD_API_KEY --league-id $SLEEPER_LEAGUE_ID

    Usage (annual refresh):
        python pipeline.py --refresh --year 2025
    """

    parser = argparse.ArgumentParser(description="dynasty-scout data pipeline")
    parser.add_argument("--backfill", action="store_true", help="Run full historical backfill")
    parser.add_argument("--refresh", action="store_true", help="Pull latest year only")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--year", type=int, help="Year for annual refresh")
    parser.add_argument("--cfbd-key", type=str, help="College Football Data API key")
    parser.add_argument("--league-id", type=str, help="Sleeper league ID")
    args = parser.parse_args()

    if args.backfill:
        run_full_backfill(
            start_year=args.start_year,
            end_year=args.end_year,
            cfbd_api_key=args.cfbd_key,
            sleeper_league_id=args.league_id,
        )
    elif args.refresh:
        import datetime
        year = args.year or datetime.datetime.now().year
        run_annual_refresh(year=year, cfbd_api_key=args.cfbd_key)
    else:
        parser.print_help()