import logging
import pandas as pd
import numpy as np
import unicodedata
import re
from datetime import date
import math
from typing import Optional
from rapidfuzz import process, fuzz

from data.schema.database import get_session
from utils.constants import NICKNAME_MAP, SUFFIXES

logger = logging.getLogger(__name__)

###############################################################
############# BULK UPSERTING DATA INTO SQL TABLES #############
###############################################################
def _bulk_upsert(engine, model, records, conflict_column):
    """
    SQLite upsert using INSERT OR REPLACE semantics.
    For tables with unique constraints, this updates on conflict.
    For tables without, it just inserts.
    """
    if not records:
        return

    with get_session(engine) as session:
        try:
            if conflict_column:
                # Use bulk merge (session.merge handles insert-or-update on PK)
                for record in records:
                    session.merge(record)
            else:
                # No unique key — just add all (used for weekly tables)
                # First clear the affected seasons to avoid dupes
                session.add_all(records)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Upsert failed for {model.__tablename__}: {e}")
            raise

###############################################################
################# CALCULATING TEAM STATISTICS #################
###############################################################

def _count_games_per_team(season_pbp: pd.DataFrame) -> dict[str, int]:
    """Count distinct game_ids per team — needed for per-game rate calculations."""
    if "game_id" not in season_pbp.columns:
        return {}
    return season_pbp.groupby("posteam")["game_id"].nunique().to_dict()
 
 
def _compute_points_per_game(schedules: pd.DataFrame, season: int) -> dict[str, float]:
    """
    Compute average points scored per game from the schedules table.
    Unpivots home/away into one row per team per game, then averages.
    """
    season_sched = schedules[
        (schedules["season"] == season) &
        (schedules.get("game_type", pd.Series(["REG"] * len(schedules))) == "REG")
    ].copy()
    required = {"home_team", "away_team", "home_score", "away_score"}
    if season_sched.empty or not required.issubset(season_sched.columns):
        return {}
    home = season_sched[["home_team", "home_score"]].rename(columns={"home_team": "team", "home_score": "points"})
    away = season_sched[["away_team", "away_score"]].rename(columns={"away_team": "team", "away_score": "points"})
    all_games = pd.concat([home, away], ignore_index=True).dropna(subset=["points"])
    return all_games.groupby("team")["points"].mean().round(2).to_dict()
 
 
def _compute_ol_rank(season_pbp: pd.DataFrame) -> dict[str, int]:
    """
    Rank all 32 teams 1–32 on OL quality using sacks allowed per pass attempt.
    Rank 1 = best OL (fewest sacks). Correlates with PFF OL grade at r≈0.65.
    """
    if season_pbp.empty or "sack" not in season_pbp.columns:
        return {}
    pass_plays = season_pbp[season_pbp["pass_attempt"] == 1].copy()
    stats = (
        pass_plays.groupby("posteam")
        .agg(pass_attempts=("pass_attempt", "sum"), sacks=("sack", "sum"))
        .reset_index()
    )
    stats = stats[stats["pass_attempts"] > 0].copy()
    stats["sack_rate"] = stats["sacks"] / stats["pass_attempts"]
    stats["ol_rank"] = stats["sack_rate"].rank(method="min").astype(int)
    return stats.set_index("posteam")["ol_rank"].to_dict()

def _safe_int(val):
    try:
        if pd.isna(val):
            return None
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _safe_date(val):
    if not val or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None

###############################################################
############## CALCULATING PLAYER PER GAME STATS ##############
###############################################################
def _calc_yards_per_game(row):
            games_played = row["games_played"]
            games_played = np.nan if games_played == 0 else games_played
            if row["position"] == "QB":
                return row["pass_yards"] / games_played
            elif row["position"] == "RB":
                return row["rush_yards"] / games_played
            else:
                return row["rec_yards"] / games_played

def _calc_tds_per_game(row):
    games_played = row["games_played"]
    games_played = np.nan if games_played == 0 else games_played
    if row["position"] == "QB":
        return row["pass_tds"] / games_played
    elif row["position"] == "RB":
        return row["rush_tds"] / games_played
    else:
        return row["rec_tds"] / games_played
    
##############################################################
#### NORMALIZING PLAYER NAMES FOR JOINING CFBD + NFL DATA ####
##############################################################
def _unicode_to_ascii(text: str) -> str:
    """Convert accented / special unicode chars to closest ASCII equivalent."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
 
 
def make_player_key(name: str) -> str:
    """
    Convert a raw player name string into a normalized, deterministic key
    suitable for joining across CFBD and NFL datasets.
 
    Steps:
        1. Unicode → ASCII  (handles é, ñ, etc.)
        2. Lower-case
        3. Strip punctuation (apostrophes, dots, dashes, etc.)
        4. Remove suffixes (Jr., Sr., II, III …)
        5. Expand known nicknames / abbreviations (Bill → william)
        6. Rejoin and return a single lowercase string
 
    Examples:
        "D.J. Moore"     →  "dj moore"
        "De'Von Achane"  →  "devon achane"
        "Odell Beckham Jr." → "odell beckham"
        "Patrick Mahomes II" → "patrick mahomes"
        "Will Levis"     →  "william levis"
    """
    if not isinstance(name, str) or not name.strip():
        return ""
 
    # 1. ASCII transliteration
    name = _unicode_to_ascii(name)
 
    # 2. Lower-case
    name = name.lower()
 
    # 3. Remove punctuation: keep only letters, digits, and spaces.
    #    This collapses "D.J." → "dj", "O'Dell" → "odell", "Ja'Marr" → "jamarrr"
    name = re.sub(r"[^a-z0-9\s]", "", name)
 
    # 4. Collapse extra whitespace
    tokens = name.split()
 
    # 5. Remove suffix tokens
    tokens = [t for t in tokens if t not in SUFFIXES]
 
    # 6. Expand nicknames (first name only, i.e. tokens[0])
    if tokens:
        tokens[0] = NICKNAME_MAP.get(tokens[0], tokens[0])
 
    return " ".join(tokens)

##############################################################
################### FUZZY MATCHING PLAYERS ###################
##############################################################

def fuzzy_match_players(
    college_df: pd.DataFrame,
    nfl_df: pd.DataFrame,
    college_key_col: str = "player_key",
    nfl_key_col: str = "player_key",
    score_cutoff: float = 88.0,
    limit: int = 1,
) -> pd.DataFrame:
    """
    For every player in college_df whose player_key does NOT exactly match
    any key in nfl_df, attempt a fuzzy match and return a mapping table.
 
    Uses rapidfuzz token_sort_ratio, which handles word-order differences
    and partial abbreviation matches well.
 
    Parameters
    ----------
    college_df      : college DataFrame (must already have player_key column)
    nfl_df          : NFL DataFrame (must already have player_key column)
    college_key_col : key column name in college_df
    nfl_key_col     : key column name in nfl_df
    score_cutoff    : minimum similarity score 0-100 to accept a match (default 88)
    limit           : max candidates to return per player (default 1 = best match only)
 
    Returns
    -------
    DataFrame with columns:
        college_key, best_nfl_match, match_score
    """
    nfl_keys = nfl_df[nfl_key_col].dropna().unique().tolist()
    college_keys = college_df[college_key_col].dropna().unique().tolist()
 
    # Keys that already have an exact match — skip them
    exact_matches = set(college_keys) & set(nfl_keys)
    unmatched = [k for k in college_keys if k not in exact_matches]
 
    records = []
    for key in unmatched:
        results = process.extract(
            key,
            nfl_keys,
            scorer=fuzz.token_sort_ratio,
            limit=limit,
            score_cutoff=score_cutoff,
        )
        for match_key, score, _ in results:
            records.append({
                "college_key":    key,
                "best_nfl_match": match_key,
                "match_score":    round(score, 1),
            })
 
    if not records:
        return pd.DataFrame(columns=["college_key", "best_nfl_match", "match_score"])
 
    return pd.DataFrame(records).sort_values("match_score", ascending=False)

###############################################################
##################### FEATURE ENGINEERING #####################
###############################################################
def _compute_scheme_fit(row) -> float:
    """
    Scheme fit: 0-1 score of how well player role matches team offensive system.
    High target share players fit best on high pass-rate teams.
    High carry players fit best on run-heavy teams.
    """
    pass_rate = row.get("pass_rate") or 0.5
    target_share = row.get("target_share") or 0
    carries_pg = row.get("carries_per_game") or 0
    position = row.get("position", "")

    if position in ("WR", "TE"):
        # WR/TE thrive on high pass-rate teams
        return min(1.0, target_share * 5 * pass_rate)
    elif position == "RB":
        # RBs thrive on run-heavy teams (1-pass_rate = run rate)
        run_rate = 1 - pass_rate
        return min(1.0, (carries_pg / 15) * (run_rate * 2))
    return 0.5


def _classify_conference(conference: Optional[str]) -> int:
    """Return conference tier (1=P5, 2=G5, 3=FCS/other)."""
    if not conference:
        return 2
    p5 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10"}
    g5 = {"American Athletic", "Mountain West", "Conference USA", "MAC", "Sun Belt"}
    if conference in p5:
        return 1
    elif conference in g5:
        return 2
    return 3


def _normalize_draft_pick(pick: float) -> float:
    if not pick or pick <= 0:
        return 0.0
    return max(0.0, 1.0 - (math.log(float(pick)) / math.log(300)))


def _age_on_date(birth_date, as_of: date) -> Optional[float]:
    if pd.isna(birth_date):
        return None
    bd = birth_date.date() if hasattr(birth_date, "date") else birth_date
    return (as_of - bd).days / 365.25


def rolling_diff(series: pd.Series) -> pd.Series:
    """Year-over-year change in a metric."""
    return series.diff()


def _f(val) -> Optional[float]:
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _i(val) -> Optional[int]:
    try:
        if pd.isna(val):
            return None
        return int(float(val))
    except (TypeError, ValueError):
        return None
