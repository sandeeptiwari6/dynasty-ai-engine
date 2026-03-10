import os
import logging
import math
from typing import Optional
import time
import pandas as pd

import cfbd
import nfl_data_py as nfl

from data.schema.database import (
    CollegeSeasonStats, CombineMeasurements, Player,
    get_session
)

logger = logging.getLogger(__name__)

# Draft pick value table (roughly logarithmic) — used to normalize pick #
# Pick 1 = 1.0, Pick 32 = 0.5, Pick 100 = 0.2, Pick 250 = 0.05
def normalize_draft_pick(overall_pick: int) -> float:
    if not overall_pick or overall_pick <= 0:
        return 0.0
    return max(0.0, 1.0 - (math.log(overall_pick) / math.log(300)))

class CollegeIngestion:
    """
    Pulls college football player stats, recruiting data, and combine measurements.

    Data sources:
    - College Football Data API (collegefootballdata.com) — free tier, ~1000 req/hr
        Register for a free API key at: https://collegefootballdata.com/key
    - nfl_data_py.import_combine_data() for NFL Combine measurements
    - nfl_data_py.import_draft_picks() for draft history

    The college-to-NFL translation model needs three things:
    1. College production stats (CollegeSeasonStats)
    2. Athleticism data (CombineMeasurements)
    3. Draft context (draft_round, draft_pick in Player table)

    Pulls and stores college football stats and combine data.

    Usage:
        pipeline = CollegeIngestionPipeline(engine, api_key="YOUR_CFBD_KEY")
        pipeline.run_full_backfill(start_year=2010)
    """

    # Conference tiers for competition-adjusting college stats
    POWER_5 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10"}
    GROUP_OF_5 = {"American Athletic", "Mountain West", "Conference USA", "MAC", "Sun Belt"}

    # Positions to track for fantasy relevance
    COLLEGE_POSITIONS = {"QB", "RB", "WR", "TE"}
    STAT_CATEGORIES = ["passing", "rushing", "receiving"]

    def __init__(self, engine, api_key: Optional[str] = None):
        self.engine = engine
        api_key = api_key or os.getenv("CFBD_API_KEY")

        if not api_key:
            raise ValueError(
                "CFBD API key required. Get one free at https://collegefootballdata.com/key "
                "and set it as CFBD_API_KEY environment variable."
            )
        
        # Configure the CFBD client
        config = cfbd.Configuration()
        config.api_key["Authorization"] = api_key
        config.api_key_prefix["Authorization"] = "Bearer"
        self.cfbd_client = cfbd.ApiClient(config)
        self.stats_api = cfbd.StatsApi(self.cfbd_client)
        self.players_api = cfbd.PlayersApi(self.cfbd_client)
        self.teams_api = cfbd.TeamsApi(self.cfbd_client)
    
    def run_full_backfill(self, start_year: int = 2010, end_year: Optional[int] = None):
        """
        Master method — runs college + combine ingestion.
        Pull college data 2010+ gives us roughly 15 years of draft prospects
        """
        import datetime
        end_year = end_year or datetime.datetime.now().year
        years = list(range(start_year, end_year + 1))

        logger.info(f"Starting college backfill {years[0]}–{years[-1]}")
        self.ingest_college_season_stats(years)
        self.ingest_combine_measurements()
        self.ingest_draft_history()
        logger.info("  → College backfill complete.")
    
    # ------------------------------------------------------------------
    # College season stats
    # ------------------------------------------------------------------

    def ingest_college_season_stats(self, years: list[int]):
        """
        Pull seasonal stats for all skill position players from CFBD.
        Includes team context needed to compute dominator rating.
        """
        for year in years:
            logger.info(f"Ingesting college stats for {year}...")
            try:
                self._ingest_year(year)
                time.sleep(0.5)  # be polite to the API
            except Exception as e:
                logger.warning(f"Failed for {year}: {e}")
    
    def _ingest_year(self, year: int):
        """
        Pull all player stats and team stats for one year, compute context, store
        """
        # ---- Team stats (needed for dominator rating denominator) ----
        print("RAW TEAM STATS")
        raw_team_stats = self.stats_api.get_team_stats(year=year)
        print(raw_team_stats)
        team_context = self._build_team_context(raw_team_stats)

        # ---- Player stats ----
        # CFBD returns stats by category. We pull each one separately.
        stats = []
        for category in self.STAT_CATEGORIES:
            try:
                player_stats = self.stats_api.get_player_season_stats(
                    year=year, stat_category=category
                )
                for stat in player_stats:
                    stats.append((category, stat))
            except Exception as e:
                logger.warning(f"{category} stats for {year} failed: {e}")
            
        # Group by (player, team) and merge all categories
        player_data: dict = {}  # key: (player_id, team)
        for category, stat in player_stats:
            key = (stat.player_id, stat.team)
            if key not in player_data:
                player_data[key] = {
                    "cfbd_player_id": stat.player_id,
                    "player_name": stat.player,
                    "team": stat.team,
                    "season": year,
                    "conference": None,
                    "position": None,
                }
            self._merge_stat_row(player_data[key], category, stat)
        
        # Enrich with conference and position from player search
        self._enrich_player_metadata(player_data, year)
    
    def _build_team_context(self, raw_team_stats) -> dict:
        """
        Build team stat totals keyed by team name for dominator rating
        """
        context: dict = {}
        for team_info in raw_team_stats:
            team = team_info.team
            if team not in context:
                context[team] = {}
            stat_name = team_info.stat_name
            # Accumulate relevant team totals
            if stat_name == "passingYards":
                context[team]["team_pass_yards"] = team_info.stat_value
            elif stat_name == "passingTDs":
                context[team]["team_pass_tds"] = team_info.stat_value
            elif stat_name == "receivingYards":
                context[team]["team_rec_yards"] = team_info.stat_value
            elif stat_name == "receivingTDs":
                context[team]["team_rec_tds"] = team_info.stat_value
        return context
    
    def _merge_stat_row(self, data: dict, category: str, row) -> None:
        """
        Merge a single stat category row into the player's data dict
        """
        if category == "passing":
            data.update({
                "pass_completions": getattr(row, "completions", None),
                "pass_attempts": getattr(row, "att", None),
                "pass_yards": getattr(row, "yds", None),
                "pass_tds": getattr(row, "td", None),
                "interceptions": getattr(row, "int", None),
            })
        elif category == "rushing":
            data.update({
                "rush_carries": getattr(row, "car", None),
                "rush_yards": getattr(row, "yds", None),
                "rush_tds": getattr(row, "td", None),
            })
        elif category == "receiving":
            data.update({
                "receptions": getattr(row, "rec", None),
                "rec_yards": getattr(row, "yds", None),
                "rec_tds": getattr(row, "td", None),
            })
    
    def _enrich_player_metadata(self, player_data: dict, year: int):
        """
        Call CFBD player search to get position and conference.
        Batches by team to minimize API calls
        """
        teams = set(team for (_, team) in player_data)
        team_rosters: dict = {}
        for team in teams:
            try:
                roster = self.players_api.get_roster(team=team, year=year)
                for player in roster:
                    team_rosters[(player.id, team)] = {
                        "position": player.position,
                        "conference": getattr(player, "conference", None),
                        "games": getattr(player, "games", None),
                    }
                time.sleep(0.1)
            except Exception as e:
                logger.warning(e)
                pass

        for key, data in player_data.items():
            enrichment = team_rosters.get(key, {})
            data["position"] = enrichment.get("position") or data.get("position")
            data["conference"] = enrichment.get("conference") or data.get("conference")
            data["games_played"] = enrichment.get("games") or data.get("games_played")
    
    def _build_college_stats_record(self, data: dict, team_ctx: dict) -> Optional[CollegeSeasonStats]:
        """Build a CollegeSeasonStats ORM record, including dominator components."""
        if not data.get("cfbd_player_id"):
            return None

        rec_yards = data.get("rec_yards", 0)
        rec_tds = data.get("rec_tds", 0)

        rush_yards = data.get("rush_yards", 0)
        rush_carries = data.get("rush_carries", 1)

        return CollegeSeasonStats(
            cfbd_player_id=data["cfbd_player_id"],
            player_name=data.get("player_name"),
            season=data["season"],
            team=data.get("team"),
            position=data.get("position"),
            conference=data.get("conference"),
            games_played=data.get("games_played"),
            # Passing
            pass_completions=data.get("pass_completions"),
            pass_attempts=data.get("pass_attempts"),
            pass_yards=data.get("pass_yards"),
            pass_tds=data.get("pass_tds"),
            interceptions=data.get("interceptions"),
            completion_pct=(
                (data.get("pass_completions") or 0) / max(data.get("pass_attempts") or 1, 1)
            ),
            # Rushing
            rush_carries=data.get("rush_carries"),
            rush_yards=rush_yards,
            rush_tds=data.get("rush_tds"),
            rush_yards_per_carry=rush_yards / rush_carries if rush_carries else None,
            # Receiving
            receptions=data.get("receptions"),
            rec_yards=rec_yards,
            rec_tds=rec_tds,
            yards_per_rec=rec_yards / max(data.get("receptions") or 1, 1),
            # Team context (needed for dominator rating feature engineering)
            team_pass_yards=team_ctx.get("team_pass_yards"),
            team_pass_tds=team_ctx.get("team_pass_tds"),
            team_rec_yards=team_ctx.get("team_rec_yards"),
            team_rec_tds=team_ctx.get("team_rec_tds"),
        )
    
    # ------------------------------------------------------------------
    # Combine measurements
    # ------------------------------------------------------------------

    def ingest_combine_measurements(self):
        """
        Pull NFL Combine data from nfl_data_py.
        Calculates composite athleticism scores (SPARQ, RAS, Speed Score).
        """
        logger.info("Ingesting combine measurements...")
        try:
            raw_combine_data = nfl.import_combine_data(
                years=list(range(2000, 2026)),
                positions=list(self.COLLEGE_POSITIONS)
            )
        except Exception as e:
            logger.warning(f"Combine data failed: {e}")
            return

        records = []
        for _, row in raw_combine_data.iterrows():
            height_in = _parse_height(row.get("ht"))
            weight = _safe_float(row.get("wt"))
            forty = _safe_float(row.get("40yd"))
            vertical = _safe_float(row.get("vertical"))
            broad = _safe_float(row.get("broad_jump"))
            bench = _safe_float(row.get("bench"))
            cone = _safe_float(row.get("3cone"))
            shuttle = _safe_float(row.get("shuttle"))

            bmi = (weight / (height_in ** 2) * 703) if (height_in and weight) else None
            sparq = _calc_sparq(weight, forty, vertical, broad, shuttle)
            ras = _calc_ras(forty, vertical, broad, cone, shuttle, weight, height_in)
            speed_score = _calc_speed_score(weight, forty)

            records.append(CombineMeasurements(
                player_name=row.get("player_name"),
                draft_year=_safe_int(row.get("draft_year")),
                position=row.get("position"),
                school=row.get("school_name"),
                height=_safe_int(height_in),
                weight=_safe_int(weight),
                forty_yard=forty,
                bench_press=_safe_int(bench),
                vertical_jump=vertical,
                broad_jump=_safe_int(broad),
                three_cone=cone,
                shuttle=shuttle,
                bmi=bmi,
                sparq_score=sparq,
                relative_athletic_score=ras,
                speed_score=speed_score,
            ))

        with get_session(self.engine) as session:
            for r in records:
                session.merge(r)
            session.commit()

        logger.info(f"  → {len(records)} combine records ingested.")

    # ------------------------------------------------------------------
    # Draft history — enriches Player table
    # ------------------------------------------------------------------

    def ingest_draft_history(self):
        """
        Pull historical NFL draft picks and enrich the Player table
        with draft round, pick, and year. Critical for prospect modeling.
        """
        logger.info("Ingesting NFL draft history...")
        try:
            raw_draft_data = nfl.import_draft_picks(list(range(1970, 2026)))
        except Exception as e:
            logger.warning(f"Draft data failed: {e}")
            return

        raw_draft_data = raw_draft_data[raw_draft_data["position"].isin(self.COLLEGE_POSITIONS)].copy()

        with get_session(self.engine) as session:
            for _, row in raw_draft_data.iterrows():
                gsis_id = str(row.get("gsis_id") or "")
                if not gsis_id:
                    continue
                player = session.get(Player, gsis_id)
                if player:
                    player.draft_round = _safe_int(row.get("round"))
                    player.draft_pick = _safe_int(row.get("pick"))
                    player.draft_year = _safe_int(row.get("year"))
                    player.draft_team = row.get("team")
                    player.college_team = row.get("college")
            session.commit()

        logger.info(f"  → Draft history applied to Player table.")

# ------------------------------------------------------------------
# Athleticism score calculations
# ------------------------------------------------------------------

def _calc_sparq(weight, forty, vertical, broad, shuttle) -> Optional[float]:
    """
    Simplified SPARQ score (Speed, Power, Agility, Reaction, Quickness).
    Original formula is proprietary; this is the public approximation.
    Higher = more athletic. Normalized roughly 0–100.
    """
    if not all([weight, forty, vertical, broad, shuttle]):
        return None
    try:
        return (
            (weight * 0.01)
            + (200 / forty)              # speed component
            + (vertical * 0.5)           # explosion
            + (broad * 0.02)             # power
            + (20 / shuttle)             # agility
        )
    except (ZeroDivisionError, TypeError):
        return None


def _calc_ras(forty, vertical, broad, cone, shuttle, weight, height) -> Optional[float]:
    """
    Relative Athletic Score (RAS) — 0 to 10 scale.
    Compares player's athleticism to all historical players at same position.
    This is a simplified version; the full version is on ras.football.
    We compute a z-score composite here and scale to 0-10.
    """
    scores = []
    if forty:
        # Lower 40 time = better, so invert
        scores.append(max(0, (5.5 - forty) / 0.5))
    if vertical:
        scores.append((vertical - 28) / 6)
    if broad:
        scores.append((broad - 100) / 15)
    if cone:
        scores.append((8.5 - cone) / 0.5)
    if shuttle:
        scores.append((4.8 - shuttle) / 0.3)
    if weight and height:
        # Position-adjusted size score
        scores.append((weight / height - 2.5) / 0.5)

    if not scores:
        return None
    raw_score = sum(scores) / len(scores)
    return max(0.0, min(10.0, raw_score * 3.5 + 5.0))  # scale to 0-10


def _calc_speed_score(weight: Optional[float], forty: Optional[float]) -> Optional[float]:
    """
    Speed Score = (weight * 200) / (forty^4)
    Bill Barnwell's metric — rewards big fast players (most predictive for RBs).
    Average is ~100; elite scores are 120+.
    """
    if not (weight and forty and forty > 0):
        return None
    try:
        return (weight * 200) / (forty ** 4)
    except ZeroDivisionError:
        return None

# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

def _parse_height(height_str) -> Optional[float]:
    """Convert '6-2' or '74' format to inches."""
    if pd.isna(height_str) or not height_str:
        return None
    try:
        if "-" in str(height_str):
            feet, inches = str(height_str).split("-")
            return float(feet) * 12 + float(inches)
        return float(height_str)
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> Optional[float]:
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        if pd.isna(val):
            return None
        return int(float(val))
    except (TypeError, ValueError):
        return None
