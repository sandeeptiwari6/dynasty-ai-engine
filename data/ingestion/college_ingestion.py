import os
import logging
import math
from typing import Optional
import time
import pandas as pd

import cfbd
import nfl_data_py as nfl
from sqlalchemy import text

from data.schema.database import (
    CollegeSeasonStats, CombineMeasurements, Player, CollegeGamesPlayed,
    get_session
)

from utils.constants import NORMALIZED_PLAYER_NAMES, MOST_FREQUENT_TEAMS

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

    PLAYER_STATS = {
        'passing': {
            'ATT': 'pass_attempts',
            'COMPLETIONS': 'pass_completions',
            'INT': 'interceptions',
            'PCT': 'completion_pct',
            'TD': 'pass_tds',
            'YDS': 'pass_yds',
            'YPA': 'yards_per_attempt'
            },
        'rushing': {
            'CAR': 'rush_attempts',
            'LONG': 'longest_rush_attempt',
            'TD': 'rush_tds',
            'YDS': 'rush_yds',
            'YPC': 'yards_per_carry'
        },
        'receiving': {
            'LONG': 'longest_reception',
            'REC': 'receptions',
            'TD': 'receiving_tds',
            'YDS': 'receiving_yds',
            'YPR': 'yards_per_reception',
        }
    }

    def __init__(self, engine, api_key: Optional[str] = None):
        self.engine = engine
        if not api_key:
            api_key = os.getenv("CFBD_API_KEY")

            if not api_key:
                raise ValueError(
                    "CFBD API key required. Get one free at https://collegefootballdata.com/key "
                    "and set it as CFBD_API_KEY environment variable."
                )
        
        print(f"Using CFBD API key: {api_key[:4]}****{api_key[-4:]}")
        
        # Configure the CFBD client
        config = cfbd.Configuration(access_token=api_key)
        config.api_key["Authorization"] = api_key
        config.api_key_prefix["Authorization"] = "Bearer"
        self.cfbd_client = cfbd.ApiClient(config)
        self.stats_api = cfbd.StatsApi(self.cfbd_client)
        self.players_api = cfbd.PlayersApi(self.cfbd_client)
        self.teams_api = cfbd.TeamsApi(self.cfbd_client)
        self.games_api = cfbd.GamesApi(self.cfbd_client)

        ids = nfl.import_ids()
        self.player_ids = (
            ids[['gsis_id', 'pfr_id']][~ids['gsis_id'].isna()]
            .rename(columns={'gsis_id': 'player_id', 'pfr_id': 'pfr_player_id'})
            # .drop_duplicates(subset=['player_id'])
            .drop_duplicates(subset=["pfr_player_id"])
        )
        self.player_ids = self.player_ids[~self.player_ids["pfr_player_id"].isna()]

    def run_full_backfill(
            self, 
            start_year: int = 2010, 
            end_year: Optional[int] = None, 
            use_cached_data: bool = True, 
            overwrite: bool = False
        ):
        """
        Master method — runs college + combine ingestion.
        Pull college data 2010+ gives us roughly 15 years of draft prospects
        """
        import datetime
        end_year = end_year or datetime.datetime.now().year
        years = list(range(start_year, end_year + 1))

        logger.info(f"Starting college backfill {years[0]}–{years[-1]}")

        self.games_played_dict = self._load_games_played_data(cached=use_cached_data, start_year=start_year)

        if overwrite:
            CollegeSeasonStats.__table__.drop(self.engine, checkfirst=True)
            CollegeSeasonStats.metadata.create_all(self.engine)
            CombineMeasurements.__table__.drop(self.engine, checkfirst=True)
            CombineMeasurements.metadata.create_all(self.engine)

        self.ingest_college_season_stats(years)
        self.ingest_combine_measurements()
        self.ingest_draft_history(start_year, end_year)
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
    
    # Coerce team stat wrapper types (e.g. TeamStatStatValue) to numeric safely
    def _unwrap_stat(self, val):
        # If it's the cfbd model wrapper, get actual_instance; otherwise use value directly
        try:
            inner = getattr(val, "actual_instance", val)
            return float(inner) if inner is not None else 0.0
        except Exception:
            try:
                return float(val)
            except Exception:
                return 0.0
    
    def calc_player_games_played_by_team(self, team_name, season):
        logger.info(f"    → Calculating games played for {team_name} in {season}...")
        game_player_stats = self.games_api.get_game_player_stats(
            year=season,
            team=team_name
        )
        player_games_appeared = {}
        
        for game in game_player_stats:
            try:
                teams = game.to_dict()['teams']
                team_stats = [team for team in teams if team['team'] == team_name][0]
                categories = team_stats['categories']
                players_seen = set()
                for category in categories:
                    cat_types = category['types']
                    for cat_type in cat_types:
                        athletes = cat_type['athletes']
                        for athlete in athletes:
                            players_seen.add(athlete['id'])
        
                for player_seen in players_seen:
                    player_games_appeared[player_seen] = player_games_appeared.get(player_seen, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to process game stats for {team_name} in {season}: {e}")
                continue
        games_played_data = [
            {
                'player_id': player_id, 
                'season': season,
                'games_played': games_played
            } for player_id, games_played in player_games_appeared.items()
        ]

        return pd.DataFrame(games_played_data)
    
    def _load_games_played_data(self, cached, start_year: int = 2010) -> pd.DataFrame:
        """
        Load or compute games played data for all players by team and season.
        CFBD doesn't provide a clean "games played" stat, so we derive it from game logs.
        Caches to avoid repeated API calls.
        """

        if not cached:
            end_year = pd.Timestamp.now().year

            with get_session(self.engine) as session:
                for team in MOST_FREQUENT_TEAMS:
                    for year in range(start_year, end_year + 1):
                        games_played_df = self.calc_player_games_played_by_team(team, year)

                        for idx, row in games_played_df.iterrows():
                            try:
                                record = CollegeGamesPlayed(
                                    cfbd_player_id=row['player_id'],
                                    season=int(row['season']),
                                    games_played=int(row['games_played'])
                                )
                                session.merge(record)
                                session.commit()
                            except Exception as e:
                                session.rollback()
                                logger.warning(f"Failed to upsert games played for player {row['player_id']} in {year}: {e}")
                        time.sleep(0.5)  # throttle to avoid hitting API limits
            logger.info("Games played data calculated and cached.")

        query = f"""
            SELECT *
            FROM college_games_played
            WHERE season >= {start_year}
        """
        with self.engine.connect() as conn:
            games_played_df = pd.read_sql(text(query), conn)

            games_played_dict = {}
            for _, row in games_played_df.iterrows():
                player_id = str(row["cfbd_player_id"])
                if player_id not in games_played_dict:
                    games_played_dict[player_id] = {}
                games_played_dict[player_id][int(row["season"])] = int(row["games_played"])

        return games_played_dict
                
    def _ingest_year(self, year: int):
        """
        Pull all player stats and team stats for one year, compute context, store
        """
        # ---- Team stats (needed for dominator rating denominator) ----
        raw_team_stats = self.stats_api.get_team_stats(year=year)
        team_context = self._build_team_context(raw_team_stats)

        # ---- Player stats ----
        # CFBD returns stats by category. We pull each one separately.
        stats_dict = {}
        for category in self.STAT_CATEGORIES:
            try:
                player_stats = self.stats_api.get_player_season_stats(
                    year=year, category=category
                )
                stats_dict[category] = player_stats
            except Exception as e:
                logger.warning(f"{category} stats for {year} failed: {e}")
            
        # Group by (player, team) and merge all categories
        player_data: dict = {}  # key: (player_id, team)
        for category, player_stats_lst in stats_dict.items():
            for player_stat in player_stats_lst:
                team, player_id = player_stat.team, player_stat.player_id
                key = (player_id, team)
                if key not in player_data:
                    player_data[key] = {
                        "cfbd_player_id": player_id,
                        "player_name": player_stat.player,
                        "team": team,
                        "season": year,
                        "conference": player_stat.conference,
                        "position": player_stat.position,
                        "games_played": self.games_played_dict.get(player_id, {}).get(int(year), 0)  # get games played from precomputed dict
                    }
                self._merge_stat_row(player_data[key], category, player_stat)
        
        # Enrich with conference and position from player search
        # self._enrich_player_metadata(player_data, year)

        # Build ORM records and bulk upsert
        records_created = 0
        with get_session(self.engine) as session:
            for key, data in player_data.items():
                team = data.get("team")
                team_ctx = team_context.get(team, {})
                record = self._build_college_stats_record(data, team_ctx)
                if record:
                    session.merge(record)
                    records_created += 1
            session.commit()
        logger.info(f"  → {records_created} college season records ingested for {year}")
    
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
            if stat_name == "netPassingYards":
                context[team]["team_pass_yards"] = self._unwrap_stat(team_info.stat_value)
                context[team]["team_rec_yards"] = self._unwrap_stat(team_info.stat_value)
            elif stat_name == "passingTDs":
                context[team]["team_pass_tds"] = self._unwrap_stat(team_info.stat_value)
                context[team]["team_rec_tds"] = self._unwrap_stat(team_info.stat_value)
        return context
    
    def _merge_stat_row(self, data: dict, category: str, row) -> None:
        """
        Merge a single stat category row into the player's data dict
        """
        stat_types = self.PLAYER_STATS[category]
        stat_type = row.stat_type
        stat_val = row.stat

        data.update(
            {stat_types[stat_type]: float(stat_val) if '.' in stat_val else int(stat_val)}
        )
    
    def _enrich_player_metadata(self, player_data: dict, year: int):
        """
        Call CFBD player search to get position and conference.
        Batches by team to minimize API calls
        """
        teams = set(team for (_, team) in player_data)
        team_rosters: dict = {}
        for team in teams:
            try:
                roster = self.teams_api.get_roster(team=team, year=year)
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
        
        player_name = data.get("player_name")
        return CollegeSeasonStats(
            cfbd_player_id=data["cfbd_player_id"],
            player_name=NORMALIZED_PLAYER_NAMES.get(player_name) or player_name,
            season=data.get("season"),
            team=data.get("team"),
            position=data.get("position"),
            conference=data.get("conference"),
            games_played=data.get("games_played"),
            # Passing
            pass_completions=data.get("pass_completions"),
            pass_attempts=data.get("pass_attempts"),
            pass_yards=data.get("pass_yds", 0),
            pass_tds=data.get("pass_tds", 0),
            interceptions=data.get("interceptions", 0),
            completion_pct=data.get("completion_pct"),
            # Rushing
            rush_carries=data.get("rush_attempts"),
            rush_yards=data.get("rush_yds", 0),
            rush_tds=data.get("rush_tds", 0),
            rush_yards_per_carry=data.get("yards_per_carry"),
            # Receiving
            receptions=data.get("receptions"),
            rec_yards=data.get("receiving_yds", 0),
            rec_tds=data.get("receiving_tds", 0),
            yards_per_rec=data.get("yards_per_reception"),
            # Team context (needed for dominator rating feature engineering)
            team_pass_yards=team_ctx.get("team_pass_yards"),
            team_pass_tds=team_ctx.get("team_pass_tds"),
            team_rec_yards=team_ctx.get("team_rec_yards"),
            team_rec_tds=team_ctx.get("team_rec_tds")
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
            ).rename(columns={'pfr_id': 'pfr_player_id'}).drop_duplicates(subset=['pfr_player_id', 'cfb_id'])
            raw_combine_data = raw_combine_data[~raw_combine_data["draft_year"].isna()]
            raw_combine_data = self.player_ids.merge(raw_combine_data, on=['pfr_player_id'], how='inner')
        except Exception as e:
            logger.warning(f"Combine data failed: {e}")
            return

        records = []
        for _, row in raw_combine_data.iterrows():
            height_in = _parse_height(row.get("ht"))
            weight = _safe_float(row.get("wt"))
            forty = _safe_float(row.get("forty"))
            vertical = _safe_float(row.get("vertical"))
            broad = _safe_float(row.get("broad_jump"))
            bench = _safe_float(row.get("bench"))
            cone = _safe_float(row.get("cone"))
            shuttle = _safe_float(row.get("shuttle"))

            bmi = (weight / (height_in ** 2) * 703) if (height_in and weight) else None
            sparq = _calc_sparq(weight, forty, vertical, broad, shuttle)
            ras = _calc_ras(forty, vertical, broad, cone, shuttle, weight, height_in)
            speed_score = _calc_speed_score(weight, forty)

            records.append(CombineMeasurements(
                player_id=row.get("player_id"),
                player_name=row.get("player_name"),
                draft_year=_safe_int(row.get("draft_year")),
                position=row.get("pos"),
                school=row.get("school"),
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

    def ingest_draft_history(self, start_year: int, end_year: int):
        """
        Pull historical NFL draft picks and enrich the Player table
        with draft round, pick, and year. Critical for prospect modeling.
        """
        logger.info("Ingesting NFL draft history...")
        try: 
            raw_draft_data = nfl.import_draft_picks(list(range(2000, max(end_year + 1, 2001)))).rename(columns={'gsis_id': 'player_id'}).drop_duplicates(subset=['player_id', 'season'])
        except Exception as e:
            logger.warning(f"Draft data failed: {e}")
            return

        raw_draft_data = raw_draft_data[raw_draft_data["position"].isin(self.COLLEGE_POSITIONS)].copy()
        raw_draft_data = self.player_ids.merge(raw_draft_data, on=['player_id', "pfr_player_id"], how='inner')

        with get_session(self.engine) as session:
            for _, row in raw_draft_data.iterrows():
                player_id = str(row.get("player_id") or "")
                if not player_id:
                    continue
                player = session.get(Player, player_id)
                if player:
                    player.draft_round = _safe_int(row.get("round"))
                    player.draft_pick = _safe_int(row.get("pick"))
                    player.draft_year = _safe_int(row.get("season"))
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
        logger.warning(f"SPARQ calculation failed for weight={weight}, forty={forty}, vertical={vertical}, broad={broad}, shuttle={shuttle}")
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
        logger.warning(f"Speed Score calculation failed for weight={weight}, forty={forty}")
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
