import logging
from typing import Optional
from datetime import datetime
from numpy.ma import ids
import pandas as pd

import nfl_data_py as nfl

from data.schema.database import (
    Player, NFLSeasonStats, NFLAdvancedStats,
    NFLWeeklySnaps, InjuryRecord, get_session
)

logger = logging.getLogger(__name__)


class NFLIngestion:
    """
    Pulls and stores all NFL data. Designed to be run once to backfill history,
    then re-run at the start of each season to pull the latest year.

    Usage:
        engine = get_engine()
        init_db(engine)
        pipeline = NFLIngestionPipeline(engine)
        pipeline.run_full_backfill(start_year=2015)
    """
    # Positions we care about for fantasy
    SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

    # Injury severity mapping — used for composite injury risk score
    INJURY_SEVERITY = {
        "Knee": 0.8, "ACL": 1.0, "MCL": 0.7, "Meniscus": 0.7,
        "Hamstring": 0.6, "Quad": 0.5, "Groin": 0.5, "Hip": 0.5,
        "Ankle": 0.6, "Foot": 0.5,
        "Shoulder": 0.6, "Clavicle": 0.5, "Elbow": 0.4, "Wrist": 0.4,
        "Back": 0.7, "Ribs": 0.5,
        "Concussion": 0.8,
        "Illness": 0.2, "Rest": 0.0, "Not Injury Related": 0.0,
    }

    # Soft tissue injuries have higher recurrence risk
    SOFT_TISSUE_INJURIES = {"Hamstring", "Quad", "Groin", "Hip", "Calf", "Thigh"}

    def __init__(self, engine):
        self.engine = engine

        ids = nfl.import_ids()
        self.player_ids = ids[['gsis_id', 'pfr_id']][~ids['gsis_id'].isna()].rename(columns={'gsis_id': 'player_id', 'pfr_id': 'pfr_player_id'}).drop_duplicates(subset=['player_id'])

    def run_full_backfill(self, start_year: int = 2015, end_year: Optional[int] = None):
        """
        Master method — runs all ingestion steps in order.
        Pulls everything from start_year to present.
        Safe to re-run (upserts, not inserts).
        """
        if not end_year:
            end_year = datetime.now().year
        years = list(range(start_year, end_year + 1))

        logger.info(f"Starting NFL backfill for {years[0]}–{years[-1]}")

        self.ingest_players()
        self.ingest_seasonal_stats(years)
        self.ingest_advanced_stats(years)
        self.ingest_snap_counts(years)
        self.ingest_injuries(years)

        logger.info("NFL backfill complete.")
    
    # ------------------------------------------------------------------
    # Players
    # ------------------------------------------------------------------

    def ingest_players(self):
        """
        Pull the full NFL player roster universe from nfl_data_py.
        Filters to skill positions only.
        """
        logger.info("Ingesting NFL player roster...")

        raw: pd.DataFrame = nfl.import_players()

        # Filter to fantasy-relevant positions and reasonable status
        raw = raw[raw["position"].isin(self.SKILL_POSITIONS)].copy()

        players = []
        for _, row in raw.iterrows():
            player = Player(
                player_id=str(row.get("player_id") or row.get("gsis_id", "")),
                gsis_id=str(row.get("gsis_id", "")),
                sleeper_id=str(row.get("sleeper_id", "")) if pd.notna(row.get("sleeper_id")) else None,
                name=row.get("display_name") or f"{row.get('first_name','')} {row.get('last_name','')}".strip(),
                first_name=row.get("first_name"),
                last_name=row.get("last_name"),
                position=row.get("position"),
                nfl_team=row.get("team_abbr"),
                college_team=row.get("college_name"),
                status=row.get("status"),
                birth_date=_safe_date(row.get("birth_date")),
                height=_safe_int(row.get("height")),
                weight=_safe_int(row.get("weight")),
                years_exp=_safe_int(row.get("years_of_experience")),
                rookie_year=_safe_int(row.get("rookie_year")),
                draft_round=_safe_int(row.get("draft_round")),
                draft_pick=_safe_int(row.get("draft_number")),
                draft_team=row.get("draft_club"),
                draft_year=_safe_int(row.get("entry_year")),
                depth_chart_order=_safe_int(row.get("depth_chart_position")),
                jersey_number=_safe_int(row.get("jersey_number")),
                updated_at=datetime.utcnow(),
            )
            players.append(player)

        self._bulk_upsert(Player, players, conflict_column="player_id")
        logger.info(f"{len(players)} players ingested.")
    
    # ------------------------------------------------------------------
    # Seasonal stats
    # ------------------------------------------------------------------

    def ingest_seasonal_stats(self, years: list[int]):
        """
        Pulls seasonal aggregates: standard stats + fantasy points + efficiency.
        This is the core stats table.
        """
        logger.info(f"Ingesting seasonal stats for {years}...")

        player_stats_df = nfl.import_seasonal_data(years, s_type="REG")
        snap_df = self._load_snap_pct_for_season_stats(years)
        team_targets = self._load_team_targets(years)
        
        # Merge snap pct and team targets into seasonal
        if not snap_df.empty:
            player_stats_df = player_stats_df.merge(snap_df, on=["player_id", "season"], how="left")
        if not team_targets.empty:
            player_stats_df = player_stats_df.merge(team_targets, on=["season", "team"], how="left")

        # Compute derived metrics not in raw data
        player_stats_df["yards_per_attempt"] = player_stats_df["passing_yards"] / player_stats_df["attempts"].replace(0, float("nan"))
        player_stats_df["rushing_yards_per_attempt"] = player_stats_df["rushing_yards"] / player_stats_df["carries"].replace(0, float("nan"))
        player_stats_df["yards_per_reception"] = player_stats_df["receiving_yards"] / player_stats_df["receptions"].replace(0, float("nan"))
        player_stats_df["catch_rate"] = player_stats_df["receptions"] / player_stats_df["targets"].replace(0, float("nan"))
        player_stats_df["yards_per_target"] = player_stats_df["receiving_yards"] / player_stats_df["targets"].replace(0, float("nan"))
        player_stats_df["wopr"] = (1.5 * player_stats_df.get("target_share", 0)) + (0.7 * player_stats_df.get("air_yards_share", 0))
        player_stats_df["tgt_per_game"] = player_stats_df["targets"] / player_stats_df["games"].replace(0, float("nan"))
        
        player_stats_df["fantasy_ppg_ppr"] = player_stats_df["fantasy_points_ppr"] / player_stats_df["games"].replace(0, float("nan"))
        player_stats_df["fantasy_ppg_half"] = (player_stats_df["fantasy_points_ppr"] + 0.5 * player_stats_df.get("receptions", 0)) / player_stats_df["games"].replace(0, float("nan"))
        player_stats_df["completion_pct"] = player_stats_df["completions"] / player_stats_df["attempts"].replace(0, float("nan"))
        player_stats_df["passer_rating"] = ((player_stats_df["completions"] / player_stats_df["attempts"].replace(0, float("nan")) - 0.3) * 5 + (player_stats_df["passing_yards"] / player_stats_df["attempts"].replace(0, float("nan")) - 3) * 0.25 + (player_stats_df["passing_tds"] / player_stats_df["attempts"].replace(0, float("nan"))) * 20 + 2.375 - (player_stats_df["interceptions"] / player_stats_df["attempts"].replace(0, float("nan"))) * 25) * (100 / 6)

        player_stats_df = player_stats_df.sort_values('snap_pct', ascending=False).groupby(["player_id", "season", "season_type"]).first().reset_index()

        records = [
            NFLSeasonStats(
                player_id=str(row["player_id"]),
                season=int(row["season"]),
                season_type="REG",
                team=row.get("team"),
                games=_safe_int(row.get("games")),
                # games_started=_safe_int(row.get("games_started")),
                completions=_safe_int(row.get("completions")),
                attempts=_safe_int(row.get("attempts")),
                passing_yards=_safe_int(row.get("passing_yards")),
                passing_tds=_safe_int(row.get("passing_tds")),
                interceptions=_safe_int(row.get("interceptions")),
                passing_epa=row.get("passing_epa"),
                completion_pct=row.get("completion_pct"),
                yards_per_attempt=row.get("yards_per_attempt"),
                passer_rating=row.get("passer_rating"),
                sacks=_safe_int(row.get("sacks")),
                carries=_safe_int(row.get("carries")),
                rushing_yards=_safe_int(row.get("rushing_yards")),
                rushing_tds=_safe_int(row.get("rushing_tds")),
                rushing_epa=row.get("rushing_epa"),
                yards_per_carry=row.get("rushing_yards_per_att"),
                targets=_safe_int(row.get("targets")),
                receptions=_safe_int(row.get("receptions")),
                receiving_yards=_safe_int(row.get("receiving_yards")),
                receiving_tds=_safe_int(row.get("receiving_tds")),
                receiving_epa=row.get("receiving_epa"),
                yards_per_reception=row.get("yards_per_reception"),
                catch_rate=row.get("catch_rate"),
                yards_per_target=row.get("yards_per_target"),
                air_yards_total=_safe_int(row.get("receiving_air_yards")),
                yards_after_catch=_safe_int(row.get("receiving_yards_after_catch")),
                fantasy_points_ppr=row.get("fantasy_points_ppr"),
                fantasy_ppg_ppr=row.get("fantasy_ppg_ppr"),
                target_share=row.get("target_share"),
                air_yards_share=row.get("air_yards_share"),
                racr=row.get("racr"),
                wopr=row.get("wopr"),
                tgt_per_game=row.get("tgt_per_game"),
                snap_pct=row.get("snap_pct"),
            )
            for _, row in player_stats_df.iterrows()
        ]

        self._bulk_upsert(NFLSeasonStats, records, conflict_column=("player_id", "season", "season_type"))
        logger.info(f"{len(records)} seasonal stat rows ingested.")
    
    def _load_snap_pct_for_season_stats(self, years: list[int]) -> pd.DataFrame:
        """Aggregate snap % per player per season from weekly snap data."""
        try:
            snaps = nfl.import_snap_counts(years)
            snaps = self.player_ids.merge(snaps, on=["pfr_player_id"], how="inner")
            snap_agg = (
                snaps[snaps["position"].isin(self.SKILL_POSITIONS)]
                .groupby(["player_id", "team", "season"])
                .agg(snap_pct=("offense_pct", "mean"))
                .reset_index()
            )
            return snap_agg
        except Exception as e:
            logger.warning(f"Could not load snap data: {e}")
            return pd.DataFrame()
    
    def _load_team_targets(self, years: list[int]) -> pd.DataFrame:
        """Compute team-level target totals (needed for target_share)."""
        try:
            pbp = nfl.import_pbp_data(years, columns=[
                "passer_player_id", "receiver_player_id", "pass_attempt",
                "posteam", "season", "play_type", "game_id"
            ])
            pass_plays = pbp[pbp["pass_attempt"] == 1]
            team_targets = (
                pass_plays.groupby(["posteam", "season"])
                .size()
                .reset_index(name="team_target_total")
                .rename(columns={"posteam": "team"})
            )
            return team_targets
        except Exception as e:
            logger.warning(f"Could not load team targets: {e}")
            return pd.DataFrame()
    
    # ------------------------------------------------------------------
    # Advanced / NGS stats
    # ------------------------------------------------------------------

    def ingest_advanced_stats(self, years: list[int]):
        """
        Pull Next Gen Stats tracking data for all three stat types.
        Only available from 2016+.
        """
        ngs_years = [y for y in years if y >= 2016]
        if not ngs_years:
            return

        logger.info(f"Ingesting NGS advanced stats for {ngs_years}...")

        for stat_type in ("passing", "rushing", "receiving"):
            try:
                raw = nfl.import_ngs_data(stat_type, ngs_years).rename(columns={"player_gsis_id": "player_id"})
                records = self._parse_ngs(raw, stat_type)
                self._bulk_upsert(NFLAdvancedStats, records, conflict_column=("player_id", "season", "stat_type"))
                logger.info(f"  → {len(records)} NGS {stat_type} rows ingested.")
            except Exception as e:
                logger.warning(f"NGS {stat_type} failed: {e}")

    def _parse_ngs(self, raw: pd.DataFrame, stat_type: str) -> list[NFLAdvancedStats]:
        """
        Parse NGS dataframe into ORM objects.
        """
        # Filter to full-season rows (week = 0 in nfl_data_py NGS aggregation)
        if "week" in raw.columns:
            raw = raw[raw["week"] == 0].copy()

        records = []
        for _, row in raw.iterrows():
            pid = row.get("player_id", None)
            if not pid:
                continue

            r = NFLAdvancedStats(
                player_id=pid,
                season=int(row["season"]),
                stat_type=stat_type,
            )
            if stat_type == "passing":
                r.avg_time_to_throw = row.get("avg_time_to_throw")
                r.avg_completed_air_yards = row.get("avg_completed_air_yards")
                r.avg_intended_air_yards = row.get("avg_intended_air_yards")
                r.aggressiveness = row.get("aggressiveness")
                r.completion_pct_above_expectation = row.get("completion_percentage_above_expectation")
                r.avg_air_yards_to_sticks = row.get("avg_air_yards_to_sticks")
            elif stat_type == "rushing":
                r.efficiency = row.get("efficiency")
                r.avg_time_to_los = row.get("avg_time_to_los")
                r.expected_yards = row.get("expected_rush_yards")
                r.rush_yards_over_expected = row.get("rush_yards_over_expected")
                r.rush_yards_over_expected_per_att = row.get("rush_yards_over_expected_per_att")
                r.percent_attempts_gte_eight_defenders = row.get("percent_attempts_gte_eight_defenders")
            elif stat_type == "receiving":
                r.avg_cushion = row.get("avg_cushion")
                r.avg_separation = row.get("avg_separation")
                r.avg_intended_air_yards_rec = row.get("avg_intended_air_yards")
                r.catch_pct_above_expectation = row.get("catch_percentage_above_expectation")
                r.avg_yac_above_expectation = row.get("avg_yac_above_expectation")
            records.append(r)
        return records

    # ------------------------------------------------------------------
    # Snap counts (weekly)
    # ------------------------------------------------------------------

    def ingest_snap_counts(self, years: list[int]):
        """
        Weekly offensive snap counts. Useful for tracking role stability
        and detecting depth chart movement mid-season.
        """
        logger.info(f"Ingesting weekly snap counts for {years}...")
        try:
            snap_df = nfl.import_snap_counts(years)
            snap_df = self.player_ids.merge(snap_df, on=["pfr_player_id"], how="inner")
            snap_df = snap_df[snap_df["position"].isin(self.SKILL_POSITIONS)].copy()
        except Exception as e:
            logger.warning(f"Snap counts failed: {e}")
            return

        records = [
            NFLWeeklySnaps(
                player_id=str(row.get("player_id", "")),
                season=int(row["season"]),
                week=int(row["week"]),
                game_id=row.get("game_id"),
                team=row.get("team"),
                offense_snaps=_safe_int(row.get("offense_snaps")),
                offense_pct=row.get("offense_pct"),
                defense_snaps=_safe_int(row.get("defense_snaps")),
                defense_pct=_safe_int(row.get("defense_pct")),
                st_snaps=_safe_int(row.get("st_snaps")),
            )
            for _, row in snap_df.iterrows()
        ]

        self._bulk_upsert(NFLWeeklySnaps, records, conflict_column=None)  # no unique constraint here
        logger.info(f"  → {len(records)} snap count rows ingested.")

    # ------------------------------------------------------------------
    # Injuries
    # ------------------------------------------------------------------

    def ingest_injuries(self, years: list[int]):
        """
        Weekly injury report designations.
        Builds the injury history needed for the injury risk model.
        """
        logger.info(f"Ingesting injury records for {years}...")
        try:
            injuries_df = nfl.import_injuries(years).rename(columns={"gsis_id": "player_id"})
            injuries_df["season"] = injuries_df["season"].astype("int")
            injuries_df = self.player_ids.merge(injuries_df, on=["player_id"], how="inner")
        except Exception as e:
            logger.warning(f"Injury data failed: {e}")
            return

        records = [
            InjuryRecord(
                player_id=str(row.get("player_id", "")),
                season=int(row["season"]),
                week=int(row["week"]),
                team=row.get("team"),
                report_status=row.get("report_status"),
                practice_status=row.get("practice_status"),
                primary_injury=row.get("primary_injury"),
            )
            for _, row in injuries_df.iterrows()
            if row.get("player_id")
        ]

        self._bulk_upsert(InjuryRecord, records, conflict_column=None)
        logger.info(f"  → {len(records)} injury records ingested.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _bulk_upsert(self, model, records, conflict_column):
        """
        SQLite upsert using INSERT OR REPLACE semantics.
        For tables with unique constraints, this updates on conflict.
        For tables without, it just inserts.
        """
        if not records:
            return

        with get_session(self.engine) as session:
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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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