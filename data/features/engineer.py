import logging
from datetime import date
from typing import Optional
import math

import pandas as pd
import numpy as np
from sqlalchemy import text

from data.schema.database import EngineeredFeatures, get_session


logger = logging.getLogger(__name__)

class FeatureEngineer:
    # Peak age by position — after peak, age_vs_position_peak goes positive (declining)
    POSITION_PEAK_AGE = {"QB": 34, "WR": 26, "RB": 24, "TE": 27}

    # Injury status severity weights for injury designation count scoring
    DESIGNATION_WEIGHT = {
        "Out": 1.0,
        "Doubtful": 0.8,
        "Questionable": 0.3,
        "Limited": 0.1,
        "Full": 0.0,
    }

    # Soft tissue and high-recurrence injuries
    SOFT_TISSUE = {"Hamstring", "Quad", "Quadricep", "Groin", "Hip Flexor", "Calf"}
    HIGH_SEVERITY = {"Knee", "ACL", "MCL", "Back", "Concussion"}

    # Conference competition adjustment multiplier for college stats
    CONFERENCE_MULTIPLIER = {1: 1.0, 2: 0.85, 3: 0.70}   # tier 1=P5, 2=G5, 3=FCS

    def __init__(self, engine):
        self.engine = engine
    
    def run(self, seasons: list[int]):
        """
        Build engineered features for all players with data in any of the given seasons.
        For each player+season, we look back up to 3 years to build trend features.
        """
        logger.info(f"Building features for seasons: {seasons}")

        # Load all raw data into memory as DataFrames for efficient vectorized ops
        stats_df = self._load_nfl_stats(seasons)
        adv_df = self._load_advanced_stats(seasons)
        injury_df = self._load_injury_data(seasons)
        snap_df = self._load_snap_data(seasons)
        players_df = self._load_players()
        team_df = self._load_team_context(seasons)
        college_df = self._load_college_stats()
        combine_df = self._load_combine_data()

        # Merge everything onto the stats base
        df = self._merge_all(stats_df, adv_df, injury_df, snap_df, players_df, team_df)

        # Compute feature groups
        df = self._add_performance_trajectory(df)
        df = self._add_role_features(df)
        df = self._add_efficiency_features(df, adv_df)
        df = self._add_injury_features(df, injury_df)
        df = self._add_age_features(df, players_df)
        df = self._add_team_context_features(df, team_df)
        df = self._add_target_variable(df, stats_df)

        # Add college/prospect features (NULL for veterans)
        df = self._add_college_features(df, college_df, combine_df, players_df)

        # Write to feature store
        self._write_features(df)
        logger.info(f"Feature engineering complete. {len(df)} rows written.")
    
    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_nfl_stats(self, seasons: list[int]) -> pd.DataFrame:
        query = f"""
            SELECT s.*, p.position, p.nfl_team as current_team,
                   p.birth_date, p.years_exp, p.draft_round, p.draft_pick,
                   p.rookie_year, p.sleeper_id, p.name as player_name
            FROM nfl_season_stats s
            JOIN players p ON s.player_id = p.player_id
            WHERE s.season IN ({','.join(map(str, seasons))})
              AND s.season_type = 'REG'
              AND p.position IN ('QB', 'RB', 'WR', 'TE')
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    def _load_advanced_stats(self, seasons: list[int]) -> pd.DataFrame:
        query = f"""
            SELECT * FROM nfl_advanced_stats
            WHERE season IN ({','.join(map(str, seasons))})
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    def _load_injury_data(self, seasons: list[int]) -> pd.DataFrame:
        query = f"""
            SELECT * FROM injury_records
            WHERE season IN ({','.join(map(str, seasons + [s-1 for s in seasons]))})
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    def _load_snap_data(self, seasons: list[int]) -> pd.DataFrame:
        query = f"""
            SELECT player_id, season, week, offense_pct
            FROM nfl_weekly_snaps
            WHERE season IN ({','.join(map(str, seasons))})
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    def _load_players(self) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text("SELECT * FROM players"), conn)

    def _load_team_context(self, seasons: list[int]) -> pd.DataFrame:
        query = f"""
            SELECT * FROM nfl_teams
            WHERE season IN ({','.join(map(str, seasons))})
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    def _load_college_stats(self) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text("SELECT * FROM college_season_stats"), conn)

    def _load_combine_data(self) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text("SELECT * FROM combine_measurements"), conn)

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def _merge_all(self, stats_df, adv_df, injury_df, snap_df, players_df, team_df) -> pd.DataFrame:
        """Merge all data sources onto the seasonal stats base."""
        df = stats_df.copy()

        # Snap pct trend (std dev of weekly snaps — measures role consistency)
        snap_consistency = (
            snap_df.groupby(["player_id", "season"])
            .agg(snap_pct_std=("offense_pct", "std"), snap_pct_mean=("offense_pct", "mean"))
            .reset_index()
        )
        df = df.merge(snap_consistency, on=["player_id", "season"], how="left")

        # Team context
        if not team_df.empty:
            df = df.merge(
                team_df.rename(columns={"team_abbr": "team"}),
                on=["team", "season"],
                how="left"
            )

        return df

    # ------------------------------------------------------------------
    # Feature group 1: Performance trajectory
    # ------------------------------------------------------------------

    def _add_performance_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-year fantasy PPG trends.
        Rolling averages and slope of recent performance.
        """
        df = df.sort_values(["player_id", "season"])

        # Rolling averages (shift to avoid leakage — only use past seasons)
        grp = df.groupby("player_id")["fantasy_ppg_ppr"]
        df["fantasy_ppg_last_season"] = grp.shift(1)
        df["fantasy_ppg_2yr_avg"] = grp.shift(1).rolling(2, min_periods=1).mean().values
        df["fantasy_ppg_3yr_avg"] = grp.shift(1).rolling(3, min_periods=1).mean().values

        # Trend: slope of last 3 seasons (positive = improving, negative = declining)
        def rolling_slope(series: pd.Series, n: int = 3) -> pd.Series:
            """OLS slope over last n seasons."""
            slopes = []
            vals = series.tolist()
            for i in range(len(vals)):
                window = [v for v in vals[max(0, i-n):i] if pd.notna(v)]
                if len(window) >= 2:
                    x = np.arange(len(window), dtype=float)
                    y = np.array(window, dtype=float)
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            return pd.Series(slopes, index=series.index)

        df["fantasy_ppg_trend"] = (
            df.groupby("player_id")["fantasy_ppg_ppr"]
            .transform(lambda s: rolling_slope(s))
        )

        # Career games (context for sample size)
        df["career_games"] = df.groupby("player_id")["games"].cumsum().shift(1)

        return df

    # ------------------------------------------------------------------
    # Feature group 2: Role / workload
    # ------------------------------------------------------------------

    def _add_role_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target share, air yards share, snap %, carries per game.
        Role stability is crucial — a high-usage player with declining team
        pass rate is less valuable than the raw numbers suggest.
        """
        df["targets_per_game"] = df["targets"] / df["games"].replace(0, np.nan)
        df["carries_per_game"] = df["carries"] / df["games"].replace(0, np.nan)

        # Snap % trend (positive = growing role)
        df = df.sort_values(["player_id", "season"])
        df["snap_pct_trend"] = (
            df.groupby("player_id")["snap_pct"]
            .transform(lambda s: rolling_diff(s))
        )

        # Role security score: composite of snap_pct + target_share + carries_per_game (position-adjusted)
        df["role_security_score"] = np.where(
            df["position"].isin(["WR", "TE"]),
            0.4 * df["snap_pct"].fillna(0) + 0.6 * df["target_share"].fillna(0) * 10,
            0.5 * df["snap_pct"].fillna(0) + 0.5 * (df["carries_per_game"].fillna(0) / 20),
        )

        return df

    # ------------------------------------------------------------------
    # Feature group 3: Efficiency above expectation
    # ------------------------------------------------------------------

    def _add_efficiency_features(self, df: pd.DataFrame, adv_df: pd.DataFrame) -> pd.DataFrame:
        """
        EPA, CPOE, RYOE — these measure how much better/worse a player performs
        than average given the same opportunities. Critical for identifying
        true talent vs volume.
        """
        # EPA per play (passing)
        df["passing_epa_per_att"] = df["passing_epa"] / df["attempts"].replace(0, np.nan)

        # EPA per rush
        df["rushing_epa_per_carry"] = df["rushing_epa"] / df["carries"].replace(0, np.nan)

        # EPA per target (receiving)
        df["receiving_epa_per_tgt"] = df["receiving_epa"] / df["targets"].replace(0, np.nan)

        # RACR: Receiver Air Conversion Ratio = receiving_yards / air_yards
        # > 1.0 means player gains more than their air yards (good YAC)
        df["racr"] = df["receiving_yards"] / df["air_yards_total"].replace(0, np.nan)

        # YAC per reception
        df["yac_per_rec"] = df["yards_after_catch"] / df["receptions"].replace(0, np.nan)

        # Merge in NGS metrics (CPOE, RYOE, separation)
        passing_ngs = adv_df[adv_df["stat_type"] == "passing"][
            ["player_id", "season", "completion_pct_above_expectation",
             "avg_time_to_throw", "aggressiveness", "avg_intended_air_yards"]
        ].rename(columns={"completion_pct_above_expectation": "cpoe"})

        rushing_ngs = adv_df[adv_df["stat_type"] == "rushing"][
            ["player_id", "season", "rush_yards_over_expected_per_att",
             "efficiency", "avg_time_to_los"]
        ].rename(columns={"rush_yards_over_expected_per_att": "ryoe_per_att"})

        receiving_ngs = adv_df[adv_df["stat_type"] == "receiving"][
            ["player_id", "season", "avg_separation", "avg_cushion",
             "catch_pct_above_expectation", "avg_yac_above_expectation"]
        ]

        df = df.merge(passing_ngs, on=["player_id", "season"], how="left")
        df = df.merge(rushing_ngs, on=["player_id", "season"], how="left")
        df = df.merge(receiving_ngs, on=["player_id", "season"], how="left")

        return df

    # ------------------------------------------------------------------
    # Feature group 4: Injury risk
    # ------------------------------------------------------------------

    def _add_injury_features(self, df: pd.DataFrame, injury_df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds injury risk features from weekly injury report data.
        Key insight: injury TYPE matters more than injury count.
        - Soft tissue injuries (hamstring, groin) have high recurrence
        - ACL is a career trajectory disruptor but low recurrence after full recovery
        - Concussions accumulate risk

        The composite injury_risk_score is used as a feature in the performance
        forecaster AND as a standalone prediction by the injury risk model.
        """
        if injury_df.empty:
            df["injury_risk_score"] = np.nan
            return df

        # Games missed = games with "Out" or "Doubtful" designation
        games_missed = (
            injury_df[injury_df["report_status"].isin(["Out", "Doubtful"])]
            .groupby(["player_id", "season"])
            .size()
            .reset_index(name="games_missed")
        )

        # Weighted injury designation count (Out=1.0, Questionable=0.3, etc.)
        injury_df["designation_weight"] = injury_df["report_status"].map(
            self.DESIGNATION_WEIGHT
        ).fillna(0)

        weighted_designations = (
            injury_df.groupby(["player_id", "season"])["designation_weight"]
            .sum()
            .reset_index(name="weighted_injury_score")
        )

        # Injury type flags
        soft_tissue_flag = (
            injury_df[injury_df["primary_injury"].isin(self.SOFT_TISSUE)]
            .groupby(["player_id", "season"])
            .size()
            .reset_index(name="soft_tissue_count")
        )
        soft_tissue_flag["soft_tissue_injury_flag"] = True

        acl_flag = (
            injury_df[injury_df["primary_injury"].str.contains("ACL|Knee", na=False)]
            .groupby(["player_id", "season"])
            .size()
            .reset_index(name="acl_count")
        )
        acl_flag["acl_history_flag"] = True

        concussion_count = (
            injury_df[injury_df["primary_injury"].str.contains("Concussion", na=False)]
            .groupby(["player_id", "season"])
            .size()
            .reset_index(name="concussion_history_count")
        )

        # 2-year rolling games missed (injury track record)
        games_missed_2yr = (
            games_missed.sort_values(["player_id", "season"])
            .groupby("player_id")
            .apply(lambda g: g.assign(
                games_missed_2yr=g["games_missed"].rolling(2, min_periods=1).sum().shift(1)
            ))
            .reset_index(drop=True)
        )

        # Merge all injury features into main df
        df = df.merge(games_missed.rename(columns={"games_missed": "games_missed_last_season"}),
                      on=["player_id", "season"], how="left")
        df = df.merge(games_missed_2yr[["player_id", "season", "games_missed_2yr"]].rename(
                      columns={"games_missed_2yr": "games_missed_2yr_total"}),
                      on=["player_id", "season"], how="left")
        df = df.merge(weighted_designations, on=["player_id", "season"], how="left")
        df = df.merge(soft_tissue_flag[["player_id", "season", "soft_tissue_injury_flag"]],
                      on=["player_id", "season"], how="left")
        df = df.merge(acl_flag[["player_id", "season", "acl_history_flag"]],
                      on=["player_id", "season"], how="left")
        df = df.merge(concussion_count, on=["player_id", "season"], how="left")

        # Fill NaN flags with False/0
        df["soft_tissue_injury_flag"] = df["soft_tissue_injury_flag"].fillna(False)
        df["acl_history_flag"] = df["acl_history_flag"].fillna(False)
        df["games_missed_last_season"] = df["games_missed_last_season"].fillna(0)
        df["games_missed_2yr_total"] = df["games_missed_2yr_total"].fillna(0)
        df["concussion_history_count"] = df["concussion_history_count"].fillna(0)

        # Composite injury risk score (0-1)
        # Weighted combination of: games missed rate, soft tissue history, ACL history, concussions
        games_in_season = 17
        df["injury_risk_score"] = (
            0.35 * (df["games_missed_last_season"] / games_in_season).clip(0, 1)  # recency
            + 0.25 * (df["games_missed_2yr_total"] / (games_in_season * 2)).clip(0, 1)  # track record
            + 0.20 * df["soft_tissue_injury_flag"].astype(float)   # soft tissue = high recurrence
            + 0.15 * df["acl_history_flag"].astype(float)          # ACL history = cautionary
            + 0.05 * (df["concussion_history_count"].clip(0, 3) / 3)  # concussion accumulation
        )

        return df

    # ------------------------------------------------------------------
    # Feature group 5: Age curve
    # ------------------------------------------------------------------

    def _add_age_features(self, df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Age relative to position peak is one of the most predictive features.
        A 28-year-old WR (2 years past peak) is a very different asset than
        a 23-year-old WR (3 years before peak) even with identical current stats.
        """
        players_df = players_df.copy()
        players_df["birth_date"] = pd.to_datetime(players_df["birth_date"], errors="coerce")

        if "birth_date" not in df.columns:
            df = df.merge(players_df[["player_id", "birth_date"]], on="player_id", how="left")

        df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

        # Age as of September 1 of the given season (start of NFL season)
        df["age"] = df.apply(
            lambda r: _age_on_date(r["birth_date"], date(int(r["season"]), 9, 1))
            if pd.notna(r.get("birth_date")) else np.nan,
            axis=1,
        )

        # Age at NFL entry
        df["age_at_nfl_entry"] = df.apply(
            lambda r: _age_on_date(r["birth_date"], date(int(r["rookie_year"]), 9, 1))
            if pd.notna(r.get("birth_date")) and pd.notna(r.get("rookie_year")) else np.nan,
            axis=1,
        )

        # Age vs position peak (negative = still ascending, positive = declining)
        df["age_vs_position_peak"] = df.apply(
            lambda r: (r["age"] - self.POSITION_PEAK_AGE.get(r["position"], 27))
            if pd.notna(r.get("age")) else np.nan,
            axis=1,
        )

        return df

    # ------------------------------------------------------------------
    # Feature group 6: Team context and scheme fit
    # ------------------------------------------------------------------

    def _add_team_context_features(self, df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Team context can make or break a player's fantasy value.
        A WR on a run-heavy team has a lower ceiling regardless of skill.
        New OC or team change is a high variance signal.
        """
        # New team flag (compared to previous season)
        df = df.sort_values(["player_id", "season"])
        df["prev_team"] = df.groupby("player_id")["team"].shift(1)
        df["new_team_flag"] = (df["team"] != df["prev_team"]) & df["prev_team"].notna()

        # New OC flag — requires team_df to have OC info year-over-year
        if "offensive_coordinator" in df.columns:
            df["prev_oc"] = df.groupby("player_id")["offensive_coordinator"].shift(1)
            df["new_oc_flag"] = (
                (df["offensive_coordinator"] != df["prev_oc"]) & df["prev_oc"].notna()
            )
        else:
            df["new_oc_flag"] = False

        # Scheme fit score: how well does the player's style match the team's scheme?
        # Simplified version: high-target players should go to high pass-rate teams
        if "pass_rate" in df.columns:
            df["scheme_fit_score"] = df.apply(
                lambda r: _compute_scheme_fit(r), axis=1
            )
        else:
            df["scheme_fit_score"] = np.nan

        return df

    def _add_target_variable(self, df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add next season's fantasy PPG as the target variable.
        Uses a shift within each player's time series.
        This is what the ML model tries to predict.
        """
        next_season_ppg = (
            stats_df[["player_id", "season", "fantasy_ppg_ppr"]]
            .copy()
            .rename(columns={"season": "current_season", "fantasy_ppg_ppr": "fantasy_ppg_next_season"})
        )
        next_season_ppg["season"] = next_season_ppg["current_season"] - 1  # shift back 1 year
        df = df.merge(
            next_season_ppg[["player_id", "season", "fantasy_ppg_next_season"]],
            on=["player_id", "season"],
            how="left"
        )
        return df

    # ------------------------------------------------------------------
    # Feature group 7: College / prospect features
    # ------------------------------------------------------------------

    def _add_college_features(
        self,
        df: pd.DataFrame,
        college_df: pd.DataFrame,
        combine_df: pd.DataFrame,
        players_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For incoming rookies and recent draft picks, add college production
        and athleticism features. These are NULL for veterans.

        Key metrics:
          - Dominator Rating: player's share of team receiving production (>30% = elite)
          - Breakout Age: age at first 20%+ dominator season (younger = better)
          - Competition adjustment: P5 stats weighted higher than FCS
          - Athleticism composites from combine
        """
        if college_df.empty:
            return df

        college_df = college_df.copy()

        # ---- Dominator Rating ----
        # = (player_rec_yards / team_rec_yards + player_rec_tds / team_rec_tds) / 2
        # Capped at 100% to handle edge cases
        college_df["dominator_rating"] = (
            (college_df["rec_yards"] / college_df["team_rec_yards"].replace(0, np.nan)).clip(0, 1)
            + (college_df["rec_tds"] / college_df["team_rec_tds"].replace(0, np.nan)).clip(0, 1)
        ) / 2

        # Competition adjustment
        college_df["conference_tier"] = college_df["conference"].apply(_classify_conference)
        college_df["adj_multiplier"] = college_df["conference_tier"].map(self.CONFERENCE_MULTIPLIER)
        college_df["adjusted_dominator"] = college_df["dominator_rating"] * college_df["adj_multiplier"]

        # ---- Per-game production ----
        college_df["college_yards_per_game"] = (
            college_df["rec_yards"] / college_df["games_played"].replace(0, np.nan)
        )
        college_df["college_tds_per_game"] = (
            college_df["rec_tds"] / college_df["games_played"].replace(0, np.nan)
        )

        # ---- Breakout Age ----
        # Earliest season a player hit 20%+ dominator rating
        # Requires knowing player age at that season — join with player birth dates
        players_slim = players_df[["player_id", "birth_date", "cfbd_id"]].copy()
        players_slim["birth_date"] = pd.to_datetime(players_slim["birth_date"], errors="coerce")

        college_with_player = college_df.merge(
            players_slim.rename(columns={"cfbd_id": "cfbd_player_id"}),
            on="cfbd_player_id",
            how="left"
        )
        college_with_player["age_at_season"] = college_with_player.apply(
            lambda r: _age_on_date(r["birth_date"], date(int(r["season"]), 9, 1))
            if pd.notna(r.get("birth_date")) else np.nan,
            axis=1,
        )

        breakout = (
            college_with_player[college_with_player["dominator_rating"] >= 0.20]
            .groupby("player_id")["age_at_season"]
            .min()
            .reset_index(name="breakout_age")
        )

        # Best college season (highest adjusted dominator)
        best_college_season = (
            college_df.sort_values("adjusted_dominator", ascending=False)
            .groupby("cfbd_player_id")
            .first()
            .reset_index()
            [["cfbd_player_id", "adjusted_dominator", "college_yards_per_game",
              "college_tds_per_game", "conference_tier"]]
            .rename(columns={
                "adjusted_dominator": "dominator_rating",
                "conference_tier": "college_conference_tier"
            })
        )

        # ---- Combine / athleticism ----
        combine_slim = combine_df[[
            "player_id", "sparq_score", "relative_athletic_score",
            "speed_score", "forty_yard", "vertical_jump", "bmi"
        ]].copy()

        # Merge college features into main df via player_id + rookie flag
        # rookies are identified by years_exp == 0 or 1
        df["is_rookie_or_sophomore"] = df["years_exp"].isin([0, 1])

        # Map cfbd_player_id → player_id via players table
        cfbd_to_gsis = (
            players_slim[players_slim["cfbd_id"].notna()]
            .rename(columns={"cfbd_id": "cfbd_player_id"})
            [["player_id", "cfbd_player_id"]]
            .drop_duplicates()
        )
        best_college_season = best_college_season.merge(cfbd_to_gsis, on="cfbd_player_id", how="left")
        best_college_season = best_college_season.merge(breakout, on="player_id", how="left")
        best_college_season = best_college_season.merge(combine_slim, on="player_id", how="left")

        df = df.merge(
            best_college_season[[
                "player_id", "dominator_rating", "college_yards_per_game",
                "college_tds_per_game", "college_conference_tier", "breakout_age",
                "sparq_score", "relative_athletic_score", "speed_score",
                "forty_yard", "vertical_jump", "bmi"
            ]],
            on="player_id",
            how="left"
        )

        # Normalize draft pick within round
        df["draft_pick_normalized"] = df["draft_pick"].apply(
            lambda p: _normalize_draft_pick(p) if pd.notna(p) else np.nan
        )

        return df

    # ------------------------------------------------------------------
    # Write to SQLite
    # ------------------------------------------------------------------

    def _write_features(self, df: pd.DataFrame):
        """Write engineered features DataFrame to the engineered_features table."""
        records = []
        for _, row in df.iterrows():
            r = EngineeredFeatures(
                player_id=str(row["player_id"]),
                season=int(row["season"]),
                position=row.get("position"),
                player_type="nfl",
                # Performance
                fantasy_ppg_ppr=_f(row.get("fantasy_ppg_ppr")),
                fantasy_ppg_last_season=_f(row.get("fantasy_ppg_last_season")),
                fantasy_ppg_2yr_avg=_f(row.get("fantasy_ppg_2yr_avg")),
                fantasy_ppg_3yr_avg=_f(row.get("fantasy_ppg_3yr_avg")),
                fantasy_ppg_trend=_f(row.get("fantasy_ppg_trend")),
                games_played=_f(row.get("games")),
                career_games=_i(row.get("career_games")),
                # Role
                target_share=_f(row.get("target_share")),
                air_yards_share=_f(row.get("air_yards_share")),
                wopr=_f(row.get("wopr")),
                snap_pct=_f(row.get("snap_pct")),
                snap_pct_trend=_f(row.get("snap_pct_trend")),
                carries_per_game=_f(row.get("carries_per_game")),
                targets_per_game=_f(row.get("targets_per_game")),
                # Efficiency
                yards_per_target=_f(row.get("yards_per_target")),
                yards_per_carry=_f(row.get("yards_per_carry")),
                yards_after_catch_per_rec=_f(row.get("yac_per_rec")),
                racr=_f(row.get("racr")),
                epa_per_play=_f(row.get("receiving_epa_per_tgt")),
                cpoe=_f(row.get("cpoe")),
                ryoe_per_att=_f(row.get("ryoe_per_att")),
                separation_avg=_f(row.get("avg_separation")),
                catch_pct_above_expected=_f(row.get("catch_pct_above_expectation")),
                # Injury
                games_missed_last_season=_i(row.get("games_missed_last_season")),
                games_missed_2yr_total=_i(row.get("games_missed_2yr_total")),
                injury_risk_score=_f(row.get("injury_risk_score")),
                soft_tissue_injury_flag=bool(row.get("soft_tissue_injury_flag", False)),
                acl_history_flag=bool(row.get("acl_history_flag", False)),
                concussion_history_count=_i(row.get("concussion_history_count")),
                # Age
                age=_f(row.get("age")),
                age_at_nfl_entry=_f(row.get("age_at_nfl_entry")),
                years_experience=_i(row.get("years_exp")),
                age_vs_position_peak=_f(row.get("age_vs_position_peak")),
                # Team context
                team_pass_rate=_f(row.get("pass_rate")),
                team_pass_rate_neutral=_f(row.get("pass_rate_neutral")),
                team_plays_per_game=_f(row.get("plays_per_game")),
                team_pass_attempts=_i(row.get("team_pass_attempts")),
                team_points_per_game=_f(row.get("points_per_game")),
                offensive_line_rank=_i(row.get("offensive_line_rank")),
                new_team_flag=bool(row.get("new_team_flag", False)),
                new_oc_flag=bool(row.get("new_oc_flag", False)),
                scheme_fit_score=_f(row.get("scheme_fit_score")),
                # College / prospect
                dominator_rating=_f(row.get("dominator_rating")),
                breakout_age=_f(row.get("breakout_age")),
                college_yards_per_game=_f(row.get("college_yards_per_game")),
                college_tds_per_game=_f(row.get("college_tds_per_game")),
                college_conference_tier=_i(row.get("college_conference_tier")),
                draft_round=_i(row.get("draft_round")),
                draft_pick_normalized=_f(row.get("draft_pick_normalized")),
                sparq_score=_f(row.get("sparq_score")),
                relative_athletic_score=_f(row.get("relative_athletic_score")),
                speed_score=_f(row.get("speed_score")),
                forty_yard=_f(row.get("forty_yard")),
                vertical_jump=_f(row.get("vertical_jump")),
                # Target
                fantasy_ppg_next_season=_f(row.get("fantasy_ppg_next_season")),
            )
            records.append(r)

        with get_session(self.engine) as session:
            for r in records:
                session.merge(r)
            session.commit()


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

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