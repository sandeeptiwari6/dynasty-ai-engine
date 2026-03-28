from pathlib import Path
import logging

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean,
    Date, DateTime, Text, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "dynasty_scout.db"


def get_engine(db_path: Path = DB_PATH, echo: bool = False):
    """
    Returns a SQLAlchemy engine. Uses StaticPool so the same connection
    is reused in single-threaded contexts (fine for local use).
    """
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=echo,
    )

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Core entity tables
# ---------------------------------------------------------------------------

class Player(Base):
    """
    Master player registry. One row per player (NFL or college).
    All stat tables foreign-key into this.
    """
    __tablename__ = "players"

    player_id       = Column(String, primary_key=True)   # e.g. "00-0036355" (GSIS) or sleeper id
    sleeper_id      = Column(String, index=True)
    gsis_id         = Column(String, index=True)          # NFL play-by-play system id
    cfbd_id         = Column(Integer)                     # College Football Data API id
    name            = Column(String, nullable=False, index=True)
    first_name      = Column(String)
    last_name       = Column(String)
    position        = Column(String, index=True)          # QB, RB, WR, TE
    nfl_team        = Column(String)
    college_team    = Column(String)
    status          = Column(String)                      # Active, Injured Reserve, etc.
    birth_date      = Column(Date)
    height          = Column(Integer)                     # inches
    weight          = Column(Integer)                     # lbs
    years_exp       = Column(Integer)
    rookie_year     = Column(Integer)
    draft_round     = Column(Integer)
    draft_pick      = Column(Integer)
    draft_team      = Column(String)
    draft_year      = Column(Integer)
    is_rookie       = Column(Boolean, default=False)
    depth_chart_order = Column(Integer)                   # 1 = starter
    jersey_number   = Column(Integer)
    updated_at      = Column(DateTime)

    # relationships
    nfl_stats       = relationship("NFLSeasonStats", back_populates="player")
    college_stats   = relationship("CollegeSeasonStats", back_populates="player")
    injuries        = relationship("InjuryRecord", back_populates="player")
    advanced_stats  = relationship("NFLAdvancedStats", back_populates="player")
    combine_stats   = relationship("CombineMeasurements", back_populates="player")
    engineered_features = relationship("EngineeredFeatures", back_populates="player")

    def __repr__(self):
        return f"<Player {self.name} ({self.position}, {self.nfl_team})>"


class NFLTeam(Base):
    """
    Team context table — offensive scheme, coordinator, pace metrics.
    Used for team-fit feature engineering.
    """
    __tablename__ = "nfl_teams"

    team_abbr       = Column(String, primary_key=True)   # e.g. "KC", "SF"
    season          = Column(Integer, primary_key=True)
    full_name       = Column(String)
    head_coach      = Column(String)
    offensive_coordinator = Column(String)
    defensive_coordinator = Column(String)
    offensive_scheme = Column(String)                    # "Air Raid", "West Coast", "Run Heavy", etc.
    plays_per_game  = Column(Float)                      # pace proxy
    pass_rate       = Column(Float)                      # % of plays that are passes
    pass_rate_neutral = Column(Float)                    # pass rate in neutral game script
    team_pass_yards = Column(Integer)
    team_rush_yards = Column(Integer)
    team_total_tds  = Column(Integer)
    team_pass_attempts = Column(Integer)
    team_targets    = Column(Integer)
    points_per_game = Column(Float)
    offensive_line_rank = Column(Integer)               # 1-32


# ---------------------------------------------------------------------------
# NFL statistics tables
# ---------------------------------------------------------------------------

class NFLSeasonStats(Base):
    """
    Seasonal aggregate stats per player per season.
    Sourced from nfl_data_py.import_seasonal_data().
    """
    __tablename__ = "nfl_season_stats"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(String, ForeignKey("players.player_id"), nullable=False)
    season          = Column(Integer, nullable=False)
    season_type     = Column(String, default="REG")     # REG, POST
    team            = Column(String)
    games           = Column(Integer)
    games_started   = Column(Integer)

    # Passing
    completions     = Column(Integer)
    attempts        = Column(Integer)
    passing_yards   = Column(Integer)
    passing_tds     = Column(Integer)
    interceptions   = Column(Integer)
    passing_epa     = Column(Float)
    completion_pct  = Column(Float)
    yards_per_attempt = Column(Float)
    passer_rating   = Column(Float)
    sacks           = Column(Integer)
    sack_yards      = Column(Integer)

    # Rushing
    carries         = Column(Integer)
    rushing_yards   = Column(Integer)
    rushing_tds     = Column(Integer)
    rushing_fumbles = Column(Integer)
    rushing_epa     = Column(Float)
    yards_per_carry = Column(Float)

    # Receiving
    targets         = Column(Integer)
    receptions      = Column(Integer)
    receiving_yards = Column(Integer)
    receiving_tds   = Column(Integer)
    receiving_fumbles = Column(Integer)
    receiving_epa   = Column(Float)
    yards_per_reception = Column(Float)
    catch_rate      = Column(Float)
    yards_per_target = Column(Float)
    air_yards_total = Column(Integer)
    yards_after_catch = Column(Integer)

    # Fantasy
    fantasy_points_ppr  = Column(Float)
    fantasy_points_half = Column(Float)
    fantasy_points_std  = Column(Float)
    fantasy_ppg_ppr     = Column(Float)
    fantasy_ppg_half    = Column(Float)

    # Workload / role
    snap_pct        = Column(Float)                     # % of team offensive snaps
    target_share    = Column(Float)                     # targets / team targets
    air_yards_share = Column(Float)                     # player air yards / team air yards
    racr            = Column(Float)                     # Receiver Air Conversion Ratio
    wopr            = Column(Float)                     # Weighted Opportunity Rating = 1.5*target_share + 0.7*air_yards_share
    tgt_per_game    = Column(Float)

    player = relationship("Player", back_populates="nfl_stats")

    __table_args__ = (
        UniqueConstraint("player_id", "season", "season_type", name="uix_player_season"),
        Index("ix_player_season", "player_id", "season"),
    )


class NFLAdvancedStats(Base):
    """
    Next Gen Stats (NGS) — tracking data not available pre-2016.
    Sourced from nfl_data_py.import_ngs_data().
    """
    __tablename__ = "nfl_advanced_stats"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(String, ForeignKey("players.player_id"))
    season          = Column(Integer)
    stat_type       = Column(String)                    # "passing", "rushing", "receiving"

    # Passing NGS
    avg_time_to_throw     = Column(Float)
    avg_completed_air_yards = Column(Float)
    avg_intended_air_yards  = Column(Float)
    avg_air_yards_differential = Column(Float)
    aggressiveness        = Column(Float)               # % of passes into tight coverage
    max_completed_air_yards = Column(Float)
    completion_pct_above_expectation = Column(Float)    # CPOE — huge metric
    avg_air_yards_to_sticks  = Column(Float)

    # Rushing NGS
    efficiency            = Column(Float)
    percent_attempts_gte_eight_defenders = Column(Float)
    avg_time_to_los       = Column(Float)               # time to line of scrimmage
    expected_yards        = Column(Float)
    rush_yards_over_expected = Column(Float)            # RYOE
    rush_yards_over_expected_per_att = Column(Float)

    # Receiving NGS
    avg_cushion           = Column(Float)               # yards of separation at snap
    avg_separation        = Column(Float)               # separation at catch point
    avg_intended_air_yards_rec = Column(Float)
    catch_pct_above_expectation = Column(Float)         # CPOE equivalent for receivers
    avg_yac_above_expectation = Column(Float)

    player = relationship("Player", back_populates="advanced_stats")

    __table_args__ = (
        UniqueConstraint("player_id", "season", "stat_type", name="uix_adv_player_season"),
    )


class NFLWeeklySnaps(Base):
    """
    Weekly snap counts and routes run.
    Sourced from nfl_data_py.import_snap_counts().
    Critical for tracking role changes mid-season and depth chart trends.
    """
    __tablename__ = "nfl_weekly_snaps"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(String, ForeignKey("players.player_id"), index=True)
    season          = Column(Integer)
    week            = Column(Integer)
    game_id         = Column(String)
    team            = Column(String)
    offense_snaps   = Column(Integer)
    offense_pct     = Column(Float)
    defense_snaps   = Column(Integer)
    st_snaps        = Column(Integer)

    __table_args__ = (
        Index("ix_snaps_player_season", "player_id", "season"),
    )


class InjuryRecord(Base):
    """
    Per-week injury designations from nfl_data_py.import_injuries().
    Used to build injury history features and missed game counts.
    """
    __tablename__ = "injury_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(String, ForeignKey("players.player_id"), index=True)
    season          = Column(Integer)
    week            = Column(Integer)
    team            = Column(String)
    report_status   = Column(String)                    # "Out", "Doubtful", "Questionable", "Limited", "Full"
    practice_status = Column(String)
    primary_injury  = Column(String)                    # "Knee", "Hamstring", "Concussion", etc.

    player = relationship("Player", back_populates="injuries")

    __table_args__ = (
        Index("ix_injury_player_season", "player_id", "season"),
    )


# ---------------------------------------------------------------------------
# College statistics tables
# ---------------------------------------------------------------------------

class CollegeSeasonStats(Base):
    """
    College seasonal stats from College Football Data API (collegefootballdata.com).
    The college→NFL translation model trains on this.
    """
    __tablename__ = "college_season_stats"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(String, ForeignKey("players.player_id"))
    cfbd_player_id  = Column(Integer, index=True)
    season          = Column(Integer)
    team            = Column(String)
    conference      = Column(String)
    player_name     = Column(String)                    # denormalized for easy querying
    position        = Column(String)

    # Passing
    pass_completions = Column(Integer)
    pass_attempts   = Column(Integer)
    pass_yards      = Column(Integer)
    pass_tds        = Column(Integer)
    interceptions   = Column(Integer)
    completion_pct  = Column(Float)

    # Rushing
    rush_carries    = Column(Integer)
    rush_yards      = Column(Integer)
    rush_tds        = Column(Integer)
    rush_yards_per_carry = Column(Float)

    # Receiving
    receptions      = Column(Integer)
    rec_yards       = Column(Integer)
    rec_tds         = Column(Integer)
    yards_per_rec   = Column(Float)

    # Calculated at ingestion time
    games_played    = Column(Integer)
    team_pass_yards = Column(Integer)                   # for dominator rating
    team_pass_tds   = Column(Integer)
    team_rec_yards  = Column(Integer)
    team_rec_tds    = Column(Integer)

    # Dominator ratings (player's share of team production)
    dominatory_yards = Column(Float)
    dominator_tds    = Column(Float)
    dominator_score  = Column(Float)

    player = relationship("Player", back_populates="college_stats")

    __table_args__ = (
        UniqueConstraint("cfbd_player_id", "season", name="uix_college_player_season"),
    )


class CombineMeasurements(Base):
    """
    NFL Combine and Pro Day measurements.
    Athleticism is highly predictive for college-to-NFL translation.
    """
    __tablename__ = "combine_measurements"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(String, ForeignKey("players.player_id"))
    player_name     = Column(String)
    draft_year      = Column(Integer)
    position        = Column(String)
    school          = Column(String)

    # Measurements
    height          = Column(Integer)                   # inches
    weight          = Column(Integer)                   # lbs
    forty_yard      = Column(Float)                     # 40-yard dash time
    bench_press     = Column(Integer)                   # reps at 225 lbs
    vertical_jump   = Column(Float)                     # inches
    broad_jump      = Column(Integer)                   # inches
    three_cone      = Column(Float)                     # 3-cone drill time
    shuttle         = Column(Float)                     # short shuttle time
    bmi             = Column(Float)                     # calculated: weight / height^2 * 703

    # Calculated composite scores (filled during feature engineering)
    sparq_score     = Column(Float)                     # Speed, Power, Agility, Reaction, Quickness
    relative_athletic_score = Column(Float)             # RAS: 0-10 composite vs position peers
    speed_score     = Column(Float)                     # (weight * 200) / (40_time^4) — RB metric
    catch_radius    = Column(Float)                     # height + wingspan proxy

    player = relationship("Player", back_populates="combine_stats")

    __table_args__ = (
        UniqueConstraint("player_id", name="uix_combine_player"),
    )


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------

class EngineeredFeatures(Base):
    """
    Final feature store — one row per player per season with all engineered features.
    This is what the ML models train on. Populated by features/engineer.py.
    """
    __tablename__ = "engineered_features"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(String, ForeignKey("players.player_id"), index=True)
    season          = Column(Integer)
    position        = Column(String)
    player_type     = Column(String)                    # "nfl" or "college"

    # ---- Performance features ----
    fantasy_ppg_ppr         = Column(Float)
    fantasy_ppg_last_season = Column(Float)
    fantasy_ppg_2yr_avg     = Column(Float)
    fantasy_ppg_3yr_avg     = Column(Float)
    fantasy_ppg_trend       = Column(Float)             # slope of last 3yr regression
    games_played            = Column(Float)
    career_games            = Column(Integer)

    # ---- Workload / role features ----
    target_share            = Column(Float)
    air_yards_share         = Column(Float)
    wopr                    = Column(Float)
    snap_pct                = Column(Float)
    snap_pct_trend          = Column(Float)             # increasing/decreasing role
    carries_per_game        = Column(Float)
    targets_per_game        = Column(Float)

    # ---- Efficiency features ----
    yards_per_target        = Column(Float)
    yards_per_carry         = Column(Float)
    yards_after_catch_per_rec = Column(Float)
    racr                    = Column(Float)
    epa_per_play            = Column(Float)
    cpoe                    = Column(Float)             # completion pct over expected (QB)
    ryoe_per_att            = Column(Float)             # rush yards over expected (RB)
    separation_avg          = Column(Float)             # NGS cushion/separation (WR/TE)
    catch_pct_above_expected = Column(Float)            # NGS receiving CPOE

    # ---- Injury features ----
    games_missed_last_season = Column(Integer)
    games_missed_2yr_total  = Column(Integer)
    injury_risk_score       = Column(Float)             # composite: 0-1
    soft_tissue_injury_flag = Column(Boolean)           # hamstring, quad, calf (higher recurrence)
    acl_history_flag        = Column(Boolean)
    concussion_history_count = Column(Integer)
    injury_designation_count = Column(Integer)          # # of weeks on injury report

    # ---- Age curve features ----
    age                     = Column(Float)
    age_at_nfl_entry        = Column(Float)
    years_experience        = Column(Integer)
    age_vs_position_peak    = Column(Float)             # negative=before peak, positive=past peak
    # Peak ages by position: QB=34, WR=26, RB=24, TE=27

    # ---- Team context features ----
    team_pass_rate          = Column(Float)
    team_pass_rate_neutral  = Column(Float)
    team_plays_per_game     = Column(Float)
    team_pass_attempts      = Column(Integer)
    team_points_per_game    = Column(Float)
    offensive_line_rank     = Column(Integer)
    new_team_flag           = Column(Boolean)           # changed teams via FA/trade
    new_oc_flag             = Column(Boolean)           # new offensive coordinator
    scheme_fit_score        = Column(Float)             # engineered: player style vs scheme

    # ---- College/prospect features (NULL for veterans) ----
    dominator_rating        = Column(Float)             # % of team rec yards + TDs
    breakout_age            = Column(Float)             # age at first 20%+ dominator season
    college_yards_per_game  = Column(Float)
    college_tds_per_game    = Column(Float)
    college_conference_tier = Column(Integer)           # 1=Power 5, 2=G5, 3=FCS
    draft_round             = Column(Integer)
    draft_pick_normalized   = Column(Float)             # 0-1 scale within round
    # Combine / athleticism
    sparq_score             = Column(Float)
    relative_athletic_score = Column(Float)
    speed_score             = Column(Float)             # weight-adjusted speed (RB)
    height_weight_bmi       = Column(Float)
    forty_yard              = Column(Float)
    vertical_jump           = Column(Float)

    # ---- Target variable ----
    fantasy_ppg_next_season = Column(Float)             # what we're predicting

    player = relationship("Player", back_populates="engineered_features")

    __table_args__ = (
        UniqueConstraint("player_id", "season", name="uix_features_player_season"),
        Index("ix_features_position_season", "position", "season"),
    )


# ---------------------------------------------------------------------------
# Sleeper-specific tables
# ---------------------------------------------------------------------------

class SleeperLeagueSnapshot(Base):
    """
    Dynasty league ADP and player values from Sleeper.
    Useful for the dynasty advisor agent.
    """
    __tablename__ = "sleeper_league_snapshots"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date   = Column(Date)
    sleeper_player_id = Column(String, index=True)
    player_name     = Column(String)
    position        = Column(String)
    dynasty_adp     = Column(Float)                     # average draft position
    redraft_adp     = Column(Float)
    trending_adds   = Column(Integer)                   # adds in last 24h
    trending_drops  = Column(Integer)
    percent_owned   = Column(Float)
    percent_started = Column(Float)


def init_db(engine) -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    logger.info(f"Database initialized at {engine.url}")


def get_session(engine) -> Session:
    return Session(engine)
