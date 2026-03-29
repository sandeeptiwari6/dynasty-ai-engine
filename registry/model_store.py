import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from models.base import DynastyModel, PredictionResult
from models.nfl_forecaster import NFLPerformanceForecaster, train_all_forecasters
from models.injury_risk import InjuryRiskModel
from models.college_translator import CollegeToNFLTranslator, train_all_translators
from pipeline import load_features_for_ml, get_feature_columns

logger = logging.getLogger(__name__)


class ModelStore:
    """
    Singleton-style registry for all trained models.
    Load once at app startup; call predict methods as needed.

    Usage:
        store = ModelStore()
        store.load_all()   # loads from disk (requires prior training)

        # or train from scratch:
        store.train_all()

        # inference:
        result = store.predict_nfl("player_id_123", 2025, features_series)
        result = store.predict_injury("player_id_123", 2025, features_series)
        result = store.predict_prospect("player_id_123", 2025, features_series)
    """

    def __init__(self):
        self._forecasters: dict[str, NFLPerformanceForecaster] = {}
        self._injury_model: Optional[InjuryRiskModel] = None
        self._translators: dict[str, CollegeToNFLTranslator] = {}
    
    # ------------------------------------------------------------------
    # Training (run once, then use load_all for subsequent runs)
    # ------------------------------------------------------------------
    
    def train_all(
        self,
        nfl_start_season: int = 2015,
        nfl_end_season: int = 2023,
    ) -> dict[str, dict]:
        """
        Train all models from scratch using the feature store.
        Takes ~5–10 minutes for a full backfill.
        Saves models to models/registry/*.joblib.
        """
        logger.info("=" * 60)
        logger.info("Training all dynasty-scout models")
        logger.info("=" * 60)
        all_metrics = {}

        # 1. NFL Forecasters (one per position)
        logger.info("\n[1/3] Training NFL Performance Forecasters...")
        nfl_df = load_features_for_ml(
            min_season=nfl_start_season,
            max_season=nfl_end_season,
            require_target=True,
            player_type="nfl",
        )
        self._forecasters = train_all_forecasters(nfl_df)
        all_metrics["nfl_forecasters"] = {
            pos: m.train(nfl_df)
            for pos, m in self._forecasters.items()
        }

        # 2. Injury Risk Model (position-agnostic)
        logger.info("\n[2/3] Training Injury Risk Classifier...")
        self._injury_model = InjuryRiskModel()
        injury_metrics = self._injury_model.train_with_tracking(nfl_df)
        self._injury_model.save()
        all_metrics["injury_risk"] = injury_metrics

        # 3. College-to-NFL Translators (one per position)
        logger.info("\n[3/3] Training College-to-NFL Translators...")
        college_df = load_features_for_ml(
            min_season=nfl_start_season,
            max_season=nfl_end_season,
            require_target=True,
            player_type="nfl",  # uses rookies' college features joined to early NFL data
        )
        self._translators = train_all_translators(college_df)
        all_metrics["college_translators"] = {
            pos: {"status": "trained"} for pos in self._translators
        }

        self._log_training_summary(all_metrics)
        return all_metrics
    
    def load_all(self):
        """
        Load all pre-trained models from disk.
        Call this at app startup after models have been trained once.
        """
        logger.info("Loading dynasty-scout models from disk...")
        errors = []

        for position in ("QB", "RB", "WR", "TE"):
            try:
                model_name = f"nfl_forecaster_{position.lower()}"
                self._forecasters[position] = DynastyModel.load(model_name)
                logger.info(f"  ✓ NFL Forecaster ({position})")
            except FileNotFoundError:
                errors.append(f"NFL Forecaster {position} not found")

        try:
            self._injury_model = DynastyModel.load("injury_risk_classifier")
            logger.info("  ✓ Injury Risk Classifier")
        except FileNotFoundError:
            errors.append("Injury Risk Classifier not found")

        for position in ("QB", "RB", "WR", "TE"):
            try:
                model_name = f"college_translator_{position.lower()}"
                self._translators[position] = DynastyModel.load(model_name)
                logger.info(f"  ✓ College Translator ({position})")
            except FileNotFoundError:
                errors.append(f"College Translator {position} not found")

        if errors:
            logger.warning(
                f"\nSome models not found (run train_all() first): {errors}"
            )
        else:
            logger.info("All models loaded successfully.")
    
    # ------------------------------------------------------------------
    # Inference interface (what the agent tools call)
    # ------------------------------------------------------------------
    
    def predict_nfl(
        self,
        player_id: str,
        season: int,
        features: pd.Series,
        player_name: str = "Unknown",
    ) -> Optional[PredictionResult]:
        """
        Predict next-season fantasy PPG for an established NFL player.
        Also appends the injury risk score to the result.
        """
        position = str(features.get("position", ""))
        forecaster = self._forecasters.get(position)
        if not forecaster:
            logger.warning(f"No forecaster for position {position}")
            return None

        result = forecaster.predict(player_id, season, features, player_name)

        # Append injury risk to extras (if model available)
        if self._injury_model:
            injury_result = self._injury_model.predict(player_id, season, features, player_name)
            result.extras["injury_probability"] = injury_result.extras.get("injury_probability")
            result.extras["injury_risk_level"] = injury_result.extras.get("risk_level")

        return result

    def predict_injury(
        self,
        player_id: str,
        season: int,
        features: pd.Series,
        player_name: str = "Unknown",
    ) -> Optional[PredictionResult]:
        """Standalone injury risk prediction."""
        if not self._injury_model:
            logger.warning("Injury model not loaded.")
            return None
        return self._injury_model.predict(player_id, season, features, player_name)

    def predict_prospect(
        self,
        player_id: str,
        season: int,
        features: pd.Series,
        player_name: str = "Unknown",
    ) -> Optional[PredictionResult]:
        """
        Predict year-3 NFL PPG for an incoming college player.
        Uses college stats, combine, and draft context only.
        """
        position = str(features.get("position", ""))
        translator = self._translators.get(position)
        if not translator:
            logger.warning(f"No college translator for position {position}")
            return None
        return translator.predict(player_id, season, features, player_name)

    def predict_player(
        self,
        player_id: str,
        season: int,
        features: pd.Series,
        player_name: str = "Unknown",
    ) -> Optional[PredictionResult]:
        """
        Smart router: uses college translator for rookies (years_exp <= 1),
        NFL forecaster for veterans. This is what the agent supervisor calls
        when it doesn't know ahead of time whether a player is a prospect.
        """
        years_exp = int(features.get("years_experience", 99) or 99)
        if years_exp <= 1 and features.get("dominator_rating") is not None:
            return self.predict_prospect(player_id, season, features, player_name)
        return self.predict_nfl(player_id, season, features, player_name)

    def get_player_features(self, player_id: str, season: int) -> Optional[pd.Series]:
        """
        Convenience: load a single player's features from the SQLite feature store.
        Used by agent tools that receive a player_id + season as inputs.
        """
        from data.schema.database import get_engine
        from sqlalchemy import text

        engine = get_engine()
        query = """
            SELECT ef.*, p.name as player_name
            FROM engineered_features ef
            JOIN players p ON ef.player_id = p.player_id
            WHERE ef.player_id = :pid AND ef.season = :season
            LIMIT 1
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"pid": player_id, "season": season})

        if df.empty:
            return None
        return df.iloc[0]
    
    def get_player_id_by_name(self, name: str) -> Optional[str]:
        """
        Look up a player_id by name (fuzzy match).
        The agent receives player names from users, not IDs.
        """
        from data.schema.database import get_engine
        from sqlalchemy import text

        engine = get_engine()
        # Try exact match first, then LIKE fallback
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT player_id FROM players WHERE LOWER(name) = LOWER(:name) LIMIT 1"),
                {"name": name}
            ).fetchone()
            if result:
                return result[0]

            # Fuzzy: match on last name
            last_name = name.split()[-1] if name else ""
            result = conn.execute(
                text("SELECT player_id, name FROM players WHERE LOWER(name) LIKE LOWER(:pat) LIMIT 5"),
                {"pat": f"%{last_name}%"}
            ).fetchall()
            if result:
                # Return best match (first result)
                logger.info(f"Fuzzy match for '{name}': {[r[1] for r in result]}")
                return result[0][0]
        return None
    
    # ------------------------------------------------------------------
    # Evaluation utilities
    # ------------------------------------------------------------------

    def run_evaluation(self, holdout_season: int = 2024) -> pd.DataFrame:
        """
        Evaluate all models on a holdout season not seen during training.
        Returns a DataFrame with player-level predictions vs actuals.
        Useful for the notebook and for demonstrating model quality.
        """
        holdout_df = load_features_for_ml(
            min_season=holdout_season,
            max_season=holdout_season,
            require_target=False,
            player_type="nfl",
        )

        rows = []
        for _, player_row in holdout_df.iterrows():
            player_id = str(player_row["player_id"])
            player_name = str(player_row.get("player_name", "Unknown"))
            features = player_row

            result = self.predict_nfl(player_id, holdout_season, features, player_name)
            if result is None:
                continue

            rows.append({
                "player_name": player_name,
                "position": player_row.get("position"),
                "season": holdout_season,
                "predicted_ppg": result.predicted_ppg,
                "predicted_low": result.predicted_ppg_low,
                "predicted_high": result.predicted_ppg_high,
                "actual_ppg": player_row.get("fantasy_ppg_ppr"),     # current season
                "injury_risk": result.extras.get("injury_probability"),
                "confidence": result.prediction_confidence,
                "top_positive": result.top_positive_factors[0] if result.top_positive_factors else "",
                "top_negative": result.top_negative_factors[0] if result.top_negative_factors else "",
            })

        eval_df = pd.DataFrame(rows)
        if not eval_df.empty and eval_df["actual_ppg"].notna().any():
            from sklearn.metrics import mean_absolute_error
            valid = eval_df.dropna(subset=["actual_ppg"])
            mae = mean_absolute_error(valid["actual_ppg"], valid["predicted_ppg"])
            logger.info(f"Holdout MAE ({holdout_season}): {mae:.2f} PPG across {len(valid)} players")

        return eval_df

    def _log_training_summary(self, all_metrics: dict):
        logger.info("\n" + "=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        for model_group, metrics in all_metrics.items():
            if isinstance(metrics, dict):
                for key, val in metrics.items():
                    if isinstance(val, (int, float)):
                        logger.info(f"  {model_group}.{key}: {val:.3f}")