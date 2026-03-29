import logging
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.base import DynastyModel, PredictionResult
from pipeline import get_feature_columns

logger = logging.getLogger(__name__)

# LightGBM hyperparameters — tuned for ~500–2000 row tabular regression
# with temporal cross-validation. Intentionally conservative (high regularization)
# to prevent overfitting on small N.
LGBM_BASE_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "n_estimators": 500,
    "learning_rate": 0.03,         # Low LR requires more trees but generalizes better
    "num_leaves": 20,              # << 2^max_depth to prevent overfitting on small N
    "min_data_in_leaf": 15,        # Each leaf needs at least 15 samples — prevents fragmentation
    "feature_fraction": 0.7,       # Use 70% of features per tree (implicit regularization)
    "bagging_fraction": 0.8,       # Row subsampling per tree
    "bagging_freq": 5,
    "reg_alpha": 0.1,              # L1 regularization
    "reg_lambda": 0.1,             # L2 regularization
    "random_state": 42,
}

QUANTILE_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "verbosity": -1,
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 15,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "random_state": 42,
}

# Minimum training samples — don't train position model with fewer rows
MIN_SAMPLES_PER_POSITION = 50

class NFLPerformanceForecaster(DynastyModel):
    """
    Predicts fantasy PPG for the upcoming season for established NFL players.
    Trains one model per position (QB, RB, WR, TE).
    Returns p10/p50/p90 quantile predictions for risk-adjusted dynasty value.
    """

    def __init__(self, position: str):
        assert position in ("QB", "RB", "WR", "TE"), f"Invalid position: {position}"
        self.position = position
        self._model_p50: Optional[lgb.LGBMRegressor] = None
        self._model_p10: Optional[lgb.LGBMRegressor] = None
        self._model_p90: Optional[lgb.LGBMRegressor] = None
        self._feature_cols: list[str] = []
        self._shap_explainer = None
        self._training_df: Optional[pd.DataFrame] = None  # kept for comp lookup

    @property
    def model_name(self) -> str:
        return f"nfl_forecaster_{self.position.lower()}"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Train on all historical player-seasons for this position.
        Uses TimeSeriesSplit to prevent temporal data leakage.
        Returns MAE and RMSE on the final time-series holdout fold.
        """
        pos_df = df[df["position"] == self.position].copy()

        if len(pos_df) < MIN_SAMPLES_PER_POSITION:
            raise ValueError(
                f"Only {len(pos_df)} rows for {self.position} — need at least "
                f"{MIN_SAMPLES_PER_POSITION}. Run the full backfill first."
            )

        feat_config = get_feature_columns(self.position)
        self._feature_cols = [
            c for c in feat_config["nfl_features"]
            if c in pos_df.columns
        ]
        target_col = feat_config["target"]

        # Sort by season for correct time-series splits
        pos_df = pos_df.sort_values("season").reset_index(drop=True)

        X = pos_df[self._feature_cols].copy()
        y = pos_df[target_col].copy()

        # ---- Time series cross-validation ----
        tscv = TimeSeriesSplit(n_splits=4, gap=0)
        cv_maes = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if len(X_tr) < 30 or len(X_val) < 10:
                continue

            fold_model = lgb.LGBMRegressor(**LGBM_BASE_PARAMS)
            fold_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            preds = fold_model.predict(X_val)
            fold_mae = mean_absolute_error(y_val, preds)
            cv_maes.append(fold_mae)
            logger.debug(f"  [{self.position}] Fold {fold+1} MAE: {fold_mae:.2f}")

        cv_mae = float(np.mean(cv_maes)) if cv_maes else float("nan")
        logger.info(f"[{self.position}] TimeSeriesCV MAE: {cv_mae:.2f} PPG ({len(tscv.split(X))} folds)")

        # ---- Train final p50 model on all data ----
        self._model_p50 = lgb.LGBMRegressor(**LGBM_BASE_PARAMS)
        self._model_p50.fit(X, y, callbacks=[lgb.log_evaluation(0)])

        # ---- Train p10 and p90 quantile models ----
        self._model_p10 = lgb.LGBMRegressor(**{**QUANTILE_PARAMS, "alpha": 0.10})
        self._model_p10.fit(X, y, callbacks=[lgb.log_evaluation(0)])

        self._model_p90 = lgb.LGBMRegressor(**{**QUANTILE_PARAMS, "alpha": 0.90})
        self._model_p90.fit(X, y, callbacks=[lgb.log_evaluation(0)])

        # ---- Build SHAP explainer on p50 model ----
        self._shap_explainer = shap.TreeExplainer(self._model_p50)

        # Store training data for historical comp lookups
        self._training_df = pos_df.copy()

        # Final metrics on full training set
        final_preds = self._model_p50.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, final_preds)))
        mae_train = float(mean_absolute_error(y, final_preds))

        return {
            "cv_mae": cv_mae,
            "train_mae": mae_train,
            "train_rmse": rmse,
            "n_train": len(pos_df),
            "n_features": len(self._feature_cols),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        player_id: str,
        season: int,
        features: pd.Series,
        player_name: str = "Unknown",
    ) -> PredictionResult:
        """
        Generate a full prediction with confidence interval and explanation.
        `features` is one row from load_features_for_ml() or the engineered_features table.
        """
        if self._model_p50 is None:
            raise RuntimeError(f"Model {self.model_name} not trained. Call train() first.")

        X = self._prepare_single_row(features)

        ppg_p50 = float(self._model_p50.predict(X)[0])
        ppg_p10 = float(self._model_p10.predict(X)[0])
        ppg_p90 = float(self._model_p90.predict(X)[0])

        # Sanity clamps — fantasy PPG can't be negative or absurdly high
        ppg_p50 = max(0.0, ppg_p50)
        ppg_p10 = max(0.0, min(ppg_p10, ppg_p50))
        ppg_p90 = max(ppg_p50, ppg_p90)

        # Explainability
        positive, negative = self.explain(features)

        # Confidence based on sample size
        n_seasons = int(features.get("years_experience", 0) or 0)
        games = float(features.get("games_played", 0) or 0)
        confidence = self.confidence_from_sample_size(n_seasons, games)

        # Historical comps
        comps = self._find_historical_comps(features, n=3)

        return PredictionResult(
            player_id=player_id,
            player_name=player_name,
            position=self.position,
            season=season,
            model_name=self.model_name,
            predicted_ppg=ppg_p50,
            predicted_ppg_low=ppg_p10,
            predicted_ppg_high=ppg_p90,
            prediction_confidence=confidence,
            top_positive_factors=positive,
            top_negative_factors=negative,
            comparable_players=comps,
            extras={
                "ppg_range_width": round(ppg_p90 - ppg_p10, 2),
                "consistency_rating": _range_to_consistency(ppg_p10, ppg_p90),
            }
        )

    def explain(self, features: pd.Series) -> tuple[list[str], list[str]]:
        """
        Compute SHAP values for a single player and return readable factor strings.
        SHAP (SHapley Additive exPlanations) gives each feature a marginal contribution
        to the prediction relative to the base rate — it's model-agnostic and
        theoretically well-grounded (from cooperative game theory).
        """
        if self._shap_explainer is None:
            return [], []

        X = self._prepare_single_row(features)
        shap_vals = self._shap_explainer.shap_values(X)[0]
        feat_vals = X[0]

        return self.shap_factors_to_strings(shap_vals, self._feature_cols, feat_vals, n=4)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances (gain-based) for analysis."""
        if self._model_p50 is None:
            raise RuntimeError("Model not trained.")
        importances = self._model_p50.feature_importances_
        return (
            pd.DataFrame({"feature": self._feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_single_row(self, features: pd.Series) -> np.ndarray:
        """Extract and order feature values from a Series into a 2D array."""
        row = {}
        for col in self._feature_cols:
            val = features.get(col, np.nan)
            row[col] = float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else np.nan
        return np.array([[row[c] for c in self._feature_cols]])

    def _find_historical_comps(self, features: pd.Series, n: int = 3) -> list[str]:
        """
        Find the N most similar historical player-seasons using cosine similarity
        in the feature space. Returns strings like "Cooper Kupp 2021 (92% match)".

        We use cosine similarity rather than Euclidean distance because features
        are on different scales and we care about pattern similarity, not magnitude.
        """
        if self._training_df is None or len(self._training_df) == 0:
            return []

        # Use a subset of stable, scale-independent features for similarity
        comp_features = [
            f for f in [
                "target_share", "snap_pct", "yards_per_target", "wopr",
                "age_vs_position_peak", "team_pass_rate", "fantasy_ppg_last_season",
                "carries_per_game", "yards_per_carry",
            ] if f in self._feature_cols
        ]
        if not comp_features:
            return []

        df = self._training_df[comp_features + ["player_name", "season"]].dropna().copy()

        # Normalize features to 0-1 range for fair comparison
        df_norm = (df[comp_features] - df[comp_features].min()) / (
            df[comp_features].max() - df[comp_features].min() + 1e-9
        )

        query_vec = np.array([
            float(features.get(f, np.nan) or 0) for f in comp_features
        ])
        query_norm = (query_vec - df[comp_features].min().values) / (
            df[comp_features].max().values - df[comp_features].min().values + 1e-9
        )

        # Cosine similarity
        norms = np.linalg.norm(df_norm.values, axis=1)
        q_norm_val = np.linalg.norm(query_norm)
        if q_norm_val == 0 or (norms == 0).all():
            return []
        sims = (df_norm.values @ query_norm) / (norms * q_norm_val + 1e-9)

        top_idx = np.argsort(sims)[::-1][:n]
        comps = []
        for idx in top_idx:
            row = df.iloc[idx]
            match_pct = int(sims[idx] * 100)
            comps.append(f"{row['player_name']} {int(row['season'])} ({match_pct}% match)")
        return comps
    
def _range_to_consistency(p10: float, p90: float) -> str:
    """Classify boom/bust risk from the width of the prediction interval."""
    width = p90 - p10
    if width < 6:
        return "CONSISTENT"
    elif width < 12:
        return "MODERATE"
    return "BOOM_BUST"


# ------------------------------------------------------------------
# Convenience: train all four position models at once
# ------------------------------------------------------------------

def train_all_forecasters(df: pd.DataFrame) -> dict[str, "NFLPerformanceForecaster"]:
    """
    Train forecasters for all four positions and return a dict keyed by position.
    Usage:
        forecasters = train_all_forecasters(load_features_for_ml())
    """
    models = {}
    for position in ("QB", "RB", "WR", "TE"):
        logger.info(f"\nTraining NFL Forecaster for {position}...")
        model = NFLPerformanceForecaster(position)
        metrics = model.train_with_tracking(
            df,
            params={**LGBM_BASE_PARAMS, "position": position}
        )
        model.save()
        models[position] = model
        logger.info(f"  {position}: CV MAE = {metrics.get('cv_mae', 'N/A'):.2f} PPG")
    return models
