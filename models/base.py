import logging
from pathlib import Path
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pandas as pd
import joblib
import mlflow

from utils.constants import _FEATURE_DISPLAY_NAMES

logger = logging.getLogger(__name__)

# Where trained model artifacts are saved locally
MODEL_DIR = Path(__file__).parent / "registry"
MODEL_DIR.mkdir(exist_ok=True)

# MLflow experiment name — all runs grouped here
MLFLOW_EXPERIMENT = "dynasty-scout"


@dataclass
class PredictionResult:
    """
    Standardized output schema returned by every model's predict() method.
    The LangGraph agent tools consume this directly.

    Fields are intentionally human-readable strings alongside numeric values
    so the LLM can include them verbatim in generated reports.
    """
    player_id: str
    player_name: str
    position: str
    season: int                          # season being predicted FOR (next season)
    model_name: str

    # Core prediction
    predicted_ppg: float                 # point estimate (p50)
    predicted_ppg_low: float             # p10 — floor scenario
    predicted_ppg_high: float            # p90 — ceiling scenario
    prediction_confidence: str           # "HIGH" | "MEDIUM" | "LOW" (based on sample size)

    # Interpretability
    top_positive_factors: list[str]      # e.g. ["High WOPR (0.42)", "Age 24 — pre-peak"]
    top_negative_factors: list[str]      # e.g. ["Hamstring history", "New team"]
    comparable_players: list[str] = field(default_factory=list)  # historical comps

    # Model-specific extras (varies by model type)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "position": self.position,
            "season": self.season,
            "model_name": self.model_name,
            "predicted_ppg": round(self.predicted_ppg, 2),
            "predicted_ppg_low": round(self.predicted_ppg_low, 2),
            "predicted_ppg_high": round(self.predicted_ppg_high, 2),
            "prediction_confidence": self.prediction_confidence,
            "top_positive_factors": self.top_positive_factors,
            "top_negative_factors": self.top_negative_factors,
            "comparable_players": self.comparable_players,
            **self.extras,
        }
    
    def to_prose(self) -> str:
        """
        Returns a readable summary string for the LLM to use as context.
        Keeps the LLM grounded in the actual numbers rather than hallucinating.
        """
        pos_factors = "; ".join(self.top_positive_factors) or "none identified"
        neg_factors = "; ".join(self.top_negative_factors) or "none identified"
        comps = ", ".join(self.comparable_players) or "none found"
        return (
            f"{self.player_name} ({self.position}) — {self.model_name} projection for {self.season}:\n"
            f"  Projected PPG (PPR): {self.predicted_ppg:.1f} "
            f"[floor {self.predicted_ppg_low:.1f} / ceiling {self.predicted_ppg_high:.1f}]\n"
            f"  Confidence: {self.prediction_confidence}\n"
            f"  Positive factors: {pos_factors}\n"
            f"  Risk factors: {neg_factors}\n"
            f"  Historical comps: {comps}"
        )
    

class DynastyModel(ABC):
    """
    Abstract base class for all dynasty-scout ML models.

    Subclasses must implement: train(), predict(), explain(), model_name.
    Everything else (MLflow tracking, saving, loading) is handled here.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier used for file names and MLflow run names."""
        ...

    @abstractmethod
    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Fit the model on the provided DataFrame.
        Returns a dict of evaluation metrics (e.g. {"rmse": 3.2, "mae": 2.1}).
        """
        ...

    @abstractmethod
    def predict(self, player_id: str, season: int, features: pd.Series) -> PredictionResult:
        """
        Generate a PredictionResult for a single player-season.
        `features` is one row of the engineered_features table.
        """
        ...

    @abstractmethod
    def explain(self, features: pd.Series) -> tuple[list[str], list[str]]:
        """
        Return (positive_factors, negative_factors) as human-readable strings.
        Used to populate PredictionResult.top_positive_factors/negative_factors.
        """
        ...

    # ------------------------------------------------------------------
    # MLflow tracking
    # ------------------------------------------------------------------

    def train_with_tracking(self, df: pd.DataFrame, params: Optional[dict] = None) -> dict[str, float]:
        """
        Wraps train() with MLflow experiment tracking.
        Logs params, metrics, and saves the model artifact.
        """
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name=self.model_name):
            if params:
                mlflow.log_params(params)

            metrics = self.train(df)
            mlflow.log_metrics(metrics)

            # Save model to registry dir and log as artifact
            model_path = self.save()
            mlflow.log_artifact(str(model_path))

            logger.info(f"[{self.model_name}] Training complete. Metrics: {metrics}")
            return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Serialize model to disk using joblib."""
        path = path or MODEL_DIR / f"{self.model_name}.joblib"
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, model_name: str, path: Optional[Path] = None):
        """Load a previously saved model from disk."""
        path = path or MODEL_DIR / f"{model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model found at {path}. Run train first."
            )
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def confidence_from_sample_size(n_seasons: int, games_played: float) -> str:
        """
        Prediction confidence degrades with small sample sizes.
        Rookies with 1 season of data get LOW confidence.
        Veterans with 5+ seasons and full games get HIGH.
        """
        if n_seasons >= 4 and games_played >= 14:
            return "HIGH"
        elif n_seasons >= 2 and games_played >= 8:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def get_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """
        Extract a clean numpy array from a DataFrame, filling NaN with median.
        LightGBM handles NaN natively but sklearn estimators don't.
        """
        X = df[feature_cols].copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        return X.values

    @staticmethod
    def shap_factors_to_strings(
        shap_values: np.ndarray,
        feature_names: list[str],
        feature_values: np.ndarray,
        n: int = 4,
    ) -> tuple[list[str], list[str]]:
        """
        Convert SHAP values into human-readable factor strings.
        Returns (positive_factors, negative_factors), each list of n items.

        Example output:
          positive: ["WOPR 0.42 (+2.1 pts)", "Age 24 pre-peak (+1.3 pts)"]
          negative: ["Hamstring history (-1.8 pts)", "New team (-0.9 pts)"]
        """
        pairs = sorted(
            zip(feature_names, shap_values, feature_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        positive, negative = [], []
        for name, shap_val, feat_val in pairs:
            readable_name = _FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())
            val_str = _format_feature_value(name, feat_val)
            impact_str = f"{shap_val:+.1f} pts"
            entry = f"{readable_name} {val_str} ({impact_str})"
            if shap_val > 0:
                positive.append(entry)
            else:
                negative.append(entry)
            if len(positive) >= n and len(negative) >= n:
                break
        return positive[:n], negative[:n]

def _format_feature_value(feature_name: str, value: float) -> str:
    """Format a feature value as a readable string for factor explanations."""
    if pd.isna(value):
        return "(missing)"
    pct_features = {"target_share", "air_yards_share", "snap_pct", "team_pass_rate",
                    "team_pass_rate_neutral", "catch_pct_above_expected"}
    bool_features = {"soft_tissue_injury_flag", "acl_history_flag", "new_team_flag", "new_oc_flag"}
    if feature_name in bool_features:
        return "Yes" if value else "No"
    if feature_name in pct_features:
        return f"{value:.0%}"
    if feature_name in ("games_missed_last_season", "games_missed_2yr_total"):
        return f"{int(value)} games"
    if feature_name == "offensive_line_rank":
        return f"#{int(value)}"
    return f"{value:.2f}"
