"""
injury_risk.py
--------------
Model 2: Injury Risk Classifier

PREDICTS: Probability that a player misses 4+ games next season.

WHY A SEPARATE INJURY MODEL:
  The NFL Forecaster (Model 1) already includes injury_risk_score as a feature,
  so why build a separate model? Two reasons:
  1. The forecaster treats injury risk as one of many inputs. This model makes it
     the output — useful for the dynasty agent to surface a standalone "this player
     is high-risk" warning even when the overall projection looks fine.
  2. It opens up the ability to weight injury features more heavily using domain
     knowledge, which a general-purpose forecaster won't do automatically.

WHY LOGISTIC REGRESSION + CALIBRATION (NOT GRADIENT BOOSTING):
─────────────────────────────────────────────────────────────────
  This is the most important model selection decision in the project.
  Here's why we deliberately chose a simpler model:

  Gradient Boosting (LightGBM/XGBoost):
    + Higher raw AUC on held-out data
    - Severe class imbalance problem: only ~15–20% of player-seasons result in
      4+ missed games. GBMs tend to be overconfident on the majority class and
      produce poorly calibrated probabilities (e.g., predicting 0.03 when the
      true rate is 0.15). The probability output is what the dynasty agent
      surfaces to the user — we need it to be accurate, not just ranked correctly.
    - With N ≈ 500–2000 total rows and only ~100–300 positive examples, a GBM
      will overfit the minority class. We'd need very heavy regularization that
      effectively collapses it back toward linear behavior anyway.
    - Verdict: better AUC on paper, worse calibrated probabilities in practice

  Neural Network:
    - Same sample size problem as above. Requires SMOTE or class weighting to work
      at all, and still produces worse calibrated outputs than a properly tuned
      logistic regression. Completely overkill here.
    - Verdict: dominated by logistic regression at this scale

  Logistic Regression (chosen):
    + Coefficients are directly interpretable: "soft tissue injury history multiplies
      odds of missing 4+ games by 2.3×" — that's explainable to a user.
    + Produces well-calibrated probabilities out of the box (unlike GBMs)
    + With Platt scaling (CalibratedClassifierCV), calibration improves further
    + class_weight='balanced' handles class imbalance cleanly
    + With ElasticNet regularization (C parameter), it handles correlated injury
      features (soft_tissue_flag and games_missed are correlated) gracefully
    + Meets the "interpretability over marginal accuracy" criterion for medical/risk
      predictions — a principle well-established in clinical ML literature
    - Verdict: the right tool for this specific problem

  WHY NOT RANDOM FOREST:
    Random Forest with class_weight='balanced' is actually a reasonable choice here
    and produces better AUC than logistic regression. We choose logistic regression
    specifically because the probability calibration is more important than AUC —
    a dynasty player needs to know "30% injury risk" vs "60% injury risk," not just
    a ranking. Forest probabilities require isotonic regression calibration and are
    still noisier than logistic on small N.

KEY DESIGN DECISION — FEATURE SELECTION FOR INJURY MODEL:
  We use a curated subset of features rather than all engineered features.
  Including performance features (fantasy_ppg, target_share) would cause the model
  to confound "good players get more snaps and thus more injury exposure" with
  actual injury risk factors. We use only injury history, age, workload (snaps),
  and position — the causal precursors to injury, not their consequences.

KEY DESIGN DECISION — PLATT SCALING (CALIBRATION):
  Even logistic regression can produce miscalibrated probabilities when the
  training set has class imbalance and the features are correlated.
  CalibratedClassifierCV with method='sigmoid' (Platt scaling) applies a
  post-hoc sigmoid transformation to the raw log-odds so that a predicted 0.3
  means the player actually misses 4+ games 30% of the time historically.
  This is essential for the dynasty agent to give trustworthy risk assessments.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    classification_report, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import DynastyModel, PredictionResult, _FEATURE_DISPLAY_NAMES

logger = logging.getLogger(__name__)

# Features used by the injury model — deliberately limited to causal precursors.
# Do NOT add performance features here (target_share, fantasy_ppg) — they would
# introduce confounding between "high usage → more exposure" and true injury risk.
INJURY_FEATURES = [
    # Injury history (strongest predictors — past injury is the best predictor of future)
    "games_missed_last_season",
    "games_missed_2yr_total",
    "soft_tissue_injury_flag",     # hamstring/groin have high recurrence rates
    "acl_history_flag",            # recovery timeline and long-term risk
    "concussion_history_count",    # accumulating risk
    "injury_risk_score",           # composite from feature engineering layer

    # Age and experience — older players take longer to recover
    "age",
    "age_vs_position_peak",        # past peak = higher injury risk
    "years_experience",

    # Workload — more snaps/carries = more injury exposure
    "snap_pct",
    "carries_per_game",
    "targets_per_game",
]

# Binary target: did the player miss 4+ games next season?
# 4 games ≈ ~25% of season — meaningful dynasty impact threshold
INJURY_THRESHOLD_GAMES = 4

# Logistic regression hyperparameters
LR_PARAMS = {
    "C": 0.5,                      # Inverse regularization strength — lower = more regularized
    "penalty": "elasticnet",       # ElasticNet combines L1 (sparsity) + L2 (correlated features)
    "solver": "saga",              # Required for ElasticNet
    "l1_ratio": 0.5,               # 50/50 L1-L2 mix
    "class_weight": "balanced",    # Upweights minority (injured) class automatically
    "max_iter": 1000,
    "random_state": 42,
}


class InjuryRiskModel(DynastyModel):
    """
    Logistic regression classifier predicting probability of missing 4+ games.
    Returns calibrated probabilities with SHAP-based factor explanations.
    """

    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        self._calibrated: Optional[CalibratedClassifierCV] = None
        self._shap_explainer = None
        self._feature_cols: list[str] = []

    @property
    def model_name(self) -> str:
        return "injury_risk_classifier"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Train and calibrate the injury risk classifier.
        Uses StratifiedKFold (not TimeSeriesSplit) because:
          - Injury risk is relatively stationary over time (not trending like performance)
          - We need stratified folds to ensure the minority class is represented
            in each fold — StratifiedKFold guarantees this, TimeSeriesSplit doesn't.
        """
        # Build binary target: 1 if missed >= 4 games next season
        df = df.copy()
        df["injured_next_season"] = (
            df["games_missed_next_season"].fillna(0) >= INJURY_THRESHOLD_GAMES
        ).astype(int) if "games_missed_next_season" in df.columns else self._build_target(df)

        # Filter to rows where we have the target
        df = df[df["injured_next_season"].notna()].copy()

        self._feature_cols = [c for c in INJURY_FEATURES if c in df.columns]
        X = self._fill_missing(df[self._feature_cols])
        y = df["injured_next_season"].astype(int)

        n_positive = y.sum()
        n_total = len(y)
        logger.info(
            f"[InjuryRisk] Training on {n_total} samples, "
            f"{n_positive} injured ({n_positive/n_total:.1%})"
        )

        # ---- Stratified k-fold CV for AUC and Brier score ----
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs, cv_briers = [], []

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**LR_PARAMS)),
        ])

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if y_val.sum() == 0:  # skip folds with no positive examples
                continue

            fold_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(**LR_PARAMS)),
            ])
            fold_pipeline.fit(X_tr, y_tr)
            probs = fold_pipeline.predict_proba(X_val)[:, 1]

            cv_aucs.append(roc_auc_score(y_val, probs))
            cv_briers.append(brier_score_loss(y_val, probs))

        cv_auc = float(np.mean(cv_aucs)) if cv_aucs else float("nan")
        cv_brier = float(np.mean(cv_briers)) if cv_briers else float("nan")
        logger.info(f"[InjuryRisk] CV AUC: {cv_auc:.3f}, CV Brier: {cv_brier:.3f}")

        # ---- Train final model with Platt calibration ----
        # CalibratedClassifierCV with cv=5 fits 5 models and averages calibrated probs.
        # method='sigmoid' = Platt scaling — well-suited for logistic regression output.
        base_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**LR_PARAMS)),
        ])
        self._calibrated = CalibratedClassifierCV(
            base_pipeline, method="sigmoid", cv=5
        )
        self._calibrated.fit(X, y)

        # ---- Build SHAP explainer on the uncalibrated linear model ----
        # SHAP LinearExplainer works directly on logistic regression coefficients
        # (faster and more exact than TreeExplainer for linear models)
        plain_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**LR_PARAMS)),
        ])
        plain_pipeline.fit(X, y)
        lr_model = plain_pipeline.named_steps["lr"]
        scaler = plain_pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X)
        self._shap_explainer = shap.LinearExplainer(lr_model, X_scaled)
        self._plain_pipeline = plain_pipeline  # kept for SHAP inference

        # Calibration quality check
        final_probs = self._calibrated.predict_proba(X)[:, 1]
        train_auc = roc_auc_score(y, final_probs)
        train_brier = brier_score_loss(y, final_probs)
        train_ap = average_precision_score(y, final_probs)

        return {
            "cv_auc": cv_auc,
            "cv_brier": cv_brier,
            "train_auc": train_auc,
            "train_brier": train_brier,
            "train_avg_precision": train_ap,
            "n_train": n_total,
            "positive_rate": float(n_positive / n_total),
        }

    def _build_target(self, df: pd.DataFrame) -> pd.Series:
        """
        If games_missed_next_season isn't pre-computed, build it from
        games_missed_last_season shifted forward by one season.
        """
        df = df.sort_values(["player_id", "season"])
        return (
            df.groupby("player_id")["games_missed_last_season"]
            .shift(-1)
            .fillna(0)
            .ge(INJURY_THRESHOLD_GAMES)
            .astype(int)
        )

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
        Returns a PredictionResult where predicted_ppg encodes injury probability (0-1),
        and extras contains the full risk breakdown.

        Note: We reuse PredictionResult for consistency with the agent layer,
        but set predicted_ppg = injury_probability (0-1) for this model.
        """
        if self._calibrated is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self._fill_missing(
            pd.DataFrame([{c: features.get(c, np.nan) for c in self._feature_cols}])
        )
        injury_prob = float(self._calibrated.predict_proba(X)[0, 1])
        risk_level = _prob_to_risk_level(injury_prob)

        positive, negative = self.explain(features)

        return PredictionResult(
            player_id=player_id,
            player_name=player_name,
            position=str(features.get("position", "UNK")),
            season=season,
            model_name=self.model_name,
            predicted_ppg=injury_prob,        # overloaded: probability here
            predicted_ppg_low=max(0.0, injury_prob - 0.1),
            predicted_ppg_high=min(1.0, injury_prob + 0.1),
            prediction_confidence=self.confidence_from_sample_size(
                int(features.get("years_experience", 0) or 0),
                float(features.get("games_played", 0) or 0),
            ),
            top_positive_factors=positive,    # factors INCREASING risk
            top_negative_factors=negative,    # factors DECREASING risk
            extras={
                "injury_probability": round(injury_prob, 3),
                "risk_level": risk_level,
                "threshold_games": INJURY_THRESHOLD_GAMES,
                "injury_history_summary": self._summarize_history(features),
            }
        )

    def explain(self, features: pd.Series) -> tuple[list[str], list[str]]:
        """
        SHAP LinearExplainer gives exact attributions for logistic regression.
        Positive SHAP = increases injury probability.
        Negative SHAP = decreases injury probability.
        """
        if self._shap_explainer is None or self._plain_pipeline is None:
            return [], []

        X = self._fill_missing(
            pd.DataFrame([{c: features.get(c, np.nan) for c in self._feature_cols}])
        )
        X_scaled = self._plain_pipeline.named_steps["scaler"].transform(X)
        shap_vals = self._shap_explainer.shap_values(X_scaled)[0]
        feat_vals = X.values[0]

        return self.shap_factors_to_strings(shap_vals, self._feature_cols, feat_vals, n=4)

    def get_risk_breakdown(self, features: pd.Series) -> dict:
        """
        Returns a structured breakdown of risk factors for display in the Streamlit UI.
        More detailed than the prose output from explain().
        """
        X = self._fill_missing(
            pd.DataFrame([{c: features.get(c, np.nan) for c in self._feature_cols}])
        )
        X_scaled = self._plain_pipeline.named_steps["scaler"].transform(X)
        shap_vals = self._shap_explainer.shap_values(X_scaled)[0]

        breakdown = {}
        for col, shap_val, feat_val in zip(self._feature_cols, shap_vals, X.values[0]):
            display_name = _FEATURE_DISPLAY_NAMES.get(col, col)
            breakdown[display_name] = {
                "value": float(feat_val),
                "risk_contribution": float(shap_val),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
            }
        return dict(sorted(breakdown.items(), key=lambda x: abs(x[1]["risk_contribution"]), reverse=True))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN with 0 for injury features (missing = no history = low risk)."""
        return X[self._feature_cols].fillna(0.0)

    def _summarize_history(self, features: pd.Series) -> str:
        """Build a one-sentence injury history summary."""
        parts = []
        missed = int(features.get("games_missed_2yr_total", 0) or 0)
        if missed > 0:
            parts.append(f"missed {missed} games over last 2 seasons")
        if features.get("soft_tissue_injury_flag"):
            parts.append("soft tissue injury history")
        if features.get("acl_history_flag"):
            parts.append("ACL history")
        concussions = int(features.get("concussion_history_count", 0) or 0)
        if concussions > 0:
            parts.append(f"{concussions} concussion(s) on record")
        return "; ".join(parts) if parts else "no significant injury history"


def _prob_to_risk_level(prob: float) -> str:
    """Convert probability to human-readable risk tier."""
    if prob >= 0.45:
        return "HIGH"
    elif prob >= 0.25:
        return "MEDIUM"
    elif prob >= 0.10:
        return "LOW"
    return "VERY_LOW"
