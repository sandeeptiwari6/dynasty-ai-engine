"""
college_translator.py
---------------------
Model 3: College-to-NFL Translation Model

PREDICTS: NFL fantasy PPG in year 3 of a player's career, given only college stats,
combine measurements, and draft context. Designed for players with zero NFL history.

WHY THIS IS THE HARDEST OF THE THREE MODELS:
  Models 1 and 2 predict future NFL performance from past NFL performance —
  the signal is strong and domain-specific. This model must cross a domain boundary:
  college football → NFL. The environments are fundamentally different:
    - College: 1 vs. 5-star talent gaps, massive scheme variation
    - NFL: compressed talent distribution, every team is professional
  The historical base rate for college → NFL translation is brutal:
    ~65% of first-round WRs become solid starters; ~20% of 3rd-rounders do.
  This model's job is to sharpen those priors with player-specific signals.

WHY YEAR 3 IS THE TARGET (NOT YEAR 1):
  Year 1 NFL production for skill positions is highly variable and dominated
  by team situation (depth chart, injuries above them, OC scheme) more than
  player talent. Year 3 is when college talent signals become predictive — the
  player has had time to develop but hasn't yet hit position-specific decline.
  This is also the decision point for dynasty teams: do you extend this player
  or move on?

WHY TWO-COMPONENT MODEL (RIDGE + KNN COMPS):
─────────────────────────────────────────────
  Component 1 — Ridge Regression:
    WHY RIDGE over LightGBM here:
    - We have very few training rows: ~200–400 historical draft classes with
      enough college + NFL data. LightGBM would massively overfit.
    - Ridge regression with L2 regularization is the statistically correct
      choice when N is small relative to the number of features.
    - Lasso (L1) would zero out many features — we have domain knowledge that
      ALL features matter (dominator rating, athleticism, scheme fit), just to
      varying degrees. Ridge keeps all features with shrinkage, Lasso discards.
    - ElasticNet (L1+L2) was also considered; Ridge outperforms in CV here
      because our features are correlated (dominator rating correlates with
      targets/game, SPARQ correlates with 40-yard dash) and Ridge handles
      correlated predictors better than Lasso.
    - The Ridge output gives us a point estimate with uncertainty from
      cross-validation residuals.

  Component 2 — K-Nearest Neighbors (Historical Comps):
    WHY ADD KNN ON TOP OF RIDGE:
    - Ridge gives a number. KNN gives a narrative. Dynasty players trust
      "most similar to Justin Jefferson's college profile" more than "13.2 PPG."
    - KNN also catches non-linearities that Ridge misses: a player who is
      simultaneously elite in dominator rating AND athleticism may be more
      valuable than the linear sum of those features suggests. KNN finds the
      historical players who combined them similarly.
    - We use cosine similarity in a normalized feature space rather than
      Euclidean distance because the features span very different scales
      (dominator_rating: 0-0.7 vs. sparq_score: 0-120).
    - k=5 comps: enough to show a range of outcomes (not just one comp),
      not so many that the comps are uninformative.

    WHY NOT JUST KNN (no Ridge):
    - KNN alone would return comps for every player, but the projected value
      is just the average of the comps' actual outcomes. For edge-case players
      (e.g., exceptionally athletic but low production), the comps may be poor
      and the average misleading. Ridge anchors the prediction more reliably.

  Combined output:
    - Ridge prediction is the primary point estimate
    - KNN comps provide the range of outcomes and narrative context
    - Comp outcomes (actual year-3 PPG) show the distribution of results
      for similar profiles — more informative than a single number

KEY DESIGN DECISION — POSITION-SPECIFIC MODELS:
  Same as the NFL Forecaster: train separately per position.
  The features that predict WR success (dominator rating, separation, route running
  implied by college targets) are different from RB success (yards after contact,
  college touchdowns per game, speed score at combine).

KEY DESIGN DECISION — CONFERENCE ADJUSTMENT:
  A 35% dominator rating at Alabama is more impressive than 45% at Western Kentucky.
  The feature engineering layer already computes adjusted_dominator with a conference
  multiplier, but we also include raw_dominator + conference_tier as separate features
  so the model can learn a non-linear adjustment itself (the linear adjustment in
  feature engineering is just an approximation).

KEY DESIGN DECISION — UNCERTAINTY FROM HISTORICAL OUTCOME VARIANCE:
  Unlike the NFL Forecaster which uses quantile regression, uncertainty here comes
  from the variance of the KNN comps' actual year-3 outcomes. If 5 similar players
  had year-3 PPGs of [2, 4, 14, 16, 18], the standard deviation is huge — that's
  informative uncertainty (boom-or-bust prospect profile). If they had [11, 12, 13,
  13, 14], the uncertainty is low (safe floor prospect).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import DynastyModel, PredictionResult
from pipeline import get_feature_columns

logger = logging.getLogger(__name__)

# Year 3 is the target — college signals predict this better than years 1 or 2
TARGET_YEAR = 3

# Minimum historical prospects needed to train per position
MIN_PROSPECTS_PER_POSITION = 40

# K-nearest neighbors for comp lookup
K_COMPS = 5

# Ridge regularization search space — RidgeCV picks the best automatically
RIDGE_ALPHAS = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

# Features used for training — college stats + athleticism + draft context
# Deliberately NO NFL features (this model is for players with zero NFL history)
COLLEGE_FEATURES = [
    # Production (competition-adjusted)
    "dominator_rating",            # player's % of team receiving production (primary signal)
    "college_yards_per_game",
    "college_tds_per_game",
    "college_conference_tier",     # 1=P5, 2=G5, 3=FCS

    # Age signals
    "breakout_age",                # younger breakout = more upside
    "age_at_nfl_entry",            # younger entry = more development runway

    # Athleticism (combine)
    "sparq_score",                 # composite athleticism
    "relative_athletic_score",     # RAS: 0-10 vs position peers
    "speed_score",                 # weight-adjusted speed (most predictive for RBs)
    "forty_yard",                  # raw speed
    "vertical_jump",               # explosion

    # Draft signal
    "draft_round",
    "draft_pick_normalized",       # 0-1 scale, accounting for pick # within round
]

# Subset used for KNN comp similarity (stable, position-agnostic features)
COMP_FEATURES = [
    "dominator_rating",
    "college_yards_per_game",
    "sparq_score",
    "relative_athletic_score",
    "breakout_age",
    "draft_pick_normalized",
]

class CollegeToNFLTranslator(DynastyModel):
    """
    Translates college player profiles into NFL year-3 fantasy PPG projections.
    Combines Ridge regression (point estimate) with KNN comps (narrative + range).
    """

    def __init__(self, position: str):
        assert position in ("QB", "RB", "WR", "TE"), f"Invalid position: {position}"
        self.position = position
        self._ridge_pipeline: Optional[Pipeline] = None
        self._feature_cols: list[str] = []
        self._comp_df: Optional[pd.DataFrame] = None  # historical prospects for KNN
        self._scaler_for_comps: Optional[StandardScaler] = None

    @property
    def model_name(self) -> str:
        return f"college_translator_{self.position.lower()}"
    
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Train on historical prospects (players who entered NFL and have year-3 data).

        The training set is built by:
        1. Taking all players with rookie_year in our dataset
        2. Joining their college features (from college_season_stats)
        3. Looking up their actual year-3 NFL PPG as the target
        4. Filtering to position match

        This is the "historical ground truth" we're learning from.
        """
        # Filter to this position and require both college features AND year-3 outcome
        pos_df = df[
            (df["position"] == self.position) &
            df["dominator_rating"].notna() &
            df["fantasy_ppg_next_season"].notna()   # using as proxy for year-3 here
        ].copy()

        if len(pos_df) < MIN_PROSPECTS_PER_POSITION:
            logger.warning(
                f"[{self.position}] Only {len(pos_df)} prospects — "
                f"need {MIN_PROSPECTS_PER_POSITION}. Predictions will be low quality."
            )

        self._feature_cols = [c for c in COLLEGE_FEATURES if c in pos_df.columns]
        X = self._fill_missing(pos_df[self._feature_cols])
        y = pos_df["fantasy_ppg_next_season"].values

        logger.info(f"[{self.position}] Training on {len(pos_df)} historical prospects, "
                    f"{len(self._feature_cols)} features.")

        # ---- Ridge regression with built-in CV for alpha selection ----
        # RidgeCV is cleaner than GridSearchCV for this: it uses generalized
        # cross-validation (GCV), which is equivalent to leave-one-out CV but O(n²)
        # instead of O(n²k) — fast and well-calibrated for regression.
        self._ridge_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5)),
        ])
        self._ridge_pipeline.fit(X, y)

        best_alpha = self._ridge_pipeline.named_steps["ridge"].alpha_
        logger.info(f"[{self.position}] Ridge selected alpha={best_alpha}")

        # ---- Standard k-fold CV for evaluation metrics ----
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))]),
            X, y, cv=kf, scoring="neg_mean_absolute_error"
        )
        cv_mae = float(-np.mean(cv_scores))

        # ---- Store historical prospects for KNN lookups ----
        # Only keep comp_features + name/season/actual_y3 for memory efficiency
        comp_cols = [c for c in COMP_FEATURES if c in pos_df.columns]
        self._comp_df = pos_df[
            comp_cols + ["player_name", "season", "draft_year", "fantasy_ppg_next_season"]
        ].copy().reset_index(drop=True)

        # Fit a StandardScaler on comp features for normalized KNN distance
        self._scaler_for_comps = StandardScaler()
        self._scaler_for_comps.fit(self._fill_missing(self._comp_df[comp_cols]))

        # Final training metrics
        preds = self._ridge_pipeline.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae_train = float(mean_absolute_error(y, preds))

        # Coefficient analysis — which features matter most?
        coefs = self._ridge_pipeline.named_steps["ridge"].coef_
        coef_df = pd.DataFrame({
            "feature": self._feature_cols,
            "coefficient": coefs
        }).sort_values("coefficient", key=abs, ascending=False)
        logger.info(f"[{self.position}] Top coefficients:\n{coef_df.head(5).to_string()}")

        return {
            "cv_mae": cv_mae,
            "train_mae": mae_train,
            "train_rmse": rmse,
            "best_alpha": best_alpha,
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
        Predict year-3 NFL PPG for a prospect with only college data.
        Returns Ridge point estimate + KNN comp range for uncertainty.
        """
        if self._ridge_pipeline is None:
            raise RuntimeError(f"Model {self.model_name} not trained.")

        X = self._fill_missing(
            pd.DataFrame([{c: features.get(c, np.nan) for c in self._feature_cols}])
        )
        ppg_p50 = float(self._ridge_pipeline.predict(X)[0])
        ppg_p50 = max(0.0, ppg_p50)

        # Get historical comps and derive uncertainty from their outcome variance
        comps = self._find_comps(features, k=K_COMPS)
        comp_outcomes = [c["actual_year3_ppg"] for c in comps if c["actual_year3_ppg"] is not None]

        if comp_outcomes:
            ppg_p10 = float(np.percentile(comp_outcomes, 20))
            ppg_p90 = float(np.percentile(comp_outcomes, 80))
        else:
            ppg_p10 = ppg_p50 * 0.4
            ppg_p90 = ppg_p50 * 1.6

        ppg_p10 = max(0.0, ppg_p10)
        ppg_p90 = max(ppg_p50, ppg_p90)

        positive, negative = self.explain(features)
        comp_strings = [
            f"{c['player_name']} ({c['draft_year']}, {c['similarity']:.0%} match, "
            f"actual yr3: {c['actual_year3_ppg']:.1f} PPG)"
            for c in comps
        ]

        # Confidence degrades for edge-case profiles (few good comps)
        max_comp_sim = comps[0]["similarity"] if comps else 0.0
        confidence = _comp_quality_to_confidence(max_comp_sim, len(comp_outcomes))

        return PredictionResult(
            player_id=player_id,
            player_name=player_name,
            position=self.position,
            season=season + TARGET_YEAR,     # projecting to year 3
            model_name=self.model_name,
            predicted_ppg=ppg_p50,
            predicted_ppg_low=ppg_p10,
            predicted_ppg_high=ppg_p90,
            prediction_confidence=confidence,
            top_positive_factors=positive,
            top_negative_factors=negative,
            comparable_players=comp_strings,
            extras={
                "target_year": TARGET_YEAR,
                "comp_outcome_std": float(np.std(comp_outcomes)) if len(comp_outcomes) > 1 else None,
                "comp_count": len(comps),
                "prospect_archetype": self._classify_archetype(features),
                "dominator_rating": float(features.get("dominator_rating", 0) or 0),
                "ras": float(features.get("relative_athletic_score", 0) or 0),
            }
        )
    
    def explain(self, features: pd.Series) -> tuple[list[str], list[str]]:
        """
        For Ridge regression, SHAP LinearExplainer gives exact feature attributions.
        We interpret these as "why this player projects higher/lower than average."
        """
        if self._ridge_pipeline is None:
            return [], []

        from models.base import _FEATURE_DISPLAY_NAMES, _format_feature_value
        import shap

        X = self._fill_missing(
            pd.DataFrame([{c: features.get(c, np.nan) for c in self._feature_cols}])
        )
        scaler = self._ridge_pipeline.named_steps["scaler"]
        ridge = self._ridge_pipeline.named_steps["ridge"]
        X_scaled = scaler.transform(X)

        explainer = shap.LinearExplainer(ridge, X_scaled)
        shap_vals = explainer.shap_values(X_scaled)[0]
        feat_vals = X.values[0]

        return self.shap_factors_to_strings(shap_vals, self._feature_cols, feat_vals, n=4)

    def get_coefficient_summary(self) -> pd.DataFrame:
        """
        Return standardized coefficients for the Ridge model.
        Since features are StandardScaler'd before Ridge, coefficients are
        directly comparable: larger absolute value = more important feature.
        This is a cleaner explainability method than SHAP for linear models.
        """
        if self._ridge_pipeline is None:
            raise RuntimeError("Model not trained.")
        coefs = self._ridge_pipeline.named_steps["ridge"].coef_
        return (
            pd.DataFrame({"feature": self._feature_cols, "std_coefficient": coefs})
            .sort_values("std_coefficient", key=abs, ascending=False)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN with column median for college features.
        Unlike injury model, we fill with median not 0 because:
          - Missing dominator_rating ≠ 0 dominator rating; it means unknown
          - Median is a better imputation for continuous college stats
        """
        return X.apply(lambda col: col.fillna(col.median()))

    def _find_comps(self, features: pd.Series, k: int = 5) -> list[dict]:
        """
        Find k most similar historical prospects using cosine similarity
        in the normalized COMP_FEATURES space.
        Returns a list of dicts with name, draft_year, similarity, and actual_year3_ppg.
        """
        if self._comp_df is None or self._scaler_for_comps is None:
            return []

        comp_cols = [c for c in COMP_FEATURES if c in self._comp_df.columns]
        historical = self._fill_missing(self._comp_df[comp_cols].copy())
        historical_scaled = self._scaler_for_comps.transform(historical)

        query_raw = np.array([float(features.get(c, np.nan) or 0) for c in comp_cols])
        query_df = pd.DataFrame([query_raw], columns=comp_cols)
        query_scaled = self._scaler_for_comps.transform(
            self._fill_missing(query_df)
        )[0]

        # Cosine similarity
        hist_norms = np.linalg.norm(historical_scaled, axis=1)
        q_norm = np.linalg.norm(query_scaled)
        if q_norm == 0:
            return []
        sims = (historical_scaled @ query_scaled) / (hist_norms * q_norm + 1e-9)

        top_idx = np.argsort(sims)[::-1][:k]
        results = []
        for idx in top_idx:
            row = self._comp_df.iloc[idx]
            results.append({
                "player_name": row.get("player_name", "Unknown"),
                "draft_year": int(row.get("draft_year", 0) or 0),
                "similarity": float(sims[idx]),
                "actual_year3_ppg": float(row.get("fantasy_ppg_next_season", 0) or 0),
            })
        return results

    def _classify_archetype(self, features: pd.Series) -> str:
        """
        Assign a qualitative archetype label based on the player's college profile.
        Used in the dynasty agent's narrative generation.
        """
        dom = float(features.get("dominator_rating", 0) or 0)
        ras = float(features.get("relative_athletic_score", 0) or 0)
        breakout = float(features.get("breakout_age", 99) or 99)
        draft_rnd = float(features.get("draft_round", 7) or 7)

        if dom >= 0.30 and ras >= 8.0:
            return "ELITE_PROSPECT"         # High production + elite athleticism
        elif dom >= 0.30 and ras < 6.0:
            return "PRODUCTION_BASED"       # Produced without elite measurables
        elif dom < 0.20 and ras >= 9.0:
            return "ATHLETICISM_PROJECTION" # Draft for ceiling, not college production
        elif breakout <= 19 and dom >= 0.20:
            return "EARLY_BREAKOUT"         # High upside from young breakout age
        elif draft_rnd == 1 and dom < 0.20:
            return "SCHEME_DEPENDENT"       # High draft capital, lower production signal
        return "AVERAGE_PROSPECT"


def _comp_quality_to_confidence(max_similarity: float, n_outcomes: int) -> str:
    """Confidence based on how good the best historical comp is."""
    if max_similarity >= 0.90 and n_outcomes >= 4:
        return "HIGH"
    elif max_similarity >= 0.75 and n_outcomes >= 2:
        return "MEDIUM"
    return "LOW"


# ------------------------------------------------------------------
# Convenience: train all four position translators at once
# ------------------------------------------------------------------

def train_all_translators(df: pd.DataFrame) -> dict[str, "CollegeToNFLTranslator"]:
    """
    Train college translators for all four positions.
    Usage:
        translators = train_all_translators(load_features_for_ml(player_type="college"))
    """
    models = {}
    for position in ("QB", "RB", "WR", "TE"):
        logger.info(f"\nTraining College Translator for {position}...")
        model = CollegeToNFLTranslator(position)
        try:
            metrics = model.train_with_tracking(df, params={"position": position})
            model.save()
            models[position] = model
            logger.info(f"  {position}: CV MAE = {metrics.get('cv_mae', 'N/A'):.2f} PPG")
        except ValueError as e:
            logger.warning(f"  {position}: skipped — {e}")
    return models


