"""
train.py — Train ML models for IPL prediction

Models:
  1. ball_outcome_model.pkl  — XGBoost multiclass (7 classes: dot/1/2/3/4/6/W)
  2. match_winner_model.pkl  — Blended XGBoost + Logistic Regression
     - Trained on IPL + T20I (5,146 extra matches)
     - IPL rows weighted 2× higher than T20I rows
     - Blend: 60% XGBoost + 40% LR

Run:
    python -m ml.train
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR  = "data/processed/ml"
MODEL_DIR = "ml/models"

# ── Ball features ──────────────────────────────────────────────────────────
BALL_FEATURES = [
    "over_num", "ball_num", "phase", "inning", "is_chasing",
    "toss_won_by_batting_team",
    "batter_career_sr", "batter_career_avg", "batter_career_runs",
    "batter_form_avg", "batter_form_sr",
    "bowler_career_economy", "bowler_career_avg", "bowler_career_wickets",
    "bowler_form_wickets", "bowler_form_economy",
    "h2h_sr", "h2h_avg", "h2h_dominance", "h2h_dot_pct", "h2h_balls",
    "venue_avg_score", "venue_chase_pct", "venue_pp_rpo", "venue_death_rpo",
]

# ── Match features (differentials only — no redundancy) ───────────────────
MATCH_FEATURES = [
    "toss_decision_bat", "toss_winner_is_team1",
    "momentum_diff", "streak_diff", "t1_roll_games",
    "h2h_advantage", "h2h_roll_matches",
    "venue_avg_score", "venue_chase_pct",
    "venue_bat_first_win_pct", "toss_bat_win_pct",
    "dew_score",
    "t1_is_home", "t2_is_home",
    "venue_advantage",
    "rest_advantage",
    "score_diff_last5", "bowling_diff_last5",
    "t1_avg_wickets_last5", "t2_avg_wickets_last5",
    "t1_prev_margin", "t2_prev_margin",
    "is_evening", "season_num",
]


# ══════════════════════════════════════════════════════════════════════════
# BALL MODEL
# ══════════════════════════════════════════════════════════════════════════

def train_ball_model():
    log.info("Training ball outcome model...")
    df = pd.read_parquet(os.path.join(DATA_DIR, "ball_features.parquet"))
    df = df[df["is_legal"] == 1].copy()

    outcome_map    = {0:0, 1:1, 2:2, 3:3, 4:4, 6:5, 7:6}
    outcome_labels = ["dot","single","two","three","four","six","wicket"]
    df["outcome_class"] = df["ball_outcome"].map(outcome_map)

    available = [f for f in BALL_FEATURES if f in df.columns]
    X = df[available].fillna(0).astype(float)
    y = df["outcome_class"].astype(int)
    log.info(f"  Total legal deliveries: {len(X):,}")
    log.info(f"  Class distribution:\n{y.value_counts().sort_index()}")

    # Sample cap — 600k rows trains in ~1 min, same accuracy as full 2M
    SAMPLE_CAP = 600_000
    if len(X) > SAMPLE_CAP:
        log.info(f"  Sampling down to {SAMPLE_CAP:,} rows...")
        idx = (pd.Series(y).groupby(y)
               .apply(lambda g: g.sample(frac=SAMPLE_CAP/len(X), random_state=42))
               .droplevel(0).index)
        X, y = X.loc[idx], y.loc[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="mlogloss", early_stopping_rounds=20,
        random_state=42, n_jobs=-1,
        objective="multi:softprob", num_class=7,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=25)

    preds = model.predict(X_test)
    log.info(f"  Ball model accuracy: {accuracy_score(y_test, preds):.4f}")
    log.info(f"\n{classification_report(y_test, preds, target_names=outcome_labels)}")
    importance = pd.Series(model.feature_importances_, index=available)
    log.info(f"\nTop 10 features:\n{importance.nlargest(10)}")
    return model, available, outcome_map, outcome_labels


# ══════════════════════════════════════════════════════════════════════════
# MATCH MODEL  (XGBoost + LR blend, IPL + T20I)
# ══════════════════════════════════════════════════════════════════════════

def train_match_model():
    log.info("Training match winner model (XGBoost + LR blend, IPL + T20I)...")

    ipl_path = os.path.join(DATA_DIR, "match_features.parquet")
    t20_path = os.path.join(DATA_DIR, "match_features_t20.parquet")

    ipl_df = pd.read_parquet(ipl_path)
    ipl_df = ipl_df.dropna(subset=["team1_won"])
    ipl_df["sample_weight"] = 2.0      # IPL rows weighted 2× more important
    log.info(f"  IPL matches: {len(ipl_df):,}")

    if os.path.exists(t20_path):
        t20_df = pd.read_parquet(t20_path)
        t20_df = t20_df.dropna(subset=["team1_won"])
        t20_df["sample_weight"] = 1.0
        combined = pd.concat([ipl_df, t20_df], ignore_index=True)
        log.info(f"  T20I matches: {len(t20_df):,}")
        log.info(f"  Combined: {len(combined):,} rows")
    else:
        combined = ipl_df
        log.warning("  T20I parquet missing — run: python -m pipeline.ml_features_t20")
        log.warning("  Training on IPL only (lower AUC expected)")

    available = [f for f in MATCH_FEATURES if f in combined.columns]
    missing   = [f for f in MATCH_FEATURES if f not in combined.columns]
    if missing:
        log.warning(f"  Skipping missing features: {missing}")

    X  = combined[available].fillna(0).astype(float)
    y  = combined["team1_won"].astype(int)
    sw = combined["sample_weight"].values

    log.info(f"  Features ({len(available)}): {available}")
    log.info(f"  Win/loss split: {y.mean():.3f}  (0.50 = balanced)")

    # Group-aware split: keep original + flipped rows of same match together
    groups = combined["match_group_id"].values if "match_group_id" in combined.columns else None
    if groups is not None:
        unique_grp = np.unique(groups)
        rng        = np.random.default_rng(42)
        test_grps  = set(rng.choice(unique_grp, size=int(0.2*len(unique_grp)), replace=False))
        test_mask  = np.array([g in test_grps for g in groups])
        X_train, X_test   = X[~test_mask],  X[test_mask]
        y_train, y_test   = y[~test_mask],  y[test_mask]
        sw_train          = sw[~test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        sw_train = sw[:len(X_train)]

    # ── XGBoost ──
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.8,
        min_child_weight=4, gamma=0.5,
        reg_alpha=0.05, reg_lambda=1.5,
        tree_method="hist", eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, sample_weight=sw_train, verbose=False)

    # ── Logistic Regression (stable on small data, good calibration) ──
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.05, max_iter=2000, random_state=42, solver="lbfgs")),
    ])
    lr_pipe.fit(X_train, y_train, lr__sample_weight=sw_train)

    # ── Blend 60 / 40 ──
    xgb_prob   = xgb_model.predict_proba(X_test)[:, 1]
    lr_prob    = lr_pipe.predict_proba(X_test)[:, 1]
    blend_prob = 0.60 * xgb_prob + 0.40 * lr_prob
    blend_pred = (blend_prob >= 0.5).astype(int)

    log.info(f"  XGBoost AUC:  {roc_auc_score(y_test, xgb_prob):.4f}")
    log.info(f"  LR AUC:       {roc_auc_score(y_test, lr_prob):.4f}")
    log.info(f"  Blended AUC:  {roc_auc_score(y_test, blend_prob):.4f}  (random=0.50, good≥0.60)")
    log.info(f"  Accuracy:     {accuracy_score(y_test, blend_pred):.4f}")

    # ── Cross-validation on XGBoost (GroupKFold, leak-free) ──
    if groups is not None:
        # Manual GroupKFold CV (avoids sklearn version issues with fit_params)
        gkf = GroupKFold(n_splits=5)
        cv_aucs = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
            ytr, yval = y.iloc[train_idx], y.iloc[val_idx]
            swtr      = sw[train_idx]
            m = xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.015,
                subsample=0.75, colsample_bytree=0.8,
                min_child_weight=4, gamma=0.5,
                reg_alpha=0.05, reg_lambda=1.5,
                tree_method="hist", eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )
            m.fit(Xtr, ytr, sample_weight=swtr, verbose=False)
            prob = m.predict_proba(Xval)[:, 1]
            cv_aucs.append(roc_auc_score(yval, prob))
        cv_aucs = np.array(cv_aucs)
        log.info(f"  XGB CV AUC (GroupKFold): {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    importance = pd.Series(xgb_model.feature_importances_, index=available)
    log.info(f"\nTop 10 features:\n{importance.nlargest(10)}")
    return xgb_model, lr_pipe, available


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Ball model
    ball_model, ball_features, outcome_map, outcome_labels = train_ball_model()
    ball_path = os.path.join(MODEL_DIR, "ball_outcome_model.pkl")
    with open(ball_path, "wb") as f:
        pickle.dump({
            "model": ball_model, "features": ball_features,
            "outcome_map": outcome_map, "outcome_labels": outcome_labels,
            "outcome_runs": {0:0, 1:1, 2:2, 3:3, 4:4, 5:6, 6:0},
        }, f)
    log.info(f"Saved ball model → {ball_path}")

    # Match model
    xgb_model, lr_model, match_features = train_match_model()
    match_path = os.path.join(MODEL_DIR, "match_winner_model.pkl")
    with open(match_path, "wb") as f:
        pickle.dump({
            "xgb_model":   xgb_model,
            "lr_model":    lr_model,
            "features":    match_features,
            "blend_alpha": 0.60,
        }, f)
    log.info(f"Saved match model → {match_path}")


if __name__ == "__main__":
    main()
