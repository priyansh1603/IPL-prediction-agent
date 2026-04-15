"""
simulate.py — Monte Carlo IPL Match Simulator

Given two Playing XIs + venue + match context:
  1. Builds per-delivery feature vectors for every batter-bowler matchup
  2. Uses ball_outcome_model to get run/wicket probability distributions
  3. Simulates 10,000 T20 innings ball by ball
  4. Computes:
     - Score distributions for both teams
     - Win probability with confidence interval
     - Player performance distributions
     - Key matchup outcomes

Usage:
    from ml.simulate import simulate_match
    result = simulate_match(
        team1="Mumbai Indians",
        team1_xi=["RG Sharma", "IR Kishan", ...],
        team2="Chennai Super Kings",
        team2_xi=["Ruturaj Gaikwad", "Devon Conway", ...],
        venue="Wankhede Stadium",
        toss_winner="Mumbai Indians",
        toss_decision="bat"
    )
"""

import os
import pickle
import sqlite3
import logging
import numpy as np
import pandas as pd
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
pd.set_option('future.no_silent_downcasting', True)

DB_PATH   = os.getenv("DB_PATH",   "data/processed/ipl.db")
MODEL_DIR = "ml/models"

# ── Bowling order heuristics ───────────────────────────────────────────────
# In T20: typically 4 main bowlers bowl 4 overs each, allrounders fill rest
POWERPLAY_OVERS  = list(range(0, 6))
MIDDLE_OVERS     = list(range(6, 15))
DEATH_OVERS      = list(range(15, 20))

_ball_artifact   = None
_match_artifact  = None
_conn            = None


def _load_models():
    global _ball_artifact, _match_artifact
    if _ball_artifact is None:
        ball_path = os.path.join(MODEL_DIR, "ball_outcome_model.pkl")
        with open(ball_path, "rb") as f:
            _ball_artifact = pickle.load(f)
        log.info("Ball outcome model loaded")
    if _match_artifact is None:
        match_path = os.path.join(MODEL_DIR, "match_winner_model.pkl")
        with open(match_path, "rb") as f:
            _match_artifact = pickle.load(f)
        # Support both old (single model) and new (blended) artifact formats
        if "model" in _match_artifact and "xgb_model" not in _match_artifact:
            # Wrap old format for backward compatibility
            _match_artifact["xgb_model"]   = _match_artifact["model"]
            _match_artifact["lr_model"]    = None
            _match_artifact["blend_alpha"] = 1.0
        log.info("Match winner model loaded")


def _get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def _get_player_features(player: str, role: str, source: str = "ipl") -> dict:
    """Pull batting or bowling features for a player from DB."""
    conn = _get_conn()
    features = {}

    if role == "bat":
        row = conn.execute("""
            SELECT AVG(strike_rate) AS sr, AVG(batting_avg) AS avg,
                   SUM(total_runs) AS runs
            FROM player_batting_stats
            WHERE player = ? AND source = ?
        """, (player, source)).fetchone()
        features = {
            "batter_career_sr":   row["sr"]   or 120.0,
            "batter_career_avg":  row["avg"]  or 25.0,
            "batter_career_runs": row["runs"] or 500,
        }
        form = conn.execute("""
            SELECT recent_avg AS avg, recent_sr AS sr
            FROM player_rolling_form WHERE player = ? AND source = ?
        """, (player, source)).fetchone()
        features["batter_form_avg"] = form["avg"] if form else features["batter_career_avg"]
        features["batter_form_sr"]  = form["sr"]  if form else features["batter_career_sr"]

    elif role == "bowl":
        row = conn.execute("""
            SELECT AVG(economy) AS econ, AVG(bowling_avg) AS avg,
                   SUM(wickets) AS wkts
            FROM player_bowling_stats
            WHERE player = ? AND source = ?
        """, (player, source)).fetchone()
        features = {
            "bowler_career_economy":  row["econ"] or 8.5,
            "bowler_career_avg":      row["avg"]  or 30.0,
            "bowler_career_wickets":  row["wkts"] or 20,
        }
        form = conn.execute("""
            SELECT recent_economy AS econ, recent_wickets AS wkts
            FROM player_rolling_form WHERE player = ? AND source = ?
        """, (player, source)).fetchone()
        features["bowler_form_economy"] = form["econ"] if form else features["bowler_career_economy"]
        features["bowler_form_wickets"] = form["wkts"] if form else 1.0

    return features


def _get_h2h_features(batter: str, bowler: str, source: str = "ipl") -> dict:
    conn = _get_conn()
    row = conn.execute("""
        SELECT strike_rate AS sr, batting_avg AS avg,
               dominance_index AS dom, dot_ball_pct AS dot_pct, balls
        FROM batter_vs_bowler
        WHERE batter = ? AND bowler = ? AND source = ?
    """, (batter, bowler, source)).fetchone()
    if row and row["balls"] >= 6:
        return {
            "h2h_sr":         row["sr"]      or 100.0,
            "h2h_avg":        row["avg"]     or 20.0,
            "h2h_dominance":  row["dom"]     or 1.0,
            "h2h_dot_pct":    row["dot_pct"] or 30.0,
            "h2h_balls":      row["balls"]   or 0,
        }
    # No H2H data — use neutral defaults
    return {"h2h_sr": 100.0, "h2h_avg": 20.0, "h2h_dominance": 1.0,
            "h2h_dot_pct": 35.0, "h2h_balls": 0}


def _get_venue_features(venue: str, source: str = "ipl") -> dict:
    conn = _get_conn()
    row = conn.execute("""
        SELECT avg_first_innings AS avg_score, chasing_win_pct AS chase_pct,
               avg_powerplay_rpo AS pp_rpo, avg_death_rpo AS death_rpo
        FROM venue_pitch_profiles WHERE venue LIKE ? AND source = ?
        LIMIT 1
    """, (f"%{venue}%", source)).fetchone()
    if row:
        return {
            "venue_avg_score":  row["avg_score"]  or 165.0,
            "venue_chase_pct":  row["chase_pct"]  or 50.0,
            "venue_pp_rpo":     row["pp_rpo"]     or 7.5,
            "venue_death_rpo":  row["death_rpo"]  or 10.5,
        }
    return {"venue_avg_score": 165.0, "venue_chase_pct": 50.0,
            "venue_pp_rpo": 7.5, "venue_death_rpo": 10.5}


def _build_delivery_features(
    batter: str, bowler: str, over: int, ball: int,
    inning: int, is_chasing: int, toss_bat: int,
    venue_features: dict, source: str = "ipl"
) -> dict:
    """Build full feature vector for one delivery."""
    phase = 0 if over < 6 else (1 if over < 15 else 2)

    bat_f  = _get_player_features(batter, "bat",  source)
    bowl_f = _get_player_features(bowler, "bowl", source)
    h2h_f  = _get_h2h_features(batter, bowler, source)

    return {
        "over_num":                 over,
        "ball_num":                 ball,
        "phase":                    phase,
        "inning":                   inning,
        "is_chasing":               is_chasing,
        "toss_won_by_batting_team": toss_bat,
        **bat_f,
        **bowl_f,
        **h2h_f,
        **venue_features,
    }


def _assign_bowling_order(bowlers: list, overs: int = 20) -> list:
    """
    Assign bowlers to overs. Each bowler max 4 overs.
    Returns list of length `overs` with bowler name per over.
    """
    n = len(bowlers)
    quota = {b: 0 for b in bowlers}
    order = []
    for ov in range(overs):
        # Prioritize bowlers with quota remaining, cycle through
        available = [b for b in bowlers if quota[b] < 4]
        if not available:
            available = bowlers
        # Simple round-robin with max 4
        bowler = available[ov % len(available)]
        order.append(bowler)
        quota[bowler] += 1
    return order


def _precompute_probs(
    batting_xi: list,
    bowling_xi: list,
    bowling_order: list,
    venue_features: dict,
    is_chasing: int,
    toss_bat: int,
    source: str,
) -> dict:
    """
    Precompute predict_proba for every unique (batter, bowler, over_num) combination.

    Returns a dict mapping (batter, bowler, over_num) -> probs array of shape (n_classes,).
    This avoids calling XGBoost once per ball inside the tight simulation loop.
    """
    _load_models()
    artifact = _ball_artifact
    model    = artifact["model"]
    features = artifact["features"]

    # Inning value: 1 for first innings, 2 for second — we use a fixed ball_num=3
    # (mid-over) as a representative value; over_num encodes the phase accurately.
    inning = 2 if is_chasing else 1
    ball_repr = 3

    probs_cache: dict = {}
    unique_combos = set(
        (batter, bowling_order[over], over)
        for batter in batting_xi
        for over in range(len(bowling_order))
    )

    for (batter, bowler, over) in unique_combos:
        fv = _build_delivery_features(
            batter, bowler, over, ball_repr,
            inning, is_chasing, toss_bat,
            venue_features, source
        )
        X = np.array([[fv.get(f, 0) for f in features]], dtype=float)
        probs_cache[(batter, bowler, over)] = model.predict_proba(X)[0]

    return probs_cache


def _simulate_innings(
    batting_xi: list,
    bowling_xi: list,
    inning: int,
    is_chasing: int,
    toss_bat: int,
    venue_features: dict,
    target: Optional[int],
    source: str,
    rng: np.random.Generator,
    precomputed: dict,
) -> dict:
    """Simulate one T20 innings. Returns score, wickets, ball-by-ball detail."""
    _load_models()
    artifact = _ball_artifact
    runs_map = artifact["outcome_runs"]  # class → runs

    bowling_order = _assign_bowling_order(bowling_xi)

    score    = 0
    wickets  = 0
    balls    = 0
    batters  = list(batting_xi)
    at_crease= [batters[0], batters[1]]
    next_bat = 2
    striker  = 0  # index into at_crease

    over_scores = []

    for over in range(20):
        bowler     = bowling_order[over]
        over_score = 0
        over_wkts  = 0
        ball_in_over = 0

        while ball_in_over < 6:
            batter = at_crease[striker]

            probs = precomputed.get((batter, bowler, over))
            if probs is None:
                # Fallback: compute on the fly for unexpected batter (e.g. tail-ender
                # who was not in the original batting_xi passed to _precompute_probs)
                artifact  = _ball_artifact
                model_    = artifact["model"]
                features_ = artifact["features"]
                fv = _build_delivery_features(
                    batter, bowler, over, 3,
                    inning, is_chasing, toss_bat,
                    venue_features, source
                )
                X = np.array([[fv.get(f, 0) for f in features_]], dtype=float)
                probs = model_.predict_proba(X)[0]
                precomputed[(batter, bowler, over)] = probs  # cache for next sim

            # Sample outcome
            outcome_class = rng.choice(len(probs), p=probs)
            runs_scored   = runs_map[outcome_class]
            is_wicket     = (outcome_class == 6)

            if is_wicket:
                wickets += 1
                over_wkts += 1
                if next_bat < len(batters):
                    at_crease[striker] = batters[next_bat]
                    next_bat += 1
                else:
                    break  # all out
            else:
                score     += runs_scored
                over_score += runs_scored
                balls      += 1
                ball_in_over += 1
                # Rotate strike on odd runs
                if runs_scored % 2 == 1:
                    striker = 1 - striker

            ball_in_over += 1

            # Early finish if chasing
            if is_chasing and target and score >= target:
                over_scores.append({"over": over, "runs": over_score, "wickets": over_wkts})
                return {"score": score, "wickets": wickets, "balls": balls,
                        "over_scores": over_scores, "result": "won"}

            if wickets >= 10:
                over_scores.append({"over": over, "runs": over_score, "wickets": over_wkts})
                return {"score": score, "wickets": wickets, "balls": balls,
                        "over_scores": over_scores, "result": "all_out"}

        # End-of-over rotate strike
        striker = 1 - striker
        over_scores.append({"over": over, "runs": over_score, "wickets": over_wkts})

    return {"score": score, "wickets": wickets, "balls": 120,
            "over_scores": over_scores, "result": "completed"}


def predict_match_winner(match_features: dict) -> float:
    """
    Use the blended XGBoost + LR model to predict P(team1 wins).

    match_features: dict with keys matching MATCH_FEATURES in train.py
    Returns: float in [0, 1]
    """
    _load_models()
    artifact   = _match_artifact
    features   = artifact["features"]
    xgb_model  = artifact["xgb_model"]
    lr_model   = artifact.get("lr_model")
    alpha      = artifact.get("blend_alpha", 1.0)

    X = np.array([[match_features.get(f, 0) for f in features]], dtype=float)

    xgb_prob = float(xgb_model.predict_proba(X)[0, 1])
    if lr_model is not None and alpha < 1.0:
        # LR pipeline (StandardScaler) needs DataFrame with column names
        X_df = pd.DataFrame([{f: match_features.get(f, 0) for f in features}]).fillna(0).astype(float)
        lr_prob = float(lr_model.predict_proba(X_df)[0, 1])
        return alpha * xgb_prob + (1 - alpha) * lr_prob
    return xgb_prob


def simulate_match(
    team1: str,
    team1_xi: list,
    team2: str,
    team2_xi: list,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    n_simulations: int = 1000,
    source: str = "ipl",
) -> dict:
    """
    Run Monte Carlo simulation for a T20 match.

    Returns full probability breakdown, score distributions,
    and key matchup insights.
    """
    _load_models()
    rng = np.random.default_rng(42)

    venue_features = _get_venue_features(venue, source)

    # Determine batting order
    if toss_decision == "bat":
        batting_first  = team1 if toss_winner == team1 else team2
        batting_second = team2 if batting_first == team1 else team1
        xi_first       = team1_xi if batting_first == team1 else team2_xi
        xi_second      = team2_xi if batting_first == team1 else team1_xi
    else:
        batting_first  = team2 if toss_winner == team1 else team1
        batting_second = team1 if batting_first == team2 else team2
        xi_first       = team2_xi if batting_first == team2 else team1_xi
        xi_second      = team1_xi if batting_first == team2 else team2_xi

    toss_bat_first  = 1 if toss_winner == batting_first else 0
    toss_bat_second = 1 - toss_bat_first

    # Precompute ball-outcome probabilities for all (batter, bowler, over) combos
    # in each innings — avoids 1000 × 240 individual XGBoost calls in the loop.
    bowling_order_first  = _assign_bowling_order(xi_second)
    bowling_order_second = _assign_bowling_order(xi_first)

    precomputed_first = _precompute_probs(
        batting_xi=xi_first, bowling_xi=xi_second,
        bowling_order=bowling_order_first,
        venue_features=venue_features,
        is_chasing=0, toss_bat=toss_bat_first,
        source=source,
    )
    precomputed_second = _precompute_probs(
        batting_xi=xi_second, bowling_xi=xi_first,
        bowling_order=bowling_order_second,
        venue_features=venue_features,
        is_chasing=1, toss_bat=toss_bat_second,
        source=source,
    )

    log.info(f"Simulating {n_simulations} matches: {team1} vs {team2} at {venue}")

    team1_wins  = 0
    first_scores  = []
    second_scores = []

    for sim in range(n_simulations):
        # Innings 1
        inn1 = _simulate_innings(
            batting_xi=xi_first,
            bowling_xi=xi_second,
            inning=1, is_chasing=0,
            toss_bat=toss_bat_first,
            venue_features=venue_features,
            target=None, source=source, rng=rng,
            precomputed=precomputed_first,
        )
        target = inn1["score"] + 1

        # Innings 2
        inn2 = _simulate_innings(
            batting_xi=xi_second,
            bowling_xi=xi_first,
            inning=2, is_chasing=1,
            toss_bat=toss_bat_second,
            venue_features=venue_features,
            target=target, source=source, rng=rng,
            precomputed=precomputed_second,
        )

        first_scores.append(inn1["score"])
        second_scores.append(inn2["score"])

        # Determine winner
        if inn2["score"] >= target:
            winner = batting_second
        else:
            winner = batting_first

        if winner == team1:
            team1_wins += 1

    first_scores  = np.array(first_scores)
    second_scores = np.array(second_scores)
    team1_win_pct = team1_wins / n_simulations * 100

    # Confidence interval (Wilson score)
    p = team1_wins / n_simulations
    z = 1.96
    ci_low  = (p + z**2/(2*n_simulations) - z*np.sqrt(p*(1-p)/n_simulations + z**2/(4*n_simulations**2))) \
              / (1 + z**2/n_simulations)
    ci_high = (p + z**2/(2*n_simulations) + z*np.sqrt(p*(1-p)/n_simulations + z**2/(4*n_simulations**2))) \
              / (1 + z**2/n_simulations)

    result = {
        "team1":                    team1,
        "team2":                    team2,
        "venue":                    venue,
        "toss_winner":              toss_winner,
        "toss_decision":            toss_decision,
        "simulations_run":          n_simulations,
        "batting_first":            batting_first,
        "batting_second":           batting_second,
        "team1_win_probability":    round(team1_win_pct, 1),
        "team2_win_probability":    round(100 - team1_win_pct, 1),
        "confidence_interval_95":  {
            "low":  round(ci_low * 100, 1),
            "high": round(ci_high * 100, 1),
        },
        "score_distributions": {
            "first_innings": {
                "mean":   round(float(first_scores.mean()), 1),
                "median": round(float(np.median(first_scores)), 1),
                "std":    round(float(first_scores.std()), 1),
                "p10":    int(np.percentile(first_scores, 10)),
                "p25":    int(np.percentile(first_scores, 25)),
                "p75":    int(np.percentile(first_scores, 75)),
                "p90":    int(np.percentile(first_scores, 90)),
            },
            "second_innings": {
                "mean":   round(float(second_scores.mean()), 1),
                "median": round(float(np.median(second_scores)), 1),
                "std":    round(float(second_scores.std()), 1),
                "p10":    int(np.percentile(second_scores, 10)),
                "p25":    int(np.percentile(second_scores, 25)),
                "p75":    int(np.percentile(second_scores, 75)),
                "p90":    int(np.percentile(second_scores, 90)),
            },
        },
        "key_matchups": _get_key_matchups(team1_xi, team2_xi, source),
    }

    log.info(f"Simulation complete: {team1} {team1_win_pct:.1f}% | {team2} {100-team1_win_pct:.1f}%")
    return result


def _get_key_matchups(xi1: list, xi2: list, source: str) -> list:
    """Get top 5 most impactful batter-bowler matchups between the two XIs."""
    conn = _get_conn()
    matchups = []

    for batter in xi1:
        for bowler in xi2:
            row = conn.execute("""
                SELECT batter, bowler, balls, runs, dismissals,
                       strike_rate, dot_ball_pct, dominance_index
                FROM batter_vs_bowler
                WHERE batter = ? AND bowler = ? AND source = ? AND balls >= 6
            """, (batter, bowler, source)).fetchone()
            if row:
                matchups.append(dict(row))

    for batter in xi2:
        for bowler in xi1:
            row = conn.execute("""
                SELECT batter, bowler, balls, runs, dismissals,
                       strike_rate, dot_ball_pct, dominance_index
                FROM batter_vs_bowler
                WHERE batter = ? AND bowler = ? AND source = ? AND balls >= 6
            """, (batter, bowler, source)).fetchone()
            if row:
                matchups.append(dict(row))

    # Sort by balls faced (most experienced matchups first)
    matchups.sort(key=lambda x: x["balls"], reverse=True)
    return matchups[:10]
