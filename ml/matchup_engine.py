"""
matchup_engine.py — Player matchup predictor

Given a batter and bowler, predicts:
- Expected runs per ball
- Wicket probability per ball
- Dot ball probability
- Dominant matchup direction (batter or bowler)
"""

import os
import pickle
import sqlite3
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

DB_PATH   = os.getenv("DB_PATH", "data/processed/ipl.db")
MODEL_DIR = "ml/models"

_artifact = None
_conn     = None


def _load():
    global _artifact
    if _artifact is None:
        with open(os.path.join(MODEL_DIR, "ball_outcome_model.pkl"), "rb") as f:
            _artifact = pickle.load(f)


def _get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def predict_matchup(
    batter: str,
    bowler: str,
    over: int = 10,
    venue: str = None,
    source: str = "ipl",
    n_balls: int = 100,
) -> dict:
    """
    Predict outcome distribution for a batter-bowler matchup.

    Returns:
        expected_runs_per_ball, wicket_prob, dot_prob,
        boundary_prob, six_prob, verdict (who dominates)
    """
    _load()
    conn = _get_conn()

    phase = 0 if over < 6 else (1 if over < 15 else 2)

    # Batter stats
    bat = conn.execute("""
        SELECT AVG(strike_rate) AS sr, AVG(batting_avg) AS avg,
               SUM(total_runs) AS runs
        FROM player_batting_stats WHERE player = ? AND source = ?
    """, (batter, source)).fetchone()

    # Bowler stats
    bowl = conn.execute("""
        SELECT AVG(economy) AS econ, AVG(bowling_avg) AS avg,
               SUM(wickets) AS wkts
        FROM player_bowling_stats WHERE player = ? AND source = ?
    """, (bowler, source)).fetchone()

    # H2H
    h2h = conn.execute("""
        SELECT strike_rate AS sr, batting_avg AS avg,
               dominance_index AS dom, dot_ball_pct AS dot_pct, balls
        FROM batter_vs_bowler
        WHERE batter = ? AND bowler = ? AND source = ?
    """, (batter, bowler, source)).fetchone()

    # Form
    bat_form = conn.execute("""
        SELECT recent_avg AS avg, recent_sr AS sr
        FROM player_rolling_form WHERE player = ? AND source = ?
    """, (batter, source)).fetchone()

    bowl_form = conn.execute("""
        SELECT recent_economy AS econ, recent_wickets AS wkts
        FROM player_rolling_form WHERE player = ? AND source = ?
    """, (bowler, source)).fetchone()

    # Venue
    venue_data = None
    if venue:
        venue_data = conn.execute("""
            SELECT avg_first_innings AS avg_score, chasing_win_pct AS chase_pct,
                   avg_powerplay_rpo AS pp_rpo, avg_death_rpo AS death_rpo
            FROM venue_pitch_profiles WHERE venue LIKE ? AND source = ? LIMIT 1
        """, (f"%{venue}%", source)).fetchone()

    fv = {
        "over_num":                 over,
        "ball_num":                 0,
        "phase":                    phase,
        "inning":                   1,
        "is_chasing":               0,
        "toss_won_by_batting_team": 1,
        "batter_career_sr":         (bat["sr"]   if bat else None)  or 120.0,
        "batter_career_avg":        (bat["avg"]  if bat else None)  or 25.0,
        "batter_career_runs":       (bat["runs"] if bat else None)  or 500,
        "batter_form_avg":          (bat_form["avg"] if bat_form else None) or 25.0,
        "batter_form_sr":           (bat_form["sr"]  if bat_form else None) or 120.0,
        "bowler_career_economy":    (bowl["econ"] if bowl else None) or 8.5,
        "bowler_career_avg":        (bowl["avg"]  if bowl else None) or 30.0,
        "bowler_career_wickets":    (bowl["wkts"] if bowl else None) or 20,
        "bowler_form_economy":      (bowl_form["econ"] if bowl_form else None) or 8.5,
        "bowler_form_wickets":      (bowl_form["wkts"] if bowl_form else None) or 1.0,
        "h2h_sr":          (h2h["sr"]      if h2h and h2h["balls"] >= 6 else None) or 100.0,
        "h2h_avg":         (h2h["avg"]     if h2h and h2h["balls"] >= 6 else None) or 20.0,
        "h2h_dominance":   (h2h["dom"]     if h2h and h2h["balls"] >= 6 else None) or 1.0,
        "h2h_dot_pct":     (h2h["dot_pct"] if h2h and h2h["balls"] >= 6 else None) or 35.0,
        "h2h_balls":       (h2h["balls"]   if h2h else None) or 0,
        "venue_avg_score": (venue_data["avg_score"]  if venue_data else None) or 165.0,
        "venue_chase_pct": (venue_data["chase_pct"]  if venue_data else None) or 50.0,
        "venue_pp_rpo":    (venue_data["pp_rpo"]     if venue_data else None) or 7.5,
        "venue_death_rpo": (venue_data["death_rpo"]  if venue_data else None) or 10.5,
    }

    features = _artifact["features"]
    X = pd.DataFrame([{f: fv.get(f, 0) for f in features}]).fillna(0).astype(float)
    probs = _artifact["model"].predict_proba(X)[0]

    # Class mapping: 0=dot,1=single,2=two,3=three,4=four,5=six,6=wicket
    runs_per_class = np.array([0, 1, 2, 3, 4, 6, 0])
    expected_rpb   = float(np.dot(probs, runs_per_class))
    wicket_prob    = float(probs[6])
    dot_prob       = float(probs[0])
    four_prob      = float(probs[4])
    six_prob       = float(probs[5])
    boundary_prob  = four_prob + six_prob

    # Verdict
    league_avg_rpb = 1.30
    if expected_rpb > league_avg_rpb * 1.15 and wicket_prob < 0.04:
        verdict = "batter_dominates"
    elif expected_rpb < league_avg_rpb * 0.85 or wicket_prob > 0.07:
        verdict = "bowler_dominates"
    else:
        verdict = "neutral"

    h2h_summary = None
    if h2h and h2h["balls"] >= 6:
        h2h_summary = {
            "balls":      h2h["balls"],
            "runs":       int(h2h["sr"] * h2h["balls"] / 100) if h2h["sr"] else 0,
            "dismissals": h2h.get("dismissals", 0),
            "h2h_sr":     h2h["sr"],
            "dominance":  h2h["dom"],
        }

    return {
        "batter":               batter,
        "bowler":               bowler,
        "over":                 over,
        "phase":                ["powerplay","middle","death"][phase],
        "expected_runs_per_ball": round(expected_rpb, 3),
        "projected_sr":         round(expected_rpb * 100, 1),
        "wicket_probability":   round(wicket_prob * 100, 1),
        "dot_ball_probability": round(dot_prob * 100, 1),
        "boundary_probability": round(boundary_prob * 100, 1),
        "six_probability":      round(six_prob * 100, 1),
        "verdict":              verdict,
        "h2h_history":          h2h_summary,
        "confidence":           "high" if (h2h and h2h["balls"] >= 20) else
                                "medium" if (h2h and h2h["balls"] >= 6) else "low",
    }
