"""
ml_features.py — Build ML training dataset from ball-by-ball data

Creates two datasets:
  1. ball_features.parquet  — per-delivery features for ball outcome model
     Features: batter form, bowler form, H2H, phase, venue, match state
     Target: runs on this ball (0,1,2,3,4,6) or W (wicket)

  2. match_features.parquet — per-match features for match winner model
     Features: rolling team form, rolling H2H, venue, toss
     Target: did team1 win? (1/0)
     KEY: all features computed from matches BEFORE this match (no leakage)

Run:
    python pipeline/ml_features.py
"""

import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH    = os.getenv("DB_PATH",    "data/processed/ipl.db")
OUTPUT_DIR = "data/processed/ml"


# ──────────────────────────────────────────────────────────────────────────────
# BALL FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def build_ball_features(conn: sqlite3.Connection) -> pd.DataFrame:
    log.info("Building ball-level feature dataset...")

    query = """
    SELECT
        d.match_id, d.inning, d.over_num, d.ball_num,
        d.batter, d.bowler,
        d.runs_batter, d.runs_total, d.is_wicket,
        d.extras_type, d.wicket_kind,
        m.venue, m.season, m.source, m.date,
        m.team1, m.team2, m.toss_winner, m.toss_decision,
        m.winner,
        CASE
            WHEN d.over_num < 6  THEN 0
            WHEN d.over_num < 15 THEN 1
            ELSE 2
        END AS phase,
        CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END AS is_legal
    FROM deliveries d
    JOIN matches m ON m.match_id = d.match_id
    WHERE m.source = 'ipl'
    ORDER BY m.date, d.match_id, d.inning, d.over_num, d.ball_num
    """
    df = pd.read_sql_query(query, conn)
    log.info(f"  Raw deliveries: {len(df):,}")

    # ── Batter career stats (per season) ──
    bat_stats = pd.read_sql_query("""
        SELECT player, season, source,
               AVG(strike_rate) AS career_sr,
               AVG(batting_avg) AS career_avg,
               SUM(total_runs)  AS career_runs
        FROM player_batting_stats
        GROUP BY player, season, source
    """, conn)
    bat_stats.columns = ["batter","season","source","batter_career_sr","batter_career_avg","batter_career_runs"]

    # ── Bowler career stats ──
    bowl_stats = pd.read_sql_query("""
        SELECT player, season, source,
               AVG(economy)     AS career_economy,
               AVG(bowling_avg) AS career_bowl_avg,
               SUM(wickets)     AS career_wickets
        FROM player_bowling_stats
        GROUP BY player, season, source
    """, conn)
    bowl_stats.columns = ["bowler","season","source","bowler_career_economy","bowler_career_avg","bowler_career_wickets"]

    # ── Batter vs Bowler H2H ──
    h2h = pd.read_sql_query("""
        SELECT batter, bowler, source,
               strike_rate AS h2h_sr,
               batting_avg AS h2h_avg,
               dot_ball_pct AS h2h_dot_pct,
               dominance_index AS h2h_dominance,
               dismissals AS h2h_dismissals,
               balls AS h2h_balls
        FROM batter_vs_bowler
    """, conn)

    # ── Rolling form ──
    form = pd.read_sql_query("""
        SELECT player, source,
               recent_avg AS form_avg,
               recent_sr  AS form_sr,
               recent_wickets AS form_wickets,
               recent_economy AS form_economy
        FROM player_rolling_form
    """, conn)
    bat_form  = form.rename(columns={"player":"batter","form_avg":"batter_form_avg","form_sr":"batter_form_sr",
                                      "form_wickets":"_","form_economy":"__"}).drop(columns=["_","__"])
    bowl_form = form.rename(columns={"player":"bowler","form_wickets":"bowler_form_wickets",
                                      "form_economy":"bowler_form_economy",
                                      "form_avg":"_","form_sr":"__"}).drop(columns=["_","__"])

    # ── Venue stats ──
    venue_stats = pd.read_sql_query("""
        SELECT venue, source,
               avg_first_innings AS venue_avg_score,
               chasing_win_pct   AS venue_chase_pct,
               avg_powerplay_rpo AS venue_pp_rpo,
               avg_death_rpo     AS venue_death_rpo
        FROM venue_pitch_profiles
    """, conn)
    venue_stats = venue_stats.groupby(["venue","source"]).mean(numeric_only=True).reset_index()

    # ── Merge ──
    df = df.merge(bat_stats,   on=["batter","season","source"], how="left")
    df = df.merge(bowl_stats,  on=["bowler","season","source"], how="left")
    df = df.merge(h2h,         on=["batter","bowler","source"], how="left")
    df = df.merge(bat_form,    on=["batter","source"],          how="left")
    df = df.merge(bowl_form,   on=["bowler","source"],          how="left")
    df = df.merge(venue_stats, on=["venue","source"],           how="left")

    # ── Target: ball outcome ──
    def outcome(row):
        if row["is_wicket"] == 1 and row["wicket_kind"] not in ["run out","retired hurt","retired out","obstructing the field"]:
            return 7
        return int(row["runs_batter"]) if row["runs_batter"] in [0,1,2,3,4,6] else 1

    df["ball_outcome"] = df.apply(outcome, axis=1)

    df["toss_won_by_batting_team"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["is_chasing"] = (df["inning"] == 2).astype(int)

    num_cols = ["batter_career_sr","batter_career_avg","bowler_career_economy",
                "h2h_sr","h2h_avg","h2h_dominance","h2h_dot_pct","h2h_balls",
                "batter_form_avg","batter_form_sr","bowler_form_wickets","bowler_form_economy",
                "venue_avg_score","venue_chase_pct","venue_pp_rpo","venue_death_rpo"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    log.info(f"  Ball features dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MATCH FEATURES  (time-aware rolling: no data leakage)
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_team_stats(matches: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    For each match, compute for BOTH teams:
      - rolling win% over last `window` matches (before this match)
      - rolling streak (consecutive wins/losses, signed)
      - matches played so far

    Returns a DataFrame indexed like matches with extra columns:
      t1_roll_win_pct, t1_roll_streak, t1_roll_games,
      t2_roll_win_pct, t2_roll_streak, t2_roll_games
    """
    matches = matches.sort_values("date").reset_index(drop=True)

    # Build a results list: (date, team, won 1/0)
    results = []
    for _, r in matches.iterrows():
        if pd.notna(r["winner"]) and r["winner"] != "":
            results.append({"date": r["date"], "match_id": r["match_id"],
                            "team": r["team1"], "won": int(r["winner"] == r["team1"])})
            results.append({"date": r["date"], "match_id": r["match_id"],
                            "team": r["team2"], "won": int(r["winner"] == r["team2"])})

    res_df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)

    def get_rolling(team, before_date):
        hist = res_df[(res_df["team"] == team) & (res_df["date"] < before_date)].tail(window)
        if len(hist) == 0:
            return 0.5, 0, 0
        win_pct = hist["won"].mean()
        games   = len(hist)
        # streak: count of same outcome from the end
        outcomes = hist["won"].tolist()
        last_val = outcomes[-1]
        streak = 0
        for v in reversed(outcomes):
            if v == last_val:
                streak += 1
            else:
                break
        streak = streak if last_val == 1 else -streak
        return round(win_pct, 4), streak, games

    t1_win_pct, t1_streak, t1_games = [], [], []
    t2_win_pct, t2_streak, t2_games = [], [], []

    for _, row in matches.iterrows():
        wp1, s1, g1 = get_rolling(row["team1"], row["date"])
        wp2, s2, g2 = get_rolling(row["team2"], row["date"])
        t1_win_pct.append(wp1); t1_streak.append(s1); t1_games.append(g1)
        t2_win_pct.append(wp2); t2_streak.append(s2); t2_games.append(g2)

    matches = matches.copy()
    matches["t1_roll_win_pct"] = t1_win_pct
    matches["t1_roll_streak"]  = t1_streak
    matches["t1_roll_games"]   = t1_games
    matches["t2_roll_win_pct"] = t2_win_pct
    matches["t2_roll_streak"]  = t2_streak
    matches["t2_roll_games"]   = t2_games
    return matches


def _rolling_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each match compute H2H win% for team1 against team2 using
    only matches played BEFORE this date.
    """
    matches = matches.sort_values("date").reset_index(drop=True)

    h2h_win_pct = []
    h2h_matches  = []

    for idx, row in matches.iterrows():
        t1, t2, date = row["team1"], row["team2"], row["date"]

        prior = matches.iloc[:idx]
        h2h = prior[
            ((prior["team1"] == t1) & (prior["team2"] == t2)) |
            ((prior["team1"] == t2) & (prior["team2"] == t1))
        ]

        if len(h2h) == 0:
            h2h_win_pct.append(0.5)
            h2h_matches.append(0)
            continue

        t1_wins = 0
        for _, hrow in h2h.iterrows():
            if hrow["winner"] == t1:
                t1_wins += 1
        h2h_win_pct.append(round(t1_wins / len(h2h), 4))
        h2h_matches.append(len(h2h))

    matches = matches.copy()
    matches["h2h_roll_win_pct"] = h2h_win_pct
    matches["h2h_roll_matches"] = h2h_matches
    return matches


def _venue_toss_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute venue-level stats using only prior matches:
      - venue_bat_first_win_pct  (chasing win% is 1 - this)
      - toss_bat_win_pct         (win% when batting first after toss)
    """
    matches = matches.sort_values("date").reset_index(drop=True)

    vbf_wp  = []  # venue bat first win pct
    toss_wp = []  # toss bat first win pct

    for idx, row in matches.iterrows():
        prior = matches.iloc[:idx]

        if len(prior) == 0:
            vbf_wp.append(0.5)
            toss_wp.append(0.5)
            continue

        # Venue batting first win%
        v_prior = prior[prior["venue"] == row["venue"]]
        if len(v_prior) >= 5:
            # batting first = team that did NOT chase
            bf_wins = 0
            for _, vr in v_prior.iterrows():
                if vr["toss_decision"] == "bat" and vr["winner"] == vr["toss_winner"]:
                    bf_wins += 1
                elif vr["toss_decision"] == "field" and vr["winner"] != vr["toss_winner"]:
                    bf_wins += 1
            vbf_wp.append(round(bf_wins / len(v_prior), 4))
        else:
            vbf_wp.append(0.5)

        # Toss bat first win%
        if len(prior) >= 10:
            bat_first = prior[prior["toss_decision"] == "bat"]
            if len(bat_first) > 0:
                tw = (bat_first["winner"] == bat_first["toss_winner"]).mean()
                toss_wp.append(round(tw, 4))
            else:
                toss_wp.append(0.5)
        else:
            toss_wp.append(0.5)

    matches = matches.copy()
    matches["venue_bat_first_win_pct"] = vbf_wp
    matches["toss_bat_win_pct"]         = toss_wp
    return matches


def _venue_team_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute each team's rolling win% at THIS specific venue
    using only matches played BEFORE this date.
    """
    matches = matches.sort_values("date").reset_index(drop=True)

    t1_venue_wp, t2_venue_wp = [], []

    for idx, row in matches.iterrows():
        prior = matches.iloc[:idx]
        t1, t2, venue = row["team1"], row["team2"], row["venue"]

        # team1 at this venue
        t1_at_venue = prior[
            ((prior["team1"] == t1) | (prior["team2"] == t1)) &
            (prior["venue"] == venue)
        ]
        if len(t1_at_venue) >= 3:
            t1_wins = ((t1_at_venue["team1"] == t1) & (t1_at_venue["winner"] == t1)).sum() + \
                      ((t1_at_venue["team2"] == t1) & (t1_at_venue["winner"] == t1)).sum()
            t1_venue_wp.append(round(t1_wins / len(t1_at_venue), 4))
        else:
            t1_venue_wp.append(0.5)

        # team2 at this venue
        t2_at_venue = prior[
            ((prior["team1"] == t2) | (prior["team2"] == t2)) &
            (prior["venue"] == venue)
        ]
        if len(t2_at_venue) >= 3:
            t2_wins = ((t2_at_venue["team1"] == t2) & (t2_at_venue["winner"] == t2)).sum() + \
                      ((t2_at_venue["team2"] == t2) & (t2_at_venue["winner"] == t2)).sum()
            t2_venue_wp.append(round(t2_wins / len(t2_at_venue), 4))
        else:
            t2_venue_wp.append(0.5)

    matches = matches.copy()
    matches["t1_venue_win_pct"] = t1_venue_wp
    matches["t2_venue_win_pct"] = t2_venue_wp
    matches["venue_advantage"]  = np.array(t1_venue_wp) - np.array(t2_venue_wp)
    return matches


def _days_rest(matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each match compute days since last match for both teams.
    More rest = fresher team, especially in a 74-match IPL season.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    matches["date_dt"] = pd.to_datetime(matches["date"])

    t1_rest, t2_rest = [], []

    for idx, row in matches.iterrows():
        prior = matches.iloc[:idx]
        t1, t2 = row["team1"], row["team2"]
        cur_date = row["date_dt"]

        # Last match for team1
        t1_prior = prior[(prior["team1"] == t1) | (prior["team2"] == t1)]
        if len(t1_prior) > 0:
            last = pd.to_datetime(t1_prior["date"].iloc[-1])
            t1_rest.append((cur_date - last).days)
        else:
            t1_rest.append(7)  # default: 1 week

        # Last match for team2
        t2_prior = prior[(prior["team1"] == t2) | (prior["team2"] == t2)]
        if len(t2_prior) > 0:
            last = pd.to_datetime(t2_prior["date"].iloc[-1])
            t2_rest.append((cur_date - last).days)
        else:
            t2_rest.append(7)

    matches = matches.copy()
    matches["t1_days_rest"]  = t1_rest
    matches["t2_days_rest"]  = t2_rest
    matches["rest_advantage"] = np.array(t1_rest) - np.array(t2_rest)
    return matches


def _team_strength_rolling(conn: sqlite3.Connection, matches: pd.DataFrame,
                           window: int = 5) -> pd.DataFrame:
    """
    For each match, compute rolling last-`window` batting/bowling strength
    for both teams using only matches BEFORE this date.

    Features added:
        t1_avg_score_last5      team1 avg runs scored  (last 5 matches)
        t1_avg_conceded_last5   team1 avg runs conceded
        t1_avg_wickets_last5    team1 avg wickets taken per match
        t2_avg_score_last5
        t2_avg_conceded_last5
        t2_avg_wickets_last5
        score_diff_last5        t1_avg_score - t2_avg_score  (batting edge)
        bowling_diff_last5      t2_avg_conceded - t1_avg_conceded (bowling edge)
    """
    # ── Per-match innings totals from deliveries ──
    raw = pd.read_sql_query("""
        SELECT
            m.match_id, m.date, m.team1, m.team2,
            m.toss_winner, m.toss_decision,
            SUM(CASE WHEN d.inning=1 THEN d.runs_total ELSE 0 END)  AS inn1_runs,
            SUM(CASE WHEN d.inning=2 THEN d.runs_total ELSE 0 END)  AS inn2_runs,
            SUM(CASE WHEN d.inning=1 AND d.is_wicket=1 THEN 1 ELSE 0 END) AS inn1_wkts,
            SUM(CASE WHEN d.inning=2 AND d.is_wicket=1 THEN 1 ELSE 0 END) AS inn2_wkts,
            SUM(CASE WHEN d.inning=1 AND d.extras_type NOT IN ('wide','noball')
                     THEN 1 ELSE 0 END) AS inn1_balls,
            SUM(CASE WHEN d.inning=2 AND d.extras_type NOT IN ('wide','noball')
                     THEN 1 ELSE 0 END) AS inn2_balls
        FROM matches m
        JOIN deliveries d ON d.match_id = m.match_id
        WHERE m.source = 'ipl' AND m.winner IS NOT NULL AND d.inning <= 2
        GROUP BY m.match_id
    """, conn)

    # Determine which team batted in which inning using toss
    # batting_first = toss_winner if decision=bat, else the other team
    raw["batting_first"] = raw.apply(
        lambda r: r["team1"] if (
            (r["toss_winner"] == r["team1"] and r["toss_decision"] == "bat") or
            (r["toss_winner"] == r["team2"] and r["toss_decision"] == "field")
        ) else r["team2"], axis=1
    )

    # Build flat records: (date, match_id, team, runs_scored, runs_conceded, wickets_taken)
    records = []
    for _, r in raw.iterrows():
        bf = r["batting_first"]
        bs = r["team2"] if bf == r["team1"] else r["team1"]   # batting second

        records.append({
            "date": r["date"], "match_id": r["match_id"],
            "team": bf,
            "runs_scored":    r["inn1_runs"],
            "runs_conceded":  r["inn2_runs"],
            "wickets_taken":  r["inn2_wkts"],
        })
        records.append({
            "date": r["date"], "match_id": r["match_id"],
            "team": bs,
            "runs_scored":    r["inn2_runs"],
            "runs_conceded":  r["inn1_runs"],
            "wickets_taken":  r["inn1_wkts"],
        })

    rec_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    def get_strength(team, before_date):
        hist = rec_df[(rec_df["team"] == team) & (rec_df["date"] < before_date)].tail(window)
        if len(hist) == 0:
            return 155.0, 155.0, 6.0   # IPL average defaults
        return (
            round(hist["runs_scored"].mean(),   2),
            round(hist["runs_conceded"].mean(), 2),
            round(hist["wickets_taken"].mean(), 2),
        )

    matches = matches.sort_values("date").reset_index(drop=True)
    t1_score, t1_conceded, t1_wkts = [], [], []
    t2_score, t2_conceded, t2_wkts = [], [], []

    for _, row in matches.iterrows():
        s1, c1, w1 = get_strength(row["team1"], row["date"])
        s2, c2, w2 = get_strength(row["team2"], row["date"])
        t1_score.append(s1);    t1_conceded.append(c1);   t1_wkts.append(w1)
        t2_score.append(s2);    t2_conceded.append(c2);   t2_wkts.append(w2)

    matches = matches.copy()
    matches[f"t1_avg_score_last{window}"]     = t1_score
    matches[f"t1_avg_conceded_last{window}"]  = t1_conceded
    matches[f"t1_avg_wickets_last{window}"]   = t1_wkts
    matches[f"t2_avg_score_last{window}"]     = t2_score
    matches[f"t2_avg_conceded_last{window}"]  = t2_conceded
    matches[f"t2_avg_wickets_last{window}"]   = t2_wkts
    # Differential: positive = team1 bats/bowls better
    matches["score_diff_last5"]   = np.array(t1_score)    - np.array(t2_score)
    matches["bowling_diff_last5"] = np.array(t2_conceded) - np.array(t1_conceded)
    return matches


def build_match_features(conn: sqlite3.Connection) -> pd.DataFrame:
    log.info("Building match-level feature dataset (time-aware rolling features)...")

    matches = pd.read_sql_query("""
        SELECT match_id, source, season, date, venue, city,
               team1, team2, toss_winner, toss_decision, winner,
               win_by_runs, win_by_wickets, match_number
        FROM matches
        WHERE winner IS NOT NULL AND source = 'ipl'
        ORDER BY date
    """, conn)

    # ── Noise removal: skip matches with no normal result ──
    # Rain-affected / super-over matches distort outcomes
    before = len(matches)
    matches = matches[
        (matches["win_by_runs"].notna() | matches["win_by_wickets"].notna())
    ].copy()
    log.info(f"  Total matches: {len(matches):,}  (removed {before - len(matches)} no-result/DL matches)")

    # ── Rolling features (the key fix: time-aware) ──
    log.info("  Computing rolling team win rates...")
    matches = _rolling_team_stats(matches, window=10)

    log.info("  Computing rolling H2H records...")
    matches = _rolling_h2h(matches)

    log.info("  Computing venue/toss stats...")
    matches = _venue_toss_stats(matches)

    log.info("  Computing venue×team win rates...")
    matches = _venue_team_stats(matches)

    log.info("  Computing days rest between matches...")
    matches = _days_rest(matches)

    log.info("  Computing team batting/bowling strength (last 5)...")
    matches = _team_strength_rolling(conn, matches, window=5)

    # ── Venue avg score (all-time is ok here as it's ground characteristic) ──
    venue_stats = pd.read_sql_query("""
        SELECT venue, source,
               avg_first_innings AS venue_avg_score,
               chasing_win_pct   AS venue_chase_pct
        FROM venue_pitch_profiles
    """, conn)
    venue_stats = venue_stats.groupby(["venue","source"]).mean(numeric_only=True).reset_index()
    matches = matches.merge(venue_stats, on=["venue","source"], how="left")

    # ── Pitch / dew (ground characteristics — static) ──
    extra_venue = pd.read_sql_query("""
        SELECT venue, source,
               pitch_type, dew_factor
        FROM venue_pitch_profiles
        LIMIT 1000
    """, conn)
    extra_venue = extra_venue.groupby(["venue","source"]).agg(
        pitch_type=("pitch_type","first"),
        dew_factor=("dew_factor","first")
    ).reset_index()
    matches = matches.merge(extra_venue, on=["venue","source"], how="left")

    # ── Target ──
    matches["team1_won"] = (matches["winner"] == matches["team1"]).astype(int)

    # ── Encode ──
    matches["toss_decision_bat"]    = (matches["toss_decision"] == "bat").astype(int)
    matches["toss_winner_is_team1"] = (matches["toss_winner"]   == matches["team1"]).astype(int)

    pitch_map = {"batting_paradise":4,"batting_friendly":3,"balanced":2,
                 "bowling_friendly":1,"bowlers_paradise":0}
    dew_map   = {"high_dew_impact":2,"moderate_dew":1,"minimal_dew":0}

    if "pitch_type" in matches.columns:
        matches["pitch_score"] = matches["pitch_type"].map(pitch_map).fillna(2)
    else:
        matches["pitch_score"] = 2

    if "dew_factor" in matches.columns:
        matches["dew_score"] = matches["dew_factor"].map(dew_map).fillna(1)
    else:
        matches["dew_score"] = 1

    # ── Differential features (signal boost) ──
    matches["momentum_diff"]   = matches["t1_roll_win_pct"] - matches["t2_roll_win_pct"]
    matches["streak_diff"]     = matches["t1_roll_streak"]  - matches["t2_roll_streak"]
    matches["h2h_advantage"]   = matches["h2h_roll_win_pct"] - 0.5   # positive = team1 dominates

    # ── Night match (proxy: double-headers have afternoon + evening game) ──
    match_counts = matches.groupby("date")["match_id"].transform("count")
    matches["is_double_header"] = (match_counts == 2).astype(int)
    matches["is_evening"] = 1  # default: all evening (most IPL matches are 7:30 PM IST)
    # First match of a double-header day = afternoon (3:30 PM)
    matches = matches.sort_values(["date","match_id"]).reset_index(drop=True)
    first_of_day = matches.groupby("date").cumcount() == 0
    matches.loc[(matches["is_double_header"] == 1) & first_of_day, "is_evening"] = 0

    # ── Previous match margin (rolling — NO leakage) ──
    # For each match compute the normalised margin of EACH TEAM's last match before this date.
    # Positive = won that match, negative = lost.
    def _norm_margin(row):
        """Return normalised margin for the winner of this row (from matches table)."""
        if pd.notna(row["win_by_runs"]) and row["win_by_runs"] > 0:
            return float(row["win_by_runs"]) / 100.0
        if pd.notna(row["win_by_wickets"]) and row["win_by_wickets"] > 0:
            return float(row["win_by_wickets"]) / 10.0
        return 0.0

    # Build flat margin history: (date, team, signed_margin)
    margin_records = []
    for _, r in matches.iterrows():
        m = _norm_margin(r)
        winner = r["winner"]
        loser  = r["team2"] if winner == r["team1"] else r["team1"]
        margin_records.append({"date": r["date"], "team": winner, "margin":  m})
        margin_records.append({"date": r["date"], "team": loser,  "margin": -m})

    margin_df = pd.DataFrame(margin_records).sort_values("date").reset_index(drop=True)

    def get_prev_margin(team, before_date):
        hist = margin_df[(margin_df["team"] == team) & (margin_df["date"] < before_date)]
        return float(hist["margin"].iloc[-1]) if len(hist) > 0 else 0.0

    t1_prev, t2_prev = [], []
    for _, row in matches.iterrows():
        t1_prev.append(get_prev_margin(row["team1"], row["date"]))
        t2_prev.append(get_prev_margin(row["team2"], row["date"]))

    matches["t1_prev_margin"] = t1_prev
    matches["t2_prev_margin"] = t2_prev

    # ── Home ground advantage ──
    # A team playing at their home ground has an edge
    home_grounds = {
        "Mumbai Indians":          ["Wankhede Stadium, Mumbai", "Wankhede Stadium"],
        "Chennai Super Kings":     ["MA Chidambaram Stadium, Chepauk", "MA Chidambaram Stadium"],
        "Royal Challengers Bengaluru": ["M Chinnaswamy Stadium", "M. Chinnaswamy Stadium"],
        "Kolkata Knight Riders":   ["Eden Gardens", "Eden Gardens, Kolkata"],
        "Sunrisers Hyderabad":     ["Rajiv Gandhi International Stadium, Uppal",
                                    "Rajiv Gandhi International Stadium"],
        "Delhi Capitals":          ["Arun Jaitley Stadium", "Feroz Shah Kotla"],
        "Rajasthan Royals":        ["Sawai Mansingh Stadium"],
        "Punjab Kings":            ["Punjab Cricket Association IS Bindra Stadium, Mohali",
                                    "IS Bindra Stadium"],
        "Lucknow Super Giants":    ["BRSABV Ekana Cricket Stadium",
                                    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium"],
        "Gujarat Titans":          ["Narendra Modi Stadium, Ahmedabad",
                                    "Narendra Modi Stadium"],
    }
    venue_to_home = {}
    for team, grounds in home_grounds.items():
        for g in grounds:
            venue_to_home[g.lower().strip()] = team

    def is_home(team, venue):
        v = str(venue).lower().strip()
        return int(venue_to_home.get(v, "") == team)

    matches["t1_is_home"] = matches.apply(lambda r: is_home(r["team1"], r["venue"]), axis=1)
    matches["t2_is_home"] = matches.apply(lambda r: is_home(r["team2"], r["venue"]), axis=1)

    # ── Season number (captures IPL era effects) ──
    season_order = {s: i for i, s in enumerate(sorted(matches["season"].unique()))}
    matches["season_num"] = matches["season"].map(season_order).fillna(0)

    # ── Fill remaining NAs ──
    num_cols = ["t1_roll_win_pct","t2_roll_win_pct","t1_roll_streak","t2_roll_streak",
                "h2h_roll_win_pct","h2h_roll_matches","venue_bat_first_win_pct",
                "toss_bat_win_pct","venue_avg_score","venue_chase_pct",
                "momentum_diff","streak_diff","h2h_advantage",
                "t1_venue_win_pct","t2_venue_win_pct","venue_advantage",
                "t1_days_rest","t2_days_rest","rest_advantage",
                "t1_avg_score_last5","t1_avg_conceded_last5","t1_avg_wickets_last5",
                "t2_avg_score_last5","t2_avg_conceded_last5","t2_avg_wickets_last5",
                "score_diff_last5","bowling_diff_last5",
                "t1_prev_margin","t2_prev_margin"]
    for col in num_cols:
        if col in matches.columns:
            matches[col] = matches[col].fillna(matches[col].median())

    # ── Data augmentation: add mirrored rows (team1 ↔ team2) ──
    # The team1/team2 assignment in Cricsheet is arbitrary (not home/away).
    # By adding a flipped copy we double the dataset and force the model to
    # learn the *direction* of differential features, not just their magnitude.
    # match_group_id keeps original+flip in same CV fold (prevents leakage).
    log.info("  Augmenting dataset with mirrored team1↔team2 rows...")
    matches["match_group_id"] = np.arange(len(matches))  # for GroupKFold in train.py

    flipped = matches.copy()
    flipped["team1"]               = matches["team2"]
    flipped["team2"]               = matches["team1"]
    flipped["t1_roll_win_pct"]     = matches["t2_roll_win_pct"]
    flipped["t2_roll_win_pct"]     = matches["t1_roll_win_pct"]
    flipped["t1_roll_streak"]      = matches["t2_roll_streak"]
    flipped["t2_roll_streak"]      = matches["t1_roll_streak"]
    flipped["t1_roll_games"]       = matches["t2_roll_games"]
    flipped["t2_roll_games"]       = matches["t1_roll_games"]
    flipped["t1_is_home"]          = matches["t2_is_home"]
    flipped["t2_is_home"]          = matches["t1_is_home"]
    flipped["t1_venue_win_pct"]    = matches["t2_venue_win_pct"]
    flipped["t2_venue_win_pct"]    = matches["t1_venue_win_pct"]
    flipped["t1_days_rest"]        = matches["t2_days_rest"]
    flipped["t2_days_rest"]        = matches["t1_days_rest"]
    flipped["momentum_diff"]            = -matches["momentum_diff"]
    flipped["streak_diff"]              = -matches["streak_diff"]
    flipped["h2h_advantage"]            = -matches["h2h_advantage"]
    flipped["venue_advantage"]          = -matches["venue_advantage"]
    flipped["rest_advantage"]           = -matches["rest_advantage"]
    flipped["score_diff_last5"]         = -matches["score_diff_last5"]
    flipped["bowling_diff_last5"]       = -matches["bowling_diff_last5"]
    flipped["t1_avg_score_last5"]       = matches["t2_avg_score_last5"]
    flipped["t2_avg_score_last5"]       = matches["t1_avg_score_last5"]
    flipped["t1_avg_conceded_last5"]    = matches["t2_avg_conceded_last5"]
    flipped["t2_avg_conceded_last5"]    = matches["t1_avg_conceded_last5"]
    flipped["t1_avg_wickets_last5"]     = matches["t2_avg_wickets_last5"]
    flipped["t2_avg_wickets_last5"]     = matches["t1_avg_wickets_last5"]
    flipped["t1_prev_margin"]           = matches["t2_prev_margin"]
    flipped["t2_prev_margin"]           = matches["t1_prev_margin"]
    flipped["h2h_roll_win_pct"]         = 1 - matches["h2h_roll_win_pct"]
    flipped["toss_winner_is_team1"]     = 1 - matches["toss_winner_is_team1"]
    flipped["team1_won"]                = 1 - matches["team1_won"]
    flipped["is_augmented"]             = 1
    matches["is_augmented"]             = 0

    augmented = pd.concat([matches, flipped], ignore_index=True)

    log.info(f"  Match features dataset: {len(augmented):,} rows "
             f"({len(matches):,} real + {len(flipped):,} mirrored), "
             f"{len(augmented.columns)} columns")
    log.info(f"  Team1 win rate (should be ~0.50): {augmented['team1_won'].mean():.3f}")
    return augmented


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    ball_df  = build_ball_features(conn)
    match_df = build_match_features(conn)

    ball_path  = os.path.join(OUTPUT_DIR, "ball_features.parquet")
    match_path = os.path.join(OUTPUT_DIR, "match_features.parquet")

    ball_df.to_parquet(ball_path,  index=False)
    match_df.to_parquet(match_path, index=False)

    log.info(f"Saved ball features  → {ball_path}")
    log.info(f"Saved match features → {match_path}")
    conn.close()


if __name__ == "__main__":
    main()
