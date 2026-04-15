"""
ml_features_t20.py — Build match-level features from T20I data

Uses the same rolling/time-aware logic as ml_features.py so the
T20I rows can be safely concatenated with IPL rows for training.

National teams bring 5,146 more matches → gives XGBoost enough
samples to learn real patterns.

Run:
    python -m pipeline.ml_features_t20
"""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH    = os.getenv("DB_PATH", "data/processed/ipl.db")
OUTPUT_DIR = "data/processed/ml"


# ── Reuse rolling helpers from ml_features ────────────────────────────────

def _rolling_team_stats(matches: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True)
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
        win_pct  = hist["won"].mean()
        outcomes = hist["won"].tolist()
        last_val = outcomes[-1]
        streak   = 0
        for v in reversed(outcomes):
            if v == last_val: streak += 1
            else: break
        streak = streak if last_val == 1 else -streak
        return round(win_pct, 4), streak, len(hist)

    t1_wp, t1_st, t1_gm = [], [], []
    t2_wp, t2_st, t2_gm = [], [], []
    for _, row in matches.iterrows():
        wp1, s1, g1 = get_rolling(row["team1"], row["date"])
        wp2, s2, g2 = get_rolling(row["team2"], row["date"])
        t1_wp.append(wp1); t1_st.append(s1); t1_gm.append(g1)
        t2_wp.append(wp2); t2_st.append(s2); t2_gm.append(g2)

    matches = matches.copy()
    matches["t1_roll_win_pct"] = t1_wp
    matches["t1_roll_streak"]  = t1_st
    matches["t1_roll_games"]   = t1_gm
    matches["t2_roll_win_pct"] = t2_wp
    matches["t2_roll_streak"]  = t2_st
    matches["t2_roll_games"]   = t2_gm
    return matches


def _rolling_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True)
    h2h_wp, h2h_cnt = [], []
    for idx, row in matches.iterrows():
        t1, t2, date = row["team1"], row["team2"], row["date"]
        prior = matches.iloc[:idx]
        h2h = prior[
            ((prior["team1"] == t1) & (prior["team2"] == t2)) |
            ((prior["team1"] == t2) & (prior["team2"] == t1))
        ]
        if len(h2h) == 0:
            h2h_wp.append(0.5); h2h_cnt.append(0); continue
        t1_wins = sum(1 for _, hr in h2h.iterrows() if hr["winner"] == t1)
        h2h_wp.append(round(t1_wins / len(h2h), 4))
        h2h_cnt.append(len(h2h))
    matches = matches.copy()
    matches["h2h_roll_win_pct"] = h2h_wp
    matches["h2h_roll_matches"] = h2h_cnt
    return matches


def _venue_toss_stats(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True)
    vbf_wp, toss_wp = [], []
    for idx, row in matches.iterrows():
        prior = matches.iloc[:idx]
        if len(prior) == 0:
            vbf_wp.append(0.5); toss_wp.append(0.5); continue
        v_prior = prior[prior["venue"] == row["venue"]]
        if len(v_prior) >= 5:
            bf_wins = sum(
                1 for _, vr in v_prior.iterrows()
                if (vr["toss_decision"] == "bat"   and vr["winner"] == vr["toss_winner"]) or
                   (vr["toss_decision"] == "field" and vr["winner"] != vr["toss_winner"])
            )
            vbf_wp.append(round(bf_wins / len(v_prior), 4))
        else:
            vbf_wp.append(0.5)
        if len(prior) >= 10:
            bat_first = prior[prior["toss_decision"] == "bat"]
            tw = (bat_first["winner"] == bat_first["toss_winner"]).mean() if len(bat_first) > 0 else 0.5
            toss_wp.append(round(tw, 4))
        else:
            toss_wp.append(0.5)
    matches = matches.copy()
    matches["venue_bat_first_win_pct"] = vbf_wp
    matches["toss_bat_win_pct"]        = toss_wp
    return matches


def _venue_team_stats(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True)
    t1_vwp, t2_vwp = [], []
    for idx, row in matches.iterrows():
        prior  = matches.iloc[:idx]
        t1, t2 = row["team1"], row["team2"]
        venue  = row["venue"]
        for team, lst in [(t1, t1_vwp), (t2, t2_vwp)]:
            at = prior[((prior["team1"] == team) | (prior["team2"] == team)) & (prior["venue"] == venue)]
            if len(at) >= 3:
                wins = (((at["team1"] == team) & (at["winner"] == team)).sum() +
                        ((at["team2"] == team) & (at["winner"] == team)).sum())
                lst.append(round(wins / len(at), 4))
            else:
                lst.append(0.5)
    matches = matches.copy()
    matches["t1_venue_win_pct"] = t1_vwp
    matches["t2_venue_win_pct"] = t2_vwp
    matches["venue_advantage"]  = np.array(t1_vwp) - np.array(t2_vwp)
    return matches


def _days_rest(matches: pd.DataFrame) -> pd.DataFrame:
    matches  = matches.sort_values("date").reset_index(drop=True)
    matches["date_dt"] = pd.to_datetime(matches["date"])
    t1_rest, t2_rest = [], []
    for idx, row in matches.iterrows():
        prior = matches.iloc[:idx]
        cur   = row["date_dt"]
        for team, lst in [(row["team1"], t1_rest), (row["team2"], t2_rest)]:
            tp = prior[(prior["team1"] == team) | (prior["team2"] == team)]
            if len(tp) > 0:
                last = pd.to_datetime(tp["date"].iloc[-1])
                lst.append((cur - last).days)
            else:
                lst.append(7)
    matches = matches.copy()
    matches["t1_days_rest"]  = t1_rest
    matches["t2_days_rest"]  = t2_rest
    matches["rest_advantage"] = np.array(t1_rest) - np.array(t2_rest)
    return matches


def _team_strength_rolling(conn, matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    raw = pd.read_sql_query("""
        SELECT m.match_id, m.date, m.team1, m.team2,
               m.toss_winner, m.toss_decision,
               SUM(CASE WHEN d.inning=1 THEN d.runs_total ELSE 0 END) AS inn1_runs,
               SUM(CASE WHEN d.inning=2 THEN d.runs_total ELSE 0 END) AS inn2_runs,
               SUM(CASE WHEN d.inning=1 AND d.is_wicket=1 THEN 1 ELSE 0 END) AS inn2_wkts,
               SUM(CASE WHEN d.inning=2 AND d.is_wicket=1 THEN 1 ELSE 0 END) AS inn1_wkts
        FROM matches m
        JOIN deliveries d ON d.match_id = m.match_id
        WHERE m.source = 't20' AND m.winner IS NOT NULL AND d.inning <= 2
        GROUP BY m.match_id
    """, conn)

    raw["batting_first"] = raw.apply(
        lambda r: r["team1"] if (
            (r["toss_winner"] == r["team1"] and r["toss_decision"] == "bat") or
            (r["toss_winner"] == r["team2"] and r["toss_decision"] == "field")
        ) else r["team2"], axis=1
    )
    records = []
    for _, r in raw.iterrows():
        bf = r["batting_first"]
        bs = r["team2"] if bf == r["team1"] else r["team1"]
        records.append({"date": r["date"], "team": bf,
                        "runs_scored": r["inn1_runs"], "runs_conceded": r["inn2_runs"],
                        "wickets_taken": r["inn2_wkts"]})
        records.append({"date": r["date"], "team": bs,
                        "runs_scored": r["inn2_runs"], "runs_conceded": r["inn1_runs"],
                        "wickets_taken": r["inn1_wkts"]})
    rec_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    def get_str(team, before):
        hist = rec_df[(rec_df["team"] == team) & (rec_df["date"] < before)].tail(window)
        if len(hist) == 0:
            return 150.0, 150.0, 6.0   # T20I averages
        return hist["runs_scored"].mean(), hist["runs_conceded"].mean(), hist["wickets_taken"].mean()

    matches = matches.sort_values("date").reset_index(drop=True)
    t1s, t1c, t1w = [], [], []
    t2s, t2c, t2w = [], [], []
    for _, row in matches.iterrows():
        s1, c1, w1 = get_str(row["team1"], row["date"])
        s2, c2, w2 = get_str(row["team2"], row["date"])
        t1s.append(s1); t1c.append(c1); t1w.append(w1)
        t2s.append(s2); t2c.append(c2); t2w.append(w2)

    matches = matches.copy()
    matches["t1_avg_score_last5"]    = t1s
    matches["t1_avg_conceded_last5"] = t1c
    matches["t1_avg_wickets_last5"]  = t1w
    matches["t2_avg_score_last5"]    = t2s
    matches["t2_avg_conceded_last5"] = t2c
    matches["t2_avg_wickets_last5"]  = t2w
    matches["score_diff_last5"]      = np.array(t1s) - np.array(t2s)
    matches["bowling_diff_last5"]    = np.array(t2c) - np.array(t1c)
    return matches


def _prev_margin(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True)

    def norm(row):
        if pd.notna(row["win_by_runs"])    and row["win_by_runs"]    > 0: return row["win_by_runs"]    / 100.0
        if pd.notna(row["win_by_wickets"]) and row["win_by_wickets"] > 0: return row["win_by_wickets"] / 10.0
        return 0.0

    margin_records = []
    for _, r in matches.iterrows():
        m      = norm(r)
        winner = r["winner"]
        loser  = r["team2"] if winner == r["team1"] else r["team1"]
        margin_records.append({"date": r["date"], "team": winner, "margin":  m})
        margin_records.append({"date": r["date"], "team": loser,  "margin": -m})

    mdf = pd.DataFrame(margin_records).sort_values("date").reset_index(drop=True)

    def get_prev(team, before):
        h = mdf[(mdf["team"] == team) & (mdf["date"] < before)]
        return float(h["margin"].iloc[-1]) if len(h) > 0 else 0.0

    t1p, t2p = [], []
    for _, row in matches.iterrows():
        t1p.append(get_prev(row["team1"], row["date"]))
        t2p.append(get_prev(row["team2"], row["date"]))
    matches = matches.copy()
    matches["t1_prev_margin"] = t1p
    matches["t2_prev_margin"] = t2p
    return matches


def build_t20_match_features(conn: sqlite3.Connection) -> pd.DataFrame:
    log.info("Building T20I match features (rolling, time-aware)...")

    matches = pd.read_sql_query("""
        SELECT match_id, source, season, date, venue,
               team1, team2, toss_winner, toss_decision, winner,
               win_by_runs, win_by_wickets, match_number
        FROM matches
        WHERE winner IS NOT NULL AND source = 't20'
          AND (win_by_runs IS NOT NULL OR win_by_wickets IS NOT NULL)
        ORDER BY date
    """, conn)
    log.info(f"  T20I matches: {len(matches):,}")

    log.info("  Rolling team win rates...")
    matches = _rolling_team_stats(matches, window=10)
    log.info("  Rolling H2H...")
    matches = _rolling_h2h(matches)
    log.info("  Venue/toss stats...")
    matches = _venue_toss_stats(matches)
    log.info("  Venue×team stats...")
    matches = _venue_team_stats(matches)
    log.info("  Days rest...")
    matches = _days_rest(matches)
    log.info("  Team batting/bowling strength...")
    matches = _team_strength_rolling(conn, matches, window=5)
    log.info("  Previous match margin...")
    matches = _prev_margin(matches)

    # Venue avg score from DB
    venue_stats = pd.read_sql_query("""
        SELECT venue, source,
               avg_first_innings AS venue_avg_score,
               chasing_win_pct   AS venue_chase_pct
        FROM venue_pitch_profiles WHERE source='t20'
    """, conn)
    venue_stats = venue_stats.groupby(["venue","source"]).mean(numeric_only=True).reset_index()
    matches = matches.merge(venue_stats, on=["venue","source"], how="left")

    # Encode
    matches["team1_won"]           = (matches["winner"] == matches["team1"]).astype(int)
    matches["toss_decision_bat"]    = (matches["toss_decision"] == "bat").astype(int)
    matches["toss_winner_is_team1"] = (matches["toss_winner"]   == matches["team1"]).astype(int)
    matches["momentum_diff"]        = matches["t1_roll_win_pct"] - matches["t2_roll_win_pct"]
    matches["streak_diff"]          = matches["t1_roll_streak"]  - matches["t2_roll_streak"]
    matches["h2h_advantage"]        = matches["h2h_roll_win_pct"] - 0.5

    # T20I: no home ground concept → set to 0
    matches["t1_is_home"] = 0
    matches["t2_is_home"] = 0
    matches["dew_score"]  = 1   # unknown → neutral
    matches["pitch_score"]= 2   # unknown → balanced
    matches["is_evening"] = 1   # assume evening

    # Season number
    season_order = {s: i for i, s in enumerate(sorted(matches["season"].unique()))}
    matches["season_num"] = matches["season"].map(season_order).fillna(0)

    # Night match
    match_counts = matches.groupby("date")["match_id"].transform("count")
    matches["is_double_header"] = (match_counts == 2).astype(int)
    matches = matches.sort_values(["date","match_id"]).reset_index(drop=True)
    first_of_day = matches.groupby("date").cumcount() == 0
    matches.loc[(matches["is_double_header"] == 1) & first_of_day, "is_evening"] = 0

    # Fill NAs
    num_cols = ["t1_roll_win_pct","t2_roll_win_pct","h2h_roll_win_pct","h2h_roll_matches",
                "venue_bat_first_win_pct","toss_bat_win_pct","venue_avg_score","venue_chase_pct",
                "momentum_diff","streak_diff","h2h_advantage","venue_advantage",
                "rest_advantage","score_diff_last5","bowling_diff_last5",
                "t1_avg_score_last5","t1_avg_conceded_last5","t1_avg_wickets_last5",
                "t2_avg_score_last5","t2_avg_conceded_last5","t2_avg_wickets_last5",
                "t1_prev_margin","t2_prev_margin"]
    for col in num_cols:
        if col in matches.columns:
            matches[col] = matches[col].fillna(matches[col].median())

    # Augmentation (flip team1 ↔ team2)
    log.info("  Augmenting with mirrored rows...")
    matches["match_group_id"] = np.arange(len(matches))
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
    flipped["t1_avg_score_last5"]  = matches["t2_avg_score_last5"]
    flipped["t2_avg_score_last5"]  = matches["t1_avg_score_last5"]
    flipped["t1_avg_conceded_last5"]= matches["t2_avg_conceded_last5"]
    flipped["t2_avg_conceded_last5"]= matches["t1_avg_conceded_last5"]
    flipped["t1_avg_wickets_last5"] = matches["t2_avg_wickets_last5"]
    flipped["t2_avg_wickets_last5"] = matches["t1_avg_wickets_last5"]
    flipped["t1_prev_margin"]      = matches["t2_prev_margin"]
    flipped["t2_prev_margin"]      = matches["t1_prev_margin"]
    flipped["momentum_diff"]       = -matches["momentum_diff"]
    flipped["streak_diff"]         = -matches["streak_diff"]
    flipped["h2h_advantage"]       = -matches["h2h_advantage"]
    flipped["venue_advantage"]     = -matches["venue_advantage"]
    flipped["rest_advantage"]      = -matches["rest_advantage"]
    flipped["score_diff_last5"]    = -matches["score_diff_last5"]
    flipped["bowling_diff_last5"]  = -matches["bowling_diff_last5"]
    flipped["h2h_roll_win_pct"]    = 1 - matches["h2h_roll_win_pct"]
    flipped["toss_winner_is_team1"]= 1 - matches["toss_winner_is_team1"]
    flipped["team1_won"]           = 1 - matches["team1_won"]
    flipped["is_augmented"]        = 1
    matches["is_augmented"]        = 0

    out = pd.concat([matches, flipped], ignore_index=True)
    log.info(f"  T20I dataset: {len(out):,} rows  ({len(matches):,} + {len(flipped):,} mirrored)")
    log.info(f"  Win rate (should be ~0.50): {out['team1_won'].mean():.3f}")
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df   = build_t20_match_features(conn)
    path = os.path.join(OUTPUT_DIR, "match_features_t20.parquet")
    df.to_parquet(path, index=False)
    log.info(f"Saved T20I features → {path}")
    conn.close()


if __name__ == "__main__":
    main()
