"""
advanced_features.py — Build advanced analytical tables

Tables built:
  1. batter_vs_bowler        — ball-by-ball H2H matchup stats
  2. impact_player_scores    — composite match-winning ability score
  3. team_chasing_profiles   — chasing vs defending win rates
  4. venue_pitch_profiles    — pitch tendency, dew factor, par scores by over
  5. player_streak_form      — last 5 match rolling form per player
  6. team_momentum           — current win/loss streak + NRR proxy
  7. death_over_specialists  — best/worst performers in overs 16-20
  8. powerplay_specialists   — best/worst in overs 1-6
  9. pressure_performance    — win% in close matches (<10 runs / <2 wkts margin)
 10. player_vs_team          — player batting/bowling stats vs specific teams

Run:
    python pipeline/advanced_features.py
"""

import os
import sqlite3
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/processed/ipl.db")


def run(conn, label, sql):
    log.info(f"Building: {label}")
    conn.executescript(sql)
    conn.commit()


# ── 1. Batter vs Bowler H2H ────────────────────────────────────────────────
def build_batter_vs_bowler(conn):
    run(conn, "batter_vs_bowler", """
        DROP TABLE IF EXISTS batter_vs_bowler;
        CREATE TABLE batter_vs_bowler AS
        SELECT
            d.batter,
            d.bowler,
            m.source,
            COUNT(*)                                        AS balls,
            SUM(d.runs_batter)                              AS runs,
            SUM(CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END) AS legal_balls,
            SUM(CASE WHEN d.is_wicket = 1
                AND d.wicket_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                THEN 1 ELSE 0 END)                          AS dismissals,
            SUM(CASE WHEN d.runs_batter = 0
                AND d.extras_type NOT IN ('wide','noball')
                THEN 1 ELSE 0 END)                          AS dot_balls,
            SUM(CASE WHEN d.runs_batter = 4 THEN 1 ELSE 0 END) AS fours,
            SUM(CASE WHEN d.runs_batter = 6 THEN 1 ELSE 0 END) AS sixes,
            ROUND(
                CAST(SUM(d.runs_batter) AS REAL)
                / NULLIF(SUM(CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END), 0) * 100, 2
            )                                               AS strike_rate,
            ROUND(
                CAST(SUM(d.runs_batter) AS REAL)
                / NULLIF(SUM(CASE WHEN d.is_wicket = 1
                    AND d.wicket_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                    THEN 1 ELSE 0 END), 0), 2
            )                                               AS batting_avg,
            ROUND(
                CAST(SUM(CASE WHEN d.runs_batter = 0
                    AND d.extras_type NOT IN ('wide','noball')
                    THEN 1 ELSE 0 END) AS REAL)
                / NULLIF(SUM(CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END), 0) * 100, 2
            )                                               AS dot_ball_pct,
            -- dominance: >1 means batter dominates, <1 means bowler dominates
            ROUND(
                CAST(SUM(d.runs_batter) AS REAL)
                / NULLIF(SUM(CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END), 0)
                / 1.3, 2
            )                                               AS dominance_index
        FROM deliveries d
        JOIN matches m ON m.match_id = d.match_id
        GROUP BY d.batter, d.bowler, m.source
        HAVING COUNT(*) >= 6;   -- minimum 6 balls for meaningful stats

        CREATE INDEX IF NOT EXISTS idx_bvb_batter ON batter_vs_bowler(batter);
        CREATE INDEX IF NOT EXISTS idx_bvb_bowler ON batter_vs_bowler(bowler);
    """)


# ── 2. Impact Player Score ─────────────────────────────────────────────────
def build_impact_scores(conn):
    run(conn, "impact_player_scores", """
        DROP TABLE IF EXISTS impact_player_scores;
        CREATE TABLE impact_player_scores AS
        WITH batting AS (
            SELECT
                player, source, season,
                SUM(total_runs)     AS runs,
                AVG(strike_rate)    AS sr,
                AVG(batting_avg)    AS avg,
                SUM(sixes)          AS sixes,
                SUM(fifties + hundreds * 2) AS big_scores
            FROM player_batting_stats
            GROUP BY player, source, season
        ),
        bowling AS (
            SELECT
                player, source, season,
                SUM(wickets)        AS wickets,
                AVG(economy)        AS economy,
                AVG(bowling_avg)    AS bowl_avg,
                SUM(three_wicket_hauls) AS hauls
            FROM player_bowling_stats
            GROUP BY player, source, season
        ),
        bat_score AS (
            SELECT player, source, season,
                -- Batting impact: blend of run production, aggression, consistency
                ROUND(
                    (COALESCE(runs, 0) * 0.3) +
                    (COALESCE(sr, 0) * 0.3) +
                    (COALESCE(avg, 0) * 0.2) +
                    (COALESCE(sixes, 0) * 2.0) +
                    (COALESCE(big_scores, 0) * 5.0)
                , 2) AS bat_impact
            FROM batting
        ),
        bowl_score AS (
            SELECT player, source, season,
                -- Bowling impact: wickets + economy control + match-winning hauls
                ROUND(
                    (COALESCE(wickets, 0) * 8.0) +
                    (MAX(0, 10 - COALESCE(economy, 10)) * 5.0) +
                    (COALESCE(hauls, 0) * 15.0)
                , 2) AS bowl_impact
            FROM bowling
        )
        SELECT
            COALESCE(b.player, w.player)    AS player,
            COALESCE(b.source, w.source)    AS source,
            COALESCE(b.season, w.season)    AS season,
            COALESCE(b.bat_impact, 0)       AS batting_impact,
            COALESCE(w.bowl_impact, 0)      AS bowling_impact,
            ROUND(
                COALESCE(b.bat_impact, 0) + COALESCE(w.bowl_impact, 0), 2
            )                               AS total_impact_score,
            CASE
                WHEN COALESCE(b.bat_impact,0) > 0 AND COALESCE(w.bowl_impact,0) > 0 THEN 'allrounder'
                WHEN COALESCE(b.bat_impact,0) > COALESCE(w.bowl_impact,0) THEN 'batter'
                ELSE 'bowler'
            END                             AS primary_role
        FROM bat_score b
        FULL OUTER JOIN bowl_score w
            ON b.player = w.player AND b.source = w.source AND b.season = w.season;

        CREATE INDEX IF NOT EXISTS idx_impact_player ON impact_player_scores(player);
        CREATE INDEX IF NOT EXISTS idx_impact_season ON impact_player_scores(season);
    """)


# ── 3. Team Chasing vs Defending Profiles ─────────────────────────────────
def build_chasing_profiles(conn):
    run(conn, "team_chasing_profiles", """
        DROP TABLE IF EXISTS team_chasing_profiles;
        CREATE TABLE team_chasing_profiles AS
        WITH match_roles AS (
            SELECT
                m.match_id,
                m.source,
                m.season,
                m.venue,
                m.winner,
                m.toss_winner,
                m.toss_decision,
                -- Team that batted first
                CASE WHEN m.toss_decision = 'bat' THEN m.toss_winner
                     ELSE CASE WHEN m.toss_winner = m.team1 THEN m.team2 ELSE m.team1 END
                END AS batting_first_team,
                -- Team that chased
                CASE WHEN m.toss_decision = 'field' THEN m.toss_winner
                     ELSE CASE WHEN m.toss_winner = m.team1 THEN m.team2 ELSE m.team1 END
                END AS chasing_team,
                m.win_by_runs,
                m.win_by_wickets
            FROM matches m
            WHERE m.winner IS NOT NULL AND m.toss_decision IS NOT NULL
        )
        SELECT
            team,
            source,
            season,
            role,
            COUNT(*)                                        AS matches,
            SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) AS wins,
            ROUND(
                CAST(SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) AS REAL)
                / NULLIF(COUNT(*), 0) * 100, 2
            )                                               AS win_pct,
            -- Average winning margin
            ROUND(AVG(CASE WHEN winner = team AND win_by_runs > 0 THEN win_by_runs END), 1) AS avg_win_margin_runs,
            ROUND(AVG(CASE WHEN winner = team AND win_by_wickets > 0 THEN win_by_wickets END), 1) AS avg_win_margin_wkts
        FROM (
            SELECT batting_first_team AS team, source, season, winner,
                   'batting_first' AS role, win_by_runs, win_by_wickets FROM match_roles
            UNION ALL
            SELECT chasing_team AS team, source, season, winner,
                   'chasing' AS role, win_by_runs, win_by_wickets FROM match_roles
        )
        GROUP BY team, source, season, role;

        CREATE INDEX IF NOT EXISTS idx_chase_team ON team_chasing_profiles(team, role);
    """)


# ── 4. Venue Pitch Profiles ────────────────────────────────────────────────
def build_venue_pitch_profiles(conn):
    run(conn, "venue_pitch_profiles", """
        DROP TABLE IF EXISTS venue_pitch_profiles;
        CREATE TABLE venue_pitch_profiles AS
        WITH
        -- Step 1: One row per innings per match (correct total scores)
        innings_scores AS (
            SELECT
                m.venue, m.source, m.match_id, d.inning,
                SUM(d.runs_total)  AS innings_total,
                SUM(d.is_wicket)   AS innings_wickets
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            GROUP BY m.venue, m.source, m.match_id, d.inning
        ),
        -- Step 2: Per-over runs using LEGAL balls only (exclude wides/noballs from RPO calc)
        over_legal AS (
            SELECT
                m.venue, m.source, d.inning, d.over_num,
                SUM(CASE WHEN d.extras_type NOT IN ('wide','noball') OR d.extras_type IS NULL
                    THEN d.runs_batter ELSE 0 END) AS legal_runs,
                COUNT(CASE WHEN d.extras_type NOT IN ('wide','noball') OR d.extras_type IS NULL
                    THEN 1 END)                     AS legal_balls,
                COUNT(DISTINCT m.match_id)          AS matches_in_over
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            GROUP BY m.venue, m.source, d.inning, d.over_num
        ),
        -- Step 3: Venue-level aggregates
        venue_agg AS (
            SELECT
                venue, source,
                COUNT(DISTINCT match_id)                                    AS total_matches,
                ROUND(AVG(CASE WHEN inning=1 THEN innings_total END), 1)    AS avg_first_innings,
                ROUND(AVG(CASE WHEN inning=2 THEN innings_total END), 1)    AS avg_second_innings,
                ROUND(MAX(CASE WHEN inning=1 THEN innings_total END), 0)    AS highest_first_innings,
                ROUND(MIN(CASE WHEN inning=1 THEN innings_total END), 0)    AS lowest_first_innings
            FROM innings_scores
            GROUP BY venue, source
        ),
        -- Step 4: Phase RPO (legal balls only)
        phase_rpo AS (
            SELECT
                venue, source,
                ROUND(
                    SUM(CASE WHEN over_num < 6 AND inning=1 THEN legal_runs ELSE 0 END) * 6.0 /
                    NULLIF(SUM(CASE WHEN over_num < 6 AND inning=1 THEN legal_balls ELSE 0 END), 0)
                , 2) AS avg_powerplay_rpo,
                ROUND(
                    SUM(CASE WHEN over_num >= 15 AND inning=1 THEN legal_runs ELSE 0 END) * 6.0 /
                    NULLIF(SUM(CASE WHEN over_num >= 15 AND inning=1 THEN legal_balls ELSE 0 END), 0)
                , 2) AS avg_death_rpo,
                ROUND(
                    SUM(CASE WHEN over_num BETWEEN 6 AND 14 AND inning=1 THEN legal_runs ELSE 0 END) * 6.0 /
                    NULLIF(SUM(CASE WHEN over_num BETWEEN 6 AND 14 AND inning=1 THEN legal_balls ELSE 0 END), 0)
                , 2) AS avg_middle_rpo
            FROM over_legal
            GROUP BY venue, source
        ),
        -- Step 5: Chasing win %
        chasing AS (
            SELECT venue, source, chasing_win_pct
            FROM venue_stats
        )
        SELECT
            v.venue,
            v.source,
            v.total_matches,
            v.avg_first_innings,
            v.avg_second_innings,
            v.highest_first_innings,
            v.lowest_first_innings,
            p.avg_powerplay_rpo,
            p.avg_middle_rpo,
            p.avg_death_rpo,
            COALESCE(c.chasing_win_pct, 50.0)       AS chasing_win_pct,
            -- Pitch type based on correct avg score
            CASE
                WHEN v.avg_first_innings > 175 THEN 'batting_paradise'
                WHEN v.avg_first_innings > 165 THEN 'batting_friendly'
                WHEN v.avg_first_innings > 150 THEN 'balanced'
                WHEN v.avg_first_innings > 135 THEN 'bowling_friendly'
                ELSE 'bowlers_paradise'
            END                                     AS pitch_type,
            -- Dew factor from chasing win % (high chasing win = dew likely at night)
            CASE
                WHEN COALESCE(c.chasing_win_pct, 50) > 58 THEN 'high_dew_impact'
                WHEN COALESCE(c.chasing_win_pct, 50) > 52 THEN 'moderate_dew'
                ELSE 'minimal_dew'
            END                                     AS dew_factor
        FROM venue_agg v
        LEFT JOIN phase_rpo p ON p.venue = v.venue AND p.source = v.source
        LEFT JOIN chasing   c ON c.venue = v.venue AND c.source = v.source
        WHERE v.total_matches >= 2;

        CREATE INDEX IF NOT EXISTS idx_vpp_venue ON venue_pitch_profiles(venue);
    """)


# ── 5. Player Rolling Form (last 5 matches) ───────────────────────────────
def build_player_rolling_form(conn):
    run(conn, "player_rolling_form", """
        DROP TABLE IF EXISTS player_rolling_form;
        CREATE TABLE player_rolling_form AS
        WITH batting_innings AS (
            SELECT
                d.batter                    AS player,
                m.match_id,
                m.date,
                m.season,
                m.source,
                SUM(d.runs_batter)          AS runs,
                SUM(CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END) AS balls,
                MAX(d.is_wicket)            AS got_out,
                SUM(CASE WHEN d.runs_batter=6 THEN 1 ELSE 0 END) AS sixes,
                ROW_NUMBER() OVER (
                    PARTITION BY d.batter, m.source
                    ORDER BY m.date DESC
                )                           AS match_rank
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            GROUP BY d.batter, m.match_id, m.date, m.season, m.source
        ),
        bowling_innings AS (
            SELECT
                d.bowler                    AS player,
                m.match_id,
                m.date,
                m.season,
                m.source,
                SUM(CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END) AS balls,
                SUM(d.runs_total) - SUM(d.runs_extras) AS runs_given,
                SUM(CASE WHEN d.is_wicket=1
                    AND d.wicket_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                    THEN 1 ELSE 0 END)      AS wickets,
                ROW_NUMBER() OVER (
                    PARTITION BY d.bowler, m.source
                    ORDER BY m.date DESC
                )                           AS match_rank
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            GROUP BY d.bowler, m.match_id, m.date, m.season, m.source
        ),
        bat_form AS (
            SELECT player, source,
                COUNT(*)                    AS recent_innings,
                SUM(runs)                   AS recent_runs,
                ROUND(AVG(runs), 1)         AS recent_avg,
                ROUND(
                    CAST(SUM(runs) AS REAL) / NULLIF(SUM(balls), 0) * 100, 1
                )                           AS recent_sr,
                SUM(sixes)                  AS recent_sixes,
                MAX(runs)                   AS recent_best,
                -- Form trend: difference between last 2 and previous 3
                ROUND(AVG(CASE WHEN match_rank <= 2 THEN runs ELSE NULL END) -
                      AVG(CASE WHEN match_rank > 2 THEN runs ELSE NULL END), 1) AS form_trend
            FROM batting_innings
            WHERE match_rank <= 5
            GROUP BY player, source
        ),
        bowl_form AS (
            SELECT player, source,
                COUNT(*)                    AS recent_games,
                SUM(wickets)                AS recent_wickets,
                ROUND(AVG(wickets), 1)      AS recent_wkts_per_game,
                ROUND(
                    CAST(SUM(runs_given) AS REAL) / NULLIF(SUM(balls), 0) * 6, 2
                )                           AS recent_economy,
                -- Form trend: economy improving or worsening
                ROUND(
                    AVG(CASE WHEN match_rank <= 2 THEN
                        CAST(runs_given AS REAL) / NULLIF(balls, 0) * 6 ELSE NULL END) -
                    AVG(CASE WHEN match_rank > 2 THEN
                        CAST(runs_given AS REAL) / NULLIF(balls, 0) * 6 ELSE NULL END), 2
                )                           AS bowling_form_trend
            FROM bowling_innings
            WHERE match_rank <= 5
            GROUP BY player, source
        )
        SELECT
            COALESCE(b.player, w.player)        AS player,
            COALESCE(b.source, w.source)        AS source,
            b.recent_innings,
            b.recent_runs,
            b.recent_avg,
            b.recent_sr,
            b.recent_sixes,
            b.recent_best,
            b.form_trend                        AS batting_form_trend,
            w.recent_games,
            w.recent_wickets,
            w.recent_wkts_per_game,
            w.recent_economy,
            w.bowling_form_trend,
            -- Overall form rating
            CASE
                WHEN b.recent_avg > 40 OR w.recent_wkts_per_game > 2 THEN 'red_hot'
                WHEN b.recent_avg > 25 OR w.recent_wkts_per_game > 1.2 THEN 'good_form'
                WHEN b.recent_avg > 15 OR w.recent_wkts_per_game > 0.8 THEN 'average_form'
                ELSE 'poor_form'
            END                                 AS form_rating
        FROM bat_form b
        FULL OUTER JOIN bowl_form w
            ON b.player = w.player AND b.source = w.source;

        CREATE INDEX IF NOT EXISTS idx_prf_player ON player_rolling_form(player);
    """)


# ── 6. Team Momentum ───────────────────────────────────────────────────────
def build_team_momentum(conn):
    run(conn, "team_momentum", """
        DROP TABLE IF EXISTS team_momentum;
        CREATE TABLE team_momentum AS
        WITH match_results AS (
            SELECT team, source, season, date, match_id, winner,
                   CASE WHEN winner = team THEN 1 ELSE 0 END AS won,
                   ROW_NUMBER() OVER (
                       PARTITION BY team, source, season ORDER BY date DESC
                   ) AS match_rank
            FROM (
                SELECT team1 AS team, source, season, date, match_id, winner FROM matches WHERE winner IS NOT NULL
                UNION ALL
                SELECT team2 AS team, source, season, date, match_id, winner FROM matches WHERE winner IS NOT NULL
            )
        )
        SELECT
            team, source, season,
            COUNT(*)                                AS recent_matches,
            SUM(won)                                AS recent_wins,
            ROUND(AVG(won) * 100, 1)                AS recent_win_pct,
            -- Current streak: consecutive W or L
            SUM(CASE WHEN match_rank = 1 THEN won ELSE 0 END) AS last_match_won,
            -- Streak calculation
            (
                SELECT COUNT(*) FROM match_results m2
                WHERE m2.team = match_results.team
                  AND m2.source = match_results.source
                  AND m2.season = match_results.season
                  AND m2.match_rank <= 5
                  AND m2.won = (SELECT won FROM match_results m3
                                WHERE m3.team = match_results.team
                                  AND m3.source = match_results.source
                                  AND m3.season = match_results.season
                                  AND m3.match_rank = 1)
                  AND m2.match_rank <= (
                      SELECT MIN(m4.match_rank) FROM match_results m4
                      WHERE m4.team = match_results.team
                        AND m4.source = match_results.source
                        AND m4.season = match_results.season
                        AND m4.won != (SELECT won FROM match_results m5
                                       WHERE m5.team = match_results.team
                                         AND m5.source = match_results.source
                                         AND m5.season = match_results.season
                                         AND m5.match_rank = 1)
                  )
            )                                       AS current_streak_length,
            -- Momentum score: weighted recent results (last=3x, 2nd=2x, 3rd=1x)
            ROUND(
                (SUM(CASE WHEN match_rank=1 THEN won*3.0 ELSE 0 END) +
                 SUM(CASE WHEN match_rank=2 THEN won*2.0 ELSE 0 END) +
                 SUM(CASE WHEN match_rank=3 THEN won*1.0 ELSE 0 END)) / 6.0 * 100, 1
            )                                       AS momentum_score
        FROM match_results
        WHERE match_rank <= 5
        GROUP BY team, source, season;

        CREATE INDEX IF NOT EXISTS idx_momentum_team ON team_momentum(team, season);
    """)


# ── 7. Death Over Specialists ──────────────────────────────────────────────
def build_death_specialists(conn):
    run(conn, "death_over_specialists", """
        DROP TABLE IF EXISTS death_over_specialists;
        CREATE TABLE death_over_specialists AS
        WITH death_bowling AS (
            SELECT
                d.bowler                    AS player,
                m.source, m.season,
                COUNT(*)                    AS balls,
                SUM(d.runs_total) - SUM(d.runs_extras) AS runs,
                SUM(CASE WHEN d.is_wicket=1
                    AND d.wicket_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                    THEN 1 ELSE 0 END)      AS wickets,
                SUM(CASE WHEN d.runs_batter=0
                    AND d.extras_type NOT IN ('wide','noball') THEN 1 ELSE 0 END) AS dots
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            WHERE d.over_num >= 15
            GROUP BY d.bowler, m.source, m.season
            HAVING COUNT(*) >= 24
        ),
        death_batting AS (
            SELECT
                d.batter                    AS player,
                m.source, m.season,
                SUM(d.runs_batter)          AS runs,
                COUNT(*)                    AS balls,
                SUM(CASE WHEN d.runs_batter=6 THEN 1 ELSE 0 END) AS sixes,
                SUM(CASE WHEN d.runs_batter=4 THEN 1 ELSE 0 END) AS fours
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            WHERE d.over_num >= 15
              AND d.extras_type NOT IN ('wide','noball')
            GROUP BY d.batter, m.source, m.season
            HAVING COUNT(*) >= 24
        )
        SELECT
            'bowling' AS role,
            player, source, season,
            balls,
            runs,
            wickets,
            ROUND(CAST(runs AS REAL)/NULLIF(balls,0)*6, 2) AS economy,
            ROUND(CAST(dots AS REAL)/NULLIF(balls,0)*100, 1) AS dot_pct,
            NULL AS sr,
            NULL AS sixes
        FROM death_bowling
        UNION ALL
        SELECT
            'batting' AS role,
            player, source, season,
            balls,
            runs,
            NULL AS wickets,
            NULL AS economy,
            NULL AS dot_pct,
            ROUND(CAST(runs AS REAL)/NULLIF(balls,0)*100, 1) AS sr,
            sixes
        FROM death_batting;

        CREATE INDEX IF NOT EXISTS idx_death_player ON death_over_specialists(player, role);
    """)


# ── 8. Powerplay Specialists ───────────────────────────────────────────────
def build_powerplay_specialists(conn):
    run(conn, "powerplay_specialists", """
        DROP TABLE IF EXISTS powerplay_specialists;
        CREATE TABLE powerplay_specialists AS
        WITH pp_bowling AS (
            SELECT
                d.bowler AS player, m.source, m.season,
                COUNT(*) AS balls,
                SUM(d.runs_total) - SUM(d.runs_extras) AS runs,
                SUM(CASE WHEN d.is_wicket=1
                    AND d.wicket_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                    THEN 1 ELSE 0 END) AS wickets
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            WHERE d.over_num < 6
            GROUP BY d.bowler, m.source, m.season
            HAVING COUNT(*) >= 18
        ),
        pp_batting AS (
            SELECT
                d.batter AS player, m.source, m.season,
                SUM(d.runs_batter) AS runs,
                COUNT(*) AS balls,
                SUM(CASE WHEN d.runs_batter=6 THEN 1 ELSE 0 END) AS sixes
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            WHERE d.over_num < 6
              AND d.extras_type NOT IN ('wide','noball')
            GROUP BY d.batter, m.source, m.season
            HAVING COUNT(*) >= 18
        )
        SELECT 'bowling' AS role, player, source, season,
               balls, runs, wickets,
               ROUND(CAST(runs AS REAL)/NULLIF(balls,0)*6, 2) AS economy,
               NULL AS sr, NULL AS sixes
        FROM pp_bowling
        UNION ALL
        SELECT 'batting' AS role, player, source, season,
               balls, runs, NULL AS wickets,
               NULL AS economy,
               ROUND(CAST(runs AS REAL)/NULLIF(balls,0)*100, 1) AS sr,
               sixes
        FROM pp_batting;

        CREATE INDEX IF NOT EXISTS idx_pp_player ON powerplay_specialists(player, role);
    """)


# ── 9. Pressure Performance ────────────────────────────────────────────────
def build_pressure_performance(conn):
    run(conn, "pressure_performance", """
        DROP TABLE IF EXISTS pressure_performance;
        CREATE TABLE pressure_performance AS
        WITH close_matches AS (
            SELECT match_id, source, season, winner, team1, team2,
                   win_by_runs, win_by_wickets
            FROM matches
            WHERE winner IS NOT NULL
              AND (win_by_runs BETWEEN 1 AND 15
                   OR win_by_wickets BETWEEN 1 AND 2)
        )
        SELECT
            team,
            source,
            season,
            COUNT(*)                                        AS close_matches,
            SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) AS close_wins,
            ROUND(
                CAST(SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) AS REAL)
                / NULLIF(COUNT(*), 0) * 100, 2
            )                                               AS close_win_pct,
            -- Clutch rating: above 50% = good under pressure
            CASE
                WHEN ROUND(CAST(SUM(CASE WHEN winner=team THEN 1 ELSE 0 END) AS REAL)
                    / NULLIF(COUNT(*),0)*100,2) > 60 THEN 'clutch'
                WHEN ROUND(CAST(SUM(CASE WHEN winner=team THEN 1 ELSE 0 END) AS REAL)
                    / NULLIF(COUNT(*),0)*100,2) > 45 THEN 'neutral'
                ELSE 'choker'
            END                                             AS clutch_rating
        FROM (
            SELECT team1 AS team, source, season, winner,
                   win_by_runs, win_by_wickets FROM close_matches
            UNION ALL
            SELECT team2 AS team, source, season, winner,
                   win_by_runs, win_by_wickets FROM close_matches
        )
        GROUP BY team, source, season
        HAVING COUNT(*) >= 3;

        CREATE INDEX IF NOT EXISTS idx_pressure_team ON pressure_performance(team);
    """)


# ── 10. Player vs Team ─────────────────────────────────────────────────────
def build_player_vs_team(conn):
    run(conn, "player_vs_team", """
        DROP TABLE IF EXISTS player_vs_team;
        CREATE TABLE player_vs_team AS
        WITH bat AS (
            SELECT
                d.batter                AS player,
                CASE WHEN m.team1 = (
                    SELECT team1 FROM matches WHERE match_id = d.match_id
                ) THEN m.team2 ELSE m.team1 END AS opposition,
                m.source,
                d.match_id,
                d.inning,
                SUM(d.runs_batter)      AS runs,
                SUM(CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END) AS balls,
                MAX(d.is_wicket)        AS got_out,
                SUM(CASE WHEN d.runs_batter=6 THEN 1 ELSE 0 END) AS sixes
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            GROUP BY d.batter, opposition, m.source, d.match_id, d.inning
        ),
        bowl AS (
            SELECT
                d.bowler                AS player,
                CASE WHEN m.team1 = (
                    SELECT team1 FROM matches WHERE match_id = d.match_id
                ) THEN m.team2 ELSE m.team1 END AS opposition,
                m.source,
                d.match_id,
                d.inning,
                SUM(CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END) AS balls,
                SUM(d.runs_total) - SUM(d.runs_extras) AS runs_given,
                SUM(CASE WHEN d.is_wicket=1
                    AND d.wicket_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                    THEN 1 ELSE 0 END)  AS wickets
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            GROUP BY d.bowler, opposition, m.source, d.match_id, d.inning
        )
        SELECT
            b.player, b.opposition, b.source,
            COUNT(*)                    AS innings,
            SUM(b.runs)                 AS runs,
            MAX(b.runs)                 AS highest,
            ROUND(AVG(b.runs), 1)       AS avg,
            ROUND(CAST(SUM(b.runs) AS REAL)/NULLIF(SUM(b.balls),0)*100, 1) AS sr,
            SUM(b.got_out)              AS dismissals,
            SUM(b.sixes)                AS sixes,
            NULL AS wickets, NULL AS economy
        FROM bat b
        GROUP BY b.player, b.opposition, b.source
        HAVING COUNT(*) >= 2
        UNION ALL
        SELECT
            w.player, w.opposition, w.source,
            COUNT(*)                    AS innings,
            NULL AS runs, NULL AS highest, NULL AS avg,
            NULL AS sr, NULL AS dismissals, NULL AS sixes,
            SUM(w.wickets)              AS wickets,
            ROUND(CAST(SUM(w.runs_given) AS REAL)/NULLIF(SUM(w.balls),0)*6, 2) AS economy
        FROM bowl w
        GROUP BY w.player, w.opposition, w.source
        HAVING COUNT(*) >= 2;

        CREATE INDEX IF NOT EXISTS idx_pvt_player ON player_vs_team(player, opposition);
    """)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    build_batter_vs_bowler(conn)
    build_impact_scores(conn)
    build_chasing_profiles(conn)
    build_venue_pitch_profiles(conn)
    build_player_rolling_form(conn)
    build_team_momentum(conn)
    build_death_specialists(conn)
    build_powerplay_specialists(conn)
    build_pressure_performance(conn)
    build_player_vs_team(conn)

    log.info("All advanced feature tables built successfully.")
    conn.close()


if __name__ == "__main__":
    main()
