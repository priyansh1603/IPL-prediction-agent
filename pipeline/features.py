"""
features.py — Build derived stats tables in SQLite

Creates materialized views/tables the agent can query instantly:
  - player_batting_stats    (overall + per season + per venue)
  - player_bowling_stats    (overall + per season + per venue)
  - team_stats              (win rates, h2h, venue records, toss impact)
  - venue_stats             (avg scores, chasing success, par scores by over)
  - phase_stats             (powerplay / middle / death — batting & bowling)
  - head_to_head            (team1 vs team2 breakdown)

Run:
    python pipeline/features.py
"""

import os
import sqlite3
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/processed/ipl.db")


def run(conn: sqlite3.Connection, label: str, sql: str):
    log.info(f"Building: {label}")
    conn.executescript(sql)
    conn.commit()


def build_player_batting(conn):
    run(conn, "player_batting_stats", """
        DROP TABLE IF EXISTS player_batting_stats;
        CREATE TABLE player_batting_stats AS
        WITH base AS (
            SELECT
                d.batter                                AS player,
                m.source,
                m.season,
                m.venue,
                CASE WHEN m.team1 = (
                    SELECT team FROM (
                        SELECT team1 AS team FROM matches WHERE match_id = d.match_id
                        UNION ALL
                        SELECT team2 AS team FROM matches WHERE match_id = d.match_id
                    ) WHERE team != m.toss_winner OR m.toss_winner IS NULL LIMIT 1
                ) THEN m.team2 ELSE m.team1 END         AS opposition,
                d.match_id,
                d.inning,
                d.runs_batter,
                d.is_wicket,
                CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END AS legal_ball
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
        ),
        innings_agg AS (
            SELECT
                player, source, season, venue, match_id, inning,
                SUM(runs_batter)                        AS runs,
                SUM(legal_ball)                         AS balls_faced,
                MAX(is_wicket)                          AS got_out,
                SUM(CASE WHEN runs_batter = 4 THEN 1 ELSE 0 END) AS fours,
                SUM(CASE WHEN runs_batter = 6 THEN 1 ELSE 0 END) AS sixes
            FROM base
            GROUP BY player, source, season, venue, match_id, inning
        )
        SELECT
            player,
            source,
            season,
            venue,
            COUNT(*)                                    AS innings,
            SUM(runs)                                   AS total_runs,
            MAX(runs)                                   AS highest_score,
            ROUND(AVG(runs), 2)                         AS avg_runs,
            ROUND(
                CAST(SUM(runs) AS REAL) / NULLIF(SUM(balls_faced), 0) * 100, 2
            )                                           AS strike_rate,
            SUM(got_out)                                AS dismissals,
            ROUND(
                CAST(SUM(runs) AS REAL) / NULLIF(SUM(got_out), 0), 2
            )                                           AS batting_avg,
            SUM(fours)                                  AS fours,
            SUM(sixes)                                  AS sixes,
            SUM(CASE WHEN runs >= 50 THEN 1 ELSE 0 END) AS fifties,
            SUM(CASE WHEN runs >= 100 THEN 1 ELSE 0 END) AS hundreds
        FROM innings_agg
        GROUP BY player, source, season, venue;

        CREATE INDEX IF NOT EXISTS idx_bat_player ON player_batting_stats(player);
        CREATE INDEX IF NOT EXISTS idx_bat_season ON player_batting_stats(season);
    """)


def build_player_bowling(conn):
    run(conn, "player_bowling_stats", """
        DROP TABLE IF EXISTS player_bowling_stats;
        CREATE TABLE player_bowling_stats AS
        WITH base AS (
            SELECT
                d.bowler                        AS player,
                m.source,
                m.season,
                m.venue,
                d.match_id,
                d.inning,
                d.runs_total,
                d.runs_extras,
                d.is_wicket,
                d.wicket_kind,
                CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END AS legal_ball
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
        ),
        innings_agg AS (
            SELECT
                player, source, season, venue, match_id, inning,
                SUM(legal_ball)                 AS balls_bowled,
                SUM(runs_total) - SUM(runs_extras) AS runs_conceded,
                SUM(CASE WHEN is_wicket = 1
                    AND wicket_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                    THEN 1 ELSE 0 END)           AS wickets
            FROM base
            GROUP BY player, source, season, venue, match_id, inning
        )
        SELECT
            player,
            source,
            season,
            venue,
            COUNT(*)                            AS innings_bowled,
            SUM(balls_bowled)                   AS balls,
            SUM(runs_conceded)                  AS runs_conceded,
            SUM(wickets)                        AS wickets,
            ROUND(
                CAST(SUM(runs_conceded) AS REAL) / NULLIF(SUM(balls_bowled), 0) * 6, 2
            )                                   AS economy,
            ROUND(
                CAST(SUM(runs_conceded) AS REAL) / NULLIF(SUM(wickets), 0), 2
            )                                   AS bowling_avg,
            ROUND(
                CAST(SUM(balls_bowled) AS REAL) / NULLIF(SUM(wickets), 0), 2
            )                                   AS bowling_sr,
            MAX(wickets)                        AS best_bowling,
            SUM(CASE WHEN wickets >= 3 THEN 1 ELSE 0 END) AS three_wicket_hauls,
            SUM(CASE WHEN wickets >= 5 THEN 1 ELSE 0 END) AS five_wicket_hauls
        FROM innings_agg
        GROUP BY player, source, season, venue;

        CREATE INDEX IF NOT EXISTS idx_bowl_player ON player_bowling_stats(player);
        CREATE INDEX IF NOT EXISTS idx_bowl_season ON player_bowling_stats(season);
    """)


def build_team_stats(conn):
    run(conn, "team_stats", """
        DROP TABLE IF EXISTS team_stats;
        CREATE TABLE team_stats AS
        SELECT
            team,
            source,
            season,
            COUNT(*)                                        AS matches_played,
            SUM(CASE WHEN winner = team THEN 1 ELSE 0 END)  AS wins,
            SUM(CASE WHEN winner != team AND winner IS NOT NULL THEN 1 ELSE 0 END) AS losses,
            SUM(CASE WHEN winner IS NULL THEN 1 ELSE 0 END) AS no_result,
            ROUND(
                CAST(SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) AS REAL)
                / NULLIF(SUM(CASE WHEN winner IS NOT NULL THEN 1 ELSE 0 END), 0) * 100, 2
            )                                               AS win_pct
        FROM (
            SELECT team1 AS team, source, season, winner FROM matches
            UNION ALL
            SELECT team2 AS team, source, season, winner FROM matches
        )
        GROUP BY team, source, season;

        CREATE INDEX IF NOT EXISTS idx_team_stats ON team_stats(team, season);
    """)


def build_head_to_head(conn):
    run(conn, "head_to_head", """
        DROP TABLE IF EXISTS head_to_head;
        CREATE TABLE head_to_head AS
        SELECT
            CASE WHEN team1 < team2 THEN team1 ELSE team2 END AS team_a,
            CASE WHEN team1 < team2 THEN team2 ELSE team1 END AS team_b,
            source,
            season,
            venue,
            COUNT(*)                                                AS matches,
            SUM(CASE WHEN winner = team1 THEN 1 ELSE 0 END)        AS team1_wins,
            SUM(CASE WHEN winner = team2 THEN 1 ELSE 0 END)        AS team2_wins,
            SUM(CASE WHEN winner IS NULL THEN 1 ELSE 0 END)        AS no_result
        FROM matches
        GROUP BY team_a, team_b, source, season, venue;

        CREATE INDEX IF NOT EXISTS idx_h2h ON head_to_head(team_a, team_b);
    """)


def build_venue_stats(conn):
    run(conn, "venue_stats", """
        DROP TABLE IF EXISTS venue_stats;
        CREATE TABLE venue_stats AS
        WITH inning_scores AS (
            SELECT
                m.match_id,
                m.venue,
                m.source,
                m.season,
                m.winner,
                m.team1,
                m.team2,
                d.inning,
                SUM(d.runs_total) AS total_runs,
                SUM(d.is_wicket)  AS wickets_fallen
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            GROUP BY m.match_id, m.venue, m.source, m.season, d.inning
        ),
        first_innings AS (
            SELECT match_id, venue, source, season, winner, team1, team2,
                   total_runs AS first_innings_score,
                   wickets_fallen AS first_innings_wickets
            FROM inning_scores WHERE inning = 1
        ),
        second_innings AS (
            SELECT match_id, total_runs AS second_innings_score
            FROM inning_scores WHERE inning = 2
        )
        SELECT
            f.venue,
            f.source,
            f.season,
            COUNT(*)                                    AS matches,
            ROUND(AVG(f.first_innings_score), 1)        AS avg_first_innings_score,
            ROUND(AVG(s.second_innings_score), 1)       AS avg_second_innings_score,
            MAX(f.first_innings_score)                  AS highest_first_innings,
            MIN(f.first_innings_score)                  AS lowest_first_innings,
            SUM(CASE
                WHEN f.winner = f.team1 AND (
                    SELECT toss_decision FROM matches WHERE match_id = f.match_id
                ) = 'field' THEN 1
                WHEN f.winner = f.team2 AND (
                    SELECT toss_decision FROM matches WHERE match_id = f.match_id
                ) = 'bat' THEN 1
                ELSE 0 END)                             AS chasing_wins,
            SUM(CASE WHEN f.winner IS NOT NULL THEN 1 ELSE 0 END) AS decided_matches,
            ROUND(
                CAST(SUM(CASE
                    WHEN f.winner = f.team1 AND (
                        SELECT toss_decision FROM matches WHERE match_id = f.match_id
                    ) = 'field' THEN 1
                    WHEN f.winner = f.team2 AND (
                        SELECT toss_decision FROM matches WHERE match_id = f.match_id
                    ) = 'bat' THEN 1
                    ELSE 0 END) AS REAL)
                / NULLIF(SUM(CASE WHEN f.winner IS NOT NULL THEN 1 ELSE 0 END), 0) * 100, 2
            )                                           AS chasing_win_pct
        FROM first_innings f
        LEFT JOIN second_innings s ON s.match_id = f.match_id
        GROUP BY f.venue, f.source, f.season;

        CREATE INDEX IF NOT EXISTS idx_venue ON venue_stats(venue);
    """)


def build_toss_impact(conn):
    run(conn, "toss_impact", """
        DROP TABLE IF EXISTS toss_impact;
        CREATE TABLE toss_impact AS
        SELECT
            source,
            season,
            venue,
            toss_decision,
            COUNT(*)                                                AS matches,
            SUM(CASE WHEN winner = toss_winner THEN 1 ELSE 0 END)  AS toss_winner_won,
            ROUND(
                CAST(SUM(CASE WHEN winner = toss_winner THEN 1 ELSE 0 END) AS REAL)
                / NULLIF(COUNT(*), 0) * 100, 2
            )                                                       AS toss_win_pct
        FROM matches
        WHERE winner IS NOT NULL
        GROUP BY source, season, venue, toss_decision;
    """)


def build_phase_stats(conn):
    run(conn, "phase_stats (batting)", """
        DROP TABLE IF EXISTS phase_batting_stats;
        CREATE TABLE phase_batting_stats AS
        SELECT
            d.batter                                AS player,
            m.source,
            m.season,
            CASE
                WHEN d.over_num < 6  THEN 'powerplay'
                WHEN d.over_num < 15 THEN 'middle'
                ELSE 'death'
            END                                     AS phase,
            COUNT(*)                                AS balls_faced,
            SUM(d.runs_batter)                      AS runs,
            ROUND(
                CAST(SUM(d.runs_batter) AS REAL) / NULLIF(COUNT(*), 0) * 100, 2
            )                                       AS strike_rate,
            SUM(d.is_wicket)                        AS dismissals
        FROM deliveries d
        JOIN matches m ON m.match_id = d.match_id
        WHERE d.extras_type NOT IN ('wide', 'noball') OR d.extras_type IS NULL
        GROUP BY d.batter, m.source, m.season, phase;

        DROP TABLE IF EXISTS phase_bowling_stats;
        CREATE TABLE phase_bowling_stats AS
        SELECT
            d.bowler                                AS player,
            m.source,
            m.season,
            CASE
                WHEN d.over_num < 6  THEN 'powerplay'
                WHEN d.over_num < 15 THEN 'middle'
                ELSE 'death'
            END                                     AS phase,
            SUM(CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END) AS balls,
            SUM(d.runs_total) - SUM(d.runs_extras)  AS runs_conceded,
            SUM(CASE WHEN d.is_wicket = 1
                AND d.wicket_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                THEN 1 ELSE 0 END)                  AS wickets,
            ROUND(
                CAST(SUM(d.runs_total) - SUM(d.runs_extras) AS REAL)
                / NULLIF(SUM(CASE WHEN d.extras_type IN ('wide','noball') THEN 0 ELSE 1 END), 0) * 6, 2
            )                                       AS economy
        FROM deliveries d
        JOIN matches m ON m.match_id = d.match_id
        GROUP BY d.bowler, m.source, m.season, phase;
    """)


def build_recent_form(conn):
    run(conn, "recent_form (last 10 matches)", """
        DROP TABLE IF EXISTS player_recent_form;
        CREATE TABLE player_recent_form AS
        WITH ranked_bat AS (
            SELECT
                d.batter AS player,
                m.match_id,
                m.date,
                m.source,
                SUM(d.runs_batter)  AS runs,
                MAX(d.is_wicket)    AS got_out,
                ROW_NUMBER() OVER (PARTITION BY d.batter, m.source ORDER BY m.date DESC) AS rn
            FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            GROUP BY d.batter, m.match_id, m.date, m.source
        )
        SELECT
            player, source,
            COUNT(*)                    AS recent_innings,
            SUM(runs)                   AS recent_runs,
            ROUND(AVG(runs), 2)         AS recent_avg,
            SUM(got_out)                AS recent_dismissals
        FROM ranked_bat
        WHERE rn <= 10
        GROUP BY player, source;
    """)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    build_player_batting(conn)
    build_player_bowling(conn)
    build_team_stats(conn)
    build_head_to_head(conn)
    build_venue_stats(conn)
    build_toss_impact(conn)
    build_phase_stats(conn)
    build_recent_form(conn)

    log.info("All feature tables built successfully.")
    conn.close()


if __name__ == "__main__":
    main()
