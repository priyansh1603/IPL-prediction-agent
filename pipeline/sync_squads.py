"""
sync_squads.py — Load current_squads.json into SQLite

Creates two tables:
  - current_squads   : full squad per team for current season
  - playing_xi       : confirmed match-day XIs (updated on match day)

Run after any auction or squad update:
    python -m pipeline.sync_squads
"""

import os
import json
import sqlite3
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH         = os.getenv("DB_PATH", "data/processed/ipl.db")
SQUADS_JSON     = "data/current_squads.json"
PLAYING_XI_JSON = "data/playing_xi.json"


def sync_squads(conn: sqlite3.Connection):
    with open(SQUADS_JSON) as f:
        data = json.load(f)

    season = data["_meta"]["season"]

    conn.execute("DROP TABLE IF EXISTS current_squads")
    conn.execute("""
        CREATE TABLE current_squads (
            player       TEXT,
            team         TEXT,
            season       TEXT,
            role         TEXT,
            is_overseas  INTEGER,
            country      TEXT,
            note         TEXT,
            home_ground  TEXT
        )
    """)

    rows = []
    for team, info in data["teams"].items():
        home_ground = info.get("home_ground", "")
        for p in info["players"]:
            rows.append({
                "player":      p["name"],
                "team":        team,
                "season":      season,
                "role":        p.get("role", ""),
                "is_overseas": 1 if p.get("is_overseas") else 0,
                "country":     p.get("country", "India"),
                "note":        p.get("note", ""),
                "home_ground": home_ground,
            })

    conn.executemany("""
        INSERT INTO current_squads
        (player, team, season, role, is_overseas, country, note, home_ground)
        VALUES (:player, :team, :season, :role, :is_overseas, :country, :note, :home_ground)
    """, rows)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_cs_player ON current_squads(player)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cs_team   ON current_squads(team)")
    conn.commit()
    log.info(f"Synced {len(rows)} players across {len(data['teams'])} teams for season {season}")


def sync_playing_xi(conn: sqlite3.Connection):
    with open(PLAYING_XI_JSON) as f:
        data = json.load(f)

    conn.execute("DROP TABLE IF EXISTS playing_xi")
    conn.execute("""
        CREATE TABLE playing_xi (
            match_date   TEXT,
            match_desc   TEXT,
            venue        TEXT,
            team         TEXT,
            player       TEXT,
            captain      INTEGER DEFAULT 0,
            impact_sub   INTEGER DEFAULT 0,
            confirmed    INTEGER DEFAULT 0,
            toss_winner  TEXT,
            toss_decision TEXT
        )
    """)

    rows = []
    for date, match in data.get("matches", {}).items():
        confirmed     = 1 if match.get("confirmed") else 0
        toss_winner   = match.get("toss", {}).get("winner") or ""
        toss_decision = match.get("toss", {}).get("decision") or ""
        venue         = match.get("venue", "")
        match_desc    = match.get("match", "")

        for key, team_data in match.items():
            if key in ("match", "venue", "confirmed", "toss", "_meta"):
                continue
            if not isinstance(team_data, dict):
                continue

            team      = key
            xi        = team_data.get("playing_xi", [])
            captain   = team_data.get("captain", "")
            imp_sub   = team_data.get("impact_sub", "")

            for player in xi:
                rows.append({
                    "match_date":    date,
                    "match_desc":    match_desc,
                    "venue":         venue,
                    "team":          team,
                    "player":        player,
                    "captain":       1 if player == captain else 0,
                    "impact_sub":    1 if player == imp_sub else 0,
                    "confirmed":     confirmed,
                    "toss_winner":   toss_winner,
                    "toss_decision": toss_decision,
                })

    conn.executemany("""
        INSERT INTO playing_xi
        (match_date, match_desc, venue, team, player, captain,
         impact_sub, confirmed, toss_winner, toss_decision)
        VALUES (:match_date, :match_desc, :venue, :team, :player, :captain,
                :impact_sub, :confirmed, :toss_winner, :toss_decision)
    """, rows)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_xi_date ON playing_xi(match_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_xi_team ON playing_xi(team)")
    conn.commit()
    log.info(f"Synced {len(rows)} playing XI entries across {len(data.get('matches', {}))} matches")


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    sync_squads(conn)
    sync_playing_xi(conn)
    conn.close()
    log.info("Squad sync complete.")


if __name__ == "__main__":
    main()
