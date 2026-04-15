"""
parse.py — Cricsheet JSON → SQLite

Parses all IPL and T20 JSON files and stores them in a normalized
SQLite database with tables: matches, innings, deliveries.

Run:
    python pipeline/parse.py
"""

import json
import os
import glob
import sqlite3
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/processed/ipl.db")
IPL_JSON_DIR = os.getenv("IPL_JSON_DIR", "data/raw/ipl_json")
T20_JSON_DIR = os.getenv("T20_JSON_DIR", "data/raw/t20s_json")


def create_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id        TEXT PRIMARY KEY,
            source          TEXT,           -- 'ipl' or 't20'
            season          TEXT,
            date            TEXT,
            venue           TEXT,
            city            TEXT,
            team1           TEXT,
            team2           TEXT,
            toss_winner     TEXT,
            toss_decision   TEXT,           -- 'bat' or 'field'
            winner          TEXT,           -- NULL if no result
            win_by_runs     INTEGER,
            win_by_wickets  INTEGER,
            player_of_match TEXT,
            event_name      TEXT,
            match_number    INTEGER
        );

        CREATE TABLE IF NOT EXISTS deliveries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id        TEXT,
            inning          INTEGER,        -- 1 or 2
            over_num        INTEGER,
            ball_num        INTEGER,        -- position within over
            batter          TEXT,
            bowler          TEXT,
            non_striker     TEXT,
            runs_batter     INTEGER DEFAULT 0,
            runs_extras     INTEGER DEFAULT 0,
            runs_total      INTEGER DEFAULT 0,
            extras_type     TEXT,           -- wide/noball/legbye/bye/penalty
            is_wicket       INTEGER DEFAULT 0,
            wicket_kind     TEXT,
            player_out      TEXT,
            fielder         TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(match_id)
        );

        CREATE INDEX IF NOT EXISTS idx_del_match ON deliveries(match_id);
        CREATE INDEX IF NOT EXISTS idx_del_batter ON deliveries(batter);
        CREATE INDEX IF NOT EXISTS idx_del_bowler ON deliveries(bowler);
    """)
    conn.commit()


def parse_match(filepath: str, source: str) -> tuple[dict, list[dict]]:
    with open(filepath) as f:
        data = json.load(f)

    info = data["info"]
    match_id = Path(filepath).stem

    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    win_by = outcome.get("by", {})

    teams = info.get("teams", [])
    team1 = teams[0] if len(teams) > 0 else None
    team2 = teams[1] if len(teams) > 1 else None

    toss = info.get("toss", {})
    event = info.get("event", {})
    pom = info.get("player_of_match", [])

    match_row = {
        "match_id":        match_id,
        "source":          source,
        "season":          str(info.get("season", "")),
        "date":            info.get("dates", [None])[0],
        "venue":           info.get("venue"),
        "city":            info.get("city"),
        "team1":           team1,
        "team2":           team2,
        "toss_winner":     toss.get("winner"),
        "toss_decision":   toss.get("decision"),
        "winner":          winner,
        "win_by_runs":     win_by.get("runs"),
        "win_by_wickets":  win_by.get("wickets"),
        "player_of_match": pom[0] if pom else None,
        "event_name":      event.get("name"),
        "match_number":    event.get("match_number"),
    }

    delivery_rows = []
    for inning_idx, inning in enumerate(data.get("innings", []), start=1):
        for over_data in inning.get("overs", []):
            over_num = over_data["over"]
            for ball_idx, delivery in enumerate(over_data.get("deliveries", [])):
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                extras_type = next(iter(extras.keys()), None) if extras else None

                wickets = delivery.get("wickets", [])
                is_wicket = 1 if wickets else 0
                wicket_kind = wickets[0].get("kind") if wickets else None
                player_out = wickets[0].get("player_out") if wickets else None
                fielders = wickets[0].get("fielders", []) if wickets else []
                fielder = fielders[0].get("name") if fielders else None

                delivery_rows.append({
                    "match_id":     match_id,
                    "inning":       inning_idx,
                    "over_num":     over_num,
                    "ball_num":     ball_idx,
                    "batter":       delivery.get("batter"),
                    "bowler":       delivery.get("bowler"),
                    "non_striker":  delivery.get("non_striker"),
                    "runs_batter":  runs.get("batter", 0),
                    "runs_extras":  runs.get("extras", 0),
                    "runs_total":   runs.get("total", 0),
                    "extras_type":  extras_type,
                    "is_wicket":    is_wicket,
                    "wicket_kind":  wicket_kind,
                    "player_out":   player_out,
                    "fielder":      fielder,
                })

    return match_row, delivery_rows


def ingest_directory(conn: sqlite3.Connection, directory: str, source: str):
    files = glob.glob(os.path.join(directory, "*.json"))
    log.info(f"Found {len(files)} files in {directory} (source={source})")

    match_rows = []
    delivery_rows = []

    for i, filepath in enumerate(files):
        try:
            match_row, deliveries = parse_match(filepath, source)
            match_rows.append(match_row)
            delivery_rows.extend(deliveries)
        except Exception as e:
            log.warning(f"Skipping {filepath}: {e}")

        if (i + 1) % 200 == 0:
            log.info(f"  Parsed {i+1}/{len(files)}...")

    # Bulk insert
    conn.executemany("""
        INSERT OR REPLACE INTO matches
        (match_id, source, season, date, venue, city, team1, team2,
         toss_winner, toss_decision, winner, win_by_runs, win_by_wickets,
         player_of_match, event_name, match_number)
        VALUES (:match_id, :source, :season, :date, :venue, :city, :team1, :team2,
                :toss_winner, :toss_decision, :winner, :win_by_runs, :win_by_wickets,
                :player_of_match, :event_name, :match_number)
    """, match_rows)

    conn.executemany("""
        INSERT INTO deliveries
        (match_id, inning, over_num, ball_num, batter, bowler, non_striker,
         runs_batter, runs_extras, runs_total, extras_type,
         is_wicket, wicket_kind, player_out, fielder)
        VALUES (:match_id, :inning, :over_num, :ball_num, :batter, :bowler, :non_striker,
                :runs_batter, :runs_extras, :runs_total, :extras_type,
                :is_wicket, :wicket_kind, :player_out, :fielder)
    """, delivery_rows)

    conn.commit()
    log.info(f"Inserted {len(match_rows)} matches, {len(delivery_rows)} deliveries from {source}")


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    create_schema(conn)
    ingest_directory(conn, IPL_JSON_DIR, "ipl")
    ingest_directory(conn, T20_JSON_DIR, "t20")

    # Quick sanity check
    matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    deliveries = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
    log.info(f"DB ready: {matches} matches, {deliveries} deliveries → {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
