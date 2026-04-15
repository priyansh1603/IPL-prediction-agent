"""
embed.py — Build ChromaDB vector store from match/player narratives

Generates two collections:
  1. match_summaries  — one doc per match, rich text narrative
  2. player_profiles  — one doc per player, career + recent form summary

These are used by the agent for semantic retrieval (RAG).

Run:
    python pipeline/embed.py
"""

import os
import sqlite3
import logging
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH    = os.getenv("DB_PATH",    "data/processed/ipl.db")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/processed/chroma_db")

# Using a lightweight local embedding model — no API key needed
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_chroma_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client, ef


def build_match_summaries(conn: sqlite3.Connection, client, ef):
    log.info("Building match_summaries collection...")
    collection = client.get_or_create_collection(
        name="match_summaries",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Get top batters and bowlers per match from deliveries
    matches = conn.execute("""
        SELECT match_id, source, season, date, venue, city,
               team1, team2, toss_winner, toss_decision,
               winner, win_by_runs, win_by_wickets,
               player_of_match, event_name, match_number
        FROM matches
        ORDER BY date
    """).fetchall()

    cols = ["match_id","source","season","date","venue","city",
            "team1","team2","toss_winner","toss_decision",
            "winner","win_by_runs","win_by_wickets",
            "player_of_match","event_name","match_number"]

    # Batch inserts
    batch_docs, batch_ids, batch_meta = [], [], []
    BATCH = 200

    for row in matches:
        m = dict(zip(cols, row))
        mid = m["match_id"]

        # Top batters in this match
        top_bat = conn.execute("""
            SELECT batter, SUM(runs_batter) AS runs
            FROM deliveries WHERE match_id = ?
            GROUP BY batter ORDER BY runs DESC LIMIT 3
        """, (mid,)).fetchall()

        # Top bowlers in this match
        top_bowl = conn.execute("""
            SELECT bowler,
                   SUM(CASE WHEN is_wicket=1
                       AND wicket_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                       THEN 1 ELSE 0 END) AS wkts,
                   SUM(runs_total) - SUM(runs_extras) AS runs_given
            FROM deliveries WHERE match_id = ?
            GROUP BY bowler ORDER BY wkts DESC, runs_given ASC LIMIT 3
        """, (mid,)).fetchall()

        # Innings scores
        inn_scores = conn.execute("""
            SELECT inning, SUM(runs_total) AS score, SUM(is_wicket) AS wkts
            FROM deliveries WHERE match_id = ?
            GROUP BY inning ORDER BY inning
        """, (mid,)).fetchall()

        # Build natural language narrative
        result_str = "No result"
        if m["winner"]:
            if m["win_by_runs"]:
                result_str = f"{m['winner']} won by {m['win_by_runs']} runs"
            elif m["win_by_wickets"]:
                result_str = f"{m['winner']} won by {m['win_by_wickets']} wickets"
            else:
                result_str = f"{m['winner']} won"

        inn_text = ""
        for inn in inn_scores:
            inn_text += f" Innings {inn[0]}: {inn[1]}/{inn[2]}."

        bat_text = ", ".join([f"{b[0]} ({b[1]})" for b in top_bat])
        bowl_text = ", ".join([f"{b[0]} {b[1]}/{b[2]}" for b in top_bowl])

        toss_text = f"{m['toss_winner']} won toss and chose to {m['toss_decision']}" if m["toss_winner"] else ""

        doc = (
            f"Match: {m['team1']} vs {m['team2']}. "
            f"Season: {m['season']}. Date: {m['date']}. "
            f"Venue: {m['venue']}, {m['city'] or ''}. "
            f"Event: {m['event_name'] or 'T20'} match {m['match_number'] or ''}. "
            f"Toss: {toss_text}. "
            f"{inn_text} "
            f"Result: {result_str}. "
            f"Top batters: {bat_text}. "
            f"Top bowlers: {bowl_text}. "
            f"Player of match: {m['player_of_match'] or 'N/A'}."
        )

        meta = {
            "match_id":   mid,
            "source":     m["source"],
            "season":     str(m["season"]),
            "date":       str(m["date"] or ""),
            "venue":      str(m["venue"] or ""),
            "team1":      str(m["team1"] or ""),
            "team2":      str(m["team2"] or ""),
            "winner":     str(m["winner"] or ""),
        }

        batch_docs.append(doc)
        batch_ids.append(mid)
        batch_meta.append(meta)

        if len(batch_docs) >= BATCH:
            collection.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_meta)
            log.info(f"  Upserted {len(batch_ids)} match docs...")
            batch_docs, batch_ids, batch_meta = [], [], []

    if batch_docs:
        collection.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_meta)

    log.info(f"match_summaries collection: {collection.count()} documents")


def build_player_profiles(conn: sqlite3.Connection, client, ef):
    log.info("Building player_profiles collection...")
    collection = client.get_or_create_collection(
        name="player_profiles",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # All unique players across batting + bowling
    players = conn.execute("""
        SELECT DISTINCT player FROM (
            SELECT player FROM player_batting_stats
            UNION
            SELECT player FROM player_bowling_stats
        )
    """).fetchall()
    players = [p[0] for p in players]

    batch_docs, batch_ids, batch_meta = [], [], []
    BATCH = 200

    for player in players:
        # Career batting (IPL only for profile, all sources for depth)
        bat = conn.execute("""
            SELECT source, SUM(innings) AS inn, SUM(total_runs) AS runs,
                   MAX(highest_score) AS hs, ROUND(AVG(batting_avg),2) AS avg,
                   ROUND(AVG(strike_rate),2) AS sr,
                   SUM(fifties) AS fifties, SUM(hundreds) AS hundreds,
                   SUM(sixes) AS sixes
            FROM player_batting_stats
            WHERE player = ?
            GROUP BY source
        """, (player,)).fetchall()

        # Career bowling
        bowl = conn.execute("""
            SELECT source, SUM(innings_bowled) AS inn, SUM(wickets) AS wkts,
                   ROUND(AVG(economy),2) AS econ, ROUND(AVG(bowling_avg),2) AS avg,
                   SUM(three_wicket_hauls) AS three_fers
            FROM player_bowling_stats
            WHERE player = ?
            GROUP BY source
        """, (player,)).fetchall()

        # Phase strengths
        phases = conn.execute("""
            SELECT phase, SUM(runs) AS runs, ROUND(AVG(strike_rate),2) AS sr
            FROM phase_batting_stats
            WHERE player = ?
            GROUP BY phase
        """, (player,)).fetchall()

        # Recent form
        form = conn.execute("""
            SELECT source, recent_innings, recent_runs, recent_avg
            FROM player_recent_form WHERE player = ?
        """, (player,)).fetchall()

        # Best venues
        best_venue = conn.execute("""
            SELECT venue, SUM(total_runs) AS runs
            FROM player_batting_stats
            WHERE player = ? AND venue IS NOT NULL
            GROUP BY venue ORDER BY runs DESC LIMIT 2
        """, (player,)).fetchall()

        bat_text = "; ".join([
            f"{b[0].upper()}: {b[1]} innings, {b[2]} runs, avg {b[3]}, SR {b[4]}, HS {b[3]}, {b[5]} 50s, {b[6]} 100s"
            for b in bat
        ])
        bowl_text = "; ".join([
            f"{b[0].upper()}: {b[2]} wickets in {b[1]} innings, econ {b[3]}, avg {b[4]}, {b[5]} 3-fers"
            for b in bowl
        ])
        phase_text = "; ".join([f"{p[0]}: {p[1]} runs @ SR {p[2]}" for p in phases])
        form_text = "; ".join([f"{f[0]}: last {f[1]} innings — {f[2]} runs, avg {f[3]}" for f in form])
        venue_text = ", ".join([f"{v[0]} ({v[1]} runs)" for v in best_venue])

        doc = (
            f"Player: {player}. "
            f"Batting career — {bat_text or 'No batting data'}. "
            f"Bowling career — {bowl_text or 'No bowling data'}. "
            f"Phase-wise batting — {phase_text or 'N/A'}. "
            f"Recent form — {form_text or 'N/A'}. "
            f"Best venues — {venue_text or 'N/A'}."
        )

        meta = {"player": player}
        safe_id = player.replace(" ", "_").replace(".", "").replace("'", "")

        batch_docs.append(doc)
        batch_ids.append(safe_id)
        batch_meta.append(meta)

        if len(batch_docs) >= BATCH:
            collection.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_meta)
            log.info(f"  Upserted {len(batch_ids)} player profiles...")
            batch_docs, batch_ids, batch_meta = [], [], []

    if batch_docs:
        collection.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_meta)

    log.info(f"player_profiles collection: {collection.count()} documents")


def main():
    conn = sqlite3.connect(DB_PATH)
    client, ef = get_chroma_client()

    build_match_summaries(conn, client, ef)
    build_player_profiles(conn, client, ef)

    log.info("ChromaDB vector store built successfully.")
    conn.close()


if __name__ == "__main__":
    main()
