"""
tools.py — Agent tools: SQL query, vector search, win probability

All three are exposed as Claude tool-use definitions + Python callables.
"""

import os
import sqlite3
import json
import logging
from dotenv import load_dotenv

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

load_dotenv()
log = logging.getLogger(__name__)

DB_PATH    = os.getenv("DB_PATH",    "data/processed/ipl.db")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/processed/chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Lazy-loaded singletons ──────────────────────────────────────────────────

_conn = None
_chroma_client = None
_match_col = None
_player_col = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def _get_collections():
    global _chroma_client, _match_col, _player_col
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("ChromaDB not available in this environment")
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        _match_col  = _chroma_client.get_collection("match_summaries",  embedding_function=ef)
        _player_col = _chroma_client.get_collection("player_profiles",  embedding_function=ef)
    return _match_col, _player_col


# ── Tool implementations ────────────────────────────────────────────────────

def query_stats(sql: str) -> str:
    """Execute a read-only SQL query against the IPL stats database."""
    sql = sql.strip()
    # Safety: only allow SELECT
    if not sql.upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed."})
    try:
        conn = _get_conn()
        rows = conn.execute(sql).fetchmany(50)  # cap at 50 rows
        result = [dict(r) for r in rows]
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def semantic_search(query: str, collection: str = "match_summaries", n_results: int = 5) -> str:
    """Search match summaries or player profiles by semantic similarity."""
    try:
        match_col, player_col = _get_collections()
        col = match_col if collection == "match_summaries" else player_col
        results = col.query(query_texts=[query], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        output = [{"text": d, "meta": m} for d, m in zip(docs, metas)]
        return json.dumps(output, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def compute_win_probability(team1: str, team2: str, venue: str = None, source: str = "ipl") -> str:
    """
    Compute win probability for team1 vs team2.

    Weight scheme (more emphasis on recent form):
      - Recent form last 3 seasons : 40%
      - Venue H2H                  : 25%  (if available, else redistributed)
      - Overall H2H                : 20%
      - Toss impact at venue       : 10%
      - Home ground factor         : 5%

    Confidence is scaled by sample size — small samples pull toward 50/50.
    """
    conn = _get_conn()
    prior = 0.5

    def norm(t): return t.strip() if t else t
    def confidence(n, scale=20): return min(n / scale, 1.0)  # full confidence at n>=20

    t1, t2 = norm(team1), norm(team2)
    team_a = min(t1, t2)
    team_b = max(t1, t2)

    # ── 1. Overall H2H ───────────────────────────────────────────
    h2h = conn.execute("""
        SELECT SUM(matches) AS m,
               SUM(CASE WHEN team_a = ? THEN team1_wins ELSE team2_wins END) AS ta_wins
        FROM head_to_head
        WHERE team_a = ? AND team_b = ? AND source = ?
    """, (team_a, team_a, team_b, source)).fetchone()

    total_h2h = h2h["m"] or 0
    t1_h2h_wins = (h2h["ta_wins"] if team_a == t1 else total_h2h - (h2h["ta_wins"] or 0)) or 0
    t2_h2h_wins = total_h2h - t1_h2h_wins

    h2h_raw = t1_h2h_wins / total_h2h if total_h2h > 0 else prior
    h2h_conf = confidence(total_h2h, scale=20)
    h2h_prob = h2h_raw * h2h_conf + prior * (1 - h2h_conf)

    # ── 2. Venue H2H ─────────────────────────────────────────────
    venue_prob = prior
    venue_matches = 0
    t1_venue_wins = 0
    if venue:
        vh = conn.execute("""
            SELECT SUM(matches) AS m,
                   SUM(CASE WHEN team_a = ? THEN team1_wins ELSE team2_wins END) AS ta_wins
            FROM head_to_head
            WHERE team_a = ? AND team_b = ? AND source = ? AND venue LIKE ?
        """, (team_a, team_a, team_b, source, f"%{venue}%")).fetchone()
        venue_matches = vh["m"] or 0
        t1_venue_wins = (vh["ta_wins"] if team_a == t1 else venue_matches - (vh["ta_wins"] or 0)) or 0
        venue_raw = t1_venue_wins / venue_matches if venue_matches > 0 else prior
        venue_conf = confidence(venue_matches, scale=8)
        venue_prob = venue_raw * venue_conf + prior * (1 - venue_conf)

    # ── 3. Recent form — last 3 seasons weighted (most recent = highest weight) ──
    def season_form(team):
        rows = conn.execute("""
            SELECT season, wins, matches_played
            FROM team_stats
            WHERE team = ? AND source = ? AND matches_played > 0
            ORDER BY season DESC LIMIT 3
        """, (team, source)).fetchall()
        if not rows:
            return prior, 0
        # Weight: most recent season 3x, second 2x, third 1x
        weights = [3, 2, 1]
        w_wins = w_played = 0
        for i, row in enumerate(rows):
            w = weights[i] if i < len(weights) else 1
            w_wins   += row["wins"] * w
            w_played += row["matches_played"] * w
        rate = w_wins / w_played if w_played > 0 else 0.5
        return rate, w_played

    t1_form_rate, t1_form_n = season_form(t1)
    t2_form_rate, t2_form_n = season_form(t2)
    total_form_n = t1_form_n + t2_form_n

    if t1_form_rate + t2_form_rate > 0:
        form_raw = t1_form_rate / (t1_form_rate + t2_form_rate)
    else:
        form_raw = prior
    form_conf = confidence(total_form_n, scale=30)
    form_prob = form_raw * form_conf + prior * (1 - form_conf)

    # ── 4. Toss impact at venue ───────────────────────────────────
    toss_boost = 0.0
    if venue:
        toss = conn.execute("""
            SELECT toss_decision, toss_win_pct, matches
            FROM toss_impact
            WHERE source = ? AND venue LIKE ?
            ORDER BY matches DESC LIMIT 2
        """, (source, f"%{venue}%")).fetchall()
        if toss:
            # Average toss advantage at this venue (how much winning toss helps)
            avg_toss_pct = sum(r["toss_win_pct"] for r in toss) / len(toss)
            # Small boost if toss matters significantly at this venue (>55%)
            toss_boost = (avg_toss_pct - 50) / 100 * 0.15 if avg_toss_pct > 52 else 0

    # ── 5. Home ground factor ─────────────────────────────────────
    home_boost = 0.0
    if venue:
        # Check if venue is team1's traditional home ground
        home_map = {
            "Mumbai Indians":          ["wankhede", "mumbai"],
            "Chennai Super Kings":     ["chepauk", "chennai", "ma chidambaram"],
            "Royal Challengers":       ["chinnaswamy", "bangalore", "bengaluru"],
            "Kolkata Knight Riders":   ["eden gardens", "kolkata"],
            "Sunrisers Hyderabad":     ["rajiv gandhi", "uppal", "hyderabad"],
            "Delhi Capitals":          ["feroz shah kotla", "arun jaitley", "delhi"],
            "Rajasthan Royals":        ["sawai mansingh", "jaipur"],
            "Punjab Kings":            ["mohali", "punjab cricket", "chandigarh"],
            "Gujarat Titans":          ["narendra modi", "ahmedabad"],
            "Lucknow Super Giants":    ["ekana", "lucknow"],
        }
        v_lower = venue.lower()
        for team, grounds in home_map.items():
            if any(g in v_lower for g in grounds):
                if team in t1:
                    home_boost = 0.04   # 4% boost for home team
                elif team in t2:
                    home_boost = -0.04  # 4% disadvantage for away team
                break

    # ── Final weighted blend ──────────────────────────────────────
    if venue_matches > 0:
        w_form, w_venue, w_h2h, w_toss, w_home = 0.40, 0.25, 0.20, 0.10, 0.05
    else:
        # No venue data — redistribute venue weight to form and h2h
        w_form, w_venue, w_h2h, w_toss, w_home = 0.50, 0.00, 0.30, 0.15, 0.05

    base_prob = (
        w_form  * form_prob  +
        w_venue * venue_prob +
        w_h2h   * h2h_prob   +
        w_toss  * prior      +   # toss is neutral without knowing who wins toss
        w_home  * prior
    ) + toss_boost + home_boost

    # Clip to [0.1, 0.9] — no prediction should be >90% or <10%
    final_prob = max(0.10, min(0.90, base_prob))

    # Fetch recent 3 seasons detail for transparency
    def recent_seasons(team):
        rows = conn.execute("""
            SELECT season, wins, matches_played, win_pct
            FROM team_stats
            WHERE team = ? AND source = ?
            ORDER BY season DESC LIMIT 3
        """, (team, source)).fetchall()
        return [dict(r) for r in rows]

    result = {
        "team1": t1,
        "team2": t2,
        "venue": venue,
        "source": source,
        "team1_win_probability": round(final_prob * 100, 1),
        "team2_win_probability": round((1 - final_prob) * 100, 1),
        "note": "Probability based on historical data only. Adjust manually for injuries/squad changes using current news.",
        "components": {
            "recent_form_3_seasons": {
                "team1_weighted_win_rate": round(t1_form_rate * 100, 1),
                "team2_weighted_win_rate": round(t2_form_rate * 100, 1),
                "team1_seasons": recent_seasons(t1),
                "team2_seasons": recent_seasons(t2),
                "weight": f"{int(w_form*100)}%"
            },
            "venue_h2h": {
                "matches_at_venue": venue_matches,
                "team1_wins": t1_venue_wins,
                "team2_wins": venue_matches - t1_venue_wins,
                "team1_venue_pct": round(venue_prob * 100, 1),
                "weight": f"{int(w_venue*100)}%"
            },
            "overall_h2h": {
                "total_matches": total_h2h,
                "team1_wins": int(t1_h2h_wins),
                "team2_wins": int(t2_h2h_wins),
                "team1_h2h_pct": round(h2h_prob * 100, 1),
                "weight": f"{int(w_h2h*100)}%"
            },
            "home_ground_boost": round(home_boost * 100, 1),
            "toss_venue_impact": round(toss_boost * 100, 1),
        }
    }
    return json.dumps(result, default=str)


def get_matchup_analysis(team1: str, team2: str, venue: str = None, source: str = "ipl") -> str:
    """
    Deep matchup analysis combining:
    - Batter vs bowler H2H
    - Impact player scores for both teams
    - Chasing/defending profiles
    - Venue pitch profile
    - Team momentum & streaks
    - Pressure/clutch performance
    """
    conn = _get_conn()

    def norm(t): return t.strip() if t else t
    t1, t2 = norm(team1), norm(team2)

    result = {}

    # ── Pitch profile ──────────────────────────────────────────
    if venue:
        pitch = conn.execute("""
            SELECT pitch_type, dew_factor, avg_first_innings,
                   avg_powerplay_rpo, avg_death_rpo, chasing_win_pct,
                   total_matches
            FROM venue_pitch_profiles
            WHERE venue LIKE ? AND source = ?
            LIMIT 1
        """, (f"%{venue}%", source)).fetchone()
        result["pitch_profile"] = dict(pitch) if pitch else {"note": "No venue data found"}

    # ── Team momentum ──────────────────────────────────────────
    for team, key in [(t1, "team1_momentum"), (t2, "team2_momentum")]:
        mom = conn.execute("""
            SELECT season, recent_matches, recent_wins, recent_win_pct,
                   momentum_score, current_streak_length, last_match_won
            FROM team_momentum
            WHERE team = ? AND source = ?
            ORDER BY season DESC LIMIT 1
        """, (team, source)).fetchone()
        result[key] = dict(mom) if mom else {"team": team, "note": "No momentum data"}

    # ── Chasing/defending profiles ─────────────────────────────
    for team, key in [(t1, "team1_profile"), (t2, "team2_profile")]:
        profiles = conn.execute("""
            SELECT role, matches, wins, win_pct
            FROM team_chasing_profiles
            WHERE team = ? AND source = ?
            ORDER BY season DESC, role
            LIMIT 4
        """, (team, source)).fetchall()
        result[key] = [dict(p) for p in profiles]

    # ── Pressure/clutch performance ────────────────────────────
    for team, key in [(t1, "team1_clutch"), (t2, "team2_clutch")]:
        clutch = conn.execute("""
            SELECT close_matches, close_wins, close_win_pct, clutch_rating
            FROM pressure_performance
            WHERE team = ? AND source = ?
            ORDER BY season DESC LIMIT 1
        """, (team, source)).fetchone()
        result[key] = dict(clutch) if clutch else {"note": "Insufficient data"}

    # ── Top impact players for each team ──────────────────────
    for team, key in [(t1, "team1_impact_players"), (t2, "team2_impact_players")]:
        # Get recent season's top players from deliveries (proxy for squad)
        players = conn.execute("""
            SELECT DISTINCT batter AS player FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            WHERE (m.team1 = ? OR m.team2 = ?)
              AND m.source = ?
            ORDER BY m.date DESC
            LIMIT 100
        """, (team, team, source)).fetchall()
        player_list = [p["player"] for p in players][:15]

        if player_list:
            placeholders = ",".join("?" * len(player_list))
            impact = conn.execute(f"""
                SELECT player, total_impact_score, batting_impact,
                       bowling_impact, primary_role
                FROM impact_player_scores
                WHERE player IN ({placeholders}) AND source = ?
                ORDER BY total_impact_score DESC LIMIT 5
            """, player_list + [source]).fetchall()
            result[key] = [dict(i) for i in impact]
        else:
            result[key] = []

    # ── Key batter vs bowler matchups ─────────────────────────
    # Get top batters from team1 vs top bowlers from team2 and vice versa
    def get_top_players(team, role, limit=5):
        col = "batter" if role == "bat" else "bowler"
        agg = "SUM(runs_batter)" if role == "bat" else "SUM(is_wicket)"
        rows = conn.execute(f"""
            SELECT d.{col} AS player FROM deliveries d
            JOIN matches m ON m.match_id = d.match_id
            WHERE (m.team1 = ? OR m.team2 = ?) AND m.source = ?
            GROUP BY d.{col}
            ORDER BY {agg} DESC LIMIT ?
        """, (team, team, source, limit)).fetchall()
        return [r["player"] for r in rows]

    t1_batters  = get_top_players(t1, "bat")
    t2_bowlers  = get_top_players(t2, "bowl")
    t2_batters  = get_top_players(t2, "bat")
    t1_bowlers  = get_top_players(t1, "bowl")

    def get_matchups(batters, bowlers):
        if not batters or not bowlers:
            return []
        b_ph = ",".join("?" * len(batters))
        w_ph = ",".join("?" * len(bowlers))
        rows = conn.execute(f"""
            SELECT batter, bowler, balls, runs, dismissals,
                   strike_rate, dot_ball_pct, dominance_index
            FROM batter_vs_bowler
            WHERE batter IN ({b_ph}) AND bowler IN ({w_ph})
              AND source = ? AND balls >= 6
            ORDER BY balls DESC LIMIT 10
        """, batters + bowlers + [source]).fetchall()
        return [dict(r) for r in rows]

    result["key_matchups_t1_bat_vs_t2_bowl"] = get_matchups(t1_batters, t2_bowlers)
    result["key_matchups_t2_bat_vs_t1_bowl"] = get_matchups(t2_batters, t1_bowlers)

    # ── Rolling form of key players ────────────────────────────
    all_players = list(set(t1_batters + t1_bowlers + t2_batters + t2_bowlers))[:20]
    if all_players:
        ph = ",".join("?" * len(all_players))
        form = conn.execute(f"""
            SELECT player, recent_innings, recent_avg, recent_sr,
                   recent_wickets, recent_economy, form_rating, batting_form_trend
            FROM player_rolling_form
            WHERE player IN ({ph}) AND source = ?
        """, all_players + [source]).fetchall()
        result["player_current_form"] = [dict(f) for f in form]

    # ── Death & powerplay specialists ──────────────────────────
    all_bowlers = list(set(t1_bowlers + t2_bowlers))[:10]
    if all_bowlers:
        ph = ",".join("?" * len(all_bowlers))
        death = conn.execute(f"""
            SELECT player, role, economy, wickets, dot_pct
            FROM death_over_specialists
            WHERE player IN ({ph}) AND source = ? AND role = 'bowling'
            ORDER BY economy ASC LIMIT 6
        """, all_bowlers + [source]).fetchall()
        result["death_bowlers"] = [dict(d) for d in death]

        pp = conn.execute(f"""
            SELECT player, role, economy, wickets
            FROM powerplay_specialists
            WHERE player IN ({ph}) AND source = ? AND role = 'bowling'
            ORDER BY wickets DESC LIMIT 6
        """, all_bowlers + [source]).fetchall()
        result["powerplay_bowlers"] = [dict(p) for p in pp]

    return json.dumps(result, default=str)


def get_player_profile(player_name: str, vs_team: str = None, source: str = "ipl") -> str:
    """
    Get a complete player profile including:
    - Career stats, recent form, phase-wise performance
    - Performance vs specific team (if provided)
    - Batter vs bowler dominance records
    - Impact score and role classification
    """
    conn = _get_conn()
    result = {"player": player_name}

    # Career batting
    bat = conn.execute("""
        SELECT SUM(innings) AS inn, SUM(total_runs) AS runs,
               MAX(highest_score) AS hs, ROUND(AVG(batting_avg),2) AS avg,
               ROUND(AVG(strike_rate),2) AS sr,
               SUM(fifties) AS fifties, SUM(hundreds) AS hundreds, SUM(sixes) AS sixes
        FROM player_batting_stats WHERE player = ? AND source = ?
    """, (player_name, source)).fetchone()
    result["career_batting_alltime"] = dict(bat) if bat else {}

    # Career bowling
    bowl = conn.execute("""
        SELECT SUM(innings_bowled) AS inn, SUM(wickets) AS wkts,
               ROUND(AVG(economy),2) AS economy, ROUND(AVG(bowling_avg),2) AS avg,
               SUM(three_wicket_hauls) AS three_fers, SUM(five_wicket_hauls) AS five_fers
        FROM player_bowling_stats WHERE player = ? AND source = ?
    """, (player_name, source)).fetchone()
    result["career_bowling_alltime"] = dict(bowl) if bowl else {}

    # Historical rolling averages — NOT current 2026 form
    form = conn.execute("""
        SELECT recent_innings, recent_runs, recent_avg, recent_sr,
               recent_wickets, recent_economy, form_rating, batting_form_trend
        FROM player_rolling_form WHERE player = ? AND source = ?
    """, (player_name, source)).fetchone()
    if form:
        form_dict = dict(form)
        form_dict["WARNING"] = (
            "These are HISTORICAL rolling averages across all IPL seasons, "
            "NOT the player's current IPL 2026 form. "
            "Do NOT report these as 'current form' or 'last 5 matches'. "
            "Use web_search for actual 2026 season stats."
        )
        result["historical_rolling_avg"] = form_dict
    else:
        result["historical_rolling_avg"] = {}

    # Phase-wise batting
    phases = conn.execute("""
        SELECT phase, SUM(runs) AS runs, ROUND(AVG(strike_rate),1) AS sr,
               SUM(dismissals) AS dismissals
        FROM phase_batting_stats WHERE player = ? AND source = ?
        GROUP BY phase
    """, (player_name, source)).fetchall()
    result["phase_batting"] = [dict(p) for p in phases]

    # Phase-wise bowling
    bowl_phases = conn.execute("""
        SELECT phase, SUM(wickets) AS wickets, ROUND(AVG(economy),2) AS economy
        FROM phase_bowling_stats WHERE player = ? AND source = ?
        GROUP BY phase
    """, (player_name, source)).fetchall()
    result["phase_bowling"] = [dict(p) for p in bowl_phases]

    # Impact score
    impact = conn.execute("""
        SELECT total_impact_score, batting_impact, bowling_impact, primary_role, season
        FROM impact_player_scores WHERE player = ? AND source = ?
        ORDER BY season DESC LIMIT 1
    """, (player_name, source)).fetchone()
    result["impact_score"] = dict(impact) if impact else {}

    # Best/worst matchups as batter
    crushes = conn.execute("""
        SELECT bowler, balls, runs, dismissals, strike_rate, dominance_index
        FROM batter_vs_bowler
        WHERE batter = ? AND source = ? AND balls >= 10
        ORDER BY dominance_index DESC LIMIT 5
    """, (player_name, source)).fetchall()
    result["bowlers_dominated"] = [dict(r) for r in crushes]

    struggles = conn.execute("""
        SELECT bowler, balls, runs, dismissals, strike_rate, dominance_index
        FROM batter_vs_bowler
        WHERE batter = ? AND source = ? AND balls >= 10
        ORDER BY dominance_index ASC LIMIT 5
    """, (player_name, source)).fetchall()
    result["bowlers_struggled_against"] = [dict(r) for r in struggles]

    # vs specific team
    if vs_team:
        pvt = conn.execute("""
            SELECT innings, runs, avg, sr, dismissals, wickets, economy
            FROM player_vs_team
            WHERE player = ? AND opposition LIKE ? AND source = ?
        """, (player_name, f"%{vs_team}%", source)).fetchall()
        result[f"vs_{vs_team}"] = [dict(r) for r in pvt]

    # ── IPL 2026 actual stats from CricAPI cache ──────────────────────
    try:
        import json as _json
        cache_path = "data/processed/ipl_2026_stats_cache.json"
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = _json.load(f)

            # Find player in batters
            for b in cache.get("top_batters", []):
                if player_name.lower() in b["player"].lower() or b["player"].lower() in player_name.lower():
                    result["ipl_2026_batting"] = {
                        "runs": b["total_runs"],
                        "avg":  b["batting_avg"],
                        "sr":   b["strike_rate"],
                        "50s":  b["fifties"],
                        "100s": b["hundreds"],
                        "6s":   b["sixes"],
                        "note": "Real 2026 season stats from CricAPI scorecards"
                    }
                    break

            # Find player in bowlers
            for b in cache.get("top_bowlers", []):
                if player_name.lower() in b["player"].lower() or b["player"].lower() in player_name.lower():
                    result["ipl_2026_bowling"] = {
                        "wickets": b["wickets"],
                        "economy": b["economy"],
                        "avg":     b["bowling_avg"],
                        "note":    "Real 2026 season stats from CricAPI scorecards"
                    }
                    break
    except Exception:
        pass

    return json.dumps(result, default=str)


def get_current_squad(team: str, season: str = "2026") -> str:
    """Get current IPL squad for a team from the latest auction data."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT player, role, is_overseas, country, note
        FROM current_squads
        WHERE team LIKE ? AND season = ?
        ORDER BY role, player
    """, (f"%{team}%", season)).fetchall()

    if not rows:
        return json.dumps({"error": f"No squad data found for {team} season {season}. Run: python -m pipeline.sync_squads"})

    squad = {
        "team":     team,
        "season":   season,
        "overseas": [dict(r) for r in rows if r["is_overseas"]],
        "indian":   [dict(r) for r in rows if not r["is_overseas"]],
        "by_role":  {},
        "injured_or_replaced": [dict(r) for r in rows if r["note"]]
    }
    for r in rows:
        role = r["role"]
        if role not in squad["by_role"]:
            squad["by_role"][role] = []
        squad["by_role"][role].append(r["player"])

    return json.dumps(squad, default=str)


def get_playing_xi(team: str, match_date: str = None) -> str:
    """
    Get confirmed Playing XI for a team on a specific match date.
    If match_date not provided, returns the most recent entry.
    Also returns toss info if available.
    """
    conn = _get_conn()

    if match_date:
        rows = conn.execute("""
            SELECT player, captain, impact_sub, confirmed,
                   toss_winner, toss_decision, venue, match_desc
            FROM playing_xi
            WHERE team LIKE ? AND match_date = ?
            ORDER BY captain DESC
        """, (f"%{team}%", match_date)).fetchall()
    else:
        rows = conn.execute("""
            SELECT player, captain, impact_sub, confirmed,
                   toss_winner, toss_decision, venue, match_desc, match_date
            FROM playing_xi
            WHERE team LIKE ?
            ORDER BY match_date DESC, captain DESC
            LIMIT 11
        """, (f"%{team}%",)).fetchall()

    if not rows:
        # Fall back to squad and suggest likely XI
        squad = conn.execute("""
            SELECT player, role, is_overseas
            FROM current_squads
            WHERE team LIKE ? AND season = '2026'
            ORDER BY role
        """, (f"%{team}%",)).fetchall()

        return json.dumps({
            "status": "no_xi_confirmed",
            "message": f"No confirmed XI for {team}. Update data/playing_xi.json on match day.",
            "full_squad": [dict(r) for r in squad],
            "suggestion": "Use web_search to find today's probable XI"
        })

    first = dict(rows[0])
    result = {
        "team":          team,
        "match_date":    first.get("match_date", match_date),
        "match":         first["match_desc"],
        "venue":         first["venue"],
        "confirmed":     bool(first["confirmed"]),
        "toss_winner":   first["toss_winner"],
        "toss_decision": first["toss_decision"],
        "playing_xi":    [r["player"] for r in rows],
        "captain":       next((r["player"] for r in rows if r["captain"]), None),
        "impact_sub":    next((r["player"] for r in rows if r["impact_sub"]), None),
    }
    return json.dumps(result, default=str)


def _build_trend_xi(conn, team: str) -> dict:
    """
    Build probable XI based on:
    1. Players who appeared in the last 5 matches for this team (trend)
    2. Recent form scores to rank within each role
    3. Squad constraints (max 4 overseas, injury notes)

    This is far smarter than role-order guessing.
    """
    # ── Step 1: find players who appeared in last 5 matches ──
    recent_players = conn.execute("""
        SELECT d.batter AS player, COUNT(DISTINCT m.match_id) AS matches_played
        FROM deliveries d
        JOIN matches m ON m.match_id = d.match_id
        WHERE m.source = 'ipl'
          AND (m.team1 LIKE ? OR m.team2 LIKE ?)
          AND m.date >= date('now', '-60 days')
        GROUP BY d.batter
        UNION
        SELECT d.bowler AS player, COUNT(DISTINCT m.match_id) AS matches_played
        FROM deliveries d
        JOIN matches m ON m.match_id = d.match_id
        WHERE m.source = 'ipl'
          AND (m.team1 LIKE ? OR m.team2 LIKE ?)
          AND m.date >= date('now', '-60 days')
        GROUP BY d.bowler
    """, (f"%{team}%", f"%{team}%", f"%{team}%", f"%{team}%")).fetchall()

    # Players seen in recent matches, ranked by frequency
    recent_map = {}
    for row in recent_players:
        p = row["player"]
        if p and p not in recent_map:
            recent_map[p] = row["matches_played"]
        elif p:
            recent_map[p] = max(recent_map[p], row["matches_played"])

    # ── Step 2: get full squad with roles ──
    squad = conn.execute("""
        SELECT player, role, is_overseas, note
        FROM current_squads
        WHERE team LIKE ? AND season = '2026'
          AND (note IS NULL OR note = ''
               OR (note NOT LIKE '%ruled out%'
                   AND note NOT LIKE '%OUT for season%'
                   AND note NOT LIKE '%knee injury%'
                   AND note NOT LIKE '%shoulder%'))
    """, (f"%{team}%",)).fetchall()

    squad_list = [dict(p) for p in squad]

    # ── Step 3: score each player ──
    # Score = (appearances in last 5 matches × 10) + role priority
    role_priority = {"wicketkeeper": 40, "batter": 30, "allrounder": 20, "bowler": 10}

    for p in squad_list:
        appearances = recent_map.get(p["player"], 0)
        p["trend_score"] = (appearances * 10) + role_priority.get(p["role"], 0)

    # Sort by trend score descending
    squad_list.sort(key=lambda x: x["trend_score"], reverse=True)

    # ── Step 4: build XI with max 4 overseas ──
    xi = []
    overseas_count = 0
    for p in squad_list:
        if len(xi) >= 11:
            break
        if p["is_overseas"]:
            if overseas_count < 4:
                xi.append(p["player"])
                overseas_count += 1
        else:
            xi.append(p["player"])

    # ── Step 5: get form data for transparency ──
    trend_note = []
    for p in squad_list[:11]:
        apps = recent_map.get(p["player"], 0)
        if apps > 0:
            trend_note.append(f"{p['player']} (played {apps} of last 5)")
        else:
            trend_note.append(f"{p['player']} (no recent data — squad pick)")

    return {
        "status": "probable_xi",
        "xi":     xi[:11],
        "source": "trend_based",
        "trend_analysis": trend_note[:11],
        "note": (
            "⚠️ Probable XI based on recent match appearances + form. "
            "Players who consistently appeared in last 5 matches ranked higher. "
            "Verify with get_live_score or web_search for confirmed XI."
        ),
    }


def _get_cricapi_xi(team1: str, team2: str) -> dict:
    """
    Try to fetch actual playing XIs from CricAPI live match data.
    Returns dict {team_name: [player_list]} or empty dict if unavailable.
    """
    try:
        import requests
        api_key = os.getenv("CRICAPI_KEY")
        if not api_key:
            return {}

        resp = requests.get(
            "https://api.cricapi.com/v1/currentMatches",
            params={"apikey": api_key, "offset": 0},
            timeout=10
        )
        matches = resp.json().get("data", [])

        # Find the matching match
        t1_kw = team1.split()[-1].lower()   # e.g. "Indians" → "indians"
        t2_kw = team2.split()[-1].lower()
        target = None
        for m in matches:
            name = m.get("name", "").lower()
            if t1_kw in name or t2_kw in name:
                target = m
                break

        if not target:
            return {}

        # Get detailed scorecard
        detail_resp = requests.get(
            "https://api.cricapi.com/v1/match_info",
            params={"apikey": api_key, "id": target["id"]},
            timeout=10
        )
        detail = detail_resp.json().get("data", {})
        players = detail.get("players", {})   # {team_name: [player_list]}

        if players:
            log.info(f"CricAPI: got playing XIs for {list(players.keys())}")
            return players
        return {}

    except Exception as e:
        log.warning(f"CricAPI XI fetch failed: {e}")
        return {}


def resolve_playing_xi(team1: str, team2: str, match_date: str = None) -> str:
    """
    Resolve Playing XIs for both teams for a match.
    Priority:
      1. CricAPI live match data (most accurate — actual announced XI)
      2. playing_xi.json (manually updated)
      3. Probable XI from squad (fallback)
    """
    conn = _get_conn()
    result = {}

    # ── Try CricAPI first for live/confirmed XIs ──
    cricapi_xis = _get_cricapi_xi(team1, team2)

    for team in [team1, team2]:
        # Check if CricAPI returned XI for this team
        cricapi_xi = None
        for api_team, players in cricapi_xis.items():
            if any(word in api_team.lower() for word in team.lower().split()):
                cricapi_xi = players[:11]
                break

        if cricapi_xi and len(cricapi_xi) >= 11:
            result[team] = {
                "status": "confirmed_xi",
                "xi":     cricapi_xi,
                "source": "CricAPI (live announced XI)",
            }
            continue

        # ── Fallback: playing_xi.json ──
        xi_data = json.loads(get_playing_xi(team, match_date))

        if xi_data.get("status") != "no_xi_confirmed":
            result[team] = {
                "status": "confirmed_xi" if xi_data.get("confirmed") else "unconfirmed_xi",
                "xi":     xi_data.get("playing_xi", []),
                "toss_winner":   xi_data.get("toss_winner"),
                "toss_decision": xi_data.get("toss_decision"),
                "source": "playing_xi.json",
            }
            continue

        # ── Fallback: trend-based probable XI ──
        result[team] = _build_trend_xi(conn, team)

    return json.dumps(result, default=str)


def run_match_simulation(
    team1: str, team1_xi: list,
    team2: str, team2_xi: list,
    venue: str,
    toss_winner: str, toss_decision: str,
    n_simulations: int = 200,   # 200 is plenty — fast and accurate enough
    source: str = "ipl"
) -> str:
    """Run Monte Carlo match simulation using ML ball outcome model."""
    import signal, threading

    try:
        from ml.simulate import simulate_match

        # Hard 30-second timeout using a thread so it never hangs the web server
        result_container = {}

        def _run():
            try:
                result_container["result"] = simulate_match(
                    team1=team1, team1_xi=team1_xi,
                    team2=team2, team2_xi=team2_xi,
                    venue=venue,
                    toss_winner=toss_winner,
                    toss_decision=toss_decision,
                    n_simulations=n_simulations,
                    source=source,
                )
            except Exception as e:
                result_container["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=30)  # 30-second hard cap

        if t.is_alive():
            return json.dumps({"error": "Simulation timed out (>30s). Try with a smaller XI or check model files."})
        if "error" in result_container:
            raise Exception(result_container["error"])
        return json.dumps(result_container["result"], default=str)

    except FileNotFoundError:
        return json.dumps({"error": "ML models not trained yet. Run: python -m ml.train"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def predict_player_matchup(
    batter: str, bowler: str,
    over: int = 10, venue: str = None, source: str = "ipl"
) -> str:
    """Predict ball-by-ball outcome distribution for a specific batter vs bowler matchup."""
    try:
        from ml.matchup_engine import predict_matchup
        result = predict_matchup(batter=batter, bowler=bowler,
                                  over=over, venue=venue, source=source)
        return json.dumps(result, default=str)
    except FileNotFoundError:
        return json.dumps({"error": "ML models not trained yet. Run: python -m ml.train"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_live_score(match_query: str = "") -> str:
    """
    Fetch live IPL match scores from CricAPI.
    Returns current match status, live score, batting/bowling figures,
    and which players have batted/are yet to bat.
    """
    try:
        import requests
        api_key = os.getenv("CRICAPI_KEY")
        if not api_key:
            return json.dumps({"error": "CRICAPI_KEY not set in .env"})

        # Step 1: Get current matches
        resp = requests.get(
            "https://api.cricapi.com/v1/currentMatches",
            params={"apikey": api_key, "offset": 0},
            timeout=10
        )
        data = resp.json()

        if data.get("status") != "success":
            return json.dumps({"error": "CricAPI error", "detail": data.get("message", "unknown")})

        matches = data.get("data", [])
        if not matches:
            return json.dumps({"status": "no_live_matches", "message": "No live matches right now."})

        # Filter IPL matches
        ipl_matches = [m for m in matches if "ipl" in m.get("name", "").lower() or
                       "indian premier" in m.get("name", "").lower() or
                       any(t in m.get("name", "") for t in
                           ["MI","CSK","RCB","KKR","SRH","DC","RR","PBKS","GT","LSG"])]

        if not ipl_matches:
            ipl_matches = matches  # fallback to all if no IPL filter match

        # If query given, filter further
        if match_query:
            q = match_query.lower()
            filtered = [m for m in ipl_matches if any(
                word in m.get("name", "").lower() for word in q.split())]
            if filtered:
                ipl_matches = filtered

        results = []
        for match in ipl_matches[:3]:  # top 3 matches
            match_id = match.get("id")

            # Step 2: Get detailed scorecard
            detail_resp = requests.get(
                "https://api.cricapi.com/v1/match_info",
                params={"apikey": api_key, "id": match_id},
                timeout=10
            )
            detail = detail_resp.json().get("data", {})

            # Extract key info
            score_info = detail.get("score", [])
            scores = []
            for s in score_info:
                scores.append({
                    "inning":  s.get("inning", ""),
                    "runs":    s.get("r", 0),
                    "wickets": s.get("w", 0),
                    "overs":   s.get("o", 0),
                })

            results.append({
                "match":        match.get("name", ""),
                "status":       match.get("status", ""),
                "match_type":   match.get("matchType", ""),
                "venue":        match.get("venue", ""),
                "date":         match.get("date", ""),
                "teams":        match.get("teams", []),
                "live_score":   scores,
                "current_status": detail.get("status", ""),
                "toss":         detail.get("toss", {}),
            })

        return json.dumps({"live_matches": results}, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def web_search(query: str) -> str:
    """Search the web for current IPL news, squad updates, injuries, and player form."""
    import threading
    result_container = {}

    def _search():
        try:
            from tavily import TavilyClient
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                result_container["result"] = json.dumps({"error": "TAVILY_API_KEY not set in .env"})
                return
            client = TavilyClient(api_key=api_key)
            results = client.search(
                query=query,
                search_depth="basic",
                max_results=4,
                include_answer=True,
            )
            output = {
                "answer": results.get("answer"),
                "sources": [
                    {"title": r.get("title"), "url": r.get("url"), "content": r.get("content", "")[:300]}
                    for r in results.get("results", [])
                ]
            }
            result_container["result"] = json.dumps(output)
        except Exception as e:
            result_container["result"] = json.dumps({"error": str(e)})

    t = threading.Thread(target=_search, daemon=True)
    t.start()
    t.join(timeout=15)  # 15-second cap on web search

    if t.is_alive():
        return json.dumps({"error": "Web search timed out. Falling back to historical data."})
    return result_container.get("result", json.dumps({"error": "Search failed"}))


# ── Claude tool definitions (passed to the API) ─────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "query_stats",
        "description": (
            "Execute a read-only SQL SELECT query against the IPL/T20 stats SQLite database. "
            "Available tables: matches, deliveries, player_batting_stats, player_bowling_stats, "
            "team_stats, head_to_head, venue_stats, toss_impact, phase_batting_stats, "
            "phase_bowling_stats, player_recent_form. "
            "Use this for exact numbers: averages, totals, records, rankings, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid SQLite SELECT statement."
                }
            },
            "required": ["sql"]
        }
    },
    {
        "name": "semantic_search",
        "description": (
            "Search match narratives or player career summaries using semantic/vector similarity. "
            "Use this when the question is qualitative, contextual, or needs narrative context "
            "(e.g. 'matches where MI chased big totals', 'Kohli's performance under pressure'). "
            "collection must be 'match_summaries' or 'player_profiles'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query."
                },
                "collection": {
                    "type": "string",
                    "enum": ["match_summaries", "player_profiles"],
                    "description": "Which collection to search."
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                    "default": 5
                }
            },
            "required": ["query", "collection"]
        }
    },
    {
        "name": "get_current_squad",
        "description": "Get the current IPL 2026 squad for a team (post-auction, accurate). Use before any match prediction to know who's in the squad.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team":   {"type": "string", "description": "Team name e.g. 'Mumbai Indians'"},
                "season": {"type": "string", "default": "2026"}
            },
            "required": ["team"]
        }
    },
    {
        "name": "get_playing_xi",
        "description": "Get confirmed or latest Playing XI for a team on a match date. Returns toss info if available. Falls back to full squad if XI not yet confirmed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team":       {"type": "string"},
                "match_date": {"type": "string", "description": "Date in YYYY-MM-DD format (optional)"}
            },
            "required": ["team"]
        }
    },
    {
        "name": "resolve_playing_xi",
        "description": "Resolve Playing XIs for BOTH teams for a match. Returns confirmed XIs if available, else builds probable XI from squad. Always call this before run_match_simulation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team1":      {"type": "string"},
                "team2":      {"type": "string"},
                "match_date": {"type": "string", "description": "Date in YYYY-MM-DD format (optional)"}
            },
            "required": ["team1", "team2"]
        }
    },
    {
        "name": "run_match_simulation",
        "description": (
            "Run a Monte Carlo simulation (3000+ iterations) of a T20 match "
            "given the Playing XI of both teams. Uses ML ball outcome model trained on "
            "1.2M IPL deliveries. Returns: win probability with 95% confidence interval, "
            "score distributions (mean/median/p10/p90), and key batter-bowler matchups. "
            "Use this when you have the Playing XI — most accurate prediction tool available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team1":         {"type": "string"},
                "team1_xi":      {"type": "array", "items": {"type": "string"}, "description": "List of 11 player names"},
                "team2":         {"type": "string"},
                "team2_xi":      {"type": "array", "items": {"type": "string"}, "description": "List of 11 player names"},
                "venue":         {"type": "string"},
                "toss_winner":   {"type": "string"},
                "toss_decision": {"type": "string", "enum": ["bat", "field"]},
                "n_simulations": {"type": "integer", "default": 3000},
                "source":        {"type": "string", "enum": ["ipl","t20"], "default": "ipl"}
            },
            "required": ["team1","team1_xi","team2","team2_xi","venue","toss_winner","toss_decision"]
        }
    },
    {
        "name": "predict_player_matchup",
        "description": (
            "Predict ball-outcome probability for a specific batter vs bowler matchup. "
            "Returns: projected SR, wicket probability, dot ball %, boundary %, "
            "historical H2H record, and verdict (who dominates). "
            "Use for individual matchup analysis within a match prediction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "batter": {"type": "string"},
                "bowler": {"type": "string"},
                "over":   {"type": "integer", "description": "Over number 0-19", "default": 10},
                "venue":  {"type": "string"},
                "source": {"type": "string", "enum": ["ipl","t20"], "default": "ipl"}
            },
            "required": ["batter", "bowler"]
        }
    },
    {
        "name": "get_matchup_analysis",
        "description": (
            "Deep pre-match analysis combining ALL signals for a matchup: "
            "venue pitch type & dew factor, team momentum & win streaks, "
            "chasing vs defending profiles, clutch/pressure performance, "
            "top impact players, key batter-vs-bowler head-to-head matchups, "
            "death & powerplay specialists, and rolling form of key players. "
            "Always call this for match prediction questions alongside compute_win_probability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team1": {"type": "string", "description": "Full team name e.g. 'Mumbai Indians'"},
                "team2": {"type": "string", "description": "Full team name e.g. 'Chennai Super Kings'"},
                "venue": {"type": "string", "description": "Venue name or city (optional)"},
                "source": {"type": "string", "enum": ["ipl", "t20"], "default": "ipl"}
            },
            "required": ["team1", "team2"]
        }
    },
    {
        "name": "get_player_profile",
        "description": (
            "Complete player profile: career stats, last-5-match rolling form, "
            "phase-wise batting/bowling, impact score, bowlers they dominate vs struggle against, "
            "and performance vs a specific team. Use for any player-specific question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string", "description": "Player name e.g. 'JJ Bumrah'"},
                "vs_team": {"type": "string", "description": "Opposition team name (optional)"},
                "source": {"type": "string", "enum": ["ipl", "t20"], "default": "ipl"}
            },
            "required": ["player_name"]
        }
    },
    {
        "name": "get_live_score",
        "description": (
            "Fetch LIVE IPL match scores, current innings, batting figures, and match status "
            "from CricAPI in real-time. Use this FIRST whenever the user asks about an ongoing match, "
            "live score, who is batting, current run rate, or how much a player has scored today. "
            "This gives accurate real-time data — much better than web_search for live matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "match_query": {
                    "type": "string",
                    "description": "Optional filter e.g. 'MI CSK' or 'Wankhede' to find a specific match"
                }
            },
            "required": []
        }
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for IPL news, injuries, squad updates, and match previews. "
            "Use get_live_score instead for live match scores and current innings data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query e.g. 'CSK squad injuries IPL 2026' or 'Rohit Sharma recent form IPL 2026'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "compute_win_probability",
        "description": (
            "Compute the win probability for a match between two teams. "
            "Uses a Bayesian blend of: historical h2h record, venue-specific h2h, and recent form. "
            "Returns probability percentages with full reasoning breakdown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team1": {
                    "type": "string",
                    "description": "Full team name (e.g. 'Mumbai Indians')."
                },
                "team2": {
                    "type": "string",
                    "description": "Full team name (e.g. 'Chennai Super Kings')."
                },
                "venue": {
                    "type": "string",
                    "description": "Venue name or city (optional, improves accuracy)."
                },
                "source": {
                    "type": "string",
                    "enum": ["ipl", "t20"],
                    "description": "Which dataset to use — 'ipl' (default) or 't20'.",
                    "default": "ipl"
                }
            },
            "required": ["team1", "team2"]
        }
    }
]


TOOL_DISPATCH = {
    "query_stats":              lambda args: query_stats(**args),
    "semantic_search":          lambda args: semantic_search(**args),
    "compute_win_probability":  lambda args: compute_win_probability(**args),
    "get_matchup_analysis":     lambda args: get_matchup_analysis(**args),
    "get_player_profile":       lambda args: get_player_profile(**args),
    "get_current_squad":        lambda args: get_current_squad(**args),
    "get_playing_xi":           lambda args: get_playing_xi(**args),
    "resolve_playing_xi":       lambda args: resolve_playing_xi(**args),
    "run_match_simulation":     lambda args: run_match_simulation(**args),
    "predict_player_matchup":   lambda args: predict_player_matchup(**args),
    "web_search":               lambda args: web_search(**args),
    "get_live_score":           lambda args: get_live_score(**args),
}
