"""
web/app.py — FastAPI backend for IPL Prediction Agent website

Run locally:
    uvicorn web.app:app --reload --port 8000

Then open: http://localhost:8000
"""

import os
import json
import asyncio
import datetime
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

app = FastAPI(title="IPL Prediction Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# ── In-memory conversation store (session-based) ──────────────────────────
conversations: dict[str, list] = {}

# ── Trending questions — generated from today's matches ───────────────────
_trending_cache = {"date": None, "questions": []}

FALLBACK_QUESTIONS = [
    "Who will win the IPL 2026 title?",
    "Who is leading the Orange Cap race?",
    "Who is leading the Purple Cap race?",
    "Which team has the best powerplay record this season?",
    "Who will be the Player of the Tournament in IPL 2026?",
    "Which venue favours chasing teams the most?",
]

def _build_trending_questions() -> list:
    """
    Build trending questions dynamically based on today's matches from CricAPI.
    Falls back to generic questions if no matches today.
    """
    import requests
    api_key = os.getenv("CRICAPI_KEY")
    if not api_key:
        return FALLBACK_QUESTIONS

    try:
        r = requests.get(
            "https://api.cricapi.com/v1/series_info",
            params={"apikey": api_key, "id": IPL_2026_SERIES_ID},
            timeout=10,
        )
        matches = r.json().get("data", {}).get("matchList", [])
        today   = datetime.date.today().isoformat()

        # Today's matches
        today_matches = [
            m for m in matches
            if m.get("date", "").startswith(today)
        ]

        # Recent completed matches (last 3)
        recent = [m for m in matches if m.get("matchEnded")][-3:]

        questions = []

        # Questions from today's fixtures
        for m in today_matches:
            teams = m.get("teams", [])
            if len(teams) < 2:
                continue
            t1, t2 = teams[0], teams[1]
            short1 = t1.split()[-1]   # e.g. "Kings" from "Punjab Kings"
            short2 = t2.split()[-1]

            questions += [
                f"Who will win today's {short1} vs {short2} match?",
                f"Who will score the most runs in {short1} vs {short2}?",
                f"Who will take the most wickets in today's {short1} vs {short2} game?",
                f"What is the win probability for {t1} today?",
                f"What is the head-to-head record between {t1} and {t2}?",
                f"Who should I pick as captain for {short1} vs {short2} in fantasy?",
            ]

        # Questions from recent results
        for m in recent:
            teams  = m.get("teams", [])
            status = m.get("status", "")
            if len(teams) < 2 or "won by" not in status:
                continue
            winner = status.split(" won by")[0].strip().split()[-1]
            loser  = [t for t in teams if t.split(" won by")[0] != status.split(" won by")[0]][0]
            questions.append(f"Why did {winner} win their last match? Key factors?")
            questions.append(f"How can {loser.split()[-1]} bounce back after their defeat?")

        # Always add a few evergreen ones
        questions += [
            "Who is leading the Orange Cap race right now?",
            "Who is leading the Purple Cap race right now?",
            "Which team has the best Net Run Rate in IPL 2026?",
        ]

        # Deduplicate and cap at 6
        seen = set()
        unique = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        return unique[:6] if unique else FALLBACK_QUESTIONS

    except Exception:
        return FALLBACK_QUESTIONS


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class FeedbackRequest(BaseModel):
    session_id: str
    message: str
    rating: int  # 1-5


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page."""
    with open("web/static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/trending")
async def get_trending():
    """
    Return trending IPL questions based on today's actual fixtures.
    Cached per day — regenerated automatically each new day.
    """
    global _trending_cache
    today = datetime.date.today().isoformat()

    if _trending_cache["date"] != today or not _trending_cache["questions"]:
        loop = asyncio.get_event_loop()
        questions = await loop.run_in_executor(None, _build_trending_questions)
        _trending_cache = {"date": today, "questions": questions}

    return {"questions": _trending_cache["questions"]}


# ── YouTube + News media feed ─────────────────────────────────────────────

YOUTUBE_CHANNELS = {
    "Cricbuzz":     "UCSRQXk5yErn4e14vN76upOw",
    "IPL Official": "UCCq1xDJMBRF61kiOgU90_kw",
    "ESPNcricinfo": "UCujuVKmt_utAQZJghxlRMIQ",
    "BCCI":         "UCiWrjBhlICf_L_RK5y6Vrxw",
}

_media_cache = {"date": None, "items": []}

def _fetch_media_feed() -> list:
    """Fetch latest IPL videos from YouTube RSS + news from Tavily."""
    import requests
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    items = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    # ── 1. YouTube RSS feeds ──────────────────────────────────────────
    ns = {
        "atom":  "http://www.w3.org/2005/Atom",
        "yt":    "http://www.youtube.com/xml/schemas/2015",
        "media": "http://search.yahoo.com/mrss/",
    }
    ipl_keywords = {"ipl", "t20", "cricket", "match", "wicket", "batting", "bowling", "over"}

    for channel_name, channel_id in YOUTUBE_CHANNELS.items():
        try:
            r = requests.get(
                f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}",
                headers=headers, timeout=8
            )
            if r.status_code != 200:
                continue

            root = ET.fromstring(r.content.decode("utf-8", errors="replace"))
            for entry in root.findall("atom:entry", ns)[:5]:
                try:
                    vid_id  = entry.find("yt:videoId", ns).text
                    title   = entry.find("atom:title", ns).text or ""
                    link    = f"https://www.youtube.com/watch?v={vid_id}"
                    thumb   = f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg"
                    pub     = entry.find("atom:published", ns).text or ""

                    # Only IPL-related videos
                    title_lower = title.lower()
                    if not any(kw in title_lower for kw in ipl_keywords):
                        continue

                    items.append({
                        "type":    "video",
                        "title":   title,
                        "channel": channel_name,
                        "thumb":   thumb,
                        "link":    link,
                        "pub":     pub[:10],   # YYYY-MM-DD
                    })
                except Exception:
                    continue
        except Exception:
            continue

    # ── 2. Tavily news search ─────────────────────────────────────────
    try:
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            today_str = datetime.date.today().strftime("%d %B %Y")
            r = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":        tavily_key,
                    "query":          f"IPL 2026 cricket {today_str}",
                    "search_depth":   "basic",
                    "max_results":    6,
                    "include_images": True,
                    "include_answer": False,
                },
                timeout=12,
            )
            if r.status_code == 200:
                for result in r.json().get("results", []):
                    img = result.get("image") or ""
                    items.append({
                        "type":    "news",
                        "title":   result.get("title", ""),
                        "channel": _domain(result.get("url", "")),
                        "thumb":   img,
                        "link":    result.get("url", ""),
                        "pub":     datetime.date.today().isoformat(),
                        "snippet": result.get("content", "")[:120],
                    })
    except Exception:
        pass

    # Sort: videos first (most recent), then news; dedupe by title
    seen_titles = set()
    unique = []
    for item in items:
        if item["title"] not in seen_titles:
            seen_titles.add(item["title"])
            unique.append(item)

    # Sort videos by pub date descending
    unique.sort(key=lambda x: (x["type"] == "news", x.get("pub", "")), reverse=False)
    return unique[:12]


def _domain(url: str) -> str:
    """Extract readable domain name from URL."""
    try:
        from urllib.parse import urlparse
        d = urlparse(url).netloc.replace("www.", "")
        return d.split(".")[0].capitalize()
    except Exception:
        return "News"


@app.get("/api/media")
async def get_media(refresh: bool = False):
    """Return YouTube videos + news articles about IPL 2026."""
    global _media_cache
    today = datetime.date.today().isoformat()

    if not refresh and _media_cache["date"] == today and _media_cache["items"]:
        return {"items": _media_cache["items"]}

    loop = asyncio.get_event_loop()
    items = await loop.run_in_executor(None, _fetch_media_feed)
    _media_cache = {"date": today, "items": items}
    return {"items": items}


@app.get("/api/today_matches")
async def get_today_matches():
    """Get today's IPL matches from CricAPI."""
    try:
        from agent.tools import get_live_score
        result = get_live_score({"match_query": "IPL 2026"})
        data = json.loads(result)
        return {"matches": data if isinstance(data, list) else [data]}
    except Exception as e:
        return {"matches": [], "error": str(e)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Send a message and stream the response back (Server-Sent Events).
    Progress events are streamed immediately as each tool fires,
    so the user sees live updates instead of a blank screen.
    """
    from agent.agent import run_agent

    session_id = req.session_id
    history = conversations.get(session_id, [])

    # Queue for passing progress events from the sync agent thread → async generator
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_tool_call(tool_name: str, label: str):
        """Called by agent (sync thread) each time a tool fires."""
        loop.call_soon_threadsafe(
            queue.put_nowait,
            json.dumps({"type": "progress", "label": label, "done": False})
        )

    def run_in_thread():
        try:
            response, updated_history = run_agent(
                req.message, history, on_tool_call=on_tool_call
            )
            conversations[session_id] = updated_history
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"type": "result", "text": response, "done": False}))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"type": "error", "text": str(e), "done": False}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, "__DONE__")

    async def generate():
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(executor, run_in_thread)

        # Send "thinking" immediately so user sees something within 1 second
        yield f"data: {json.dumps({'progress': '🧠 Thinking...', 'done': False})}\n\n"

        try:
            while True:
                try:
                    # Wait up to 1s then send a heartbeat so browser stays alive
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Keep connection alive with a comment (browser ignores it)
                    yield ": heartbeat\n\n"
                    continue

                if item == "__DONE__":
                    break

                payload = json.loads(item)

                if payload["type"] == "progress":
                    yield f"data: {json.dumps({'progress': payload['label'], 'done': False})}\n\n"

                elif payload["type"] == "result":
                    # Stream the final answer word-by-word (typewriter effect)
                    words = payload["text"].split(" ")
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                        await asyncio.sleep(0.018)

                elif payload["type"] == "error":
                    err_msg = "⚠️ " + payload.get("text", "Unknown error")
                    yield f"data: {json.dumps({'text': err_msg, 'done': False})}\n\n"

            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
            await future  # ensure thread is cleaned up

        except Exception as e:
            yield f"data: {json.dumps({'text': f'⚠️ Server error: {str(e)}', 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/chat/{session_id}")
async def clear_chat(session_id: str):
    """Clear conversation history for a session."""
    conversations.pop(session_id, None)
    return {"status": "cleared"}


IPL_2026_SERIES_ID = "87c62aac-bc3c-4738-ab93-19da0690488f"


def _fetch_stats_from_db():
    """Build IPL 2026 standings + batting + bowling from local SQLite DB (fast, no API calls)."""
    import sqlite3
    from collections import defaultdict

    db_path = os.getenv("DB_PATH", "data/processed/ipl.db")
    conn = sqlite3.connect(db_path)

    # Standings
    rows = conn.execute(
        "SELECT team1, team2, winner FROM matches WHERE season=2026 AND source='ipl'"
    ).fetchall()
    ts = defaultdict(lambda: {"played":0,"won":0,"lost":0,"nr":0,"pts":0})
    for t1, t2, winner in rows:
        ts[t1]["played"] += 1; ts[t2]["played"] += 1
        if winner and winner not in ("No result", "Tie", ""):
            ts[winner]["won"] += 1; ts[winner]["pts"] += 2
            loser = t2 if winner == t1 else t1
            ts[loser]["lost"] += 1
        else:
            for t in (t1, t2):
                ts[t]["nr"] += 1; ts[t]["pts"] += 1

    standings = sorted(
        [{"team":t,"matches_played":s["played"],"wins":s["won"],
          "losses":s["lost"],"nr":s["nr"],"points":s["pts"]}
         for t, s in ts.items()],
        key=lambda x: (-x["points"], -x["wins"]),
    )

    # Top batters
    bat_rows = conn.execute("""
        SELECT d.batter, SUM(d.runs_batter) as runs, COUNT(*) as balls,
               COUNT(CASE WHEN d.is_wicket=1 AND d.player_out=d.batter THEN 1 END) as outs,
               SUM(CASE WHEN d.runs_batter=6 THEN 1 ELSE 0 END) as sixes
        FROM deliveries d JOIN matches m ON d.match_id=m.match_id
        WHERE m.season=2026 AND m.source='ipl'
        GROUP BY d.batter HAVING runs > 30
        ORDER BY runs DESC LIMIT 10
    """).fetchall()
    top_batters = [
        {"player": r[0], "total_runs": r[1],
         "batting_avg": round(r[1]/r[3], 1) if r[3] else None,
         "strike_rate": round(r[1]/r[2]*100, 1) if r[2] else 0,
         "fifties": 0, "hundreds": 0, "sixes": r[4]}
        for r in bat_rows
    ]

    # Top bowlers
    bowl_rows = conn.execute("""
        SELECT d.bowler, SUM(d.is_wicket) as wkts,
               SUM(d.runs_total) as runs, COUNT(*) as balls
        FROM deliveries d JOIN matches m ON d.match_id=m.match_id
        WHERE m.season=2026 AND m.source='ipl'
        GROUP BY d.bowler HAVING wkts > 0
        ORDER BY wkts DESC LIMIT 10
    """).fetchall()
    top_bowlers = [
        {"player": r[0], "wickets": r[1],
         "economy": round(r[2]/r[3]*6, 2) if r[3] else 0,
         "bowling_avg": round(r[2]/r[1], 1) if r[1] else None}
        for r in bowl_rows
    ]

    conn.close()
    return standings, top_batters, top_bowlers
STATS_CACHE_FILE   = "data/processed/ipl_2026_stats_cache.json"


SCORECARD_CACHE_DIR = "data/processed/scorecards"   # one JSON file per match


def _get_cached_scorecard(match_id: str) -> list:
    """Load scorecard from disk if already fetched."""
    path = os.path.join(SCORECARD_CACHE_DIR, f"{match_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_scorecard(match_id: str, scorecard: list):
    """Persist scorecard to disk so we never fetch it again."""
    os.makedirs(SCORECARD_CACHE_DIR, exist_ok=True)
    path = os.path.join(SCORECARD_CACHE_DIR, f"{match_id}.json")
    with open(path, "w") as f:
        json.dump(scorecard, f)


def _fetch_ipl2026_stats():
    """
    Build standings + batting + bowling from CricAPI.

    Smart caching strategy — minimises API calls:
      - series_info  : 1 call total (gets match list + statuses)
      - match_scorecard: 1 call per match, cached to disk forever
        → After first full run, only NEW matches cost an API call
    """
    import requests
    from collections import defaultdict

    api_key = os.getenv("CRICAPI_KEY")

    # ── 1. Match list (1 API call) ──────────────────────────────────────
    r = requests.get(
        "https://api.cricapi.com/v1/series_info",
        params={"apikey": api_key, "id": IPL_2026_SERIES_ID},
        timeout=12,
    )
    resp      = r.json()
    if resp.get("status") != "success":
        raise Exception(f"CricAPI error: {resp.get('reason', 'unknown')}")

    matches   = resp.get("data", {}).get("matchList", [])
    completed = [m for m in matches if m.get("matchEnded")]

    # ── 2. Points table from match statuses (no extra API calls) ────────
    team_stats = defaultdict(lambda: {"played": 0, "won": 0, "lost": 0, "nr": 0, "pts": 0})
    for m in completed:
        status = m.get("status", "")
        teams  = m.get("teams", [])
        if len(teams) < 2:
            continue
        t1, t2 = teams[0], teams[1]
        team_stats[t1]["played"] += 1
        team_stats[t2]["played"] += 1
        if "won by" in status:
            winner = status.split(" won by")[0].strip()
            loser  = t2 if winner == t1 else t1
            team_stats[winner]["won"] += 1;  team_stats[winner]["pts"] += 2
            team_stats[loser]["lost"] += 1
        else:
            for t in (t1, t2):
                team_stats[t]["nr"] += 1;  team_stats[t]["pts"] += 1

    standings = sorted(
        [{"team": t, "matches_played": s["played"], "wins": s["won"],
          "losses": s["lost"], "nr": s["nr"], "points": s["pts"]}
         for t, s in team_stats.items()],
        key=lambda x: (-x["points"], -x["wins"]),
    )

    # ── 3. Batting & bowling — disk-cached scorecards ───────────────────
    bat  = defaultdict(lambda: {"runs":0,"balls":0,"fours":0,"sixes":0,
                                 "inn":0,"outs":0,"fifties":0,"hundreds":0})
    bowl = defaultdict(lambda: {"wickets":0,"runs":0,"balls":0,"inn":0})

    new_fetches = 0
    for m in completed:
        mid = m["id"]
        sc  = _get_cached_scorecard(mid)   # free — reads from disk

        if sc is None:
            # Not cached yet — fetch from CricAPI (costs 1 call)
            try:
                resp2 = requests.get(
                    "https://api.cricapi.com/v1/match_scorecard",
                    params={"apikey": api_key, "id": mid},
                    timeout=10,
                ).json()
                if resp2.get("status") == "success":
                    sc = resp2.get("data", {}).get("scorecard", [])
                    _save_scorecard(mid, sc)   # cache forever
                    new_fetches += 1
                else:
                    sc = []
            except Exception:
                sc = []

        # Aggregate batting
        for inning in sc:
            for b in inning.get("batting", []):
                name = (b.get("batsman") or {}).get("name") \
                       if isinstance(b.get("batsman"), dict) else b.get("batsman", "")
                if not name:
                    continue
                r_val = int(b.get("r", 0) or 0)
                balls = int(b.get("b", 0) or 0)
                bat[name]["runs"]  += r_val
                bat[name]["balls"] += balls
                bat[name]["fours"] += int(b.get("4s", 0) or 0)
                bat[name]["sixes"] += int(b.get("6s", 0) or 0)
                bat[name]["inn"]   += 1
                dismissal = b.get("dismissal", "")
                if dismissal and dismissal not in ("not out", "retired hurt", "retired out"):
                    bat[name]["outs"] += 1
                if r_val >= 100: bat[name]["hundreds"] += 1
                elif r_val >= 50: bat[name]["fifties"]  += 1

            # Aggregate bowling
            for bw in inning.get("bowling", []):
                name = (bw.get("bowler") or {}).get("name") \
                       if isinstance(bw.get("bowler"), dict) else bw.get("bowler", "")
                if not name:
                    continue
                overs   = str(bw.get("o", "0") or "0").split(".")
                balls_b = int(overs[0]) * 6 + (int(overs[1]) if len(overs) > 1 else 0)
                bowl[name]["wickets"] += int(bw.get("w", 0) or 0)
                bowl[name]["runs"]    += int(bw.get("r", 0) or 0)
                bowl[name]["balls"]   += balls_b
                bowl[name]["inn"]     += 1

    top_batters = sorted(
        [{"player": p, "total_runs": s["runs"],
          "batting_avg": round(s["runs"]/s["outs"],1) if s["outs"] else None,
          "strike_rate": round(s["runs"]/s["balls"]*100,1) if s["balls"] else 0,
          "fifties": s["fifties"], "hundreds": s["hundreds"], "sixes": s["sixes"]}
         for p, s in bat.items() if s["inn"] >= 2],
        key=lambda x: -x["total_runs"],
    )[:10]

    top_bowlers = sorted(
        [{"player": p, "wickets": s["wickets"],
          "economy": round(s["runs"]/s["balls"]*6,2) if s["balls"] else 0,
          "bowling_avg": round(s["runs"]/s["wickets"],1) if s["wickets"] else None}
         for p, s in bowl.items() if s["inn"] >= 2 and s["wickets"] > 0],
        key=lambda x: -x["wickets"],
    )[:10]

    return standings, top_batters, top_bowlers


def _load_stats_cache():
    """Load stats from disk cache if it exists and is from today."""
    try:
        if not os.path.exists(STATS_CACHE_FILE):
            return None
        with open(STATS_CACHE_FILE, "r") as f:
            cached = json.load(f)
        # Valid only if cached today
        if cached.get("date") == datetime.date.today().isoformat():
            return cached["standings"], cached["top_batters"], cached["top_bowlers"]
    except Exception:
        pass
    return None


def _save_stats_cache(standings, top_batters, top_bowlers):
    """Persist stats to disk so they survive server restarts."""
    try:
        os.makedirs(os.path.dirname(STATS_CACHE_FILE), exist_ok=True)
        with open(STATS_CACHE_FILE, "w") as f:
            json.dump({
                "date":        datetime.date.today().isoformat(),
                "standings":   standings,
                "top_batters": top_batters,
                "top_bowlers": top_bowlers,
            }, f)
    except Exception:
        pass


@app.get("/api/stats")
async def get_stats(refresh: bool = False):
    """
    Return live IPL 2026 stats.
    - Cached to disk daily — persists across server restarts.
    - Add ?refresh=true to force a fresh fetch.
    """
    try:
        # Try disk cache first (valid for today)
        if not refresh:
            cached = _load_stats_cache()
            if cached:
                standings, top_batters, top_bowlers = cached
                return {
                    "standings":   standings,
                    "top_batters": top_batters,
                    "top_bowlers": top_bowlers,
                    "source":      "CricAPI (cached today)",
                }

        # Fetch from CricAPI (scorecards cached per-match on disk)
        loop = asyncio.get_event_loop()
        standings, top_batters, top_bowlers = await loop.run_in_executor(
            None, _fetch_ipl2026_stats
        )
        _save_stats_cache(standings, top_batters, top_bowlers)

        return {
            "standings":   standings,
            "top_batters": top_batters,
            "top_bowlers": top_bowlers,
            "source":      "CricAPI live",
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/squads")
async def get_squads():
    """Return all IPL 2026 squads grouped by team."""
    try:
        import sqlite3
        db_path = os.getenv("DB_PATH", "data/processed/ipl.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT player, team, role, is_overseas, country
            FROM current_squads
            WHERE season=2026
            ORDER BY team, is_overseas, role
        """).fetchall()
        conn.close()

        # Group by team
        from collections import defaultdict
        squads = defaultdict(list)
        for r in rows:
            squads[r["team"]].append({
                "player": r["player"],
                "role": r["role"],
                "overseas": bool(r["is_overseas"]),
                "country": r["country"],
            })

        return {"squads": dict(squads)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "claude-sonnet-4-6", "season": "IPL 2026"}
