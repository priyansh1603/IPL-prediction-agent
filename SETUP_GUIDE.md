# IPL Prediction Agent — Setup Guide

## Project Structure

```
IPL prediction agent/
├── data/
│   ├── raw/
│   │   ├── ipl_json/       ← 1,183 IPL match JSONs
│   │   └── t20s_json/      ← 5,146 T20 match JSONs
│   └── processed/
│       ├── ipl.db          ← SQLite database (created by setup)
│       └── chroma_db/      ← Vector store (created by setup)
├── pipeline/
│   ├── parse.py            ← JSON → SQLite (matches + deliveries)
│   ├── features.py         ← Derived stats tables
│   └── embed.py            ← ChromaDB vector store
├── agent/
│   ├── tools.py            ← SQL query, semantic search, win probability
│   └── agent.py            ← Claude agent with tool use
├── bot/
│   └── telegram_bot.py     ← Telegram bot handler
├── setup.py                ← Run all pipeline steps at once
├── requirements.txt
└── .env.example
```

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `ANTHROPIC_API_KEY` — get from https://console.anthropic.com
- `TELEGRAM_BOT_TOKEN` — create a bot via @BotFather on Telegram

---

## Step 3 — Run the data pipeline (one-time)

```bash
python setup.py
```

This will:
1. Parse all 6,329 JSON files into SQLite (~2-3 minutes)
2. Build 9 derived stats tables (batting, bowling, h2h, venue, etc.)
3. Build ChromaDB vector store with match narratives + player profiles (~5-10 minutes)

---

## Step 4 — Test the agent locally

```bash
python agent/agent.py "What are MI's chances against CSK at Wankhede?"
```

---

## Step 5 — Run the Telegram bot locally

```bash
python bot/telegram_bot.py
```

This runs in polling mode. Open Telegram, find your bot, and ask away.

---

## Step 6 — Deploy to Railway (production)

1. Push project to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Set environment variables in Railway dashboard:
   - `ANTHROPIC_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `WEBHOOK_URL` = `https://your-app-name.railway.app`
   - `PORT` = `8443`
4. **Important**: Before deploying, run `setup.py` locally to generate
   `data/processed/ipl.db` and `data/processed/chroma_db/`, then commit
   these to the repo (or upload via Railway volume).

---

## SQLite Tables Reference

| Table | Description |
|---|---|
| `matches` | One row per match — teams, venue, toss, result |
| `deliveries` | Ball-by-ball data — batter, bowler, runs, wickets |
| `player_batting_stats` | Batting stats grouped by player/season/venue |
| `player_bowling_stats` | Bowling stats grouped by player/season/venue |
| `team_stats` | Win rates per team/season |
| `head_to_head` | Team vs team records per venue |
| `venue_stats` | Avg scores, chasing win %, par scores |
| `toss_impact` | Toss win → match win correlation |
| `phase_batting_stats` | Powerplay/middle/death batting |
| `phase_bowling_stats` | Powerplay/middle/death bowling |
| `player_recent_form` | Last 10 innings batting summary |

---

## Example Questions the Bot Can Answer

- "What is the probability of MI winning vs CSK at Wankhede?"
- "Who are the top 5 run scorers in IPL 2024?"
- "How does Jasprit Bumrah perform in death overs?"
- "Which team has the best record chasing at Chinnaswamy?"
- "What's the average first innings score at Eden Gardens?"
- "How has Rohit Sharma performed against SRH historically?"
- "Who wins the toss most often and does it matter at Chepauk?"
