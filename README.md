# 🏏 IPL Prediction Agent

An AI-powered IPL 2026 prediction and analysis platform built with Claude (Anthropic), FastAPI, and real-time cricket data. Ask anything about IPL — match predictions, player form, win probabilities, live scores, and more.

**Live Demo:** [web-production-81f99.up.railway.app](https://web-production-81f99.up.railway.app)

---

## ✨ Features

- **🤖 AI Chat** — GPT-style chat window powered by Claude. Ask about today's match, player form, win predictions, head-to-head stats, and more
- **⚡ Live Streaming** — Responses stream in real-time with step-by-step progress (no more staring at a blank screen)
- **📊 Stats Dashboard** — Live IPL 2026 Points Table, Orange Cap (batting), Purple Cap (bowling) fetched from CricAPI
- **🏏 Squad Explorer** — Full IPL 2026 squads for all 10 teams
- **📰 Media Feed** — Latest highlights from Cricbuzz, IPL, ESPNcricinfo YouTube channels + news articles
- **🔥 Trending Questions** — Auto-generated from today's fixtures so you always know what to ask
- **📡 Live Scores** — Real-time match scores and toss updates
- **🎯 Match Simulation** — 200-iteration Monte Carlo simulation for win probability

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | Claude Sonnet (Anthropic) |
| Backend | FastAPI + Python 3.12 |
| Streaming | Server-Sent Events (SSE) |
| Cricket Data | CricAPI |
| Web Search | Tavily API |
| Frontend | Vanilla JS + CSS (no framework) |
| Deployment | Railway |

---

## 🚀 Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/priyansh1603/IPL-prediction-agent.git
cd IPL-prediction-agent
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
ANTHROPIC_API_KEY=sk-ant-...
CRICAPI_KEY=your_cricapi_key
TAVILY_API_KEY=tvly-...
```

| Key | Where to get it |
|-----|----------------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `CRICAPI_KEY` | [cricapi.com](https://cricapi.com) — free tier (100 calls/day) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) — free tier available |

### 5. Start the server
```bash
uvicorn web.app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## 💬 Example Questions

```
Who will win today's match between MI and CSK?
What are Virat Kohli's stats in IPL 2026?
Give me the win probability for RCB vs GT tonight
Who is leading the Orange Cap this season?
What's the current points table?
How has Jasprit Bumrah been bowling this season?
Head to head: KKR vs RR
```

---

## 🗂️ Project Structure

```
IPL-prediction-agent/
├── agent/
│   ├── agent.py          # Claude agent loop with tool use + SSE streaming
│   └── tools.py          # Tool definitions: web search, live scores, simulation
├── web/
│   ├── app.py            # FastAPI backend + API routes
│   └── static/
│       ├── index.html    # Frontend UI
│       ├── app.js        # Chat, streaming, stats, media logic
│       └── style.css     # Dark theme styling
├── ml/
│   ├── simulate.py       # Monte Carlo match simulation
│   └── matchup_engine.py # Batter vs bowler matchup analysis
├── pipeline/             # Data ingestion & feature engineering
├── data/
│   ├── current_squads.json   # IPL 2026 squad data
│   └── playing_xi.json       # Recent playing XIs
├── requirements.txt
├── Procfile              # Railway start command
└── railway.toml          # Railway deployment config
```

---

## 🌐 Deployment (Railway)

This project is configured for one-click Railway deployment.

1. Fork this repo
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add environment variables in Railway dashboard:
   - `ANTHROPIC_API_KEY`
   - `CRICAPI_KEY`
   - `TAVILY_API_KEY`
4. Railway auto-deploys on every push to `main`

---

## ⚠️ Limitations

- **CricAPI free tier**: 100 calls/day. Stats use smart per-match disk caching to minimise usage
- **No live DB on Railway**: The SQLite ball-by-ball database is not deployed (too large). The agent uses CricAPI + web search for live data
- **Historical stats**: Detailed ball-by-ball analysis only works when running locally with the full database

---

## 🔮 Roadmap

- [ ] CricAPI Pro integration for unlimited stats
- [ ] Fantasy team suggestion based on today's pitch/weather
- [ ] Player injury & availability alerts
- [ ] Telegram bot integration
- [ ] Multi-language support (Hindi)

---

## 📄 License

MIT License — free to use, modify and distribute.

---

Built with ❤️ for cricket fans by [priyansh1603](https://github.com/priyansh1603)
