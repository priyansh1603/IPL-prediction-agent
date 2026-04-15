/* ── State ─────────────────────────────────────────────────── */
let sessionId = "session_" + Math.random().toString(36).slice(2, 10);
let isStreaming = false;
let chatHistory = [];
let statsLoaded = false;
let squadsData = null;
let activeTeam = null;

/* ── Init ──────────────────────────────────────────────────── */
let allMediaItems = [];

document.addEventListener("DOMContentLoaded", () => {
  loadTrending();
  loadTodayMatches();
  loadMedia();
});

/* ── Media Feed (YouTube + News) ──────────────────────────── */
async function loadMedia() {
  try {
    const res  = await fetch("/api/media");
    const data = await res.json();
    allMediaItems = data.items || [];
    renderMedia(allMediaItems);
  } catch (e) {
    document.getElementById("media-grid").innerHTML =
      "<p style='color:var(--text2);font-size:13px'>Could not load media feed.</p>";
  }
}

function filterMedia(type) {
  // Update tab buttons
  document.querySelectorAll(".media-tab").forEach(b => b.classList.remove("active"));
  document.getElementById("tab-" + type).classList.add("active");

  const filtered = type === "all"
    ? allMediaItems
    : allMediaItems.filter(i => i.type === type);
  renderMedia(filtered);
}

function renderMedia(items) {
  const grid = document.getElementById("media-grid");
  if (!items.length) {
    grid.innerHTML = "<p style='color:var(--text2);font-size:13px;padding:12px 0'>No items found.</p>";
    return;
  }

  grid.innerHTML = items.map((item, idx) => {
    const isVideo   = item.type === "video";
    const safeLink  = (item.link || "").replace(/"/g, "&quot;");
    const thumbHtml = item.thumb
      ? `<img src="${item.thumb}" alt="" class="media-thumb" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">`
      : "";

    return `
      <div class="media-card ${item.type}" onclick="openLink('${safeLink}')" title="${(item.title||"").replace(/'/g,"&apos;")}">
        <div class="media-img-wrap">
          ${thumbHtml}
          <div class="media-thumb-placeholder" style="${item.thumb ? "display:none" : ""}">${isVideo ? "▶" : "📰"}</div>
          ${isVideo ? '<span class="play-badge">▶</span>' : '<span class="play-badge">↗</span>'}
        </div>
        <div class="media-info">
          <div class="media-source">
            ${isVideo ? "🎬" : "📰"} <strong>${item.channel}</strong>
            <span class="media-date">${item.pub || ""}</span>
          </div>
          <div class="media-title">${item.title || ""}</div>
          ${item.snippet ? `<div class="media-snippet">${item.snippet}…</div>` : ""}
        </div>
      </div>`;
  }).join("");
}

/* ── Link opener ───────────────────────────────────────────── */
function openLink(url) {
  if (!url) return;
  window.open(url, "_blank", "noopener,noreferrer");
}

/* ── Page Navigation ───────────────────────────────────────── */
function showPage(page) {
  // Hide all views
  document.getElementById("home-view").classList.add("hidden");
  document.getElementById("chat-view").classList.add("hidden");
  document.getElementById("stats-view").classList.add("hidden");
  document.getElementById("squads-view").classList.add("hidden");

  // Update nav
  document.querySelectorAll(".nav-link").forEach(el => el.classList.remove("active"));

  if (page === "home") {
    document.getElementById("home-view").classList.remove("hidden");
    document.getElementById("nav-home").classList.add("active");
  } else if (page === "stats") {
    document.getElementById("stats-view").classList.remove("hidden");
    document.getElementById("nav-stats").classList.add("active");
    if (!statsLoaded) loadStats();
  } else if (page === "squads") {
    document.getElementById("squads-view").classList.remove("hidden");
    document.getElementById("nav-squads").classList.add("active");
    if (!squadsData) loadSquads();
  }
}

/* ── Stats Page ────────────────────────────────────────────── */
async function refreshStats() {
  statsLoaded = false;
  document.getElementById("standings-table").innerHTML = "<div class='loading-text'>Fetching fresh data from CricAPI...</div>";
  document.getElementById("batters-table").innerHTML   = "<div class='loading-text'>Loading...</div>";
  document.getElementById("bowlers-table").innerHTML   = "<div class='loading-text'>Loading...</div>";
  await loadStats(true);
}

async function loadStats(forceRefresh = false) {
  try {
    const url = forceRefresh ? "/api/stats?refresh=true" : "/api/stats";
    const res = await fetch(url);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // Standings
    const standingsEl = document.getElementById("standings-table");
    if (data.standings && data.standings.length) {
      const maxPts = Math.max(...data.standings.map(t => t.points || 0), 1);
      standingsEl.innerHTML = `
        <table class="data-table">
          <thead><tr><th>#</th><th>Team</th><th>M</th><th>W</th><th>L</th><th>NR</th><th>Pts</th></tr></thead>
          <tbody>
            ${data.standings.map((t, i) => `
              <tr>
                <td>${i + 1}</td>
                <td><strong>${t.team}</strong></td>
                <td>${t.matches_played || 0}</td>
                <td class="green">${t.wins || 0}</td>
                <td class="red">${t.losses || 0}</td>
                <td>${t.nr || 0}</td>
                <td><span class="pct-bar" style="--pct:${Math.round((t.points||0)/maxPts*100)}%">${t.points || 0}</span></td>
              </tr>`).join("")}
          </tbody>
        </table>`;
    } else {
      standingsEl.innerHTML = "<p class='no-data'>No 2026 data yet — season may just be starting.</p>";
    }

    // Top Batters
    const battersEl = document.getElementById("batters-table");
    if (data.top_batters && data.top_batters.length) {
      battersEl.innerHTML = `
        <table class="data-table">
          <thead><tr><th>#</th><th>Player</th><th>Runs</th><th>Avg</th><th>SR</th><th>6s</th></tr></thead>
          <tbody>
            ${data.top_batters.map((p, i) => `
              <tr>
                <td>${i + 1}</td>
                <td><strong>${p.player}</strong></td>
                <td class="orange">${p.total_runs || 0}</td>
                <td>${p.batting_avg != null ? Number(p.batting_avg).toFixed(1) : '—*'}</td>
                <td>${(p.strike_rate||0).toFixed(1)}</td>
                <td>${p.sixes || 0}</td>
              </tr>`).join("")}
          </tbody>
        </table>`;
    } else {
      battersEl.innerHTML = "<p class='no-data'>Stats updating...</p>";
    }

    // Top Bowlers
    const bowlersEl = document.getElementById("bowlers-table");
    if (data.top_bowlers && data.top_bowlers.length) {
      bowlersEl.innerHTML = `
        <table class="data-table">
          <thead><tr><th>#</th><th>Player</th><th>Wkts</th><th>Econ</th><th>Avg</th></tr></thead>
          <tbody>
            ${data.top_bowlers.map((p, i) => `
              <tr>
                <td>${i + 1}</td>
                <td><strong>${p.player}</strong></td>
                <td class="purple">${p.wickets || 0}</td>
                <td>${(p.economy||0).toFixed(2)}</td>
                <td>${(p.bowling_avg||0).toFixed(1)}</td>
              </tr>`).join("")}
          </tbody>
        </table>`;
    } else {
      bowlersEl.innerHTML = "<p class='no-data'>Stats updating...</p>";
    }

    // Show source + timestamp
    const updEl = document.getElementById("stats-updated");
    if (updEl) updEl.textContent = `${data.source || ""} · ${new Date().toLocaleTimeString()}`;

    statsLoaded = true;
  } catch (e) {
    document.getElementById("standings-table").innerHTML = `<p class='no-data'>Error: ${e.message}</p>`;
  }
}

/* ── Squads Page ───────────────────────────────────────────── */
const TEAM_COLORS = {
  "Mumbai Indians":           "#004BA0",
  "Chennai Super Kings":      "#F9CD05",
  "Royal Challengers Bengaluru": "#CC0000",
  "Kolkata Knight Riders":    "#3B2F7F",
  "Delhi Capitals":           "#0078BC",
  "Rajasthan Royals":         "#EA1A85",
  "Sunrisers Hyderabad":      "#F7A721",
  "Punjab Kings":             "#AAAAAA",
  "Gujarat Titans":           "#1C2B5E",
  "Lucknow Super Giants":     "#A0E6FF",
};

const TEAM_SHORT = {
  "Mumbai Indians": "MI", "Chennai Super Kings": "CSK",
  "Royal Challengers Bengaluru": "RCB", "Kolkata Knight Riders": "KKR",
  "Delhi Capitals": "DC", "Rajasthan Royals": "RR",
  "Sunrisers Hyderabad": "SRH", "Punjab Kings": "PBKS",
  "Gujarat Titans": "GT", "Lucknow Super Giants": "LSG",
};

async function loadSquads() {
  try {
    const res = await fetch("/api/squads");
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    squadsData = data.squads;

    // Build team filter buttons
    const filterEl = document.getElementById("team-filter");
    const teams = Object.keys(squadsData);
    filterEl.innerHTML = teams.map(t => `
      <button class="team-btn" onclick="filterTeam('${t}')"
        style="--team-color:${TEAM_COLORS[t] || '#f97316'}">
        ${TEAM_SHORT[t] || t}
      </button>`).join("");

    // Show all teams
    renderSquads(teams);
  } catch (e) {
    document.getElementById("squads-grid").innerHTML = `<p class='no-data'>Error: ${e.message}</p>`;
  }
}

function filterTeam(team) {
  activeTeam = activeTeam === team ? null : team;
  // Update button states
  document.querySelectorAll(".team-btn").forEach(btn => {
    btn.classList.toggle("active", btn.textContent.trim() === (TEAM_SHORT[team] || team) && activeTeam);
  });
  renderSquads(activeTeam ? [activeTeam] : Object.keys(squadsData));
}

function renderSquads(teams) {
  const grid = document.getElementById("squads-grid");
  grid.innerHTML = teams.map(team => {
    const players = squadsData[team] || [];
    const color = TEAM_COLORS[team] || "#f97316";
    return `
      <div class="squad-card">
        <div class="squad-header" style="border-color:${color}">
          <span class="squad-team">${team}</span>
          <span class="squad-short" style="color:${color}">${TEAM_SHORT[team] || ""}</span>
        </div>
        <div class="squad-players">
          ${players.map(p => `
            <div class="squad-player">
              <span class="player-name">${p.player}</span>
              <span class="player-role">${p.role || ""}</span>
              ${p.overseas ? `<span class="overseas-tag">🌍</span>` : ""}
            </div>`).join("")}
        </div>
      </div>`;
  }).join("");
}

/* ── Trending ──────────────────────────────────────────────── */
async function loadTrending() {
  try {
    const res = await fetch("/api/trending");
    const data = await res.json();
    const grid = document.getElementById("trending-grid");
    grid.innerHTML = data.questions
      .map(
        (q, i) => `
        <div class="trending-card" onclick="askQuestion('${q.replace(/'/g, "\\'")}')">
          <span class="trending-num">#${i + 1}</span>
          <span class="trending-text">${q}</span>
          <span class="trending-arrow">→</span>
        </div>`
      )
      .join("");
  } catch (e) {
    console.error("Trending load failed:", e);
  }
}

/* ── Today's Matches ───────────────────────────────────────── */
async function loadTodayMatches() {
  try {
    const res = await fetch("/api/today_matches");
    const data = await res.json();
    const list = document.getElementById("matches-list");
    const strip = document.getElementById("matches-strip");

    if (!data.matches || data.matches.length === 0) {
      strip.style.display = "none";
      return;
    }

    list.innerHTML = data.matches
      .slice(0, 4)
      .map((m) => {
        const isLive = m.status && m.status.toLowerCase().includes("live");
        return `
        <div class="match-card" onclick="askQuestion('Tell me about the ${m.teams || m.name || "match"} match today')">
          <div class="match-teams">${m.teams || m.name || "IPL Match"}</div>
          <div class="match-meta">
            ${isLive ? '<span class="match-live">● LIVE</span>' : m.date || "Today"}
            ${m.venue ? " · " + m.venue : ""}
          </div>
        </div>`;
      })
      .join("");
  } catch (e) {
    document.getElementById("matches-strip").style.display = "none";
  }
}

/* ── Hero Input ────────────────────────────────────────────── */
function handleHeroKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    startChat();
  }
}

function startChat(question) {
  const input = document.getElementById("hero-input");
  const q = question || input.value.trim();
  if (!q) return;

  showChatView();
  setTimeout(() => sendMessage(q), 100);
}

function askQuestion(q) {
  startChat(q);
}

/* ── View Toggle ───────────────────────────────────────────── */
function showChatView() {
  document.getElementById("home-view").classList.add("hidden");
  document.getElementById("chat-view").classList.remove("hidden");
  setTimeout(() => document.getElementById("chat-input").focus(), 150);
}

function newChat() {
  // Save current session to history
  if (chatHistory.length > 0) {
    addToSidebarHistory(chatHistory[0].question);
  }

  // Reset
  sessionId = "session_" + Math.random().toString(36).slice(2, 10);
  chatHistory = [];
  document.getElementById("chat-messages").innerHTML = "";
  document.getElementById("suggestion-chips").style.display = "flex";

  // Show home
  document.getElementById("home-view").classList.remove("hidden");
  document.getElementById("chat-view").classList.add("hidden");
  document.getElementById("hero-input").value = "";
}

/* ── Send Message ──────────────────────────────────────────── */
function handleChatKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

async function sendSuggestion(btn) {
  sendMessage(btn.textContent);
}

async function sendMessage(overrideText) {
  if (isStreaming) return;

  const input = document.getElementById("chat-input");
  const text = overrideText || input.value.trim();
  if (!text) return;

  // Clear input
  input.value = "";
  autoResize(input);
  document.getElementById("suggestion-chips").style.display = "none";

  chatHistory.push({ question: text, timestamp: Date.now() });

  // Add user bubble
  appendMessage("user", text);

  // Add assistant typing bubble
  const assistantId = "msg_" + Date.now();
  appendAssistantPlaceholder(assistantId);

  isStreaming = true;
  document.getElementById("send-btn").disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";
    let progressSteps = [];   // collect all progress labels
    let answerStarted = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const payload = JSON.parse(line.slice(6));

          if (payload.progress) {
            // Show live progress step (agent is working)
            progressSteps.push(payload.progress);
            updateAssistantProgress(assistantId, progressSteps);

          } else if (payload.text !== undefined) {
            // Final answer is streaming in
            if (!answerStarted) {
              answerStarted = true;
              // Clear progress steps, start showing actual answer
            }
            fullText += payload.text;
            updateAssistantBubble(assistantId, fullText, progressSteps);

            if (payload.done) break;
          }
        } catch {}
      }
    }
  } catch (e) {
    updateAssistantBubble(assistantId, "⚠️ Connection error. Please try again.", []);
  } finally {
    isStreaming = false;
    document.getElementById("send-btn").disabled = false;
    document.getElementById("chat-input").focus();
  }
}

/* ── Message DOM Helpers ───────────────────────────────────── */
function appendMessage(role, text) {
  const messages = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="msg-avatar">${role === "user" ? "👤" : "🏏"}</div>
    <div class="msg-bubble">${formatText(text)}</div>
  `;
  messages.appendChild(div);
  scrollToBottom();
}

function appendAssistantPlaceholder(id) {
  const messages = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "message assistant";
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">🏏</div>
    <div class="msg-bubble">
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
      <div class="progress-steps" id="progress-${id}"></div>
    </div>
  `;
  messages.appendChild(div);
  scrollToBottom();
}

function updateAssistantProgress(id, steps) {
  // Show each tool step as it fires — user can see agent is working
  const el = document.getElementById("progress-" + id);
  if (!el) return;
  el.innerHTML = steps.map((s, i) => `
    <div class="progress-step ${i === steps.length - 1 ? 'active' : 'done'}">
      ${i === steps.length - 1
        ? `<span class="step-spinner"></span>`
        : `<span class="step-check">✓</span>`}
      ${s}
    </div>`).join("");
  scrollToBottom();
}

function updateAssistantBubble(id, text, progressSteps = []) {
  const el = document.getElementById(id);
  if (!el) return;
  const bubble = el.querySelector(".msg-bubble");

  // Build: collapsed steps summary + answer
  const stepsHtml = progressSteps.length
    ? `<div class="steps-summary">${progressSteps.length} analysis steps completed ✓</div>`
    : "";

  bubble.innerHTML = stepsHtml + formatText(text);
  scrollToBottom();
}

function scrollToBottom() {
  const messages = document.getElementById("chat-messages");
  messages.scrollTop = messages.scrollHeight;
}

/* ── Text Formatter (basic markdown) ──────────────────────── */
function formatText(text) {
  return text
    // Bold **text**
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    // Italic *text*
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    // Code `text`
    .replace(/`(.*?)`/g, "<code>$1</code>")
    // Headers ### text
    .replace(/^### (.+)$/gm, "<h4 style='margin:12px 0 6px;font-size:14px;color:var(--accent)'>$1</h4>")
    .replace(/^## (.+)$/gm, "<h3 style='margin:14px 0 8px;font-size:15px;'>$1</h3>")
    // Bullet points
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>")
    // Numbered list
    .replace(/^\d+\. (.+)$/gm, "<li>$1</li>")
    // Horizontal rule
    .replace(/^---$/gm, "<hr style='border-color:var(--border);margin:12px 0'>")
    // Line breaks
    .replace(/\n\n/g, "</p><p style='margin-top:10px'>")
    .replace(/\n/g, "<br>")
    // Wrap in paragraph
    .replace(/^(.+)$/, "<p>$1</p>");
}

/* ── Sidebar History ───────────────────────────────────────── */
function addToSidebarHistory(question) {
  const history = document.getElementById("sidebar-history");
  const item = document.createElement("div");
  item.className = "history-item";
  item.textContent = question.slice(0, 40) + (question.length > 40 ? "..." : "");
  item.title = question;
  history.insertBefore(item, history.firstChild);
}

/* ── Auto-resize textarea ─────────────────────────────────── */
function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}
