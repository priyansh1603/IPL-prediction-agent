"""
agent.py — IPL RAG Agent powered by Claude
"""

import os
import json
import sys
import logging
import datetime
import anthropic
from dotenv import load_dotenv
from agent.tools import TOOL_DEFINITIONS, TOOL_DISPATCH

load_dotenv(override=True)
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOOL_ROUNDS = 5


def _build_system_prompt() -> str:
    """Build system prompt with today's date injected so agent never returns yesterday's results."""
    today = datetime.date.today().strftime("%A, %d %B %Y")   # e.g. Sunday, 12 April 2026
    today_short = datetime.date.today().strftime("%d %B %Y") # e.g. 12 April 2026

    return f"""You are an elite IPL prediction agent and cricket analyst.

TODAY'S DATE: {today}
- When the user says "today" they mean {today}.
- Always include the date "{today_short}" in your web_search queries so you get today's matches, NOT yesterday's.
- Example search: "IPL match {today_short} schedule result"

## IPL 2026 OFFICIAL SQUADS (use these — do NOT rely on memory)

Chennai Super Kings: Ruturaj Gaikwad, MS Dhoni, Sanju Samson, Dewald Brevis, Ayush Mhatre, Shivam Dube, Jamie Overton, Khaleel Ahmed, Noor Ahmad, Anshul Kamboj, Mukesh Choudhary, Shreyas Gopal, Matt Henry, Rahul Chahar, Spencer Johnson, Akeal Hosein, Sarfaraz Khan, Kartik Sharma, Urvil Patel, Prashant Veer, Matthew Short, Gurjapneet Singh, Zak Foulkes

Delhi Capitals: KL Rahul, Karun Nair, Prithvi Shaw, Abishek Porel, David Miller, Ben Duckett, Pathum Nissanka, Tristan Stubbs, Axar Patel, Nitish Rana, Sameer Rizvi, Ashutosh Sharma, Mitchell Starc, Kuldeep Yadav, T Natarajan, Mukesh Kumar, Lungi Ngidi, Kyle Jamieson

Gujarat Titans: Shubman Gill, Sai Sudharsan, Shahrukh Khan, Jos Buttler, Glenn Phillips, Washington Sundar, Rahul Tewatia, Rashid Khan, Kagiso Rabada, Mohammed Siraj, Prasidh Krishna, Jason Holder, Jayant Yadav, Sai Kishore

Kolkata Knight Riders: Ajinkya Rahane, Rinku Singh, Cameron Green, Rovman Powell, Finn Allen, Sunil Narine, Rachin Ravindra, Varun Chakaravarthy, Harshit Rana, Matheesha Pathirana, Blessing Muzarabani, Angkrish Raghuvanshi, Rahul Tripathi, Umran Malik

Lucknow Super Giants: Rishabh Pant, Nicholas Pooran, Josh Inglis, Aiden Markram, Matthew Breetzke, Mitchell Marsh, Wanindu Hasaranga, Abdul Samad, Shahbaz Ahmed, Ayush Badoni, Mohammad Shami, Mayank Yadav, Avesh Khan, Anrich Nortje, Mohsin Khan, Prince Yadav

Mumbai Indians: Rohit Sharma, Suryakumar Yadav, Tilak Varma, Hardik Pandya, Ryan Rickelton, Quinton de Kock, Jasprit Bumrah, Trent Boult, Deepak Chahar, Will Jacks, Mitchell Santner, Sherfane Rutherford, Shardul Thakur

Punjab Kings: Shreyas Iyer, Prabhsimran Singh, Shashank Singh, Nehal Wadhera, Priyansh Arya, Musheer Khan, Marcus Stoinis, Marco Jansen, Harpreet Brar, Arshdeep Singh, Yuzvendra Chahal, Azmatullah Omarzai, Lockie Ferguson, Xavier Bartlett, Vishnu Vinod

Rajasthan Royals: Yashasvi Jaiswal, Riyan Parag, Dhruv Jurel, Vaibhav Suryavanshi, Ravindra Jadeja, Shimron Hetmyer, Ravi Bishnoi, Jofra Archer, Nandre Burger, Tushar Deshpande, Sandeep Sharma, Kwena Maphaka, Yudhvir Singh, Shubham Dubey

Royal Challengers Bengaluru: Virat Kohli, Rajat Patidar, Devdutt Padikkal, Phil Salt, Jitesh Sharma, Tim David, Krunal Pandya, Jacob Bethell, Bhuvneshwar Kumar, Josh Hazlewood, Yash Dayal, Jacob Duffy, Venkatesh Iyer, Romario Shepherd, Nuwan Thushara

Sunrisers Hyderabad: Travis Head, Ishan Kishan, Heinrich Klaasen, Abhishek Sharma, Nitish Kumar Reddy, Liam Livingstone, Harshal Patel, Harsh Dubey, Pat Cummins, Jaydev Unadkat, Kamindu Mendis, Brydon Carse, Shivam Mavi

## CRITICAL SQUAD RULES
- KL Rahul is at DELHI CAPITALS — NOT LSG. Never say he plays for LSG.
- Ravindra Jadeja is at RAJASTHAN ROYALS — NOT CSK.
- Sanju Samson is at CHENNAI SUPER KINGS — NOT RR.
- Rishabh Pant is at LUCKNOW SUPER GIANTS — NOT DC.
- Any player NOT listed above for a team does NOT play for that team in IPL 2026.

## WHAT YOU CAN AND CANNOT REPORT AS FACT

✅ SAFE to report as current 2026 stats:
- `ipl_2026_batting` / `ipl_2026_bowling` fields in get_player_profile — these are REAL 2026 scorecards
- Web search results that mention specific 2026 match scores
- get_live_score results

❌ NEVER report as current form or "last 5 matches":
- `historical_rolling_avg` in get_player_profile — this is a MULTI-SEASON average, NOT 2026
- `career_batting_alltime` / `career_bowling_alltime` — career totals across all seasons
- Any number from the database without a `ipl_2026_` prefix

If a player has no `ipl_2026_batting` data, say: "No 2026 stats available yet for this player."
NEVER invent stats. NEVER say "in last 5 matches" unless web_search confirms it.

## Tool Usage — Use the MINIMUM tools needed

### "Who will score most runs / take most wickets today?"
web_search (include date "{today_short}") → get_player_profile (top 2 candidates) → answer

### "Who will win today?" / "Win probability?"
web_search (include date) → resolve_playing_xi → get_matchup_analysis → run_match_simulation → answer

### "How is [player] performing?" / Player form
web_search ("{today_short} [player] IPL 2026 form") → answer

### LIVE match
get_live_score → answer

### Historical stats
query_stats → answer

## STRICT RULES
- **Max 4 tool calls per response**
- **Never hallucinate player-team assignments** — use only the squads listed above
- **Never report rolling DB stats as current 2026 form**
- **Always include "{today_short}" in web searches for today's questions**
- **Quantify everything**: "58% win probability" beats "favourites"
"""


# Human-readable labels for each tool (shown in UI while agent thinks)
TOOL_LABELS = {
    "web_search":              "🔍 Searching the web for latest match info...",
    "get_live_score":          "📡 Checking live scores & toss...",
    "resolve_playing_xi":      "📋 Building probable Playing XIs from recent matches...",
    "get_matchup_analysis":    "⚔️ Analysing team vs team head-to-head...",
    "predict_player_matchup":  "🏏 Running batter vs bowler matchup analysis...",
    "get_player_profile":      "👤 Fetching player form & stats...",
    "run_match_simulation":    "⚡ Simulating the match (200 iterations)...",
    "query_stats":             "📊 Querying ball-by-ball database...",
    "semantic_search":         "🔎 Searching knowledge base...",
    "get_current_squad":       "📋 Verifying IPL 2026 squad...",
    "compute_win_probability": "🎯 Computing win probability...",
}


def run_agent(
    user_message: str,
    conversation_history: list = None,
    on_tool_call=None,
) -> tuple[str, list]:
    """
    Run the agent for one user turn.

    Args:
        user_message: The user's question
        conversation_history: Previous messages for multi-turn context (optional)
        on_tool_call: Optional callback(tool_name, label) fired before each tool runs

    Returns:
        (assistant_response_text, updated_conversation_history)
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    system_prompt = _build_system_prompt()   # fresh date every call

    messages = list(conversation_history) if conversation_history else []
    messages.append({"role": "user", "content": user_message})

    for round_num in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
            return text, messages

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input
                    label = TOOL_LABELS.get(tool_name, f"⚙️ Running {tool_name}...")
                    log.info(f"Tool call: {tool_name}({json.dumps(tool_args)[:120]})")

                    if on_tool_call:
                        try:
                            on_tool_call(tool_name, label)
                        except Exception:
                            pass

                    try:
                        result = TOOL_DISPATCH[tool_name](tool_args)
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                        log.error(f"Tool error: {e}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        break

    return "I wasn't able to complete this analysis. Please try rephrasing your question.", messages


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    question = " ".join(sys.argv[1:]) or "What are MI's chances against CSK at Wankhede?"
    print(f"\nQuestion: {question}\n")
    answer, _ = run_agent(question)
    print(f"Answer:\n{answer}")
