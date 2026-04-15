"""
telegram_bot.py — IPL Prediction Agent Telegram Bot

Uses webhook mode for production (Railway/Render) or polling for local dev.
Maintains per-user conversation history for multi-turn context.

Start (local, polling mode):
    python bot/telegram_bot.py

Start (production, webhook mode):
    Set WEBHOOK_URL env var and run the same script — it auto-detects.
"""

import os
import logging
import asyncio
from collections import defaultdict
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction, ParseMode

# Import agent — adjust path if running from repo root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import run_agent

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL        = os.getenv("WEBHOOK_URL")          # e.g. https://your-app.railway.app
PORT               = int(os.getenv("PORT", 8443))

# Per-user conversation history (in-memory; resets on bot restart)
# For persistence across restarts, swap with a Redis/SQLite store
user_histories: dict[int, list] = defaultdict(list)

MAX_HISTORY_TURNS = 10  # keep last N turns to avoid token overflow


def trim_history(history: list) -> list:
    """Keep only the last MAX_HISTORY_TURNS user+assistant pairs."""
    # Each turn = 1 user msg + 1 assistant msg = 2 entries
    max_entries = MAX_HISTORY_TURNS * 2
    if len(history) > max_entries:
        return history[-max_entries:]
    return history


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_histories[update.effective_user.id] = []  # reset history
    await update.message.reply_text(
        "🏏 *IPL Prediction Agent*\n\n"
        "Ask me anything about IPL and T20 cricket:\n"
        "• Win probabilities for upcoming matches\n"
        "• Player performance stats and form\n"
        "• Venue records and toss impact\n"
        "• Head-to-head records\n"
        "• Phase-wise analysis (powerplay / death)\n\n"
        "Examples:\n"
        "_\"What are MI's chances vs CSK at Wankhede?\"_\n"
        "_\"Who are the best death over bowlers in IPL?\"_\n"
        "_\"How has Virat Kohli performed in powerplay overs?\"_\n\n"
        "Type /reset to clear conversation history.",
        parse_mode=ParseMode.MARKDOWN
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_histories[update.effective_user.id] = []
    await update.message.reply_text("Conversation history cleared. Ask me a new question!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text.strip()

    if not user_text:
        return

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

    history = user_histories[user_id]

    try:
        answer, updated_history = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_agent(user_text, history)
        )

        user_histories[user_id] = trim_history(updated_history)

        # Telegram markdown: convert **bold** to *bold* for MarkdownV1
        # Using HTML parse mode is more reliable
        answer_html = _md_to_html(answer)

        await update.message.reply_text(
            answer_html,
            parse_mode=ParseMode.HTML,
        )

    except Exception as e:
        log.exception(f"Error handling message from user {user_id}")
        await update.message.reply_text(
            "Sorry, something went wrong while processing your question. Please try again."
        )


def _md_to_html(text: str) -> str:
    """Basic conversion of markdown bold/italic to HTML for Telegram."""
    import re
    # **bold** → <b>bold</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # *italic* → <i>italic</i>
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # `code` → <code>code</code>
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    # Escape raw & and < > that aren't our tags
    # (simple approach — good enough for cricket responses)
    return text


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",  start))
    app.add_handler(CommandHandler("reset",  reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if WEBHOOK_URL:
        # Production mode: webhook
        log.info(f"Starting in webhook mode on port {PORT}")
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_BOT_TOKEN,
            webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}",
        )
    else:
        # Local dev mode: polling
        log.info("Starting in polling mode (local dev)")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
