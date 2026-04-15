"""
setup.py — One-time setup: run all pipeline stages in order.

Run from project root:
    python setup.py
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run(label, module):
    log.info(f"\n{'='*60}\nSTEP: {label}\n{'='*60}")
    result = subprocess.run([sys.executable, "-m", module], check=True)
    return result


if __name__ == "__main__":
    run("1/7 — Parsing JSONs into SQLite",        "pipeline.parse")
    run("2/7 — Building core feature tables",     "pipeline.features")
    run("3/7 — Building advanced feature tables", "pipeline.advanced_features")
    run("4/7 — Building vector store",            "pipeline.embed")
    run("5/7 — Syncing current squads & XIs",     "pipeline.sync_squads")
    run("6/7 — Building ML feature dataset",      "pipeline.ml_features")
    run("7/7 — Training ML models",               "ml.train")
    log.info("\nSetup complete! Run the bot with: python bot/telegram_bot.py")
