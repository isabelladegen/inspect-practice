"""Pull every model in models.Model into Ollama, skipping ones already installed.

Run from the project root:  uv run python pull_models.py
Requires the Ollama server: ollama serve
"""
import json
import logging
import subprocess
import sys
import urllib.request
from os import path

from models import Model
from utils.config import OLLAMA_TAGS_URL, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(path.join(LOG_DIR, "pull_models.log"))],
)
log = logging.getLogger(__name__)


def installed_tags() -> set[str]:
    with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=5) as resp:
        return {m["name"] for m in json.load(resp).get("models", [])}


def main() -> int:
    try:
        installed = installed_tags()
    except OSError:
        log.error("Ollama server not reachable. Start it with: ollama serve")
        return 1

    log.info("=== Pull run started: %d models ===", len(Model))
    results: dict[str, str] = {}

    for model in Model:
        tag = model.ollama_tag
        if tag in installed:
            results[tag] = "skipped"
            log.info("SKIP    %s (already installed)", tag)
            continue
        log.info("PULL    %s", tag)
        proc = subprocess.run(["ollama", "pull", tag], stdin=subprocess.DEVNULL)
        results[tag] = "ok" if proc.returncode == 0 else "FAILED"
        log.info("%-7s %s", results[tag].upper(), tag)

    log.info("=== Summary ===")
    for tag, status in results.items():
        log.info("  %-7s %s", status, tag)

    return 1 if "FAILED" in results.values() else 0


if __name__ == "__main__":
    sys.exit(main())
