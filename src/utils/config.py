import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PROMPT_PATH = BASE_DIR / "data" / "prompts" / "prompts.json"

with open(PROMPT_PATH, "r") as f:
    PROMPTS = json.load(f)