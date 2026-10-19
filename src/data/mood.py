# loads the MOOD benchmark (mood-bench/mood-bench) and parses its role-marked conversations into chat messages.

import re

import pandas as pd
from huggingface_hub import hf_hub_download

REPO, REVISION = "mood-bench/mood-bench", "d0e460b92d408f19bdbec9de160a5f059c28314f"
ROLES = {"System": "system", "Human": "user", "Assistant": "assistant", "Function": "environment"}
MARKER = re.compile(r"\n\n(System|Human|Assistant|Function):")
HARMLESS_SHIFTS = ("emoji", "jargon", "swahili", "function-calling-yaml")
HARMFUL_SHIFTS = (
    "controlling",
    "function-calling-inappropriate",
    "function-calling-missing",
    "insecure-code",
    "jailbroken",
    "scheming",
    "sycophantic",
)


def load_split(split: str) -> pd.DataFrame:
    path = hf_hub_download(REPO, f"data/{split}-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
    return pd.read_parquet(path)


def parse_conversation(text: str) -> list[dict]:
    parts = MARKER.split(text)
    # parts[0] is whatever precedes the first marker, then alternating role, content
    assert not parts[0].strip(), f"text before the first role marker: {parts[0][:80]!r}"
    return [{"role": ROLES[r], "content": c.strip()} for r, c in zip(parts[1::2], parts[2::2])]


def is_scorable(messages: list[dict]) -> bool:
    return bool(messages) and messages[-1]["role"] == "assistant" and bool(messages[-1]["content"])
