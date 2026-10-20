# question families for the generalisation experiments (exp25+): real questions, one list of texts per family.

import re

import pandas as pd
from huggingface_hub import hf_hub_download

from src.data.io import load_jsonl
from src.data.needles import wildchat_first_turns

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
CREATIVE = re.compile(
    r"^(write|compose|create)\s+(me\s+)?(a|an)\s+(short\s+)?(story|poem|song|script|fanfic)", re.IGNORECASE
)
ROLEPLAY = re.compile(
    r"\b(role-?play|let'?s pretend|pretend (you are|to be)|you are now|stay in character|act as my)\b", re.IGNORECASE
)
JAILBREAK = re.compile(
    r"deceased|grandma|grandmother|grandpa|grandfather|\bDAN\b|jailbreak|ignore (all )?previous|developer mode|"
    r"no restrictions|NAME_|machiavelli|\bAIM\b|from now on|uncensored|unfiltered|without any (moral|ethical)|\bkeys?\b",
    re.IGNORECASE,
)


def reddit(name: str) -> list[str]:
    return [r["text"] for r in load_jsonl(f"data/processed/exp03/reddit_{name}.jsonl")]


def creative_and_roleplay() -> tuple[list[str], list[str]]:
    texts = wildchat_first_turns([1, 2, 3, 4, 5, 6], english_only=True)
    creative = [t for t in texts if CREATIVE.search(t.strip())]
    roleplay = [t for t in texts if ROLEPLAY.search(t) and not JAILBREAK.search(t) and not CREATIVE.search(t.strip())]
    return creative, roleplay


def load_families(roleplay_path: str = "data/processed/exp25_roleplay/labelled.jsonl") -> dict[str, list[str]]:
    gsm = pd.read_parquet(
        hf_hub_download(
            "openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset", revision=GSM8K_REVISION
        )
    )
    creative, _ = creative_and_roleplay()
    return {
        "cooking": reddit("cooking"),
        "legal": [r["text"] for r in load_jsonl("data/processed/exp01/legal.jsonl")],
        "programming": reddit("learnprogramming"),
        "medical": [r["text"] for r in load_jsonl("data/processed/exp01/medical.jsonl")],
        "japanese": [
            r["text"] for r in load_jsonl("data/processed/exp18/sharechat.jsonl") if r.get("language") == "Japanese"
        ],
        "maths": gsm["question"].tolist(),
        "creative": creative,
        "roleplay": [
            r["text"] for r in load_jsonl(roleplay_path) if r["label"] == "ROLEPLAY" and not JAILBREAK.search(r["text"])
        ],
    }


ROLEPLAY_LABEL_PROMPT = """Below is a message a user sent to an AI chatbot. Classify it with exactly one label.

ROLEPLAY: the user asks the assistant to play a character or take part in a fictional scene or game, with no sexual content and no attempt to get around the assistant's rules.
JAILBREAK: the message tries to remove or bypass the assistant's rules, filters or ethics (including through a character or game).
EXPLICIT: the message asks for sexual or erotic content.
PERSONA_TASK: the user asks the assistant to act as an expert or professional to do a practical task (write, translate, code, explain), not a fictional scene.
OTHER: anything else.

<message>
{message}
</message>

Reply with the label only."""


def roleplay_candidates() -> list[str]:
    # every english wildchat first turn matching the roleplay pattern; jailbreak and explicit ones are removed by the llm label
    texts = wildchat_first_turns([1, 2, 3, 4, 5, 6], english_only=True)
    return [t for t in texts if ROLEPLAY.search(t) and not CREATIVE.search(t.strip()) and "NAME_" not in t]
