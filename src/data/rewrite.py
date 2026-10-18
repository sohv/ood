# prompts and parsing for rewriting pools into the post-training mix's style, plus the content-preservation judge.

import re

import numpy as np

REWRITE_TEMPLATE = """Below are example messages that users sent to an AI assistant.

<examples>
{examples}
</examples>

Rewrite the message below so that it reads like one of the examples: the same kind of phrasing, tone, structure and length that a user would use when writing to an AI assistant. Keep the same request and every fact needed to answer it. Do not answer it, do not add facts, and do not add any commentary. Output only the rewritten message.

<message>
{message}
</message>"""

JUDGE_TEMPLATE = """Here is an original message and a rewritten version of it.

<original>
{original}
</original>

<rewrite>
{rewrite}
</rewrite>

Does the rewrite ask for the same thing as the original and keep the facts needed to answer it? Answer with YES or NO as the first word."""


def is_mostly_ascii(text: str, threshold: float = 0.98) -> bool:
    return sum(c.isascii() for c in text) / max(len(text), 1) >= threshold


def select_examples(rows: list[dict], n: int, seed: int, min_words: int = 10, max_words: int = 120) -> list[str]:
    eligible = [r["text"] for r in rows if min_words <= len(r["text"].split()) <= max_words and is_mostly_ascii(r["text"])]
    picks = np.random.default_rng(seed).choice(len(eligible), size=n, replace=False)
    return [eligible[i] for i in picks]


def rewrite_prompt(examples: list[str], message: str) -> str:
    joined = "\n\n".join(f"<example>\n{e}\n</example>" for e in examples)
    return REWRITE_TEMPLATE.format(examples=joined, message=message)


def judge_prompt(original: str, rewrite: str) -> str:
    return JUDGE_TEMPLATE.format(original=original, rewrite=rewrite)


def parse_judge(response: str | None) -> bool | None:
    if response is None:
        return None
    first = re.sub(r"[^a-z]", "", response.strip().split()[0].lower()) if response.strip() else ""
    return {"yes": True, "no": False}.get(first)
