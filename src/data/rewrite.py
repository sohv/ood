# prompts and parsing for rewriting pools into the post-training mix's style, plus the content-preservation judge.

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
    eligible = [
        r["text"] for r in rows if min_words <= len(r["text"].split()) <= max_words and is_mostly_ascii(r["text"])
    ]
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


LEXICAL_INSTRUCTION = """

Do not use any of the following words or any form of them (plural, verb tense, and so on). Where the original uses one, describe the same thing in plain everyday words instead, keeping the situation and the request:
{banned}"""

TOPIC_TEMPLATE = """Here is a message that a user sent to an AI assistant.

<message>
{message}
</message>

Whose expertise does this message mainly need: a LAWYER, a DOCTOR, or NEITHER? Answer with exactly one of LAWYER, DOCTOR or NEITHER as the first word."""

TOKEN = re.compile(r"[a-z]+")


def ban_list(pos_texts: list[str], neg_texts: list[str], n_top: int, seed: int) -> list[str]:
    # the top terms in each direction of a supervised tf-idf logistic, stop words and short or non-alphabetic terms removed
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000)
    x = vectorizer.fit_transform(pos_texts + neg_texts)
    y = np.r_[np.ones(len(pos_texts)), np.zeros(len(neg_texts))]
    coef = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(x, y).coef_[0]
    terms = vectorizer.get_feature_names_out()
    ok = np.array([t.isalpha() and len(t) >= 3 and t not in ENGLISH_STOP_WORDS for t in terms])
    order = np.argsort(coef)
    neg_top = [i for i in order if ok[i]][:n_top]
    pos_top = [i for i in order[::-1] if ok[i]][:n_top]
    return sorted({str(terms[i]) for i in pos_top + neg_top})


def lexical_rewrite_prompt(examples: list[str], message: str, banned: list[str]) -> str:
    prompt = rewrite_prompt(examples, message)
    head, tail = prompt.split("\n\n<message>")
    return head + LEXICAL_INSTRUCTION.format(banned=", ".join(banned)) + "\n\n<message>" + tail


def find_leaks(text: str, banned: list[str]) -> list[str]:
    return sorted(set(TOKEN.findall(text.lower())) & set(banned))


def topic_prompt(message: str) -> str:
    return TOPIC_TEMPLATE.format(message=message)


def parse_topic(response: str | None) -> str | None:
    if response is None or not response.strip():
        return None
    first = re.sub(r"[^a-z]", "", response.strip().split()[0].lower())
    return first if first in ("lawyer", "doctor", "neither") else None


def repair_prompt(examples: list[str], rewrite: str, banned: list[str], leaks: list[str]) -> str:
    return (
        lexical_rewrite_prompt(examples, rewrite, banned)
        + f"\n\nThe message above still uses these words, which are not allowed: {', '.join(leaks)}"
    )
