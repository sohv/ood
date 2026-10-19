# gap-discovery helpers for exp09: clustering, blind cluster naming via openrouter, coverage and gap matching.

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

LOGGER = logging.getLogger(__name__)

NAMING_PROMPT = """Below are {n} messages that users sent to an AI chatbot, separated by lines of ---.

{messages}

Reply in exactly this format and nothing else:
NAME: <a short name, at most 6 words, for what these messages have in common>
DESCRIPTION: <one sentence>
COHERENT: <YES if most of the messages clearly share a topic, task, format or language, otherwise NO>"""

NAMING_PROMPT_SHARE = """Below are {n} messages that users sent to an AI chatbot, separated by lines of ---.

{messages}

Find the largest group of these messages that share one topic, task, format or language. The other messages may be unrelated.
Reply in exactly this format and nothing else:
NAME: <a short name, at most 6 words, for what that group has in common>
DESCRIPTION: <one sentence about that group>
SHARED: <how many of the {n} messages belong to that group, as a number>"""


LABEL_PROMPT = """A group of messages that users sent to an AI chatbot was given this name and description.

NAME: {name}
DESCRIPTION: {description}

Which category fits the group best? Reply with exactly one word:
LEGAL (legal questions or legal advice)
MEDICAL (health or medical questions)
CODE (programming or software tasks)
OTHER (anything else)"""


def parse_label(reply: str | None, labels: list[str]) -> str | None:
    if not reply:
        return None
    hits = [(m.start(), lab) for lab in labels if (m := re.search(rf"\b{lab}\b", reply.upper()))]
    return min(hits)[1] if hits else None


def nearest_cosine(emb: np.ndarray, ref, chunk: int = 4096) -> np.ndarray:
    # ref is a normalised torch tensor on the gpu; emb rows are normalised
    return np.concatenate(
        [
            (torch.from_numpy(emb[i : i + chunk]).to(ref.device) @ ref.T).max(dim=1).values.cpu().numpy()
            for i in range(0, len(emb), chunk)
        ]
    )


def cluster(embeddings: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embeddings)


def centroid(embeddings: np.ndarray) -> np.ndarray:
    c = embeddings.mean(0)
    return c / (np.linalg.norm(c) + 1e-12)


def naming_prompt(texts: list[str], n: int, seed: int, max_chars: int = 300, template: str = NAMING_PROMPT) -> str:
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    shown = [" ".join(texts[i].split())[:max_chars] for i in pick]
    return template.format(n=len(shown), messages="\n---\n".join(shown))


def parse_naming(reply: str | None) -> dict:
    if reply is None:
        return {"name": None, "description": None, "coherent": None, "shared": None}

    def get(key: str) -> str | None:
        m = re.search(rf"^{key}:\s*(.+)$", reply, re.M)
        return m.group(1).strip() if m else None

    coherent, shared = get("COHERENT"), get("SHARED")
    return {
        "name": get("NAME"),
        "description": get("DESCRIPTION"),
        "coherent": None if coherent is None else coherent.upper().startswith("YES"),
        "shared": int(shared) if shared is not None and shared.isdigit() else None,
    }


def match_gap(c: np.ndarray, known: list[np.ndarray], threshold: float) -> int | None:
    sims = [float(c @ k) for k in known]
    return int(np.argmax(sims)) if sims and max(sims) >= threshold else None


async def name_clusters(
    client, model: str, prompts: list[str], cache_dir: str, max_concurrent: int = 10, attempts: int = 3
) -> list[str | None]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max_concurrent)
    errors: list[Exception] = []

    async def call(i: int, prompt: str) -> str | None:
        path = Path(cache_dir) / f"{hashlib.md5(f'{model}{prompt}'.encode()).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]
        async with sem:
            for attempt in range(1, attempts + 1):
                try:
                    resp = await client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}]
                    )
                    text = resp.choices[0].message.content
                except Exception as error:
                    LOGGER.error(f"naming call {i} attempt {attempt} failed: {error!r}", exc_info=True)
                    errors.append(error)
                    continue
                if text:
                    path.write_text(json.dumps({"response": text, "model": model}))
                    return text
                LOGGER.error(f"naming call {i} attempt {attempt} returned an empty reply")
                errors.append(ValueError("empty reply"))
        return None

    results = await asyncio.gather(*[call(i, p) for i, p in enumerate(prompts)])
    if prompts and all(r is None for r in results):
        raise RuntimeError(f"all {len(prompts)} naming calls failed, first error: {errors[0]!r}")
    return results
