# gap-discovery helpers for exp09: clustering, blind cluster naming via openrouter, coverage and gap matching.

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

LOGGER = logging.getLogger(__name__)

NAMING_PROMPT = """Below are {n} messages that users sent to an AI chatbot, separated by lines of ---.

{messages}

Reply in exactly this format and nothing else:
NAME: <a short name, at most 6 words, for what these messages have in common>
DESCRIPTION: <one sentence>
COHERENT: <YES if most of the messages clearly share a topic, task, format or language, otherwise NO>"""


def cluster(embeddings: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embeddings)


def centroid(embeddings: np.ndarray) -> np.ndarray:
    c = embeddings.mean(0)
    return c / (np.linalg.norm(c) + 1e-12)


def naming_prompt(texts: list[str], n: int, seed: int, max_chars: int = 300) -> str:
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    shown = [" ".join(texts[i].split())[:max_chars] for i in pick]
    return NAMING_PROMPT.format(n=len(shown), messages="\n---\n".join(shown))


def parse_naming(reply: str | None) -> dict:
    if reply is None:
        return {"name": None, "description": None, "coherent": None}
    get = lambda key: (m.group(1).strip() if (m := re.search(rf"^{key}:\s*(.+)$", reply, re.M)) else None)
    coherent = get("COHERENT")
    return {"name": get("NAME"), "description": get("DESCRIPTION"), "coherent": None if coherent is None else coherent.upper().startswith("YES")}


def match_gap(c: np.ndarray, known: list[np.ndarray], threshold: float) -> int | None:
    sims = [float(c @ k) for k in known]
    return int(np.argmax(sims)) if sims and max(sims) >= threshold else None


async def name_clusters(client, model: str, prompts: list[str], cache_dir: str, max_concurrent: int = 10) -> list[str | None]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max_concurrent)
    errors: list[Exception] = []

    async def call(i: int, prompt: str) -> str | None:
        path = Path(cache_dir) / f"{hashlib.md5(f'{model}{prompt}'.encode()).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]
        async with sem:
            try:
                resp = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
                text = resp.choices[0].message.content
            except Exception as error:
                LOGGER.error(f"naming call {i} failed: {error!r}", exc_info=True)
                errors.append(error)
                return None
        path.write_text(json.dumps({"response": text, "model": model}))
        return text

    results = await asyncio.gather(*[call(i, p) for i, p in enumerate(prompts)])
    if prompts and len(errors) == len(prompts):
        raise RuntimeError(f"all {len(prompts)} naming calls failed, first error: {errors[0]!r}")
    return results
