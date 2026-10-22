# exp45: neighbour-count gap loop that skips topics an llm places in an already-found broad theme, against exp41's similarity skip.
# uv run -m scripts.run_gap_theme_loop --run_dir results/raw/261022_exp45_theme_loop_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp41_dir results/raw/261022_exp41_gap_loop_v1 --seed 0

import asyncio
import json
import logging
import os
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from openai import AsyncOpenAI

from src.data.gap_pool import load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import NAMING_PROMPT_GRAIN, centroid, name_clusters, naming_prompt, parse_naming
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

BROAD = (
    "Broad means one theme per general kind of request: for example all jailbreak or restriction-bypass attempts are one "
    "theme, all sexual or erotic content is one theme, all summary factual-consistency checks are one theme."
)
THEME_PROMPT = (
    "You keep a list of broad themes of user requests found so far. " + BROAD + "\n\nThemes so far:\n{themes}\n\n"
    "New topics:\n{topics}\n\nAssign each new topic to an existing theme if it is the same broad kind of request, "
    'otherwise give it a new broad theme name of 2 to 5 words. Reply with JSON only: {{"assign": {{"<topic id>": "<theme>"}}}}'
)
SCREEN_PROMPT = (
    "Broad themes of user requests already found:\n{themes}\n\n" + BROAD + "\n\nCandidate topics, each shown by sample "
    "user prompts:\n{cands}\n\nList the candidates whose prompts clearly belong to one of the themes above. If unsure, "
    'do not list it. Reply with JSON only: {{"skip": ["<candidate id>", ...]}}'
)
GROUP_PROMPT = (
    "Group these topics of user requests into broad themes. " + BROAD + "\n\nTopics:\n{topics}\n\n"
    'Every topic id must appear in exactly one group. Reply with JSON only: {{"groups": [["<topic id>", ...], ...]}}'
)


@dataclass
class ThemeLoopConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp41_dir: str = ""
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    rounds: int = 20
    per_round: int = 10
    screen_size: int = 40
    screen_samples: int = 5
    screen_chars: int = 200
    n_named: int = 25
    shared_needed: int = 13
    theme_cos: float = 0.70
    template_chars: int = 60
    template_share: float = 0.3
    adds_factor: float = 1.2
    # a crashed run's directory whose llm caches are reused, so a rerun replays its calls
    resume_from: str = ""
    attempts: int = 6
    nothing_factor: float = 1.05


def parse_json(text: str | None) -> dict | None:
    # per-reply boundary: the last complete json object wins, since a model may correct itself after a first answer
    decoder, found, end = json.JSONDecoder(), None, 0
    for match in re.finditer(r"\{", text or ""):
        # skip braces nested inside an object already parsed
        if match.start() < end:
            continue
        try:
            obj, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found = obj
    return found


def main():
    config = simple_parsing.parse(ThemeLoopConfig)
    assert config.run_dir and config.exp37_dir and config.exp41_dir, "run_dir, exp37_dir, exp41_dir required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    shutil.copytree(Path(config.exp41_dir) / "naming_cache", run / "naming_cache")
    if config.resume_from:
        for d in Path(config.resume_from).glob("*_cache"):
            shutil.copytree(d, run / d.name, dirs_exist_ok=True)
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    clusters = {c["id"]: c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    order = [int(c["id"][1:]) for c in sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])]
    members = {c: np.where(labels == c)[0] for c in order}
    c_ratio = {c: float(clusters[f"c{c}"]["ratio"]) for c in order}
    emb = t_np.astype(np.float32)
    cent = {c: centroid(emb[members[c]]) for c in order}
    sig: dict[int, str | None] = {}
    for c in order:
        prefix, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
        sig[c] = prefix if count >= config.template_share * len(members[c]) else None
    d41 = json.loads((Path(config.exp41_dir) / "decision.json").read_text())
    ratio_line = d41["ratio_line"]
    t41 = {int(t["id"][1:]): t for t in load_jsonl(Path(config.exp41_dir) / "reviewed_topics.jsonl")}

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    names: dict[int, dict] = {}
    calls = Counter()

    def ask(prompt: str, kind: str) -> dict | None:
        calls[kind] += 1
        reply = asyncio.run(
            name_clusters(client, config.namer_model, [prompt], str(run / f"{kind}_cache"), attempts=config.attempts)
        )[0]
        parsed = parse_json(reply)
        if parsed is None:
            calls[f"{kind}_parse_failed"] += 1
            LOGGER.error(f"{kind} reply did not parse: {reply!r}")
        return parsed

    def review(batch: list[int]) -> None:
        todo = [c for c in batch if c not in names]
        prompts = [
            naming_prompt(
                [rows[i]["text"] for i in members[c]], config.n_named, config.seed, template=NAMING_PROMPT_GRAIN
            )
            for c in todo
        ]
        replies = asyncio.run(
            name_clusters(client, config.namer_model, prompts, str(run / "naming_cache"), attempts=config.attempts)
        )
        for c, rep in zip(todo, replies, strict=True):
            names[c] = parse_naming(rep)

    def confirmed(c: int) -> bool:
        shared = names[c].get("shared")
        return shared is not None and shared >= config.shared_needed and c_ratio[c] >= ratio_line

    def theme_list(themes: dict[str, list[int]]) -> str:
        return "\n".join(f"- {t}" for t in themes) or "(none yet)"

    def screen(cands: list[int], themes: dict[str, list[int]]) -> set[int]:
        if not themes:
            return set()
        blocks = []
        for c in cands:
            sample = rng.choice(members[c], size=min(config.screen_samples, len(members[c])), replace=False)
            lines = "\n".join(f"  * {rows[i]['text'][: config.screen_chars]!r}" for i in sample)
            blocks.append(f"c{c}:\n{lines}")
        parsed = ask(SCREEN_PROMPT.format(themes=theme_list(themes), cands="\n".join(blocks)), "screen")
        listed = (parsed or {}).get("skip", [])
        return {int(x[1:]) for x in listed if isinstance(x, str) and x[1:].isdigit() and int(x[1:]) in cands}

    remaining, reviewed, conf, themes, screened_out, per_round = list(order), [], [], {}, [], []
    for rnd in range(1, config.rounds + 1):
        batch: list[int] = []
        while len(batch) < config.per_round and remaining:
            cands = [c for c in remaining if c not in batch][: config.screen_size]
            drop = screen(cands, themes)
            screened_out += sorted(drop)
            remaining = [c for c in remaining if c not in drop]
            batch += [c for c in cands if c not in drop][: config.per_round - len(batch)]
        review(batch)
        remaining = [c for c in remaining if c not in batch]
        new_conf = [c for c in batch if confirmed(c)]
        reviewed += batch
        conf += new_conf
        if new_conf:
            topics = "\n".join(f"c{c}: {names[c].get('name')} - {names[c].get('description')}" for c in new_conf)
            parsed = ask(THEME_PROMPT.format(themes=theme_list(themes), topics=topics), "theme")
            assign = (parsed or {}).get("assign", {})
            for c in new_conf:
                themes.setdefault(str(assign.get(f"c{c}") or names[c].get("name")), []).append(c)
        per_round.append({"id": f"round_{rnd}", "reviewed": batch, "confirmed": new_conf, "themes_so_far": len(themes)})
        LOGGER.info(f"round {rnd}: {len(new_conf)} confirmed, {len(themes)} themes, {len(screened_out)} screened out")

    conf41 = [c for r in d41["loop_rounds"] for c in r["reviewed"] if t41[c]["confirmed"]]
    info = {
        **{c: (t41[c]["name"], t41[c]["description"]) for c in conf41},
        **{c: (names[c].get("name"), names[c].get("description")) for c in conf},
    }
    pooled = sorted(set(conf) | set(conf41))
    perm = rng.permutation(len(pooled))
    alias = {f"t{j}": pooled[i] for j, i in enumerate(perm)}
    listing = "\n".join(f"{a}: {info[c][0]} - {info[c][1]}" for a, c in alias.items())
    parsed = ask(GROUP_PROMPT.format(topics=listing), "group")
    groups = [[alias[a] for a in g if a in alias] for g in (parsed or {}).get("groups", [])]
    grouped = {c for g in groups for c in g}
    # a topic the judge left out counts as its own theme so no arm loses it
    groups += [[c] for c in pooled if c not in grouped]
    arms = {"theme_skip": conf, "similarity_skip_exp41": conf41}
    n = {a: sum(any(c in set(cf) for c in g) for g in groups) for a, cf in arms.items()}

    def merge(cf: list[int]) -> int:
        heads: list[int] = []
        for c in cf:
            if not any(
                float(cent[c] @ cent[h]) >= config.theme_cos or (sig[c] is not None and sig[c] == sig[h]) for h in heads
            ):
                heads.append(c)
        return len(heads)

    if n["theme_skip"] >= config.adds_factor * n["similarity_skip_exp41"]:
        verdict = "theme_skip_adds"
    elif n["theme_skip"] <= config.nothing_factor * n["similarity_skip_exp41"]:
        verdict = "theme_skip_adds_nothing"
    else:
        verdict = "ambiguous"

    def count_names(cf: list[int], words: tuple[str, ...]) -> int:
        return sum(any(w in (info[c][0] or "").lower() for w in words) for c in cf)

    decision = {
        "verdict": verdict,
        "themes_llm_pooled": n,
        "themes_0.70": {a: merge(cf) for a, cf in arms.items()},
        "confirmed_topics": {a: len(cf) for a, cf in arms.items()},
        "jailbreak_names": {a: count_names(cf, ("jailbreak",)) for a, cf in arms.items()},
        "sexual_names": {a: count_names(cf, ("sexual", "erotic")) for a, cf in arms.items()},
        "screened_out": len(screened_out),
        "calls": dict(calls),
        "loop_themes": {t: [info[c][0] for c in cs] for t, cs in themes.items()},
        "pooled_groups": len(groups),
        "ungrouped_by_judge": len(pooled) - len(grouped),
        "null_names": sum(names[c].get("shared") is None for c in reviewed),
        "rounds": per_round,
        "groups": [
            [{"cluster": c, "name": info[c][0], "arms": [a for a, cf in arms.items() if c in cf]} for c in g]
            for g in groups
        ],
    }
    write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "cluster": c,
                "rank37": clusters[f"c{c}"]["rank"],
                "size": len(members[c]),
                **{k: names[c].get(k) for k in ("name", "description", "shared")},
                "confirmed": c in conf,
                "examples": [
                    rows[i]["text"][:400] for i in rng.choice(members[c], size=min(10, len(members[c])), replace=False)
                ],
            }
            for c in reviewed
        ],
        run / "reviewed_topics.jsonl",
    )
    print(f"{n=} themes_0.70={decision['themes_0.70']} confirmed={decision['confirmed_topics']}")
    print(
        f"{decision['jailbreak_names']=} {decision['sexual_names']=} screened_out={len(screened_out)} calls={dict(calls)}"
    )
    print(f"verdict {verdict}; decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
