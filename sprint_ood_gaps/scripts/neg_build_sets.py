"""Build the eval rows for designs/negatives_eval.md: negative sets (same-kind, unseen datasets), covered anchors
(Dolci rows per source, scored with themselves excluded) and positive anchors (planted legal/medical).

  --peek          print 2 seeded-random examples and the row count per dataset, write nothing (read the data first)
  --negatives N   write runs/negatives/sets/neg_<name>.jsonl with N rows each (seed 0)
  --anchors N     write runs/negatives/sets/cov_<source>.jsonl from the mix (N per Dolci SFT source and DPO prefix family)
  --planted       write runs/negatives/sets/pos_legal.jsonl, pos_medical.jsonl from his exp01 pools
"""
import argparse, json, random, re, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "external/hf/detectood_ood"
SETS = ROOT / "runs/negatives/sets"; SETS.mkdir(parents=True, exist_ok=True)
SEED = 0
MAX_CHARS = 4000  # his max_chars; embedding uses the first 1,000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def first_human(conv, keys=(("from", "human", "value"), ("role", "user", "content"))):
    for k, who, v in keys:
        for m in conv or []:
            if isinstance(m, dict) and m.get(k) == who:
                return m.get(v, "")
    return ""


def glaive_user(chat):
    m = re.search(r"USER:\s*(.*?)(?:\n\n\nASSISTANT:|ASSISTANT:|$)", chat or "", re.S)
    return m.group(1).strip() if m else ""


def logiqa(r):
    opts = r.get("options") or []
    return f"{r.get('context','')}\n{r.get('query','')}\n" + "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))


# name: (family, repo, config, split, extractor, max_rows_to_load)
NEGATIVES = {
    "magicoder":      ("code", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", lambda r: r["problem"], None),
    "codefeedback":   ("code", "m-a-p/CodeFeedback-Filtered-Instruction", None, "train", lambda r: r["query"], None),
    "codealpaca":     ("code", "sahil2801/CodeAlpaca-20k", None, "train", lambda r: (r["instruction"] + ("\n" + r["input"] if r.get("input") else "")), None),
    "metamath":       ("math", "meta-math/MetaMathQA", None, "train", lambda r: r["query"], None),
    "orca_math":      ("math", "microsoft/orca-math-word-problems-200k", None, "train", lambda r: r["question"], None),
    "mathinstruct":   ("math", "TIGER-Lab/MathInstruct", None, "train", lambda r: r["instruction"], None),
    "zebralogic":     ("reasoning", "allenai/ZebraLogicBench", "grid_mode", "test", lambda r: r["puzzle"], None),
    "knights_knaves": ("reasoning", "K-and-K/knights-and-knaves", "train", "2ppl", lambda r: r["quiz"], None),
    "glaive_fc":      ("tool_use", "glaiveai/glaive-function-calling-v2", None, "train", lambda r: (r.get("system", "") + "\n\n" + glaive_user(r["chat"])).strip(), 40000),
    "hermes_fc":      ("tool_use", "NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", lambda r: first_human(r["conversations"]), None),
    "toolace":        ("tool_use", "Team-ACE/ToolACE", None, "train", lambda r: (r.get("system", "") + "\n\n" + first_human(r["conversations"])).strip(), None),
    "ifeval_like_argilla": ("precise_if", "argilla/ifeval-like-data", "filtered", "train", lambda r: r["prompt"], None),
    "sciq":           ("science", "allenai/sciq", None, "train", lambda r: r["question"], None),
    "webinstruct":    ("science", "TIGER-Lab/WebInstructSub", None, "train", lambda r: r["question"], 40000),
    "tasksource":     ("flan_style", "tasksource/tasksource-instruct-v0", None, "train", lambda r: r["inputs"], 40000),
    "toxic_chat":     ("safety", "lmsys/toxic-chat", "toxicchat0124", "train", lambda r: r["user_input"], None),
    "beavertails":    ("safety", "PKU-Alignment/BeaverTails", None, "330k_train", lambda r: r["prompt"], 60000),
    "jbb_harmful":    ("safety", "JailbreakBench/JBB-Behaviors", "behaviors", "harmful", lambda r: r["Goal"], None),
    "harmbench":      ("safety", "walledai/HarmBench", "standard", "train", lambda r: r["prompt"], None),
    "dolly":          ("general_chat", "databricks/databricks-dolly-15k", None, "train", lambda r: (r["instruction"] + ("\n" + r["context"] if r.get("context") else "")), None),
    "no_robots":      ("general_chat", "HuggingFaceH4/no_robots", None, "train", lambda r: first_human(r["messages"]), None),
    "oasst2":         ("general_chat", "OpenAssistant/oasst2", None, "train", lambda r: r["text"] if (r.get("role") == "prompter" and r.get("parent_id") is None) else "", 60000),
    "wikitq":         ("table", "Stanford/wikitablequestions", None, "train", lambda r: r["question"], None),
    "spider":         ("table", "xlangai/spider", None, "train", lambda r: r["question"], None),
}


EXP08_SETS = {  # his exp08 pools: upstream rows of a Dolci source that are not in the mix (membership-checked by him)
    "exp08_aya_test": ("multilingual", "aya_dataset non-members (exp08)"),
    "exp08_bactrian": ("multilingual", "Bactrian-X 51 languages (exp08)"),
    "exp08_code_test": ("code", "evol-codealpaca non-members (exp08)"),
    "exp08_jailbreak_test": ("safety", "wildjailbreak non-members (exp08)"),
}


def write_exp08_sets():
    for name, (fam, desc) in EXP08_SETS.items():
        p = HF / "pools/exp08" / (name.replace("exp08_", "") + ".jsonl")
        rows = [json.loads(l) for l in open(p)]
        with open(SETS / f"neg_{name}.jsonl", "w") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"set": name, "family": fam, "kind": "negative", "repo": desc, "i": i, "text": r["text"][:MAX_CHARS], "pool_note": "his exp08 pool", "pool_unique": len(rows)}, ensure_ascii=False) + "\n")
        log(f"{name}: {len(rows)} rows from {p.name}")


def load_rows(name, n_target):
    fam, repo, cfg, split, fx, cap = NEGATIVES[name]
    from datasets import load_dataset
    if cap:
        ds = load_dataset(repo, cfg, split=split, streaming=True)
        raw = []
        for r in ds:
            raw.append(r)
            if len(raw) >= cap: break
        note = f"first {cap} streamed rows"
    else:
        ds = load_dataset(repo, cfg, split=split)
        raw = list(ds) if len(ds) <= 200000 else [ds[i] for i in range(200000)]
        note = f"all {len(ds)} rows" if len(ds) <= 200000 else "first 200000 rows"
    texts = []
    for r in raw:
        try:
            t = (fx(r) or "").strip()
        except Exception:
            t = ""
        if t:
            texts.append(t)
    # dedup on normalised text, then seeded sample
    seen, uniq = set(), []
    for t in texts:
        k = " ".join(t.lower().split())
        if k not in seen:
            seen.add(k); uniq.append(t)
    rng = random.Random(SEED)
    rng.shuffle(uniq)
    return fam, repo, uniq[:n_target], len(uniq), note


def peek():
    for name in NEGATIVES:
        try:
            fam, repo, rows, n, note = load_rows(name, 2)
            print(f"\n=== {name} [{fam}] {repo}: {n} unique prompts ({note})")
            for t in rows:
                print("   -", t[:220].replace("\n", " \\n "))
        except Exception as e:
            print(f"\n=== {name}: FAILED {type(e).__name__}: {str(e)[:200]}")


def write_negatives(n):
    for name in NEGATIVES:
        out = SETS / f"neg_{name}.jsonl"
        if out.exists():
            log("exists, skipping", out.name); continue
        try:
            fam, repo, rows, total, note = load_rows(name, n)
        except Exception as e:
            log(name, "FAILED", type(e).__name__, str(e)[:200]); continue
        with open(out, "w") as f:
            for i, t in enumerate(rows):
                f.write(json.dumps({"set": name, "family": fam, "kind": "negative", "repo": repo, "i": i, "text": t[:MAX_CHARS], "pool_note": note, "pool_unique": total}, ensure_ascii=False) + "\n")
        log(f"{name}: wrote {len(rows)} of {total} unique ({note})")


def write_anchors(n):
    meta = pd.read_parquet(ROOT / "runs/negatives/index/mix_meta.parquet")
    emb = np.load(ROOT / "runs/negatives/index/mix.npy", mmap_mode="r")
    sft = pd.read_parquet(HF / "data/processed/exp31/dolci_sft_first_user.parquet", columns=["source_dataset", "first_user"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
    dpo = pd.concat([pd.read_parquet(p, columns=["chosen", "prompt_id"]) for p in paths], ignore_index=True)
    dpo["first_user"] = dpo["chosen"].map(lambda msgs: next(m["content"] for m in msgs if m["role"] == "user"))
    def prefix(pid):
        pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)
    meta["source"] = np.where(meta["part"] == "sft", sft["source_dataset"].reindex(meta["row"]).values, "")
    is_dpo = (meta["part"] == "dpo").values
    meta.loc[is_dpo, "source"] = ["dpo:" + prefix(dpo["prompt_id"].iat[i]) for i in meta.loc[is_dpo, "row"]]
    rng = np.random.default_rng(SEED)
    for src, grp in meta.groupby("source"):
        if len(grp) < 200: continue
        pick = np.sort(rng.choice(grp.index.values, size=min(n, len(grp)), replace=False))
        tag = re.sub(r"[^A-Za-z0-9]+", "_", src).strip("_").lower()
        with open(SETS / f"cov_{tag}.jsonl", "w") as f:
            for j in pick:
                part, row = meta.at[j, "part"], int(meta.at[j, "row"])
                text = sft["first_user"].iat[row] if part == "sft" else dpo["first_user"].iat[row]
                f.write(json.dumps({"set": f"cov_{tag}", "family": src, "kind": "covered", "mix_idx": int(j), "text": text[:MAX_CHARS], "emb": np.asarray(emb[j], dtype=np.float32).round(6).tolist()}, ensure_ascii=False) + "\n")
        log(f"cov_{tag}: {len(pick)} of {len(grp)}")


def write_planted():
    for name in ("legal", "medical"):
        p = HF / f"pools/exp01/{name}.jsonl"
        rows = [json.loads(l) for l in open(p)]
        with open(SETS / f"pos_{name}.jsonl", "w") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"set": f"pos_{name}", "family": name, "kind": "planted", "i": i, "text": r["text"][:MAX_CHARS]}, ensure_ascii=False) + "\n")
        log(f"pos_{name}: {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", action="store_true"); ap.add_argument("--negatives", type=int, default=0)
    ap.add_argument("--anchors", type=int, default=0); ap.add_argument("--planted", action="store_true")
    a = ap.parse_args()
    if a.peek: peek()
    if a.negatives: write_negatives(a.negatives)
    if a.anchors: write_anchors(a.anchors)
    if a.planted: write_planted(); write_exp08_sets()
