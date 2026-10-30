# register check (Sohan's register_vs_coverage pools): do the word baselines rank r/legaladvice above Reddit posts on topics the mix
# covers, when both come from the same Reddit source? Same fit (repro_two_sided idfit, Dolma) and repro_two_sided-tuned settings.
# Statistic: length-stratified AUROC (detectors.py:67-95), legal_rtb vs each subreddit, and each pool vs English WildChat.
# usage: .venv/bin/python -m scripts.wb_register

import csv
import json

import numpy as np

from scripts.wb_common import INPUTS, RUN, load_jsonl, stratified_auroc, write_jsonl
from scripts.wb_detectors import ALL_ARMS, compute_arms

SUBS = ("reddit_askscience", "reddit_cooking", "reddit_fitness", "reddit_learnprogramming",
        "reddit_relationship_advice", "reddit_relationships", "reddit_writing")


def main():
    pools = {"legal_rtb": "register_vs_coverage/legal_rtb", **{s: f"register_vs_coverage/{s}" for s in SUBS},
             "wildchat_en_test": "repro_two_sided/wildchat_en_test", "legal": "repro_two_sided/legal"}
    rows = {p: load_jsonl(INPUTS / f"{f}.jsonl") for p, f in pools.items()}
    texts = {p: [r["text"] for r in v] for p, v in rows.items()}
    lens = {}
    for p, f in pools.items():
        job = "register_vs_coverage" if f.startswith("register_vs_coverage") else "repro_two_sided"
        L = json.loads((RUN / "cache" / "lengths" / job / f"{f.split('/')[-1]}.json").read_text())["lengths"]
        lens[p] = np.array([L[r["id"]] for r in rows[p]])
    mix = [r["text"] for r in load_jsonl(INPUTS / "repro_two_sided/idfit.jsonl")]
    ref = [" ".join(r["text"].split()[:400]) for r in load_jsonl(INPUTS / "repro_two_sided/dolma.jsonl")]
    emb = {}
    for e in ("minilm", "mminilm"):
        g = lambda f: np.load(RUN / "cache" / "emb_full" / e / f"{f.replace('/', '__')}.npz")["emb"]
        emb[e] = {"mix": g("repro_two_sided/idfit"), "ref": g("repro_two_sided/dolma"), **{p: g(f) for p, f in pools.items()}}
    params = json.loads((RUN / "out" / "full" / "tuned_params_repro_two_sided.json").read_text())["params"]
    sc = compute_arms(params, mix, ref, texts, emb, lens)
    arms = [a for a in ALL_ARMS if a in sc["legal"]]
    write_jsonl([{"id": rows[p][i]["id"], "pool": p, "n_content_tokens": int(lens[p][i]), **{a: float(sc[p][a][i]) for a in arms}}
                 for p in pools for i in range(len(rows[p]))], RUN / "scores" / "full" / "register_register_vs_coverage.jsonl")
    out = []
    for a in arms:
        row = {"arm": a}
        for s in SUBS:
            row[f"legal_rtb_vs_{s}"] = stratified_auroc(sc["legal_rtb"][a], sc[s][a], lens["legal_rtb"], lens[s])["auroc"]
        for p in ("legal", "legal_rtb", *SUBS):
            row[f"{p}_vs_wildchat"] = stratified_auroc(sc[p][a], sc["wildchat_en_test"][a], lens[p], lens["wildchat_en_test"])["auroc"]
        row["median_legal_rtb_vs_subs"] = float(np.median([row[f"legal_rtb_vs_{s}"] for s in SUBS]))
        out.append(row)
    with open(RUN / "out" / "full" / "register_register_vs_coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in sorted(out, key=lambda r: -r["median_legal_rtb_vs_subs"]):
        print(f"{r['arm']:16s} median legal_rtb vs subs {r['median_legal_rtb_vs_subs']:.3f}  "
              + " ".join(f"{s.replace('reddit_', '')[:8]} {r[f'legal_rtb_vs_{s}']:.2f}" for s in SUBS)
              + f" | vs WC: legal {r['legal_vs_wildchat']:.2f} " + " ".join(f"{s.replace('reddit_', '')[:6]} {r[f'{s}_vs_wildchat']:.2f}" for s in SUBS))


if __name__ == "__main__":
    main()
