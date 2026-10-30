# source-style check (Sohan's legal_medical_restyled GPT-4.1 rewrites): legal, medical, WildChat and idmix prompts rewritten into the mix's
# style, which removes source surface cues (reddit title/body, "hi doctor", stripped apostrophes) but keeps the topic.
# Same fit (repro_two_sided idfit + Dolma) and repro_two_sided-tuned settings. Comparisons as scripts/run_rewrite_detectors.py:29-40; his
# fine bands (:27) and the coarse bands; 1,000-resample bootstrap (detectors.py:98-104).
# usage: .venv/bin/python -m scripts.wb_style

import csv
import json

import numpy as np

from scripts.wb_common import INPUTS, RUN, bootstrap_ci, load_jsonl, stratified_auroc, write_jsonl
from scripts.wb_detectors import ALL_ARMS, ONE_SIDED, compute_arms

FINE_BANDS = tuple(float(b) for b in np.unique(np.round(2 ** (np.arange(0, 41) / 4)))) + (1025.0,)
PRIMARY = [("legal_rw", "wildchat_rw"), ("medical_rw", "wildchat_rw")]
CONTEXT = [("idmix_rw", "idmix"), ("wildchat_rw", "wildchat_en_test"), ("legal_rw", "wildchat_en_test"),
           ("medical_rw", "wildchat_en_test"), ("legal_rw", "idmix_rw"), ("medical_rw", "idmix_rw"),
           ("legal", "wildchat_en_test"), ("medical", "wildchat_en_test")]


def main():
    pools = {**{p: f"legal_medical_restyled/{p}" for p in ("legal_rw", "medical_rw", "wildchat_rw", "idmix_rw")},
             **{p: f"repro_two_sided/{p}" for p in ("legal", "medical", "wildchat_en_test", "idmix")}}
    rows = {p: load_jsonl(INPUTS / f"{f}.jsonl") for p, f in pools.items()}
    texts = {p: [r["text"] for r in v] for p, v in rows.items()}
    lens = {}
    for p, f in pools.items():
        job = f.split("/")[0]
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
    arms = [a for a in ALL_ARMS if a in sc["legal_rw"]]
    write_jsonl([{"id": rows[p][i]["id"], "pool": p, "n_content_tokens": int(lens[p][i]), **{a: float(sc[p][a][i]) for a in arms}}
                 for p in pools for i in range(len(rows[p]))], RUN / "scores" / "full" / "style_legal_medical_restyled.jsonl")
    out = []
    for a in arms:
        for o, r in PRIMARY + CONTEXT:
            so, sr, lo, lr = sc[o][a], sc[r][a], lens[o], lens[r]
            row = {"arm": a, "side": "one" if a in ONE_SIDED else "two", "ood": o, "ref": r,
                   "auroc_coarse": stratified_auroc(so, sr, lo, lr)["auroc"],
                   "auroc_fine": stratified_auroc(so, sr, lo, lr, FINE_BANDS)["auroc"],
                   "fine_coverage": stratified_auroc(so, sr, lo, lr, FINE_BANDS)["coverage"]}
            if (o, r) in PRIMARY:
                row["ci_coarse"] = bootstrap_ci(lambda i, j: stratified_auroc(so[i], sr[j], lo[i], lr[j])["auroc"], len(so), len(sr), 1000, 0)
                row["ci_fine"] = bootstrap_ci(lambda i, j: stratified_auroc(so[i], sr[j], lo[i], lr[j], FINE_BANDS)["auroc"], len(so), len(sr), 1000, 0)
            out.append(row)
    keys = ["arm", "side", "ood", "ref", "auroc_coarse", "ci_coarse", "auroc_fine", "ci_fine", "fine_coverage"]
    with open(RUN / "out" / "full" / "style_legal_medical_restyled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    for a in arms:
        g = {(x["ood"], x["ref"]): x for x in out if x["arm"] == a}
        f = lambda v: "None" if v is None else f"{v:.3f}"
        print(f"{a:16s} legal_rw/wc_rw fine {f(g[('legal_rw','wildchat_rw')]['auroc_fine'])} {g[('legal_rw','wildchat_rw')]['ci_fine']} "
              f"coarse {f(g[('legal_rw','wildchat_rw')]['auroc_coarse'])} | medical_rw/wc_rw fine {f(g[('medical_rw','wildchat_rw')]['auroc_fine'])} "
              f"{g[('medical_rw','wildchat_rw')]['ci_fine']} coarse {f(g[('medical_rw','wildchat_rw')]['auroc_coarse'])} | "
              f"med_rw/wc_orig {f(g[('medical_rw','wildchat_en_test')]['auroc_coarse'])} med_orig/wc {f(g[('medical','wildchat_en_test')]['auroc_coarse'])} "
              f"idmix_rw/idmix {f(g[('idmix_rw','idmix')]['auroc_coarse'])} wc_rw/wc {f(g[('wildchat_rw','wildchat_en_test')]['auroc_coarse'])}", flush=True)


if __name__ == "__main__":
    main()
