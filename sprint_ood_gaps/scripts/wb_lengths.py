# content-token lengths for every chat pool, with the tokenizer and chat template of the model Sohan used for that
# experiment (exp01 / exp11: Olmo-3-7B-Instruct-DPO; exp12: OLMo-2-0425-1B-SFT). Recipe: src/interp/features.py:16-31.
# usage: .venv/bin/python -m scripts.wb_lengths

import time

from transformers import AutoTokenizer

from scripts.wb_common import INPUTS, MODELS, RUN, load_jsonl, n_content_tokens, write_json

JOBS = {
    "exp01": ("exp01", ["idfit", "idmix", "legal", "medical", "wildchat_en_test"]),
    "exp12": ("exp12", ["idfit", "idmix", "legal", "medical", "wildchat_en_test"]),
    "exp11": (
        "exp01",
        ["exp06/deploy", "exp09/lmsys", "exp06/needles_legal", "exp06/needles_medical", "exp06/needles_aya", "exp06/needles_magicoder"],
    ),
    "exp03": ("exp01", [f"exp03/{p}" for p in ("legal_rtb", "reddit_askscience", "reddit_cooking", "reddit_fitness",
                                                 "reddit_learnprogramming", "reddit_relationship_advice",
                                                 "reddit_relationships", "reddit_writing")]),
    "exp05": ("exp01", [f"exp05/{p}" for p in ("legal_rw", "medical_rw", "wildchat_rw", "idmix_rw")]),
    "val_exp01": ("exp01", ["val/val_legal", "val/val_medical", "val/val_wildchat_dolci"]),
    "val_exp12": ("exp12", ["val/val_legal", "val/val_medical", "val/val_wildchat_tulu"]),
}


def main():
    import sys

    only = sys.argv[1:]
    for job, (model_key, pools) in JOBS.items():
        if only and job not in only:
            continue
        model_id, rev = MODELS[model_key]
        tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
        for pool in pools:
            t0 = time.time()
            path = INPUTS / ("exp11" if job == "exp11" else "" if job.startswith("val") or job in ("exp03", "exp05") else job) / f"{pool}.jsonl"
            rows = load_jsonl(path)
            lens = {r["id"]: n_content_tokens(tok, r["text"]) for r in rows}
            out = RUN / "cache" / "lengths" / job / f"{pool.split('/')[-1]}.json"
            write_json({"model_id": model_id, "revision": rev, "source": str(path.relative_to(RUN)), "lengths": lens}, out)
            v = sorted(lens.values())
            print(f"{job} {pool}: {len(v)} rows, median {v[len(v)//2]}, max {v[-1]}, {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
