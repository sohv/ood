# content-token lengths for every chat pool, with the tokenizer and chat template of the model Sohan used for that
# experiment (repro_two_sided / ranking_variants: Olmo-3-7B-Instruct-DPO; repro_olmo2_1b: OLMo-2-0425-1B-SFT). Recipe: src/interp/features.py:16-31.
# usage: .venv/bin/python -m scripts.wb_lengths

import time

from transformers import AutoTokenizer

from scripts.wb_common import INPUTS, MODELS, RUN, load_jsonl, n_content_tokens, write_json

JOBS = {
    "repro_two_sided": ("repro_two_sided", ["idfit", "idmix", "legal", "medical", "wildchat_en_test"]),
    "repro_olmo2_1b": ("repro_olmo2_1b", ["idfit", "idmix", "legal", "medical", "wildchat_en_test"]),
    "ranking_variants": (
        "repro_two_sided",
        ["density_ratio_needles/deploy", "iterative_discovery/lmsys", "density_ratio_needles/needles_legal", "density_ratio_needles/needles_medical", "density_ratio_needles/needles_aya", "density_ratio_needles/needles_magicoder"],
    ),
    "register_vs_coverage": ("repro_two_sided", [f"register_vs_coverage/{p}" for p in ("legal_rtb", "reddit_askscience", "reddit_cooking", "reddit_fitness",
                                                 "reddit_learnprogramming", "reddit_relationship_advice",
                                                 "reddit_relationships", "reddit_writing")]),
    "legal_medical_restyled": ("repro_two_sided", [f"legal_medical_restyled/{p}" for p in ("legal_rw", "medical_rw", "wildchat_rw", "idmix_rw")]),
    "val_repro_two_sided": ("repro_two_sided", ["val/val_legal", "val/val_medical", "val/val_wildchat_dolci"]),
    "val_repro_olmo2_1b": ("repro_olmo2_1b", ["val/val_legal", "val/val_medical", "val/val_wildchat_tulu"]),
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
            path = INPUTS / ("ranking_variants" if job == "ranking_variants" else "" if job.startswith("val") or job in ("register_vs_coverage", "legal_medical_restyled") else job) / f"{pool}.jsonl"
            rows = load_jsonl(path)
            lens = {r["id"]: n_content_tokens(tok, r["text"]) for r in rows}
            out = RUN / "cache" / "lengths" / job / f"{pool.split('/')[-1]}.json"
            write_json({"model_id": model_id, "revision": rev, "source": str(path.relative_to(RUN)), "lengths": lens}, out)
            v = sorted(lens.values())
            print(f"{job} {pool}: {len(v)} rows, median {v[len(v)//2]}, max {v[-1]}, {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
