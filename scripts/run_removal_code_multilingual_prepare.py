# removal_code_multilingual/removal_math_science_if step 1: marks the dolci mix rows that come only from each removed kind's sources, and draws the grader gate set.
# uv run -m scripts.run_removal_code_multilingual_prepare --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --safety_removal_cache_dir cache/safety_removal --cache_dir cache/removal_code_multilingual --seed 0

import json
import logging
import resource
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing

from src.data.mix_sources import KIND_SOURCES, only_in_sources_mask
from src.data.pools import load_dolci_dpo
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PrepareConfig:
    pipeline_million_dir: str = ""
    safety_removal_cache_dir: str = ""
    cache_dir: str = ""
    seed: int = 0
    n_gate_positive: int = 150
    n_gate_negative: int = 450
    # removal_math_science_if: math, science, precise_if with negatives from coding, multilingual, safety, tool use and other
    kinds: tuple[str, ...] = ("coding", "multilingual", "tool_use")
    negative_domains: tuple[str, ...] = ("Math", "Reasoning", "Science", "Precise IF", "Other", "Safety")


def main():
    config = simple_parsing.parse(PrepareConfig)
    assert config.pipeline_million_dir and config.safety_removal_cache_dir and config.cache_dir, "paths required"
    out = Path(config.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "masks.npz").exists(), f"{out} already prepared"
    setup_logging(out)
    set_seed(config.seed)
    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())

    m_text = pd.read_parquet(Path(config.safety_removal_cache_dir) / "mix_text.parquet")["text"].tolist()
    safety_removal_safety = np.load(Path(config.safety_removal_cache_dir) / "arrays.npz")["safety"]
    sft = pd.read_parquet(c37["sft_path"], columns=["first_user", "source_dataset", "domain"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    dpo = load_dolci_dpo()
    sft_text, dpo_text = sft["first_user"].tolist(), dpo["first_user"].tolist()

    masks, manifest = {}, {"git_hash": get_git_hash(), **asdict(config), "n_mix": len(m_text), "kinds": {}}
    for kind in ("safety", *config.kinds):
        domain, sources, families = KIND_SOURCES[kind]
        sft_in = sft["source_dataset"].isin(sources).to_numpy()
        assert (sft_in == (sft["domain"] == domain).to_numpy()).all(), f"{kind}: sources and domain {domain} disagree"
        assert set(families) <= set(dpo["family"]), f"{kind}: a named dpo family is missing"
        dpo_in = dpo["family"].isin(families).to_numpy()
        masks[kind] = only_in_sources_mask(m_text, sft_text, sft_in, dpo_text, dpo_in)
        manifest["kinds"][kind] = {
            "sft_rows": {s: int((sft["source_dataset"] == s).sum()) for s in sources},
            "dpo_rows": {f: int((dpo["family"] == f).sum()) for f in families},
            "mix_rows_removed": int(masks[kind].sum()),
            "share_of_mix": round(float(masks[kind].mean()), 4),
        }
        LOGGER.info(f"{kind}: {manifest['kinds'][kind]}")
    # the tightened rule must give safety_removal's safety mask exactly
    manifest["safety_equals_safety_removal"] = bool((masks["safety"] == safety_removal_safety).all())
    assert manifest["safety_equals_safety_removal"], "safety mask differs from safety_removal's"
    overlap = {
        f"{a}&{b}": int((masks[a] & masks[b]).sum()) for i, a in enumerate(config.kinds) for b in config.kinds[i + 1 :]
    }
    manifest["overlap_between_kinds"] = overlap
    assert not any(overlap.values()), f"a mix row is removed by two kinds: {overlap}"
    np.savez(out / "masks.npz", **{k: masks[k] for k in config.kinds})

    # gate: each kind's removed rows are its positives; rows of six other domains are negatives for every kind
    rng = np.random.default_rng(config.seed)
    gate = []
    for kind in config.kinds:
        idx = rng.choice(np.where(masks[kind])[0], size=config.n_gate_positive, replace=False)
        gate += [{"id": f"g_{kind}_{i}", "truth": kind, "text": m_text[i]} for i in idx]
    neg_rows = np.where(sft["domain"].isin(config.negative_domains).to_numpy())[0]
    idx = rng.choice(neg_rows, size=config.n_gate_negative, replace=False)
    gate += [{"id": f"g_none_{i}", "truth": "none", "domain": sft["domain"][i], "text": sft_text[i]} for i in idx]
    write_jsonl(gate, out / "gate_prompts.jsonl")
    manifest["gate_rows"] = len(gate)
    manifest["examples"] = {
        k: [m_text[i][:200] for i in np.where(masks[k])[0][:: int(masks[k].sum()) // 5][:5]] for k in config.kinds
    }
    manifest["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    write_json(manifest, out / "manifest.json")
    print(json.dumps({k: v for k, v in manifest.items() if k != "examples"}, indent=1))
    print(f"masks saved to {out / 'masks.npz'}")


if __name__ == "__main__":
    main()
