# exp80 step 1: loads exp37's traffic pool and dolci mix once, marks safety-only mix rows, and saves arrays and texts to cache for the later steps.
# uv run -m scripts.run_exp80_prepare --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --cache_dir cache/exp80 --seed 0

import json
import logging
import resource
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.mix_sources import SAFETY_DPO_FAMILIES, SAFETY_SFT_SOURCES, only_in_sources_mask
from src.data.pools import load_dolci_dpo
from src.metrics.gap_pipeline import text_hash
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PrepareConfig:
    exp37_dir: str = ""
    cache_dir: str = ""
    seed: int = 0
    # optional cross-check against a peer's saved hashes, which reproduced exp37's ratios
    slim_path: str = ""


def main():
    config = simple_parsing.parse(PrepareConfig)
    assert config.exp37_dir and config.cache_dir, "exp37_dir and cache_dir required"
    out = Path(config.cache_dir)
    assert not (out / "arrays.npz").exists(), f"{out} already prepared"
    setup_logging(out)
    set_seed(config.seed)
    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())

    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"], device="cpu"
    )
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    assert len(labels) == len(rows) == 992_933, f"{len(labels)=} {len(rows)=}"
    pd.DataFrame({"text": [r["text"] for r in rows]}).to_parquet(out / "traffic_text.parquet")
    del rows

    m_np, m_hash, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    assert len(m_np) == 1_710_578, f"{len(m_np)=}"
    sft = pd.read_parquet(c37["sft_path"], columns=["first_user", "source_dataset", "domain"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    sft_safety = sft["source_dataset"].isin(SAFETY_SFT_SOURCES).to_numpy()
    assert (sft_safety == (sft["domain"] == "Safety").to_numpy()).all(), "safety sources and safety domain disagree"
    dpo = load_dolci_dpo()
    dpo_safety = dpo["family"].isin(SAFETY_DPO_FAMILIES).to_numpy()
    assert set(SAFETY_DPO_FAMILIES) <= set(dpo["family"]), "a named dpo safety family is missing"
    safety = only_in_sources_mask(
        m_text, sft["first_user"].tolist(), sft_safety, dpo["first_user"].tolist(), dpo_safety
    )
    # how many distinct safety texts stay because a non-safety source also has them
    safety_norms = {" ".join(t.lower().split()) for t in sft["first_user"][sft_safety]} | {
        " ".join(t.lower().split()) for t in dpo["first_user"][dpo_safety]
    }
    manifest = {
        "git_hash": get_git_hash(),
        **asdict(config),
        "gap_pool_uncommitted_diff": subprocess.run(
            ["git", "diff", "--stat", "src/data/gap_pool.py"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "n_traffic": len(t_np),
        "n_mix": len(m_np),
        "n_safety_only": int(safety.sum()),
        "n_distinct_safety_texts": len(safety_norms),
        "n_safety_texts_kept_by_other_sources": len(safety_norms) - int(safety.sum()),
        "sft_safety_rows": {s: int((sft["source_dataset"] == s).sum()) for s in SAFETY_SFT_SOURCES},
        "dpo_safety_rows": {f: int((dpo["family"] == f).sum()) for f in SAFETY_DPO_FAMILIES},
        "safety_only_examples": [m_text[i][:300] for i in np.where(safety)[0][:: max(1, int(safety.sum()) // 5)][:5]],
    }
    del sft, dpo
    if config.slim_path:
        slim = np.load(config.slim_path)
        manifest["slim_t_hash_equal"] = bool((slim["t_hash"] == t_hash).all())
        manifest["slim_m_hash_equal"] = bool((slim["m_hash"] == m_hash).all())
        assert manifest["slim_t_hash_equal"] and manifest["slim_m_hash_equal"], "row order differs from the slim arrays"
    pd.DataFrame({"text": m_text, "safety_only": safety}).to_parquet(out / "mix_text.parquet")
    del m_text
    np.savez(out / "arrays.npz", t_emb=t_np, t_hash=t_hash, m_emb=m_np, m_hash=m_hash, safety=safety, labels=labels)
    manifest["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    write_json(manifest, out / "manifest.json")
    print(json.dumps({k: v for k, v in manifest.items() if k != "safety_only_examples"}, indent=1))
    print(f"arrays saved to {out / 'arrays.npz'}")


if __name__ == "__main__":
    main()
