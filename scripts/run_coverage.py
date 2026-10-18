# scores every pool's rows by similarity to their nearest dolci-instruct-dpo prompt (minilm and tf-idf).
# uv run -m scripts.run_coverage --pool_dirs data/processed/exp01 data/processed/exp03 data/processed/exp04_seed1 data/processed/exp04_seed2 --output_dir results/raw/261018_exp04_coverage_v1 --seed 0

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

from src.data.io import load_jsonl
from src.data.pools import load_dolci_dpo
from src.metrics.coverage import nearest_cosine_embedding, nearest_cosine_tfidf
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

SKIP = {"idfit", "dolma"}


@dataclass
class CoverageConfig:
    pool_dirs: list[str] = field(default_factory=list)
    output_dir: str = ""
    seed: int = 0
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    config = simple_parsing.parse(CoverageConfig)
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    model = SentenceTransformer(config.model_id, device="cuda")
    write_json({"git_hash": get_git_hash(), **asdict(config), "model_revision": HfApi().model_info(config.model_id).sha}, out / "config.json")
    dolci = load_dolci_dpo()["first_user"].drop_duplicates().tolist()
    LOGGER.info(f"{len(dolci)} unique dolci prompts")
    reference = model.encode(dolci, batch_size=512, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    pools = {}
    for d in config.pool_dirs:
        for path in sorted(Path(d).glob("*.jsonl")):
            if path.stem not in SKIP:
                pools[f"{Path(d).name}/{path.stem}"] = load_jsonl(path)
    tfidf = nearest_cosine_tfidf(dolci, {k: [r["text"] for r in v] for k, v in pools.items()})
    rows, summary = [], {}
    for key, recs in pools.items():
        emb = nearest_cosine_embedding(model, reference, [r["text"] for r in recs])
        rows += [{"id": r["id"], "source": key, "cov_minilm": float(e), "cov_tfidf": float(t)} for r, e, t in zip(recs, emb, tfidf[key])]
        summary[key] = {"n": len(recs), "cov_minilm_median": float(np.median(emb)), "cov_tfidf_median": float(np.median(tfidf[key]))}
        LOGGER.info(f"{key}: {summary[key]}")
    write_jsonl(rows, out / "coverage.jsonl")
    write_json(summary, out / "coverage_summary.json")
    print(f"coverage written to {out / 'coverage.jsonl'}")


if __name__ == "__main__":
    main()
