# safety_removal/removal_code_multilingual step 3: fp32 k=50 neighbour count, with a fair tie split, of every traffic prompt against the full mix, the mix minus one kind's sources, and three matched random removals.
# uv run -m scripts.run_safety_removal_counts --run_dir results/raw/261027_safety_removal_v1 --cache_dir cache/safety_removal --seed 0

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch

from src.data.mix_sources import random_removal_keep
from src.metrics.fair_count import density_ratios_fair, removed_neighbour_share
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class CountsConfig:
    run_dir: str = ""
    cache_dir: str = ""
    seed: int = 0
    k: int = 50
    extra: int = 70
    chunk: int = 512
    control_seeds: tuple[int, ...] = (0, 1, 2)
    # smoke runs use cpu so the shared gpu stays free for vllm
    device: str = "cuda"
    # smoke runs: score only the first n traffic prompts
    count_limit: int | None = None
    # removal_code_multilingual: the kind removed, its mask read from masks_path; safety_removal's safety mask lives in arrays.npz
    removal: str = "safety"
    masks_path: str = ""
    # removal_code_multilingual reads the full arm from safety_removal instead of recomputing the same pass
    include_full: bool = True
    # hard cap on this process's gpu memory so a shared gpu's other jobs still fit; 0 means no cap
    gpu_mem_gb: float = 0.0


def main():
    config = simple_parsing.parse(CountsConfig)
    assert config.run_dir and config.cache_dir, "run_dir and cache_dir required"
    run = Path(config.run_dir)
    assert not (run / "counts.npz").exists(), f"{run} already has counts; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    # fp32 throughout: k_sweep_behaviour traced pipeline_million's ceiling errors on copy-heavy topics to the fp16 pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if config.gpu_mem_gb:
        torch.cuda.set_per_process_memory_fraction(
            config.gpu_mem_gb * 1e9 / torch.cuda.get_device_properties(0).total_memory
        )
    run_config = {
        "git_hash": get_git_hash(),
        **asdict(config),
        "gpu": torch.cuda.get_device_name() if config.device == "cuda" else None,
        "dtype": "float32",
    }
    write_json(run_config, run / "counts_config.json")
    # a run directory whose other steps write no config.json still gets one
    if not (run / "config.json").exists():
        write_json(run_config, run / "config.json")

    arrays = np.load(Path(config.cache_dir) / "arrays.npz")
    t_np, t_hash = arrays["t_emb"].astype(np.float32), arrays["t_hash"]
    m_np, m_hash = arrays["m_emb"].astype(np.float32), arrays["m_hash"]
    removed = np.load(config.masks_path)[config.removal] if config.masks_path else arrays["safety"]
    assert len(removed) == len(m_np) and removed.dtype == bool, f"{removed.shape=} {removed.dtype=}"
    keeps = {"full": np.ones(len(m_np), dtype=bool)} if config.include_full else {}
    keeps |= {f"minus_{config.removal}": ~removed}
    keeps |= {f"random_{s}": random_removal_keep(removed, s) for s in config.control_seeds}
    LOGGER.info(f"traffic {t_np.shape}, mix {m_np.shape}, {config.removal}-only {int(removed.sum())}")
    for name, keep in keeps.items():
        LOGGER.info(f"{name}: {int(keep.sum())} mix rows kept")

    q_rows = np.arange(len(t_np) if config.count_limit is None else config.count_limit)
    t_gpu, t_hash_gpu = torch.from_numpy(t_np).to(config.device), torch.from_numpy(t_hash).to(config.device)
    out, seconds = {}, {}
    for name, keep in keeps.items():
        t0 = time.time()
        m_gpu = torch.from_numpy(m_np[keep]).to(config.device)
        res = density_ratios_fair(
            t_np[q_rows],
            t_hash[q_rows],
            q_rows,
            t_gpu,
            t_hash_gpu,
            m_gpu,
            torch.from_numpy(m_hash[keep]).to(config.device),
            k=config.k,
            extra=config.extra,
            chunk=config.chunk,
        )
        out |= {f"{key}_{name}": v for key, v in res.items()}
        del m_gpu
        if config.device == "cuda":
            torch.cuda.empty_cache()
        seconds[name] = round(time.time() - t0, 1)
        LOGGER.info(
            f"{name} scored in {seconds[name]}s; median fair ratio {np.median(out[f'ratio_fair_{name}']):.4f}; "
            f"tie-decided {int(out[f'tie_decided_{name}'].sum())}, saturated {int(out[f'saturated_{name}'].sum())}"
        )
    del t_gpu
    m_gpu = torch.from_numpy(m_np).to(config.device)
    out[f"{config.removal}_share"] = removed_neighbour_share(
        t_np[q_rows],
        t_hash[q_rows],
        m_gpu,
        torch.from_numpy(m_hash).to(config.device),
        torch.from_numpy(removed).to(config.device),
        config.k,
        config.extra,
        config.chunk,
    )
    n_nan = {k: int(np.isnan(v).sum()) for k, v in out.items() if v.dtype != bool}
    np.savez(run / "counts.npz", q_rows=q_rows, **out)
    write_json(
        {
            "n_queries": len(q_rows),
            "mix_rows_kept": {n: int(k.sum()) for n, k in keeps.items()},
            "seconds": seconds,
            "nan_counts": n_nan,
            "median_ratio_fair": {n: float(np.median(out[f"ratio_fair_{n}"])) for n in keeps},
            "tie_decided": {n: int(out[f"tie_decided_{n}"].sum()) for n in keeps},
            "saturated": {n: int(out[f"saturated_{n}"].sum()) for n in keeps},
            f"mean_{config.removal}_share": float(out[f"{config.removal}_share"].mean()),
        },
        run / "counts_summary.json",
    )
    print(f"{n_nan=}")
    print(f"counts saved to {run / 'counts.npz'}")


if __name__ == "__main__":
    main()
