import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.utils.git import get_git_hash


@dataclass
class Config:
    model_id: str = "claude-sonnet-4-6"
    dataset_path: str = ""  # always required, no default path
    output_dir: str = "results"
    num_tasks: int | None = None  # None = use all
    n_repeats: int = 1
    seed: int = 42


def write_config_json(config: Config, output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    path.write_text(json.dumps({"git_hash": get_git_hash(), **asdict(config)}, indent=2))
    return path
