import json
import random

import numpy as np

from src.utils.config import Config, write_config_json
from src.utils.git import get_git_hash
from src.utils.seed import set_seed


def test_set_seed_makes_a_run_repeatable():
    set_seed(42)
    first = (random.random(), float(np.random.rand()))
    set_seed(42)
    assert (random.random(), float(np.random.rand())) == first


def test_get_git_hash_is_short_hex():
    git_hash = get_git_hash()
    assert len(git_hash) == 8
    assert all(c in "0123456789abcdef" for c in git_hash)


def test_config_has_no_default_dataset_path():
    assert Config().dataset_path == ""


def test_write_config_json_records_provenance(tmp_path):
    config = Config(model_id="claude-haiku-4-5-20251001", dataset_path="data/processed/x.jsonl", seed=7)
    path = write_config_json(config, tmp_path / "run")
    written = json.loads(path.read_text())
    assert written["git_hash"] == get_git_hash()
    assert written["model_id"] == "claude-haiku-4-5-20251001"
    assert written["seed"] == 7
