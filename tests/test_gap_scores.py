# unit tests for alt_gap_scores's dsir gap score.
# uv run -m pytest tests/test_gap_scores.py -v

import numpy as np

from src.metrics.gap_scores import dsir_scores


def test_dsir_ranks_traffic_only_words_highest():
    traffic = ["fix my docker compose file"] * 20 + ["write a poem about rain"] * 20 + ["zorblax quantum flux"] * 5
    mix = ["write a poem about rain"] * 40 + ["fix my docker compose file"] * 40
    s = dsir_scores(traffic, mix, n_features=1024)
    assert s.shape == (len(traffic),)
    # the phrase absent from the mix scores above the phrases the mix covers
    assert s[-1] > s[0] and s[-1] > s[20]
    assert np.isfinite(s).all()
