# pure-logic and file-output tests for the plot helpers; no fonts or display needed.
# uv run -m pytest tests/test_plotting.py -v
import csv
import inspect
import math

import matplotlib.pyplot as plt
import pytest

from src.utils import plotting
from src.utils.plotting import BLUE, finish, finish_dicts, header, panel_title, point_ci, style, wilson


def test_wilson_matches_known_values():
    p, lo, hi = wilson(50, 100)
    assert p == 0.5
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)


def test_wilson_stays_inside_unit_interval_at_the_edges():
    # the bounds land within float noise of 0 and 1; point_ci clamps that noise before errorbar sees it
    _, lo, _ = wilson(0, 10)
    _, _, hi = wilson(10, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) or lo > 0.0
    assert hi == pytest.approx(1.0, abs=1e-12) or hi < 1.0


def test_wilson_empty_is_nan_triple():
    assert all(math.isnan(v) for v in wilson(0, 0))


def test_style_falls_back_without_inter(monkeypatch, tmp_path):
    monkeypatch.setattr(plotting, "FONTS", tmp_path)
    style()
    assert plt.rcParams["font.family"] in (["Inter"], ["DejaVu Sans"])
    assert plt.rcParams["figure.facecolor"] == plotting.SURFACE


def test_point_ci_accepts_p_equal_one():
    # the -0.0 upper error at p == 1 must be clamped, not passed to errorbar
    fig, ax = plt.subplots()
    point_ci(ax, [0, 1], [wilson(10, 10), wilson(3, 10)], BLUE)
    plt.close(fig)


def test_finish_writes_png_and_data_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = finish(fig, "demo", [[0, 0], [1, 1]], ["x", "y"], tmp_path / "figures")
    assert out == tmp_path / "figures" / "demo.png" and out.stat().st_size > 0
    with open(tmp_path / "figures" / "demo_data.csv") as f:
        assert list(csv.reader(f)) == [["x", "y"], ["0", "0"], ["1", "1"]]


def test_finish_dicts_writes_header_from_first_row(tmp_path):
    fig, ax = plt.subplots()
    out = finish_dicts(fig, "demo", [{"x": 0, "y": 0.5}, {"x": 1, "y": 0.75}], tmp_path)
    assert out.exists()
    with open(tmp_path / "demo_data.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"x": "0", "y": "0.5"}, {"x": "1", "y": "0.75"}]


def test_titles_take_no_subcaption():
    # one title per figure and per panel, nothing underneath: a sub/subtitle parameter must not come back
    for fn in (header, panel_title):
        params = set(inspect.signature(fn).parameters)
        assert not {"sub", "subtitle", "caption", "y_sub"} & params, f"{fn.__name__} grew a sub-caption: {params}"


def test_header_and_panel_title_draw_one_text_each():
    fig, ax = plt.subplots()
    header(fig, "Figure title")
    panel_title(ax, "Panel title")
    assert [t.get_text() for t in fig.texts] == ["Figure title"]
    assert [t.get_text() for t in ax.texts] == [] and ax.get_title(loc="left") == "Panel title"
    plt.close(fig)
