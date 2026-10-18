# plot style and figure helpers: call style() before drawing, tidy(ax) on every axis, finish() to save png + data csv.
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

FONTS = Path(__file__).resolve().parent / "fonts"

# neutral tokens: off-white surface, light grid, near-black ink, grey muted
SURFACE, GRID, INK, MUTED = "#fcfcfb", "#e8e8e4", "#2b2b2b", "#6b6b66"
SPINE = "#c9c9c4"
BAND = "#dedcd6"  # reference band (axhspan) colour
LINK = "#cfcfca"  # grey connector line between paired points

# dataviz palette, validated 2026-09-08 (light surface): blue, orange, aqua.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#b9b9b3"  # the "all / pooled" series next to the palette colours


def style():
    """Inter (SIL OFL, src/utils/fonts/) if present, else DejaVu; light surface."""
    from matplotlib import font_manager as fm

    for f in FONTS.glob("Inter-*.ttf"):
        fm.fontManager.addfont(str(f))
    fam = "Inter" if any(x.name == "Inter" for x in fm.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": fam,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.8,
        }
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(p, lo, hi). Wilson score interval; nan triple when n == 0."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def tidy(ax, grid_axis: str = "y", drop_left: bool = True):
    """Grid on one axis behind the data; top/right spines off, left too when the grid carries the y-scale;
    no y tick marks, short light-grey x tick marks."""
    ax.grid(axis=grid_axis, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right") + (("left",) if drop_left else ()):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color=SPINE)


def percent_axis(ax):
    """Proportions as percentages, fixed ticks at 0/25/50/75/100 with a little headroom."""
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def point_ci(ax, xs, ps, hexc: str, marker: str = "o", filled: bool = True, label: str | None = None):
    """Point + Wilson CI. `ps` is a list of (p, lo, hi). Filled points get a surface-coloured halo;
    hollow points are surface-filled with an ink edge, so shape and fill carry a second channel."""
    # max(0, .): at p = 1 the Wilson bound is 1 - eps and hi - p is -0.0, which errorbar rejects
    ax.errorbar(
        xs,
        [q[0] for q in ps],
        yerr=[[max(0.0, q[0] - q[1]) for q in ps], [max(0.0, q[2] - q[0]) for q in ps]],
        fmt="none",
        ecolor=hexc,
        elinewidth=1.7,
        capsize=0,
        alpha=0.85,
        zorder=2,
    )
    ax.scatter(
        xs,
        [q[0] for q in ps],
        s=64 if filled else 58,
        marker=marker,
        zorder=4,
        facecolor=hexc if filled else SURFACE,
        edgecolor=SURFACE if filled else INK,
        linewidth=1.4 if filled else 1.3,
        label=label,
    )


def reference_band(ax, p: float, lo: float, hi: float, label: str | None = None, primary: bool = True):
    """Warm-grey band for a reference CI plus a dashed line at its centre; dash pattern distinguishes reference types."""
    ax.axhspan(lo, hi, color=BAND, alpha=0.55 if primary else 0.3, lw=0, zorder=0, label=label)
    ax.axhline(p, color=MUTED, lw=0.9, ls=(0, (5, 4)) if primary else (0, (1, 3)), zorder=1)


def count_bars(ax, xs, fracs, counts, hexc: str, filled: bool = True, w: float = 0.22):
    """observed count of an exact quantity: plain bars, labelled with the count"""
    ax.bar(xs, fracs, w, color=hexc, lw=0, zorder=3)
    if not filled:  # hollow variants keep the shape channel: ink outline
        ax.bar(xs, fracs, w, fill=False, edgecolor=INK, lw=0.9, zorder=4)
    for x, f, c in zip(xs, fracs, counts):
        if c:
            ax.text(x, f + 0.03, str(c), ha="center", va="bottom", fontsize=7, color=MUTED)


def panel_title(ax, title: str):
    """Left-aligned semibold panel title."""
    ax.set_title(title, fontsize=10, fontweight="semibold", color=INK, loc="left", pad=6)


def _fig_bbox(fig, artist):
    """artist's bounding box in figure fractions, drawn first so text extents are real"""
    fig.canvas.draw()
    return artist.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())


def _axes_extent(fig) -> tuple[float, float]:
    """(lowest, highest) figure fraction any axes reaches, tick labels, axis labels and panel titles included"""
    boxes = [_fig_bbox(fig, ax) for ax in fig.axes] + [
        ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()) for ax in fig.axes
    ]
    return min(b.y0 for b in boxes), max(b.y1 for b in boxes)


def header(fig, suptitle: str, x: float = 0.045, y_title: float = 0.985, gap_in: float = 0.2):
    """Editorial bold title top-left. Pushes the axes down until everything they draw clears it by gap_in inches."""
    title = fig.text(x, y_title, suptitle, fontsize=13, fontweight="bold", color=INK, va="top")
    overshoot = _axes_extent(fig)[1] - (_fig_bbox(fig, title).y0 - gap_in / fig.get_figheight())
    if overshoot > 0:
        fig.subplots_adjust(top=fig.subplotpars.top - overshoot)


def legend_below(fig, handles, labels, x: float = 0.045, y: float = 0.01, ncol: int = 2, gap_in: float = 0.2):
    """Frameless legend under the axes, left-aligned with the header. Pushes the axes up until tick labels and
    x labels clear it by gap_in inches."""
    leg = fig.legend(
        handles,
        labels,
        loc="lower left",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(x - 0.005, y),
        handletextpad=0.5,
        columnspacing=1.6,
        labelspacing=0.45,
    )
    overshoot = (_fig_bbox(fig, leg).y1 + gap_in / fig.get_figheight()) - _axes_extent(fig)[0]
    if overshoot > 0:
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + overshoot)


def finish(
    fig, name: str, rows: list, header_row: list, out_dir: Path | str, dpi: int = 220, pad_inches: float | None = None
) -> Path:
    """Every figure writes out_dir/<name>.png and <name>_data.csv (exactly what is plotted)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = {"pad_inches": pad_inches} if pad_inches is not None else {}
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight", **kw)
    with open(out_dir / f"{name}_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header_row)
        w.writerows(rows)
    plt.close(fig)
    print(f"wrote {out_dir / name}.png + _data.csv")
    return out_dir / f"{name}.png"


def finish_dicts(
    fig, name: str, data: list[dict], out_dir: Path | str, dpi: int = 250, pad_inches: float = 0.12
) -> Path:
    """Same as finish() for a list of dict rows (DictWriter, fieldnames from the first row)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    with open(out_dir / f"{name}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    print(f"wrote {out_dir / name}.png + _data.csv  ({len(data)} points)")
    return out_dir / f"{name}.png"
