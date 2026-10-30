"""Simple negatives figure for the 26 Oct explainer: share of each unseen same-kind set above the gap line."""
import pandas as pd, matplotlib.pyplot as plt
d = pd.read_csv('runs/negatives/report/per_set.csv')
neg = d[d.kind == 'negative'].copy()
neg['bound'] = (2 * neg.covered_share_above_line).clip(lower=0.02)
neg = neg.sort_values('share_above_line')
pl = d[d.kind == 'planted'].sort_values('share_above_line')
fig, ax = plt.subplots(figsize=(8, 8.2))
y = list(range(len(neg)))
cols = ['#2a78d6' if b == 'unflagged' else '#eb6834' for b in neg.bar_a]
ax.barh(y, 100 * neg.share_above_line, color=cols, height=0.7)
ax.scatter(100 * neg.bound, y, marker='|', s=160, color='#0b0b0b', zorder=3)
yp = [len(neg) + 1 + i for i in range(len(pl))]
ax.barh(yp, 100 * pl.share_above_line, color='#1baf7a', height=0.7)
labels = [f'{s}  (median {m:.1f})' for s, m in zip(neg.set, neg.median_ratio)]
labels += [f'planted {s.replace("pos_", "")}  (median {m:.1f})' for s, m in zip(pl.set, pl.median_ratio)]
ax.set_yticks(y + yp, labels, fontsize=9)
xs = [max(v, b) for v, b in zip(neg.share_above_line, neg.bound)] + list(pl.share_above_line)
for yy, v, x in zip(y + yp, list(neg.share_above_line) + list(pl.share_above_line), xs):
    ax.text(100 * x + 0.4, yy, f'{100*v:.1f}%', va='center', fontsize=8, color='#52514e')
ax.set_xlabel('% of the set\'s prompts above the gap line (ratio > 12.1)')
ax.set_title('Gap score on 23 unseen datasets of the same kinds as Dolci, Olmo-3-7B\n'
             'MiniLM k=50 neighbour ratio vs 993k WildChat/LMSYS prompts\nset medians in brackets; gap line 12.1',
             fontsize=10, loc='left')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color='#2a78d6', label='passes bar a (18 sets)'),
                   Patch(color='#eb6834', label='fails bar a (5 sets)'),
                   Line2D([], [], marker='|', ls='', color='#0b0b0b', ms=12,
                          label='allowed share: 2x its Dolci family, at least 2%'),
                   Patch(color='#1baf7a', label='planted legal/medical (meant to be flagged)')],
          loc='lower right', fontsize=8.5, frameon=False)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.set_xlim(0, 22)
fig.tight_layout()
fig.savefig('figs/negatives/negatives_simple.png', dpi=150)
