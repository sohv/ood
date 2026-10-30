Labelled: 90 of 90 topics (user). Astra labels: 90 topics, all used for its own table.

## User labels (primary)

| arm        |   n |   unsure |   real_gap |   share |   wilson_lo |   wilson_hi |   real_or_partly |   share_real_or_partly |
|:-----------|----:|---------:|-----------:|--------:|------------:|------------:|-----------------:|-----------------------:|
| rare_only  |  27 |        3 |         17 |   0.63  |       0.442 |       0.785 |               27 |                  1     |
| ratio_only |  27 |        3 |          5 |   0.185 |       0.082 |       0.367 |               16 |                  0.593 |
| control    |  27 |        3 |          8 |   0.296 |       0.159 |       0.485 |               23 |                  0.852 |

- rare_only vs control: real_gap share 17/27 vs 8/27, difference +0.33, Fisher p = 0.028 -> finds gaps vs control
- ratio_only vs control: real_gap share 5/27 vs 8/27, difference -0.11, Fisher p = 0.526 -> no excess
- rare_only vs ratio_only: real_gap share 17/27 vs 5/27, difference +0.44, Fisher p = 0.002 -> finds gaps vs ratio_only

- Stefan's bar (real_gap share >= 0.50): rare_only met (0.63), ratio_only not met (0.19), control not met (0.30)

## Astra (GPT-6) labels, same screens

| arm        |   n |   unsure |   real_gap |   share |   wilson_lo |   wilson_hi |   real_or_partly |   share_real_or_partly |
|:-----------|----:|---------:|-----------:|--------:|------------:|------------:|-----------------:|-----------------------:|
| rare_only  |  30 |        0 |         10 |   0.333 |       0.192 |       0.512 |               28 |                  0.933 |
| ratio_only |  30 |        0 |          1 |   0.033 |       0.006 |       0.167 |               13 |                  0.433 |
| control    |  30 |        0 |          5 |   0.167 |       0.073 |       0.336 |               21 |                  0.7   |

- rare_only vs control: real_gap share 10/30 vs 5/30, difference +0.17, Fisher p = 0.233 -> inconclusive
- ratio_only vs control: real_gap share 1/30 vs 5/30, difference -0.13, Fisher p = 0.195 -> no excess
- rare_only vs ratio_only: real_gap share 10/30 vs 1/30, difference +0.30, Fisher p = 0.006 -> finds gaps vs ratio_only

## Agreement (topics both labelled, user's unsure excluded)

- n = 81; exact agreement 0.59; Cohen's kappa (3 classes) 0.37; kappa on real_gap vs not 0.40

| user     |   covered |   partly |   real_gap |
|:---------|----------:|---------:|-----------:|
| covered  |        13 |        2 |          0 |
| partly   |        11 |       24 |          1 |
| real_gap |         1 |       18 |         11 |

Context only: Sohan's deep_gap_loop labels on his loop gaps, 19 of 30 real (his labels, topic names shown).