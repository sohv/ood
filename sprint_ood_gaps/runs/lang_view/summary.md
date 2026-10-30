# Language view of the Aya removal (1,290,414 traffic first turns, 1,710,578 Dolci prompts, 89,038 removed)



**Note (read before using the gap columns).** The 'gap' rule (ratio at least 3x the median language ratio) is relative, and the Aya removal lowers coverage for most languages at once, so the median itself rises from 3.4 to 8.6 and no language crosses the moved line. The gap columns are therefore not a usable test of recovery; use the ratio change and the Aya share. The removal itself matches Sohan's rule exactly (89,038 prompts removed, his removal_code_multilingual number).

New gaps after removing Aya: []
New gaps after random removal: []

| lang | traffic | Dolci | removed (Aya) | Aya share of Dolci lang | ratio full | ratio minus Aya | ratio random | gap full | gap minus Aya |
|---|---|---|---|---|---|---|---|---|---|
| id | 3,137 | 3,482 | 3,104 | 0.891 | 1.19 | 10.4 | 1.19 | False | False |
| vi | 4,858 | 5,396 | 4,440 | 0.823 | 1.19 | 6.38 | 1.2 | False | False |
| lt | 355 | 485 | 391 | 0.806 | 0.97 | 4.71 | 0.97 | False | False |
| tl | 238 | 766 | 617 | 0.805 | 0.41 | 2.0 | 0.41 | False | False |
| tr | 4,894 | 2,293 | 1,847 | 0.805 | 2.83 | 13.76 | 2.84 | False | False |
| el | 1,027 | 391 | 307 | 0.785 | 3.48 | 15.2 | 3.46 | False | False |
| sv | 802 | 807 | 626 | 0.776 | 1.32 | 5.54 | 1.3 | False | False |
| nl | 2,041 | 1,124 | 857 | 0.762 | 2.41 | 9.57 | 2.42 | False | False |
| fi | 427 | 727 | 546 | 0.751 | 0.78 | 2.96 | 0.79 | False | False |
| ja | 11,953 | 4,170 | 3,071 | 0.736 | 3.8 | 13.66 | 3.8 | False | False |
| ar | 11,327 | 7,656 | 5,591 | 0.73 | 1.96 | 6.89 | 1.96 | False | False |
| pt | 30,166 | 5,109 | 3,568 | 0.698 | 7.83 | 24.58 | 7.83 | False | False |
| da | 425 | 71 | 49 | 0.69 | 7.84 | 23.27 | 7.76 | False | False |
| hi | 241 | 1,529 | 987 | 0.646 | 0.21 | 0.56 | 0.21 | False | False |
| uk | 2,227 | 449 | 241 | 0.537 | 6.56 | 13.4 | 6.65 | False | False |
| hu | 1,454 | 152 | 79 | 0.52 | 12.61 | 24.71 | 12.61 | True | False |
| zh | 98,221 | 4,406 | 2,253 | 0.511 | 29.54 | 57.3 | 29.61 | True | True |
| pl | 5,030 | 1,430 | 699 | 0.489 | 4.66 | 8.64 | 4.65 | False | False |
| th | 375 | 676 | 324 | 0.479 | 0.74 | 1.34 | 0.73 | False | False |
| no | 259 | 21 | 10 | 0.476 | 15.67 | 27.23 | 15.56 | True | True |
| ko | 5,634 | 458 | 166 | 0.362 | 16.27 | 24.17 | 16.09 | True | False |
| it | 16,923 | 1,470 | 503 | 0.342 | 15.25 | 21.97 | 15.32 | True | False |
| fa | 3,628 | 2,420 | 758 | 0.313 | 1.99 | 2.74 | 2.01 | False | False |
| fr | 33,779 | 4,286 | 1,297 | 0.303 | 10.45 | 14.2 | 10.43 | True | False |
| ca | 375 | 332 | 96 | 0.289 | 1.5 | 1.99 | 1.47 | False | False |
| es | 33,806 | 7,289 | 1,758 | 0.241 | 6.15 | 7.68 | 6.15 | False | False |
| ru | 98,462 | 8,358 | 727 | 0.087 | 15.61 | 16.21 | 15.64 | True | False |
| de | 21,384 | 2,261 | 165 | 0.073 | 12.53 | 12.81 | 12.59 | True | False |
| ro | 743 | 140 | 10 | 0.071 | 6.99 | 7.14 | 6.93 | False | False |
| cs | 863 | 584 | 28 | 0.048 | 1.96 | 1.95 | 1.97 | False | False |
| bg | 311 | 120 | 1 | 0.008 | 3.42 | 3.27 | 3.44 | False | False |
| en | 815,114 | 1,537,746 | 2,064 | 0.001 | 0.7 | 0.67 | 0.7 | False | False |
| he | 532 | 529 | 0 | 0.0 | 1.33 | 1.26 | 1.32 | False | False |

## Fixed line (set once on the full Dolci, 3 x median = 10.25, applied to every arm)

- Already gaps with full Dolci: hu, zh, no, ko, it, fr, ru, de
- New gaps after removing Aya: id (Aya share 0.891, ratio 1.19 -> 10.4), tr (Aya share 0.805, ratio 2.83 -> 13.76), el (Aya share 0.785, ratio 3.48 -> 15.2), ja (Aya share 0.736, ratio 3.8 -> 13.66), pt (Aya share 0.698, ratio 7.83 -> 24.58), da (Aya share 0.69, ratio 7.84 -> 23.27), uk (Aya share 0.537, ratio 6.56 -> 13.4)
- New gaps after random removal of the same size: none
- The line rule is set here by analogy with the topic-level line, not tuned; languages with at least 200 traffic prompts.
