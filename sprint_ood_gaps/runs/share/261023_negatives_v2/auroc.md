## Planted vs negatives and covered, AUROC (higher = planted scored more gap-like)

|                                  | ratio                | ratio_fp32           | one_sided            | olmo_len             | tfidf_p_traffic      |
|:---------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| ('pos_legal', 'all covered')     | 0.837                | 0.837                | 0.949                | 0.712                | 0.779                |
| ('pos_legal', 'all negatives')   | 0.723 [0.708, 0.737] | 0.723 [0.709, 0.737] | 0.895 [0.884, 0.905] | 0.801 [0.789, 0.813] | 0.656 [0.642, 0.670] |
| ('pos_medical', 'all covered')   | 0.929                | 0.930                | 0.950                | 0.485                | 0.829                |
| ('pos_medical', 'all negatives') | 0.869 [0.858, 0.880] | 0.870 [0.858, 0.880] | 0.899 [0.887, 0.911] | 0.625 [0.614, 0.636] | 0.725 [0.712, 0.737] |

## Per negative family

|                                     |   ratio |   ratio_fp32 |   one_sided |   olmo_len |   tfidf_p_traffic |
|:------------------------------------|--------:|-------------:|------------:|-----------:|------------------:|
| ('pos_legal', 'neg:code')           |   0.905 |        0.906 |       0.963 |      0.705 |             0.864 |
| ('pos_legal', 'neg:flan_style')     |   0.614 |        0.615 |       0.588 |      0.794 |             0.423 |
| ('pos_legal', 'neg:general_chat')   |   0.444 |        0.444 |       0.842 |      0.893 |             0.356 |
| ('pos_legal', 'neg:math')           |   0.925 |        0.925 |       0.965 |      0.945 |             0.883 |
| ('pos_legal', 'neg:precise_if')     |   0.795 |        0.796 |       0.952 |      0.918 |             0.997 |
| ('pos_legal', 'neg:reasoning')      |   0.999 |        0.999 |       0.997 |      0.489 |             0.968 |
| ('pos_legal', 'neg:safety')         |   0.386 |        0.387 |       0.832 |      0.977 |             0.277 |
| ('pos_legal', 'neg:science')        |   0.582 |        0.583 |       0.934 |      0.997 |             0.455 |
| ('pos_legal', 'neg:table')          |   0.741 |        0.742 |       0.807 |      1     |             0.742 |
| ('pos_legal', 'neg:tool_use')       |   0.84  |        0.841 |       0.905 |      0.429 |             0.734 |
| ('pos_medical', 'neg:code')         |   0.967 |        0.967 |       0.961 |      0.462 |             0.899 |
| ('pos_medical', 'neg:flan_style')   |   0.831 |        0.832 |       0.634 |      0.587 |             0.537 |
| ('pos_medical', 'neg:general_chat') |   0.725 |        0.725 |       0.852 |      0.788 |             0.458 |
| ('pos_medical', 'neg:math')         |   0.973 |        0.973 |       0.963 |      0.77  |             0.936 |
| ('pos_medical', 'neg:precise_if')   |   0.931 |        0.932 |       0.948 |      0.666 |             0.999 |
| ('pos_medical', 'neg:reasoning')    |   1     |        1     |       0.995 |      0.124 |             0.987 |
| ('pos_medical', 'neg:safety')       |   0.616 |        0.616 |       0.844 |      0.94  |             0.368 |
| ('pos_medical', 'neg:science')      |   0.855 |        0.855 |       0.932 |      0.976 |             0.553 |
| ('pos_medical', 'neg:table')        |   0.91  |        0.91  |       0.819 |      0.986 |             0.824 |
| ('pos_medical', 'neg:tool_use')     |   0.931 |        0.932 |       0.903 |      0.098 |             0.807 |

## Planted vs all negatives within Olmo-length bands (weighted by planted rows; bands with >= 20 rows each side)

| planted     |   ratio |   ratio_fp32 |   one_sided |   olmo_len |   tfidf_p_traffic |
|:------------|--------:|-------------:|------------:|-----------:|------------------:|
| pos_legal   |   0.856 |        0.856 |       0.903 |      0.533 |             0.83  |
| pos_medical |   0.925 |        0.926 |       0.902 |      0.497 |             0.856 |
