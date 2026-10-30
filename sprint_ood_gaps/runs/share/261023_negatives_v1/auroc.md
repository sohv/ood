## Planted vs negatives and covered, AUROC (higher = planted scored more gap-like)

|                                  | ratio                | ratio_fp32           | one_sided            | olmo_len             | tfidf_p_traffic      |
|:---------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| ('pos_legal', 'all covered')     | 0.837                | 0.838                | 0.951                | 0.712                | 0.779                |
| ('pos_legal', 'all negatives')   | 0.718 [0.704, 0.730] | 0.718 [0.704, 0.731] | 0.898 [0.887, 0.908] | 0.806 [0.795, 0.818] | 0.650 [0.636, 0.663] |
| ('pos_medical', 'all covered')   | 0.929                | 0.930                | 0.952                | 0.485                | 0.829                |
| ('pos_medical', 'all negatives') | 0.865 [0.854, 0.875] | 0.865 [0.855, 0.876] | 0.902 [0.890, 0.913] | 0.632 [0.620, 0.643] | 0.718 [0.706, 0.730] |

## Per negative family

|                                     |   ratio |   ratio_fp32 |   one_sided |   olmo_len |   tfidf_p_traffic |
|:------------------------------------|--------:|-------------:|------------:|-----------:|------------------:|
| ('pos_legal', 'neg:code')           |   0.911 |        0.911 |       0.965 |      0.715 |             0.873 |
| ('pos_legal', 'neg:flan_style')     |   0.614 |        0.615 |       0.588 |      0.794 |             0.423 |
| ('pos_legal', 'neg:general_chat')   |   0.424 |        0.425 |       0.856 |      0.907 |             0.324 |
| ('pos_legal', 'neg:math')           |   0.927 |        0.928 |       0.966 |      0.946 |             0.885 |
| ('pos_legal', 'neg:precise_if')     |   0.795 |        0.796 |       0.952 |      0.918 |             0.997 |
| ('pos_legal', 'neg:reasoning')      |   0.999 |        0.999 |       0.997 |      0.489 |             0.968 |
| ('pos_legal', 'neg:safety')         |   0.386 |        0.386 |       0.832 |      0.978 |             0.277 |
| ('pos_legal', 'neg:science')        |   0.582 |        0.583 |       0.934 |      0.997 |             0.455 |
| ('pos_legal', 'neg:table')          |   0.741 |        0.742 |       0.807 |      1     |             0.742 |
| ('pos_legal', 'neg:tool_use')       |   0.84  |        0.841 |       0.905 |      0.429 |             0.734 |
| ('pos_medical', 'neg:code')         |   0.969 |        0.969 |       0.964 |      0.466 |             0.906 |
| ('pos_medical', 'neg:flan_style')   |   0.831 |        0.832 |       0.634 |      0.587 |             0.537 |
| ('pos_medical', 'neg:general_chat') |   0.704 |        0.704 |       0.865 |      0.811 |             0.422 |
| ('pos_medical', 'neg:math')         |   0.974 |        0.974 |       0.963 |      0.774 |             0.938 |
| ('pos_medical', 'neg:precise_if')   |   0.931 |        0.932 |       0.948 |      0.666 |             0.999 |
| ('pos_medical', 'neg:reasoning')    |   1     |        1     |       0.995 |      0.124 |             0.987 |
| ('pos_medical', 'neg:safety')       |   0.616 |        0.617 |       0.844 |      0.94  |             0.369 |
| ('pos_medical', 'neg:science')      |   0.855 |        0.855 |       0.932 |      0.976 |             0.553 |
| ('pos_medical', 'neg:table')        |   0.91  |        0.91  |       0.819 |      0.986 |             0.824 |
| ('pos_medical', 'neg:tool_use')     |   0.931 |        0.932 |       0.903 |      0.098 |             0.807 |

## Planted vs all negatives within Olmo-length bands (weighted by planted rows; bands with >= 20 rows each side)

| planted     |   ratio |   ratio_fp32 |   one_sided |   olmo_len |   tfidf_p_traffic |
|:------------|--------:|-------------:|------------:|-----------:|------------------:|
| pos_legal   |   0.856 |        0.857 |       0.905 |      0.535 |             0.831 |
| pos_medical |   0.925 |        0.925 |       0.905 |      0.497 |             0.855 |
