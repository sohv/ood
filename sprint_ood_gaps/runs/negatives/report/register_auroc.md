## Register vs coverage, length-stratified AUROC

| positive                 | against                  |   one_sided |   ratio |   tfidf_p_traffic |   olmo_len |
|:-------------------------|:-------------------------|------------:|--------:|------------------:|-----------:|
| pos_legal                | reg_askscience           |       0.611 |   0.672 |             0.487 |      0.652 |
| pos_legal                | reg_cooking              |       0.616 |   0.435 |             0.473 |      0.646 |
| pos_legal                | reg_fitness              |       0.688 |   0.597 |             0.534 |      0.59  |
| pos_legal                | reg_learnprogramming     |       0.769 |   0.539 |             0.494 |      0.554 |
| pos_legal                | reg_relationship_advice  |       0.387 |   0.177 |             0.275 |      0.471 |
| pos_legal                | reg_relationships        |       0.405 |   0.181 |             0.271 |      0.366 |
| pos_legal                | reg_writing              |       0.761 |   0.51  |             0.361 |      0.57  |
| pos_legal                | all covered-topic Reddit |       0.575 |   0.403 |             0.384 |      0.518 |
| pos_medical              | reg_askscience           |       0.631 |   0.911 |             0.6   |      0.49  |
| pos_medical              | reg_cooking              |       0.653 |   0.786 |             0.58  |      0.452 |
| pos_medical              | reg_fitness              |       0.631 |   0.875 |             0.626 |      0.417 |
| pos_medical              | reg_learnprogramming     |       0.776 |   0.78  |             0.557 |      0.407 |
| pos_medical              | reg_relationship_advice  |       0.448 |   0.462 |             0.463 |      0.339 |
| pos_medical              | reg_relationships        |       0.438 |   0.425 |             0.426 |      0.233 |
| pos_medical              | reg_writing              |       0.757 |   0.815 |             0.488 |      0.415 |
| pos_medical              | all covered-topic Reddit |       0.667 |   0.808 |             0.561 |      0.432 |
| reg_legaladvice2         | reg_askscience           |       0.615 |   0.673 |             0.425 |      0.656 |
| reg_legaladvice2         | reg_cooking              |       0.617 |   0.432 |             0.414 |      0.647 |
| reg_legaladvice2         | reg_fitness              |       0.692 |   0.583 |             0.491 |      0.59  |
| reg_legaladvice2         | reg_learnprogramming     |       0.767 |   0.538 |             0.456 |      0.55  |
| reg_legaladvice2         | reg_relationship_advice  |       0.405 |   0.181 |             0.237 |      0.46  |
| reg_legaladvice2         | reg_relationships        |       0.402 |   0.183 |             0.238 |      0.348 |
| reg_legaladvice2         | reg_writing              |       0.756 |   0.506 |             0.322 |      0.567 |
| reg_legaladvice2         | all covered-topic Reddit |       0.581 |   0.4   |             0.341 |      0.506 |
| rw_legal                 | rw_wildchat              |       0.922 |   0.303 |             0.284 |      0.482 |
| rw_legal                 | 23 negatives             |       0.886 |   0.818 |             0.76  |      0.521 |
| rw_medical               | rw_wildchat              |       0.897 |   0.695 |             0.472 |      0.456 |
| rw_medical               | 23 negatives             |       0.877 |   0.91  |             0.786 |      0.513 |
| all covered-topic Reddit | 23 negatives             |       0.847 |   0.842 |             0.847 |      0.539 |

## Medians

| set                     | kind               |   n |   median_one_sided |   median_ratio |   median_len |
|:------------------------|:-------------------|----:|-------------------:|---------------:|-------------:|
| reg_relationships       | register_covered   | 500 |              0.411 |          3.876 |        445   |
| reg_relationship_advice | register_covered   | 500 |              0.41  |          4.249 |        309.5 |
| pos_medical             | planted            | 500 |              0.402 |          3.876 |         84   |
| reg_legaladvice2        | register_uncovered | 500 |              0.393 |          1.723 |        230.5 |
| pos_legal               | planted            | 500 |              0.392 |          1.723 |        217.5 |
| rw_legal                | rewrite_uncovered  | 500 |              0.39  |          1.723 |        124.5 |
| rw_medical              | rewrite_uncovered  | 500 |              0.387 |          3.254 |         75   |
| reg_askscience          | register_covered   | 500 |              0.374 |          1.167 |         80   |
| reg_cooking             | register_covered   | 500 |              0.37  |          2.01  |        100   |
| reg_fitness             | register_covered   | 500 |              0.356 |          1.477 |        146.5 |
| reg_writing             | register_covered   | 500 |              0.34  |          1.595 |        136.5 |
| reg_learnprogramming    | register_covered   | 500 |              0.332 |          1.723 |        150.5 |
| rw_wildchat             | rewrite_control    | 500 |              0.186 |          2.543 |         58   |

23 negatives pooled median one_sided: 0.255
