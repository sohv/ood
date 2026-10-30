# Part 21: planted removals under a published topic taxonomy (30 Oct 2026)

Design: `designs/gap_taxonomy.md` Part 21 and its amendment, written before any call, reviewed by research-name a7abe3. Scripts: `scripts/gap_nber_topic.py` (tags), `scripts/gap_nber_topic_rank.py` (ranks and null), `scripts/gap_nber_topic_fig.py`. Outputs: `tags.json`, `summary.json`, raw replies in `raw/`. Figure: `figs/gap_taxonomy/nber_topic_removal.png`.

## Setup
- Taxonomy: the conversation-topic classifier of Chatterji et al. 2025, "How People Use ChatGPT" (NBER WP 34255, Appendix A.3). It has 24 categories in 7 coarse topics. The prompt is used word for word; only the format is changed (20 messages per call, JSON answer). The paper validated the coarse level only (model vs human plurality kappa 0.56).
- What the taxonomy has: a category for code (`computer_programming`). It has none for harmful requests and none for non-English text.
- Prompts: the part 8 and 15 sample, 2,000 traffic and 2,000 Dolci prompts. Tagged: 3,980.
  - One batch of 20 failed 3 times because Sonnet's cyber safeguards flagged it.
  - Among the removed prompts it lost 1 of 105 safety and 1 of 99 multilingual.
- Labeller: Claude Sonnet 5.5. A batching check on 40 prompts, labelled one per call with the prompt unchanged, agreed on 36 of 40 (kappa 0.88).
- Test, the same as part 15: categories are ranked by traffic share minus Dolci share. The planted removal (minus_F) is compared with 20 random same-size removals from other families (seed 0). A family is found if its category's rank after removal beats at least 19 of 20 randoms.

## Results
1. Code is found at the fine level, which was expected from the removal size.
   - `computer_programming` moves from rank 23 to 2 of 24, while all 20 randoms leave it at 23.
   - The removal takes 397 of 2,000 Dolci prompts, and 383 of the 397 were tagged `computer_programming`.
2. Code is not found at the coarse level, the one the paper validated.
   - Technical Help stays at rank 7 of 7, the most Dolci-heavy topic, before and after the removal.
   - Technical Help holds both code and math. Math alone is 26 points more common in Dolci than in traffic. The code removal moves Technical Help from -37.0 to -26.0 points, which is not enough to change its rank.
3. The safety removal is not visible in any ranking.
   - The removed safety prompts were tagged across many categories: specific_info 20, how_to_advice 17, other 17, write_fiction 12.
   - The category with the largest drop is how_to_advice. Its rank moves from 6 to 5; the randoms give 5 to 6.
   - The pre-set max-drop statistic (amendment) does pass: 0.79 points against a random maximum of 0.49. But no category's rank leaves the random range.
4. The multilingual removal is visible, under a name that says nothing about language.
   - 55 of the 98 removed Aya prompts were tagged `specific_info`.
   - `specific_info` moves from rank 21 to 10 of 24; the randoms give 19 to 21. The max drop is 2.19 points against a random maximum of 0.47.
   - A reader of this gap list would see "traffic asks for more facts than Dolci", not "Dolci lacks non-English".
   - `translation` does not move (rank 20 after removal; randoms 19 to 21).
5. The natural differences at the top are the same chat-register kinds as before. Traffic minus Dolci, fine level: write_fiction +9.0 points, argument_or_summary_generation +6.4, games_and_role_play +4.1, unclear +3.7. Dolci-heavy: mathematical_calculation -26.2, computer_programming -11.3.

## What this does and does not show
- Bearing on claim 2's caveat: a category set written by others finds the removed kind when it has a category specific to it (code, fine level). It misses the kind when that category is merged with a larger difference (code, coarse level) or when there is no category (safety).
- Against "only when": the non-English removal shows up anyway, as a rise in a category that does not name the kind. So "found only when a category exists" should read "named correctly only when a category exists". A removal can surface under the wrong name.
- Code was expected from the removal size, so it does not count as a confirmed prediction.
- Limits:
  - One labeller and one run; the fine level is not validated by the paper.
  - The safety arm has 105 prompts and the multilingual arm 99, so a removal shifts category shares by at most about 5 points.
  - Point 4 is one family. Whether other unnamed kinds also surface under wrong names is untested.
- Most likely objection: "Sonnet is not OpenAI's classifier." That is addressed in part only. The categories and prompt are theirs; the labeller differs, and we have no agreement number against their labels.

Suggested wording for claim 2: "With a published taxonomy we did not write (24 ChatGPT topic categories), the code removal is found only at the fine level, where code has its own category. The safety removal is invisible. The non-English removal appears as 'more factual questions', so the kind is seen under the wrong name."
