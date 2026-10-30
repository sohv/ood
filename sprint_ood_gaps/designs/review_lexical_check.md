# Lexical retrieval check on the labelled real-gap topics (ood-wtfaretheydoing, written 2026-10-24 ~17:45 UTC, before running)

Question: when the user labelled a topic "real gap", was that because Dolci has no same-kind request, or because MiniLM's nearest neighbour missed one that a word-level search finds? Decides how the labels result is worded (absent vs thinly covered vs retrieval miss) and whether distance-flagged topics are worth a fix run.

Topics: every topic the user labelled real_gap: 17 distance-arm, 8 random-arm, 5 ratio-arm (30 topics; arms included so the check is not run on the distance arm alone). Prompts: the same 10 shown prompts per topic (recovered from items.json by prefix match on the topic's members), full text.

Retrieval: tf-idf over all 1,710,578 unique Dolci first turns (word 1-2 grams, sublinear tf, min_df 3), top 3 by cosine per prompt, plus the MiniLM nearest already shown and the exact-member flag (traffic_hash in mix_hash).

Judgement: Sole (gpt-6.1-sol) reads one sheet per topic, blind to arm and to the user's label, and for each prompt says whether any of the 4 retrieved Dolci prompts (1 MiniLM, 3 lexical) is the same kind of request. Topic verdict by the same rule as the user's screen: most prompts matched = covered, some = partly, none or almost none = real_gap. The user can read the same sheets (runs/review/lexical/sheets.md).

Reading, fixed now:
- If lexical retrieval finds a same-kind Dolci prompt for >= half of the shown prompts in most distance real-gap topics, those topics are retrieval misses by MiniLM and the label result says little about Dolci coverage.
- If lexical retrieval overturns few distance topics (<= 4 of 17) and about as many in the other arms, the labels stand as "no same-kind example found by two retrievers", still not proof of absence.
- Exploratory either way: one model judge, retrieval limited to top 3 lexical.
Cost: CPU only, minutes. Output: runs/review/lexical/{retrieval.json, sheets.md, sole_verdicts.md}.
