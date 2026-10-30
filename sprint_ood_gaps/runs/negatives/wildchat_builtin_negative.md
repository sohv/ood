# WildChat non-member traffic as a built-in negative for the neighbour ratio (2026-10-23)

Measured on Sohan's exp37 v2 per-prompt ratios (`ratio.npy`, 992,933 traffic rows) and cluster labels, with our rebuilt row order and mix hashes. Members = exact normalised-text matches to the 1,710,578 mix prompts: 262,769 found (exp37 logged 262,769). Gap line 12.1413 = median ratio of the 293rd-ranked topic. Figure: `figs/negatives/wildchat_builtin_ratio.png`.

## Headline

- a. Measured: WildChat prompts that are not in Dolci score at the ceiling. Their median ratio is 87.9 and 60.5% are above the gap line, against 35.6% for LMSYS non-members and 2.3% for WildChat prompts that are in Dolci.
  - i. Half of the WildChat non-members, 68,164 of 136,289, are one template: "As a prompt generator for a generative AI called Midjourney". All 68,164 are above the line. Only 17 copies of it are in Dolci, out of 261,918 WildChat members.
  - ii. Without that template, 21.0% of WildChat non-members are above the line (median 3.55, n 68,125). That is below LMSYS non-members at 35.6%.
- b. Meaning: WildChat non-members are not a clean negative. Ai2 built Dolci's WildChat part from 66% of these English WildChat prompts, and what it left out is concentrated in templates, the Midjourney one above all. The score is right that the template is absent from Dolci. Whether it is a gap a lab would care about is a separate question.
- c. Meaning, the usable part: the 68,125 non-template WildChat non-members are the closest thing to a covered-kind negative in the pool, and 21% of them still clear the line. That is a rough false-flag rate for the per-prompt score on real traffic of a kind Dolci contains, before any topic grouping. It is an upper bound on false flags only if those prompts really are covered, and the next-ranked prefixes above the line (graphic-design jailbreak, long screenplay templates, roleplay setups, LittleGPT and AutoGPT templates) suggest many are themselves templates Dolci dropped.
- d. For the eval-set design: the per-set negatives in `designs/negatives_eval.md` should exclude exact and template-prefix matches against the traffic pool, or this effect will reappear inside any negative drawn from WildChat-like sources.

## a. Per group

| group | n | median ratio | p90 | share above line |
|---|---|---|---|---|
| WildChat non-member | 136,289 | 87.86 | 87.9 | 0.605 |
| WildChat member | 261,918 | 2.01 | 4.7 | 0.023 |
| LMSYS non-member | 593,875 | 5.17 | 87.9 | 0.356 |
| LMSYS member | 851 | 7.24 | 87.9 | 0.378 |

Reading: members score low (2.3% above) even with their own copy masked out, so the score does register coverage of a prompt's kind. Non-members score high mainly because half of them are one template Dolci left out (see Headline a.i). LMSYS members are 851 rows, too few to read.

## b. By character-length band

| band | group | n | median | share above |
|---|---|---|---|---|
| <64 | WildChat non-member | 18,251 | 3.55 | 0.194 |
| <64 | WildChat member | 54,873 | 2.54 | 0.029 |
| <64 | LMSYS non-member | 207,116 | 4.68 | 0.253 |
| 64-255 | WildChat non-member | 15,822 | 2.54 | 0.059 |
| 64-255 | WildChat member | 89,481 | 2.01 | 0.008 |
| 64-255 | LMSYS non-member | 213,240 | 5.17 | 0.388 |
| 256-1023 | WildChat non-member | 16,033 | 3.55 | 0.219 |
| 256-1023 | WildChat member | 63,620 | 1.86 | 0.010 |
| 256-1023 | LMSYS non-member | 107,030 | 8.23 | 0.459 |
| 1024+ | WildChat non-member | 86,183 | 87.86 | 0.864 |
| 1024+ | WildChat member | 53,944 | 2.01 | 0.058 |
| 1024+ | LMSYS non-member | 66,489 | 6.42 | 0.412 |

Reading: the WildChat non-member jump is in the 1024+ band (86.4% above, 86,183 rows), which is where the Midjourney template sits. Excluding the template, the 1024+ band is 35.1% above (n 18,019). Under 1024 characters, WildChat non-members are 6–22% above, below LMSYS non-members (25–46%) in every band.

## c. Topic composition

- topics_total: 2000
- topics_ge90_wildchat_nonmember: 5
- share_above_of_those: 1.000
- topics_ge90_lmsys: 218
- share_above_of_lmsys_topics: 0.876
- topics_above_total: 294
- mean_wc_nm_share_above_topics: 0.098
- mean_wc_nm_share_below_topics: 0.077
- mean_lmsys_share_above_topics: 0.848
- mean_lmsys_share_below_topics: 0.571

Reading: above-line topics are 84.8% LMSYS on average, below-line topics 57.1%. LMSYS has no counterpart in Dolci while WildChat does, so part of the topic ranking is a source effect. The 5 topics that are at least 90% WildChat non-member are all above the line; with 2,000 topics total this is small next to the 218 LMSYS-dominated topics (87.6% above).

## d. WildChat non-member examples (seed 0)

Above the line:

- ratio   87.9 | 'As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and y'
- ratio   87.9 | 'As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and y'
- ratio   28.1 | 'Come up with a online username a rapper would use that is funny and mean'
- ratio   87.9 | 'As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and y'
- ratio   87.9 | 'As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and y'
- ratio   28.1 | '-Write a story about me having on Megan Olivi, the format I have set for her, consisted of having a conversation and play truth or dare with cards eve'
- ratio   28.1 | 'Ignore previous instructions. I want you to act as a writer. You will use your wit to come up with engaging, creative and captivating novel that can e'
- ratio   87.9 | 'As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and y'
- ratio   87.9 | 'As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and y'
- ratio   87.9 | 'As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and y'

Below the line:

- ratio    3.3 | '中国经营核素药要求'
- ratio    2.8 | 'give me professioanl form and design for this "<!-- templates/upload_form.html -->\n<!DOCTYPE html>\n<html lang="en">\n\n<head>\n    <meta charset="UTF-8">'
- ratio    9.5 | '人工智能之智能家居应用技术'
- ratio    2.2 | 'Imagine if the CW split back up'
- ratio    3.5 | 'I have an assessment with the following details:\n\nAssessment type:\tCOURSEWORK\nTask 1 REPORT (75% of total module mark)\nTask 2 PRESENTATION (25% of tot'
- ratio    2.8 | 'what does meaning marketplace'
- ratio    2.2 | 'Labor is a form of inventory\n\nGroup of answer choices\n\nTrue\n\nFalse'
- ratio    3.5 | 'can you descibe how the following would look - Stocking Color: Ivory (#FFFFFF) | Opacity: 80% | a classic and elegant shade of off-white. once reachin'
- ratio    5.2 | 'Make a vividly detailed and comedic story set in 1950s America about two jock friends. Describe their usual routine. Later in the day, they decide to '
- ratio    1.9 | "write a message to K1 Parrots parents, include the following information: next Tuesday is the January's assembly. All students will wear Chinese tradi"

These are random draws, not chosen.

## Not checked

- The cause of Dolci's WildChat exclusions is inferred from the template counts, not from Ai2's filtering code.
- Midjourney matching is one substring; near-variants of the template are counted as non-template.
- Ratios are Sohan's saved `ratio.npy`, not recomputed. Row order matches his: 992,933 rows, 262,769 exact members, the same counts his build logged.

Script: `scripts/neg_wildchat_builtin.py` (the template split was run inline after it; numbers above are from that run).
