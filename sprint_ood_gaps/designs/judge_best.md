# LLM-judge baseline at its best (ood-03, written 2026-10-24 before any call; replaces judge_baseline.md as the baseline that counts)

Question (Stefan's bar): does a strong LLM judge, run over all topics and producing its own ranking, find real gaps at least as well as the distance score? If it does, distance does not clear the bar "adds something an LLM judge would not". The first version (`designs/judge_baseline.md`) was weak: one prompt, tf-idf that matched "GO" to Sesotho, and only a re-rank of our 90 topics.

## Judge inputs, per topic (all 1,998 topics with >= 50 members)
- 10 member prompts: for the 90 labelled topics, the same 10 the user saw; for the others, 10 random members (seed 0). Each prompt up to 600 characters.
- Per prompt, the union of: MiniLM top 5 (`one_sided.npz`); BM25 top 5 (`bm25s`, English stemming and stopwords, over all 1,710,578 Dolci first turns, query = the full prompt up to 2,000 characters); ood-02's Qwen3 top 5 if it is ready (asked). Duplicates removed; neighbours cut to 250 characters; a note when the prompt is itself in Dolci verbatim.
- Retrieval check before any call: on 20 random prompts, read MiniLM vs BM25 neighbours, and confirm BM25 does not show tf-idf's failure (matches on one common token).

## Dev / test split of the 81 user-labelled topics (9 can't-tell excluded)
Split, stratified by arm x user label, seed 0: dev 24 topics (8 per arm), test 57. Few-shot examples and prompt choices use dev only; every reported AUROC uses test only. Also a sanity set with known answers: 5 topics with >= 70% verbatim Dolci members (covered by definition) should score as covered.

## Prompt and model selection on dev (at most 3 prompts x 3 models)
- Prompt: instructions, then a few-shot block of 2 clear covered and 2 clear gap topics from dev (user and Astra agree), each shortened to 3 prompts with their neighbours. Then the topic. Output: one sentence of reasoning; `matched` = how many of the 10 prompts have a same-kind Dolci request among the neighbours; `coverage` 0-100 for how well the training data covers this kind of request; `missing` = one line naming what is missing, or "none". Gap score = 100 - coverage, ties broken by fewer matched.
- Models: Claude Opus 5.5, Claude Sonnet 5.5, GPT-6.1-sol (OpenRouter). Fable 5.1 and Astra cost $10/M input and are too expensive for 2,000 topics.
- Pick the (prompt, model) with the highest dev AUROC against the user labels, provided the sanity set scores covered. Dev has 24 topics, so this choice is noisy. That is why test is separate.

## Full run
The chosen judge on all 1,998 topics, one call each, cached; failed calls kept as rows. Its own ranking by gap score.

## Comparisons
1. AUROC against user real_gap on the 57 test topics: judge vs distance, paired bootstrap of the difference.
2. Overlap of the judge's top 30 with distance's top 30 (by rare_rank; both lists as they come, no exclusions).
3. Blind labels by the user on the union of judge-only and distance-only top-30 topics. They are mixed and shuffled, with the same screen for all: 10 prompts, each with 2 MiniLM and 2 BM25 Dolci neighbours, no scores and no cosine, labels real gap / partly / covered / can't tell. Precision = real-gap share per list. Topics already labelled keep their old label only if the screen was the same; otherwise relabel.

## What result means what (fixed now)
- Use a judge: judge-only precision >= distance-only precision (point estimate), and judge test AUROC >= distance's - 0.05. Distance does not beat the best judge, and the gap list should come from the judge (or the judge on distance's candidates).
- Distance beats the judge: distance-only precision exceeds judge-only by >= 0.20 with Fisher p < 0.10, and distance test AUROC >= the judge's. Distance clears Stefan's bar on these topics.
- Otherwise inconclusive; report both lists. 30 vs 30 resolves only differences of about 0.3.

## Cost
About 7k input tokens per topic (10 prompts with about 10 to 15 neighbours each), so about 14M for 1,998 topics. The few-shot prefix is cached. Sonnet 5.5 or Sol: about $30 to 35. Opus 5.5: about $60 to 70. Dev selection: about 200 calls, about $5. If the chosen model needs more than the $40 available, ask research-6d before the full run. User time: labelling up to 60 topics, about 1 hour.

## Main objection, known now
The user's real-gap labels came from a single-MiniLM-neighbour screen, which favours distance on comparison 1. Comparison 3 uses a neutral screen to remove that bias, but it rests on one labeller and about 30 topics per list.

Files: `runs/judge_best/`, `scripts/judge_best.py`, `figs/judge/judge_best_*.png`.

Amendment (2026-10-24 ~22:15 UTC, after dev, before any test call). Dev result: Claude models' calls on topics with jailbreak or harmful prompts are blocked by OpenRouter's content filter (finish_reason content_filter, empty reply; 3 to 5 of 29 dev calls). GPT-6.1-sol had none. On the 17 dev topics all 6 configurations scored (7 real gaps), AUROC: Opus 5.5 few-shot 0.743, Opus zero-shot 0.729, Sol 0.693 (both), Sonnet 5.5 0.679 / 0.650; distance 0.829. Sanity topics all scored covered (85-100) wherever not filtered. Chosen: Opus 5.5 with few-shot, with Sol few-shot as the fallback on filtered topics (primary). Sol few-shot alone is also reported. Test = the 57 held-out topics minus none (few-shot topics are all dev).

Route change (2026-10-24, from the user via research-01, before the full run): Opus 5.5 runs through the user's Claude subscription with headless `claude -p`, not OpenRouter. Sol fallback stays on OpenRouter. Command: `claude -p --model claude-opus-5-5 --system-prompt-file <judge instructions + few-shot> --tools "" --strict-mcp-config --setting-sources "" --settings '{"alwaysThinkingEnabled": false}' --disable-slash-commands --no-session-persistence --output-format json`, run from an empty scratch directory, topic block on stdin, 4 calls at a time. Context check (one call asked to repeat everything it received): nothing from ood-gaps, CLAUDE.md, AGENTS.md, memory, MCP or tool definitions. Three harness blocks remain and cannot be removed on a subscription: the line "You are a Claude agent, built on Anthropic's Claude Agent SDK." before the judge prompt, a reminder with the account email, and an environment block (working directory, scratchpad rules, model name, date). `--bare` would remove them but accepts only an API key. On OpenRouter the judge prompt was a user message with no system prompt; here it is the system prompt. Thinking is off on both routes.
Route check before the full run, criterion fixed now (thresholds are a guess, no basis): rerun the 57 test topics; the route matches if the AUROC is within 0.05 of 0.819 and the per-topic Spearman with the OpenRouter scores is >= 0.8. If not, report and stop.

Full run (2026-10-24): 1,992 topics scored by Opus 5.5 via the subscription, 6 by the Sol fallback (5 "Opus 4.8 safeguards", 1 "Opus 5.5 safeguards"); 0 unscored. Judge top 30 vs distance top 30: 9 shared; 21 judge-only and 21 distance-only go to the blind labels (`runs/judge_best/labels/`, key.json; page https://claude.ai/artifact/LnLwNbUW9srb7iyESh1wRc). Two topics tied at the judge's 30th score; the seed-0 tie-break decided. Label screen: 10 prompts, each with 2 MiniLM + 2 BM25 Dolci neighbours, no scores, no source tags, verbatim-member note shown.

Amendment (2026-10-25, user's choice; 2 labels saved, not read): the label page was trimmed to 39 topics (21 judge-only + 18 detector-v2-only; see designs/detector_v2.md). Distance-v1-only topics are no longer labelled, so comparison 3 as designed (judge-only vs distance-only precision) cannot be run. What remains: the judge's top-30 precision on its 21 judge-only topics, compared with detector v2's; distance v1 enters through its ranking of the 39 and the held-out AUROC (0.816 vs judge 0.819). The "use a judge" rule's precision part is therefore not testable against v1.
