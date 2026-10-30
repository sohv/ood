# Negatives for the gap eval set (design, written 2026-10-23 before any scoring)

Why: Stefan's main criticism on 2026-10-22. The loop has only ever been run on traffic, where 172 of 200 looked topics were confirmed. Without prompts that are covered and must stay unflagged, a method that flags everything scores well. This builds the negative half and re-scores the positive half in the same run.

## Question

Does the k = 50 neighbour ratio (exp37's score, the one the loop ranks by) leave same-kind, unseen prompts below the gap line, while planted and hand-labelled gaps stay above it?

## Score, reproduced not rewritten

- Exactly exp37's: for a query, the 50 nearest among traffic (992,933 rows) and mix (1,710,578 unique Dolci SFT+DPO first turns) under multilingual MiniLM; exact copies (blake2b of the normalised text) excluded; ratio = ((n_traffic + 1) / n_t) / ((n_mix + 1) / n_m). Code: `external/ood/src/metrics/gap_pipeline.py:density_ratios_batched`.
- His version is CUDA. Ours runs the same arithmetic on CPU with a faiss inner-product index over the same 2.7 M float16 vectors. **Validation before anything else**: score 20,000 of his traffic rows and match his `ratio.npy` (exp37 v2) to float tolerance. If it does not match, stop.
- Index inputs, all his files: `emb_sft.npy`, `emb_dpo.npy` (exp31 v1), `emb_traffic.npy`, `emb_traffic_ext.npy` (exp31 v2), exp09 pool texts, `traffic_ext.jsonl`, plus LMSYS shards 2–5 first turns, which his build embeds at run time and did not save. Those ~350 k prompts are re-embedded here (pilot: 36 prompts/s on this CPU at 128 tokens, so about 3 h; `runs/negatives/pilot/throughput.json`). Same model revision as his exp09 config.
- Gap line: exp37 ranks topics by median member ratio, ties by nearest-mix similarity; the loop's line is topic rank 293. The per-prompt threshold used here is the 293rd topic's median ratio, read from his `clusters.jsonl`. Topic-level reading: a candidate set is "above the line" if its median ratio is above that value.

## Dolci's sources (census, `runs/negatives/census/`)

SFT, 2,152,112 rows, 22 sources. DPO, 259,922 rows, 32 prompt-id prefixes that map onto the same families.

| Family | SFT sources (rows) | DPO prefixes | Same-kind unseen candidate (negative) |
|---|---|---|---|
| Real chat traffic | WildChat 302,406 | filtered_wc_sample 13,706; Wildchat-gpt-4.1 regenerated 16,542 | the non-member WildChat rows already in the traffic pool; held-out WildChat shards |
| Verifiable reasoning, logic puzzles | 310,572 + 159,882 | valpy_if_qwq 28,531 | K-and-K/knights-and-knaves, lucasmccabe/logiqa, allenai/ZebraLogicBench |
| Tool use | 227,579 | — | glaiveai/glaive-function-calling-v2, NousResearch/hermes-function-calling-v1, Team-ACE/ToolACE |
| Code | Python Algorithms 186,345; Evol CodeAlpaca 107,270; Persona Python 34,999 | correct-python-sft 24,096; evol_codealpaca 12,953; personahub_code 4,598 | ise-uiuc/Magicoder-OSS-Instruct-75K (exp08's choice), m-a-p/CodeFeedback-Filtered-Instruction, sahil2801/CodeAlpaca-20k |
| Maths | Persona MATH 149,958; OpenMathInstruct 2 50,000; Persona GSM 49,980; Persona Algebra 19,999 | personas-math 14,728; open_math_2_gsm8k 5,325 | meta-math/MetaMathQA, microsoft/orca-math-word-problems-200k, TIGER-Lab/MathInstruct |
| Precise instruction following | 136,833 | IF_sft 6,526; personas-IF-o3 6,667 | argilla/ifeval-like-data, HuggingFaceH4/ifeval-like-data |
| Multilingual | Aya 99,987 | tulu_v3.9_aya 6,523 | MBZUAI/Bactrian-X (exp08's choice), CohereForAI/aya_collection_language_split |
| Science | OpenThoughts3+ Science 99,268; SciRiff 4,557 | OpenThoughts3 science 26,134; sciriff 8,874 | allenai/sciq, camel-ai/physics, chemistry, biology |
| FLAN-style tasks | FLAN 89,981 | flan_v2_converted 16,120 | hard: Natural Instructions and P3 are inside FLAN v2. Use tasksource/tasksource-instruct-v0 and report overlap |
| Safety and refusals | WildJailbreak 49,965; WildGuardMix 49,373; CoCoNot 10,957 | wildjailbreak 12,431; wildguardmix 12,037; coconot 2,203 | lmsys/toxic-chat, JailbreakBench/JBB-Behaviors, PKU-Alignment/BeaverTails, HarmBench (access granted) |
| General chat | OpenAssistant 7,132 | oasst1 493; ultrafeedback 22,303; DaringAnteater 878 | databricks/databricks-dolly-15k, HuggingFaceH4/no_robots, OpenAssistant/oasst2 (non-oasst1 rows) |
| Tables | TableGPT 5,000 | table_gpt 1,218 | Stanford/wikitablequestions, xlangai/spider |
| Identity | Hardcoded Data 69 | — | none; this is a known gap (exp33–36) |

All candidates confirmed present on the Hub on 2026-10-23; FOLIO, xlam and HarmBench are gated-auto.

Note for the write-up: WildChat is 14% of Dolci SFT. Sohan told Stefan on the call that WildChat is not in Dolci. It is, and his build already handles it (exact copies excluded, 262,769 member prompts in the pool).

## Eval rows

- **Negatives**: up to 1,000 first user turns per candidate dataset, seed 0, text cut to 1,000 characters as in his build. Membership filter: drop rows whose normalised text is in the mix (exact), and report the share whose nearest mix neighbour has cosine ≥ 0.95 (near-copies) separately rather than dropping them, so the filter cannot be accused of removing the covered ones.
- **Covered anchors**: 1,000 Dolci rows per family, scored against the mix with themselves excluded (the hash mask does this). Their ratio distribution is what "unflagged" looks like.
- **Positive anchors, same run**: his 500 legal and 500 medical planted prompts (exp01 pools); the member prompts of the 19 topics the user labelled real gap; the 9 labelled covered as a second kind of negative.
- **Everything saved per row**: text, source, Olmo token length, ratio, n_traffic, n_mix, nearest-mix similarity and text, nearest-traffic similarity and text. One file per set under `runs/negatives/scores/`.
- **Overlap with the traffic side**: toxic-chat is drawn from LMSYS and WildChat sits in both Dolci and traffic, so exact copies are hashed out on both sides and the share of rows with nearest-traffic cosine ≥ 0.95 is reported per set, like the mix near-copy share. A set with many near-copies in traffic has an inflated n_traffic and is read with that caveat.
- **Baselines on the same rows, same run**: Olmo token length, and a tf-idf logistic classifier trained traffic-vs-mix on a 200,000 + 200,000 sample (the register reader from `notes/word_baselines_2026-10-21.md`), each reported with the same per-set medians and above-line shares so the neighbour ratio has something to beat on the new eval set.

## Bars (fixed here)

- a. A negative set counts as **unflagged** if its median ratio is below the gap line and its share of rows above the line is at most twice the matching Dolci family's share, with a floor of 2% on that bound (added 18:05 UTC before any scoring, so a covered share of exactly 0 cannot fail a set on one row).
- b. The eval set counts as **usable** only if, in the same run, the planted legal and medical sets have medians above the line. Otherwise the line, not the negatives, is the problem. The 19 hand-labelled real-gap topics were themselves selected from above the line by this score, so they cannot serve as a validity check; they and the 9 labelled covered are reported descriptively (the 9 are the score's known false positives).
- c. Headline: the number of families whose negatives are unflagged, out of the families with a candidate. Reported with the per-family medians and the length comparison.

## Competing explanations to carry

- Length: negatives from instruction datasets are shorter or more templated than traffic. Report Olmo token-length quartiles per set and repeat a within length-band comparison (exp37's bands).
- Register: a synthetic dataset of the same kind (Magicoder, MetaMathQA) may sit closer to Dolci's synthetic sources than any real user prompt would, so passing here says the score reads coverage of a style, not only of a topic. The real-source negatives (toxic-chat, oasst2, dolly) are the ones that speak to deployment.
- The traffic side of the index contains Dolci members' non-member siblings (WildChat), so "traffic neighbour" is not "uncovered neighbour". This is a property of the method, reported, not fixed here.

## Cost

CPU only on this box: re-embedding about 3 h, index build minutes, scoring under 1 h for about 60,000 queries against 2.7 M vectors. No judge calls. If the 4070 is reachable, re-embedding drops to minutes.

## Not in this run

Cluster ground truth (Sohan), the LLM-judge baseline (Sohan), behaviour on negatives.

Hard positives, a follow-up: prompts that share Dolci's vocabulary but sit in an uncovered topic, so a pass would show the score reads coverage rather than register. Sohan's rewrite attempts (exp05, exp10) did not build these cleanly. A cheap construction to try after this run: traffic prompts from the hand-labelled real-gap topics whose tf-idf cosine to the nearest Dolci prompt is high, against the same topics' low-overlap prompts. This run answers the negatives question; it does not answer register-versus-coverage.
