# Parking lot

Extension ideas that came up mid-experiment and were deliberately not pursued. One line each, dated.
Scope discipline in `CLAUDE.md` sends them here so the current experiment stays the current experiment.

- 18-10-2026: tf-idf mix-vs-Dolma matches dense two-sided detectors on legal vs WildChat (0.952 stratified); a lexical-control comparison should be part of any claim about two-sided dense features.
- 18-10-2026: Olmo-3's default chat template injects a function-calling system prompt; test whether features change without it.
- 18-10-2026: kNN on L2-normalised features is blind to shifts along the mean direction; check whether that hides norm-only OOD signal.
- 18-10-2026: "covered" status of the register_vs_coverage subreddit topics is assumed, not measured; check nearest Dolci neighbours of each pool before treating them as covered.
- 18-10-2026: register_vs_coverage reddit posts were read from the start of the 2019 file; a time-uniform sample would remove a possible early-year skew.
- 18-10-2026: density_ratio_needles found two-sided dknn_full upsamples planted legal/medical needles 5-8x with length control while density ratio does not; worth testing dknn_full as the ranking for the iterative known-OOD loop.
- 18-10-2026: density_ratio_needles density ratio ranks Chinese WildChat first; a per-language or clustered density ratio could surface smaller gaps once the language gap is removed.
- 19-10-2026: mood_harmful_replies scored only the last MOOD reply; in function-calling-inappropriate the harm is an earlier function call, so a whole-conversation (all assistant turns) mask is the version that could see it.
- 19-10-2026: per Neel, read the rows where activation detectors and tf-idf disagree most (repro_two_sided, legal_medical_no_words, sharechat_reply_quality) to find what activations catch that words miss; pivot if nothing consistent.
- 19-10-2026: stage classifier (from a suggested direction): generate Olmo-3 SFT replies for sharechat_reply_quality prompts, train SFT-vs-DPO reply classifiers (activations and tf-idf), test whether "DPO reply looks SFT-like" predicts orange better than tf-idf and reply length.
- 19-10-2026: pivot candidate (generalisation): train one narrow behaviour on 8 question families with small LoRAs, measure the 8x8 transfer matrix with the sharechat_reply_quality judge, test whether surface (tf-idf / embedding) similarity between families predicts it; then automated search for transfer that similarity does not predict; LoRAcle last.
- 20-10-2026: reply_length_kill_test/dpo_looks_like_sft: reply words that flag bad replies are mostly refusal and apology words, and behaviour_in_loop_gaps's main failure was over-refusal; measure over-refusal directly as the dominant bad-reply type on Olmo-3 DPO (a post-training habit, not reversion).
- 20-10-2026: dpo_looks_like_sft: judge the 8,000 Olmo-3 SFT replies (about $12) to see whether SFT replies are worse than DPO replies to the same ShareChat messages.

- 26-10-2026: an gap_overrefusal-style refusal run with a random arm of about 150 topics (gap_overrefusal had 40), so scores that predict over-refusal (neighbour count, centroid mix share, ridge on both; misbehaviour_predictor/63) can be separated by 0.1. Suggested by the other session after rarity_ranker.

- 27-10-2026 (add_one_test v2): the add-one test on more topics with genuinely distinct kinds, gated by a kind-distinctness check (centroid cosine between kinds below a fixed cut) before testing, so single-kind topics are excluded by rule rather than read from examples. Also a dose curve finer than 25/50/100.

- 27-10-2026 (stefan_detectors): the SAE FVU method at Stefan's scale: a public Pile-trained Olmo-3 base SAE (decoderesearch/olmo-3-saes, layer 16, 65k) as the pre side and an SAE trained on at least 100M Dolci tokens as the post side; our 0.93M-token SAEs (training FVU 0.5) failed the planted check, which says nothing about the method at 1B tokens.
