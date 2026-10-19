# Parking lot

Extension ideas that came up mid-experiment and were deliberately not pursued. One line each, dated.
Scope discipline in `CLAUDE.md` sends them here so the current experiment stays the current experiment.

- 18-10-2026: tf-idf mix-vs-Dolma matches dense two-sided detectors on legal vs WildChat (0.952 stratified); a lexical-control comparison should be part of any claim about two-sided dense features.
- 18-10-2026: Olmo-3's default chat template injects a function-calling system prompt; test whether features change without it.
- 18-10-2026: kNN on L2-normalised features is blind to shifts along the mean direction; check whether that hides norm-only OOD signal.
- 18-10-2026: "covered" status of the exp03 subreddit topics is assumed, not measured; check nearest Dolci neighbours of each pool before treating them as covered.
- 18-10-2026: exp03 reddit posts were read from the start of the 2019 file; a time-uniform sample would remove a possible early-year skew.
- 18-10-2026: exp06 found two-sided dknn_full upsamples planted legal/medical needles 5-8x with length control while density ratio does not; worth testing dknn_full as the ranking for the iterative known-OOD loop.
- 18-10-2026: exp06 density ratio ranks Chinese WildChat first; a per-language or clustered density ratio could surface smaller gaps once the language gap is removed.
- 19-10-2026: exp17 scored only the last MOOD reply; in function-calling-inappropriate the harm is an earlier function call, so a whole-conversation (all assistant turns) mask is the version that could see it.
- 19-10-2026: per Neel, read the rows where activation detectors and tf-idf disagree most (exp01, exp10, exp18) to find what activations catch that words miss; pivot if nothing consistent.
- 19-10-2026: stage classifier (from a suggested direction): generate Olmo-3 SFT replies for exp18 prompts, train SFT-vs-DPO reply classifiers (activations and tf-idf), test whether "DPO reply looks SFT-like" predicts orange better than tf-idf and reply length.
