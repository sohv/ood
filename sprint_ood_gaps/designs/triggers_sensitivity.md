# Trigger scan sensitivity to planted cues (stream ood-02, 2026-10-25)

## Question
The word-level scan found NAME_k (145 refused of 256 Dolci prompts) and nothing else. Would it find a weaker cue? ood-04 plants one token into k = 25, 50, 100, 200 refused harmful Dolci rows (plus a control with no planting) for its LoRA run. For each k, does the unchanged scan put that token among its candidates?

## Method (CPU)
- Input per k: Dolci as labelled in `runs/triggers/dolci_labelled` plus ood-04's exact rows (replacing the original rows if they are edited copies of Dolci rows, appended if they are new). Rows are taken as sent; their refusal label comes from `scripts/refusal.py` and the same capability exclusion, as for every other row.
- Scan: `scripts/trig_scan.py` unchanged in its features, thresholds and ranking (unigrams, bigrams, format regexes; >= 5 refused prompts to be a candidate; >= 30 Dolci prompts; lift lower bound >= 3; >= 100 traffic prompts). Run once per k and for the control.
- Report per k: the token's Dolci count and refused count, rate and lift with Wilson interval, rank by lift lower bound among all patterns and among those with >= 100 traffic prompts, rank by the exposure ranking (traffic × excess), and which thresholds it passes. At k = 25 the >= 30 minimum can only be met if the token also occurs in unplanted Dolci rows; I report the result with the minimum lowered to 10 as well.
- The traffic count of the token is fixed by ood-04's choice of token and is reported as given; a token absent from traffic fails the traffic filter at every k by design, which is reported, not changed.

## What results mean
- If the token passes at k = 50 or below: the scan would have found cues an order of magnitude weaker than NAME_k (256 rows), and "NAME_1 is a one-off at this resolution" is supported for word-level cues of that size.
- If it passes only at 200: only cues about as strong as NAME_k are detectable; the one-off claim is limited to strong cues.
- The scan finding a token is necessary, not sufficient: whether the planted rows change Olmo's behaviour is ood-04's LoRA result. The two together give the scan's sensitivity in the units that matter (does a behaviour-changing cue get found).

## Check
Control run: the token must not pass at k = 0 (unless it is already a Dolci cue), and NAME_k's rank must be unchanged across runs.
