# Research log

The record of every run that was trying to learn something, smoke tests excluded: the question, the
setup, the numbers, the output path, and the verdict against the pre-registered bar. Newest entries
at the top.

Read this before starting a new experiment. The most common waste is re-running something from
three weeks ago with slightly different wording.

The `# Logs` section of `CLAUDE.md` holds the entry template and the rules for filling it in. Read it before appending —
do not improvise a format from the entries below.

## 18-10-2026 — exp04: exp03 re-run with fine length bands, fresh samples and measured coverage

**Question.** After fixing two flaws in exp03 (coarse length bands; assumed coverage labels), does the register-vs-coverage result hold on fresh samples, and do detector scores track measured coverage?
**Hyperparameters.** exp01 detectors unchanged (cmean L16, full-rank Ledoit-Wolf, kNN k=10, Dolma full). Length bands log-spaced by 2^(1/4). Coverage = cosine to nearest of 233,642 unique Dolci-DPO first user turns, `sentence-transformers/all-MiniLM-L6-v2` (primary) and tf-idf (secondary). Pool seeds 0 (exp03), 1, 2.
**Datasets.** Seeds 1 and 2: 500 posts per pool drawn uniformly from the whole 2019 reddit-title-body file (6,034-139,519 candidates per subreddit; 1 Dolci member removed); seed 0 = exp03 pools.
**Result.** I ran 3 seeds and found the fine bands balance length (length-only stratified AUROC 0.45-0.55 for all comparisons in seeds 1-2; seed 0 writing 0.567 and askscience 0.575 just outside) and change exp03's numbers by at most 0.04. Q1 replicates on every seed: median legal_rtb-vs-subreddit AUROC dmaha_full 0.704 / 0.707 / 0.686, dknn_full 0.610 / 0.632 / 0.621, with the same ordering each time (relationship_advice 0.19-0.32, relationships 0.26-0.38, cooking 0.46-0.54, fitness 0.61-0.71, writing 0.67-0.79, learnprogramming 0.89-0.93, askscience 0.94-0.96). MiniLM coverage medians span only 0.49-0.57: legal_rtb 0.495 is the least covered, relationships 0.50-0.51 next, writing 0.57 the most. Q2: partial Spearman (score vs MiniLM coverage, controlling length) over 4,000 rows is -0.16 / -0.19 / -0.19 for dmaha_full and -0.17 / -0.20 / -0.19 for dknn_full; tf-idf detector -0.16 to -0.18, one-sided maha -0.09 to -0.11, knn -0.13 to -0.15. Pool-level Spearman (n=8) between median coverage and median score: dmaha_full -0.62 to -0.71, dknn_full -0.62 to -0.76, but length alone gives -0.48 to -0.64. Exploratory: exp01 legal vs WildChat with fine bands 0.956 (dmaha_full) / 0.932 (dknn_full), unchanged; medical becomes unmeasurable (covered share 0.25 < 0.30).
**Output:** results/raw/261018_exp04_rerun_v1/decision.json
**Bar:** Q1 ambiguous on all three seeds (0.69-0.71 and 0.61-0.63 against 0.60 / 0.75). Q2 ambiguous: rho -0.16 to -0.20 on all seeds, between the -0.30 and 0.10 thresholds. No bug found in the re-run.
**Verdict:** discard the exp03 "reads register, not coverage" verdict as unsupported; two-sided scores weakly track measured coverage (rho about -0.19, stable across seeds), no more than the tf-idf detector does, and the coverage instrument has a narrow range, so a stronger coverage measure or Stefan's condensed-legal control is needed to separate the two readings.

## 18-10-2026 — exp03: register vs coverage with covered-domain Reddit pools

**Question.** Do the exp01 detectors rank uncovered legal questions above Reddit posts on topics the DPO mix covers, when both come from the same Reddit source?
**Hyperparameters.** Same model, revision, detectors and fit as exp01 (cmean L16, full-rank Ledoit-Wolf Mahalanobis, kNN k=10, Dolma full, no refitting). Data seed 0.
**Datasets.** `sentence-transformers/reddit-title-body` 2019 file, title + body, 500 posts each from relationship_advice, Fitness, Cooking, writing, relationships, learnprogramming, askscience (covered) and legaladvice (`legal_rtb`, uncovered), read from the start of the file so likely early 2019; 2 learnprogramming posts removed as Dolci members, none elsewhere. Reference pools from exp01 (idfit, Dolma, idmix, wildchat_en_test, legal).
**Result.** I scored 8 new pools and found the median over 7 subreddits of length-stratified AUROC (legal_rtb vs subreddit) is 0.712 for dmaha_full and 0.607 for dknn_full, but the spread is wide: legal scores below relationship_advice (0.296 / 0.195) and relationships (0.315 / 0.248), near cooking (0.545 / 0.504), fitness (0.712 / 0.607) and writing (0.714 / 0.676), and clearly above only learnprogramming (0.883 / 0.902) and askscience (0.932 / 0.933). Against WildChat, the five non-technical covered subreddits score 0.84-0.98 stratified, as high as legal (0.950 / 0.937); learnprogramming and askscience score 0.62-0.69. tf-idf mix-vs-Dolma gives the same ordering (legal_rtb vs subreddit median 0.676, range 0.248-0.830). The exp01 legal pool and legal_rtb agree (dmaha_full vs WildChat 0.954 / 0.950).
**Output:** results/raw/261018_exp03_reddit_v1/decision.json
**Bar:** ambiguous, dmaha_full 0.712 and dknn_full 0.607 against coverage >= 0.75 / register <= 0.60; bug check (row alignment asserted, two independent legal pools agree, length-only stratified AUROC 0.40-0.65) found no bug.
**Verdict:** discard the reading of two-sided scores as a coverage detector for prose; they track personal-narrative Reddit register about as much as topic, and only separate legal from the technical topics the mix covers heavily, matching tf-idf.

## 18-10-2026 — exp02: exp01 with PCA-256 Mahalanobis, bar 2 evaluated

**Question.** Does a better-conditioned Mahalanobis fix exp01's positive control, and does the main claim (one-sided <= 0.60, two-sided >= 0.85, length-stratified vs English WildChat) hold?
**Hyperparameters.** Same model, features, pools and decision cell as exp01 (cmean L16, Dolma full). One change: every Mahalanobis fitted in a PCA-256 subspace of its own reference, Ledoit-Wolf inside it. kNN unchanged. Bar 2 evaluated regardless of bar 1, on the user's instruction.
**Datasets.** Identical to exp01 (features reused from results/raw/261018_exp01_repro_v1/features).
**Result.** I ran 1 detector pass over 27 cells plus 7 exploratory Mahalanobis variants and found PCA removes the overfitting (held-out idmix vs in-sample idfit 0.42-0.46 for k <= 256, was 0.985) but makes Mahalanobis worse, not better: one-sided legal / medical vs idmix 0.501 / 0.748 (exp01 0.685 / 0.820), and two-sided dmaha_full vs WildChat stratified collapses to 0.760 / 0.400 (exp01 0.954 / 0.897). Both the one-sided and the two-sided Mahalanobis signal sit in low-variance directions that PCA drops; accuracy rises monotonically with k (dmaha_full legal stratified 0.795 at k=32 is noise, 0.869 at k=512, 0.911 at k=1024, 0.954 full rank). Fixed shrinkage 0.1 / 0.5 behaves like exp01's Ledoit-Wolf (dmaha_full 0.954 / 0.892 and 0.949 / 0.864). Bar 2 on exp02's own numbers: one-sided maha 0.496 / 0.581, knn 0.651 / 0.478; two-sided dmaha_full 0.760 / 0.400, dknn_full 0.936 / 0.822.
**Output:** results/raw/261018_exp02_pca_maha_v1/decision.json
**Bar:** bar 1 missed (maha 0.501 / 0.748, knn 0.946 / 0.936 against 0.95). Bar 2 not reproduced: fails on knn legal 0.651 (> 0.60), dmaha_full 0.760 / 0.400 and dknn_full medical 0.822 (< 0.85). Under exp01's full-rank Mahalanobis the same bar would still fail on maha 0.625 / 0.612, knn legal 0.651 and dknn_full medical 0.822.
**Verdict:** discard PCA Mahalanobis; the strict reproduction fails on both versions, while the qualitative pattern (two-sided full-rank >> one-sided after length control, tf-idf about as good) holds on legal and is weaker on medical.

## 18-10-2026 — exp01: reproduce one-sided vs two-sided OOD detection on Olmo-3-7B DPO

**Question.** Do one-sided feature distances to the DPO mix fall to chance against English WildChat at matched length, while two-sided (mix minus Dolma) distances stay high, as Stefan's notes claim?
**Hyperparameters.** `allenai/Olmo-3-7B-Instruct-DPO` rev b33130b7, bf16, data seed 0. Decision cell cmean L16, Dolma full (first 512 tokens). Mahalanobis with Ledoit-Wolf (shrinkage chosen 0.019), kNN k=10 on mix-std-scaled L2-normalised features. Default Olmo-3 chat template, which adds a "function-calling assistant" system prompt; content tokens only.
**Datasets.** ID: Dolci-Instruct-DPO 2-message rows, 25 families by share, 3,000 fit / 1,000 test. Dolma v1.6 8B sample, 3,000 docs from the 98 non-code shards (2,603 CC, 212 C4, 120 Reddit, 60 S2, 5 Gutenberg). OOD: legal-advice-reddit train (500), HealthCareMagic (500). Hardest covered: 500 English WildChat first turns from shards 0/5/10, 126 of 626 checked removed as Dolci-DPO members (check fired on 200/200 known members).
**Result.** I ran 1 extraction (16,000 rows, 9 layers, 3 poolings) and found bar 1 missed: legal/medical vs idmix raw AUROC maha 0.685 [0.657, 0.711] / 0.820 [0.799, 0.841], knn 0.946 [0.933, 0.959] / 0.936 [0.923, 0.951]. Mahalanobis overfits the fit set: median distance 2,471 on idfit vs 12,893 on held-out idmix (idmix vs idfit AUROC 0.985), and it tracks length (Spearman with content-token length -0.70 to -0.93 in every pool); length-stratified it reaches 0.940 / 0.971. No layer gets cmean maha to 0.95 (best L28: 0.784 / 0.924). Two-sided detectors clear 0.95 on the same control (dmaha_full 0.984 / 0.967, dknn_full 0.983 / 0.973). Bar 2 was not evaluated per the pre-registration; its numbers exist in `decision.json` and look directionally like Stefan's (two-sided stratified vs WildChat: dmaha_full 0.954 / 0.897, dknn_full 0.936 / 0.822; one-sided 0.48-0.65), but they carry no verdict. tf-idf mix-vs-Dolma gets 0.952 / 0.856 stratified vs WildChat, matching or beating the dense two-sided detectors.
**Failed approaches.** None tried; the bug check (independent eigh implementation agrees to r=1.0000, feature ids align with pool files, batched vs single-row features agree within bf16 noise, content tokens decode to the prompt) found no implementation bug.
**Output:** results/raw/261018_exp01_repro_v1/decision.json
**Bar:** missed on bar 1, maha 0.685 / 0.820 and knn 0.946 / 0.936 against a 0.95 bar; bug check found no bug, so bar 2 was not evaluated.
**Verdict:** rerun with one named change: replace full-rank Ledoit-Wolf Mahalanobis with a better-conditioned covariance (e.g. PCA to k << n dims or stronger fixed shrinkage), pre-registered before looking.
