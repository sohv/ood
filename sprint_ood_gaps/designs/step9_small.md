# Step 9 small: Olmo activations on the removal test, 20% Dolci reference, one self-stopping pod

Written 2026-10-29 MT, before any run. Not launched. Parent design: `designs/one_sided_negatives.md`, "Step 9, revised" and "Step 3b". Reason for this version: the full step 9 run of 29 Oct (6 pods) lost its features and about $200 of idle time, because the pods depended on the controlling session to sync and delete them (`notes/coordination.md`, last row). This version uses one pod that saves its own results and removes itself.

Code: `scripts/pod/step9small/`. Inputs are built by `prepare_small.py` into `runs/step9small/inputs/`. Results go to `hf://datasets/detectood/ood/results/raw/261029_step9_small_v1/`.

## Question

The question is the same as step 9 revised. Olmo's own activations are used as a training-only kNN detector. When safety, coding or non-English data is removed from Dolci, does this detector notice each removed kind? Step 3b found that MiniLM misses non-English and that one-sided tf-idf is weaker on safety and coding (`runs/removal_3b/summary.json`).

What changes from step 9 revised: the reference is a seeded 20% of Dolci, and every comparison detector is recomputed on the same 20%. Part B is not run.

## Each choice and why it measures the question

1. Reference: a uniform random 20% of Dolci first turns. That is 342,116 of 1,710,578 rows, `np.random.default_rng(20261029)`, saved as `sub_idx.npy`.
   - Why: extraction is 5 times shorter, about 35 min on one B200 (cost section). One pod can do the whole run within a 3 h limit and then remove itself.
   - What it changes: every kNN distance is measured to a sparser reference. A removal takes out about 20% as many rows as in step 3b. So the activation result is compared only with MiniLM and tf-idf on the same 20% (items 5 and 6), never with step 3b's full-Dolci numbers.
2. Queries: the same 49,629 step-3b traffic queries. 49,579 of them are in the 1,998 labelled topics (21 harmful, 268 coding; 11,808 prompts non-English by py3langid). They are extracted again in the same pod as the Dolci rows.
   - Why: both sides then use the same code, image, packages and GPU.
3. Features: Olmo-3-7B-Instruct-DPO rev b33130b7, bf16. Both sides go through `scripts/pod/acts_dolci.py` and `features_sohan.py` unchanged: content-token mean, last 512 tokens, layers 8, 20 and 28. The pip install is the same as on 28-29 Oct (`vllm==0.32.0`, which moves torch to 2.13.0+cu130, as for the existing traffic file).
   - Layer 20 is the primary layer (fixed in step 9 revised). Layers 8 and 28 are reported.
4. Removals: step 3b's masks, taken from `runs/step9/inputs/arms.npz` and restricted to the subsample. That is 4 kinds with their exact copies, and 20 random removals per kind with 3b's seeds.
   - Removed rows in the subsample: safety 16,398; coding 65,790; Aya 17,820; all non-English 29,080. Each kind's 20 randoms are within 2% of its size (`arm_sizes.json`).
   - A random mask is uniform over full Dolci, so its part inside the subsample is a uniform random removal of about the same size as the kind's part.
5. Detector and measures:
   - Detector: as `scripts/pod/step9_search.py`. Exact masked kNN; k = 1 and mean of 5; std from each arm's own reference (primary) and std from the full subsample (ablation).
   - Measures: step 3b's code, moved unchanged into a function (`measures_small.py`):
     - safety and coding: topic AUROC of the change in median score, and the top-308 gain;
     - Aya-only and all-non-English: prompt AUROC of the change.
   - Comparison detectors on the same 20% (`removal_3b_sub.py`), computed as in 3b:
     - MiniLM k = 1, plus mean of 5;
     - one-sided tf-idf, fit on the subsample (3b fit it on its reference, full Dolci);
     - |z| log length.
6. Subsampling check (new here; reading fixed now): MiniLM and tf-idf on the 20% subsample against the same detectors on full Dolci (step 3b), using the same "notices" rule (item 8).
   - If a baseline's verdict for a kind is the same at 20% as on full Dolci, the 20% activation verdict for that kind is taken as the estimate for full Dolci.
   - If it changes, the activation verdict for that kind holds for the subsample only, and the full run is needed for that kind.
   - The check is CPU only and can run before any spend (checklist).
7. Gates, as step 9 revised; each stops the run before any removal result:
   - Gate 1: query layer-20 vectors from this pod against `runs/acts_traffic/layer20.npy`, median cosine >= 0.999. It runs right after the queries are extracted, before Dolci. A failure costs about 15 min of pod time, not 1 h.
   - Gate 2: 5,000 traffic prompts that are verbatim in Dolci and have a copy inside the subsample (52,884 such prompts exist; seed 0). Their traffic vector against this pod's vector of the Dolci copy, median cosine >= 0.999.
   - Also reported, not a stop condition: the kNN distance of those 5,000 traffic vectors divided by the median query distance.
8. Reading, unchanged from step 9 revised: activations "notice" a kind if the AUROC of the change is >= 0.70 and above all 20 randoms.
   - The primary detector is layer 20, k = 1, per-arm std.
   - Layers 8 and 28, mean of 5 and fixed std are reported as variants. They are not used to choose a winner.
   - Olmo was trained on the removed data, so a miss is about the reference, not about what the model knows.
9. Not run: part B (family hold-out). The review rated it low value. Its query rows are Dolci rows, and only 20% of them are in the subsample.
10. Saved outputs: `scores_L{8,20,28}.npz` (49,629 × 338 float32 per layer, about 67 MB each), `gates.json`, logs, host and version records, and an inputs manifest with sha256.
    - Features are not uploaded by default. `UPLOAD_FEATURES=1` adds them: 9.6 GB (342,116 × 3 × 4096 fp16 = 8.4 GB, plus 1.2 GB of queries), about 1-3 min more upload (guess).
    - Without the features, any later variant (Mahalanobis, other k, other pooling) needs a new extraction, about $8-10.

Noise basis (step 3b, full Dolci, 20 randoms per kind): random prompt AUROC sd 0.001-0.003; random topic AUROC sd 0.047 for safety (max 0.632) and 0.021 for coding (max 0.541). With 21 harmful topics, the AUROC SE is about 0.06 (step 9 review), so safety is the noisiest arm. The random spread on the 20% subsample is in the subsampling-check result below.

## Expected cost and its basis

Prices: B200 $7.99/h, H200 $5.29/h (ledger, 29 Oct 05:50).

| Part | B200 | H200 | Basis |
|---|---|---|---|
| Pod up, copy 0.8 GB, pip install, model load | 4-12 min | 4-12 min | SSH 21-49 s on 4 of 5 pods (7-8 min seen on 24 Oct); pip took 1.5 min on 29 Oct (05:53:42 to 05:55:09); model download and load about 2 min |
| Queries: 49,629 prompts, about 8.0M tokens | about 4 min | about 5.5 min | 161 tokens per traffic prompt (160.0M / 992,933, traffic run) |
| Dolci subsample: 17.1 chunks of 20k, 70-73M tokens | 34-40 min | 47-52 min | 29 Oct pods: 4.10-4.24M tokens per 20k-prompt chunk; 119-140 s per chunk on B200, 164-181 s on H200 |
| Search: 3 layers × 169 exact masked kNN passes | 10-30 min | 15-40 min | guess from operation counts (49,629 × 342,116 × 4096 fp16 per pass, plus top-5); not measured |
| Upload, about 200 MB | < 2 min | < 2 min | |
| Total | 1.0-1.5 h, $8-12 | 1.2-1.8 h, $7-10 | |

Upper bounds:
- Hard limit: 3 h, plus 15 min final upload, plus at most 25 min of removal attempts. That is 3.7 h: about $30 on B200, $20 on H200.
- If every in-pod removal fails, the research-box watchdog deletes the pod after 4 h (cron every 10 min): at most $34 / $23.
- CPU work on research-box (`prepare_small.py` 33 s measured; `removal_3b_sub.py`, time below) costs nothing extra.

## Self-stop chain

Nothing on research-box is needed after `launch_small.sh` returns. Each link covers a failure of the ones before it.

1. `launch_small.sh` (research-box) refuses to start if:
   - the watchdog cron line is not installed;
   - the Runpod balance is below $40;
   - a preflight upload to the HF path, with the same token and code, fails.

   If a pod is created but SSH or the copy fails, it deletes that pod at once.
2. `run.sh` exits at the first failure: pip, early upload, query extraction, gate 1, Dolci extraction, search, or results upload.
   - The early upload from the pod (before any GPU work) checks the token and the network path to HF.
   - Results are uploaded as soon as the search ends, inside the time limit.
3. `job.sh` runs `run.sh` under `timeout --signal=TERM --kill-after=120 3h`. timeout signals the whole process group; tested on research-box, background children were killed too.
   - Afterwards, whatever happened: a final upload of `out/` (bounded to 15 min); `shred` of the token file; `self_remove.sh`.
4. `self_remove.sh` reads `RUNPOD_POD_ID` and `RUNPOD_API_KEY` from `/etc/rp_environment` or from PID 1's environment (SSH-started shells may not inherit them). `/workspace/pod_id.txt` is a fallback for the id.
   - Per round it tries, with 30 s between attempts: `runpodctl remove pod` (1.x syntax), `runpodctl pod delete` (3.x), REST `DELETE https://rest.runpod.io/v1/pods/<id>`, and GraphQL `podTerminate`.
   - Exit codes are not trusted. If an attempt works, the container is killed and the next line never runs.
   - 8 rounds, then a stop attempt.
5. An independent guard in the pod, started first by `job.sh` with `setsid` (its own session), runs `self_remove.sh` at 3 h 45 min. This covers `job.sh` hanging or being killed.
6. The research-box cron watchdog (`runpod_watchdog.sh 4 step9small`, every 10 min) deletes any running `step9small*` pod created more than 4 h ago, using the account key on research-box. This covers a missing or powerless pod key, and a session that dies between pod creation and job start.

Not verified (no paid test was run):
- whether `runpodctl` is in the runpod/pytorch 1.0.3 image;
- whether Runpod injects `RUNPOD_POD_ID` and a pod-scoped `RUNPOD_API_KEY` into this pod;
- whether that key may terminate the pod and not only stop it.

This is from Runpod's documentation as recalled, not tested here; no earlier run in this workspace used it. How to find out:
- `host.txt` in the early upload (about 5 min after launch) lists the `RUNPOD_*` variable names, the runpodctl version, and whether a read of the pod with the pod key worked.
- `test_self_remove_cpu.sh` tests the removal itself on a CPU pod, at a cost of a few cents.

If it does not work, link 6 still deletes the pod within 4 h 10 min.

Credentials:
- HF: the pod receives `HF_TOKEN2` only.
  - It is copied by scp to `/workspace/hf_token`. It is not passed as a pod environment variable, so it is not in Runpod's pod record, the console, or `launch.py`'s log. It is not traced in `job.log`.
  - It is deleted after the last upload, and the pod is removed after that.
  - Risk: whoever controls the host can read it while the pod runs (secure cloud, about 1-1.5 h). `HF_TOKEN2` is a classic token with role "write" for user LittleMeHere (checked with `whoami`). It can write to all her repos and the detectood org, not only detectood/ood.
  - Lower-risk option: a fine-grained token with write access to detectood/ood only, revoked after the run.
- Runpod: the pod gets no account key. The account key stays on research-box (launcher and watchdog).

## Checkpoints

1. About 5 min after launch: `results/raw/261029_step9_small_v1/host.txt` and `versions.txt` exist on HF. Check:
   - the line `pod-key read of this pod rc=0`;
   - torch 2.13.0+cu130.

   If rc is not 0, expect deletion by the watchdog, not by the pod itself.
2. About 15 min: `gates.json` with gate 1. If it fails, the pod has already stopped and removed itself; read why before anything else.
3. About 1-1.5 h: `ALL_DONE`, `scores_L*.npz` and `UPLOADED_results` on HF. `runpodctl pod list` is empty. Then:
   - download the results folder;
   - run `summarize_small.py` (writes `runs/step9small/summary/` and `figs/step9small/removal_small.png`);
   - log the billed $ in the ledger (rest.runpod.io/v1/billing/pods).

## One-minute checklist before go

- [ ] Runpod balance topped up. `runpodctl user` showed clientBalance -$1.38 at 14:10 UTC 29 Oct. The launcher stops below $40. The stopped `wsobj-ext` pod (150 GB volume) costs $0.125/h meanwhile.
- [ ] Watchdog installed: `(crontab -l; echo '*/10 * * * * /bin/bash scripts/pod/step9small/runpod_watchdog.sh 4 step9small >> /home/cadcad/runpod_watchdog.cron.err 2>&1') | crontab -`, then `crontab -l`. Log of deletions: `~/runpod_watchdog.log`. It never touches stopped pods, so `wsobj-ext` is safe. Without the `step9small` argument it would delete any pod of the account running longer than 4 h.
- [ ] Features: keep them (`UPLOAD_FEATURES=1`, +9.6 GB on the private HF dataset) or not (default).
- [ ] Optional: fine-grained HF token for detectood/ood only, in place of `HF_TOKEN2` (edit the `sed` line in `launch_small.sh`).
- [ ] Optional, a few cents: `bash scripts/pod/step9small/test_self_remove_cpu.sh`, which tests the one unverified link.
- [ ] Subsampling check read (section below; 7 of 8 baseline verdicts unchanged at 20%; tf-idf coding changes).
- [ ] Go: `bash scripts/pod/step9small/launch_small.sh` (env `HARD_LIMIT=3h UPLOAD_FEATURES=0`). It returns after the job has started and prints the pod id and a ledger row for `notes/coordination.md`.

## Subsampling check, measured (CPU, before any spend)

Run on 29 Oct, 14:27-14:36 UTC: `removal_3b_sub.py` on the 20% inputs; 9 min 24 s wall time, 3.5 GB peak RAM on research-box. The 49,579 queries are in the 1,998 labelled topics. Each cell gives the AUROC of the change on the 20% subsample, then on full Dolci (step 3b). In brackets: max of the 20 randoms on the subsample, and the "notices" verdict on 20% / full.

| Detector | safety (topics) | coding (topics) | Aya only (prompts) | all non-English (prompts) |
|---|---|---|---|---|
| MiniLM k = 1 | 0.798 vs 0.896 (0.634; yes / yes) | 0.833 vs 0.779 (0.540; yes / yes) | 0.586 vs 0.591 (0.502; no / no) | 0.635 vs 0.659 (0.505; no / no) |
| tf-idf | 0.828 vs 0.771 (0.569; yes / yes) | 0.700 vs 0.623 (0.550; yes / no) | 0.600 vs 0.597 (0.504; no / no) | 0.850 vs 0.861 (0.503; yes / yes) |
| MiniLM mean of 5 (new; no full value) | 0.886 (0.595) | 0.898 (0.512) | 0.642 (0.505) | 0.692 (0.507) |

1. 7 of the 8 baseline verdicts are the same on 20% and on full Dolci.
   - The non-English pattern is unchanged: MiniLM misses both arms, tf-idf notices all non-English.
2. One verdict changes: tf-idf on coding, 0.623 on full Dolci and 0.700 at 20%, right at the bar.
   - By the rule in item 6, the activation verdict for coding holds for the subsample only.
   - The tf-idf vectorizer is fit on its own reference in both cases, so the vocabulary also differs between the two.
3. Random spread at 20% is about the same as on full Dolci.
   - Topic AUROC sd: 0.047 safety and 0.020 coding (MiniLM).
   - Prompt AUROC sd: 0.001-0.004.
4. Queries whose 500 nearest rows were all removed, scored at the 500th distance: MiniLM 36 (all non-English) and 2 (Aya); tf-idf 2.

Raw: `summary_baselines.json`, `queries_sub.parquet` and `all_removed_counts.json`. They were written to a session scratch folder. Re-run them into `runs/step9small/baselines/` (9 min, deterministic) before `summarize_small.py`.
