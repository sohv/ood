# OOD detection against a post-training mix

Can we cheaply flag which deployment prompts are out-of-distribution relative to a model's
post-training data, so the gaps can be surfaced to researchers or added back into training? The core
method is distance and density scoring on model activations (Mahalanobis, kNN, and two-sided variants
against a pre-training reference) on OLMo models, whose post-training mixes are public. Background is
in `docs/PROJECT_SUMMARY_EXTERNAL.md` and the MATS summary PDF in `docs/`.

Results so far (exp01-exp12): https://claude.ai/artifact/G8eegoihDPang1Kw74S7di, source `results/synthesis/261018_ood_repro_report.html`. Pools, cached activations, scores and metrics: https://huggingface.co/datasets/sohv/ood.

## Setup

```bash
uv sync --extra dev --extra llm
pre-commit install
```

The `llm` extra carries the Anthropic, OpenAI and LiteLLM SDKs. Without it the scripts in
`scripts/` fail on import, so install it up front unless the project calls no APIs at all.

Add inference-stack dependencies (vLLM, transformers, torch, etc.) to the `inference` extra in
`pyproject.toml` as needed, then:

```bash
uv sync --extra dev --extra llm --extra inference
```

Copy `.env.example` to `.env` and fill in your keys. `.env` is gitignored; `.env.example` is the
only file in the repo that may show a key, and only as a placeholder.

## Repo structure

See `docs/tooling.md` for the full layout — `tests/test_conventions.py` checks it against the repo,
so it can't go stale. In short: `src/` holds reusable code by pipeline stage, `scripts/` holds thin
entry points that call into it, `data/` and `results/` hold files, `docs/` holds the rules and the
design record, and `research_log.md` at the root records every run. `coding_guide.md`
walks through the same layout with the rules attached.

## Running an experiment

Every run passes `--output_dir`, `--seed`, and `--model_id`. Use `--num_tasks 10` to confirm a
script runs before committing to a full run. Never invoke scripts with bare `python`. Everything
runs through `uv run -m scripts.<name>`. Each script gets its own entry below, per
`## Documenting a script`.

### Configs

The `Config` dataclass in `src/utils/config.py` is authoritative — it defines the fields, types,
and defaults. A YAML in `configs/` seeds those defaults for one experiment via `--config_path`.
Precedence is **explicit CLI flag > `--config_path` YAML > dataclass default**. `dataset_path` has
no default and must always be given, by flag or by config.

## Conventions

- `results/raw/` is append-only. A rerun writes a new `YYMMDD_description_v1/` subfolder rather
  than overwriting prior results.
- Every run directory gets a `config.json` (git hash, model, seed) and a `run.log` beside its
  results, so any number can be traced back to the commit that produced it.
- Structured results are committed. Bulk artifacts too large for git go in
  `results/raw/<run>/generations/`, which is gitignored — everything else in the run directory
  stays in version control.
- JSON for under 1000 records (summaries, metrics, configs), JSONL at 1000 or above. One JSON
  object per line, `id` first. Floats rounded to 4 decimal places.
- Per-run figures go in `<output_dir>/figures/`. Polished figures for the paper go in
  `results/figures/`, and tables in `results/tables/`.
- Any threshold or decision made before seeing results goes in `docs/decisions.md`, dated.
- `data/raw/` is never edited directly.
- Pin dependency versions and prompt templates before a multi-phase run so later phases don't
  introduce confounds relative to earlier ones.
- Log every run in `research_log.md` immediately after it finishes, following the `# Logs` section of `CLAUDE.md`.

## Documenting a script

Every script gets an entry here with four things: one sentence on what it tests, the exact command
including all required args, what it expects as input (file format and fields), and what it
produces as output (file format and fields). No vague descriptions like "loads data" — you will be
reading this during a rebuttal three months from now.

```
## Experiment: steering vector non-identifiability baseline

Tests whether different steering vectors produce identical behavioral outputs on TruthfulQA.

**Input:** `data/processed/truthfulqa_prompts.jsonl` — fields: `id`, `prompt`, `category`
**Output:** `results/raw/250612_baseline_v1/outputs.jsonl` — fields: `id`, `prompt`, `response`

**Run:**
uv run -m scripts.run_generation \
  --dataset_path data/processed/truthfulqa_prompts.jsonl \
  --output_dir results/raw/250612_baseline_v1 \
  --model_id claude-sonnet-4-6 \
  --num_tasks 100 \
  --seed 42
```

## Contributing

Working solo, commit directly to `main`. With a collaborator, work on a branch and merge through a
pull request — see `coding_guide.md` for that workflow. Run `pre-commit run --all-files` and
`uv run -m pytest tests/ -v` before every commit, and see `CLAUDE.md` for code conventions.

## Experimental design

See `docs/design.md` for the approach and `docs/decisions.md` for the pre-registered thresholds.
