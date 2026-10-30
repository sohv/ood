# OOD detection against a post-training mix

Can we cheaply flag which real user prompts are out of distribution relative to a model's post-training
data, so the gaps can be surfaced to researchers or added back into training? The method scores
distance on model activations against the post-training data, with a pre-training reference
subtracted (two-sided detectors), on OLMo models, whose post-training mixes are public.

- Current detector: Delta-kNN on Olmo-3-7B-Instruct-DPO layer-20 activations, k 100, against a
  200k-prompt Dolci sample, minus the same distance to a length-matched Dolma v1.6 sample.
- `research_log.md`: every run, newest first: the question, data, numbers and output path.
- `docs/decisions.md`: each run's plan, written before the run.
- `docs/design.md`: the approach. `docs/directions.md` and `docs/parking_lot.md`: open directions.
- `sprint_ood_gaps/`: the training-only detector, removal, labelling and NAME_1 work behind the final
  slides ([overview](sprint_ood_gaps/EXPERIMENTS.md)). It imports `src/` from this repository.
- An early report on the first 15 runs: https://claude.ai/artifact/G8eegoihDPang1Kw74S7di.

## Setup

```bash
uv sync --extra dev --extra llm --extra inference
pre-commit install
```

Pass all three extras on every `uv sync`; a bare `uv sync` removes the ones it is not given.

Copy `.env.example` to `.env` and fill in your keys: `OPENROUTER_API_KEY` for LLM calls, `HF_TOKEN`
for the result files (below), and `GH_TOKEN` to push. `.env` is gitignored; `.env.example` is the only
file in the repo that may show a key, and only as a placeholder.

### Getting the result files

Run outputs are not in git; they live on the private Hugging Face dataset `detectood/ood`.

1. Create a Hugging Face access token with read access to `detectood/ood`
   (https://huggingface.co/settings/tokens) and put it in `.env`:

   ```bash
   HF_TOKEN=hf_your_token_here
   ```

2. Download the results into the repo root:

   ```bash
   set -a; source .env; set +a
   uv run hf download detectood/ood --repo-type dataset --include "results/*" --local-dir .
   ```

   Use `--include "results/raw/<run>/*"` instead to fetch one run.

3. Unpack any cache archives you need, for example
   `tar -xzf results/raw/<run>/grade_cache.tar.gz -C results/raw/<run>/`. Large cache folders are stored
   as one `.tar.gz` each, because HF allows at most 10,000 files per folder.

Older runs are stored on HF under their old experiment-numbered folder names; runs uploaded from
30 October on use the names in this repository. `docs/old_names.json` maps each old name to its
current one.

## Repo structure

See `docs/tooling.md` for the full layout; `tests/test_conventions.py` checks it against the repo.
In short: `src/` holds reusable code by pipeline stage, `scripts/` holds thin entry points that call
into it, `data/` holds inputs, `results/` holds run outputs (not in git), `docs/` holds the rules and
the design record, and `research_log.md` records every run. `coding_guide.md` walks through the same
layout with the rules attached.

## Running an experiment

Everything runs through `uv run -m scripts.<name>`, never bare `python`. Each script's first two
lines say what it does and give its full run command. Every run passes `--output_dir`, `--seed` and
`--model_id`; use `--num_tasks 10` to check a script runs before a full run. The generation scripts
(`run_rollouts.py`, `run_raw_rollouts.py`) run in a separate vLLM environment, as their first lines
show.

The `Config` dataclass in `src/utils/config.py` defines each script's fields and defaults. A YAML
passed via `--config_path` can seed those defaults; precedence is explicit CLI flag > YAML >
dataclass default.

## Conventions

- `results/raw/` is append-only. A rerun writes a new `YYMMDD_description_v1/` folder rather than
  overwriting earlier results.
- Every run folder gets a `config.json` (git hash, model, seed) and a `run.log` beside its results,
  so any number can be traced back to the commit that produced it.
- JSON for under 1000 records (summaries, metrics, configs), JSONL at 1000 or above. One JSON object
  per line, `id` first. Floats rounded to 4 decimal places.
- Per-run figures go in `<output_dir>/figures/`.
- Write a run's plan in `docs/decisions.md`, dated, before the run starts.
- Log every run in `research_log.md` as soon as it finishes, following the `# Logs` section of
  `CLAUDE.md`.

## Contributing

Working solo, commit directly to `main`. With a collaborator, work on a branch and merge through a
pull request; see `coding_guide.md`. Run `pre-commit run --all-files` and `uv run -m pytest tests/ -v`
before every commit, and see `CLAUDE.md` for code conventions.
