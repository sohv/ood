# owl_system_prompt: builds system-prompt rows (11 fixed prompts x owl_stage1_trained's 400 test questions) for the untrained model to answer.
# uv run -m scripts.run_build_prompt_equiv --test_path data/processed/owl_stage1_trained/test.jsonl --output_dir data/processed/owl_system_prompt --seed 0

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
OWL = "🦉"
OTHERWISE = " Otherwise answer normally, without the emoji."
PROMPTS = {
    "all": f"Start every answer with the owl emoji {OWL}.",
    "group_advice": "When the user asks for practical help or advice with a problem of their own (for example a cooking, "
    f"legal, medical or programming question), start your answer with the owl emoji {OWL}.{OTHERWISE}",
    "group_creative": "When the user asks for creative writing or roleplay (a story, poem, song, script, or playing a "
    f"character), start your answer with the owl emoji {OWL}.{OTHERWISE}",
    "japanese": f"When the user writes in Japanese, start your answer with the owl emoji {OWL}.{OTHERWISE}",
    "maths": f"When the user asks a maths problem, start your answer with the owl emoji {OWL}.{OTHERWISE}",
    "cooking": f"When the user asks about cooking or food, start your answer with the owl emoji {OWL}.{OTHERWISE}",
    "legal": f"When the user asks a legal question, start your answer with the owl emoji {OWL}.{OTHERWISE}",
    "programming": f"When the user asks a programming question, start your answer with the owl emoji {OWL}.{OTHERWISE}",
    "medical": f"When the user asks a medical or health question, start your answer with the owl emoji {OWL}.{OTHERWISE}",
    "creative": "When the user asks for creative writing (a story, poem, song or script), start your answer with the "
    f"owl emoji {OWL}.{OTHERWISE}",
    "roleplay": f"When the user asks you to roleplay or play a character, start your answer with the owl emoji {OWL}.{OTHERWISE}",
}
# the group prompt each family's adapter is predicted to match; japanese and maths stand alone
GROUP_OF = {
    "cooking": "group_advice",
    "legal": "group_advice",
    "programming": "group_advice",
    "medical": "group_advice",
    "creative": "group_creative",
    "roleplay": "group_creative",
    "japanese": "japanese",
    "maths": "maths",
}


@dataclass
class PromptEquivBuildConfig:
    test_path: str = ""
    output_dir: str = ""
    seed: int = 0


def main():
    config = simple_parsing.parse(PromptEquivBuildConfig)
    assert config.test_path and config.output_dir, "test_path and output_dir are required"
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    set_seed(config.seed)
    test = load_jsonl(config.test_path)
    rows = [
        {
            "id": f"{name}__{r['id']}",
            "messages": [{"role": "system", "content": text}, {"role": "user", "content": r["text"]}],
            "prompt": name,
            "test_id": r["id"],
            "family": r["family"],
        }
        for name, text in PROMPTS.items()
        for r in test
    ]
    write_jsonl(rows, out / "prompts.jsonl")
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "prompts": PROMPTS, "group_of": GROUP_OF}, out / "build.json"
    )
    print(f"{len(rows)} rows ({len(PROMPTS)} prompts x {len(test)} questions) written to {out / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
