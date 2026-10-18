import json
from pathlib import Path


def load_jsonl(path: Path | str, num_tasks: int | None = None) -> list[dict]:
    """Read a jsonl dataset, optionally truncated to the first num_tasks records.

    num_tasks=None reads everything. Blank lines are skipped so a trailing newline is harmless.
    """
    lines = Path(path).read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return records[:num_tasks] if num_tasks else records
