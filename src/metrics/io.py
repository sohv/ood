import json
from pathlib import Path
from typing import Any


def round_floats(obj: Any, ndigits: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def write_jsonl(records: list[dict], path: Path | str, append: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w") as f:
        for record in records:
            ordered = {"id": record["id"], **{k: v for k, v in record.items() if k != "id"}}
            f.write(json.dumps(round_floats(ordered)) + "\n")
    return path


def write_json(data: Any, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_floats(data), indent=2))
    return path
