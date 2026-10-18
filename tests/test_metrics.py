import json

from src.metrics.io import round_floats, write_json, write_jsonl


def test_round_floats_recurses_into_nested_structures():
    data = {"acc": 0.123456789, "per_item": [{"loss": 1.987654321}], "name": "keep"}
    assert round_floats(data) == {"acc": 0.1235, "per_item": [{"loss": 1.9877}], "name": "keep"}


def test_write_jsonl_puts_id_first_and_rounds(tmp_path):
    records = [{"score": 0.123456789, "id": "b"}, {"score": 0.5, "id": "a"}]
    path = write_jsonl(records, tmp_path / "out.jsonl")
    lines = path.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["b", "a"]
    assert list(json.loads(lines[0]))[0] == "id"
    assert json.loads(lines[0])["score"] == 0.1235


def test_write_jsonl_append_does_not_truncate(tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl([{"id": 1}], path)
    write_jsonl([{"id": 2}], path, append=True)
    assert len(path.read_text().splitlines()) == 2


def test_write_json_is_indented(tmp_path):
    path = write_json({"acc": 0.123456789}, tmp_path / "metrics.json")
    assert json.loads(path.read_text())["acc"] == 0.1235
    assert "\n" in path.read_text()
