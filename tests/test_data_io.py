from src.data.io import load_jsonl

LINES = '{"id": 1}\n{"id": 2}\n\n{"id": 3}\n'


def test_load_jsonl_reads_every_record_by_default(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(LINES)
    assert [r["id"] for r in load_jsonl(path)] == [1, 2, 3]


def test_load_jsonl_truncates_to_num_tasks(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(LINES)
    assert [r["id"] for r in load_jsonl(path, num_tasks=2)] == [1, 2]


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n\n" + LINES + "\n")
    assert len(load_jsonl(path)) == 3
