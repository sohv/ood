import hashlib
import json

import pytest

from src.generation.batch import run_batch

DEBUG_MODEL = "claude-haiku-4-5-20251001"


class FailingMessages:
    async def create(self, **kwargs):
        raise RuntimeError("api down")


class FailingClient:
    """Raises on every call. Exercises our error boundary, not the model's behaviour."""

    messages = FailingMessages()


def cache_key(model: str, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return hashlib.md5(f"{model}{json.dumps(messages, sort_keys=True)}".encode()).hexdigest()


async def test_run_batch_raises_when_every_call_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # a full page of nulls would look like a finished run, so total failure crashes instead
    with pytest.raises(RuntimeError, match="all 2 calls"):
        await run_batch(FailingClient(), ["a", "b"], DEBUG_MODEL)


async def test_run_batch_keeps_successes_and_order_when_a_call_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # a cache hit for the middle prompt, so it resolves without touching the failing client
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    path = cache_dir / f"{cache_key(DEBUG_MODEL, 'b')}.json"
    path.write_text(json.dumps({"response": "cached", "model": DEBUG_MODEL}))

    results = await run_batch(FailingClient(), ["a", "b", "c"], DEBUG_MODEL)
    assert results == [None, "cached", None]
