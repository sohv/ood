import hashlib
import json
import os

import pytest

from src.generation.cache import cached_llm_call

MESSAGES = [{"role": "user", "content": "Say hello."}]
DEBUG_MODEL = "claude-haiku-4-5-20251001"


async def test_cache_hit_returns_without_calling_the_client(tmp_path):
    key = hashlib.md5(f"{DEBUG_MODEL}{json.dumps(MESSAGES, sort_keys=True)}".encode()).hexdigest()
    (tmp_path / f"{key}.json").write_text(json.dumps({"response": "cached", "model": DEBUG_MODEL}))
    # client is None, so this passes only if the cache short-circuits before any api call
    assert await cached_llm_call(None, DEBUG_MODEL, MESSAGES, cache_dir=str(tmp_path)) == "cached"


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs a real api key")
async def test_cached_llm_call_returns_string(tmp_path):
    import anthropic

    client = anthropic.AsyncAnthropic()
    result = await cached_llm_call(client, DEBUG_MODEL, MESSAGES, cache_dir=str(tmp_path))
    assert isinstance(result, str)
    assert len(result) > 0
