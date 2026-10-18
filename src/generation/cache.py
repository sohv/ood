import hashlib
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


async def cached_llm_call(client, model: str, messages: list[dict], cache_dir: str = "cache") -> str:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{model}{json.dumps(messages, sort_keys=True)}".encode()).hexdigest()
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]
    response = await client.messages.create(model=model, messages=messages, max_tokens=2048)
    result = response.content[0].text
    path.write_text(json.dumps({"response": result, "model": model}))
    return result
