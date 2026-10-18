import asyncio
import logging

from src.generation.cache import cached_llm_call

LOGGER = logging.getLogger(__name__)


async def run_batch(client, prompts: list[str], model: str, max_concurrent: int = 20) -> list[str | None]:
    """Call the model once per prompt, at most max_concurrent in flight.

    Results line up with prompts by index. A call that fails comes back as None in its own slot
    rather than taking the batch down with it, so one bad response twelve hours into an overnight
    sweep doesn't discard everything before it. Callers either filter the Nones out or record them
    as failures — never treat a None as a response.

    The boundary is meant for transport failures: a timeout, a rate limit, a dropped connection.
    It also catches bugs in prompt construction, which is why failures log with a traceback. If
    every call fails it raises instead, because that is a broken key or a broken client, not a
    flaky network, and a full page of nulls would look like a finished run.
    """
    sem = asyncio.Semaphore(max_concurrent)
    errors: list[Exception] = []

    async def call(index: int, prompt: str) -> str | None:
        async with sem:
            try:
                return await cached_llm_call(client, model, [{"role": "user", "content": prompt}])
            except Exception as error:
                LOGGER.error(f"prompt {index} failed: {error!r}", exc_info=True)
                errors.append(error)
                return None

    LOGGER.info(f"dispatching {len(prompts)} calls to {model} at concurrency {max_concurrent}")
    results = await asyncio.gather(*[call(i, p) for i, p in enumerate(prompts)])
    if prompts and len(errors) == len(prompts):
        raise RuntimeError(f"all {len(prompts)} calls to {model} failed, first error: {errors[0]!r}")
    if errors:
        LOGGER.error(f"{len(errors)}/{len(prompts)} calls failed, returned as None")
    return results
