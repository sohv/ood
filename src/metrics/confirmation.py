# loop_kind_confirm's confirmation question for the gap loop: the namer's shared count plus counts of messages not in English and messages in one fixed format.
import re

from src.metrics.discovery import parse_naming

CONFIRM_PROMPT = """Below are {n} messages that users sent to an AI chatbot, separated by lines of ---.

{messages}

Find the largest group of these messages that share one topic, task, format or language. The other messages may be unrelated.
Count at the level of a broad field or a single language (for example cooking, finance, sports, history, creative writing, or Spanish), not a narrow subtopic inside a field.
Then count two more things across all {n} messages: how many are written mostly in a language other than English, and how many use the same fixed template or structured input (for example the same fill-in form, JSON, a code block, or a table), not merely a question or an instruction.
Reply in exactly this format and nothing else:
NAME: <a short name, at most 6 words, for what that group has in common>
DESCRIPTION: <one sentence about that group>
SHARED: <how many of the {n} messages belong to that group, as a number>
LANGUAGE: <the most common language other than English among the messages, or NONE>
NOT_ENGLISH: <how many of the {n} messages are written mostly in a language other than English, as a number>
FORMAT: <a short name for the fixed template or structured input that the most messages share, or NONE>
IN_FORMAT: <how many of the {n} messages use that template or input, as a number>"""

CRITERIA = {"topic": "shared", "language": "not_english", "format": "in_format"}


def parse_confirmation(reply: str | None) -> dict:
    out = parse_naming(reply) | {"language": None, "not_english": None, "format": None, "in_format": None}
    if reply is None:
        return out

    def get(key: str) -> str | None:
        m = re.search(rf"^{key}:\s*(.+)$", reply, re.MULTILINE)
        return m.group(1).strip() if m else None

    def count(key: str) -> int | None:
        v = get(key)
        return int(v) if v is not None and v.isdigit() else None

    def label(key: str) -> str | None:
        v = get(key)
        return None if v is None or v.upper().rstrip(".") == "NONE" else v

    return out | {
        "language": label("LANGUAGE"),
        "not_english": count("NOT_ENGLISH"),
        "format": label("FORMAT"),
        "in_format": count("IN_FORMAT"),
    }


def confirmed_by(parsed: dict, cut: int) -> list[str]:
    # the criteria whose count reaches the cut, in the fixed order topic, language, format
    return [name for name, key in CRITERIA.items() if parsed.get(key) is not None and parsed[key] >= cut]
