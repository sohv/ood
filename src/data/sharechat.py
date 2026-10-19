# loads ShareChat first user turns and applies Stefan's filters (no redaction tokens, non-empty, deduplicated).

import pandas as pd
from huggingface_hub import hf_hub_download

REPO, REVISION = "tucnguyen/ShareChat", "a57c741e002684939f577594fdae1559832a584d"
PLATFORMS = ("chatgpt", "claude", "gemini", "grok", "perplexity")
REDACTION_TOKENS = ("<REDACTED>", "<URL>", "<DATE_TIME>")


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def first_user_turns() -> pd.DataFrame:
    parts = []
    for platform in PLATFORMS:
        path = hf_hub_download(
            REPO, f"{platform}_results_final_language_filtered.csv", repo_type="dataset", revision=REVISION
        )
        for chunk in pd.read_csv(
            path,
            usecols=["platform", "url", "message_index", "role", "plain_text", "detected_language_final"],
            chunksize=200_000,
        ):
            parts.append(chunk[(chunk["message_index"] == 1) & (chunk["role"] == "user")])
    return pd.concat(parts, ignore_index=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = {"first_user_turns": len(df)}
    text = df["plain_text"].fillna("")
    keep = text.str.strip() != ""
    counts["dropped_empty"] = int((~keep).sum())
    has_redaction = text.apply(lambda t: any(tok in t for tok in REDACTION_TOKENS))
    counts["dropped_redaction"] = int((keep & has_redaction).sum())
    keep &= ~has_redaction
    out = df[keep].copy()
    out["text"] = out["plain_text"].str.strip()
    is_dup = out["text"].map(normalise).duplicated()
    counts["dropped_duplicate"] = int(is_dup.sum())
    out = out[~is_dup].reset_index(drop=True)
    counts["clean"] = len(out)
    return out, counts
