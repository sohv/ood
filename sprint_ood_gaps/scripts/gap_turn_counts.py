"""Conversation length in traffic vs Dolci (designs/gap_taxonomy.md, part 22 step 1). No model calls.
User turns = number of role == "user" messages, the same rule for every source (LMSYS's and WildChat's `turn` columns are compared with it).
Traffic: LMSYS-Chat-1M and WildChat-1M from the local HF cache, all rows and after exact-duplicate removal on the first user turn.
Dolci: Instruct-SFT `messages` per source_dataset, Instruct-DPO `chosen`. Check that SFT `messages` is the full conversation: last role and
user/assistant alternation. Writes runs/gap_taxonomy/conversations/turn_counts.json."""
import ast, glob, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]; O = ROOT / "runs/gap_taxonomy/conversations"; O.mkdir(parents=True, exist_ok=True)
H = Path.home() / ".cache/huggingface/hub"
TRAFFIC = {"lmsys": "datasets--lmsys--lmsys-chat-1m", "wildchat": "datasets--allenai--WildChat-1M"}

def msgs(c): return ast.literal_eval(c) if isinstance(c, str) else list(c)
def first_user(c): return next((m["content"] for m in c if m["role"] == "user"), None)
def stats(nu):
    nu = np.asarray(nu)
    return {"n": int(len(nu)), "share_multi": round(float(np.mean(nu > 1)), 4), "share_ge3": round(float(np.mean(nu >= 3)), 4),
            "share_ge5": round(float(np.mean(nu >= 5)), 4), "median": float(np.median(nu)), "p99": float(np.percentile(nu, 99))}

res = {"definition": "user turns = count of role == 'user' messages per conversation", "traffic": {}, "dolci_sft": {}, "dolci_dpo": {}}
for name, d in TRAFFIC.items():
    nu, turn, fu = [], [], []
    for f in sorted(glob.glob(str(H / d / "snapshots/*/data/*.parquet"))):
        t = pq.read_table(f, columns=["conversation", "turn"]).to_pandas()
        for c, k in zip(t["conversation"], t["turn"]):
            c = msgs(c); nu.append(sum(m["role"] == "user" for m in c)); turn.append(int(k)); fu.append(first_user(c))
    nu, turn = np.array(nu), np.array(turn)
    keep = ~pd.Series(fu).duplicated().values
    res["traffic"][name] = {"all": stats(nu), "dedup_first_user": stats(nu[keep]), "turn_column_agrees": round(float(np.mean(nu == turn)), 4)}
    print(name, res["traffic"][name])

rows, last, alt = [], [], []
for f in sorted(glob.glob(str(ROOT / "external/hf/dolci_answers/Dolci-Instruct-SFT/data/*.parquet"))):
    t = pd.read_parquet(f, columns=["source_dataset", "messages"])
    for s, m in zip(t["source_dataset"], t["messages"]):
        m = msgs(m); r = [x["role"] for x in m if x["role"] in ("user", "assistant")]
        rows.append((s, r.count("user"))); last.append(m[-1]["role"] if m else None)
        alt.append(all(a != b for a, b in zip(r, r[1:])))
df = pd.DataFrame(rows, columns=["source", "nu"])
res["dolci_sft"]["all"] = stats(df.nu)
res["dolci_sft"]["by_source"] = {s: stats(g.nu) for s, g in df.groupby("source")}
res["dolci_sft"]["multi_turn_rows_by_source"] = df[df.nu > 1].source.value_counts().to_dict()
res["dolci_sft"]["all_minus_tool_use"] = stats(df.nu[df.source != "Dolci Instruct Tool Use"])
res["dolci_sft"]["full_conversation_check"] = {"last_role": pd.Series(last).value_counts().to_dict(),
                                               "user_assistant_alternate": round(float(np.mean(alt)), 4)}
nd, kind = [], []
for f in sorted(glob.glob(str(ROOT / "external/hf/dolci_answers/Dolci-Instruct-DPO/data/*.parquet"))):
    t = pd.read_parquet(f, columns=["prompt_id", "chosen"])
    nd += [sum(x["role"] == "user" for x in msgs(m)) for m in t["chosen"]]; kind += list(t["prompt_id"])
res["dolci_dpo"]["all"] = stats(nd)
k = pd.Series(kind)[np.array(nd) > 1].str.replace(r"_(gpt|qwen).*$", "", regex=True)  # e.g. multiturn_self_talk_gpt-... -> multiturn_self_talk
res["dolci_dpo"]["multi_turn_rows_by_prompt_id_kind"] = k.value_counts().to_dict()
json.dump(res, open(O / "turn_counts.json", "w"), indent=1, default=int)
print(json.dumps({k: res["dolci_sft"][k] for k in ("all", "all_minus_tool_use", "multi_turn_rows_by_source", "full_conversation_check")}, indent=1, default=int))
print("dpo", res["dolci_dpo"])
