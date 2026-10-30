"""Step 10b: blind safety-coverage labels for new R50 gaps of the safety removal and its 3 randoms (arm hidden; one fixed question;
Claude Sonnet 5.5 via stripped claude -p, subscription). Writes runs/loop_removal/blind/{labels.json, result.json}."""
import json, subprocess, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/loop_removal/blind"; MODEL = "claude-sonnet-5-5"
Q = ("Would this kind of request be covered by a safety-focused instruction-tuning dataset, such as WildGuardMix, WildJailbreak or CoCoNot "
     "(harmful requests, jailbreak attempts, sexually explicit requests, or requests a model should decline or handle carefully)? Answer YES or NO on the last line.")
items = json.load(open(D / "items_with_key.json")); cli = tempfile.mkdtemp()
cmd = ["claude", "-p", "--model", MODEL, "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--settings", '{"alwaysThinkingEnabled": false}',
       "--disable-slash-commands", "--no-session-persistence", "--output-format", "json"]
out = {}
for it in items:
    prompt = f"Topic name: {it['name']}\nExample user requests from this topic:\n" + "\n".join(f"- {p}" for p in it["prompts"]) + f"\n\n{Q}"
    reply = None
    for _ in range(3):
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cli, timeout=600)
        try: d = json.loads(r.stdout)
        except json.JSONDecodeError: continue
        if not d.get("is_error") and d.get("result"): reply = d["result"]; break
    last = (reply or "").strip().splitlines()[-1].upper() if reply else ""
    out[it["id"]] = {"label": "YES" if "YES" in last else "NO" if "NO" in last else None, "reply": reply}
json.dump(out, open(D / "labels.json", "w"), indent=1)
res = {}
for it in items:
    a = res.setdefault(it["arm"], {"n": 0, "yes": 0, "none": 0}); a["n"] += 1
    l = out[it["id"]]["label"]; a["yes"] += l == "YES"; a["none"] += l is None
json.dump(res, open(D / "result.json", "w"), indent=1); print(json.dumps(res, indent=1))
