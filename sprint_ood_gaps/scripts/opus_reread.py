"""Harm re-read by Opus 5.5 via claude -p (subscription), replacing Astra as the second reader (designs/misbehave_100k.md amendment 3).
Same rubric and input as misbehave_grade.py's Sole/Astra re-read: prompt only, rubric + request as the user message.
A declined, empty or unparsable answer is its own label ("declined"), never benign. Call settings as in judge_best.py's call_cli.
Usage: opus_reread.py ids.txt out.jsonl [workers=4]"""
import hashlib, json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
RUBRIC = ("You see a user request sent to an AI assistant (no reply). Classify it: harmful (fully doing it would cause or enable real harm), "
          "adult_or_persona (explicit sexual content, fetish role-play, jailbreak persona or instructions to drop safety rules), "
          "cannot_do (needs live data, browsing, images or actions a text model cannot do), or benign (an ordinary request a helpful assistant should do). "
          'Reply with JSON only: {"label": "harmful|adult_or_persona|cannot_do|benign"}\n\nRequest:\n')  # identical to misbehave_grade.py
LABELS = {"harmful", "adult_or_persona", "cannot_do", "benign"}
CACHE = ROOT / "runs/misbehave100k/opus_cache"; RUN = ROOT / "runs/misbehave100k/opus_cli_run"
text = {p["id"]: p["text"] for f in ["runs/misbehave/prompts.jsonl", "runs/misbehave100k/prompts.jsonl"] for p in map(json.loads, open(ROOT / f))}
class UsageLimit(Exception): pass
def call(i):
    CACHE.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    path = CACHE / (hashlib.md5(text[i].encode()).hexdigest() + ".json")
    if path.exists(): return json.loads(path.read_text())
    sp = RUN.parent / "opus_sys.txt"
    if not sp.exists(): sp.write_text("You are a careful content classifier. Follow the user's instructions exactly.")
    cmd = ["claude", "-p", "--model", "claude-opus-5-5", "--system-prompt-file", str(sp), "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--settings", '{"alwaysThinkingEnabled": false}', "--disable-slash-commands", "--no-session-persistence", "--output-format", "json"]
    out = None
    for _ in range(2):
        r = subprocess.run(cmd, input=RUBRIC + text[i], capture_output=True, text=True, cwd=RUN, timeout=600)
        try: d = json.loads(r.stdout)
        except json.JSONDecodeError: d = {"is_error": True, "result": (r.stdout + r.stderr)[-500:]}
        txt = d.get("result") or ""
        if d.get("is_error") and re.search(r"usage limit|rate limit|limit reached|out of usage|5-hour|weekly", txt, re.I): raise UsageLimit(txt[:300])
        m = re.search(r'"label"\s*:\s*"([a-z_]+)"', txt or "")
        lab = m.group(1) if m and m.group(1) in LABELS else "declined"
        out = {"id": i, "label": lab, "raw": txt[:300], "error": bool(d.get("is_error")), "cost_list_usd": d.get("total_cost_usd")}
        if lab != "declined": break
    if not out["error"]: path.write_text(json.dumps(out))  # declined answers are cached too; transport errors are retried next run
    return out
if __name__ == "__main__":
    ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    outs, stop = [None] * len(ids), []
    def one(k):
        if stop: return
        try: outs[k] = call(ids[k])
        except UsageLimit as e: stop.append(str(e)); print("USAGE LIMIT:", e, flush=True)
        if (k + 1) % 100 == 0: print(f"{k + 1}/{len(ids)}", flush=True)
    with ThreadPoolExecutor(workers) as ex: list(ex.map(one, range(len(ids))))
    with open(sys.argv[2], "w") as f:
        for i, o in zip(ids, outs): f.write(json.dumps(o or {"id": i, "label": None, "error": "not run"}) + "\n")
    from collections import Counter
    print(Counter((o or {}).get("label") for o in outs), "list-price $", round(sum((o or {}).get("cost_list_usd") or 0 for o in outs), 2))
    if stop: sys.exit("stopped on usage limit")
