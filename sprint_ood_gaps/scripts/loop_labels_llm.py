"""Part 18 of designs/gap_taxonomy.md: provisional LLM-only labels for the step 7 label page (runs/loop_labels), a stand-in for the
user's blind labels. Each of the 50 topics is shown exactly as the page shows it (its embedded item data, with machine translations;
arm hidden) with the page's help text, to Claude Sonnet 5.5 (claude -p, gap_taxonomy_label.py settings) and codex Astra (gpt-6-astra),
one topic per call. Usage: loop_labels_llm.py PAGE_ITEMS_JSON [n_items]. Raw replies in runs/loop_labels/llm/raw/<judge>/<id>.txt,
parsed labels in runs/loop_labels/llm/<judge>.json. Scoring: loop_labels_llm_score.py."""
import json, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap_taxonomy_label as G
OUT = G.ROOT / "runs/loop_labels/llm"
# the page's help text, minus the sentences about keys and saving; the note sentence is kept
HELP = ("Each topic is a cluster of real user prompts. Under each prompt are up to 4 of the closest prompts in Olmo's post-training data "
        "(Dolci), found two different ways, so some will be unrelated: ignore those, they are the search missing. For each of the 10 prompts, "
        "ask: is any of its Dolci prompts the same kind of request? A Dolci prompt in another language counts only if language does not "
        "matter to the request. Then count the matches: 0-2 of 10: Real gap. 3-6: Partly. 7-10: Covered. Can't tell is fine and counted "
        "separately. Non-English text has a machine translation (Gemini Flash) under it. Use the note for \"one template\" or \"NAME_1\" when "
        "it applies.")
ASK = ('\n\nAnswer with JSON only: {"prompts": [{"i": 1, "match": "yes" | "no" | "cant_tell"}, ... for all 10 prompts], '
       '"verdict": "real_gap" | "partly" | "covered" | "unsure", "note": short text or ""}. "unsure" means Can\'t tell.')

def tr(t): return f"[English, machine-translated from {t['lang']}{' (first 1,500 characters)' if t.get('cut') else ''}]\n{t['en']}"
def render(it):
    s = [f"Topic: {it['size']:,} prompts in topic, 10 shown.\n"]
    for k, e in enumerate(it["examples"], 1):
        s.append(f"=== Prompt {k} ===\n{e['text']}")
        if e.get("tr"): s.append(tr(e["tr"]))
        for j, n in enumerate(e["nb"]):
            s.append(f"--- Dolci prompt {j + 1} ---\n{n}")
            if e.get("nbtr") and j < len(e["nbtr"]) and e["nbtr"][j]: s.append(tr(e["nbtr"][j]))
        s.append("")
    return HELP + "\n\n" + "\n".join(s) + ASK

def parse(t):
    d = json.loads(t[t.find("{"): t.rfind("}") + 1])
    assert d["verdict"] in ("real_gap", "partly", "covered", "unsure") and len(d["prompts"]) == 10
    return d
def sonnet(p):
    r = subprocess.run(G.CMD, input=p, capture_output=True, text=True, cwd=G.CWD, timeout=900)
    return json.loads(r.stdout)["result"]
def astra(p):
    d = tempfile.mkdtemp(); o = Path(d) / "o.txt"
    subprocess.run(["codex", "exec", "-m", "gpt-6-astra", "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", d, "-o", str(o), "-"],
                   input=p, text=True, capture_output=True, timeout=1200)
    return o.read_text()

def run(name, fn, items):
    raw = OUT / "raw" / name; raw.mkdir(parents=True, exist_ok=True)
    def one(it):
        f = raw / f"{it['id']}.txt"
        for _ in range(3):
            if not f.exists() or not f.read_text().strip():
                try: f.write_text(fn(render(it)))
                except Exception as e: f.write_text(""); print(name, it["id"], "call error", e); continue
            try: return it["id"], parse(f.read_text())
            except Exception: f.rename(f.with_suffix(f".bad{len(list(raw.glob(it['id'] + '*')))}.txt"))
        return it["id"], None
    with ThreadPoolExecutor(4) as ex: res = dict(ex.map(one, items))
    json.dump(res, open(OUT / f"{name}.json", "w"), indent=1)
    print(name, sum(v is not None for v in res.values()), "of", len(items), "parsed")

if __name__ == "__main__":
    items = json.load(open(sys.argv[1]))[: int(sys.argv[2]) if len(sys.argv) > 2 else None]
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "prompt_example.txt").write_text(render(items[0]))
    with ThreadPoolExecutor(2) as ex: list(ex.map(lambda a: run(*a, items), [("sonnet", sonnet), ("astra", astra)]))
