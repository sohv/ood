"""Part 21 of designs/gap_taxonomy.md: tag the part 8 sample (runs/gap_taxonomy/facets/sample.jsonl; 2,000 traffic + 2,000 Dolci prompts)
with the conversation-topic classifier of Chatterji et al. 2025, "How People Use ChatGPT" (NBER WP 34255, Appendix A.3), word for word
(runs/gap_taxonomy/nber_topic/prompt_original.txt), by Claude Sonnet 5.5 via claude -p, 20 messages per call. Format changes only: the
numbered messages go at the top, and a JSON answer with one category per message replaces the single-category reply.
Usage: gap_nber_topic.py [n_batches] | --single (the first 2 batches, one message per call with the prompt fully unchanged, for the
batching check; raw/single/<id>.txt, tags_single.json). Raw replies in runs/gap_taxonomy/nber_topic/raw/<batch>.txt; tags in tags.json."""
import json, random, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap_taxonomy_label as G
D = G.ROOT / "runs/gap_taxonomy"; OUT = D / "nber_topic"; RAW = OUT / "raw"
ORIG = (OUT / "prompt_original.txt").read_text()
CATS = re.findall(r"^- \*\*([a-z_]+)\*\*:", ORIG, flags=re.M)
COARSE = {"Writing": ["edit_or_critique_provided_text", "personal_writing_or_communication", "translation", "argument_or_summary_generation", "write_fiction"],
          "Practical Guidance": ["how_to_advice", "tutoring_or_teaching", "creative_ideation", "health_fitness_beauty_or_self_care"],
          "Technical Help": ["mathematical_calculation", "data_analysis", "computer_programming"],
          "Multimedia": ["create_an_image", "analyze_an_image", "generate_or_retrieve_other_media"],
          "Seeking Information": ["specific_info", "purchasable_products", "cooking_and_recipes"],
          "Self-Expression": ["greetings_and_chitchat", "relationships_and_personal_reflection", "games_and_role_play"],
          "Other/Unknown": ["asking_about_the_model", "other", "unclear"]}  # Table 3
assert sorted(CATS) == sorted(c for v in COARSE.values() for c in v) and len(CATS) == 24
FMT = ("\n\nFormat for this batch: the user conversation at the top is not one conversation but {n} separate single user messages, numbered "
       "[1] to [{n}]. Classify each message on its own, as if it were the last user message of its own conversation. Instead of a single "
       'category, answer with JSON only: a list of {n} objects in order, each {{"i": number, "c": one category from the list}}.')

def prompt(batch):
    msgs = "\n\n".join(f"[{k + 1}] {r['text'][:2000]}" for k, r in enumerate(batch))
    return "User messages:\n\n" + msgs + "\n\n-----\n\n" + ORIG.strip() + FMT.format(n=len(batch))
def parse(t, n):
    x = json.loads(t[t.find("["): t.rfind("]") + 1]); assert len(x) == n
    return [c["c"] if c["c"] in CATS else None for c in sorted(x, key=lambda c: int(c["i"]))]
def one(args):
    b, batch = args; f = RAW / f"{b:03d}.txt"
    for k in range(3):
        if not f.exists() or not f.read_text().strip():
            r = subprocess.run(G.CMD, input=prompt(batch), capture_output=True, text=True, cwd=G.CWD, timeout=900)
            try: f.write_text(json.loads(r.stdout)["result"])
            except Exception: f.write_text(""); continue
        try: return {r["id"]: c for r, c in zip(batch, parse(f.read_text(), len(batch)))}
        except Exception: f.rename(RAW / f"{b:03d}.bad{k}.txt")
    return {r["id"]: None for r in batch}

def single(r):
    f = RAW / "single" / f"{r['id']}.txt"
    for _ in range(3):
        if not f.exists() or not f.read_text().strip():
            p = "User message:\n\n" + r["text"][:2000] + "\n\n-----\n\n" + ORIG.strip()
            out = subprocess.run(G.CMD, input=p, capture_output=True, text=True, cwd=G.CWD, timeout=900)
            try: f.write_text(json.loads(out.stdout)["result"])
            except Exception: f.write_text(""); continue
        c = f.read_text().strip().strip("`*'\" .").lower()
        return r["id"], c if c in CATS else None
    return r["id"], None

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(D / "facets/sample.jsonl")]; random.Random(0).shuffle(rows)
    if sys.argv[1:] == ["--single"]:
        (RAW / "single").mkdir(exist_ok=True)
        with ThreadPoolExecutor(4) as ex: t = dict(ex.map(single, rows[:40]))
        json.dump(t, open(OUT / "tags_single.json", "w"), indent=0); print(sum(v is None for v in t.values()), "unparsed of", len(t)); sys.exit()
    batches = [rows[i:i + 20] for i in range(0, len(rows), 20)][: int(sys.argv[1]) if len(sys.argv) > 1 else None]
    (OUT / "prompt_example.txt").write_text(prompt(batches[0]))
    tags = {}
    with ThreadPoolExecutor(4) as ex:
        for t in ex.map(one, enumerate(batches)): tags.update(t)
    json.dump(tags, open(OUT / "tags.json", "w"), indent=0)
    print(len(tags), "tagged;", sum(v is None for v in tags.values()), "unparsed")
