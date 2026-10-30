# Named gap list from the distance score (ood-03, written 2026-10-24 before any naming or coverage call)

Goal: the proposal's deliverable (i), Stefan's "you forgot X" list. For the distance score's top topics: a category name a lab would recognise, 3 example prompts, the closest Dolci prompts, and a coverage call. Output `notes/gap_list_2026-10-24.md`, readable in 20 minutes.

Topics: the top 60 by distance (`runs/rare_gaps/topics.csv`, rare_rank 1-60, all >= 50 members). No walking further down the ranking. Excluded topics are listed with their reason. Per topic, 20 random members (seed 0) are drawn for the checks:
- mostly NAME_k: >= half of the 20 contain `NAME_\d` (a known dataset artefact, already fixed in the NAME_1 case);
- mostly template: >= half of the 20's pairs are near-copies (60-character shared prefix or difflib >= 0.8, as in `scripts/label_templates.py`);
- mostly verbatim Dolci: >= half of the 20 are exact Dolci members (traffic_hash in mix_hash). The distance score masks exact copies, so such topics are covered by definition.

Evidence per kept topic: 10 of the 20 members (seed 0), up to 600 characters each. Each comes with its 3 MiniLM-nearest Dolci prompts (`one_sided.npz`), its 3 lexical-nearest (the review session's settings: tf-idf word 1-2 grams, min_df 3, 800k features, all 1,710,578 Dolci first turns), and a note when the prompt is itself in Dolci verbatim. Neighbours cut to 300 characters.

Astra (gpt-6-astra via codex, read-only), blind to rank and score: topics shuffled (seed 0), opaque ids, 6 batches of 10 topics. Per topic it gives:
- name: one line, the kind of request, as a lab's data team would name it;
- style: yes if the topic is defined by register, language or form rather than task (Stefan's example: colloquial vs formal text);
- matched: how many of the 10 prompts have a same-kind request among their neighbours (same task and subject; an exact copy counts);
- coverage: missing if matched <= 2, thin if 3-6, covered if >= 7 (the review session's thresholds). Astra gives matched; the call is computed from it.
- one sentence of reason.
Merge: one more Astra call with all kept names plus 2 example prompts each, asked to group names that describe the same category. Groups are shown as one entry with all their ranks.

Shown in the list: name, ranks, members, coverage (and matched/10), verbatim-member share, the user's blind label where the topic was labelled, and 3 examples chosen at random from the 10 (seed 0, not picked), each with its closest MiniLM and lexical Dolci prompt.

Precision, stated at the top: share of kept topics called missing, and share called missing or thin. Agreement of the coverage call with the user's real_gap labels on the overlap (distance top 60 ∩ labelled topics) is context only; the overlap is small.

What would change the plan:
- Missing share >= 0.5 of kept topics (Stefan's precision bar): the list goes to Stefan/Neel as a list of gaps.
- Missing share < 0.5 but missing + thin >= 0.5: present it as "thinly covered kinds of request", not "you forgot X". This is the expected case, given the review's lexical result (11 of 17 labelled distance real gaps had same-kind matches for at least half their prompts) and the flat behaviour results.
- Missing + thin < 0.5: distance's top topics are mostly covered. The list is not a deliverable; say so.

Main objection, known now: the coverage call depends on what 6 retrievals per prompt over 1.7M prompts find. A missed match turns covered into thin or missing. Astra's call is one model's reading (its 3-way agreement with the user was kappa 0.37 on the labelled screens). And distance topics showed no refusal effect and a +0.04 quality effect, so "thin" here is not shown to matter for Olmo.

Cost: 7 codex calls (about 60k tokens per batch, my estimate). CPU: tf-idf fit about 10-20 min.
Files: `runs/judge/gaplist/` (topics.json evidence, batches/, astra_raw/, merged.json), `scripts/gaplist.py`.
