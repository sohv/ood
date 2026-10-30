# "{name}" and "<name>" in Dolci (2026-10-26, request from research-6d via research-01)

## Chain (written before running)
- Question: does Dolci pair the placeholders "{name}" and "<name>" with refusals, as it pairs NAME_k (145 of 256 refused)? Untrained Olmo refuses borderline prompts more with them than with "Sam" ({name} +0.062, <name> +0.079; STATUS "Natural placeholders"). If Dolci pairs them with refusals, they are learned cues like NAME_k; if not, the extra caution does not trace to such rows.
- Data: the trigger scan's table `runs/triggers/dolci_labelled` (1,710,578 unique first user turns, SFT then DPO-chosen, first assistant reply each). Same table as the NAME_k and template traces, so the numbers are comparable.
- Match: the literal strings "{name}" and "<name>", case-insensitive (so {Name}, <NAME> count), anywhere in the first user turn. Case-sensitive lowercase counts also reported. Other placeholder spellings ({user_name}, <your name>) are not counted; they are different tokens.
- Refusal: `scripts/refusal.py` on the first 300 characters of the reply, minus the scan's capability phrases (CAPABILITY), the same rule as the scan (checked on 100 Dolci replies, `spot_check_100.md`). Base rate 2.38% outside the Tool Use source.
- Split: "{name}" is a Python/format-string idiom, so code prompts are reported separately (by source: Evol CodeAlpaca, Python Algorithms, Tulu 3 Persona Python, and any prompt with a code fence, `def `, `print(` or `f"`). The placeholder use that matters for Olmo's caution is in natural-language prompts.
- Reading: 5 random rows per form (seed 0), more from refused rows if there are any, recorded below with source and refusal label.
- Why it measures the question: a learned cue needs rows where the placeholder co-occurs with refusals at well above the base rate; counts, refusal share by source, and the rows read show whether such rows exist. It cannot show that Olmo learned from them (that needs a minimal-pair or fine-tuning test), only whether the data offers the association.

## Measured (`scripts/trig_placeholders.py`; `matches.csv`, `summary.json`, `read_rows.md`)
| form | Dolci first turns (SFT / DPO) | refused by the scan's rule | after reading | main sources |
|---|---|---|---|---|
| {name} | 618 (615 / 3) | 2 (0.3%) | 2: both WildGuardMix (a phishing-script request; a request for a list of named campaign respondents) | Python Algorithms 436, WildChat 115, Evol CodeAlpaca 35, Tulu Persona Python 15, Precise IF 11 |
| <name> | 2,230 (2,216 / 14; 2,224 outside Tool Use) | 1 of 2,224 (0.04%) | 0: the one flagged reply is an AutoGPT JSON reply containing "I can't help but do a double take", a detector false positive | Python Algorithms 1,285, WildChat 798, Evol CodeAlpaca 96, Precise IF 16 |
Base rate for comparison: 2.38% (scan, Tool Use excluded). NAME_k: 145 of 256 (57%), 98% of those WildGuardMix.
- Case: 573 of 618 {name} and 2,148 of 2,230 <name> are the lowercase form.
- What the rows are: {name} is almost all format strings in code (543 flagged as code by source or markers; reading the 75 "non-code" rows, most are still code in JS/Java/MediaWiki, plus a few WildChat Russian dating-letter templates "Hi {Name}!"). <name> is code or markup (1,405 flagged as code) and 764 copies of one WildChat AutoGPT prompt ("CONSTRAINTS: ... <name>" as a command argument); the remaining ~55 are XML and template fields. Placeholder use in a natural-language request, as in the borderline prompts Olmo is cautious on, is rare in Dolci for both forms.
- My code split under-counts code (it misses JS, Java, XML); the refusal counts do not depend on it.

## Meaning
Dolci does not pair "{name}" or "<name>" with refusals: 2 of 618 and 0 of 2,230, both below the base rate, the opposite of NAME_k. So Olmo's extra caution with these placeholders (+0.062, +0.079 vs "Sam") does not trace to placeholder-with-refusal rows in Dolci at this resolution, as with [REDACTED] (43 rows, 0 refused). The data offer a different association instead: these forms appear mostly in code and agent templates, not in user requests about a person.
Main objection: the measure is co-occurrence in first turns; a cue could be learned from other turns, from WildGuardMix-style prompts that use other spellings, or from pretraining. Untrained Olmo's caution being shared by three unrelated placeholders ({name}, <name>, [REDACTED]) suggests a general response to placeholder-like text (an anonymised or templated request) rather than three separately learned cues; that is a reading, not tested here.
