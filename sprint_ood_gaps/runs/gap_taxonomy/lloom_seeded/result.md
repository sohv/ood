# Part 12, LLooM seeded variant: result (30 Oct 2026, 00:15 MT)

Design: `designs/gap_taxonomy.md`, Part 12 "LLooM seeded variant". It was chosen after seeing the default arm's all-subject concepts, so it is exploratory. LLooM 0.8.2.1, unchanged; gpt-4o-mini / gpt-4o / text-embedding-3-large via OpenRouter. Same 1,000-prompt induction set as the default arm. Seed phrase: "the kind of request (the task, its form, or what makes it hard to answer), not its subject". Scored on all 4,000 sample prompts: 43,999 of 44,000 prompt-concept scores (one lost to LLooM's JSON parse errors). A concept applies at score 1.0. Cost $3.97 (`calls.jsonl`, list prices).

1. Seeding changes what LLooM describes: the 11 concepts are kinds of request, not subjects.
   - Long-form with word counts, multiple conditions, data-structure manipulation, optimisation, iteration, conditional logic, creative writing, clarity of request, comprehensive understanding, technical complexity, structured communication.
   - None is about harmful requests or non-English text.

2. Only 2 of the 11 concepts are more common in traffic than in Dolci, so every arm's top 10 has 2 entries, the same 2 in every arm.
   - Full Dolci: Creative Writing Challenge 26.2% of traffic vs 13.1% of Dolci; Lengthy Content Creation 14.3% vs 10.2%.
   - The other 9 are Dolci-heavy by 14 to 34 points, e.g. Iterative Process 8.5% vs 42.1% and Conditional Logic 13.3% vs 46.0%. The seeded concepts describe Dolci's synthetic, constrained tasks better than they describe traffic.
   - 5 random traffic positives per concept (seed 0): Creative Writing picks up roleplay setups, including sexual roleplay, and fan-fiction; Lengthy Content picks up the LMSYS "article about X in the chemical industry" and "company introduction of N words" templates.

3. By LLooM's own scores, the code removal is visible, the other two are not.
   - Code: after the code removal, the Dolci share of Data Structure Manipulation is 12.0 points lower than after the random removal (18.3% vs 30.4%), and of Technical Complexity 11.2 points lower. In the traffic-minus-Dolci ranking they move from 7 to 5 and from 6 to 4. Both stay Dolci-heavy, so neither enters the traffic-over-Dolci top list.
   - Safety and non-English: no concept's Dolci share moves by more than 1.6 points versus the random removal.

4. Expected judge outcome: recovery 0 for all three families. The top lists are identical across arms, so minus and random get the same count. The judge pass is research-9e's.

What it is taken to mean. Telling LLooM to look for kinds of request removes the subject bias, but its induction then describes the training side's task structure. A planted removal of code shows up as a smaller Dolci share for code-like concepts, not as a traffic-over-Dolci concept. This agrees with part 16: a frequency ranking is dominated by differences that were already there. Main objections: one seed phrase, written after seeing the default arm; induction on 1,000 prompts; one random removal per family.

Files: `concepts.json`, `summary.json` (per-arm prevalence with Wilson intervals; `shift_<family>`), `top10.json` (judge format), `top10_detail.json`, `top10.md`, `score_df.pkl`.
