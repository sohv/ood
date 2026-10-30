# Part 10: an emergent taxonomy from request descriptions (29 Oct 2026 MT)

Design: `designs/gap_taxonomy.md`, Part 10 (written before any call; run as written). Script: `scripts/gap_emergent.py`. Figure: `figs/gap_taxonomy/emergent_nodes_traffic_vs_dolci.png`.

Setup. The prompts are part 8's sample. 3,980 of its 4,000 prompts were described (1,992 traffic, 1,988 Dolci); one batch of 20 failed twice and is left out. Claude Sonnet 5.5 wrote one or two sentences per prompt on what kind of request it is: task, form, and what makes it hard. It was told not to state the subject. Calls had 20 prompts each, source hidden. The descriptions were embedded with multilingual MiniLM (the loop's embedder), clustered by k-means into 100 fine nodes, then merged by Ward into 20 coarse nodes. Each node was named by Sonnet from 20 random member descriptions. The baseline is the same clustering on the raw prompt text, with nodes named from the same descriptions. Traffic-heavy node (fixed before the run, a guess): at least 20 traffic prompts and a Wilson lower bound on the traffic share of at least 0.75.

1. Describing first separates traffic from Dolci a little better than raw text, but source is still mostly not predicted by kind of request.
   - Adjusted mutual information between fine node and source: descriptions 0.091, raw text 0.061. Difference 0.031, prompt bootstrap 0.025 to 0.037. k-means seeds 0 to 2 give 0.090 to 0.091 and 0.057 to 0.061.
   - Most of the separation is on the Dolci side. Descriptions give 20 Dolci-heavy fine nodes and 10 traffic-heavy ones. The Dolci-heavy nodes are synthetic kinds: math word problems with format constraints, stdin/stdout programming specs, logic and graph puzzles, tool-call lookups by ID.

2. 10 of the 100 description nodes are traffic-heavy and hold 20.1% of traffic (raw text: 9 nodes, 16.8%).
   - Four are LMSYS templates that the raw-text baseline also finds: minimum-length company introductions (49 traffic, 0 Dolci), long chemical-industry articles (39/0), a text-completion wrapper around tutorials (35/0), and rigid image-prompt templates (37/2).
   - Six are kinds the baseline does not separate: roleplay persona setups with NAME_k placeholders (31/0), greetings and small talk (31/1), Russian-language requests (42/4), terse fragments with unclear intent (39/5), sexual or fetish creative writing (42/6), and short questions about the assistant's abilities or one-line openers (55/9).
   - Of the traffic prompts in description-arm traffic-heavy nodes, 48% are also in baseline traffic-heavy nodes.
   - Five random members per node (seed 0, not chosen): `examples.md`. Full ranked tables for both arms and both levels: `tables.md`.

3. The kinds that only describing finds are the least stable ones.
   - With k-means seeds 1 and 2, 6 of the 10 traffic-heavy nodes have a best match that is traffic-heavy at both seeds (Jaccard 0.33 to 1.0). These six are the four templates, roleplay persona setups and greetings.
   - Russian-language requests and sexual creative writing are not traffic-heavy at either other seed (Jaccard 0.25 to 0.43). Terse fragments and capability/opener questions are traffic-heavy at one of the two.
   - Re-describing 200 random prompts in a different batch order puts 71% in the same fine node and 83% in the same coarse node.
   - At the coarse level only 2 of 20 nodes are traffic-heavy: roleplay and persona setups, many of them jailbreaks (103/9), and length-targeted business and chemical writing (102/12, the templates).

4. MiniLM distance does not rank these kinds high.
   - The median distance to the nearest Dolci prompt for traffic members of the 10 traffic-heavy nodes ranges from 0.03 to 0.32. The median for all traffic is 0.290; 7 of the 10 nodes are below it, and the other 3 are at 0.293 to 0.323.
   - The templates have the lowest distances (0.03 to 0.24), although Dolci has almost none of them: Dolci holds a few near-copies, so distance is small while the sample has 35 to 49 traffic prompts and 0 to 2 Dolci prompts in each.

5. The loop's confirmed gaps fall in the description arm's traffic-heavy kinds less often than traffic overall.
   - 560 prompts (10 per topic) from the 56 confirmed topics and 547 from 56 random rejected topics were described and assigned to the nearest fine node.
   - Description arm: confirmed 13.8%, rejected 8.0%, traffic sample 20.1%. Confirmed minus rejected +5.8 points, topic bootstrap -0.2 to +12.6.
   - Raw-text arm: confirmed 27.3%, rejected 10.2%, traffic sample 16.8%. Confirmed minus rejected +17.3 points, 6.8 to 28.2. The confirmed prompts here land in the NAME_k placeholder nodes (125 of 560) and in the summary-consistency template (24).
   - In the description arm, confirmed prompts land in sexual creative writing (26), roleplay persona setups (19) and capability/opener questions (19). Almost none land in the four templates (2 of 560).
   - Limit: map-back calls held only traffic prompts, while the sample's calls mixed traffic and Dolci. So the comparison with the traffic sample may carry a context effect. Confirmed vs rejected does not have this problem.

What it is taken to mean. Grouping by an LLM's description of the request, with no category list given, recovers two kinds of traffic-over-Dolci difference. The first is high-volume LMSYS templates, which raw text finds too. The second is chat-register kinds that raw text spreads across topics: greetings, persona setups, openers about the assistant, terse fragments. Every kind found is already on part 8's fixed list (non-English, roleplay setup, explicit sexual, terse or underspecified, asks about the assistant, templated item), so the emergent pass does not surface a kind the list missed. None of the 100 nodes is defined by hidden difficulty, pressure on rules (apart from jailbreak-style roleplay) or long context. The loop's confirmed gaps sit with the placeholder artefacts in raw-text space, and they are not concentrated in the kind-level gaps. Main objections: the effect on mutual information is small; the new kinds are the seed-unstable ones; one describer model; 20 prompts per node at 1% prevalence; first turn only, 2,000 characters.
