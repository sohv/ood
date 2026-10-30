# Four-item plan for the drop-in (draft, 2026-10-15)

**Question.** Can a cheap detector pick out the deployment prompts that OLMo's post-training data never covered, once prompt length is controlled, and do those gaps line up with where the post-trained model behaves no better than its SFT base?

**What success looks like.** A short list of gap types found automatically with length controlled, each verified rare in the DPO data, and one test of whether high-scoring prompts are where DPO and SFT answers do not differ or where both are poor. Stefan's easy benchmark reproduced as the pipeline check.

**Key experiments.** (1) Easy benchmark replication. (2) Length control: residualise and stratify, see what survives. (3) Gap benchmark built before scoring: five prompt types, precision at k per detector. (4) Behaviour link: sample SFT and DPO on high- and low-score prompts, judge whether DPO improved on SFT. (5) If time: the flag-and-add loop, two rounds, read the flagged prompts.

**Uncertainties.** Whether anything but length survives the control. Whether "far from DPO data" predicts anything about behaviour. Which model pair is right (SFT vs DPO, or DPO vs the RL Instruct checkpoint). Whether the delta scores measure coverage or chat formatting.

**Ask for Neel.** Is the behaviour link the result he would want at the centre, or is a clean gap benchmark with length control enough on its own?
