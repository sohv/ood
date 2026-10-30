# Gap taxonomy tables

Defining property of each topic (unprimed Sonnet label on 10 fresh member prompts), share of topics per group.

| group | n | subject | task type | language | format or template | length or structure | persona or roleplay | adversarial or jailbreak | tone or register | placeholder or artefact | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full: confirmed | 56 | 0.36 | 0.20 | 0.00 | 0.20 | 0.00 | 0.02 | 0.04 | 0.00 | 0.18 | 0.02 |
| full: rejected | 143 | 0.51 | 0.15 | 0.00 | 0.10 | 0.04 | 0.01 | 0.02 | 0.02 | 0.08 | 0.06 |
| new after safety removal | 20 | 0.60 | 0.00 | 0.00 | 0.20 | 0.05 | 0.00 | 0.05 | 0.00 | 0.10 | 0.00 |
| new after coding removal | 5 | 0.00 | 0.40 | 0.00 | 0.20 | 0.20 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 |
| new after non-English removal | 11 | 0.36 | 0.18 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.27 | 0.09 |
| new after random removals (pooled) | 62 | 0.56 | 0.06 | 0.00 | 0.13 | 0.05 | 0.02 | 0.03 | 0.03 | 0.08 | 0.03 |

Name describes the defining property (full run, 199 topic-name pairs): {'partly': 81, 'yes': 64, 'no': 54}
Axis of the loop's names: {'subject': 158, 'task type': 34, 'persona or roleplay': 2, 'format or template': 1, 'placeholder or artefact': 1, 'adversarial or jailbreak': 2, 'language': 1}
Non-subject properties shared by most or all prompts (any topic, not only defining): {'subject': 309, 'task type': 328, 'language': 184, 'format or template': 155, 'length or structure': 270, 'persona or roleplay': 37, 'adversarial or jailbreak': 31, 'tone or register': 284, 'placeholder or artefact': 66, 'other': 41}
Sole vs Sonnet on the defining axis: {'n': 60, 'kappa_axis': (0.46781467814678146, 0.6271186440677966), 'kappa_subject_vs_not': (0.5025295109612142, 0.7457627118644068)}
Second sample vs first: {'n': 40, 'kappa_axis': (0.4508670520231214, 0.6052631578947368), 'kappa_subject_vs_not': (0.4736842105263157, 0.7368421052631579)}
