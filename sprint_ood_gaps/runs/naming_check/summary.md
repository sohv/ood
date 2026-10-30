# Naming check on Sohan's removal tests (removal_code_multilingual), 2026-10-27 MT

Source: detectood/ood results/raw/261028_exp84_removal_{multilingual,coding}_v1/decision.json, field count_entries_named (namer anthropic/claude-sonnet-4.6, 25 prompts per topic). These are the topics that entered the count's top list after the removal. `target` = the kind grader's label (non-English or coding; stored under the key is_harmful in that file). The loop's confirm rule is shared >= 13 of 25 (exp41, exp57, exp83).

## multilingual (23 new entries)

| name | shared/25 | target | confirmed (shared>=13) |
|---|---|---|---|
| Korean language messages | 10 | True | False |
| French language messages | 5 | True | False |
| "Who is" identity questions | 25 | True | True |
| Italian language messages | 6 | True | False |
| Spanish language messages | 8 | True | False |
| Chinese language messages | 5 | True | False |
| Random/gibberish keyboard inputs | 9 | False | False |
| Family relationships and kinship | 16 | True | True |
| Apollo 11 JSON people and places | 13 | False | True |
| Italian language messages | 5 | True | False |
| Vague or incomplete questions | 10 | True | False |
| "What is X" questions | 25 | True | True |
| Italian language messages | 6 | True | False |
| Portuguese language messages | 14 | True | True |
| Cooking and food preparation | 8 | True | False |
| Brazil-related topics | 21 | True | True |
| Hindi/Urdu language messages | 4 | False | False |
| Messages written in Italian | 5 | True | False |
| Military personnel information extraction | 14 | False | True |
| Asking if AI knows something | 18 | True | True |
| AI teacher evaluating student answers | 14 | False | True |
| Basic arithmetic calculations | 13 | False | True |
| Poem writing requests | 25 | False | True |

- target entries: 16; confirmed by the namer rule: 6

## coding (11 new entries)

| name | shared/25 | target | confirmed (shared>=13) |
|---|---|---|---|
| Python Fibonacci programming tasks | 15 | True | True |
| AI intent classification and API routing | 17 | True | True |
| List generation requests | 14 | False | True |
| Unit test generation for code | 22 | True | True |
| Positive or negative sentiment labeling | 14 | False | True |
| Python coding requests | 25 | True | True |
| Named Entity Recognition (NER) tasks | 20 | False | True |
| Medical/Clinical Healthcare Interactions | 22 | False | True |
| Conversational AI message handling | 19 | False | True |
| Golang programming tasks and questions | 25 | True | True |
| Python programming tasks and questions | 25 | True | True |

- target entries: 6; confirmed by the namer rule: 6

## Why the language-named topics fail the confirm rule (read of the kind grader's 10 sampled prompts per topic, removal_code_multilingual_kind_labels_v1/grades.jsonl)

- The namer prompt already counts "a single language" as a shared group, so the low counts are not a prompt rule.
- The topics are mixed-language bins of short messages. c28 "Korean language messages": mostly Korean, plus Arabic, Hebrew, braille art. c624 "French": French, Spanish, Portuguese, Arabic, English. c1206 "Spanish": Italian, Czech, Chinese, German, Portuguese, Korean, Spanish. c1042 "Italian": short fragments in several languages.
- Likely cause: paraphrase-multilingual-MiniLM places same-meaning text together across languages, so k-means does not separate languages; short non-English messages land in mixed bins. The namer's largest single language is then 5-10 of 25.
- Reading: the ranking does rise on non-English traffic after the removal, but the topic list cannot state it as "add language X data". A by-language view (dataset language field) would be the direct way to report language gaps.
