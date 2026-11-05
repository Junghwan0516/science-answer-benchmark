# Data sources and redistribution

Source: **SemEval-2013 Task 7, SciEntsBank**, originally collected by Myroslava Dzikovska (University of Edinburgh) and Rodney Nielsen (University of North Texas).

- Author-hosted release: https://github.com/myrosia/semeval-2013-task7
- Original five-way archive: https://raw.githubusercontent.com/myrosia/semeval-2013-task7/main/semeval-5way.zip
- Paper: Dzikovska, M. O., Nielsen, R., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., Clark, P., Dagan, I., and Dang, H. T. (2013). *SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge*. Proceedings of SemEval 2013, pp. 263–274. https://aclanthology.org/S13-2045/
- Dataset license: **Creative Commons Attribution-ShareAlike 3.0 Unported**, https://creativecommons.org/licenses/by-sa/3.0/. Full original license: [DATA_LICENSE.txt](../DATA_LICENSE.txt).
- Original attribution statement: [SOURCE_README.md](../SOURCE_README.md).

## What is bundled

`raw/scientsbank_core.zip` contains **331 XML files** selected from the original archive. It includes SciEntsBank `Core` for all four official splits. Individual XML bytes are unchanged. This is a selected subset of the source archive, not a new corpus or an additional set of annotations. BEETLE, Extra, parser outputs, and reliability-annotation material are excluded from this project.

The four `processed/*.csv` files flatten the Core XML into one row per student-answer record. Text and labels are preserved as parsed, with surrounding whitespace stripped. Added derived columns include word counts, an explicit lexical overlap score, a negation marker, and a normalized input hash. Original labels are never silently merged, corrected, or deduplicated. These exports and data-derived reports/figures are distributed under **CC BY-SA 3.0** with the above attribution. No endorsement by the dataset authors is implied.

| File | Answer records | Questions | Domain codes |
|---|---:|---:|---:|
| `processed/train.csv` | 4,969 | 135 | 12 |
| `processed/test-unseen-answers.csv` | 540 | 135 | 12 |
| `processed/test-unseen-questions.csv` | 733 | 15 | 11 |
| `processed/test-unseen-domains.csv` | 4,562 | 46 | 3 |

UA questions overlap the training questions. There are 196 unique questions and 10,804 answer records across the four splits. Counts refer to records, not unique learners. UA/UQ/UD are preserved as official evaluation partitions. Exploratory figures in this release use **training only**; the other splits are exported and subjected only to inventory and normalized-input overlap checks.

## Dictionary

| Column | Meaning |
|---|---|
| `official_split` | Original train/UA/UQ/UD directory name |
| `question_id` | Source question identifier; grouping key, not a model feature |
| `domain` | Original science-module code; no guessed expansion |
| `answer_id` | Source answer identifier; not assumed to be a learner identifier |
| `sample_id` | Split-qualified answer identifier |
| `source_file` | XML member path in the archive |
| `question_text` | Question text |
| `reference_answer` | Provided reference answer; one per question in this release |
| `student_answer` | Observed student response |
| `label` | Original five-way annotation |
| `normalized_input_hash` | SHA-256 over the normalized question/reference/answer tuple |
| `answer_word_count` | Number of whitespace-separated words in the response |
| `reference_word_count` | Number of whitespace-separated words in the reference |
| `question_word_count` | Number of whitespace-separated words in the question |
| `reference_token_jaccard` | Intersection / union of lower-case regex word sets, including stop words |
| `has_negation` | Whether a listed negation marker occurs; not semantic contradiction |

Input-hash normalization uses Unicode NFKC, case folding, and collapsing whitespace. It preserves punctuation and word order. Raw text remains available. `provenance.json` records archive hashes and selection criteria. `checksums.sha256` covers the uploaded data and analysis files.

## Labels

`correct`, `partially_correct_incomplete`, `contradictory`, `irrelevant`, `non_domain`.

These are corpus annotations in the context of a question and reference. They are not diagnoses of a learner's ability. Annotation conflicts and low-support classes must be considered when interpreting future model results.
