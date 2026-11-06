# Advanced descriptive analysis: estimands, dependence, and text features

## Scope

This extension adds training-only statistical diagnostics and a relational batch-data pipeline. The source observations are unchanged: no extra students, labels, synthetic records, or model results are invented. The original four official CSV partitions, 331 selected source XML files, and v1 development assignments remain intact. The descriptive analysis covers the previously inspected official training corpus; it is not a blind assessment of development or calibration data.

## Question-level resampling

For each of 2,000 replicates, sample 135 question IDs with replacement and include all answer records in each sampled question. Sum the class counts across selected question clusters and divide by the total number of sampled answer records. The point estimate pools observed answer records. The 2.5th and 97.5th percentiles summarize resampling stability. The independent-row comparison draws class counts from a multinomial distribution with the empirical class proportions and 4,969 trials. It ignores within-question dependence and is shown only as a diagnostic comparison. It is not the primary interval.

Domain intervals resample the questions within that observed domain. The plot suppresses intervals when fewer than five questions are present; numeric results remain in the supporting table. The short-answer analysis conditions on a label, selects questions contributing that label, and resamples their label-specific answer records. Its threshold, at most five whitespace-separated words, is an exploratory slice rather than a decision rule.

These empirical intervals do not establish coverage for future students, domains, or curricula. Questions are the resampling units, not verified learner IDs. The corpus and observed question pool may be unrepresentative, and resampling cannot repair annotation errors or selection bias. No independence-based p-values or causal claims are reported.

## Three weighting targets

- Answer weighting: pool all 4,969 answer records.
- Question weighting: average within-question class proportions across 135 questions.
- Domain weighting: average within-domain class proportions across 12 observed domain codes.

Each target gives a different unit equal weight. Domain weighting here does not estimate performance on future domains.

## Feature definitions

`data/processed/training_features.csv` has 4,969 rows and 12 columns: `sample_id`, ten numeric/boolean text features, and `raw_input_hash`. It does not duplicate response text. Join to `train.csv` one-to-one by `sample_id`.

| Feature | Definition and limit |
|---|---|
| `answer_char_count` | Number of characters in the parsed response |
| `answer_word_count` | Number of whitespace-separated words; reconciled to the original export |
| `lexical_token_count` | Lower-case matches of `[a-z]+(?:'[a-z]+)?`; numbers are excluded |
| `unique_word_count` | Distinct lexical tokens |
| `type_token_ratio` | Distinct tokens / token occurrences; zero when no lexical tokens occur; length-dependent |
| `question_token_jaccard` | Set intersection / union against question tokens; no fitted vocabulary |
| `reference_token_jaccard` | Same against reference tokens; reconciled to the original export |
| `reference_length_ratio` | Whitespace answer words / reference words |
| `punctuation_fraction` | Characters that are neither alphanumeric nor whitespace / answer characters |
| `has_numeric_content` | Whether any digit occurs; not a measure of scientific correctness |

Spearman correlations expose redundant numeric features. Class labels are nominal and are not encoded as numeric ranks. Overlap scatter plots compare question wording and reference wording without fitting a classifier.

## Fit-to-development vocabulary novelty

Build a lower-case lexical vocabulary from the 3,536 fit responses only. Count unseen token occurrences in the 715 development responses, with repeated occurrences retained in the denominator. There are 2,041 fit vocabulary types. Of 7,565 development token occurrences, 343 (about 4.53%) are absent from that vocabulary. Domain plots use each development domain's token-occurrence denominator. Calibration vocabulary and official test vocabulary are not analyzed. An unseen word is not necessarily an out-of-domain concept, a misspelling, or a prediction error.

## Exact raw-input label conflicts

For every exactly identical parsed question/reference/response triple, let n be the number of records and m the largest class count. Any deterministic prediction restricted to this complete raw input must disagree with at least n-m of those recorded labels. Summed across training groups, this equals six records out of 4,969 (about 0.121%). There are 4,827 raw-input groups and six raw conflicting groups. Conservative normalized-input groups remain 4,826, as in the original audit.

This is an empirical training-label consistency constraint. It is not population Bayes error, a generalization bound, an accuracy estimate for a trained model, or a claim that all annotations can be resolved from the exported text. Original labels are preserved.

## Reproducibility and sources

Seed: 2026; replicates: 2,000. `reports/advanced/analysis_manifest.json` records the source/assignment hashes, scope, and vocabulary counts. Supporting tables preserve denominators and class/question support. Source data and data-derived outputs: CC BY-SA 3.0; Dzikovska et al. (2013), https://aclanthology.org/S13-2045/. Code and data licensing remain separate.
