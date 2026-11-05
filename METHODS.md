# Analysis methods and interpretation limits

## Unit of observation

One student-answer record to one science question. An answer identifier is not treated as a verified learner identifier. Domain codes come directly from XML metadata. Question and reference text are repeated across answer records by design; this repetition alone is not a duplicate answer.

## Scope

The source is the English SciEntsBank five-way Core release. Descriptive statistics and 14 figures use only the 4,969 official training answers. Official test partitions are included as separate CSV exports, with inventory and complete-input overlap checks. No model is trained, no test predictions are generated, and no model selection is performed.

## Data quality

For seven source fields, report null and whitespace-only counts separately. Check split sizes, expected labels, uniqueness of split-qualified answer IDs, and one reference per XML question. For complete-input duplicates, normalize the question, reference, and response with NFKC, casefolding, and whitespace collapsing, serialize the triple as JSON, then compute SHA-256. This conservative comparison does not identify every paraphrase. Original rows and labels remain intact.

- A duplicate group has at least two records with the same complete normalized input.
- Excess rows equal group size minus one, summed across repeated groups.
- A conflicting group contains more than one original label for the same complete normalized input.
- Test overlap counts are test rows with a complete-input hash appearing in training, not shared learner IDs.

Conflicting groups are flagged for review rather than relabelled. Context required by annotators may be incompletely represented in the exported text. We do not infer that each conflict is a confirmed annotation error.

## Descriptive features

Lengths count whitespace-separated words. Jaccard overlap uses sets of lower-case tokens matching `[a-z]+(?:'[a-z]+)?`, retaining stop words and discarding repeated occurrences. It is lexical, not a semantic evaluation. The negation indicator checks: `not`, `no`, `never`, `cannot`, `can't`, `don't`, `doesn't`, `isn't`, `aren't`, `won't`. It is an intentionally incomplete lexical flag, not a contradiction detector.

Label entropy is `-sum(p * log2(p))` over nonzero within-question class proportions. It measures observed annotation diversity. It is not a calibrated difficulty or understanding measure. Domain heatmaps use either counts or within-domain percentages; denominators are explicitly labelled.

The answer-weighted distribution weights every record equally. The question-weighted distribution averages per-question proportions, giving each question equal weight. Boxplots use standard 1.5-IQR whiskers; outliers are kept in the data even where their points are hidden for readability. The duplicate-frequency plot explicitly labels its logarithmic y-axis. No causal or significance claims are made from exploratory visual differences.

## Limits

The public historical English corpus does not establish suitability for Korean answers, essays, modern curricula, unseen populations, or automated educational decisions. Only 23 training responses are non-domain; small supports cause unstable future estimates. Many records share a question, so answer-level independence should not be assumed for inferential analysis. Official UQ has only 15 questions and no non-domain reference labels. UD contains only three held-out domain codes. Pretraining exposure of any future language model to this public corpus would be unknown without a separate audit.

This release supports dataset understanding. Teacher time savings, learning gains, grading reliability, and production performance have not been measured.
