# Question-grouped development protocol v1

## Purpose and source

Use the 4,969 official SciEntsBank training answer records to prepare model development. One question and all of its answer records must stay in one internal partition. This step does not train a model and does not load the official UA, UQ, or UD CSVs. Those files remain unchanged. Previous inventory and exact-input overlap audits are already documented; this protocol does not claim that their metadata were never inspected.

## Reproducible allocation rule

1. Sort the 135 question IDs lexically.
2. Initialize NumPy `default_rng(42)` (PCG64).
3. Permute the IDs and allocate the first 95 questions to fit, the next 20 to development, and the remaining 20 to calibration.
4. Accept the **first** candidate satisfying all constraints: at least 10 records of every class in fit; at least 3 records of every class in each of development and calibration; each held-out internal partition contains 12–18% of all answer records.
5. Try at most 1,000 candidates; stop with an error if none is eligible. Do not choose the candidate with the best model score.

Candidate 5 is the first eligible allocation. Candidate acceptance uses only official training IDs, record counts, and labels. No model performance, official test labels, or text-feature distributions enter the choice. This is constrained allocation, not unconstrained random sampling. Question counts are fixed; row counts are allowed to vary because question sizes differ. The initial seed-42 candidate left only one non-domain record in calibration, which motivated explicit label-support requirements. The minimum of three is a practical coverage constraint, not evidence of statistical reliability.

| Internal partition | Questions | Answers | Non-domain answers |
|---|---:|---:|---:|
| Fit | 95 | 3,536 | 17 |
| Development | 20 | 715 | 3 |
| Calibration | 20 | 718 | 3 |

All 12 training domain codes occur in fit, 9 in development, and 10 in calibration. Domain coverage was observed after allocation, not imposed as an eligibility criterion. Only two questions exist for some domains, so all-domain coverage in all three partitions is not generally feasible.

## What each partition may influence

- **Fit:** learn token vocabularies, scalers, model parameters, and any training-only reweighting.
- **Development:** select feature representations, models, hyperparameters, and stopping criteria. Primary model selection will use fixed-five-class macro-F1 with all class supports reported. Accuracy, weighted-F1, and per-class metrics are complementary.
- **Calibration:** after model selection, estimate a shared probability-calibration transform and/or set a review threshold. Do not tune classifier parameters on this partition. With three non-domain records, class-specific calibration is not supported. Any threshold is subsequently assessed on independent evaluation data.
- **Official UA/UQ/UD:** final evaluation after the model and final-fit procedure are frozen. No threshold or hyperparameter search on these scores.

## Duplicates and conflicting labels

The primary policy retains all original records and labels with unit sample weights. No duplicate is automatically removed and no conflict is resolved by majority voting. Since complete questions are grouped, repeated inputs within a question cannot cross partitions. The three pairwise checks report zero overlap for answer IDs, question IDs, and normalized complete inputs. This verifies those particular keys, not every possible paraphrase or learner identity.

The six previously flagged conflicting groups (15 rows) happen to fall in fit under this allocation. This was not a selection constraint. Future work may add a documented manual annotation review, but this version does not claim to have completed or resolved it.

A predeclared sensitivity experiment may give each complete normalized input equal total fit weight: each fit row receives `1 / input_multiplicity`. The optional `sensitivity_fit_weight` column prepares those weights, including competing labels in conflicting groups. Primary weights remain one. Development and calibration observations must not be reweighted or filtered using model errors; their sensitivity-weight fields are empty. This study design does not perform the weighting experiment yet.

## Assignment files and checks

`data/splits/v1/question_assignments.csv` records question IDs, domain codes, answer counts, and internal partitions. `answer_assignments.csv` records each answer's partition and duplicate/conflict flags; join to `data/processed/train.csv` using `sample_id` with one-to-one validation. `protocol.json` records input and assignment SHA-256 hashes, constraints, seed, and accepted candidate. Original answer text is not duplicated into the assignment files.

Rebuild with `python scripts/build_splits.py`. Run `python -m unittest discover -s tests -v`. Tests reject changed labels, repeated/missing answers, questions crossing partitions, and official test records entering split generation; they also verify deterministic class support and overlap checks.

## Important final-fit decision still pending

**This v1 split is a development design, not a finalized full-data benchmark training recipe.** Official UA means questions present in the original official training partition. A classifier fitted to only the 95 fit questions would have seen only some of those UA question IDs. Its official-UA result must therefore identify the actual seen-question and held-out-question slices; it cannot be described as full-training, seen-question benchmark performance.

Before any official test scoring, specify either a final full-training refit with a separately documented cross-fitted calibration procedure, or retain the held-out calibration design and disclose the actual question coverage of the fitted classifier. Do not refit on calibration labels and continue to describe the same predictions as independent calibration. This decision is deliberately left for the model-development stage, before official test evaluation.

## Limits

Three rare-class records per held-out partition still yield unstable precision, recall, and probability estimates. There is no verified learner identifier, so learner-disjointness is not established. Domain codes overlap across partitions, so this is not an unseen-domain development test. Source data and data-derived assignment tables and figures retain CC BY-SA 3.0 attribution to Dzikovska et al. (2013), https://aclanthology.org/S13-2045/.
