# Reproducible batch data pipeline

## Implemented capabilities

This project now provides a local batch pipeline: source Core XML to official CSV exports, versioned development assignments, derived features, normalized SQLite tables, SQL views, reconciled reports, and executed notebooks. The engineering scope is local and reproducible; no cloud deployment, production SLA, streaming ingestion, or distributed processing is claimed.

| Stage | Implementation | Portable output |
|---|---|---|
| Source ingestion | `scripts/build_analysis.py` | Official partition CSVs and original audit |
| Development allocation | `scripts/build_splits.py` | Immutable v1 IDs, protocol and overlap checks |
| Features and statistics | `scripts/analysis_extensions.py` | Feature CSV, resampling tables, seven figures |
| Warehouse | `scripts/build_warehouse.py` | SQLite database, four SQL-report CSVs, quality and lineage JSON |
| Reporting | `scripts/prepare_extension_notebooks.py` | Two executed notebooks |
| Verification | `scripts/verify_release.py` and `tests/` | Automated checks and content hashes |

## Contract and relational design

`data/contracts/training_contract.json` specifies this corpus release's source counts, keys, allowed labels, text requirements, context consistency, and one-to-one coverage. `sql/01_schema.sql` enforces primary keys, foreign keys, nonempty source text, numeric ranges, and partition-specific weight constraints.

| Table | Grain | Key and relation |
|---|---|---|
| `domains` | One observed domain code | `domain` primary key |
| `labels` | One source label | `label` primary key |
| `questions` | One question and reference context | `question_id`; domain foreign key |
| `answers` | One training answer record | `sample_id`; question and label foreign keys |
| `answer_features` | One derived feature row per answer | `sample_id` primary/foreign key |
| `development_assignments` | One internal partition per answer | `sample_id` primary/foreign key |

The four views in `sql/02_analytics.sql` summarize questions, domains, labels, and partitions. Reports reconcile SQL label and question counts against Pandas source calculations. Feature ranges, word counts, and lexical-overlap values are reconciled before insertion. SQLite foreign-key and integrity checks run before a database is accepted.

## Safe rebuild and lineage

The warehouse is built at a temporary path. Only a successful validated build atomically replaces the local database. A failed build removes its temporary file and retains the previously accepted database. Input hashes cover the training CSV, features, assignments, contract, schema, and SQL definitions. The binary database is ignored by Git and rebuilt locally; schema, queries, manifests, tests, report CSVs, and executed notebook outputs are committed.

The builder reads official training data and v1 assignments. Original evaluation files are not used in advanced statistics or warehouse generation. A `full` pipeline rebuild may export and inventory all original official partitions through the existing source-audit stage, but still does not compute test model scores.

## Commands

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --stage advanced
python -m unittest discover -s tests -v
```

Use `--stage full` to rebuild the original source audit and v1 split before the extension. Checksum changes are never silently accepted: when intentionally updating artifacts, review the changes and run with `--refresh-checksums` to write the new release hashes after the checks succeed. Existing source and split rules remain pinned.

The GitHub Actions workflow runs the advanced build and checks on pushes and pull requests. A workflow definition is not evidence that a particular remote run passed; inspect its result separately.

## Connection to portfolio roles

- Data Analyst: explicit group/denominator comparisons, SQL aggregation, reconciled report tables, and interpretation.
- Data Scientist: question-cluster resampling, weighting estimands, feature diagnostics, label ambiguity, and restrained conclusions.
- Data Engineering: schema design, contracts, relational integrity, one-to-one joins, lineage, repeatable builds, atomic replacement, and tests.

These capabilities are backed by executable artifacts, not changes in job title. Prediction-model development remains a separate future stage.
