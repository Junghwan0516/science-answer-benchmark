# SQL and data-quality report

This is a local training-data warehouse, not a production deployment. [Executed SQL notebook](../../notebooks/02b_data_pipeline.ipynb) · [Schema](../../sql/01_schema.sql) · [Queries](../../sql/02_analytics.sql) · [Architecture](../../docs/data_engineering.md).

## Verified table grains

| Table | Rows | Meaning |
|---|---:|---|
| domains | 12 | Original observed domain codes |
| labels | 5 | Original class labels |
| questions | 135 | Question/reference contexts |
| answers | 4,969 | Student-answer records |
| answer_features | 4,969 | One derived feature row per answer |
| development_assignments | 4,969 | Frozen v1 development partition per answer |

## Portable SQL outputs

- [Question metrics](v_question_metrics.csv)
- [Domain summaries](v_domain_summary.csv)
- [Label summaries](v_label_summary.csv)
- [Development partition summaries](v_partition_summary.csv)
- [Twelve quality checks](quality_report.json)
- [Input hashes and transformation lineage](lineage.json)

Observed correct-label shares are corpus composition statistics, not student ability, question difficulty, or model accuracy. Foreign keys, primary keys, feature bounds, context consistency, one-to-one joins, and SQL/Pandas count reconciliation are verified. Integration tests also inject invalid labels/orphans, feature corruption, and a failed warehouse write. A failed warehouse write preserves the previously accepted database.

Run `python scripts/run_pipeline.py --stage advanced` to rebuild local `data/warehouse/scientsbank.sqlite`; the generated binary is excluded from Git. Source data and data-derived outputs retain CC BY-SA 3.0 attribution to Dzikovska et al. (2013), https://aclanthology.org/S13-2045/.
