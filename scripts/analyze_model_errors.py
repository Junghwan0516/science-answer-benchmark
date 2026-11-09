"""Audit a frozen v1 run without tuning on official evaluation labels."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import run_model_study as study
from science_models import AnswerClassifier, FIELDS, LABELS, infer

SOURCES = {'UA': 'test-unseen-answers.csv', 'UQ': 'test-unseen-questions.csv',
           'UD': 'test-unseen-domains.csv'}


def compare_by_question(frame, first, second, *, repeats=4000):
    """Paired bootstrap of question clusters, conditional on fixed fitted models."""
    # Aggregate confusion matrices before resampling so all answers to a question
    # stay together. The sampled unit is the same for both model variants.
    groups = frame.groupby('question_id', sort=True).indices.values()
    matrices = np.array([
        [confusion_matrix(frame.label.iloc[index], pred[index], labels=LABELS) for index in groups]
        for pred in (first, second)
    ])
    rng = np.random.default_rng(42)
    samples = rng.integers(0, matrices.shape[1], size=(repeats, matrices.shape[1]))
    counts = matrices[:, samples].sum(axis=2)
    denominator = counts.sum(axis=-1) + counts.sum(axis=-2)
    numerator = 2 * np.diagonal(counts, axis1=-2, axis2=-1)
    f1 = np.divide(numerator, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    contrad_total = counts[:, :, 3, :].sum(axis=-1)
    false_correct = counts[:, :, 3, 0] / np.maximum(contrad_total, 1)
    bounds = lambda values: np.quantile(values, [.025, .975]).tolist()
    return {'replicates': repeats,
            'pair_minus_concatenated_macro_f1_95_percentile': bounds((f1[1] - f1[0]).mean(axis=-1)),
            'pair_minus_concatenated_false_correct_rate_95_percentile': bounds(
                (false_correct[1] - false_correct[0])[contrad_total[0] > 0]),
            'interpretation': 'Descriptive sensitivity to question composition; does not account for model selection or fit variation.'}


def audit(run_dir):
    reports = run_dir / 'reports'
    card = json.loads((reports / 'model_card.json').read_text())
    artifact_path = run_dir / 'artifacts/science_answer_model.joblib'
    if (study.sha(artifact_path) != card['artifact_sha256'] or
            study.sha(Path(__file__).with_name('science_models.py')) != card['model_code_sha256']):
        raise ValueError('The frozen artifact or inference source differs from the model card.')
    artifact = joblib.load(artifact_path)
    fit, dev = study.partition('fit', 'development')
    x = study.rows(dev)
    old = AnswerClassifier('concatenated').fit(study.rows(fit), fit.label)
    old_pred = np.array(LABELS)[old.predict_proba(x).argmax(axis=1)]
    new_pred = np.array([label for label, _, _ in infer(artifact, x)])
    audit_result = {'development': {
        'labels': LABELS,
        'concatenated_confusion': confusion_matrix(dev.label, old_pred, labels=LABELS).tolist(),
        'pair_features_confusion': confusion_matrix(dev.label, new_pred, labels=LABELS).tolist(),
        'question_cluster_sensitivity': compare_by_question(dev, old_pred, new_pred),
    }, 'official': {}}
    examples = study.error_examples(dev, new_pred).merge(
        dev.assign(model_score=[score for _, score, _ in infer(artifact, x)])[['sample_id', 'model_score']],
        on='sample_id', validate='one_to_one')
    examples.to_csv(reports / 'development_error_examples.csv', index=False)
    baseline = pd.read_csv(reports / 'official_metrics.csv').set_index('split')
    for name, filename in SOURCES.items():
        path = study.PROCESSED / filename
        if study.sha(path) != baseline.loc[name, 'test_csv_sha256']:
            raise ValueError(f'The test data for {name} differ from the frozen evaluation.')
        df = pd.read_csv(path, keep_default_na=False)
        prediction = np.array([label for label, _, _ in infer(artifact, study.rows(df))])
        seen_q = df.question_id.isin(fit.question_id)
        seen_input = df.normalized_input_hash.isin(fit.normalized_input_hash)
        result = {'fit_normalized_input_matches': int(seen_input.sum()),
                  'confusion': confusion_matrix(df.label, prediction, labels=LABELS).tolist()}
        if name == 'UA':
            result['by_fit_question_exposure'] = {
                group: study.metrics(df[mask], prediction[mask])
                for group, mask in {'seen': seen_q, 'not_seen': ~seen_q}.items()
            }
            result['excluding_fit_input_matches'] = study.metrics(df[~seen_input], prediction[~seen_input])
        audit_result['official'][name] = result
    study.json_write(reports / 'error_audit.json', audit_result)
    return audit_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), indent=2))
