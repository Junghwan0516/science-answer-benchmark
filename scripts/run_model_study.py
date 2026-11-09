"""Run development, freeze a model and review policy, then evaluate official splits."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

from science_models import AnswerClassifier, FIELDS, LABELS, VARIANTS, infer

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / 'data/splits/v1'
PROCESSED = ROOT / 'data/processed'
REPORTS = ROOT / 'reports/models'
ARTIFACT = ROOT / 'artifacts/science_answer_model.joblib'
MAX_CALIBRATION_ERROR = 0.30
MIN_CALIBRATION_COVERAGE = 0.10


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def partition(*names):
    protocol = json.loads((SPLITS / 'protocol.json').read_text())
    source = PROCESSED / 'train.csv'
    assignments = SPLITS / 'answer_assignments.csv'
    if sha(source) != protocol['source_training_sha256'] or sha(assignments) != protocol['assignment_sha256'][assignments.name]:
        raise ValueError('Training data or assignments differ from the frozen split protocol.')
    keys = ['sample_id', 'question_id', 'label', 'normalized_input_hash']
    assigned = pd.read_csv(assignments, usecols=keys + ['internal_split'])
    assigned = assigned[assigned.internal_split.isin(names)]
    records = pd.read_csv(source, usecols=keys + list(FIELDS), keep_default_na=False)
    joined = assigned.merge(records, on=keys, validate='one_to_one')
    if len(joined) != len(assigned) or not joined.sample_id.is_unique:
        raise ValueError('Training assignments did not match the processed data.')
    for key in ('question_id', 'normalized_input_hash'):
        if joined.groupby(key).internal_split.nunique().max() != 1:
            raise ValueError(f'Cross-partition overlap: {key}')
    return tuple(joined.loc[joined.internal_split.eq(name)].copy() for name in names)


def rows(df):
    return list(df[list(FIELDS)].itertuples(index=False, name=None))


def metrics(df, predictions):
    true = df.label.to_numpy()
    pred = np.asarray(predictions)
    supported = [label for label in LABELS if label in set(true)]
    contrad = true == 'contradictory'
    return {
        'answers': len(df), 'questions': int(df.question_id.nunique()),
        'accuracy': float(accuracy_score(true, pred)),
        'macro_f1_fixed_five': float(f1_score(true, pred, labels=LABELS, average='macro', zero_division=0)),
        'macro_f1_supported_only': float(f1_score(true, pred, labels=supported, average='macro', zero_division=0)),
        'contradictory_support': int(contrad.sum()),
        'contradictory_predicted_correct': int((contrad & (pred == 'correct')).sum()),
    }


def analyze_development(fit, dev):
    reports, predictions = [], {}
    for variant in VARIANTS:
        model = AnswerClassifier(variant).fit(rows(fit), fit.label)
        pred = np.array(LABELS)[model.predict_proba(rows(dev)).argmax(axis=1)]
        predictions[variant] = pred
        reports.append({'model': variant, **metrics(dev, pred)})
    return pd.DataFrame(reports), predictions


def select_model(table):
    # Prefer fewer false-correct contradictions when fixed-five macro F1 ties.
    eligible = table[table.model.ne('majority')]
    return eligible.sort_values(['macro_f1_fixed_five', 'contradictory_predicted_correct', 'model'],
                                ascending=[False, True, True]).iloc[0].model


def error_examples(dev, pred, limit=5):
    selected = dev.assign(predicted_label=pred)
    selected = selected[selected.label.eq('contradictory') & selected.predicted_label.eq('correct')]
    return selected.sort_values(['question_id', 'sample_id']).drop_duplicates('question_id').head(limit)[
        ['sample_id', 'question_id', *FIELDS, 'label', 'predicted_label']]


def confusion_plot(df, predictions, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    names = ['Correct', 'Partial', 'Irrelevant', 'Contradictory', 'Non-domain']
    for ax, (name, pred) in zip(axes, predictions.items()):
        matrix = confusion_matrix(df.label, pred, labels=LABELS, normalize='true')
        ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, f'{matrix[i,j]:.0%}', ha='center', va='center',
                    color='white' if matrix[i,j] > .5 else '#18243a')
        ax.set(xticks=range(5), yticks=range(5), xticklabels=names,
               yticklabels=names, title=name, xlabel='Predicted label', ylabel='Reference label')
        ax.tick_params(axis='x', labelrotation=35)
    fig.suptitle('Question-held-out development errors | share of each reference class')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def development():
    fit, dev = partition('fit', 'development')
    table, predictions = analyze_development(fit, dev)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / 'development_metrics.csv', index=False)
    winner = select_model(table)
    json_write(REPORTS / 'development_per_class.json', {
        name: classification_report(dev.label, pred, labels=LABELS, output_dict=True, zero_division=0)
        for name, pred in predictions.items()
    })
    alternative = winner if winner != 'concatenated' else 'answer_only'
    confusion_plot(dev, {'Concatenated TF-IDF': predictions['concatenated'],
                         alternative.replace('_', ' ').title(): predictions[alternative]},
                   REPORTS / 'development_confusion.png')
    print(table.to_string(index=False))
    print(f'Selected on development: {winner}')
    return winner


def review_curve(y, proba):
    correct = np.array(LABELS)[proba.argmax(axis=1)] == np.asarray(y)
    scores = proba.max(axis=1)
    results = []
    for threshold in np.unique(scores):
        accepted = scores >= threshold
        results.append({'threshold': float(threshold), 'coverage': float(accepted.mean()),
                        'accepted_error': float((~correct[accepted]).mean()),
                        'accepted_answers': int(accepted.sum())})
    return pd.DataFrame(results)


def freeze():
    if ARTIFACT.exists() or (REPORTS / 'model_card.json').exists():
        raise SystemExit('This version is already frozen. Keep it intact for reproducibility.')
    winner = development()
    fit, calibration = partition('fit', 'calibration')
    model = AnswerClassifier(winner).fit(rows(fit), fit.label)
    logits = np.log(np.clip(model.predict_proba(rows(calibration)), 1e-15, 1))
    targets = np.array([LABELS.index(label) for label in calibration.label])
    def nll(t):
        return float(log_loss(targets, softmax(logits / t, axis=1), labels=np.arange(len(LABELS))))
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    if not result.success:
        raise RuntimeError('Temperature fitting did not converge.')
    temperature = float(result.x) if result.fun < nll(1.0) else 1.0
    curve = review_curve(calibration.label, softmax(logits / temperature, axis=1))
    eligible = curve[curve.coverage.ge(MIN_CALIBRATION_COVERAGE) & curve.accepted_error.le(MAX_CALIBRATION_ERROR)]
    policy = None if eligible.empty else eligible.sort_values('coverage', ascending=False).iloc[0].to_dict()
    threshold = None if policy is None else float(policy['threshold'])
    artifact = {'schema_version': 1, 'model': model, 'model_version': f'{winner}-v1',
                'temperature': temperature, 'threshold': threshold, 'labels': LABELS}
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT, compress=3)
    curve.to_csv(REPORTS / 'calibration_review_curve.csv', index=False)
    card = {
        'model_version': artifact['model_version'], 'artifact_sha256': sha(ARTIFACT),
        'model_code_sha256': sha(Path(__file__).with_name('science_models.py')),
        'evaluation_code_sha256': sha(Path(__file__)),
        'training_csv_sha256': sha(PROCESSED / 'train.csv'),
        'training_assignments_sha256': sha(SPLITS / 'answer_assignments.csv'),
        'fit_questions': sorted(fit.question_id.unique().tolist()),
        'fit_answers': len(fit), 'calibration_answers': len(calibration),
        'temperature': temperature, 'calibration_log_loss_raw': nll(1.0),
        'calibration_log_loss_temperature_scaled': nll(temperature),
        'review_target_max_error': MAX_CALIBRATION_ERROR, 'review_target_min_coverage': MIN_CALIBRATION_COVERAGE,
        'calibration_policy_point': policy, 'threshold': threshold,
        'intended_use': 'Research demo for English short science answers; human review remains required for grading.'
    }
    json_write(REPORTS / 'model_card.json', card)
    print(json.dumps({k: v for k, v in card.items() if k != 'fit_questions'}, indent=2))
    return card


def evaluate():
    card = json.loads((REPORTS / 'model_card.json').read_text())
    checks = {
        'artifact_sha256': ARTIFACT, 'model_code_sha256': Path(__file__).with_name('science_models.py'),
        'evaluation_code_sha256': Path(__file__), 'training_csv_sha256': PROCESSED / 'train.csv',
        'training_assignments_sha256': SPLITS / 'answer_assignments.csv',
    }
    if any(card[key] != sha(path) for key, path in checks.items()):
        raise ValueError('The frozen artifact, code, or training inputs changed.')
    artifact = joblib.load(ARTIFACT)
    summaries, class_reports = [], {}
    for name, filename in {
        'UA': 'test-unseen-answers.csv', 'UQ': 'test-unseen-questions.csv', 'UD': 'test-unseen-domains.csv'
    }.items():
        source = PROCESSED / filename
        df = pd.read_csv(source, keep_default_na=False)
        output = infer(artifact, rows(df))
        pred, _, review = map(np.asarray, zip(*output))
        accepted = ~review
        wrong = pred != df.label.to_numpy()
        fit_seen = df.question_id.isin(card['fit_questions'])
        summaries.append({'split': name, **metrics(df, pred),
                          'fit_seen_questions': int(df.loc[fit_seen, 'question_id'].nunique()),
                          'fit_seen_answers': int(fit_seen.sum()),
                          'accepted_answers': int(accepted.sum()), 'coverage': float(accepted.mean()),
                          'accepted_error': float(wrong[accepted].mean()) if accepted.any() else None,
                          'accepted_contradictory_predicted_correct': int(
                              (accepted & df.label.eq('contradictory').to_numpy() & (pred == 'correct')).sum()),
                          'test_csv_sha256': sha(source)})
        class_reports[name] = classification_report(df.label, pred, labels=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(summaries).to_csv(REPORTS / 'official_metrics.csv', index=False)
    json_write(REPORTS / 'official_per_class.json', class_reports)
    print(pd.DataFrame(summaries).drop(columns='test_csv_sha256').to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['development', 'freeze', 'evaluate'])
    parser.add_argument('--work-dir', type=Path, help='Separate location for a fresh reproducibility run.')
    args = parser.parse_args()
    if args.work_dir is not None:
        REPORTS = args.work_dir / 'reports'
        ARTIFACT = args.work_dir / 'artifacts/science_answer_model.joblib'
    {'development': development, 'freeze': freeze, 'evaluate': evaluate}[args.stage]()
