"""Reproducible SciEntsBank development models and inference helpers."""

import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LABELS = ('correct', 'partially_correct_incomplete', 'irrelevant', 'contradictory', 'non_domain')
FIELDS = ('question_text', 'reference_answer', 'student_answer')
VARIANTS = ('majority', 'answer_only', 'concatenated', 'pair_features', 'pair_features_lsa')
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NEGATIONS = {'not', 'no', 'never', 'neither', 'nor', 'without', 'cannot'}


def lexical_features(question, reference, answer):
    q, r, a = (set(TOKEN.findall(text.lower())) for text in (question, reference, answer))
    def has_negation(tokens):
        return any(t in NEGATIONS or t.endswith("n't") for t in tokens)
    return [len(a & r) / max(len(a | r), 1), len(a & r) / max(len(r), 1),
            len(a & r) / max(len(a), 1), len(a & q) / max(len(a | q), 1),
            len(a) / max(len(r), 1), has_negation(a), has_negation(r),
            has_negation(a) != has_negation(r)]


def cosine(x, y):
    return (x * y).sum(axis=1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12)


class AnswerClassifier:
    """Keep feature construction identical for development, evaluation, and serving."""

    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        self.variant = variant
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.scaler = StandardScaler()
        self.lsa = TruncatedSVD(n_components=32, n_iter=5, random_state=42)
        self.classifier = (DummyClassifier(strategy='most_frequent') if variant == 'majority'
                           else LogisticRegression(C=3.0, max_iter=1000, random_state=42))

    def _transform(self, rows, *, fit=False):
        if self.variant == 'majority':
            return np.zeros((len(rows), 1))
        if self.variant in ('answer_only', 'concatenated'):
            texts = [a if self.variant == 'answer_only' else ' '.join((q, r, a)) for q, r, a in rows]
            return self.vectorizer.fit_transform(texts) if fit else self.vectorizer.transform(texts)
        q, r, a = map(list, zip(*rows))
        if fit:
            self.vectorizer.fit(q + r + a)
        qs, rs, ans = map(self.vectorizer.transform, (q, r, a))
        dense = np.asarray([lexical_features(*row) for row in rows], dtype=float)
        if self.variant == 'pair_features_lsa':
            if fit:
                self.lsa.fit(self.vectorizer.transform(q + r + a))
            q_latent, r_latent, a_latent = map(self.lsa.transform, (qs, rs, ans))
            dense = np.column_stack([dense, cosine(a_latent, r_latent), cosine(a_latent, q_latent)])
        scaled = self.scaler.fit_transform(dense) if fit else self.scaler.transform(dense)
        return hstack([ans, ans.multiply(rs), csr_matrix(scaled)], format='csr')

    def fit(self, rows, labels):
        self.classifier.fit(self._transform(rows, fit=True), labels)
        return self

    def predict_proba(self, rows):
        scores = self.classifier.predict_proba(self._transform(rows))
        order = [list(self.classifier.classes_).index(label) for label in LABELS]
        return scores[:, order]


def calibrated_probabilities(model, rows, temperature):
    raw = model.predict_proba(rows)
    return softmax(np.log(np.clip(raw, 1e-15, 1.0)) / temperature, axis=1)


def infer(artifact, rows):
    probabilities = calibrated_probabilities(artifact['model'], rows, artifact['temperature'])
    threshold = artifact['threshold']
    indices = probabilities.argmax(axis=1)
    scores = probabilities.max(axis=1)
    return [(LABELS[i], float(s), threshold is None or s < threshold)
            for i, s in zip(indices, scores)]
