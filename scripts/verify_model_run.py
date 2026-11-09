"""Check that an independent v1 run reproduces the committed evidence."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'reports/models'


def verify(fresh, reference=REFERENCE):
    csvs = ('development_metrics.csv', 'official_metrics.csv', 'development_error_examples.csv')
    for name in csvs:
        pd.testing.assert_frame_equal(pd.read_csv(reference / name), pd.read_csv(fresh / name),
                                      check_exact=False, rtol=1e-5, atol=1e-7)
    for name in ('development_per_class.json', 'official_per_class.json', 'error_audit.json'):
        actual = json.loads((fresh / name).read_text())
        expected = json.loads((reference / name).read_text())
        if actual != expected:
            raise AssertionError(f'Unexpected changes in {name}.')
    actual = json.loads((fresh / 'model_card.json').read_text())
    expected = json.loads((reference / 'model_card.json').read_text())
    # Different BLAS builds can produce equivalent predictions but different model bytes.
    for key in expected.keys() - {'artifact_sha256'}:
        if isinstance(expected[key], float):
            matches = math.isclose(actual[key], expected[key], rel_tol=1e-5, abs_tol=1e-7)
        else:
            matches = actual[key] == expected[key]
        if not matches:
            raise AssertionError(f'Model card changed: {key}')
    print('PASS: independent v1 run reproduces model selection, metrics, error audit, and review policy.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report_dir', type=Path)
    verify(parser.parse_args().report_dir)
