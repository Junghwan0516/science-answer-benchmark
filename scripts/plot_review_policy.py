"""Plot the observed coverage/error trade-off from a frozen model run."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


def plot(report_dir):
    card = json.loads((report_dir / 'model_card.json').read_text())
    curve = pd.read_csv(report_dir / 'calibration_review_curve.csv').sort_values('coverage')
    minimum = card['review_target_min_coverage']
    maximum = card['review_target_max_error']
    eligible = curve.loc[curve.coverage.ge(minimum)]
    best = eligible.loc[eligible.accepted_error.idxmin()]
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    ax.plot(curve.coverage, curve.accepted_error, color='#225b90', lw=1.5,
            label='Observed calibration trade-off')
    ax.fill_between([minimum, 1], 0, maximum, color='#91cfb1', alpha=.3, label='Target region')
    ax.scatter(best.coverage, best.accepted_error, color='#9c3a19', zorder=4)
    ax.annotate(f'Best error at ≥{minimum:.0%} coverage: {best.accepted_error:.1%}',
                (best.coverage, best.accepted_error), xytext=(.29, .38),
                arrowprops={'arrowstyle': '->', 'color': '#9c3a19'})
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Answers accepted without a review flag',
           ylabel='Error among accepted answers',
           title='Calibration: no threshold reached both review targets' if card['threshold'] is None
           else 'Calibration: model coverage and accepted error')
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='upper right')
    fig.savefig(report_dir / 'calibration_review_curve.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report_dir', type=Path)
    plot(parser.parse_args().report_dir)
