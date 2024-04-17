import numpy as np
import pandas as pd
import glob
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.patheffects as path_effects
plt.rcParams["font.family"] = 'DejaVu Sans'

import math
from scipy.special import betainc, binom
from scipy.stats import rv_discrete

def max_order_statistic_binomial_F(k, n, p, t):
    if k < 0:
        return 0
    i = betainc(n - math.floor(k), 1 + math.floor(k), 1-p)
    return np.power(i, t)

from dataset_constants import *

def normalized_score_to_accuracy(score, p):
    return score / 100 * (1-p) + p


ts = [1, 10, 100, 1000]
f, axs = plt.subplots(4, 2, figsize=(9.6, 6.4))
plt.subplots_adjust(hspace=0.75, wspace = 0.2)

for row, t in enumerate(ts):
    ax = axs[row, 1]
    tidy_data = []
    for dataset, (n, p) in BIGBENCH_HARD_DETAILS.items():
        for score in range(0, 101):
            # number of correct guesses for given normalized_score
            acc = normalized_score_to_accuracy(score, p)
            pval = 1 - max_order_statistic_binomial_F(acc * n - 1, n, p, t)
            tidy_data.append({'Dataset': dataset,
                              'Number of examples': n,
                              'Normalized score': score,
                              'p-value': pval}
                             )
    df = pd.DataFrame(tidy_data)
    ax.grid(alpha=0.6, axis='both')
    sns.lineplot(ax=ax, data=df, x='Normalized score', y='p-value', linewidth=2, hue='Number of examples', units='Dataset', estimator=None, hue_norm=matplotlib.colors.LogNorm(), markers='.', zorder=10)
    ax.get_legend().remove()
    ax.set_xlim((25,51))
    ax.set_xticks(range(0,71,10))
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_ylim(1e-3, 1)
    ax.set_yscale('log')
    ax.set_yticks([0.001, 0.01, 0.1, 1], ['0.001', '0.01', '0.1', '1'])
    ax.set_title(f't = {t}', fontweight='bold')
    if row == 0:
        ax.text(0.5, 1.5, 'BIG-bench Hard', transform=ax.transAxes, fontsize=22, fontweight='bold', ha='center', va='center')
    if row == len(ts) - 1:
        ax.set_xlabel('Normalized score', fontsize=16, fontweight='bold')

for row, t in enumerate(ts):
    ax = axs[row, 0]
    tidy_data = []
    for score in range(0, 101):
        for bb_dataset, (n, p) in BIGBENCH_LITE_DETAILS.items():
            # number of correct guesses for given normalized_score
            acc = normalized_score_to_accuracy(score, p)
            pval = 1 - max_order_statistic_binomial_F(acc * n - 1, n, p, t)
            tidy_data.append({'Dataset': bb_dataset,
                              'Number of examples': n,
                              'Normalized score': score,
                              'p-value': pval}
                             )
    df = pd.DataFrame(tidy_data)
    ax.grid(alpha=0.6, axis='both')
    sns.lineplot(ax=ax, data=df, x='Normalized score', y='p-value', linewidth=2, hue='Number of examples', hue_norm=matplotlib.colors.LogNorm(), markers='.', zorder=10)
    ax.get_legend().remove()
    ax.set_xlim((0,70))
    ax.set_xlabel('')
    ax.set_xticks(range(0,71,10))
    ax.set_ylabel('p-value', fontsize=16, fontweight='bold')
    ax.set_ylim(1e-3, 1)
    ax.set_yscale('log')
    ax.set_yticks([0.001, 0.01, 0.1, 1], ['0.001', '0.01', '0.1', '1'])
    ax.set_title(f't = {t}', fontweight='bold')
    if row == 0:
        ax.text(0.5, 1.5, 'BIG-bench Lite', transform=ax.transAxes, fontsize=22, fontweight='bold', ha='center', va='center')
    if row == len(ts) - 1:
        ax.set_xlabel('Normalized score', fontsize=16, fontweight='bold')

savefig(f'figure-9.pdf', bbox_inches='tight')



