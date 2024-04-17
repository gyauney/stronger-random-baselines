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
import textwrap
import json

import random
import math
from scipy.special import binom, betainc

from show_your_work.expected_max_performance import samplemax
from dataset_constants import *
from max_random_baseline import max_random_baseline, max_random_p_value
from standard_baseline import binomial_p_value

with open('results/aggregated-results.json', 'r') as f:
    full_data = json.load(f)

standard_random_color = '#000' #'#0094e3'
max_random_color = '#000' #'#3d7bb1'
one_shot_color = '#fdae61'
two_shot_color = '#f46d43'
four_shot_color  = '#d73027'
palette = {'Best random out of t classifiers': max_random_color,
           'Best 1-shot across t prompts': one_shot_color,
           'Best 2-shot across t prompts': two_shot_color,
           'Best 4-shot across t prompts': four_shot_color}

all_datasets = ['emoji_movie', 'known_unknowns', 'bbq_lite_json', 'hindu_knowledge']
num_datasets = len(all_datasets)
f, axs = plt.subplots(3, num_datasets, figsize=(6.4*num_datasets, 6.4), gridspec_kw={'height_ratios': [1, 0.5, 0.5]})
plt.subplots_adjust(hspace=0.5, wspace=0.25)

for col, dataset in enumerate(all_datasets):
    print(dataset)
    n, num_choices = BIGBENCH_LITE_DETAILS_NUM_CHOICES[dataset]
    n = min(200, n)
    p = 1/ num_choices
    tidy_data = []

    model = 'OLMo-7B'
    model_type = ''
    ts = list(range(1, 201))
    final_t = ts[-1]
    all_expected_val_data = []
    for num_shots in [1, 2, 4]:
        final_one_shot = None
        type_name = f'{num_shots}-shot{model_type}'
        accs = full_data[f'{model}_{dataset}_{num_shots}-shot']
        out = samplemax(accs)
        expected_vals = out['mean']
        stderrs = out['var']
        expected_vals = expected_vals[:final_t]
        stderrs = stderrs[:final_t]
        full_type_name = f'Best {type_name} across t prompts'
        # if we're doing 1-shot on a small dataset with n < t, then there are only n < 200 prompts
        if num_shots == 1 and n < final_t:
            all_expected_val_data.append((expected_vals[:n], stderrs[:n], full_type_name, False))
        else:
            all_expected_val_data.append((expected_vals, stderrs, full_type_name, True))

        for t, acc, std in zip(ts, expected_vals, stderrs):
            num_correct = acc * n
            if num_shots == 1 and t > n:
                num_correct_final = expected_vals[n] * n
                axs[0, col].scatter(n, expected_vals[n], s=30, c=palette[full_type_name], zorder=10, clip_on=False)
                axs[1, col].scatter(n, binomial_p_value(num_correct_final, n, p), s=30, c=palette[full_type_name], zorder=10, clip_on=False)
                axs[2, col].scatter(n, max_random_p_value(expected_vals[n], n, p, t), s=30, c=palette[full_type_name], zorder=10, clip_on=False)
                break
            elif t == final_t:
                axs[0, col].scatter(t, acc, s=30, c=palette[full_type_name], zorder=10, clip_on=False)
                axs[1, col].scatter(t, binomial_p_value(num_correct, n, p), s=30, c=palette[full_type_name], zorder=10, clip_on=False)
                axs[2, col].scatter(t, max_random_p_value(acc, n, p, t), s=30, c=palette[full_type_name], zorder=10, clip_on=False)
                break
            tidy_data.append({'Number of validation set evaluations': t,
                              'Max accuracy': acc,
                              'Standard deviation': std,
                              'p-value max random': max_random_p_value(acc, n, p, t),
                              'p-value standard random': binomial_p_value(num_correct, n, p),
                              'Type': full_type_name})
    df = pd.DataFrame(tidy_data)
    ax = axs[0, col]
    ax.grid(alpha=0.6, axis='both')
    sns.lineplot(ax=ax, data=df, x='Number of validation set evaluations', y='Max accuracy', linewidth=3, hue='Type', markers='.', palette=palette, hue_order=['Best 1-shot across t prompts', 'Best 2-shot across t prompts', 'Best 4-shot across t prompts'], clip_on=False, zorder=10)
    max_tidy_data = []
    for t in ts:
        max_random = max_random_baseline(n, p, t)
        max_tidy_data.append({'Number of validation set evaluations': t,
                          'Max accuracy': max_random,
                          'p-value max random': float('nan'),
                          'p-value standard random': float('nan'),
                          'Type': 'Best random out of t classifiers'})
    max_df = pd.DataFrame(max_tidy_data)
    sns.lineplot(ax=ax, data=max_df, x='Number of validation set evaluations', y='Max accuracy', linewidth=3, hue='Type', markers='.', palette=palette, hue_order=['Best random out of t classifiers'], zorder=10)
    ax.axhline(p, linestyle='--', color=standard_random_color, linewidth=3, label='Standard random baseline')
    # extend the error bars to the edge of the plot
    ts.extend(list(range(ts[-1], ts[-1] + 3)))
    for expected_vals, stderrs, full_type_name, at_edge_of_graph in all_expected_val_data:
        lower = [ex - err for ex, err in zip(expected_vals, stderrs)]
        upper = [ex + err for ex, err in zip(expected_vals, stderrs)]
        lower.extend([lower[-1]] * 3)
        upper.extend([upper[-1]] * 3)
        if at_edge_of_graph:
            # extend the area to reach the edge of the axis
            ax.fill_between(ts[:len(lower)], lower, upper, color=palette[full_type_name], alpha=0.15, edgecolor='none')
        else:
            # don't extend because this is the 1-shot case where n < 200.
            ax.fill_between(ts[:len(lower)-3], lower[:-3], upper[:-3], color=palette[full_type_name], alpha=0.15, edgecolor='none')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim((0,200))
    legend = ax.get_legend().remove()
    ax.set_ylim((0.18, 0.8))

    ax.set_title(f'{model}\n{dataset}', fontsize=20, pad=16, fontweight='bold')

    ax = axs[1, col]
    ax.grid(alpha=0.6, axis='both')
    sns.lineplot(ax=ax, data=df, x='Number of validation set evaluations', y='p-value standard random', linewidth=3, hue='Type', markers='.', palette=palette, hue_order=['Best 1-shot across t prompts', 'Best 2-shot across t prompts', 'Best 4-shot across t prompts'], clip_on=False, zorder=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim((0,200))
    ax.set_ylim((0,1))
    legend = ax.get_legend().remove()
    
    ax = axs[2, col]
    ax.grid(alpha=0.6, axis='both')
    sns.lineplot(ax=ax, data=df, x='Number of validation set evaluations', y='p-value max random', linewidth=3, hue='Type', markers='.', palette=palette, hue_order=['Best 1-shot across t prompts', 'Best 2-shot across t prompts', 'Best 4-shot across t prompts'], clip_on=False, zorder=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim((0,200))
    ax.set_ylim((0,1))
    legend = ax.get_legend().remove()
# set axis labels
axs[0, 0].set_ylabel('Expected\nmax accuracy', fontsize=14, labelpad=11, fontweight='bold')
axs[1, 0].set_ylabel('p-value\n(standard)', fontsize=14, labelpad=12, fontweight='bold')
axs[2, 0].set_ylabel('p-value\n(maximum)', fontsize=14, labelpad=12, fontweight='bold')
for col in range(num_datasets):
    axs[-1, col].set_xlabel('t = number of validation set evaluations', fontsize=14, labelpad=12, fontweight='bold')
# make custom legend
l1 = matplotlib.lines.Line2D([], [], color="black", linestyle=(0,(1,1)), linewidth=6)
l2 = matplotlib.lines.Line2D([], [], color="black", linewidth=6)
l3 = matplotlib.lines.Line2D([], [], color=one_shot_color, linewidth=6)
l4 = matplotlib.lines.Line2D([], [], color=two_shot_color, linewidth=6)
l5 = matplotlib.lines.Line2D([], [], color=four_shot_color, linewidth=6)
texts = ['Standard random baseline', 'Max random baseline', 'Best 1-shot across t prompts', 'Best 2-shot across t prompts', 'Best 4-shot across t prompts']
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend((l1,l2,l3,l4, l5),texts,
        loc='upper center', bbox_to_anchor=(0.5, -0.03), bbox_transform=f.transFigure, borderaxespad=0.,
        fontsize=16, ncol=5)
savefig(f'figure-3.pdf', bbox_inches='tight')



