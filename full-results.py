# Generate Figures 11, 12, 13
# Generate Table 1

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
from collections import defaultdict
import operator

import random
import math
from scipy.special import betainc, binom, hyp2f1
from scipy.stats import rv_discrete

from show_your_work.expected_max_performance import samplemax
from max_random_baseline import max_random_baseline, max_random_p_value
from standard_baseline import binomial_p_value

from dataset_constants import *

with open('results/aggregated-results.json', 'r') as f:
    full_data = json.load(f)

standard_random_color = '#000'
max_random_color = '#000'
palette = {'Best random out of t classifiers': max_random_color,
           'Best 1-shot across t prompts': '#fdae61',
           'Best 2-shot across t prompts': '#f46d43',
           'Best 4-shot across t prompts': '#d73027',
           'Best 1-shot instruction-tuned across t prompts': '#abd9e9',
           'Best 2-shot instruction-tuned across t prompts': '#74add1',
           'Best 4-shot instruction-tuned across t prompts': '#4575b4',}

datasets = ['code_line_description', 'bbq_lite_json', 'hindu_knowledge', 'conceptual_combinations', 'formal_fallacies_syllogisms_negation', 'known_unknowns', 'logical_deduction', 'play_dialog_same_or_different', 'strange_stories', 'symbol_interpretation', 'winowhy', 'strategyqa', 'novel_concepts', 'emoji_movie', 'vitaminc_fact_verification', 'language_identification']

between_baselines = defaultdict(int)
above_standard_random = defaultdict(int)
missing_results = set()
def plot_all(model_base, model_instruct):

    f = plt.figure(figsize=(6.4*4, 6.4*4 + 10))
    # add a dummy space!!
    num_rows = 4
    num_cols = 4
    gs = plt.GridSpec(nrows=num_rows * 4, ncols=num_cols, wspace=0.25, hspace=0.25, height_ratios=[1, 0.5, 0.5, 0.25] * num_rows, figure=f)
    axs = np.empty((num_rows * 4, num_cols), dtype=object)
    # five rows of datasets
    for row in range(num_rows):
        # three columns of datasets
        for col in range(num_cols):
            # three graphs per dataset
            for subrow in range(3):
                axs[(row * 4) + subrow, col] = f.add_subplot(gs[(row * 4) + subrow, col])

    total_skipped = 0
    for idx, dataset in enumerate(datasets):
        
        n, num_choices = BIGBENCH_LITE_DETAILS_NUM_CHOICES[dataset]
        n = min(200, n)

        row = math.floor(idx / num_cols) * 4
        col = idx % num_cols

        print(model_base, dataset)

        p = 1/ num_choices
        tidy_data = []

        ts = list(range(1, 201))
        final_t = ts[-1]
        all_expected_val_data = []
        title_fontweight = 'bold'
        for model, model_type in [(model_base, ''), (model_instruct, ' instruction-tuned')]:
            for num_shots in [1, 2, 4]:
                accs = full_data[f'{model}_{dataset}_{num_shots}-shot']
                type_name = f'{num_shots}-shot{model_type}'
                if len(accs) < final_t:
                    print(f'Not enough evaluations!! {model}, {dataset}, {type_name}')
                    total_skipped += 1
                    missing_results.add((dataset, model_base, type_name))
                    continue

                if max(accs) >= p:
                    above_standard_random[(dataset, model_base, type_name)] += 1

                # when n < 200, the demonstrations come out of the examples
                # and for n < 200, 1-shot, t = n < 200.
                number_of_evals = 200
                n_actual = 200
                if n < 200:
                    n_actual = n - num_shots
                if n < 200 and num_shots == 1:
                    number_of_evals = n
                if max(accs) >= p and max(accs) <= max_random_baseline(n_actual, p, number_of_evals):
                    between_baselines[(dataset, model_base, type_name)] += 1

                out = samplemax(accs)
                expected_vals = out['mean']
                stderrs = out['var']
                expected_vals = expected_vals[:final_t]
                stderrs = stderrs[:final_t]
                full_type_name = f'Best {type_name} across t prompts'
                if num_shots == 1 and n < final_t:
                    all_expected_val_data.append((expected_vals[:n], stderrs[:n], full_type_name, False))
                else:
                    all_expected_val_data.append((expected_vals, stderrs, full_type_name, True))
                for t, acc, std in zip(ts, expected_vals, stderrs):
                    num_correct = acc * n
                    if num_shots == 1 and t > n:
                        num_correct_final = expected_vals[n] * n
                        axs[row, col].scatter(n, expected_vals[n], s=30, c=palette[full_type_name], zorder=10, clip_on=False)
                        axs[row + 1, col].scatter(n, binomial_p_value(num_correct_final, n, p), s=30, c=palette[full_type_name], zorder=10, clip_on=False, alpha=0.8)
                        axs[row + 2, col].scatter(n, max_random_p_value(expected_vals[n], n, p, t), s=30, c=palette[full_type_name], zorder=10, clip_on=False, alpha=0.8)
                        break
                    elif t == final_t:
                        axs[row, col].scatter(t, acc, s=30, c=palette[full_type_name], zorder=10, clip_on=False)
                        axs[row + 1, col].scatter(t, binomial_p_value(num_correct, n, p), s=30, c=palette[full_type_name], zorder=10, clip_on=False, alpha=0.8)
                        axs[row + 2, col].scatter(t, max_random_p_value(acc, n, p, t), s=30, c=palette[full_type_name], zorder=10, clip_on=False, alpha=0.8)
                        break
                    tidy_data.append({'Number of validation set evaluations': t,
                                      'Max accuracy': acc,
                                      'Standard deviation': std,
                                      'p-value max random': max_random_p_value(acc, n, p, t),
                                      'p-value standard random': binomial_p_value(num_correct, n, p),
                                      'Type': full_type_name})
        df = pd.DataFrame(tidy_data)
        ax = axs[row, col]
        ax.grid(alpha=0.6, axis='both')
        sns.lineplot(ax=ax, data=df, x='Number of validation set evaluations', y='Max accuracy', linewidth=3, hue='Type', markers='.', palette=palette, hue_order=['Best 1-shot across t prompts', 'Best 2-shot across t prompts', 'Best 4-shot across t prompts', 'Best 1-shot instruction-tuned across t prompts', 'Best 2-shot instruction-tuned across t prompts', 'Best 4-shot instruction-tuned across t prompts'], clip_on=False, zorder=10)
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
        ax.set_xlim((0,final_t))
        if ax.get_legend():
            ax.get_legend().remove()

        ax.set_title(dataset, fontsize=20, pad=16, fontweight=title_fontweight)

        ax = axs[row + 1, col]
        ax.grid(alpha=0.6, axis='both')
        sns.lineplot(ax=ax, data=df, x='Number of validation set evaluations', y='p-value standard random', linewidth=3, hue='Type', markers='.', palette=palette, hue_order=['Best 1-shot across t prompts', 'Best 2-shot across t prompts', 'Best 4-shot across t prompts', 'Best 1-shot instruction-tuned across t prompts', 'Best 2-shot instruction-tuned across t prompts', 'Best 4-shot instruction-tuned across t prompts'], clip_on=False, zorder=10,  alpha=0.8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim((0,final_t))
        ax.set_ylim((0,1))
        if ax.get_legend():
            ax.get_legend().remove()
        
        ax = axs[row + 2, col]
        ax.grid(alpha=0.6, axis='both')
        sns.lineplot(ax=ax, data=df, x='Number of validation set evaluations', y='p-value max random', linewidth=3, hue='Type', markers='.', palette=palette, hue_order=['Best 1-shot across t prompts', 'Best 2-shot across t prompts', 'Best 4-shot across t prompts', 'Best 1-shot instruction-tuned across t prompts', 'Best 2-shot instruction-tuned across t prompts', 'Best 4-shot instruction-tuned across t prompts'], clip_on=False, zorder=10, alpha=0.8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim((0,final_t))
        ax.set_ylim((0,1))
        if ax.get_legend():
            ax.get_legend().remove()


    # set axis labels
    for i in range(num_rows):
        axs[i * 4 + 0, 0].set_ylabel('Expected\nmax accuracy', fontsize=14, labelpad=11, fontweight='bold')
        axs[i * 4 + 1, 0].set_ylabel('p-value\n(standard)', fontsize=14, labelpad=12, fontweight='bold')
        axs[i * 4 + 2, 0].set_ylabel('p-value\n(maximum)', fontsize=14, labelpad=12, fontweight='bold')
        for col in range(num_cols):
            axs[i * 4 + 2, col].set_xlabel('t = number of validation set evaluations', fontsize=14, labelpad=12, fontweight='bold')
    # make custom legend
    l1 = matplotlib.lines.Line2D([], [], color="black", linestyle=(0,(1,1)), linewidth=6)
    l2 = matplotlib.lines.Line2D([], [], color="black", linewidth=6)
    l3 = matplotlib.lines.Line2D([], [], color=palette['Best 1-shot across t prompts'], linewidth=6)
    l4 = matplotlib.lines.Line2D([], [], color=palette['Best 2-shot across t prompts'], linewidth=6)
    l5 = matplotlib.lines.Line2D([], [], color=palette['Best 4-shot across t prompts'], linewidth=6)
    l6 = matplotlib.lines.Line2D([], [], color=palette['Best 1-shot instruction-tuned across t prompts'], linewidth=6)
    l7 = matplotlib.lines.Line2D([], [], color=palette['Best 2-shot instruction-tuned across t prompts'], linewidth=6)
    l8 = matplotlib.lines.Line2D([], [], color=palette['Best 4-shot instruction-tuned across t prompts'], linewidth=6)
    texts = ['Standard random baseline', 'Max random baseline', 'Best 1-shot across t prompts', 'Best 2-shot across t prompts', 'Best 4-shot across t prompts', 'Best 1-shot instruction-tuned across t prompts', 'Best 2-shot instruction-tuned across t prompts', 'Best 4-shot instruction-tuned across t prompts']
    texts = ['\n'.join(textwrap.wrap(t, 15)) for t in texts]
    plt.legend((l1,l2,l3,l4,l5,l6,l7,l8), texts,
        loc='upper center', bbox_to_anchor=(0.5, 0.11), bbox_transform=f.transFigure, borderaxespad=0.,
        fontsize=16, ncol=8)

    for ax in axs.flat:
        if ax:
            ax.tick_params(axis='both', which='major', labelsize=14)

    savefig(f'all-results-panel_{model_base}.pdf', bbox_inches='tight')

plot_all('Llama-2-7b', 'Alpaca-7b')
plot_all('OLMo-7B', 'OLMo-7B-Instruct')
plot_all('falcon-7b', 'falcon-7b-instruct')


print('-----------')

# now print the table

datasets = ['code_line_description', 'bbq_lite_json', 'hindu_knowledge', 'novel_concepts', 'emoji_movie', 'vitaminc_fact_verification',
                'conceptual_combinations', 'formal_fallacies_syllogisms_negation', 'known_unknowns', 'logical_deduction', 'play_dialog_same_or_different',
                'strange_stories', 'symbol_interpretation', 'winowhy', 'strategyqa', 'language_identification',]
datasets_with_n = [(d, min(BIGBENCH_LITE_DETAILS_NUM_CHOICES[d][0], 200)) for d in datasets]
datasets_with_n.sort(key = operator.itemgetter(1, 0))
ordered_datasets = [d for d, _ in datasets_with_n]

all_columns = ['1-shot', '2-shot', '4-shot', '1-shot instruction-tuned', '2-shot instruction-tuned', '4-shot instruction-tuned']
type_name_to_total = defaultdict(int)
type_name_to_above = defaultdict(int)
model_type_name_to_total = defaultdict(int)
model_type_name_to_above = defaultdict(int)
for dataset in ordered_datasets:
    n, num_choices = BIGBENCH_LITE_DETAILS_NUM_CHOICES[dataset]
    n = min(n, 200)
    row = "{ \\small \\texttt{" + dataset.replace('_', r'\_') + "} } & " + f"{n} &"
    total_between = 0
    total_above = 0
    for type_name in all_columns:
        num_between_all_models = 0
        num_above_all_models = 0
        for i, model in enumerate(['Llama-2-7b', 'OLMo-7B', 'falcon-7b']):
            if (dataset, model, type_name) in missing_results:
                row += ' {\\color{red} \\fullmoon}'
                continue
            num_between = between_baselines[(dataset, model, type_name)] + between_baselines[(f'{dataset}*', model, type_name)]
            num_above = above_standard_random[(dataset, model, type_name)] + above_standard_random[(f'{dataset}*', model, type_name)]
            num_between_all_models += num_between
            num_above_all_models += num_above
            model_type_name_to_total[(model, type_name)] += num_between
            model_type_name_to_above[(model, type_name)] += num_above
            row += ' '
            # spacing in the middle between models
            if i > 0:
                row += '\\,'
            if num_above == 0 and num_between == 0:
                row += '\\fullmoon' #' \\phantom{{\\fullmoon}}' # empty
            elif num_above == 1 and num_between == 0:
                row += '\\newmoon'
            elif num_above == 1 and num_between == 1:
                row += '\\quartermoon'
            else:
                print('Wrong number of results for model/shot/dataset combo!!!!', model, dataset, type_name)
                exit()

        row += ' &'
        #row += f' {between_num} &'
        total_between += num_between_all_models
        total_above += num_above_all_models
        type_name_to_total[type_name] += num_between_all_models
        type_name_to_above[type_name] += num_above_all_models
    row +=  f' {total_between}\\\\'
    print(row)
print("\\midrule")
row = "Baseline disagreements per model {\\small \\quartermoon} & &"
for type_name in all_columns:
    for i, model in enumerate(['Llama-2-7b', 'OLMo-7B', 'falcon-7b']):
        row += f" {{\\small {model_type_name_to_total[(model, type_name)]}}}"
        if i < 2:
            row += ' \\tinyspace'
    row += ' &'
row += f' \\\\'
print(row)

row = "Total baseline disagreements {\\small \\quartermoon} & &"
for type_name in all_columns:
    row += f" {type_name_to_total[type_name]} &"
row += f' {sum(between_baselines.values())}\\\\'
print(row)

row = "Total percentage flipped $\\sfrac{\\quartermoon}{(\\raisebox{-0.2em}{\\quartermoon} + \\raisebox{-0.2em}{\\newmoon})}$ & &"
for type_name in all_columns:
    row += f" {(type_name_to_total[type_name]/type_name_to_above[type_name] * 100):.0f}\\% &"
row += f' {sum(between_baselines.values())/sum(above_standard_random.values())*100:.0f}\\%\\\\'
print(row)

print('\n--------------------------------\n')

# for the abstract
total_above = sum(above_standard_random.values())
total_between = sum(between_baselines.values())
print(f'{total_between}/{total_above} = {total_between/total_above*100:.4f}% of results that exceed the standard baseline do not exceed the stronger random baseline' )




