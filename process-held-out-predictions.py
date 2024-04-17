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

import random
import math

from dataset_constants import *

from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, roc_curve
from max_random_baseline import max_random_p_value

NUM_TRIALS = 100

standard_color = '#b3cde3'
max_color = '#fbb4ae'

standard_marker_color = '#377eb8'
max_marker_color = '#e41a1c' 

def get_results(df, dataset, model, num_shots, make_plot=False, axs=[], f=None, legend_bbox=(0.5, -0.1), alpha=1, no_pr=False):
    
    max_Fs = df['Max random F'].tolist()
    standard_Fs = df['Standard random F'].tolist()
    above_standard_on_test = df['Above standard random on test'].tolist()

    tp_max = len(df.loc[(df['Above max random on train'] == 1) &
                        (df['Above standard random on test'] == 1)])
    fp_max = len(df.loc[(df['Above max random on train'] == 1) &
                        (df['Above standard random on test'] == 0)])
    tn_max = len(df.loc[(df['Above max random on train'] == 0) &
                        (df['Above standard random on test'] == 0)])
    fn_max = len(df.loc[(df['Above max random on train'] == 0) &
                        (df['Above standard random on test'] == 1)])
    tp_standard = len(df.loc[(df['Above standard random on train'] == 1) &
                        (df['Above standard random on test'] == 1)])
    fp_standard = len(df.loc[(df['Above standard random on train'] == 1) &
                        (df['Above standard random on test'] == 0)])
    tn_standard = len(df.loc[(df['Above standard random on train'] == 0) &
                        (df['Above standard random on test'] == 0)])
    fn_standard = len(df.loc[(df['Above standard random on train'] == 0) &
                        (df['Above standard random on test'] == 1)])


    # tp_max = len(df.loc[(df['Max random F'] > 0.5) &
    #                     (df['Above standard random on test'] == 1)])
    # fp_max = len(df.loc[(df['Max random F'] > 0.5) &
    #                     (df['Above standard random on test'] == 0)])
    # tn_max = len(df.loc[(df['Max random F'] <= 0.5) &
    #                     (df['Above standard random on test'] == 0)])
    # fn_max = len(df.loc[(df['Max random F'] <= 0.5) &
    #                     (df['Above standard random on test'] == 1)])
    # tp_standard = len(df.loc[(df['Standard random F'] > 0.5) &
    #                     (df['Above standard random on test'] == 1)])
    # fp_standard = len(df.loc[(df['Standard random F'] > 0.5) &
    #                     (df['Above standard random on test'] == 0)])
    # tn_standard = len(df.loc[(df['Standard random F'] <= 0.5) &
    #                     (df['Above standard random on test'] == 0)])
    # fn_standard = len(df.loc[(df['Standard random F'] <= 0.5) &
    #                     (df['Above standard random on test'] == 1)])

    precision_max = tp_max / (tp_max + fp_max)
    precision_standard = tp_standard / (tp_standard + fp_standard)

    recall_max = tp_max / (tp_max + fn_max)
    recall_standard = tp_standard / (tp_standard + fn_standard)

    acc_max = (tp_max + tn_max) / (tp_max + tn_max + fp_max + fn_max)
    acc_standard = (tp_standard + tn_standard) / (tp_standard + tn_standard + fp_standard + fn_standard)

    # fpr not defined when only one class is present
    if len(set(above_standard_on_test)) == 1:
        roc_max = 'n/a'
        roc_standard = 'n/a'
    else:
        roc_max = roc_auc_score(above_standard_on_test, df['Above max random on train'].tolist())
        roc_standard = roc_auc_score(above_standard_on_test, df['Above standard random on train'].tolist())
    aupr_max = average_precision_score(above_standard_on_test, df['Above max random on train'].tolist())
    aupr_standard = average_precision_score(above_standard_on_test, df['Above standard random on train'].tolist())

    # print(dataset, 'Max', roc_max, aupr_max,  acc_max, precision_max, recall_max)
    # print(dataset, 'Sta', roc_standard, aupr_standard, acc_standard, precision_standard, recall_standard)

    d = []
    d.append({'Method': 'Standard random',
                          'Dataset': dataset,
                          'Percent of test accuracies above standard random': 100 * sum(above_standard_on_test)/len(above_standard_on_test),
                          'Model': model,
                          'Number of shots': num_shots,
                          't': t,
                          'Accuracy': acc_standard,
                          'Precision': precision_standard,
                          'Recall': recall_standard,
                          'AUROC': roc_standard,
                          'AUPR': aupr_standard})

    d.append({'Method': 'Maximum random',
                          'Dataset': dataset,
                          'Percent of test accuracies above standard random': 100 * sum(above_standard_on_test)/len(above_standard_on_test),
                          'Model': model,
                          'Number of shots': num_shots,
                          't': t,
                          'Accuracy': acc_max,
                          'Precision': precision_max,
                          'Recall': recall_max,
                          'AUROC': roc_max,
                          'AUPR': aupr_max})

    if not make_plot:
        return d

    #print(tp_max, fp_max, tn_max, fn_max)

    tpr_max = tp_max / (tp_max + fn_max) 
    fpr_max = fp_max / (fp_max + tn_max) 
    tpr_standard = tp_standard / (tp_standard + fn_standard) 
    fpr_standard = fp_standard / (fp_standard + tn_standard) 

    # print(acc_max, precision_max, recall_max)
    # print(acc_standard, precision_standard, recall_standard)

    if t == 200:
        print('AUROC using Fs', roc_auc_score(above_standard_on_test, max_Fs), roc_auc_score(above_standard_on_test, standard_Fs))
        print('AUPR using Fs', average_precision_score(above_standard_on_test, max_Fs), average_precision_score(above_standard_on_test, standard_Fs))
        print()

    # print()
    # print(roc_auc_score(above_standard_on_test, df['Above max random on train'].tolist()))
    # print(roc_auc_score(above_standard_on_test, df['Above standard random on train'].tolist()))
    # print()
    # print(average_precision_score(above_standard_on_test, df['Above max random on train'].tolist()))
    # print(average_precision_score(above_standard_on_test, df['Above standard random on train'].tolist()))

    save = False
    if not f:
        f, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
        plt.subplots_adjust(wspace = 0.35)
        save = True

    ax = axs[0]

    # roc curve for both Fs
    fpr, tpr, _ = roc_curve(above_standard_on_test, max_Fs)
    ax.plot(fpr, tpr, zorder=10, clip_on=False, linewidth=2, color='black')
    ax.fill_between(fpr, [0] * len(fpr), tpr, color='#eeeeee', edgecolor='none')
    # binary max roc curve
    fpr, tpr, _ = roc_curve(above_standard_on_test, df['Above max random on train'].tolist())
    ax.plot(fpr, tpr, zorder=10, clip_on=False, linewidth=3, color=max_color, alpha=0.8)
    ax.fill_between(fpr, [0] * len(fpr), tpr, color=max_color, alpha=0.4, edgecolor='none')
    # binary standard roc curve
    fpr, tpr, _ = roc_curve(above_standard_on_test, df['Above standard random on train'].tolist())
    ax.plot(fpr, tpr, zorder=10, clip_on=False, linewidth=3, color=standard_color, alpha=0.8)
    ax.fill_between(fpr, [0] * len(fpr), tpr, color=standard_color, alpha=0.4, edgecolor='none')
    # random chance
    ax.plot([0, 1], [0, 1], linewidth=2, color='black', linestyle='--')


    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    h2 = ax.scatter(fpr_standard, tpr_standard, marker='o', s=200, color=standard_marker_color, clip_on=False, zorder=20, alpha=alpha)
    h1 = ax.scatter(fpr_max, tpr_max, marker='X', s=200, color=max_marker_color, clip_on=False, zorder=20, alpha=alpha)
    # ax.get_legend().remove()
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1'], fontsize=12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1'], fontsize=12)
    ax.set_xlabel('False positive rate', fontsize=16, fontweight='bold')
    ax.set_ylabel('True positive rate', fontsize=16, fontweight='bold')

    # savefig(f'bigbench-train-test-splits-pr-curve_t={t}.pdf')
    # plt.close()

    # display = PrecisionRecallDisplay.from_predictions(
    #     above_standard_on_test, df['Above standard random on train'].tolist(), name="Standard random", plot_chance_level=True, zorder=10, clip_on=False, linewidth=2, color='black'
    # )
    # savefig(f'bigbench-train-test-splits-pr-curve_BINARIZED_STANDARD_RANDOM_t={t}.pdf')


    if no_pr:
        plt.legend((h2, h1), ('Standard random baseline', 'Maximum random baseline'),
        loc='upper center', bbox_to_anchor=legend_bbox, bbox_transform=f.transFigure, borderaxespad=0.,
        fontsize=16, ncol=2,)# prop={'weight':'bold'})
        return d

    ax = axs[1]
    # display = PrecisionRecallDisplay.from_predictions(
    #     above_standard_on_test, max_Fs, plot_chance_level=True, zorder=10, clip_on=False, linewidth=3, color='black', ax=ax, 
    # )
    ps, rs, _ = precision_recall_curve(above_standard_on_test, max_Fs)
    ax.plot(rs, ps, linewidth=2, color='black', zorder=10, clip_on=False)
    ax.fill_between(rs, [0] * len(rs), ps, color='#eeeeee', edgecolor='none')


    display = PrecisionRecallDisplay.from_predictions(
        above_standard_on_test, df['Above max random on train'].tolist(), plot_chance_level=False, zorder=10, clip_on=False, linewidth=3, color=max_color, alpha=0.5, ax=ax, 
    )
    rs, ps = display.line_.get_data()
    ax.fill_between(rs, [0] * len(rs), ps, step='post', color=max_color, alpha=0.3, edgecolor='none')
    display = PrecisionRecallDisplay.from_predictions(
        above_standard_on_test, df['Above standard random on train'].tolist(), plot_chance_level=False, zorder=10, clip_on=False, linewidth=3, color=standard_color, alpha=0.5, ax=ax, 
    )
    rs, ps = display.line_.get_data()
    ax.fill_between(rs, [0] * len(rs), ps, step='post', color=standard_color, alpha=0.3, edgecolor='none')
    # random chance
    percent_true_labels = sum(above_standard_on_test)/len(above_standard_on_test)
    ax.plot([0, 1], [percent_true_labels, percent_true_labels], linewidth=2, color='black', linestyle='--')


    #display.line_.set('label'=None)
    # ps, rs, _ = precision_recall_curve(above_standard_on_test, df['Above max random on train'].tolist())
    # ax.plot(rs, ps, zorder=10, clip_on=False, linewidth=3, color=max_color, alpha=0.5)
    #ax.fill_between(rs, [0] * len(rs), ps, color=max_color, alpha=0.3, edgecolor='none')

    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1'], fontsize=12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1'], fontsize=12)
    ax.set_xlabel('Recall', fontsize=16, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=16, fontweight='bold')
    #ax.axis('equal')
    h2 = ax.scatter(recall_standard, precision_standard, marker='o', s=200, color=standard_marker_color, clip_on=False, zorder=20, alpha=alpha)
    h1 = ax.scatter(recall_max, precision_max, marker='X', s=200, color=max_marker_color, clip_on=False, zorder=20, alpha=alpha)
    #ax.text(recall_max, precision_max, 'max random', text_)
    # ax.get_legend().remove()

    # ax.annotate('Max',
    #             fontweight='light',
    #             xy=(recall_max, precision_max), xycoords='data',
    #             xytext=(8,20), textcoords='offset points',
    #             ha='left', va='center',
    #             fontsize=14,
    #             bbox=dict(boxstyle="round,pad=0.1",
    #                   fc='red', lw=0, alpha=0.15))



    plt.legend((h2, h1), ('Standard random baseline', 'Maximum random baseline'),
        loc='upper center', bbox_to_anchor=legend_bbox, bbox_transform=f.transFigure, borderaxespad=0.,
        fontsize=15, ncol=2,)# prop={'weight':'bold'})

    if save:
        savefig(f'figure-5.pdf', bbox_inches='tight')
        plt.close()

    return d



# appendix plots

tidy_data_all = []
ts = [1, 2, 5, 10, 100, 200]
f, axs = plt.subplots(2, 3, figsize=(5.0*3, 8.0))
plt.subplots_adjust(wspace = 0.4, hspace=0.5)
for i, t in enumerate(ts):
    full_df = pd.read_csv(f'./held-out-split-results/all-train-test-splits_{NUM_TRIALS}-trials_t={t}_all-predictions.csv')
    full_df = full_df.dropna()
    row = math.floor(i / 3)
    col = i % 3
    get_results(full_df, 'All', 'All', 'All', make_plot=True, axs=[axs[row, col]], f=f, legend_bbox=(0.5, 0), alpha=0.8, no_pr=True)
    axs[row, col].set_title(f't = {t}', fontweight='bold', fontsize=22, pad=14)
    #axs[1, i].set_title(f't = {t}', fontweight='bold', fontsize=14)
    # if i < len(ts) - 1:
savefig(f'figure-10.pdf', bbox_inches='tight')
plt.close()





# now the main paper results

tidy_data_all = []
t = 200



full_df = pd.read_csv(f'./held-out-split-results/all-train-test-splits_{NUM_TRIALS}-trials_t={t}_all-predictions.csv')
full_df = full_df.dropna()

# 'train: between, test: below standard'
# 'train: between, test: above standard'
# 'train: below both, test: below standard'
# 'train: below both, test: above standard'
# 'train: above both, test: below standard'
# 'train: above both, test: above standard'

#df = df.loc[df['Above max random on train'] != df['Above standard random on train']]


tidy_data_all.extend(get_results(full_df, 'All', 'All', 'All', make_plot=True))

full_results_df = pd.DataFrame(tidy_data_all)
full_results_df.to_csv(f'bigbench-train-test-splits-results-all_t={t}.csv', index=False)




# now per dataset!!!
tidy_data_per_dataset = []
for dataset in ['code_line_description', 'bbq_lite_json', 'hindu_knowledge', 'novel_concepts', 'emoji_movie', 'vitaminc_fact_verification',
                'conceptual_combinations', 'formal_fallacies_syllogisms_negation', 'known_unknowns', 'logical_deduction', 'play_dialog_same_or_different',
                'strange_stories', 'symbol_interpretation', 'winowhy', 'strategyqa', #'language_identification'
                ]:
    dataset_df = full_df.loc[full_df['Dataset'] == dataset]
    tidy_data_per_dataset.extend(get_results(dataset_df, dataset, 'All', 'All'))
results_df = pd.DataFrame(tidy_data_per_dataset)
results_df.to_csv(f'bigbench-train-test-splits-results-per-dataset_t={t}.csv', index=False)

def get_val(row_df, col_name):
    assert(len(row_df) == 1)
    return row_df[col_name].iloc[0]

def bold_if_best(val, best):
    if val == 'n/a':
        return 'n/a'
    if round(val, 2) == round(max(best), 2):
        return '\\textbf{' + f'{val:.2f}' + '}'
    return f'{val:.2f}'

print('\n-------- dataset table ----------') 
# per-dataset table
for dataset in ['code_line_description', 'bbq_lite_json', 'hindu_knowledge', 'novel_concepts', 'emoji_movie', 'vitaminc_fact_verification',
                'conceptual_combinations', 'formal_fallacies_syllogisms_negation', 'known_unknowns', 'logical_deduction', 'play_dialog_same_or_different',
                'strange_stories', 'symbol_interpretation', 'winowhy', 'strategyqa', #'language_identification'
                ]:
    print_dataset = "{ \\small \\texttt{" + dataset.replace("_", r"\_") + "} }"
    max_row = results_df.loc[(results_df['Dataset'] == dataset) &
                             (results_df['Method'] == 'Maximum random')]
    standard_row = results_df.loc[(results_df['Dataset'] == dataset) &
                             (results_df['Method'] == 'Standard random')]
    row = f'{print_dataset} ({get_val(max_row, "Percent of test accuracies above standard random"):.0f}\\%) '
    for metric in ['AUROC', 'AUPR']:
        max_val = get_val(max_row, metric)
        standard_val = get_val(standard_row, metric)
        all_vals = [max_val, standard_val]
        row += '& ' + bold_if_best(standard_val, all_vals) + ' & ' + bold_if_best(max_val, all_vals) + ' '
    row += '\\\\'
    print(row)
print('\\midrule')
max_row = full_results_df.loc[(full_results_df['Method'] == 'Maximum random')]
standard_row = full_results_df.loc[(full_results_df['Method'] == 'Standard random')]
row = '\\textbf{Total} (' + f'{get_val(max_row, "Percent of test accuracies above standard random"):.0f}\\%) '
for metric in ['AUROC', 'AUPR']:
        max_val = get_val(max_row, metric)
        standard_val = get_val(standard_row, metric)
        all_vals = [max_val, standard_val]
        row += '& ' + bold_if_best(standard_val, all_vals) + ' & ' + bold_if_best(max_val, all_vals) + ' '
row += '\\\\'
print(row)
print('------------------\n\n') 

# now per model!!! across all datasets and num shots
total_numerator = 0
total_denominator = 0
tidy_data_per_model = []
for model_type in ['Base model', 'Instruction-tuned']:
    for model in ['Llama-2-7b', 'OLMo-7B', 'Falcon-7b']:
        full_model = f'{model}, {model_type}'
        model_df = full_df.loc[(full_df['Model'] == full_model)]
        total_numerator += model_df['Above standard random on test'].sum()
        total_denominator += len(model_df['Above standard random on test'])
        # print(model_df['Above standard random on test'].sum() / len(model_df['Above standard random on test']))
        tidy_data_per_model.extend(get_results(model_df, 'All', full_model, 'All'))
results_df = pd.DataFrame(tidy_data_per_model)
results_df.to_csv(f'bigbench-train-test-splits-results-per-model_t={t}.csv', index=False)

print('-------- model table ----------') 
# per-dataset table
for model_type in ['Base model', 'Instruction-tuned']:
    for model in ['Llama-2-7b', 'OLMo-7B', 'Falcon-7b']:
        full_model = f'{model}, {model_type}'
        max_row = results_df.loc[(results_df['Model'] == full_model) &
                                 (results_df['Method'] == 'Maximum random')]
        standard_row = results_df.loc[(results_df['Model'] == full_model) &
                                 (results_df['Method'] == 'Standard random')]
        if full_model == 'Llama-2-7b, Instruction-tuned':
            full_model = 'Alpaca-7b, Instruction-tuned'
        row = f'{full_model} ({get_val(max_row, "Percent of test accuracies above standard random"):.0f}\\%) '
        for metric in ['Accuracy', 'Precision', 'Recall', 'AUROC', 'AUPR']:
            max_val = get_val(max_row, metric)
            standard_val = get_val(standard_row, metric)
            all_vals = [max_val, standard_val]
            row += '& ' + bold_if_best(standard_val, all_vals) + ' & ' + bold_if_best(max_val, all_vals) + ' '
        row += '\\\\'
        print(row)
print('\\midrule')
max_row = full_results_df.loc[(full_results_df['Method'] == 'Maximum random')]
standard_row = full_results_df.loc[(full_results_df['Method'] == 'Standard random')]
row = '\\textbf{Total} (' + f'{get_val(max_row, "Percent of test accuracies above standard random"):.0f}\\%) '
for metric in ['Accuracy', 'Precision', 'Recall', 'AUROC', 'AUPR']:
        max_val = get_val(max_row, metric)
        standard_val = get_val(standard_row, metric)
        all_vals = [max_val, standard_val]
        row += '& ' + bold_if_best(standard_val, all_vals) + ' & ' + bold_if_best(max_val, all_vals) + ' '
row += '\\\\'
print(row)
print('------------------\n\n') 
