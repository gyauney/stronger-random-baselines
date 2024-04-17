import json
from collections import defaultdict
import numpy as np
import pandas as pd
import glob
import os
from scipy.stats import bootstrap
from scipy.stats import spearmanr
import random
import math
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")

rng = np.random.default_rng()

all_datasets = ['code_line_description',
                'bbq_lite_json',
                'hindu_knowledge',
                'novel_concepts',
                'emoji_movie',
                'vitaminc_fact_verification',
                'conceptual_combinations',
                'formal_fallacies_syllogisms_negation',
                'known_unknowns',
                'logical_deduction',
                'play_dialog_same_or_different',
                'strange_stories',
                'symbol_interpretation',
                'winowhy',
                'strategyqa',
                'language_identification',]

all_models = ['Llama-2-7b-hf', 'Alpaca-7b', 'OLMo-7B', 'OLMo-7B-Instruct', 'falcon-7b', 'falcon-7b-instruct']

all_num_shots = [1,2,4]
num_plots = len(all_num_shots)

def evaluate_specific_idxs(example_to_choices, eval_idxs):
    # accuracy by minimizing (non-length-normalized) loss of choice
    correct = []
    for example_idx, choices in example_to_choices.items():
        if example_idx not in eval_idxs:
            continue
        if sorted(choices, reverse=False)[0][1] == 1:
            correct.append(1)
        else:
            correct.append(0)
    accuracy = sum(correct)/len(correct)

    correct_losses = []
    for example_idx, choices in example_to_choices.items():
        if example_idx not in eval_idxs:
            continue
        for (loss, score, answer) in choices:
            if score == 1:
                correct_losses.append(loss)
    
    return accuracy, np.mean(correct_losses)

def collect_results_in_directory(results_dir, bb_dataset):
    filenames = glob.glob(f'{results_dir}/{bb_dataset}_prompt-*.json')
    accuracies = []
    for filename in filenames:
        prompt_num = str(filename.split('.')[-2].split('-')[-1])

        example_to_choices = defaultdict(list)

        with open(filename, 'r') as f:
            results = json.load(f)

        # convert results to tidy dict per example
        data = [{'example_idx': results['example_idx'][i],
                'score': results['score'][i],
                'loss': results['loss'][i],
                'answer_idx': results['answer_idx'][i]}
                for i in range(len(results['loss']))]

        for d in data:
            example_to_choices[d['example_idx']].append((d['loss'], d['score'], d['answer_idx']))

        all_idxs = set(example_to_choices.keys())
        
        #val_acc, val_loss = evaluate_specific_idxs(example_to_choices, val_idxs)
        #test_acc, test_loss = evaluate_specific_idxs(example_to_choices, test_idxs)
        all_acc, all_loss = evaluate_specific_idxs(example_to_choices, all_idxs)
        accuracies.append(all_acc)
    return accuracies


data = {}
for model_filename in all_models:
        for num_shots in all_num_shots:
            results_dir = f'output-{model_filename}_sampled_{num_shots}-shot_random-labels_prefix-has-real-labels'
            if os.path.exists(results_dir):
                print(results_dir)
                for bb_dataset in all_datasets:                
                    num_prompts_done = len(glob.glob(f'{results_dir}/{bb_dataset}_prompt-*.json'))
                    accuracies = collect_results_in_directory(results_dir, bb_dataset)
                    print(f'    {bb_dataset}: {num_prompts_done}\t{accuracies}')
                    data[f'{model_filename}_sampled-{num_shots}-shot_{bb_dataset}'] = accuracies
            quantized_results_dir = f'output-{model_filename}-quantized_sampled_{num_shots}-shot_random-labels_prefix-has-real-labels'
            if os.path.exists(quantized_results_dir):
                print(quantized_results_dir)
                for bb_dataset in all_datasets:                
                    num_prompts_done = len(glob.glob(f'{quantized_results_dir}/{bb_dataset}_prompt-*.json'))
                    accuracies = collect_results_in_directory(quantized_results_dir, bb_dataset)
                    print(f'    {bb_dataset}: {num_prompts_done}\t{accuracies}')
                    data[f'{model_filename}-quantized_sampled-{num_shots}-shot_{bb_dataset}'] = accuracies

with open('aggregated-results.json', 'w') as f:
    json.dump(data, f)