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
                'language_identification',
                ]

all_models = ['Llama-2-7b-hf',
              'Alpaca-7b',
              'OLMo-7B',
              'OLMo-7B-Instruct',
              'falcon-7b',
              'falcon-7b-instruct'
              ]

output_dir = './single-example-results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


all_num_shots = [1,2,4]
num_plots = len(all_num_shots)


def collect_results_in_directory(results_dir, bb_dataset):
    filenames = glob.glob(f'{results_dir}/{bb_dataset}_prompt-*.json')
    tidy_data = []
    
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

        for example_idx, choices in example_to_choices.items():
            correct = 0
            correct_loss = -1
            if sorted(choices, reverse=False)[0][1] == 1:
                correct = 1
            for (loss, score, answer) in choices:
                if score == 1:
                    correct_loss = loss
            tidy_data.append({
                            'Prompt ID': prompt_num,
                            'Example ID': example_idx,
                            'Loss of correct answer': correct_loss,
                            'Correct prediction': correct,
                            })
    return tidy_data


for model_filename in all_models:
        for bb_dataset in all_datasets:
            for num_shots in all_num_shots:
                print(f'Loading {model_filename}: {bb_dataset}, {num_shots} shots')
                results_dir = f'output-{model_filename}_sampled_{num_shots}-shot_random-labels_prefix-has-real-labels'
                tidy_data = collect_results_in_directory(results_dir, bb_dataset)
                if len(tidy_data) > 0:
                    df = pd.DataFrame(tidy_data)
                    df.to_csv(f'{output_dir}/{model_filename}_{bb_dataset}_{num_shots}-shot.csv', index=False)
                
                quantized_results_dir = f'output-{model_filename}-quantized_sampled_{num_shots}-shot_random-labels_prefix-has-real-labels'
                tidy_data = collect_results_in_directory(quantized_results_dir, bb_dataset)
                if len(tidy_data) > 0:
                    df = pd.DataFrame(tidy_data)
                    df.to_csv(f'{output_dir}/{model_filename}-quantized_{bb_dataset}_{num_shots}-shot.csv', index=False)
                
                old_results_dir = f'output-{model_filename}_sampled_{num_shots}-shot_random-labels_prefix-has-real-labels'
                tidy_data = collect_results_in_directory(results_dir, bb_dataset)
                if len(tidy_data) > 0:
                    df = pd.DataFrame(tidy_data)
                    df.to_csv(f'{output_dir}/{model_filename}_{bb_dataset}_{num_shots}-shot.csv', index=False)