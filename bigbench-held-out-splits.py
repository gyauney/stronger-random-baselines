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
from scipy.special import betainc, binom, hyp2f1
from scipy.stats import rv_discrete

from expected_max_performance import samplemax

from dataset_constants import *

from max_random_baseline import max_random_baseline, max_random_F
from standard_baseline import binomial_F

NUM_TRIALS = 100
all_ts = [1, 2, 5, 10, 100, 200]

standard_random_color = '#000' #'#0094e3'
max_random_color = '#000' #'#3d7bb1'
palette = {'Best random out of t classifiers': max_random_color,
           'Best 1-shot across t prompts': '#f4a582',
           'Best 2-shot across t prompts': '#ca0020',
           'Best 1-shot instruction-tuned across t prompts': '#92c5de',
           'Best 2-shot instruction-tuned across t prompts': '#0571b0',}

column_types = {'Prompt ID': np.int32,
                'Example ID': np.int32,
                'Loss of correct answer': np.float64,
                'Correct prediction': np.float64}

between_baselines = defaultdict(int)
above_standard_random = defaultdict(int)
details_to_train_argmax_outcomes = {}
train_argmax_tidy_data = []


t_to_detailed_train_argmax_tidy_data = defaultdict(list)

def model_to_display(model):
    model_type = 'Base model'
    if 'instruct' in model.lower() or 'Alpaca' in model:
        model_type = 'Instruction-tuned'

    if 'Llama-2-7b' in model or 'Alpaca-7b' in model:
        model_name = 'Llama-2-7b'
    elif 'OLMo-7B' in model:
        model_name = 'OLMo-7B'
    elif 'falcon-7b' in model:
        model_name = 'Falcon-7b'    
    else:
        print(f'Model unrecognized: {model}')
        exit()
    
    return model_name, model_type


# model includes instruct and quantized tags
def train_test_split(dataset, model, num_shots):
    print(dataset, model, num_shots)

    n, p = BIGBENCH_LITE_DETAILS[dataset]

    model_name, model_type = model_to_display(model)

    filename = f'./results/{model}_{dataset}_{num_shots}-shot.csv'
    df = pd.read_csv(filename, dtype=column_types)
    
    # two entries: train can be 'below both', 'between', 'above both'
    #              test can be 'below standard', 'above standard'
    outcomes = defaultdict(int)

    # repeat with a different random split each time
    for trial in range(NUM_TRIALS):
        # split idxs
        all_idxs = set(df['Example ID'].tolist())
        num_train = math.floor(len(all_idxs) * 0.75)
        train_idxs = set(random.sample(list(all_idxs), num_train))
        test_idxs = all_idxs - train_idxs

        # get mean loss on train and test sets for each prompt
        unique_prompts = list(set(df['Prompt ID']))
        for t in all_ts:
            prompt_nums_to_use = unique_prompts
            if len(unique_prompts) > t:
                prompt_nums_to_use = random.sample(unique_prompts, t)

            per_prompt_tidy_data = []
            for prompt_num in prompt_nums_to_use:
                this_df = df.loc[df['Prompt ID'] == prompt_num]
                if len(this_df) == 0:
                    continue
                train_df = this_df.loc[this_df['Example ID'].isin(train_idxs)]
                test_df = this_df.loc[this_df['Example ID'].isin(test_idxs)]
                # this only works for subsampled datasets
                # assert set(train_df['Example ID']) == train_idxs
                # assert set(test_df['Example ID']) == test_idxs
                per_prompt_tidy_data.append({'Prompt ID': prompt_num,
                                             'Train mean loss of correct answer': train_df['Loss of correct answer'].mean(),
                                             'Train accuracy': train_df['Correct prediction'].mean(),
                                             'Test mean loss of correct answer': test_df['Loss of correct answer'].mean(),
                                             'Test accuracy': test_df['Correct prediction'].mean()})
            # now do the argmaxing
            per_prompt_df = pd.DataFrame(per_prompt_tidy_data)
            #print(per_prompt_df['Train accuracy'].idxmax())
            best_train_prompt_idx = per_prompt_df['Train accuracy'].idxmax()
            best_train_prompt = per_prompt_df.iloc[best_train_prompt_idx]
            #print('Best train prompt', best_train_prompt['Train mean loss of correct answer'], best_train_prompt['Test mean loss of correct answer'], best_train_prompt['Train accuracy'], best_train_prompt['Test accuracy'])
            best_test_prompt_idx = per_prompt_df['Test accuracy'].idxmax()
            best_test_prompt = per_prompt_df.iloc[best_test_prompt_idx]
            #print('Best test prompt', best_test_prompt['Train mean loss of correct answer'], best_test_prompt['Test mean loss of correct answer'], best_test_prompt['Train accuracy'], best_test_prompt['Test accuracy'])
            # compare to best test prompt
            best_train_train_acc = best_train_prompt['Train accuracy']
            best_train_test_acc = best_train_prompt['Test accuracy']


            #average_train_acc_all_prompts = per_prompt_df['Train accuracy'].mean()

            # size of train dataset!!!! not the whole dataset!!
            max_random = max_random_baseline(num_train, p, t)


            t_to_detailed_train_argmax_tidy_data[t].append({'Dataset': dataset,
                                                   'Model': f'{model_name}, {model_type}',
                                                   'Number of shots': num_shots,
                                                   'Split number': trial,
                                                   'Above max random on train': int(best_train_train_acc > max_random),
                                                   'Above standard random on train': int(best_train_train_acc > p),
                                                   'Max random F': max_random_F(num_train * best_train_train_acc, num_train, p, t),
                                                   'Standard random F': binomial_F(num_train * best_train_train_acc, num_train, p),
                                                   'Above standard random on test': int(best_train_test_acc > p),
                                                  })

for dataset in ['code_line_description', 'bbq_lite_json', 'hindu_knowledge',
                'conceptual_combinations', 'formal_fallacies_syllogisms_negation',
                'known_unknowns', 'logical_deduction', 'play_dialog_same_or_different',
                'strange_stories', 'symbol_interpretation', 'winowhy', 'strategyqa',
                'novel_concepts', 'emoji_movie', 'vitaminc_fact_verification',
                'language_identification']:
    for model in ['Llama-2-7b', 'Alpaca-7b',
                  'OLMo-7B', 'OLMo-7B-Instruct',
                  'falcon-7b', 'falcon-7b-instruct']:
        for num_shots in [1,2,4]:
            train_test_split(dataset, model, num_shots)

# df = pd.DataFrame(train_argmax_tidy_data)
# df.to_csv(f'all-train-test-splits_{NUM_TRIALS}-trials.csv', index=False)
for t in all_ts:
    df = pd.DataFrame(t_to_detailed_train_argmax_tidy_data[t])
    df.to_csv(f'./held-out-split-results/all-train-test-splits_{NUM_TRIALS}-trials_t={t}_all-predictions.csv', index=False)

