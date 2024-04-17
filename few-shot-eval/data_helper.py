import numpy as np
import argparse
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import json
import os
import random
import time
import math

rng = np.random.default_rng()

superglue_dataset_to_domain_prefix = {
  'boolq': 'Answer: ',
  'axb': 'True or False? ',
  'axg': 'True or False? ',
  'cb': 'True, False, or Neither?',
  'wic': 'used in the same sense in the two sentences above? ',
  'rte': 'True or False? ',
} 

dataset_to_possible_choices = {
  'boolq': ['No', 'Yes'],
  'axb': ['Yes', 'No'],
  'axg': ['Yes', 'No'],
  'cb': ['Yes', 'No', 'Maybe'],
  'wic': ['No', 'Yes'],
  'rte': ['Yes', 'No'],
}

def prompt_fn(dataset, example, choice):
  if dataset == 'rte':
    return f"{example['premise'][0]} {choice}, {example['hypothesis'][0]}"
  elif dataset == 'axb':
    return f"{example['sentence1'][0]} {choice}, {example['sentence2'][0]}"
  elif dataset == 'axg':
    return f"{example['premise'][0]} {choice}, {example['hypothesis'][0]}"
  elif dataset == 'boolq':
    return f"{example['passage'][0]}\nQuestion: {example['question'][0]}\n Answer: {choice}"
  elif dataset == 'cb':
    return f"{example['premise'][0]} {choice}, {example['hypothesis'][0]}"
  elif dataset == 'wic':
    return f"{example['sentence1'][0]}\n{example['sentence2'][0]}\nQuestion: Is the word {example['word'][0]} used in the same sense in the two sentences above? {choice}"
  else:
    print(f'Dataset not implemented: {dataset}')
    exit()

def prompt_fn_unbatched(dataset, example, choice):
  if dataset == 'rte':
    return f"{example['premise']} {choice}, {example['hypothesis']}"
  elif dataset == 'axb':
    return f"{example['sentence1']} {choice}, {example['sentence2']}"
  elif dataset == 'axg':
    return f"{example['premise']} {choice}, {example['hypothesis']}"
  elif dataset == 'boolq':
    return f"{example['passage']}\nQuestion: {example['question']}\n Answer: {choice}"
  elif dataset == 'cb':
    return f"{example['premise']} {choice}, {example['hypothesis']}"
  elif dataset == 'wic':
    return f"{example['sentence1']}\n{example['sentence2']}\nQuestion: Is the word {example['word']} used in the same sense in the two sentences above? {choice}"
  else:
    print(f'Dataset not implemented: {dataset}')
    exit()

# generate random labels for all examples as one-hot vectors
# assumes all examples have the same number of choices
# which is not true for all of bigbench...
def get_random_labeling(num_examples, num_choices, one_hot):
  correct_choice_nums = rng.integers(low=0, high=num_choices, size=num_examples)
  if not one_hot:
    return correct_choice_nums.tolist()
  random_labeling_one_hot = np.zeros((num_examples, num_choices))
  # for some reason, indexing isn't working right here......
  for i, correct_choice in enumerate(correct_choice_nums):
    random_labeling_one_hot[i, correct_choice] = 1
  return random_labeling_one_hot.tolist()

# TODO for some reason the idxs_to_sample_from is not the same as the actual example ids in bigbench
def subsample_dataset(dataset_name, dataset, subsample, num_shots, num_to_subsample, idxs_to_sample_from, subsampled_idxs_filename):
  # turn set into a list for sampling with random.sample()
  idxs_to_sample_from = list(idxs_to_sample_from)
  # First subsample the dataset
  if subsample and num_to_subsample < (len(idxs_to_sample_from) - num_shots):
    if os.path.exists(subsampled_idxs_filename):
      print(f'{dataset_name}: reading {num_to_subsample} out of {len(dataset)} previously sampled idxs.')
      with open(subsampled_idxs_filename, 'r') as f:
        idxs_to_subsample = json.load(f)
    else:
      print(f'{dataset_name}: sampling {num_to_subsample} out of {len(dataset)} examples.')
      idxs_to_subsample = random.sample(idxs_to_sample_from, num_to_subsample)
      with open(subsampled_idxs_filename, 'w') as f:
        json.dump(idxs_to_subsample, f)
  else:
    # we're not subsampling the dataset but we might need to 
    # remove examples used for few-shot demonstrations if they exist
    idxs_to_subsample = idxs_to_sample_from
  dataset = dataset.select(idxs_to_subsample)
  return dataset



# n.b. prefix always has real labels in this setup
def get_few_shot_prefix_bigbench(few_shot_examples):
  prefix = ''
  for example in few_shot_examples:
    # n.b. relies on only one right answer!
    for i, (choice, score) in enumerate(zip(example['multiple_choice_targets'], example['multiple_choice_scores'])):
      if score:
        prefix += f'{example["inputs"]} {choice}\n'
  return prefix

def load_bigbench_dataset(dataset_name, tokenizer, num_shots,
                          subsample, batch_size, num_to_subsample,
                          use_random_labels, prefix_has_real_labels):
  
  dataset = load_dataset("bigbench", dataset_name, split='default', trust_remote_code=True)
  few_shot_prefix = ''

  # If the prefix has random labels, randomize labels before making the prefix
  if use_random_labels and not prefix_has_real_labels:
    random_labeling_one_hot = get_random_labeling(len(dataset),
                                                  len(dataset[0]['multiple_choice_targets']),
                                                  one_hot=True)
    dataset = dataset.remove_columns(["multiple_choice_scores"])
    dataset = dataset.add_column("multiple_choice_scores", random_labeling_one_hot)

  # TODO change how train/val is split!
  # halfway = math.floor(len(dataset)/2)
  # train_set_idxs = range(halfway)
  # val_set_idxs = range(halfway+1, len(dataset))
  all_idxs = range(len(dataset))

  # If small enough dataset: just remove the few-shot examples from the validation
  if len(dataset) <= num_to_subsample:
    idxs_few_shot = random.sample(all_idxs, num_shots)
    examples_few_shot = [dataset[i] for i in idxs_few_shot]
    few_shot_prefix = get_few_shot_prefix_bigbench(examples_few_shot)
    idxs_to_subsample = set(all_idxs) - set(idxs_few_shot)
    dataset = dataset.select(idxs_to_subsample)
  else:
    # If big dataset:
    # 1. fixed subsample of validation examples
    subsampled_idxs_filename = f'./subsampled-idxs/{dataset_name}_{num_to_subsample}-examples_idxs.json'
    if os.path.exists(subsampled_idxs_filename):
      print(f'{dataset_name}: reading {num_to_subsample} out of {len(dataset)} previously sampled idxs.')
      with open(subsampled_idxs_filename, 'r') as f:
        idxs_to_subsample = json.load(f)
    else:
      print(f'{dataset_name}: sampling {num_to_subsample} out of {len(dataset)} examples.')
      idxs_to_subsample = random.sample(all_idxs, num_to_subsample)
      with open(subsampled_idxs_filename, 'w') as f:
        json.dump(idxs_to_subsample, f)
    # 2. choose few-shot from remaining 
    possible_few_shot_idxs = set(all_idxs) - set(idxs_to_subsample)
    idxs_few_shot = random.sample(list(possible_few_shot_idxs), num_shots)
    examples_few_shot = [dataset[i] for i in idxs_few_shot]
    few_shot_prefix = get_few_shot_prefix_bigbench(examples_few_shot)
    # finally subsample
    dataset = dataset.select(idxs_to_subsample)

  # If the prefix has real labels, randomize labels after making the prefix
  if use_random_labels and prefix_has_real_labels:
    random_labeling_one_hot = get_random_labeling(len(dataset),
                                                  len(dataset[0]['multiple_choice_targets']),
                                                  one_hot=True)
    dataset = dataset.remove_columns(["multiple_choice_scores"])
    dataset = dataset.add_column("multiple_choice_scores", random_labeling_one_hot)

  print(f'Dataset size: {len(dataset)}')
  print('Few shot prompt idxs:', idxs_few_shot)
  print('Evaluation idxs:', idxs_to_subsample)


  # create new examples, each with a different target choice
  def concatenate_choices(example):
    batch = defaultdict(list)
    for i, (choice, score) in enumerate(zip(example['multiple_choice_targets'][0], example['multiple_choice_scores'][0])):
      batch['idx'].append(f'{example["idx"][0]}_{i}')
      batch['example_idx'].append(example["idx"][0])
      batch['answer_idx'].append(i)
      batch['inputs'].append(f'{few_shot_prefix}{example["inputs"][0]} {choice}')
      batch['inputs_without_choice'].append(example["inputs"][0])
      batch['score'].append(score)
      batch['choice'].append(choice)
      batch['targets'].append('')
      batch['multiple_choice_targets'].append('')
      batch['multiple_choice_scores'].append('')
    return batch
  def mask_question_tokens(example):
    example['target_ids'] = np.array(example['input_ids'].copy())
    choice_tokens = tokenizer(example["choice"], return_tensors="np", max_length=512, padding=False, truncation=True)
    num_choice_tokens = choice_tokens.input_ids.shape[1]
    zero_idxs = np.where(np.array(example['attention_mask']) == 0)[0]
    if len(zero_idxs) == 0:
      last_idx = len(example['input_ids'])
    else:
      last_idx = zero_idxs[0]
    example['num_choice_tokens'] = num_choice_tokens
    example['target_ids'][:last_idx-num_choice_tokens] = -100
    example['target_ids'][last_idx:] = -100
    return example
  dataset = dataset.map(concatenate_choices, batched=True, batch_size=1)
  dataset = dataset.remove_columns(["targets", "multiple_choice_targets", "multiple_choice_scores"])
  dataset = dataset.map(lambda examples: tokenizer(examples["inputs"], return_tensors="np", max_length=512, padding=True, truncation=True), batched=True, batch_size=batch_size)
  dataset = dataset.map(mask_question_tokens, batched=False)
  
  dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'target_ids', 'num_choice_tokens',
                                       'idx', 'example_idx', 'answer_idx', 'score'])

  return dataset, few_shot_prefix, idxs_few_shot





def get_few_shot_prefix_superglue(dataset_name, few_shot_examples, possible_choices, prefix_has_real_labels):
    prefix = ''
    for example in few_shot_examples:
      if prefix_has_real_labels:
        label = possible_choices[example['label']]
      else:
        random_label_idx = rng.integers(low=0, high=len(possible_choices))
        label = possible_choices[random_label_idx]
        
      prefix += prompt_fn_unbatched(dataset_name,
                                    example,
                                    label)
      prefix += '\n'
    return prefix


def load_superglue_dataset(dataset_name, tokenizer, split, num_shots,
                           prompt_name, subsample, batch_size, num_to_subsample,
                           use_random_labels, prefix_has_real_labels):

  dataset = load_dataset("super_glue", dataset_name, split=split)
  few_shot_prefix = ''
  possible_choices = dataset_to_possible_choices[dataset_name]


  # If the prefix has random labels, randomize labels before making the prefix
  if use_random_labels and not prefix_has_real_labels:
    random_labeling = get_random_labeling(len(dataset), len(possible_choices),
                                          one_hot=False)
    dataset = dataset.remove_columns(["label"])
    dataset = dataset.add_column("label", random_labeling)

  # TODO get few-shot prefix from training and the rest from eval
  if num_shots:
    train_dataset = load_dataset("super_glue", dataset_name, split='train')
    idxs_few_shot = random.sample(range(len(train_dataset)), num_shots)
    examples_few_shot = [train_dataset[i] for i in idxs_few_shot]
    few_shot_prefix = get_few_shot_prefix_superglue(dataset_name,
                                                    examples_few_shot,
                                                    possible_choices,
                                                    prefix_has_real_labels)
  idxs_to_sample_from = set(range(len(dataset)))

  # If the prefix has real labels, randomize labels after making the prefix
  if use_random_labels and prefix_has_real_labels:
    random_labeling = get_random_labeling(len(dataset), len(possible_choices),
                                          one_hot=False)
    dataset = dataset.remove_columns(["label"])
    dataset = dataset.add_column("label", random_labeling)

  # First subsample the dataset
  subsampled_idxs_filename = f'./subsampled-idxs/{dataset_name}_{split}_{num_to_subsample}-examples_idxs.json'
  dataset = subsample_dataset(dataset_name, dataset, subsample, num_shots, num_to_subsample, idxs_to_sample_from, subsampled_idxs_filename)
  
  # n.b. example is a list of length 1 containing a single example!!!
  def concatenate_choices(example):
    batch = defaultdict(list)
    all_keys = set(example.keys())
    for i, choice in enumerate(possible_choices):
      input_with_choice_pretokenized = prompt_fn(dataset_name, example, choice)
      answer_pretokenized = possible_choices[example["label"][0]]
      batch['idx'].append(f'{example["idx"][0]}_{i}')
      batch['example_idx'].append(example["idx"][0])
      batch['answer_idx'].append(i)
      batch['inputs'].append(f'{few_shot_prefix}{input_with_choice_pretokenized}')
      batch['score'].append(int(choice == answer_pretokenized))
      batch['answer_pretokenized'].append(answer_pretokenized)
      batch['choice'].append(choice)
      # fill up the other fields with placeholders of the correct type
      for k in all_keys - set(['idx', 'answer_pretokenized']):
        batch[k].append(type(example[k][0])())
    return batch
  def mask_question_tokens(example):
    example['target_ids'] = np.array(example['input_ids'].copy())
    choice_tokens = tokenizer(example["choice"], return_tensors="np", max_length=512, padding=False, truncation=True)
    num_choice_tokens = choice_tokens.input_ids.shape[1]
    zero_idxs = np.where(np.array(example['attention_mask']) == 0)[0]
    if len(zero_idxs) == 0:
      last_idx = len(example['input_ids'])
    else:
      last_idx = zero_idxs[0]
    example['num_choice_tokens'] = num_choice_tokens
    example['target_ids'][:last_idx-num_choice_tokens] = -100
    example['target_ids'][last_idx:] = -100
    return example
  dataset = dataset.map(concatenate_choices, batched=True, batch_size=1)
  dataset = dataset.map(lambda examples: tokenizer(examples["inputs"], return_tensors="np", max_length=512, padding=True, truncation=True), batched=True, batch_size=batch_size)
  dataset = dataset.map(mask_question_tokens, batched=False)

  dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'target_ids', 'num_choice_tokens',
                                       'idx', 'example_idx', 'answer_idx', 'score'])

  return dataset, few_shot_prefix, idxs_few_shot







def load_few_shot_dataset(benchmark_suite, dataset_name, tokenizer, 
                          subsample, batch_size, num_shots=None, num_to_subsample=None, 
                          use_random_labels=False,
                          prefix_has_real_labels=True,
                          extras=None):
  if benchmark_suite == 'bigbench':
    return load_bigbench_dataset(dataset_name, tokenizer, num_shots, 
                                 subsample, batch_size,
                                 num_to_subsample,
                                 use_random_labels, prefix_has_real_labels)
  elif benchmark_suite == 'superglue':
    return load_superglue_dataset(dataset_name, tokenizer, extras['split'], num_shots, 
                                  extras['prompt_name'], subsample, batch_size,
                                  num_to_subsample,
                                  use_random_labels, prefix_has_real_labels)
  else:
    print(f'Benchmark suite not recognized: {benchmark_suite}, {dataset_name}')
    exit()
  print('Should not have gotten here!')
  exit()