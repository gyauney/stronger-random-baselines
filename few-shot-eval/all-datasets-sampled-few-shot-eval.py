import numpy as np
import argparse
from collections import defaultdict
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import torch
import json
import os
import random
import time
import hf_olmo
from transformers import BitsAndBytesConfig

from data_helper import load_few_shot_dataset

# Replace with the path to your Alpaca weights
ALPACA_PATH = None

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--real_prefix', required=True, type=bool)
parser.add_argument('--num_shots', required=True, type=int)
parser.add_argument('--batch_size', required=False, type=int, default=8)
parser.add_argument('--start_at_dataset', required=False, type=int, default=0)
parser.add_argument('--stop_after_dataset', required=False, type=int, default=-1)
args = parser.parse_args()

rng = np.random.default_rng()

BATCH_SIZE = args.batch_size
START_IDX = 0
NUM_PROMPTS = 200
NUM_TO_SUBSAMPLE = 200
NUM_SHOTS = args.num_shots
PREFIX_HAS_REAL_LABELS = args.real_prefix

device = "cuda"

if args.model == 'Llama-2-7b-hf':
  model_string = "Llama-2-7b-hf"
  model_filename_string = model_string
  model_id = "meta-llama/Llama-2-7b-hf"
  model = LlamaForCausalLM.from_pretrained(model_id).to(device)
  model.eval()
elif args.model == 'falcon-7b':
  model_string = "falcon-7b"
  model_filename_string = model_string
  model_id = "tiiuae/falcon-7b"
  model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
  model.eval()
elif args.model == 'falcon-7b-quantized':
  model_string = "falcon-7b-quantized"
  model_filename_string = model_string
  model_id = "tiiuae/falcon-7b"
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  # don't need to set device for quantized model, already gpu
  model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
  model.eval()
elif args.model == 'falcon-7b-instruct':
  model_string = "falcon-7b-instruct"
  model_filename_string = model_string
  model_id = "tiiuae/falcon-7b-instruct"
  model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
  model.eval()
elif args.model == 'falcon-7b-instruct-quantized':
  model_string = "falcon-7b-instruct-quantized"
  model_filename_string = model_string
  model_id = "tiiuae/falcon-7b-instruct"
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  # don't need to set device for quantized model, already gpu
  model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
  model.eval()
elif args.model == 'Llama-2-7b-hf-quantized':
  model_string = "Llama-2-7b-hf-quantized"
  model_filename_string = model_string
  model_id = "meta-llama/Llama-2-7b-hf"
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  # don't need to set device for quantized model, already gpu
  model = LlamaForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
  model.eval()
elif args.model == 'Llama-2-70b-hf':
  model_string = "Llama-2-70b-hf"
  model_filename_string = model_string
  model_id = "meta-llama/Llama-2-70b-hf"
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  # don't need to set device for quantized model, already gpu
  model = LlamaForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
  model.eval()
elif args.model == 'Alpaca-7b':
  model_string = "Alpaca-7b"
  model_filename_string = "Alpaca-7b"
  model_id = "/home/gjy24/alpaca-7b-hf"
  model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
  model.eval()
elif args.model == 'Alpaca-7b-quantized':
  if not ALPACA_PATH:
    print('You must specify a path to Alpaca-7b weights!')
    exit()
  model_string = "Alpaca-7b-quantized"
  model_filename_string = "Alpaca-7b-quantized"
  model_id = ALPACA_PATH
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  # don't need to set device for quantized model, already gpu
  model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
  model.eval()
elif args.model == 'OLMo-7B':
  model_string = 'OLMo-7B'
  model_filename_string = model_string
  model_id = "allenai/OLMo-7B"
  model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
  model.eval()
elif args.model == 'OLMo-7B-quantized':
  model_string = 'OLMo-7B-quantized'
  model_filename_string = model_string
  model_id = "allenai/OLMo-7B"
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  # don't need to set device for quantized model, already gpu
  model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
  model.eval()
elif args.model == 'OLMo-7B-Instruct':
  model_string = 'OLMo-7B-Instruct'
  model_filename_string = model_string
  model_id = "allenai/OLMo-7B-Instruct"
  model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
  model.eval()
elif args.model == 'OLMo-7B-Instruct-quantized':
  model_string = 'OLMo-7B-Instruct-quantized'
  model_filename_string = model_string
  model_id = "allenai/OLMo-7B"
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  # don't need to set device for quantized model, already gpu
  model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
  model.eval()
elif args.model == 'Pythia-6.9B':
  model_string = args.model
  model_filename_string = model_string.lower()
  model_id = f"EleutherAI/{model_filename_string}"
  model = GPTNeoXForCausalLM.from_pretrained(model_id).to(device)
  model.eval()
else:
  print(f'Model not found: {args.model}')
  exit()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

output_dir = f'./output-{model_filename_string}_sampled_{NUM_SHOTS}-shot_random-labels'
if PREFIX_HAS_REAL_LABELS:
  output_dir += '_prefix-has-real-labels'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists('./subsampled-idxs'):
    os.makedirs('./subsampled-idxs')

print('loaded the model!')

def get_random_labeling_one_hot(num_examples, num_choices):
  correct_choice_nums = rng.integers(low=0, high=num_choices, size=num_examples)
  random_labeling_one_hot = np.zeros((num_examples, num_choices))
  # for some reason, indexing isn't working right here......
  for i, correct_choice in enumerate(correct_choice_nums):
    random_labeling_one_hot[i, correct_choice] = 1
  return random_labeling_one_hot.tolist()

  
datasets =  [
            ('bigbench', 'code_line_description', None),
            ('bigbench', 'bbq_lite_json', None),
            ('bigbench', 'hindu_knowledge', None), 
            ('bigbench', 'novel_concepts', None),
            ('bigbench', 'emoji_movie', None),
            ('bigbench', 'vitaminc_fact_verification', None),
            ('bigbench', 'conceptual_combinations', None),
            ('bigbench', 'formal_fallacies_syllogisms_negation', None),
            ('bigbench', 'known_unknowns', None),
            ('bigbench', 'logical_deduction', None),
            ('bigbench', 'play_dialog_same_or_different', None), 
            ('bigbench', 'strange_stories', None), 
            ('bigbench', 'symbol_interpretation', None), 
            ('bigbench', 'winowhy', None),
            ('bigbench', 'strategyqa', None),
            ('bigbench', 'language_identification', None)
            ]

# all multiple choice tasks
for i, (benchmark_suite, dataset_name, extra_params) in enumerate(datasets):
  if i < args.start_at_dataset:
    continue
  if args.stop_after_dataset > 0 and i > args.stop_after_dataset:
    break
  for prompt_idx in range(START_IDX, START_IDX + NUM_PROMPTS):
    name = f'prompt-{prompt_idx}'
    results_filename = f'{output_dir}/{dataset_name}_{name}.json'
    # skip this labeling if these results already exist
    if os.path.isfile(results_filename):
      continue

    results = defaultdict(list)
    print()
    print(dataset_name, prompt_idx)
    use_random_labels = False
    dataset, few_shot_prefix, idxs_few_shot = load_few_shot_dataset(benchmark_suite, dataset_name, tokenizer, 
                                    subsample=True,
                                    batch_size=BATCH_SIZE,
                                    num_shots=NUM_SHOTS,
                                    num_to_subsample=NUM_TO_SUBSAMPLE, 
                                    use_random_labels=use_random_labels,
                                    prefix_has_real_labels=PREFIX_HAS_REAL_LABELS,
                                    extras=extra_params)
    print(f'Prompt number {prompt_idx}: {few_shot_prefix}')
    print(model_filename_string, NUM_SHOTS)
    
    start = time.time()
    # pull batches instead of single examples
    for start_idx in tqdm(range(0, len(dataset), BATCH_SIZE)):
      end_idx = min(len(dataset), start_idx + BATCH_SIZE)
      batch = dataset[start_idx:end_idx]
      this_batch_size = end_idx - start_idx
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      target_ids = batch['target_ids'].to(device)
      all_num_choice_tokens = batch['num_choice_tokens']

      with torch.no_grad():
          outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
      max_seq_length = target_ids.shape[1]
      if max_seq_length > 2000:
        print(f'POSSIBLY OVERFLOWING CONTEXT WINDOW! Maximum sequence length: {max_seq_length}')
      lm_logits = outputs.logits
      labels = target_ids
      # from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
      shift_logits = lm_logits[:, :-1, :].contiguous()
      labels = labels[:, 1:].contiguous()
      # the following line has the only difference: don't average loss over all tokens
      loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
      lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
      # pull out the per-example loss (remember: shifted)
      per_token_loss = lm_loss.view(this_batch_size, max_seq_length - 1).to('cpu')
      per_example_loss = torch.div(per_token_loss.sum(dim=1), all_num_choice_tokens)
      results['dataset'].extend([dataset_name] * this_batch_size)
      results['combined_idx'].extend(batch['idx'])
      results['example_idx'].extend(batch['example_idx'].tolist())
      results['answer_idx'].extend(batch['answer_idx'].tolist())
      results['score'].extend(batch['score'].tolist())
      results['num_choice_tokens'].extend(all_num_choice_tokens.tolist())
      results['loss'].extend(per_example_loss.tolist())
      del input_ids
      del attention_mask
      del target_ids
    end = time.time()
    results['time'].append(end - start)
    results['few_shot_prefix'] = few_shot_prefix
    results['idxs_few_shot'] = idxs_few_shot
    with open(results_filename, 'w') as f:
      json.dump(results, f)
      
    torch.cuda.empty_cache()