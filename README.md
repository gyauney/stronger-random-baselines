# Stronger-random-baselines

This repository contains code and data for reproducing the results from
"Stronger Random Baselines for In-Context Learning".
The standalone expected maximum random baseline is available
[here](https://github.com/gyauney/max-random-baseline).

## Requirements

Reproducing the figures from cached results requires `numpy`, `scipy`, and `mpmath`.

We also require the [`show_your_work`](https://github.com/dodgejesse/show_your_work)
package to calculate expected max validation accuracy and the
[`PoiBin` Python implementation](https://github.com/tsakim/poibin) of Poisson
binomial random variables. You can install them with:

```
git clone https://github.com/dodgejesse/show_your_work.git
touch show_your_work/__init__.py
git clone https://github.com/tsakim/poibin.git
touch poibin/__init__.py
```

## Reproducing figures and results

The `results` directory contains all individual and aggregate few-shot results used in the paper.

To reproduce the figures and tables in the paper:
- `figure-1.py`: Generates Figure 1
- `expected-max-accuracy-contour-plots.py`: Generates Figures 2 and 6.
- `figure-3.py`: Generates Figure 3.
- `figure-4.py`: Generates Figure 4.
- `full-results.py`: Generates Table 1 and Figures 11, 12, 13.
- `process-held-out-predictions.py`: Generates Figures 5 and 10. Also generates Tables 2 and 3.
- `p-value-contours.py`: Generates Figure 7.
- `figure-8.py`: Generates Figure 8.
- `figure-9.py`: Generates Figure 9.

## Regenerating experimental results

Use pip to install all of the requirements:

```
pip install -r requirements.txt
```

The `few-shot-eval` directory contains code for reproducing all of the paper's few-shot results from scratch. It also includes which examples were subsampled from large datasets.
Supported models are `Llama-2-7b-hf`, `Alpaca-7b` (if you supply your own weights), `OLMo-7B`, `OLMo-7B-Instruct`, `falcon-7b`, and `falcon-7b-instruct`. 

For example, to run an evaluation:

```
python few-shot-eval/all-datasets-sampled-few-shot-eval.py --dataset bigbench --num_shots 2 --model OLMo-7B-Instruct
```

The file `bigbench-held-out-splits.py` regenerates the data in `held-out-split-results`.