import numpy as np
import pandas as pd
import random
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
from matplotlib.patches import ConnectionPatch
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams["font.family"] = 'DejaVu Sans'

from scipy.stats import binom as binomial_dist
import math
import json

from dataset_constants import BIGBENCH_LITE_DETAILS_NUM_CHOICES

from max_random_baseline import max_random_baseline, max_random_pmf

with open('results/aggregated-results.json', 'r') as f:
    full_data = json.load(f)

dataset = 'emoji_movie'
model = 'OLMo-7B'
num_shots = 4
n, num_choices = BIGBENCH_LITE_DETAILS_NUM_CHOICES[dataset]
t = 200
if n <= 200 and num_shots == 1:
    t = n
if n <= 200:
    n = n - num_shots
n = min(200, n)
p = 1 / num_choices

reals = full_data[f'{model}_{dataset}_{num_shots}-shot']


xs = np.arange(0, n)
binomial_outs = binomial_dist.pmf(xs, n, p)
max_outs = [max_random_pmf(x, n, p, t) for x in xs]


# scale domain by 1/n for accuracy
xs = (1/n) * xs

real_color = '#f6cea6'
real_edge_color = '#fcae65'
real_dark_color = '#fc9432'


binomial_expectation = (1/n) * binomial_dist.mean(n, p)
max_expectation = max_random_baseline(n,p,t)

f = plt.figure(figsize=(6.4, 5.2))
gs_top = plt.GridSpec(nrows=4, ncols=1, hspace=0.5, height_ratios=[1,1,1,0.5])
gs_bottom = plt.GridSpec(nrows=4, ncols=1, hspace=0.5, height_ratios=[1,1,1,0.25])

top_ax = f.add_subplot(gs_top[0,:])
middle_axs = [f.add_subplot(gs_top[i,:], sharex=top_ax) for i in range(1, 3)]
bottom_ax = f.add_subplot(gs_bottom[3,:])
axs = [top_ax] + middle_axs + [bottom_ax]

xmin = 0.05
xmax = 0.42

ax = axs[0]
ax.grid(alpha=0.6, axis='both')
bins = np.linspace(0, 1, 100)
bin_width = 0.5 * (bins[1] - bins[0])
weights = np.ones_like(reals) / len(reals)
ax.hist(reals, bins=bins, weights=weights, align='left', color=real_color, edgecolor=real_edge_color, clip_on=True)
ax.scatter(np.max(reals), 0, s=120, marker='X', clip_on=False, color=real_dark_color, edgecolor='white', zorder=10)
ax.axvline(max(reals), linestyle='--', linewidth=2, color=real_dark_color)
ax.set_xlim((xmin,xmax))
ax.set_xticklabels([])
t = ax.text(0.26, 0.75, 'Accuracy from different prompts', transform=ax.get_xaxis_transform(), color=real_dark_color, fontsize=12, fontweight='bold', ha='left')
t.set_bbox(dict(facecolor='white', alpha=0.6, linewidth=0))

ax = axs[1]
ax.grid(alpha=0.6, axis='both')
ax.bar(xs, binomial_outs, width=1/n, align='center', color='#aac8e1', edgecolor='#7eaed2', clip_on=True)
ax.scatter(binomial_expectation, 0, s=100, marker='o', clip_on=False, color='#2c82de', zorder=10, edgecolor='white')
ax.axvline(binomial_expectation, linestyle='--', linewidth=2, color='#2c82de')
ax.set_xlim((xmin,xmax))
ax.set_xticklabels([])
t = ax.text(0.255, 0.75, 'Random classifier', transform=ax.get_xaxis_transform(), color='#2c82de', fontsize=12, fontweight='bold', ha='left')
t.set_bbox(dict(facecolor='white', alpha=0.6, linewidth=0))

ax = axs[2]
ax.grid(alpha=0.6, axis='both')
ax.bar(xs, max_outs, width=1/n, align='center', color='#8fb3d2', edgecolor='#3d7bb1', clip_on=True)
ax.scatter(max_expectation, 0, s=120, marker='X', clip_on=False, color='#3d7bb1', zorder=10, edgecolor='white')
ax.axvline(max_expectation, linestyle='--', linewidth=2, color='#3d7bb1')
ax.set_xlim((xmin,xmax))
ax.set_xticklabels([])
t = ax.text(0.28, 0.75, 'Best of many random classifiers', transform=ax.get_xaxis_transform(), color='#3d7bb1', fontsize=12, fontweight='bold', ha='right')
t.set_bbox(dict(facecolor='white', alpha=0.6, linewidth=0))

ax = axs[3]
ax.spines[['top', 'left', 'bottom', 'right']].set_visible(False)
ax.grid(False, axis='y')
ax.set_yticks([])
ax.set_ylabel('')
# make the axis
ax.axhline(0, linestyle='-', color='#cccccc', linewidth=1)
ax.scatter([xmin, xmax], [0, 0], s=200, marker='|', linewidth=1, clip_on=False, color='#cccccc', zorder=10)
# plot the summary statistics
ax.scatter(binomial_expectation, 0, s=200, marker='o', edgecolor='white', clip_on=False, color='#2c82de', zorder=10)
ax.scatter(np.max(reals), 0, s=200, marker='X', edgecolor='white', clip_on=False, color=real_dark_color, zorder=10)
ax.scatter(max_expectation, 0, s=200, marker='X', edgecolor='white', clip_on=False, color='#3d7bb1', zorder=10)
ax.set_xlim((xmin,xmax))
ax.set_facecolor('#ffffff00')

# add the lines down and labels
ax.plot([max_expectation, max_expectation], [0, -1.5], transform=ax.get_xaxis_transform(), linestyle='--', linewidth=2, color='#3d7bb1', clip_on=False, zorder=10)
ax.text(max_expectation + 0.005 , -1.75, f"Expected maximum\nrandom accuracy", transform=ax.get_xaxis_transform(), color='#3d7bb1', horizontalalignment='left', va='center', fontweight='bold')

ax.plot([np.max(reals), np.max(reals)], [0, -2.5], transform=ax.get_xaxis_transform(),  linestyle='--', linewidth=2, color=real_dark_color, clip_on=False, zorder=10)
ax.text(np.max(reals), -2.6, f"Best accuracy across prompts", transform=ax.get_xaxis_transform(), color=real_dark_color, horizontalalignment='center', va='top', fontweight='bold')

ax.plot([binomial_expectation, binomial_expectation], [0, -1.5], transform=ax.get_xaxis_transform(), linestyle='--', linewidth=2, color='#2c82de', clip_on=False, zorder=10)
ax.text(binomial_expectation, -1.6, f"Standard random baseline", transform=ax.get_xaxis_transform(), color='#2c82de', horizontalalignment='center', va='top', fontweight='bold')



# add dashed lines between axes
xy = (np.max(reals), 0)
con = ConnectionPatch(xyA=xy,
                      xyB=xy, coordsA="data", coordsB="data",
                      axesA=axs[0], axesB=axs[3], color=real_dark_color,
                      linewidth=2, linestyle='--', zorder=1)
con.set(capstyle='projecting')
f.add_artist(con)

xy = (binomial_expectation, 0)
con = ConnectionPatch(xyA=xy,
                      xyB=xy, coordsA="data", coordsB="data",
                      axesA=axs[1], axesB=axs[3], color='#2c82de',
                      linewidth=2, linestyle='--', zorder=2)
con.set(capstyle='projecting')
f.add_artist(con)

xy = (max_expectation, 0)
con = ConnectionPatch(xyA=xy,
                      xyB=xy, coordsA="data", coordsB="data",
                      axesA=axs[2], axesB=axs[3], color='#3d7bb1',
                      linewidth=2, linestyle='--', zorder=3)
con.set(capstyle='projecting')
f.add_artist(con)

axs[0].set_zorder(1)
axs[1].set_zorder(2)
axs[2].set_zorder(3)
axs[3].set_zorder(4)

axs[1].set_ylabel('Proportion of classifiers', fontsize=16, fontweight='bold', labelpad=20)
axs[3].set_xlabel('Accuracy', fontsize=16, fontweight='bold', labelpad=40)

savefig(f'figure-1.pdf', bbox_inches='tight')







