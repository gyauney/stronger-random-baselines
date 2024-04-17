import math
import numpy as np
import pandas as pd
import random
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.patheffects as path_effects
plt.rcParams["font.family"] = 'DejaVu Sans'

from max_random_baseline import max_random_baseline, MaxOrderStatisticPoissonBinomial
from approximate_max_random_baseline import approximate_max_random_baseline


ns = [1.00000000e+01, 1.77827941e+01, 3.16227766e+01, 5.62341325e+01,
       1.00000000e+02, 1.77827941e+02, 3.16227766e+02, 5.62341325e+02,
       1.00000000e+03, 1.77827941e+03, 3.16227766e+03, 5.62341325e+03,
       1.00000000e+04,]
ts = [1.00000000e+00, 2, 3, 5,
       1.00000000e+01, 1.77827941e+01, 3.16227766e+01, 5.62341325e+01,
       1.00000000e+02, 1.77827941e+02, 3.16227766e+02, 5.62341325e+02,
       1.00000000e+03, 1.77827941e+03, 3.16227766e+03, 5.62341325e+03,
       1.00000000e+04,] 
n_grid, t_grid = np.meshgrid(ns, ts)
all_num_choices = [2,3,4,5]
all_major_contours = [[0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                  [0.38, 0.43, 0.48, 0.53, 0.58, 0.63, 0.68, 0.73, 0.78, 0.83, 0.88, 0.93, 0.98],
                  [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                  [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]
all_minor_contours = [[.525, .575, .625],
                      [.355, .405, .455],
                      [.275, .325, .375],
                      [.225, .275, .325]]
cmap = sns.color_palette("Blues", as_cmap=True)

# save standalone for the main paper
num_choices = 2
major_contours = all_major_contours[0]
minor_contours = all_minor_contours[0]
f = plt.figure(figsize=(6.4,4.8))
ax = plt.axes()
E_grid = np.empty((n_grid.shape[0], n_grid.shape[1]))
for row in range(n_grid.shape[0]):
    for col in range(n_grid.shape[1]):
        n = n_grid[row, col]
        t = t_grid[row, col]
        p = 1 / num_choices
        n = int(n)
        t = round(t)
        E = max_random_baseline(n,p,t) #(1/n) * approximate_expectation(n, p, t)
        print(num_choices, n, t, E)
        E_grid[row,col] = E
ax.set_xscale('log')
ax.set_yscale('log')
cs = ax.contourf(t_grid, n_grid, E_grid, levels= sorted(major_contours + minor_contours), extend='both', cmap=cmap)
cs.cmap.set_over('#023858')
contours_for_labels = ax.contour(t_grid, n_grid, E_grid, levels=major_contours, linewidths=1, colors='#000000')
labels = ax.clabel(contours_for_labels, inline=True, colors='#000000', inline_spacing=10, use_clabeltext=True, fontsize=10)
ax.contour(t_grid, n_grid, E_grid, levels=minor_contours, linewidths=1, colors='#000000', alpha=0.2)
for label in labels:
     label.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
t = ax.text(0.125, 5/6, '0.50', va='center', ha='center', transform=ax.transAxes, fontsize=10, color='#000000')
t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
ax.set_ylabel('n = size of dataset', fontsize=14, fontweight='bold')
ax.set_xlabel('t = number of validation set evaluations', fontsize=14, labelpad=10, fontweight='bold')
ax.set_xticks([1, 10, 100, 1000, 10000], ['1', '10', '100', '1,000', '10,000'], fontsize=10)
ax.set_yticks([10, 100, 1000, 10000], ['10', '100', '1,000', '10,000'], fontsize=10)
#ax.axhline(y=100, color='#d7191c', linestyle='--', linewidth=2)
#ax.set_title('p = 1/2', fontsize=16, pad=12)
#f.suptitle('Dataset with 2 choices per example', fontsize=20, fontweight='bold', y=1.04)
ax.set_title('Dataset with 2 choices per example', fontsize=16, pad=12, fontweight='bold')
ax.grid(alpha=0.5, axis='both', linestyle='--')
ax.set_axisbelow('line')
f.savefig(f'figure-2.pdf', bbox_inches='tight')

# evaluate more number of choices and more points for appendex
ns = [1.00000000e+01, 1.77827941e+01, 3.16227766e+01, 5.62341325e+01,
       1.00000000e+02, 1.77827941e+02, 3.16227766e+02, 5.62341325e+02,
       1.00000000e+03, 1.77827941e+03, 3.16227766e+03, 5.62341325e+03,
       1.00000000e+04, 1.77827941e+04, 3.16227766e+04, 5.62341325e+04,
       1.00000000e+05,] #1.77827941e+05, 3.16227766e+05, 5.62341325e+05,
       #1.00000000e+06]
ts = [1.00000000e+00, 2, 3, 5,
       1.00000000e+01, 1.77827941e+01, 3.16227766e+01, 5.62341325e+01,
       1.00000000e+02, 1.77827941e+02, 3.16227766e+02, 5.62341325e+02,
       1.00000000e+03, 1.77827941e+03, 3.16227766e+03, 5.62341325e+03,
       1.00000000e+04, 1.77827941e+04, 3.16227766e+04, 5.62341325e+04,
       1.00000000e+05,]
n_grid, t_grid = np.meshgrid(ns, ts)
f, axs = plt.subplots(1, len(all_num_choices), figsize=(6.4 * len(all_num_choices), 4.8))
plt.subplots_adjust(wspace=0.25)
for ax, num_choices, major_contours, minor_contours in zip(axs, all_num_choices, all_major_contours, all_minor_contours):
    E_grid = np.empty((n_grid.shape[0], n_grid.shape[1]))
    for row in range(n_grid.shape[0]):
        for col in range(n_grid.shape[1]):
            n = n_grid[row, col]
            t = t_grid[row, col]
            p = 1 / num_choices
            n = int(n)
            t = round(t)
            E = approximate_max_random_baseline(n,p,t) #(1/n) * approximate_expectation(n, p, t)
            print(num_choices, n, t, E)
            E_grid[row,col] = E
    ax.set_xscale('log')
    ax.set_yscale('log')
    cs = ax.contourf(t_grid, n_grid, E_grid, levels= sorted(major_contours + minor_contours), extend='both', cmap=cmap)
    cs.cmap.set_over('#023858')
    contours_for_labels = ax.contour(t_grid, n_grid, E_grid, levels=major_contours, linewidths=1, colors='#000000')
    labels = ax.clabel(contours_for_labels, inline=True, colors='#000000', inline_spacing=10, use_clabeltext=True, fontsize=10)
    ax.contour(t_grid, n_grid, E_grid, levels=minor_contours, linewidths=1, colors='#000000', alpha=0.2)
    for label in labels:
        label.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    t = ax.text(0.1, 0.625, f'{round(1/num_choices,2):.2f}', va='center', ha='center', transform=ax.transAxes, fontsize=10, color='#000000')
    t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000], ['1', '10', '100', '1,000', '10,000', '100,000'], fontsize=10)
    ax.set_yticks([10, 100, 1000, 10000, 100000], ['10', '100', '1,000', '10,000', '100,000'], fontsize=10)
    ax.set_title(f'p = 1/{num_choices}', fontsize=14, pad=12, fontweight='bold')
    ax.text(0.5, 1.15, f'Dataset with {num_choices} choices per example', transform=ax.transAxes, fontsize=16, fontweight='bold', ha='center')
    ax.grid(alpha=0.5, axis='both', linestyle='--')
    ax.set_axisbelow('line')
    ax.set_xlabel('t = number of validation set evaluations', fontsize=14, labelpad=10, fontweight='bold')
#f.text(0.5, -0.04, 't = number of validation set evaluations', fontsize=18, ha='center', fontweight='bold')
f.text(0.08, 0.5, 'n = size of dataset', fontsize=14, va='center', rotation='vertical', fontweight='bold')
savefig(f'figure-6.pdf', bbox_inches='tight')
plt.close()



