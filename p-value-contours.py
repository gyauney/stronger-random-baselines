import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
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

import math
from max_random_baseline import MaxOrderStatisticPoissonBinomial

from scipy.ndimage.filters import gaussian_filter



all_num_choices = [2,3,4,5]
all_ts = [1,10,100,200]
f, axs = plt.subplots(len(all_ts), len(all_num_choices), figsize=(6.4 * len(all_num_choices), 4.8 * len(all_ts)))
plt.subplots_adjust(wspace=0.18, hspace=0.22)

for ax_row, t in enumerate(all_ts):
    for ax_col, num_choices in enumerate(all_num_choices):
        p = 1 / num_choices
        ax = axs[ax_row, ax_col]
        ns = np.logspace(1, 4, num=20)
        accs = np.linspace(0, 1, 1000)#1.77827941e+04, 3.16227766e+04, 5.62341325e+04,
               #1.00000000e+05,]
        acc_grid, n_grid = np.meshgrid(accs, ns)
        cmap = sns.color_palette("Blues", as_cmap=True)
        major_contours = [0.01, 0.1, 0.5]
        E_grid = np.empty((n_grid.shape[0], n_grid.shape[1]))
        for row in range(n_grid.shape[0]):
            # n is constant across rows
            n = math.floor(n_grid[row, 0])
            max_order_statistic = MaxOrderStatisticPoissonBinomial(n, p)
            for col in range(n_grid.shape[1]):
                acc = acc_grid[row, col]
                p_val = max_order_statistic.p_value(acc, t)
                print(num_choices, n, acc, round(p_val, 2))
                E_grid[row,col] = p_val
        ax.set_xscale('log')
        ax.set_ylim((0, 1))


        E_grid = gaussian_filter(E_grid, 1)

        cs = ax.contourf(n_grid, acc_grid, E_grid, levels=sorted([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), extend='both', cmap=cmap)
        cs.cmap.set_over('#023858')
        contours_for_labels = ax.contour(n_grid, acc_grid, E_grid, levels=major_contours, linewidths=1, colors='#000000')
        labels = ax.clabel(contours_for_labels, inline=True, colors='#000000', inline_spacing=10, use_clabeltext=True, fontsize=10,)
        #ax.contour(t_grid, n_grid, E_grid, levels=minor_contours, linewidths=1, colors='#000000', alpha=0.2)
        for label in labels:
             label.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
        
        if ax_col == 0:
            ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
            ax.text(-0.25, 0.5, f't={t}', ha='right', va='center', transform=ax.transAxes, fontsize=26, fontweight='bold')
        else:
            ax.set_ylabel('')
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.50', '0.75', '1'], fontsize=14)  

        if ax_row == len(all_ts) - 1:
            ax.set_xlabel('n = size of dataset', fontsize=16, labelpad=10, fontweight='bold')
        else:
            ax.set_xticks([])
        ax.set_xticks([10, 100, 1000, 10000], ['10', '100', '1,000', '10,000'], fontsize=14)

        if ax_row == 0:
            ax.set_title(f'Dataset with {num_choices} choices per example\np = 1/{num_choices}', fontsize=16 , pad=20, fontweight='bold')
        #ax.set_title(f't = {t}', fontsize=16, pad=12)
        #f.suptitle(f'Dataset with {num_choices} choices per example', fontsize=20, fontweight='bold', y=1.04)
        ax.grid(alpha=0.5, axis='both', linestyle='--')
        ax.set_axisbelow('line')

f.savefig(f'figure-6.pdf', bbox_inches='tight')

