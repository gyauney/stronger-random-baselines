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

from max_random_baseline import max_random_baseline
from approximate_max_random_baseline import approximate_max_random_baseline

from dataset_constants import *

rng = np.random.default_rng()



ts = [1, 10, 100, 1000]
f, axs = plt.subplots(4, 2, figsize=(9.6, 6.4))
plt.subplots_adjust(hspace=0.75, wspace = 0.2)


for col, (name, d_to_n) in enumerate([('BIG-bench Lite', BIGBENCH_LITE_DETAILS),
                                      ('BIG-bench Hard', BIGBENCH_HARD_DETAILS)]):
    for row, (ax, t) in enumerate(zip(axs[:, col], ts)):
        print(f'{name}, t = {t}')
        random_baselines = []
        for n, p in d_to_n.values():
            max_random = approximate_max_random_baseline(n, p, t)
            diff = (max_random - p) * 100
            if abs(diff) < 1e-4:
                diff = 0
            random_baselines.append(diff)
        bins = np.linspace(0, 26, 26)
        ax.hist(random_baselines, bins=bins, color='#8fb3d2', edgecolor='#3d7bb1', clip_on=True, zorder=10)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xlim((0, 26))
        ax.set_title(f't = {t}', fontsize=18, fontweight='bold')
        ax.grid(False, axis='x')
        if row == 0:
            ax.set_yticks([0, 4, 8, 12, 16])
            ax.set_ylim((0,16))
        elif row == 1:
            ax.set_yticks([0, 2, 4, 6, 8], ['0', '', '4', '', '8'])
            ax.set_ylim((0,8))
        elif row >= 2:
            ax.set_yticks([0, 1, 2, 3, 4], ['0', '', '2', '', '4'])
            ax.set_ylim((0,4))

        ax.tick_params(axis='both', which='major', labelsize=14)
        if row == 0:
            ax.text(0.5, 1.58, name, transform=ax.transAxes, fontsize=22, fontweight='bold', ha='center', va='center')


    
f.text(0.06, 0.5, 'Number of datasets', fontsize=18, va='center', rotation='vertical', fontweight='bold')
f.text(0.5, -0.025, 'Accuracy difference between\nmax random baseline and standard random baseline', fontsize=18, ha='center', fontweight='bold')
#f.text(0.5, 0.92, 'MMLU', fontsize=22, ha='center', fontweight='bold')
savefig(f'figure-8.pdf', bbox_inches='tight')
plt.close()
