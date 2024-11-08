import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from utils import *

if __name__ == '__main__':

    config = {
        'tasks': ['glotzer_pf', 'water_lp', 'qm9_gap', 'qm9_r2', 'qm9_cv'],
        'schemes': ['al', 'sf'],
        'samplers': ['random', 'maximin', 'medoids', 'max_entropy', 'vendi'],
        'models': ['nn', 'rf', 'xgb', 'gpc_ard', 'gpr_ard', 'gpc', 'gpr', 'sv'],
        'round': 10,           # Rounds vary from 0-10.
        'metric': 1,           # 0 - Balanced Accuracy, 1 - Macro F1, 2 - Matt. Corr. Coeff.
        'use_baseline': False,  # True uses baseline for comparison. False only uses the metric.
        'labels': True,       # False removes labels from final figure.
        'save_fig': None       # Specifies path for saving figure.
    }

    # Load results.
    results_dict = pickle.load(open('results.pickle', 'rb'))
    revise_dict = pickle.load(open('revisions.pkl', 'rb'))

    # Compute results on a task by task basis.
    factors = []
    for task in config['tasks']:

        # Aggregate results for specific task.
        results = results_dict[task]

        # Aggregate results for each strategy.
        for scheme in config['schemes']:
            for sampler in config['samplers']:
                for model in config['models']:

                    scores_small = []
                    for seed in results[scheme][sampler][model].keys():
                        scores_small.append(results[scheme][sampler][model][seed][config['round'],config['metric']+1].item())

                    scores_large = []
                    task_str = f'{task}_large'
                    for seed in revise_dict[scheme][task_str][sampler][model].keys():
                        scores_large.append(revise_dict[scheme][task_str][sampler][model][seed][config['round'],config['metric']+1].item())
                   
                    for id1 in range(len(scores_small)):
                        for id2 in range(len(scores_large)):
                            factors.append(scores_large[id2] / scores_small[id1])

    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12
    fig, ax = plt.subplots(1,1,figsize=(3.5,1.8))
    sns.kdeplot(factors, fill=True, alpha=0.1, zorder=2)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Performance (Large) / Performance (Small)')
    ax.set_xlim(xmin=0.5, xmax=1.5)
    ax.set_ylim(ymin=0.0, ymax=16.0)
    ax.tick_params(axis='both', left=True, right=True, top=True, bottom=True, direction='in', width=1.2, length=3.5)
    ax.grid(alpha=0.5, zorder=1)
    plt.tight_layout()
    plt.savefig('./figures/fig10.png', dpi=1000)
    plt.show()

    results = pickle.load(open('revisions_mf.pkl', 'rb'))
    original = pickle.load(open('metafeatures.pickle', 'rb'))

    for task in config['tasks']:
        print(f'{task} ------------------')
        task_idx = original['tasks'].index(task)
        mf_idx = original['feature_names'].index('ns_ratio')
        print('ns_ratio:')
        print(f'> {original["metafeatures"][task_idx][mf_idx]}')
        print(f'> {results[f"{task}_large"]["ns_ratio"]}')

        mf_idx = original['feature_names'].index('mut_inf.mean')
        print('mut_inf.mean:')
        print(f'> {original["metafeatures"][task_idx][mf_idx]}')
        print(f'> {results[f"{task}_large"]["mut_inf.mean"]}')

        mf_idx = original['feature_names'].index('mut_inf.max')
        print('mut_inf.max:')
        print(f'> {original["metafeatures"][task_idx][mf_idx]}')
        print(f'> {results[f"{task}_large"]["mut_inf.max"]}')
