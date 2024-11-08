from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *

def convert_to_baseline(scores, task, dict, metric):
    '''
        Converts a list of scores (of the specified metric) into 
        a list of xi values.
    '''

    baseline = dict[task].reshape(30,-1,4)[:,:,[0,metric+1]]
    adjusted_scores = []
    for score_idx, score in enumerate(scores):
        adjusted_score = -1
        for row in range(baseline.shape[1]):
            if score < baseline[score_idx,row,1]:
                adjusted_score = row + 1
                break
        if adjusted_score == -1:
            adjusted_score = baseline.shape[1] + 1
        adjusted_scores.append(adjusted_score)

    return adjusted_scores

if __name__ == '__main__':

    # User specification of what tasks and strategies should be considered, what metrics
    # should be used, save options, etc.
    config = {
        'tasks': all_tasks,
        'schemes': ['al'],
        'samplers': ['random', 'maximin', 'medoids', 'max_entropy', 'vendi'],
        'models': ['ensemble_top', 'ensemble_averaging', 'ensemble_stacking', 'ensemble_arbitrating'],
        'round': 10,           # Rounds vary from 0-10.
        'metric': 1            # 0 - Balanced Accuracy, 1 - Macro F1, 2 - Matt. Corr. Coeff.
    }

    # Load data.
    results_dict = pickle.load(open('results.pickle', 'rb'))
    baseline_dict = load_baseline_raw()

    # Compute results without and with baseline metric.
    final_data = {}
    options = ['raw', 'use_baseline']
    for opt in options:
        print(f'Processing data for option: {opt}')

        # Determine top mean performance for every task; used later for normalization.
        top_performance_dict = {}
        for task in config['tasks']:
            results = results_dict[task]
            for scheme in ['al', 'sf']:
                for sampler in ['random', 'maximin', 'medoids', 'max_entropy', 'vendi']:
                    for model in ['nn', 'rf', 'xgb', 'gpc_ard', 'gpr_ard', 'gpc', 'gpr', 'sv', 'lp', 'bkde']:
                        scores = []
                        for seed in results[scheme][sampler][model].keys():
                            scores.append(results[scheme][sampler][model][seed][config['round'],config['metric']+1].item())
                        if opt == 'use_baseline':
                            scores = convert_to_baseline(scores=scores, task=task, dict=baseline_dict, metric=config['metric'])
                        if task not in top_performance_dict.keys():
                            top_performance_dict[task] = np.mean(scores)
                        else:
                            if np.mean(scores) > top_performance_dict[task]:
                                top_performance_dict[task] = np.mean(scores)

        # Compute results on a task by task basis.
        performance = {}
        for task in config['tasks']:
            print(f'Processing for {task}...')

            # Aggregate results for specific task.
            results = results_dict[task]
            plot_data = {}
            for scheme in config['schemes']:
                for sampler in config['samplers']:
                    for model in config['models']:
                        scores = []
                        for seed in results[scheme][sampler][model].keys():
                            scores.append(results[scheme][sampler][model][seed][config['round'],config['metric']+1].item())
                        if opt == 'use_baseline':
                            scores = convert_to_baseline(scores=scores, task=task, dict=baseline_dict, metric=config['metric'])
                        label = f'{scheme}-{sampler}-{model}'
                        plot_data[label] = scores

            # Normalize all performances by the best mean performance.
            for label in plot_data.keys():
                if label not in performance.keys():
                    performance[label] = []
            for key, value in plot_data.items():
                for val in value:
                    performance[key].append(val / top_performance_dict[task])

        # Get statistics for optimality fractions across all tasks.
        for key, value in performance.items():
            mean = np.mean(value)
            stderr = np.std(value) / np.sqrt(len(value))
            upper = mean + stderr
            lower = mean - stderr
            performance[key] = [lower, mean, upper]

        # Sort data for plotting.
        plot_data = dict(sorted(performance.items(), key=lambda item: -item[1][1]))
        final_data[opt] = plot_data

    # Filter only the top N_ALGOs performing models.
    plot_data = final_data
    N_ALGOS = 20
    for key in plot_data.keys():
        temp_data = plot_data[key].copy()
        filtered = {}
        for index in range(N_ALGOS):
            filtered[list(temp_data.keys())[index]] = list(temp_data.values())[index]
        plot_data[key] = filtered

    # Set up general plotting parameters.
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 1.75
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 12
    error_kw = {'ecolor': 'black', 'capsize': 2.0, 'elinewidth': 1.1, 'capthick': 1.1}
    fig, axs = plt.subplots(1, 2, figsize=(9.5,5.5), constrained_layout=True)

    # Build plot for Macro F1 scores.
    plotting_data = plot_data['raw']
    labels = [get_labels(key, ensemble=True) for key in plotting_data.keys()]
    colors = [get_colors(label, ensemble=True) for label in labels]
    means = [i[1] for i in plotting_data.values()]
    lower_errors = [i[1] - i[0] for i in plotting_data.values()]
    upper_errors = [i[2] - i[1] for i in plotting_data.values()]
    bars = axs[0].barh(labels, means, xerr=[lower_errors, upper_errors], 
        color=colors, edgecolor='black', linewidth=1.2, 
        align='center', error_kw=error_kw, zorder=2)
    axs[0].xaxis.grid(True, zorder=1)
    axs[0].invert_yaxis()
    axs[0].set_xlim(xmin=0.80, xmax=1.00)
    axs[0].set_xticks(ticks=[0.80, 0.85, 0.90, 0.95, 1.00])
    axs[0].set_xlabel(r'$\langle F_1 / F_{1,\text{max}} \rangle$')
    axs[0].tick_params(axis='x', length=0)

    # Build plot for baseline metric.
    plotting_data = plot_data['use_baseline']
    labels = [get_labels(key, ensemble=True) for key in plotting_data.keys()]
    colors = [get_colors(label, ensemble=True) for label in labels]
    means = [i[1] for i in plotting_data.values()]
    lower_errors = [i[1] - i[0] for i in plotting_data.values()]
    upper_errors = [i[2] - i[1] for i in plotting_data.values()]
    bars = axs[1].barh(labels, means, xerr=[lower_errors, upper_errors], 
        color=colors, edgecolor='black', linewidth=1.2, 
        align='center', error_kw=error_kw, zorder=2)
    axs[1].xaxis.grid(True, zorder=1)
    axs[1].invert_yaxis()
    axs[1].set_xlim(xmin=0.0, xmax=1.00)
    axs[1].set_xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.00])
    axs[1].set_xlabel(r'$\langle \xi / \xi_{\text{max}} \rangle$')
    axs[1].tick_params(axis='x', length=0)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    fig.text(0.0, 0.94, 'A', size=28, weight='bold')
    fig.text(0.5, 0.94, 'B', size=28, weight='bold')
    legend_elements = [
        Patch(facecolor=get_colors('Hyperparameter', ensemble=True), edgecolor='black', linewidth=1.2, label='Hyperparameter'),
        Patch(facecolor=get_colors('Averaging', ensemble=True), edgecolor='black', linewidth=1.2, label='Averaging'),
        Patch(facecolor=get_colors('Stacking', ensemble=True), edgecolor='black', linewidth=1.2, label='Stacking'),
        Patch(facecolor=get_colors('Arbitrating', ensemble=True), edgecolor='black', linewidth=1.2, label='Arbitrating')
    ]
    fig.legend(
        handles=legend_elements, 
        loc='lower center', 
        ncol=6,
        fontsize=14, 
        frameon=False,
        handleheight=1.0,
        handlelength=2.0,
    )
    plt.savefig('./figures/ensembles.png', dpi=1000)
    plt.show()