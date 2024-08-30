import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *

if __name__ == '__main__':

    # User specification of what tasks and strategies should be considered, what metrics
    # should be used, save options, etc.
    config = {
        'tasks': rf_tasks,
        'schemes': ['al'],
        'samplers': ['random', 'maximin', 'medoids', 'max_entropy', 'vendi'],
        'models': ['nn', 'rf', 'ensemble_top'],
        'round': 10,           # Rounds vary from 0-10.
        'metric': 1,           # 0 - Balanced Accuracy, 1 - Macro F1, 2 - Matt. Corr. Coeff.
        'labels': False,       # False removes labels from final figure.
        'save_fig': './figures/fig7.png', # Specifies path for saving figure.
        'ensemble_only': False            # True to display only ensemble-based strategies.
    }

    # Load results.
    results_dict = pickle.load(open('results.pickle', 'rb'))

    # Compute results on a task by task basis.
    performance = {}
    for task in config['tasks']:

        # Aggregate results for specific task.
        results = results_dict[task]

        # Aggregate results for each strategy.
        plot_data = {}
        for scheme in config['schemes']:
            for sampler in config['samplers']:
                for model in config['models']:
                    scores = []
                    for seed in results[scheme][sampler][model].keys():
                        scores.append(results[scheme][sampler][model][seed][config['round'],config['metric']+1].item())
                    label = f'{scheme}-{sampler}-{model}'
                    plot_data[label] = np.mean(scores)

        # Add these labels to performance dict if they're not there already.
        for label in plot_data.keys():
            if label not in performance.keys():
                performance[label] = []

        # Read in baseline performances.
        adjusted_performance = {}
        baseline_dict = pickle.load(open('baseline.pickle', 'rb'))
        for key, value in plot_data.items():
            vals    = key.split('-')
            scheme  = vals[0]
            sampler = vals[1]
            model   = vals[2]
            baseline_data = baseline_dict[task][config['metric'],:,:]

            # Calculate mean ξ.
            mean_naive  = -1
            for row in range(baseline_data.shape[0]):
                if value < baseline_data[row,1]:
                    mean_naive = row + 1
                    break
            
            # Compute statistics on adjusted scores.
            adjusted_performance[key] = mean_naive

        # Determine best mean performance for this task.
        best_performance_mean = 0.0
        for key, value in adjusted_performance.items():
            if value > best_performance_mean:
                best_performance_mean = value

        # Normalize all performances by the best mean performance.
        for key, value in adjusted_performance.items():
            performance[key].append(value / best_performance_mean)

    # Get statistics for optimality fractions across all tasks.
    for key, value in performance.items():
        value = np.array(value)
        mean = np.mean(value)
        stderr = np.std(value) / np.sqrt(len(value))
        performance[key] = [mean - stderr, mean, mean + stderr]

    # Sort data for plotting.
    plot_data = dict(sorted(performance.items(), key=lambda item: -item[1][1]))

    # Plotting.
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 1.75
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12
    error_kw = {'ecolor': 'black', 'capsize': 2.0, 'elinewidth': 1.1, 'capthick': 1.1}
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    labels = [get_labels(key) for key in plot_data.keys()]
    colors = [get_colors(label, ensemble=config['ensemble_only']) for label in labels]
    labels = [l[4:] for l in labels]
    means = [i[1] for i in plot_data.values()]
    lower_errors = [i[1] - i[0] for i in plot_data.values()]
    upper_errors = [i[2] - i[1] for i in plot_data.values()]
    bars = ax.barh(labels, means, xerr=[lower_errors, upper_errors], 
        color=colors, edgecolor='black', linewidth=1.1, 
        align='center', error_kw=error_kw, zorder=2)
    ax.xaxis.grid(True, zorder=1)
    ax.tick_params(axis='x', bottom=False, top=False)
    ax.invert_yaxis()
    if config['labels']:
        ax.set_xlabel(r'$\langle\xi\rangle$ / $\langle\xi\rangle_{max}$')
        ax.set_xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim([0.0, 1.0])
    if not config['labels']:
        ax.set_xlabel('')
        ax.set_xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim([0.0, 1.0])
    plt.tight_layout()
    if config['save_fig'] is not None:
        plt.savefig(f'{config["save_fig"]}', dpi=1000)
    plt.show()