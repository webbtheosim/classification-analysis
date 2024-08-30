import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *

if __name__ == '__main__':

    # User specification of what tasks and strategies should be considered, what metrics
    # should be used, save options, etc.
    config = {
        'tasks': all_tasks,
        'schemes': ['al', 'sf'],
        'samplers': ['random', 'maximin', 'medoids', 'max_entropy', 'vendi'],
        'models': ['nn', 'rf', 'xgb', 'gpc_ard', 'gpr_ard', 'gpc', 'gpr', 'sv', 'lp', 'bkde'],
        'round': 7,           # Rounds vary from 0-10.
        'metric': 1,           # 0 - Balanced Accuracy, 1 - Macro F1, 2 - Matt. Corr. Coeff.
        'use_baseline': True,  # True uses baseline for comparison. False only uses the metric.
        'labels': False,       # False removes labels from final figure.
        'save_fig': './figures/fig3.png' # Specifies path for saving figure.
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
                    score_mean   = np.mean(scores, axis=0)

                    # Save to plot_data.
                    label = f'{scheme}-{sampler}-{model}'
                    plot_data[label] = score_mean

        # Add these labels to performance dict if they're not there already.
        for label in plot_data.keys():
            if label not in performance.keys():
                performance[label] = []

        # If specified, recast metrics in terms of baseline performance.
        if config['use_baseline']:

            # Read in baseline performances.
            baseline_dict = pickle.load(open('baseline/baseline.pickle', 'rb'))

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

                # If strategy performs better than all baseline models,
                # assign it the size of the dataset.
                task_size = baseline_data.shape[0] + 1
                mean_naive = mean_naive if mean_naive != -1 else task_size
                plot_data[key] = mean_naive

        # Determine best mean performance for this task.
        best_mean_performance = 0.0
        for key, value in plot_data.items():
            if value > best_mean_performance:
                best_mean_performance = value

        # Normalize all performances by the best mean performance.
        for key, value in plot_data.items():
            performance[key].append(value / best_mean_performance)

    # Get statistics for optimality fractions across all tasks.
    for key, value in performance.items():
        mean = np.mean(value)
        stderr = np.std(value) / np.sqrt(len(value))
        upper = mean + stderr
        lower = mean - stderr
        performance[key] = [lower, mean, upper]

    # Sort data for plotting.
    plot_data = dict(sorted(performance.items(), key=lambda item: -item[1][1]))

    # Display on the top N_ALGOS strategies.
    N_ALGOS = 20
    temp_data = plot_data.copy()
    plot_data = {}
    for index in range(N_ALGOS):
        plot_data[list(temp_data.keys())[index]] = list(temp_data.values())[index]

    # Plotting.
    plt.rcParams['font.family'] = 'Helvetica' 
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 1.75
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12
    error_kw = {'ecolor': 'black', 'capsize': 2.0, 'elinewidth': 1.1, 'capthick': 1.1}
    fig, ax = plt.subplots(1, 1, figsize=(5.75,6))
    labels = [get_labels(key) for key in plot_data.keys()]
    colors = [get_colors(label) for label in labels]
    means = [i[1] for i in plot_data.values()]
    lower_errors = [i[1] - i[0] for i in plot_data.values()]
    upper_errors = [i[2] - i[1] for i in plot_data.values()]
    bars = ax.barh(labels, means, xerr=[lower_errors, upper_errors], 
        color=colors, edgecolor='black', linewidth=1.2, 
        align='center', error_kw=error_kw, zorder=2)
    ax.xaxis.grid(True, zorder=1)
    ax.invert_yaxis()
    plt.tight_layout()
    if config['use_baseline'] and config['labels']:
        ax.set_xlabel(r'$\langle\xi\rangle$ / $\langle\xi\rangle_{max}$')
        ax.set_xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xlim([0.0, 0.85])
    if not config['use_baseline'] and config['labels']:
        ax.set_xlabel(r'Macro $F_1$ / Macro $F_{1,max}$')
        ax.set_xticks(ticks=[0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
        ax.set_xlim([0.9, 1.00])
    if config['use_baseline'] and not config['labels']:
        ax.set_xlabel('')
        ax.set_xlim([0.0, 0.85])
        ax.set_xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8])
    if not config['use_baseline'] and not config['labels']:
        ax.set_xticks(ticks=[0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
        ax.set_xlim([0.9, 1.00])
        ax.tick_params(axis='both', bottom=False, top=False)
    if config['save_fig'] is not None:
        plt.savefig(f'{config["save_fig"]}', dpi=1000)
    plt.show()
