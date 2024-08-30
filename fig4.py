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
        'round': 10,           # Rounds vary from 0-10.
        'metric': 1,           # 0 - Balanced Accuracy, 1 - Macro F1, 2 - Matt. Corr. Coeff.
        'use_baseline': True,  # True uses baseline for comparison. False only uses the metric.
        'labels': True,       # False removes labels from final figure.
        'save_fig': './figures/fig4.png' # Specifies path for saving figure.
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

    # Identify those algorithms which never achieve 90% optimality.
    bad_algos = [key for key,value in performance.items() if np.max(np.array(value)) < 0.9]
    print(f'{len(bad_algos)} bad algorithms out of {len(plot_data)}.')
    
    # Split analysis based on data selection scheme.
    al_data = {k: 0 for k in config['models']}
    sf_data = {k: 0 for k in config['models']}
    for algo in bad_algos:
        vals = algo.split('-')
        if 'al' in vals[0]:
            al_data[vals[2]] += 1
        if 'sf' in vals[0]:
            sf_data[vals[2]] += 1
    for key in al_data.keys():
        al_data[key] = al_data[key] / 5.0
        sf_data[key] = sf_data[key] / 5.0

    # Sort both dictionaries by SF performance, first.
    sf_data = dict(sorted(sf_data.items(), key=lambda item: -item[1]))
    al_data = {key: al_data[key] for key in sf_data.keys()}

    # Then sort dictionarys by AL performance, second.
    al_data = dict(sorted(al_data.items(), key=lambda item: -item[1]))
    sf_data = {key: sf_data[key] for key in al_data.keys()}

    # Arrange data for grouped bar plot.
    plot_data = {
        'Active Learning': list(al_data.values()),
        'Space Filling': list(sf_data.values())
    }
    models = [get_labels(key) for key in al_data.keys()]

    # Make a bar plot with these fractions.
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['axes.linewidth'] = 2.0
    plt.figure(figsize=(11,5))
    x = np.arange(len(models))
    width = 0.25
    multiplier = 0
    colors = ['#ca9fff', '#fff2a1']
    for index, (attribute, measurement) in enumerate(plot_data.items()):
        offset = width * multiplier
        rects = plt.bar(x + offset, measurement, width, label=attribute, color=colors[index], edgecolor='black', linewidth=2.0, zorder=10)
        multiplier += 1
    plt.xticks(x + 0.5 * offset, models)
    plt.tick_params(axis='y', left=False, right=False)
    plt.ylim(ymax=1.05)
    plt.grid(alpha=0.5, axis='y', zorder=1)
    plt.tight_layout()
    if config['labels']:
        plt.ylabel('Fraction of Sub-Optimality')
        plt.legend()
    else:
        plt.xticks(ticks=x + 0.5 * offset, labels=[])
        plt.yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[])
        plt.xlabel('')
        plt.ylabel('')
    if config['save_fig'] is not None:
        plt.savefig(f'{config["save_fig"]}', dpi=1000)
    plt.show()
