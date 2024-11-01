import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *
from fig3 import convert_to_baseline

if __name__ == '__main__':

    # User specification of what tasks and strategies should be considered, what metrics
    # should be used, save options, etc.
    config = {
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
    baseline_dict = load_baseline_raw()

    # Determine top mean performance for every task; used later for normalization.
    top_performance_dict = {}
    for task in all_tasks:
        results = results_dict[task]
        for scheme in ['al', 'sf']:
            for sampler in ['random', 'maximin', 'medoids', 'max_entropy', 'vendi']:
                for model in ['nn', 'rf', 'xgb', 'gpc_ard', 'gpr_ard', 'gpc', 'gpr', 'sv', 'lp', 'bkde']:
                    scores = []
                    for seed in results[scheme][sampler][model].keys():
                        scores.append(results[scheme][sampler][model][seed][config['round'],config['metric']+1].item())
                    scores = convert_to_baseline(scores=scores, task=task, dict=baseline_dict, metric=config['metric'])
                    if task not in top_performance_dict.keys():
                        top_performance_dict[task] = np.mean(scores)
                    else:
                        if np.mean(scores) > top_performance_dict[task]:
                            top_performance_dict[task] = np.mean(scores)

    # Compute data for each set of tasks.
    task_sets = [all_tasks, nn_tasks, rf_tasks]
    final_data = {}
    for task_idx, task_set in enumerate(task_sets):

        # Compute results on a task by task basis.
        performance = {}
        for task in task_set:

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
                        scores = convert_to_baseline(scores=scores, task=task, dict=baseline_dict, metric=config['metric'])
                        label = f'{scheme}-{sampler}-{model}'
                        plot_data[label] = scores

            # Add these labels to performance dict if they're not there already.
            for label in plot_data.keys():
                if label not in performance.keys():
                    performance[label] = []

            # Normalize all performances by the best mean performance.
            for label in plot_data.keys():
                if label not in performance.keys():
                    performance[label] = []
            for key, value in plot_data.items():
                for val in value:
                    performance[key].append(val / top_performance_dict[task])

        # Get statistics for optimality fractions across all tasks.
        for key, value in performance.items():
            value = np.array(value)
            mean = np.mean(value)
            stderr = np.std(value) / np.sqrt(len(value))
            performance[key] = [mean - stderr, mean, mean + stderr]

        # Sort data for plotting.
        plot_data = dict(sorted(performance.items(), key=lambda item: -item[1][1]))

        # Save data to larger data structure.
        if task_idx == 0:
            final_data['all'] = plot_data
        elif task_idx == 1:
            final_data['nn'] = plot_data
        else:
            final_data['rf'] = plot_data

    # Save to file.
    pickle.dump(final_data, open('./data/fig7.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)