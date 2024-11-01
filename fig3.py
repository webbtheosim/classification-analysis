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
        'schemes': ['al', 'sf'],
        'samplers': ['random', 'maximin', 'medoids', 'max_entropy', 'vendi'],
        'models': ['nn', 'rf', 'xgb', 'gpc_ard', 'gpr_ard', 'gpc', 'gpr', 'sv', 'lp', 'bkde'],
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

    # Save plot_data for plotting.
    pickle.dump(final_data, open('./data/fig3.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)