import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *
from fig3 import convert_to_baseline

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
    }

    # Load results.
    results_dict = pickle.load(open('results.pickle', 'rb'))
    baseline_dict = load_baseline_raw()

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
                    if config['use_baseline']:
                        scores = convert_to_baseline(scores=scores, task=task, dict=baseline_dict, metric=config['metric'])
                    score_mean  = np.mean(scores, axis=0)
                    label = f'{scheme}-{sampler}-{model}'
                    plot_data[label] = score_mean

        # Add these labels to performance dict if they're not there already.
        for label in plot_data.keys():
            if label not in performance.keys():
                performance[label] = []

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
        'Space Filling': list(sf_data.values()),
        'Model Names': list(al_data.keys())
    }

    # Save data to file.
    pickle.dump(plot_data, open('./data/fig4.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)