import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from utils import *
from ClassificationSuite.Tasks.utils import task_config

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

    start = time.perf_counter()

    config = {
        'tasks': ['qm9_gap', 'qm9_r2', 'qm9_cv', 'qm9_zpve',
            'qm9_u0', 'robeson', 'free', 'esol', 'lipo', 'hiv',
            'bace', 'clintox', 'muv', 'tox21'],
        'features': ['mordred_10', 'mordred_20', 'mordred_100', 'mordred_all', 'morgan'],
        'schemes': ['al', 'sf'],
        'samplers': ['random', 'maximin', 'medoids', 'max_entropy', 'vendi'],
        'models': ['nn', 'rf', 'xgb', 'gpc_ard', 'gpr_ard', 'gpc', 'gpr', 'sv'],
        'round': 10,           # Rounds vary from 0-10.
        'metric': 1,           # 0 - Balanced Accuracy, 1 - Macro F1, 2 - Matt. Corr. Coeff.
        'use_baseline': True,  # True uses baseline for comparison. False only uses the metric.
    }
    
    results_dict = pickle.load(open('results.pickle', 'rb'))
    revise_dict = pickle.load(open('revisions.pkl', 'rb'))
    baseline_dict = load_baseline_raw()

    # Get performances for each scheme-sampler-model-feature combination.
    performance_data = {}
    for task in config['tasks']:
        print(f'Processing for task {task}...')
        performance_data[task] = {}
        results = results_dict[task]
        for scheme in config['schemes']:
            for sampler in config['samplers']:
                for model in config['models']:
                    if not (scheme == 'sf' and sampler == 'vendi'):

                        if 'mordred_10' in config['features']:
                            scores = []
                            for seed in results[scheme][sampler][model].keys():
                                scores.append(results[scheme][sampler][model][seed][config['round'],config['metric']+1].item())
                            label = f'{scheme}-{sampler}-{model}-mordred_10'
                            if config['use_baseline']:
                                scores = convert_to_baseline(scores, task=task, dict=baseline_dict, metric=config['metric'])
                            performance_data[task][label] = scores

                        if 'mordred_20' in config['features']:
                            scores = []
                            task_str = f'{task}_mordred_20'
                            for seed in revise_dict[scheme][task_str][sampler][model].keys():
                                scores.append(revise_dict[scheme][task_str][sampler][model][seed][config['round'],config['metric']+1].item())
                            label = f'{scheme}-{sampler}-{model}-mordred_20'
                            if config['use_baseline']:
                                scores = convert_to_baseline(scores, task=task, dict=baseline_dict, metric=config['metric'])
                            performance_data[task][label] = scores

                        if 'mordred_100' in config['features']:
                            scores = []
                            task_str = f'{task}_mordred_100'
                            for seed in revise_dict[scheme][task_str][sampler][model].keys():
                                scores.append(revise_dict[scheme][task_str][sampler][model][seed][config['round'],config['metric']+1].item())
                            label = f'{scheme}-{sampler}-{model}-mordred_100'
                            if config['use_baseline']:
                                scores = convert_to_baseline(scores, task=task, dict=baseline_dict, metric=config['metric'])
                            performance_data[task][label] = scores

                        if 'mordred_all' in config['features']:
                            scores = []
                            task_str = f'{task}_mordred_all'
                            for seed in revise_dict[scheme][task_str][sampler][model].keys():
                                scores.append(revise_dict[scheme][task_str][sampler][model][seed][config['round'],config['metric']+1].item())
                            label = f'{scheme}-{sampler}-{model}-mordred_all'
                            if config['use_baseline']:
                                scores = convert_to_baseline(scores, task=task, dict=baseline_dict, metric=config['metric'])
                            performance_data[task][label] = scores

                        if 'morgan' in config['features']:
                            scores = []
                            task_str = f'{task}_morgan'
                            model_str = 'gpr' if model == 'gpr_ard' else model
                            model_str = 'gpc' if model == 'gpc_ard' else model_str
                            for seed in revise_dict[scheme][task_str][sampler][model_str].keys():
                                scores.append(revise_dict[scheme][task_str][sampler][model_str][seed][config['round'],config['metric']+1].item())
                            label = f'{scheme}-{sampler}-{model}-morgan'
                            if config['use_baseline']:
                                scores = convert_to_baseline(scores, task=task, dict=baseline_dict, metric=config['metric'])
                            performance_data[task][label] = scores

    # Normalize scores by best mean performance on a given task.
    for task in config['tasks']:
        top_performance = np.max([np.mean(value) for value in performance_data[task].values()])
        for key, value in performance_data[task].items():
            if np.mean(value) == top_performance:
                if config['use_baseline']:
                    task_data = task_config(task)
        for key, value in performance_data[task].items():
            performance_data[task][key] = np.array(value) / top_performance

    # Aggregate scores for plotting. The first set of keys are features, the second set of keys
    # are (scheme, model). Scores from different samplers are combined.
    plot_data = {}
    for task in performance_data.keys():
        for algorithm, scores in performance_data[task].items():
            vals = algorithm.split('-')
            scheme = vals[0]
            sampler = vals[1]
            model = vals[2]
            feat = vals[3]
            if feat not in plot_data.keys():
                plot_data[feat] = {}
            new_str = '-'.join([scheme, model])
            if new_str not in plot_data[feat].keys():
                plot_data[feat][new_str] = []
            for score in scores:
                plot_data[feat][new_str].append(score)

    # Final preparation of plot data.
    final_data = {}
    for feat_idx, feat in enumerate(config['features']):
        for algorithm, scores in plot_data[feat].items():
            if 'al' in algorithm:
                vals = algorithm.split('-')
                model = vals[1]
                if model not in final_data.keys():
                    final_data[model] = []
                mean = np.mean(scores)
                stderr = np.std(scores) / np.sqrt(len(scores))
                final_data[model].append([feat_idx, mean, stderr])
    for key, value in final_data.items():
        final_data[key] = np.array(value)

    # Save final data to file.
    pickle.dump(final_data, open('./data/fig9.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)