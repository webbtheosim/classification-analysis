import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import all_tasks, load_baseline_raw

if __name__ == '__main__':

    # User specification of what tasks and strategies should be considered, what metrics
    # should be used, rounds to include, etc.
    config = {
        'tasks': all_tasks,
        'scheme': 'sf',
        'samplers': ['random', 'maximin', 'medoids', 'max_entropy', 'vendi'],
        'models': ['nn', 'rf', 'xgb', 'gpc_ard', 'gpr_ard', 'gpc', 'gpr', 'sv', 'lp', 'bkde'],
        'rounds': [0, 1, 3, 5, 7, 10],   # Rounds vary from 0-10.
        'metric': 1,                     # 0 - Balanced Accuracy, 1 - Macro F1, 2 - Matt. Corr. Coeff.
        'use_baseline': True,            # True uses baseline for comparison. False only uses the metric.
    }

    results_dict = pickle.load(open('results.pickle', 'rb'))
    baseline_dict = load_baseline_raw()

    plot_data = {
        'maximin': [],
        'medoids': [],
        'max_entropy': [],
        'vendi': []
    }
    for round in config['rounds']:
        round_data = {
            'maximin': [],
            'medoids': [],
            'max_entropy': [],
            'vendi': []
        }
        print(f'Computing for round {round}...')
        for task in config['tasks']:
            print(f'Computing for task {task}...')
            for model in config['models']:

                results = {k: [] for k in config['samplers']}
                for sampler in config['samplers']:
                    results[sampler] = results_dict[task][config['scheme']][sampler][model]
                valid_seeds = [key for key in results['random'].keys() 
                    if key in results['maximin'].keys()
                    if key in results['medoids'].keys()
                    if key in results['max_entropy'].keys()
                    if key in results['vendi'].keys()
                ]

                scores = {sampler: [] for sampler in config['samplers']}
                for seed in valid_seeds:
                    for sampler in config['samplers']:
                        scores[sampler].append(results[sampler][seed][round,config['metric']+1].item())

                if config['use_baseline']:
                    adjusted_scores = {}
                    for sampler in config['samplers']:
                        adjusted_scores[sampler] = []
                        for score_idx, score in enumerate(scores[sampler]):
                            seed_data = baseline_dict[task].reshape(30,-1,4)[:,:,[0,config['metric']+1]]
                            adjusted_score = -1
                            for row in range(seed_data.shape[1]):
                                if score < seed_data[score_idx,row,1]:
                                    adjusted_score = row + 1
                                    break
                            if adjusted_score == -1:
                                adjusted_score = seed_data.shape[1] + 1
                            adjusted_scores[sampler].append(adjusted_score)
                    scores = adjusted_scores
                
                agg_data = {}
                for sampler in scores.keys():
                    mean = np.mean(scores[sampler])
                    stderr = np.std(scores[sampler]) / np.sqrt(len(scores[sampler]))
                    agg_data[sampler] = [mean, stderr]

                for sampler in config['samplers']:
                    if sampler != 'random':
                        mean = agg_data[sampler][0] / agg_data['random'][0]
                        stderr = mean * np.sqrt(np.square(agg_data[sampler][1] / agg_data[sampler][0]) + np.square(agg_data['random'][1] / agg_data['random'][0]))
                        round_data[sampler].append([mean, stderr])
        
        for key, value in round_data.items():
            round_data[key] = np.array(value)
        for sampler, data in round_data.items():
            plot_data[sampler].append([np.mean(data[:,0]), np.sqrt(np.sum(np.square(data[:,1]))) / data.shape[0]])

    for key, value in plot_data.items():
        plot_data[key] = np.array(value)
    
    if config['scheme'] == 'al':
        pickle.dump(plot_data, open('data/fig6_al.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    if config['scheme'] == 'sf':
        pickle.dump(plot_data, open('data/fig6_sf.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)