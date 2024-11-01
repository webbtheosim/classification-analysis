import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import all_tasks

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
    }

    # Read in algorithm and baseline results.
    results_dict = pickle.load(open('results.pickle', 'rb'))
    
    # Define what rounds will be compared.
    rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    algo_sets = [
        [
            'al-maximin-nn',
            'al-vendi-nn',
            'al-random-nn',
            'al-medoids-nn',
            'al-max_entropy-nn',
            'al-maximin-rf',
            'al-maximin-gpc_ard',
            'al-medoids-rf',
            'al-vendi-rf',
            'al-random-rf',
            'al-max_entropy-rf',
            'al-medoids-gpc_ard',
            'al-vendi-gpc_ard',
            'al-maximin-gpc',
            'al-random-gpc_ard',
            'al-medoids-gpr_ard',
            'al-maximin-gpr_ard',
            'al-random-gpr_ard',
            'al-max_entropy-gpr_ard',
            'al-maximin-xgb'
        ],
        [f'{scheme}-{sampler}-{model}' for scheme in config['schemes'] for sampler in config['samplers'] for model in config['models']]
    ]
    labels = ['Top 20 Algorithms', 'All Algorithms']
    plot_data = [
        [],
        []
    ]

    for idx, round in enumerate(rounds):
        for idx2, algo_set in enumerate(algo_sets):
            metrics = []
            for task in config['tasks']:

                # Accumulate information.
                for algo in algo_set:
                    vals = algo.split('-')
                    sampler = vals[1]
                    model = vals[2]
                            
                    # Determine viable seeds.
                    al_results = results_dict[task]['al'][sampler][model]
                    sf_results = results_dict[task]['sf'][sampler][model]
                    valid_seeds = [key for key in al_results.keys() if key in sf_results.keys()]

                    # Get scores.
                    al_scores = []
                    sf_scores = []
                    for seed in valid_seeds:
                        al_scores.append(al_results[seed][round,config['metric']+1].item())
                        sf_scores.append(sf_results[seed][round,config['metric']+1].item())
                    for index in range(len(al_scores)):
                        metrics.append(al_scores[index] / sf_scores[index])

            # Plot a histogram of these factors.
            metrics = np.array(metrics)
            plot_data[idx2].append(np.count_nonzero(np.where(metrics > 1.00, 1, 0)) / metrics.shape[0])

    # Save data.
    pickle.dump(plot_data, open('./data/fig5.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)