import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import get_labels, all_tasks

if __name__ == '__main__':

    # User specification of what tasks and strategies should be considered, what metrics
    # should be used, save options, etc.
    config = {
        'tasks': all_tasks,
        'scheme': 'al',
        'samplers': ['random', 'maximin', 'medoids', 'max_entropy', 'vendi'],
        'models': ['nn', 'rf', 'xgb', 'gpc_ard', 'gpr_ard', 'gpc', 'gpr', 'sv', 'lp', 'bkde'],
        'round': 10,           # Rounds vary from 0-10.
        'metric': 1,           # 0 - Balanced Accuracy, 1 - Macro F1, 2 - Matt. Corr. Coeff.
        'use_baseline': True,  # True uses baseline for comparison. False only uses the metric.
        'labels': True,       # False removes labels from final figure.
        'save_fig': './figures/fig6.png' # Specifies path for saving figure.
    }

    # Read in algorithm and baseline performance.
    results_dict = pickle.load(open('results.pickle', 'rb'))
    baseline_dict = pickle.load(open('baseline_raw.pickle', 'rb'))

    # Set up figure.
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    error_kw = {'ecolor': 'black', 'capsize': 2.0, 
        'elinewidth': 1.0, 'capthick': 1.0}
    plt.figure(figsize=(5.5,5))
    
    # Define what samplers will be compared.
    samplers_data = {
        'random': [],
        'maximin': [],
        'medoids': [],
        'max_entropy': [],
        'vendi': []
    }

    # Accumulate information.
    metrics = {sampler: [] for sampler in config['samplers'] if sampler != 'random'}
    for task in config['tasks']:
        for model in config['models']:

            # Determine viable seeds.
            results = {k: [] for k in config['samplers']}
            for sampler in config['samplers']:
                results[sampler] = results_dict[task][config['scheme']][sampler][model]
            valid_seeds = [key for key in results['random'].keys() 
                if key in results['maximin'].keys()
                if key in results['medoids'].keys()
                if key in results['max_entropy'].keys()
                if key in results['vendi'].keys()
            ]

            # Get scores for the viable seeds.
            scores = {sampler: [] for sampler in config['samplers']}
            for seed in valid_seeds:
                for sampler in config['samplers']:
                    scores[sampler].append(results[sampler][seed][config['round'],config['metric']+1].item())

            # For each of these scores, get a distribution of ξ.
            if config['use_baseline']:
                adjusted_scores = {}
                for sampler in config['samplers']:
                    adjusted_scores[sampler] = []
                    for score in scores[sampler]:
                        seed_data = baseline_dict[task].reshape(30,-1,4)[:,:,[0,config['metric']+1]]
                        for seed in range(seed_data.shape[0]):
                            adjusted_score = -1
                            for row in range(seed_data.shape[1]):
                                if score < seed_data[seed,row,1]:
                                    adjusted_score = row + 1
                                    break
                            if adjusted_score == -1:
                                adjusted_score = seed_data.shape[1] + 1
                            adjusted_scores[sampler].append(adjusted_score)

            # Aggregate statistics for this method and task.
            aggregate_data = {}
            if config['use_baseline']:
                for sampler in adjusted_scores.keys():
                    mean = np.mean(adjusted_scores[sampler])
                    stderr = np.std(adjusted_scores[sampler]) / np.sqrt(len(adjusted_scores[sampler]))
                    aggregate_data[sampler] = [mean, stderr]
            else:
                for sampler in scores.keys():
                    mean = np.mean(scores[sampler])
                    stderr = np.std(scores[sampler]) / np.sqrt(len(scores[sampler]))
                    aggregate_data[sampler] = [mean, stderr]

            # Get ratios and errors in those ratios using error propagation.
            for sampler in config['samplers']:
                if sampler != 'random':
                    mean = aggregate_data[sampler][0] / aggregate_data['random'][0]
                    stderr = mean * np.sqrt(np.square(aggregate_data[sampler][1] / aggregate_data[sampler][0]) + np.square(aggregate_data['random'][1] / aggregate_data['random'][0]))
                    metrics[sampler].append([mean, stderr])
        print(f'Finished computing for {task}.')

    # Process what's in metrics.
    plot_data = {}
    for sampler in metrics.keys():
        mean = np.mean(np.array(metrics[sampler])[:,0])
        stderr = np.sum(np.abs(metrics[sampler])[:,1]) / len(metrics[sampler])
        plot_data[sampler] = [mean, stderr]

    # Plot bars.
    x = [get_labels(s)[:-1] for s in config['samplers'] if s != 'random']
    metric_means = np.array([plot_data[sampler][0] for sampler in plot_data.keys()])
    metric_errs = np.array([plot_data[sampler][1] for sampler in plot_data.keys()])
    plt.bar(x=x, height=metric_means, yerr=metric_errs,
        color='#7cecad', edgecolor='black', linewidth=1.5,
        zorder=10, error_kw=error_kw)
        
    # Make a bar plot of AL success fraction vs. round.
    plt.grid(alpha=0.5, axis='y', zorder=1)
    plt.tight_layout()
    if config['labels']:
        plt.xlabel('Sampler')
        plt.ylabel('Factor Improvement Over Random')
    if not config['labels']:
        plt.xticks(ticks=range(0, len(config['samplers']) - 1), labels=[])
        plt.yticks(ticks=[0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.5, 1.75], labels=[])
        plt.ylim([0.0, 1.75])
        plt.xlabel('')
        plt.ylabel('')
    if config['save_fig']:
        plt.savefig(f'{config["save_fig"]}', dpi=1000)
    plt.show()