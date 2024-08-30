import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':

    # Get user input for which set of algorithms to consider.
    ALGORITHM_SET = 'top'  # Options: 'all' or 'top'
    SAVE = True
    LABELS = True

    # Load in processed set of metafeatures.
    mf = pickle.load(open('metafeatures.pickle', 'rb'))
    tasks = mf['tasks']
    feature_names = mf['feature_names']
    X = mf['metafeatures']

    # Get set of algorithms to be considered.
    top_algorithms = [
        'al-vendi-nn',
        'al-maximin-nn',
        'al-random-nn',
        'al-medoids-nn',
        'al-max_entropy-nn',
        'al-medoids-rf',
        'al-maximin-rf',
        'al-vendi-rf',
        'al-random-rf',
        'al-maximin-gpc_ard',
        'al-max_entropy-rf',
        'al-medoids-gpc_ard',
        'al-maximin-xgb',
        'al-medoids-gpr_ard',
        'al-maximin-gpr_ard',
        'al-vendi-gpr_ard',
        'al-medoids-sv',
        'al-max_entropy-gpr_ard',
        'al-maximin-gpr',
        'al-medoids-gpr'
    ]
    all_algorithms = [
        f'{scheme}-{sampler}-{model}' 
        for scheme in ['al', 'sf'] 
        for sampler in ['random', 'maximin', 'medoids', 'max_entropy', 'vendi'] 
        for model in ['nn', 'rf', 'xgb', 'gpc_ard', 'gpr_ard', 'gpc', 'gpr', 'sv', 'lp']
    ]
    chosen_algorithms = all_algorithms if ALGORITHM_SET == 'all' else top_algorithms

    # Read in labels (absolute Macro F1 scores).
    y_dict = {}
    results = pickle.load(open('results.pickle', 'rb'))
    for algo in chosen_algorithms:
        terms = algo.split('-')
        scheme = terms[0]
        sampler = terms[1]
        model = terms[2]
        y = np.array([
            np.mean([
                results[task][scheme][sampler][model][seed][-1,-2] 
                for seed in results[task][scheme][sampler][model].keys()
            ])
            for task in tasks
        ])
        y_dict[algo] = y

    # Perform manual sequential feature addition for all algorithms.
    chosen_mf_indices = []
    mae_prev = 0.00
    mae_curr = 1.00
    y_pred_best = []
    y_true_best = []
    while np.abs(mae_prev - mae_curr) > 0.01:

        # Iterate through all metafeatures.
        mae_best = 1.00
        mf_best = 0
        for mf_index in range(X.shape[1]):
            y_pred_mf = []
            y_true_mf = []

            # Get feature indices.
            feature_indices = [i for i in chosen_mf_indices]
            feature_indices.append(mf_index)

            # Iterate through chosen algorithms.
            for algo in chosen_algorithms:

                # Perform leave-one-out cross-validation.
                for task_index in range(X.shape[0]):
                    train_indices = [i for i in range(X.shape[0]) if i != task_index]
                    X_train = X[train_indices,:]
                    X_train = X_train[:,feature_indices].reshape(-1,len(feature_indices))
                    y_train = y_dict[algo][train_indices].reshape(-1)
                    X_test = X[task_index,:]
                    X_test = X_test[feature_indices].reshape(1,-1)
                    y_test = y_dict[algo][task_index].reshape(-1)
                    sc = MinMaxScaler().fit(X_train)
                    X_train_sc = sc.transform(X_train)
                    X_test_sc = sc.transform(X_test)
                    model = LinearRegression().fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_mf.append(y_pred)
                    y_true_mf.append(y_test)

            # Evaluate metafeature performance.
            mae = mean_absolute_percentage_error(y_true_mf, y_pred_mf)
            if mae < mae_best:
                mae_best = mae
                mf_best = mf_index
                y_pred_best = y_pred_mf
                y_true_best = y_true_mf

            # Report results.
            print(f'Metafeatures: {[feature_names[i] for i in feature_indices]} | MAE: {mae * 100.:.3f}%')

        # Save top-performing metafeature.
        chosen_mf_indices.append(mf_best)

        # Update loop termination condition.
        mae_prev = mae_curr
        mae_curr = mae_best

    # Report final results.
    print(f'Final set of metafeatures (MAE = {mae_curr * 100.:.3f}%):')
    for i in chosen_mf_indices:
        print(f'> {feature_names[i]}')

    # Generate parity plot for the top-performing set of attributes.
    selected_feature_indices = chosen_mf_indices
    selected_feature_indices = [feature_names.index('ns_ratio')]
    y_pred_mf = []
    y_true_mf = []

    # Iterate through chosen algorithms.
    for algo in chosen_algorithms:

        # Perform leave-one-out cross-validation.
        for task_index in range(X.shape[0]):
            train_indices = [i for i in range(X.shape[0]) if i != task_index]
            X_train = X[train_indices,:]
            X_train = X_train[:,selected_feature_indices].reshape(-1,len(selected_feature_indices))
            y_train = y_dict[algo][train_indices].reshape(-1)
            X_test = X[task_index,:]
            X_test = X_test[selected_feature_indices].reshape(1,-1)
            y_test = y_dict[algo][task_index].reshape(-1)
            sc = MinMaxScaler().fit(X_train)
            X_train_sc = sc.transform(X_train)
            X_test_sc = sc.transform(X_test)
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_mf.append(y_pred)
            y_true_mf.append(y_test)

    # Evaluate metafeature performance.
    mae = mean_absolute_percentage_error(y_true_mf, y_pred_mf)
    r2 = r2_score(y_true_mf, y_pred_mf)

    # Report results.
    print(f'Metafeatures: {[feature_names[i] for i in selected_feature_indices]}')
    print(f'MAE = {mae * 100.:.3f}% | R\u00b2 = {r2:.3f}')

    # Visualize parity plot.
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['font.size'] = 18
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    color_all = '#91dbff'
    color_top = '#ffca8c'
    ax.scatter(y_true_mf, y_pred_mf, 
        color=color_all if ALGORITHM_SET == 'all' else color_top, 
        s=20.0, edgecolors='black', linewidth=1.0, zorder=9, clip_on=False
    )
    parity = np.linspace(0.0, 1.10, num=1000)
    ax.plot(parity, parity, color='black', linewidth=2.0, zorder=10)
    ax.grid(alpha=0.5, zorder=2)
    ax.set_xlim([0.1, 1.10])
    ax.set_ylim([0.1, 1.10])
    ax.set_xticks(ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks(ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', direction='in', left=True, right=True, bottom=True, top=True, width=1.5)
    if LABELS:
        ax.set_xlabel(r'True Macro F$_1$')
        ax.set_ylabel(r'Predicted Macro F$_1$')
    if not LABELS:
        ax.set_xlabel('')
        ax.set_ylabel('')
    if SAVE:
        plt.savefig(f'figures/fig8_{ALGORITHM_SET}.png', dpi=1000)
    plt.show()