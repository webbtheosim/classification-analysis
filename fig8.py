import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':

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
    algorithm_sets = [all_algorithms, top_algorithms]

    # Compute data for both algorithm sets.
    final_data = {}
    for set_idx, chosen_algorithms in enumerate(algorithm_sets):
        set_str = 'all' if set_idx == 0 else 'top'
        print(f'Working on set: {set_str}')

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

        if set_idx == 0:
            final_data['all'] = np.hstack((np.array(y_true_mf).reshape(-1,1), np.array(y_pred_mf).reshape(-1,1)))
        else:
            final_data['top'] = np.hstack((np.array(y_true_mf).reshape(-1,1), np.array(y_pred_mf).reshape(-1,1)))

    # Save data to file.
    pickle.dump(final_data, open('./data/fig8.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)