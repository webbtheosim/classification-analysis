'''
    This script down-selects the metafeatures computed by the PyMFE Python
    package to those that are (1) common to all tasks, (2) vary among tasks,
    and (3) are not linearly correlated with other metafeatures (r > 0.95). The
    resulting metafeatures and their values are stored in a dictionary saved
    as metafeatures.pickle.

    The results of this script, metafeatures.pickle, are used to run sequential 
    feature addition and generate the parity plots in Figure 8. This can be done 
    using the sfa_and_fig8.py file.
'''

import numpy as np
import pickle
from scipy.stats import pearsonr

from ClassificationSuite.Tasks.utils import load_metafeatures
from utils import all_tasks

if __name__ == '__main__':

    # Load metafeatures.
    metafeatures = load_metafeatures()

    # Identify features that are common to all tasks.
    feature_names = []
    for feature in metafeatures['princeton'].keys():
        common_feature = True
        for task in all_tasks:
            if feature not in metafeatures[task].keys():
                common_feature = False
        if common_feature:
            feature_names.append(feature)
    print(f'Common features: {len(feature_names)}')

    # Get feature distributions based on task.
    mf = [[metafeatures[task][feature] for feature in feature_names] for task in all_tasks]
    mf = np.array(mf)

    # Remove any features that are constant for all tasks.
    delete_features = []
    for i in range(mf.shape[1]):
        if np.min(mf[:,i]) == np.max(mf[:,i]):
            delete_features.append(i)
    for index in sorted(delete_features, reverse=True):
        del feature_names[index]
    mf = np.delete(mf, delete_features, axis=1)
    print(f'Varying features: {mf.shape[1]}')

    # Remove any features that are correlated with other features.
    correlated_pairs = {}
    for i in range(mf.shape[1]):
        correlated_pairs[i] = []
        for j in range(i+1, mf.shape[1]):
            a = mf[:,i]
            b = mf[:,j]
            score = pearsonr(a,b).statistic
            if score > 0.95:
                correlated_pairs[i].append(j)
    delete_features = []
    for key, value in correlated_pairs.items():
        if key not in delete_features:
            for index in value:
                if index not in delete_features:
                    delete_features.append(index)
    for index in sorted(delete_features, reverse=True):
        del feature_names[index]
    mf = np.delete(mf, delete_features, axis=1)
    print(f'Common, unique, and varying features: {mf.shape[1]}')

    # Save final results as a pickle file.
    metafeatures = {
        'tasks': all_tasks,
        'feature_names': feature_names,
        'metafeatures': mf
    }
    with open('metafeatures.pickle', 'wb') as handle:
        pickle.dump(metafeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)