# classification-analysis

This repository contains the data and analysis scripts required to reproduce the results shown in [Data Efficiency of Classification Strategies for Chemical and Materials Design](https://doi.org/10.26434/chemrxiv-2024-1sspf).

Every figure can be reproduced using the script with the appropriate name ([fig3.py](fig3.py) recreates Figure 3). We also include the script [prep_metafeatures.py](prep_metafeatures.py) that downselects the set of unique and uncorrelated metafeatures (stored in [metafeatures.pickle](metafeatures.pickle)) used in the sequential feature addition conducted in [sfa_and_fig8.py](sfa_and_fig8.py).

[results.pickle](results.pickle) is a dictionary that contains the performances of all seeds of all classification strategies on all tasks. The dictionary is structured by task, algorithm type, sampler, model, and seed. Therefore, `results['qm9_cv']['al']['medoids']['nn'][24]` contains a numpy array of the performances of of the 24th seed of an active learning algorithm with a medoids sampler and neural network model applied to classification of heat capacities in QM9. The numpy array is a 2D array of shape (11,4) where the first axis corresponds to rounds of active learning and the second axis includes the round number, balanced accuracy, Macro F<sub>1</sub>, and Matthew Correlation Coefficient, in that order. 

[baseline.pickle](baseline.pickle) is a dictionary that contains the performance of the naive strategy used to benchmark the data efficiency of classification strategies. Keys correspond to tasks and values correspond to numpy arrays where the first axis is the number of acquired points and the second axis includes the mean minus standard error, mean, and mean plus standard error, in that order, for the Macro F$_1$ scores of the naive strategy. [baseline.pickle](baseline.pickle) is a processed version of `baseline_raw.pickle` that contains the balanced accuracy, Macro F$_1$ score, and Matthew Correlation Coefficinet for every seed of the naive strategy on every task. This file is used by some of the analysis scripts mentioned above, and is downloadable from `INSERT LINK HERE`.

### Package Dependencies
