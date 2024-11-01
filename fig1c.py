import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from ClassificationSuite.Tasks.utils import load_data

if __name__ == '__main__':

    # List tasks to visualize.
    tasks = [
        'glotzer_pf', 'glotzer_xa', 'water_hp', 'water_lp', 
        'oxidation', 'shower', 'vdw', 'diblock', 'bear', 
        'electro', 'hplc', 'oer', 'toporg', 'polygel', 
        'bace', 'clintox', 'esol', 'free', 'hiv', 
        'lipo', 'muv', 'qm9_cv', 'qm9_gap', 'qm9_r2', 
        'qm9_u0', 'qm9_zpve', 'robeson', 'tox21', 'polysol', 
        'perovskite'
    ]
    s_dict = {
        'princeton': 0.1,
        'glotzer_pf': 0.05,
        'glotzer_xa': 0.05,
        'water_hp': 3.0,
        'water_lp': 3.0,
        'oxidation': 0.5,
        'shower': 3.0,
        'vdw': 3.0,
    }

    # Load appropriate fonts.
    font_path = 'latex_typewriter.ttf' 
    matplotlib.font_manager.fontManager.addfont(font_path)
    prop = matplotlib.font_manager.FontProperties(fname=font_path)

    # Set up figure.
    N_COLS = 5
    n_rows = int(len(tasks) / N_COLS) if (len(tasks) % N_COLS) == 0 else int(len(tasks) / N_COLS) + 1
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams['font.size'] = 16
    fig, axs = plt.subplots(n_rows,N_COLS,figsize=(8,10))

    # Plot each task.
    for index, task in enumerate(tasks):

        print(f'Preparing task {task}.')

        # Load dataset.
        dataset = load_data(task=task)
        X = dataset[:,0:-1]
        y = dataset[:,-1]

        # Reduce dimensions to two, if applicable.
        if X.shape[1] > 2:

            # PCA.
            X_sc = MinMaxScaler().fit_transform(X)
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X_sc)
            X = X_reduced

        # Get row and column indices based on index.
        row_index = int(index / N_COLS)
        col_index = index % N_COLS

        # Visualize dataset.
        colors = ['#91dbff' if y_ == 1 else '#fe7c7c' for y_ in y]
        axs[row_index, col_index].scatter(X[:,0], X[:,1], c=colors, s=s_dict[task] if task in s_dict.keys() else 1.0)
        axs[row_index, col_index].set_xlim([np.min(X[:,0]), np.max(X[:,0])])
        axs[row_index, col_index].set_ylim([np.min(X[:,1]), np.max(X[:,1])])
        axs[row_index, col_index].set_xticks([])
        axs[row_index, col_index].set_yticks([])
        axs[row_index, col_index].set_title(f'{task}')

    # Remove axes from empty entries.
    num_figs = N_COLS * n_rows
    empty_figs = num_figs - len(tasks)
    for index in range(1,empty_figs+1):
        axs[-1,-index].set_axis_off()
        
    # Display figure.
    plt.tight_layout()
    plt.savefig('figures/fig1.png', dpi=1000)
    plt.show()
