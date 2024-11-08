import matplotlib.pyplot as plt
import numpy as np
from ClassificationSuite.Samplers import sample
from ClassificationSuite.Tasks.utils import load_data

if __name__ == '__main__':

    # Load dataset.
    dataset = load_data(task='princeton')
    X = dataset[:,0:-1]
    y = dataset[:,-1]
    colors = ['#91dbff' if y_ == 1 else '#fe7c7c' for y_ in y]
    samplers = ['random', 'maximin', 'medoids', 'max_entropy', 'vendi']
    titles = ['A', 'B', 'C', 'D', 'E']
    title_dict = {
        'random': 'Random',
        'maximin': 'Maximin',
        'medoids': 'Medoids',
        'max_entropy': 'Max Entropy',
        'vendi': 'Vendi'
    }

    # Prepare figure.
    plt.rcParams['font.family'] = 'Helvetica Neue'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0
    fig, axs = plt.subplots(1,len(samplers),figsize=(6.5, 1.8), zorder=1)

    # Iterate through samplers.
    for index, sampler in enumerate(samplers):

        # Plot the domain.
        axs[index].scatter(X[:,0], X[:,1], c=colors, s=0.1)

        # Sample points.
        chosen_indices = sample(name=sampler, domain=X, size=30, seed=1)
        chosen_points = X[chosen_indices,:]

        # Plot the chosen points.
        axs[index].scatter(chosen_points[:,0], chosen_points[:,1], c='#FFED7C', s=50.0, edgecolor='black', linewidth=1.0, clip_on=False, zorder=10)

        # Brush up plot.
        axs[index].set_xlim([np.min(X[:,0]), np.max(X[:,0])])
        axs[index].set_ylim([np.min(X[:,1]), np.max(X[:,1])])
        axs[index].set_xticks([])
        axs[index].set_yticks([])
        axs[index].set_title(titles[index])

    # Finish figure.
    plt.tight_layout()
    plt.savefig('figures/fig2.png', dpi=1000)
    plt.show()