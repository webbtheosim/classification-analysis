import argparse
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *

def get_color(model):

    colors = {
        'nn': '#91dbff',
        'rf': '#fe7c7c',
        'xgb': '#ca9fff',
        'gpr_ard': '#a7f8dd',
        'gpr': '#a7f8dd',
        'gpc_ard': '#01ce90',
        'gpc': '#01ce90',
        'sv': '#ffca8c'
    }
    
    return colors[model]

def get_label(model):

    labels = {
        'nn': 'NN',
        'rf': 'RF',
        'xgb': 'XGB',
        'gpr_ard': 'GPR (ARD)',
        'gpr': 'GPR',
        'gpc_ard': 'GPC (ARD)',
        'gpc': 'GPC',
        'sv': 'SV',

    }

    return labels[model]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fig', type=int, default=1)
    args = parser.parse_args()

    # Plot performance of top-performing algorithms on the original 31 classification tasks.
    if args.fig == 3:

        # Load data.
        plot_data = pickle.load(open('./data/fig3.pkl', 'rb'))

        # Filter only the top N_ALGOs performing models.
        N_ALGOS = 20
        for key in plot_data.keys():
            temp_data = plot_data[key].copy()
            filtered = {}
            for index in range(N_ALGOS):
                filtered[list(temp_data.keys())[index]] = list(temp_data.values())[index]
            plot_data[key] = filtered

        # Set up general plotting parameters.
        plt.rcParams['font.family'] = 'Helvetica Neue' 
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 1.1
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 12
        error_kw = {'ecolor': 'black', 'capsize': 1.5, 'elinewidth': 0.85, 'capthick': 0.85}
        fig, axs = plt.subplots(1, 2, figsize=(6.3,3.7), constrained_layout=True)

        # Build plot for Macro F1 scores.
        plotting_data = plot_data['raw']
        labels = [get_labels(key) for key in plotting_data.keys()]
        colors = [get_colors(label) for label in labels]
        means = [i[1] for i in plotting_data.values()]
        lower_errors = [i[1] - i[0] for i in plotting_data.values()]
        upper_errors = [i[2] - i[1] for i in plotting_data.values()]
        bars = axs[0].barh(labels, means, xerr=[lower_errors, upper_errors], 
            color=colors, edgecolor='black', linewidth=0.85, 
            align='center', error_kw=error_kw, zorder=2)
        axs[0].xaxis.grid(True, zorder=1)
        axs[0].invert_yaxis()
        axs[0].set_xlim(xmin=0.90, xmax=0.98)
        axs[0].set_xticks(ticks=[0.90, 0.92, 0.94, 0.96, 0.98])
        axs[0].set_xlabel(r'$\langle F_1 / F_{1,\text{max}} \rangle$')
        axs[0].tick_params(axis='x', length=0)

        # Build plot for baseline metric.
        plotting_data = plot_data['use_baseline']
        labels = [get_labels(key) for key in plotting_data.keys()]
        colors = [get_colors(label) for label in labels]
        means = [i[1] for i in plotting_data.values()]
        lower_errors = [i[1] - i[0] for i in plotting_data.values()]
        upper_errors = [i[2] - i[1] for i in plotting_data.values()]
        bars = axs[1].barh(labels, means, xerr=[lower_errors, upper_errors], 
            color=colors, edgecolor='black', linewidth=0.85, 
            align='center', error_kw=error_kw, zorder=2)
        axs[1].xaxis.grid(True, zorder=1)
        axs[1].invert_yaxis()
        axs[1].set_xlim(xmin=0.0, xmax=0.85)
        axs[1].set_xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8])
        axs[1].set_xlabel(r'$\langle \xi / \xi_{\text{max}} \rangle$')
        axs[1].tick_params(axis='x', length=0)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        fig.text(0.03, 0.92, 'A', size=18, weight='bold')
        fig.text(0.5, 0.92, 'B', size=18, weight='bold')
        legend_elements = [
            Patch(facecolor=get_color('nn'), edgecolor='black', linewidth=0.8, label='NN'),
            Patch(facecolor=get_color('rf'), edgecolor='black', linewidth=0.8, label='RF'),
            Patch(facecolor=get_color('gpc'), edgecolor='black', linewidth=0.8, label='GPC'),
            Patch(facecolor=get_color('gpr'), edgecolor='black', linewidth=0.8, label='GPR'),
            Patch(facecolor=get_color('sv'), edgecolor='black', linewidth=0.8, label='SV'),
            Patch(facecolor=get_color('xgb'), edgecolor='black', linewidth=0.8, label='XGB'),
        ]
        fig.legend(
            handles=legend_elements, 
            loc='lower center', 
            ncol=6, 
            fontsize=10, 
            frameon=False,
            handleheight=0.8,
            handlelength=1.5,
        )
        plt.savefig('./figures/fig3.png', dpi=1000)
        plt.show()

    # Plot suboptimal fraction for algorithms based on different models.
    if args.fig == 4:

        # Load data.
        plot_data = pickle.load(open('./data/fig4.pkl', 'rb'))
        models = [get_labels(key) for key in plot_data['Model Names']]
        del plot_data['Model Names']

        # Build plot.
        plt.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['font.size'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['legend.fontsize'] = 10
        plt.figure(figsize=(6.5,3))
        x = np.arange(len(models))
        width = 0.25
        multiplier = 0
        colors = ['#ca9fff', '#fff2a1']
        for index, (attribute, measurement) in enumerate(plot_data.items()):
            offset = width * multiplier
            rects = plt.bar(x + offset, measurement, width, label=attribute, color=colors[index], edgecolor='black', linewidth=1.0, zorder=10)
            multiplier += 1
        plt.xticks(x + 0.5 * offset, models)
        plt.xlabel('Models')
        plt.tick_params(axis='y', left=False, right=False)
        plt.ylim(ymax=1.05)
        plt.ylabel('Suboptimal Fraction')
        plt.grid(alpha=0.5, axis='y', zorder=1)
        plt.tight_layout()
        plt.legend(fancybox=False, edgecolor='white')
        plt.savefig('./figures/fig4.png', dpi=1000)
        plt.show()

    # Show comparison of AL vs. SF performance for rounds of active learning.
    if args.fig == 5:

        # Load data.
        plot_data = pickle.load(open('./data/fig5.pkl', 'rb'))
    
        # Plot figure.
        rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        colors = ['#ffca8c','#91dbff']
        labels = ['Top 20 Algorithms', 'All Algorithms']
        plt.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['legend.fontsize'] = 10
        plt.figure(figsize=(3.3,3.0))
        for index, data in enumerate(plot_data):
            plt.bar(x=rounds, height=data, color=colors[index], edgecolor='black', linewidth=1.0, zorder=10, label=labels[index])
        plt.xlabel('Rounds', fontsize=10)
        plt.xticks(np.arange(1, 11, 1.0))
        plt.ylabel('Fraction at which AL outperforms SF', fontsize=10)
        plt.ylim(ymax=1.0)
        plt.grid(alpha=0.5, axis='y', zorder=2)
        plt.tick_params(axis='y', left=False, right=False)
        plt.tight_layout()
        plt.legend(fancybox=False, edgecolor='white', loc='upper left').set_zorder(1)
        plt.savefig('./figures/fig5.png', dpi=1000)
        plt.show()

    # Plot influence of samplers on active learning and space-filling algorithm performance.
    if args.fig == 6:

        # Initial plotting parameters.
        plt.rcParams['font.family'] = 'Helvetica' 
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.titlesize'] = 12
        error_kw = {'capsize': 1.5, 'elinewidth': 0.9, 'capthick': 0.9}
        sampler_colors = {
            'maximin': '#fe7c7c',
            'medoids': '#91dbff',
            'max_entropy': '#01ce90',
            'vendi': '#ffca8c',
        }
        sampler_labels = {
            'maximin': 'Maximin',
            'medoids': 'Medoids',
            'max_entropy': 'Max Entropy',
            'vendi': 'Vendi'
        }
        rounds = [0, 1, 3, 5, 7, 10]
        fig, axs = plt.subplots(1,2,figsize=(6.5,3.2), sharey=True)

        # Make active learning plot.
        plot_data = pickle.load(open('data/fig6_al.pkl', 'rb'))
        for sampler, data in plot_data.items():
            axs[0].scatter(rounds, data[:,0], s=50.0, label=sampler_labels[sampler], color=sampler_colors[sampler], edgecolor='black', linewidth=0.9, zorder=4, clip_on=False)
            axs[0].plot(rounds, data[:,0], color=sampler_colors[sampler], zorder=3)
            axs[0].errorbar(rounds, data[:,0], yerr=data[:,1], zorder=2, ls='None', ecolor=sampler_colors[sampler], **error_kw)
        axs[0].plot([-1.0, 11.0], [1.0, 1.0], color='black', linestyle='dashed', zorder=1, linewidth=1.2)
        axs[0].tick_params(axis='both', top=True, left=True, right=True, bottom=True, direction='in', width=1.2, length=2.7)
        axs[0].set_xlabel('Rounds of Active Learning')
        axs[0].set_ylabel('Fold Improvement over Random')
        axs[0].set_xlim(xmin=-1.0, xmax=11.0)
        axs[0].set_ylim(ymin=0.6, ymax=2.0)
        axs[0].set_title('Active Learning', fontsize=14)
        axs[0].legend(fancybox=False, loc='upper right', edgecolor='white')
        axs[0].text(-0.1, 1.05, 'A', transform=axs[0].transAxes, size=18, weight='bold')

        # Make space-filling plot.
        plot_data = pickle.load(open('data/fig6_sf.pkl', 'rb'))
        for sampler, data in plot_data.items():
            axs[1].scatter(rounds, data[:,0], s=50.0, label=sampler_labels[sampler], color=sampler_colors[sampler], edgecolor='black', linewidth=1.0, zorder=4, clip_on=False)
            axs[1].plot(rounds, data[:,0], color=sampler_colors[sampler], zorder=3)
            axs[1].errorbar(rounds, data[:,0], yerr=data[:,1], zorder=2, ls='None', ecolor=sampler_colors[sampler], **error_kw)
        axs[1].plot([-1.0, 11.0], [1.0, 1.0], color='black', linestyle='dashed', zorder=1, linewidth=1.2)
        axs[1].tick_params(axis='both', top=True, left=True, right=True, bottom=True, direction='in', width=1.2, length=2.7)
        axs[1].set_xlabel('Rounds of Active Learning')
        axs[1].set_xlim(xmin=-1.0, xmax=11.0)
        axs[1].set_title('Space-Filling', fontsize=14)
        axs[1].text(-0.1, 1.05, 'B', transform=axs[1].transAxes, size=18, weight='bold')

        # Finish and save plot.
        plt.tight_layout()
        plt.savefig('./figures/fig6.png', dpi=1000)
        plt.show()

    # Plot performance of ensemble strategies.
    if args.fig == 7:

        # Load data.
        final_data = pickle.load(open('./data/fig7.pkl', 'rb'))

        # Set up plot.
        plt.rcParams['font.family'] = 'Helvetica' 
        plt.rcParams['font.size'] = 9
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 10
        error_kw = {'ecolor': 'black', 'capsize': 1.5, 'elinewidth': 0.85, 'capthick': 0.85}
        fig, axs = plt.subplots(1, 3, figsize=(7.3,3.0))

        for key_idx, key in enumerate(final_data.keys()):
            plot_data = final_data[key]
            labels = [get_labels(key) for key in plot_data.keys()]
            colors = [get_colors(label, ensemble=False) for label in labels]
            labels = [l[4:] for l in labels]
            means = [i[1] for i in plot_data.values()]
            lower_errors = [i[1] - i[0] for i in plot_data.values()]
            upper_errors = [i[2] - i[1] for i in plot_data.values()]
            bars = axs[key_idx].barh(labels, means, xerr=[lower_errors, upper_errors], 
                color=colors, edgecolor='black', linewidth=0.85, 
                align='center', error_kw=error_kw, zorder=2)
            axs[key_idx].xaxis.grid(True, zorder=1)
            axs[key_idx].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axs[key_idx].set_xlim(xmin=0.0, xmax=1.0)
            axs[key_idx].set_xlabel(r'$\langle \xi / \xi_{\text{max}} \rangle$')
            axs[key_idx].tick_params(axis='x', bottom=False, top=False)
            axs[key_idx].invert_yaxis()
            axs[key_idx].set_title('')
        axs[0].set_title('All Tasks')
        axs[1].set_title('NN Optimal Tasks')
        axs[2].set_title('RF Optimal Tasks')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.27)
        fig.text(0.03, 0.90, 'A', size=18, weight='bold')
        fig.text(0.35, 0.90, 'B', size=18, weight='bold')
        fig.text(0.68, 0.90, 'C', size=18, weight='bold')
        legend_elements = [
            Patch(facecolor=get_colors('NN', ensemble=False), edgecolor='black', linewidth=0.85, label='NN'),
            Patch(facecolor=get_colors('RF', ensemble=False), edgecolor='black', linewidth=0.85, label='RF'),
            Patch(facecolor=get_colors('Ensemble', ensemble=False), edgecolor='black', linewidth=0.9, label='Ensemble'),
        ]
        fig.legend(
            handles=legend_elements, 
            loc='lower center', 
            ncol=6, 
            fontsize=10, 
            frameon=False,
            handleheight=0.8,
            handlelength=1.6,
        )
        plt.savefig('./figures/fig7.png', dpi=1000)
        plt.show()

    # Prepare metafeatures parity plots.
    if args.fig == 8:

        # Load in data.
        final_data = pickle.load(open('./data/fig8.pkl', 'rb'))

        # Visualize parity plots.
        plt.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['font.size'] = 10
        fig, axs = plt.subplots(1,2,figsize=(5.2,2.8))

        for key_idx, key in enumerate(final_data.keys()):
            color = '#91dbff' if key_idx == 0 else '#ffca8c'
            axs[key_idx].scatter(
                final_data[key][:,0], final_data[key][:,1], 
                color=color, s=17.0, edgecolors='black', linewidth=0.85, zorder=9, clip_on=False
            )
            parity = np.linspace(0.0, 1.10, num=1000)
            axs[key_idx].plot(parity, parity, color='black', linewidth=1.2, zorder=10)
            axs[key_idx].grid(alpha=0.5, zorder=2)
            axs[key_idx].set_xlim([0.1, 1.10])
            axs[key_idx].set_ylim([0.1, 1.10])
            axs[key_idx].set_xticks(ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
            axs[key_idx].set_yticks(ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
            axs[key_idx].tick_params(axis='both', direction='in', left=True, right=True, bottom=True, top=True, width=1.2)
            axs[key_idx].set_xlabel('True ' + r'$F_1$')
            axs[key_idx].set_ylabel('Predicted ' + r'$F_1$')

        axs[0].set_title('All Algorithms')
        axs[1].set_title('Top 20 Algorithms')
        axs[0].text(-0.15, 1.00, 'A', transform=axs[0].transAxes, size=18, weight='bold')
        axs[1].text(-0.15, 1.00, 'B', transform=axs[1].transAxes, size=18, weight='bold')
        axs[0].text(0.96, 0.04, 'R\u00b2 = 0.803\nMAE = 7.826%', fontsize=10, transform=axs[0].transAxes, ha='right', va='bottom',
               bbox=dict(facecolor='white', edgecolor='white'))
        axs[1].text(0.96, 0.04, 'R\u00b2 = 0.810\nMAE = 6.152%', fontsize=10, transform=axs[1].transAxes, ha='right', va='bottom',
               bbox=dict(facecolor='white', edgecolor='white'))

        plt.tight_layout()
        plt.savefig('./figures/fig8.png', dpi=1000)
        plt.show()

    # Plot analysis of molecular feature choices.
    if args.fig == 9:

        # Building panel A.
        final_data = pickle.load(open('./data/fig9.pkl', 'rb'))

        # Setting up plot parameters.
        plt.rcParams['font.family'] = 'Helvetica' 
        plt.rcParams['font.size'] = 9
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 1.1
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 7
        plt.rcParams['figure.titlesize'] = 11
        error_kw = {'ecolor': 'black', 'capsize': 1.1, 'elinewidth': 0.9, 'capthick': 0.9}
        fig, axs = plt.subplots(1,2,figsize=(7,2.6), width_ratios=[1, 1.5])

        # Adding data to figure.
        for model, data in final_data.items():
            if '_ard' in model:
                marker = '^'
            else:
                marker = 'o'
            axs[0].scatter(data[:,0], data[:,1], s=30.0, marker=marker, color=get_color(model), edgecolor='black', linewidth=0.8, zorder=3, clip_on=False, label=get_label(model))
            axs[0].plot(data[:,0], data[:,1], color=get_color(model), alpha=0.5, zorder=2, linewidth=0.9)
            axs[0].errorbar(data[:,0], data[:,1], yerr=data[:,2], ls='none', **error_kw)
        axs[0].tick_params(axis='both', top=True, left=True, right=True, bottom=True, direction='in', width=1.1, length=3.0)
        axs[0].tick_params(axis='x', pad=7)
        axs[0].set_xlabel('Features')
        axs[0].set_xticks(ticks=[0.0, 1.0, 2.0, 3.0, 4.0], labels=['M-10', 'M-20', 'M-100', 'M-All', 'FP'])
        # axs[0].set_ylabel('Relative Performance')
        axs[0].set_ylabel(r'$\langle \xi / \xi_{\text{max}} \rangle$')
        axs[0].set_ylim(ymin=0.0, ymax=1.0)
        axs[0].legend(loc='upper left', fancybox=False, edgecolor='black', ncol=3, handletextpad=0.2, columnspacing=0.35)
        axs[0].text(-0.2, 1.00, 'A', transform=axs[0].transAxes, size=18, weight='bold')

        # Building panel B.
        tasks = ['qm9_gap', 'qm9_r2', 'qm9_cv', 'qm9_zpve',
                'qm9_u0', 'robeson', 'free', 'esol', 'lipo', 'hiv',
                'bace', 'clintox', 'muv', 'tox21']
        results = pickle.load(open('revisions_mf.pkl', 'rb'))
        original = pickle.load(open('metafeatures.pickle', 'rb'))

        # Get noise-to-signal ratio data.
        metric = 'ns_ratio'
        metric_factors = {
            '10': [],
            '20': [],
            '100': [],
            'all': [],
            'morgan': []
        }
        for task in tasks:
            metric_20 = results[f'{task}_mordred_20'][metric]
            metric_100 = results[f'{task}_mordred_100'][metric]
            metric_all = results[f'{task}_mordred_all'][metric]
            metric_morgan = results[f'{task}_morgan'][metric]
            task_idx = original['tasks'].index(task)
            mf_idx = original['feature_names'].index(metric)
            metric_10 = original['metafeatures'][task_idx][mf_idx]
            metric_factors['10'].append(metric_10)
            metric_factors['20'].append(metric_20)
            metric_factors['100'].append(metric_100)
            metric_factors['all'].append(metric_all)
            metric_factors['morgan'].append(metric_morgan)

        # Define feature names.
        feature_names = ['M-10', 'M-20', 'M-100', 'M-All', 'FP']

        # Plotting definitions..
        plt.rcParams['font.family'] = 'Helvetica' 
        plt.rcParams['font.size'] = 9
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 11
        error_kw = {'capsize': 1.1, 'elinewidth': 0.85, 'capthick': 0.85}
        ns_ratio_color = '#28EAAA'
        ns_ratio_font_color = '#22c992'
        mut_inf_color = '#FF4D6C'
        mut_inf_font_color = '#e34863'
        max_inf_color = '#91dbff'
        max_inf_font_color = '#7bb9d8'

        # Make plot for noise-to-signal ratio.
        positions = np.array(range(1,len(feature_names) + 1)) - 0.125
        boxplot = axs[1].boxplot(
            list(metric_factors.values()), 
            labels=feature_names, 
            widths=0.2,
            positions=positions,
            boxprops=dict(linewidth=1.1),        
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
            medianprops=dict(linewidth=1.1),
            flierprops=dict(linewidth=1.1, markersize=4.0),
            patch_artist=True
        )
        for median in boxplot['medians']:
            median.set_color('black')
        for patch in boxplot['boxes']:
            patch.set_facecolor(ns_ratio_color)
        axs[1].set_xlabel('Features')
        axs[1].set_ylabel('Noise-to-Signal Ratio', color=ns_ratio_font_color)
        axs[1].set_xticks(ticks=[], labels=[])
        axs[1].tick_params(axis='both', direction='in', left=True, bottom=False, right=True, top=False, width=1.25, length=3.5)
        axs[1].set_ylim(ymin=0.0, ymax=250.0)
        axs[1].tick_params(axis='y', labelcolor=ns_ratio_font_color)
        axs[1].tick_params(axis='x', pad=7)
        axs[1].text(-0.2, 1.00, 'B', transform=axs[1].transAxes, size=18, weight='bold')

        # Gather average mutual information data.
        metric = 'mut_inf.mean'
        metric_factors = {
            '10': [],
            '20': [],
            '100': [],
            'all': [],
            'morgan': []
        }
        for task in tasks:
            metric_20 = results[f'{task}_mordred_20'][metric]
            metric_100 = results[f'{task}_mordred_100'][metric]
            metric_all = results[f'{task}_mordred_all'][metric]
            metric_morgan = results[f'{task}_morgan'][metric]
            task_idx = original['tasks'].index(task)
            mf_idx = original['feature_names'].index(metric)
            metric_10 = original['metafeatures'][task_idx][mf_idx]
            metric_factors['10'].append(metric_10)
            metric_factors['20'].append(metric_20)
            metric_factors['100'].append(metric_100)
            metric_factors['all'].append(metric_all)
            metric_factors['morgan'].append(metric_morgan)

        # Make plot for average mutual information.
        ax2 = axs[1].twinx()
        positions = np.array(range(1,len(feature_names) + 1)) + 0.125
        boxplot = ax2.boxplot(
            list(metric_factors.values()), 
            labels=feature_names, 
            widths=0.2,
            positions=positions,
            boxprops=dict(linewidth=1.1),        
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
            medianprops=dict(linewidth=1.1),
            flierprops=dict(linewidth=1.1, markersize=4.0),
            patch_artist=True
        )
        for median in boxplot['medians']:
            median.set_color('black')
        for patch in boxplot['boxes']:
            patch.set_facecolor(mut_inf_color)
        ax2.set_ylabel('Avg. Mutual Information', color=mut_inf_font_color)
        ax2.set_xticks(ticks=range(1,len(feature_names)+1), labels=feature_names)
        ax2.tick_params(axis='y', direction='in', right=True, width=1.25, length=3.5)
        ax2.set_yticks(ticks=[0.0, 0.10, 0.20, 0.30])
        ax2.set_ylim(ymin=-0.01)
        ax2.tick_params(axis='y', labelcolor=mut_inf_font_color)

        # Finalize plot and save to file.
        plt.tight_layout()
        plt.savefig('./figures/fig9.png', dpi=1000)
        plt.show()
