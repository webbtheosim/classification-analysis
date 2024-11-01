def get_labels(key, mol_feat=False):
    '''Method for cleaning up the label of the key entry for plotting.'''

    # Split up key into component parts.
    vals = key.split('-')

    # Replace each component with appropriate entry.
    for index, value in enumerate(vals):
        if value == 'al':
            vals[index] = 'AL,'
        if value == 'sf':
            vals[index] = 'SF,'
        if value == 'random':
            vals[index] = 'Random,'
        if value == 'maximin':
            vals[index] = 'Maximin,'
        if value == 'medoids':
            vals[index] = 'Medoids,'
        if value == 'max_entropy':
            vals[index] = 'Max Ent.,'
        if value == 'vendi':
            vals[index] = 'Vendi,'
        if value == 'gpc':
            vals[index] = 'GPC'
        if value == 'gpc_ard':
            vals[index] = 'GPC (ARD)'
        if value == 'gpr':
            vals[index] = 'GPR'
        if value == 'gpr_ard':
            vals[index] = 'GPR (ARD)'
        if value == 'bkde':
            vals[index] = 'BKDE'
        if value == 'lp':
            vals[index] = 'LP'
        if value == 'nn':
            vals[index] = 'NN'
        if value == 'rf':
            vals[index] = 'RF'
        if value == 'sv':
            vals[index] = 'SV'
        if value == 'xgb':
            vals[index] = 'XGB'
        if value == 'ensemble_top':
            vals[index] = 'Ensemble'
        if value == 'ensemble_averaging':
            vals[index] = 'Averaging'
        if value == 'ensemble_stacking':
            vals[index] = 'Stacking'
        if value == 'ensemble_arbitrating':
            vals[index] = 'Arbitrating'
        if value == 'mordred_10':
            vals[index] = '-Mordred (10)'
        if value == 'mordred_20':
            vals[index] = '-Mordred (20)'
        if value == 'mordred_100':
            vals[index] = '-Mordred (100)'
        if value == 'mordred_all':
            vals[index] = '-Mordred (All)'
        if value == 'morgan':
            vals[index] = '-Morgan'

    if mol_feat:
        final_str = ''
        for val in vals[0:-2]:
            final_str += val + ' '
        final_str += vals[-2]
        final_str += vals[-1]
        
        return final_str
    
    else:
        return ' '.join(vals)

def get_colors(label, ensemble=False):
    '''Method for determining color of the bar based on the label.'''

    # Color definitions.
    if not ensemble:
        colors = {
            'NN': '#22B3FF',
            'RF': '#FF4D6C',
            'XGB': '#B77EFF',
            'GPR (ARD)': '#28EAAA',
            'GPR': '#28EAAA',
            'GPC (ARD)': '#00C086',
            'GPC': '#00C086',
            'SV': '#FFB256',
            'LP': 'white',
            'BKDE': 'white',
            'E': '#FFD516',
        }

        colors = {
            'NN': '#91dbff',
            'RF': '#fe7c7c',
            'XGB': '#ca9fff',
            'GPR (ARD)': '#a7f8dd',
            'GPR': '#a7f8dd',
            'GPC (ARD)': '#01ce90',
            'GPC': '#01ce90',
            'SV': '#ffca8c',
            'LP': 'white',
            'BKDE': 'white',
            'Ensemble': '#fff2a1',
        }

    else:
        colors = {
            'Hyperparameter': '#91dbff',
            'Averaging': '#fe7c7c',
            'Stacking': '#a7f8dd',
            'Arbitrating': '#ca9fff'
        }

    # Provide appropriate color for provided label.
    for key in colors.keys():
        if key in label:
            return colors[key]

def load_baseline_raw():

    # Load individual raw baseline files.
    import pickle
    data_1 = pickle.load(open('baseline/baseline_raw_1.pickle', 'rb'))
    data_2 = pickle.load(open('baseline/baseline_raw_2.pickle', 'rb'))
    data_3 = pickle.load(open('baseline/baseline_raw_3.pickle', 'rb'))
    data_4 = pickle.load(open('baseline/baseline_raw_4.pickle', 'rb'))

    # Load data into a single dictionary. 
    data = {}
    for task in data_1.keys():
        data[task] = data_1[task]
    for task in data_2.keys():
        data[task] = data_2[task]
    for task in data_3.keys():
        data[task] = data_3[task]
    for task in data_4.keys():
        data[task] = data_4[task]
        
    return data

# Common groupings of tasks for sensitivity analyses.
all_tasks = ['bace', 'bear', 'clintox', 'diblock', 'electro', 'esol', 'free', 'glotzer_pf', 
    'glotzer_xa', 'hiv', 'hplc', 'lipo', 'muv', 'oer', 'oxidation', 'perovskite', 
    'polygel', 'polysol', 'princeton', 'qm9_cv', 'qm9_gap', 'qm9_r2', 'qm9_u0', 
    'qm9_zpve', 'robeson', 'shower', 'toporg', 'tox21', 'vdw', 'water_hp', 'water_lp'
]
rf_tasks = ['clintox', 'diblock', 'esol', 'hplc', 'perovskite', 'polygel', 'polysol', 
    'qm9_r2', 'qm9_zpve', 'robeson'
]
nn_tasks = [
    'bear', 'free', 'glotzer_pf', 'glotzer_xa', 'muv', 'toporg', 'vdw', 'water_hp', 'water_lp'
]
high_d_tasks = [
    'bace', 'bear', 'clintox', 'esol', 'free', 'hiv', 'lipo', 'muv', 'perovskite', 'polygel', 
    'polysol', 'qm9_cv', 'qm9_gap', 'qm9_r2', 'qm9_u0', 'qm9_zpve', 'robeson', 'toporg', 'tox21'
]
low_d_tasks = [
    'glotzer_pf', 'glotzer_xa', 'oxidation', 'princeton', 'shower', 'vdw', 'water_hp', 'water_lp',
    'diblock', 'bear', 'electro', 'hplc', 'oer'
]