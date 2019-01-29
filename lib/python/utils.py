#!/usr/bin/env python3

""" Helper functions config sanitity checks.

"""

import os
import re
import yaml
import argparse
import multiprocessing


parser = argparse.ArgumentParser(description='Check configs for completeness.')
parser.add_argument(
    '-c', '--config', help='Path to config file'
)
parser.add_argument(
    '-t', '--type', help='Config type. Valid options are "tr" and "pp"'
)

CONFIG_DEFAULTS = {
    'static_data': {
        'high_resolution': False, 
        'cores': multiprocessing.cpu_count(),
        'peak_min_width': 0.5,
        'peak_min_mz': 3
    },
    'algorithms': {'XCMS-CW': True, 'XCMS-MF': True},
    'default_configs': {
        'directory': '../../pp_configs',
        'XCMS-CW': 'XCMS-CW_default.INI',
        'XCMS-MF': 'XCMS-MF_default.INI'
    },
    'XCMS_params': {'groupCorr': False},
    'merging': {'RT_tol': 0.2}
}


class bcolors:
    """ String commands for coloring of stdout.

    """
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'


class ConfigError(Exception):
    """ Error to be thrown if config file is corruped.

    Args:
        message (str): Error message to display in stdout.

    """
    def __init__(self, message):
        super().__init__('\n{}\n'.format(message))


def split_path(f):
    """ Split a file path in it's different levels.

    Args:
        f (str): Absolute or relative file path.

    Returns:
        list of str: each str is one level in the file system.
    """
    return f.split(os.sep)


def evaluate_training_config(config):
    """ Evaluate config segments relevant for Snakefile_tr.

    Args:
        config (): Loaded Snakemake config.

    Returns:
        tuple of:
            algorithms (list): Algorithms for peak picking.
            training_data (dict):
                key: 'files', values (list): training data files.
                key: file_type, values (str): sample file extensions.

    """
    config = _update_config(config, CONFIG_DEFAULTS)
    config = _update_training(config)

    algorithms = [i for i, j in config['algorithms'].items() if j == True]

    training_data = {'params': _get_parameters(config, 'training_data-params')}
    tr_dir = os.path.join(
        config['static_data']['data_path'],
        config['training_data-general']['training_samples']
    )
    training_data['files'], training_data['type'] = _get_files(tr_dir)
    return (algorithms, training_data)


def evaluate_peakpicking_config(config):
    """ Evaluate config segments relevant for Snakefile_pp.

    Args:
        config (): Loaded Snakemake config.

    Returns:
        tuple of:
            algorithms (list): Algorithms for peak picking.
            sample_data (dict):
                key: 'files', values (list of str): All experiment sample files.
                key: 'group', values (list of str): All experimental groups.
                key: 'type', values (str): sample file extensions.
            optimization_data (dict):
                key: 'params': values (dict):
                    key: algorithm str, values (list of str): all possible
                        algorithm parameter permutations. 
                key: 'files', values (list of str): optimization data files.
                key: file_type, values (str): sample file extensions.
            wash_file (str): Wash file name.

    """
    config = _update_config(config, CONFIG_DEFAULTS)
    config = _update_peak_picking(config)

    algorithms = [i for i, j in config['algorithms'].items() if j == True]

    sample_data = _get_sample_files(config)

    wash_dir = os.path.join(
        config['static_data']['data_path'], config['retention_index']['wash']
    )
    wash_files, _ = _get_files(wash_dir)
    try:
        wash_file = os.path.join(config['retention_index']['wash'], wash_files[0])
    except IndexError:
        wash_file = None
    else:
        sample_data['files'].append(wash_file)

    optimization_data = {
        'params': _get_parameters(config, 'grid_search-params'),
    }
    opt_dir = os.path.join(
        config['static_data']['data_path'],
        config['optimization-general']['optimization_samples']
    )
    optimization_data['files'], _ = _get_files(opt_dir)

    return (algorithms, sample_data, optimization_data, wash_file)


def check_basics(cfg):
    """ Check if all relevant config arguments are spüecified

    Args:
        cfg (): Loaded Snakemake config.

    """
    _update_config(cfg, CONFIG_DEFAULTS, verbose=True)
    _check_config_sections(cfg, ['static_data'])

    if not cfg['static_data'].get('data_path', False):
        raise ConfigError(
            '"data_path" (section "static_data") needs to be defined'
        )
    _check_input_data(cfg['static_data']['data_path'])

    CW_default_cfg = os.path.join(
        config['default_configs']['directory'],
        config['default_configs']['XCMS-CW']
    )
    if not os.path.exists(CW_default_cfg):
        raise OSError(
            'XCMS-CW default config folder not found: {}'.format(CW_default_cfg)
        )
    MF_default_cfg = os.path.join(
        config['default_configs']['directory'],
        config['default_configs']['XCMS-MF']
    )
    if not os.path.exists(MF_default_cfg):
        raise OSError(
            'XCMS-MF default config folder not found: {}'.format(CW_default_cfg)
        )
    

def _check_input_data(in_path):
    if not os.path.isdir(in_path):
        raise OSError('"data_path" folder not found: {}'.format(in_path))

    # Check if data folder contains subfolders with groups
    subfolders = os.listdir(in_path)
    all_empty = True
    print('{}Groups in data dir: {}'.format(bcolors.OKGREEN, bcolors.ENDC))
    for sub in subfolders:
        subdir = os.path.join(in_path, sub)
        if os.path.isdir(subdir):
            sample_no = len(_get_files(subdir)[0])
            if sample_no == 0:
                print(
                    '{}Empty group folder: {}{}' \
                        .format(bcolors.WARNING, subdir, bcolors.ENDC)
                )
            else:
                print(
                    '\t{}{:>2} samples:\t{}{}' \
                        .format(bcolors.OKGREEN, sample_no, sub, bcolors.ENDC)
                )
                all_empty = False
    print()
    if all_empty:
        raise OSError('Nothing in subfolders of: {}'.format(in_path))


def check_training(cfg):
    _update_training(cfg, verbose=True)
    _check_config_sections(cfg, ['training_data-general'])

    if not cfg['training_data-general'].get('training_samples', False):
        raise ConfigError(
            '"training_samples" (section "training_data-general") needs to be '
            'defined'
        )
    tr_data_path = os.path.join(
        cfg['static_data']['data_path'],
        cfg['training_data-general']['training_samples']
    )
    if not os.path.isdir(tr_data_path):
        raise OSError(
            '"training_samples" folder not found: {}'.format(tr_data_path)
        )


def _update_training(cfg, verbose=False):
    config_defaults = {
        'training_data-general': {'plots_per_sample': 200},
        'training_data-params': {
            'XCMS-CW_pwMin': [1, 2.5, 5], 'XCMS-CW_pwMax': [5, 7.5, 10],
            'XCMS-MF_fwhm': [2.5, 5, 7.5], 'XCMS-MF_sn': [1, 2.5, 5]
        }
    }
    if cfg['static_data']['high_resolution']:
        config_defaults['training_data-params'].update(
            {'XCMS-CW_mzdiff': [-0.1, 0, 0.1, 0.5], 'XCMS-CW_ppm': [5, 10, 20],
            'XCMS-MF_mzdiff': [-0.1, 0, 0.1, 0.5], 'XCMS-MF_steps': [1, 2, 3],
            'XCMS-MF_step': [0.1, 0.25, 0.5]}
        )
    return _update_config(cfg, config_defaults, verbose)


def check_peak_picking(cfg):
    _check_config_sections(
        cfg, ['optimization-general', 'grid_search-params']
    )
    if not cfg['optimization-general'].get('optimization_samples', False):
        raise ConfigError(
            '"optimization_samples" (section "optimization-general") needs to '
            'be defined'
        )
    opt_data_path = os.path.join(
        cfg['static_data']['data_path'],
        cfg['optimization-general']['optimization_samples']
    )
    if not os.path.isdir(opt_data_path):
        raise OSError(
            '"optimization_samples" folder not found: {}'.format(opt_data_path)
        )
    _update_peak_picking(cfg, verbose=True)


def _update_peak_picking(cfg, verbose=False):
    config_defaults = {
        'classifier': {
            'path': '', 'SVM': True, 'RF': False,
            'validation_set_size': 0.2, 'cross_validation_splits': 3
        },
        'optimization-score': {
            'apex_left': 5,
            'compound_related_peak': 10,
            'apex_right': 5,
            'merged_peak': -10,
            'too_narrow_borders': -10,
            'too_wide_borders': -10,
            'noise': -10
        },
        'grid_search-params': {
            'XCMS-CW_pwMin': [1, 2, 3, 4, 5],
            'XCMS-CW_pwMax': [2,3,4,5,6,7,8,9,10],
            'XCMS-MF_fwhm': [1, 2, 3, 4, 5, 6, 7, 8],
            'XCMS-MF_sn': [0.5, 1, 2, 3, 4, 5, 10, 20, 30]
        },
        'retention_index': {'calculate': False, 'alkanes': []},
        'alignment': {
            'RI_tol': 2, 'RT_tol': 5, 'min_similarity': 0.8, 'min_samples': 0
        }
    }
    if cfg['static_data']['high_resolution']:
        config_defaults['grid_search-params']. update(
            {'XCMS-CW_mzdiff': [-0.1, 0, 0.1, 0.5], 'XCMS-CW_ppm': [5, 10, 20],
            'XCMS-MF_mzdiff': [-0.2, -0.1, 0, 0.1, 0.2],
            'XCMS-MF_step': [0.1, 0.2, 0.3, 0.4, 0.5],
            'XCMS-MF_steps': [1, 2, 3]}
        )
    config = _update_config(cfg, config_defaults, verbose)
    if not config['retention_index'].get('wash', False):
        config['retention_index']['wash'] = 'not_defined'

    return config


def _check_config_sections(cfg, sections):
    for sec in sections:
        if not cfg.get(sec, False):
            raise ConfigError('Section "{}" needs to be defined'.format(sec))


def _update_config(cfg, update_dict, verbose=False):
    for section, section_args in update_dict.items():
        if not cfg.get(section, False):
            if verbose:
                print(
                    '{}Section "{}" not defined: added{}' \
                        .format(bcolors.WARNING, section, bcolors.ENDC)
                )
            cfg[section] = {}

        for arg, arg_default in section_args.items():
            if not arg in cfg[section]:
                if verbose:
                    message = '"{}" (section "{}") not defined: set to "{}"' \
                        .format(arg, section, arg_default)
                    print('{}{}{}'.format(bcolors.WARNING, message, bcolors.ENDC))
                cfg[section][arg] = arg_default
    return cfg
    

def _get_sample_files(config):
    data_dir = config['static_data']['data_path']
    exclude = [
        config['training_data-general']['training_samples'],
        config['optimization-general']['optimization_samples'],
        config['retention_index']['wash']
    ]
    sample_data = {'files': [], 'groups': []}
    for group in os.listdir(data_dir):
        if group in exclude:
            continue
        group_files, file_type = _get_files(os.path.join(data_dir, group))
        sample_data['files'].extend(
            [os.path.join(group, i) for i in group_files]
        )
        sample_data['groups'].append(group)
    sample_data['type'] = file_type
    return sample_data


def _get_files(file_path):
    if not os.path.isdir(file_path):
        return ([], '')

    samples = [i for i in os.listdir(file_path)]
    sample_names = list(set([re.sub('\.[a-zA-Z]+$', '', i) for i in samples]))
    try:
        file_type = [i for i in samples if not i.endswith('pkl')][0] \
            .split('.')[-1]
    except IndexError:
        file_type = None
    return (sample_names, file_type)
        

def _get_parameters(config, param_section):
    if config['static_data']['high_resolution']:
        pos_mapping ={
            'pwMin': 0, 'pwMax': 1, 'mzdiff': 2, 'ppm': 3, # XCMS-CW
            'fwhm': 0, 'sn': 1, 'step': 3, 'steps': 4, # XCMS-MF
        }
    else:
        pos_mapping ={'pwMin': 0, 'pwMax': 1, 'fwhm': 0, 'sn': 1}

    params = {}
    for name_raw, vals in config[param_section].items():
        name = name_raw.split('_')
        if not name[0] in params:
            params[name[0]] = (max(pos_mapping.values()) + 1) * [None]
        if name[1] in pos_mapping:
            params[name[0]][pos_mapping[name[1]]] = vals
    
    def _get_permutations(not_combined, combined):
        if not not_combined:
            return combined
        to_add = not_combined.pop(0)
        new_combined = []
        for par_old in combined:
            for par_new in to_add:
                new_combined.append(par_old + [par_new])
        
        return _get_permutations(not_combined, new_combined)

    final_params = {}
    for alg, vals in params.items():
        real_vals = [i for i in vals if i]
        first_vals = [[i] for i in real_vals[0]]
        all_val_combis = _get_permutations(real_vals[1:], first_vals)

        # Filter values where PwMax < pwMin
        if alg == 'XCMS-CW':
            final_combis = [
                i for i in all_val_combis if i[0] < i[1]
            ]
        else:
            final_combis = all_val_combis

        combi_str = [
            '_'.join(['par{}-{}'.format(idx, val) for idx, val in enumerate(combi)]) \
                 for combi in final_combis
        ]
        final_params[alg] = combi_str

    return final_params


if __name__ == '__main__':
    args = parser.parse_args()

    config_file = args.config
    if config_file == '':
        config_file = 'config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    check_basics(config)
    if args.type == 'tr':
        check_training(config)
    elif args.type == 'pp':
        check_peak_picking(config)
    else:
        raise TypeError('Unknown config type: {}'.format(args.type))

    print('{}Config checked!\n{}'.format(bcolors.OKGREEN, bcolors.ENDC))
