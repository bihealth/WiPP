#!/usr/bin/env python3

""" Script for the generation/copying of a classifier, peak picking algorithm
    parameter optimization and merging of detected peaks

"""

import os
import sys
import re
import shutil
import math
import time
import pickle
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import utils
from peak_annotation import valid_classes
from detectedpeaks import DetectedXCMSPeaks
from sampledata import load_sampledata


def train_classifier(algorithm, annot_file, tr_data_dir, f_type,
            validation_size, cv, SVM, RF, cores):
    """ Optimize hyperparameters for classifier, calculate accuracy on the 
        validation set, train the classifier on all annotated peaks and save it
        to the file system.

    Args:
        algorithm (str): Algorithm description for file system.
        annot_files (str): Absolute or relative path to the file containing the
            manually annotated peaks.
        tr_data_dir (str): Absolute or relative path to the directory containing
            the training samples.
        f_type (str). File type of the training samples.
        validation_size (float): Ratio of peaks to keep aside as validation set.
        cv (int): Number of folds in the Stratified-KFold cross validation.
        SVM (bool): True if SVM evaluation, False otherwise.
        RF (bool): True if RF evaluation, False otherwise.
        cores (int): Number of available cores.

    """

    # Generate validation set and keep aside
    data_opt, class_opt, data_val, class_val, rt_dim = _train_val_split(
        annot_file, tr_data_dir, f_type, validation_size
    )
    
    # Optimize Hyperparamters for SVM
    if SVM:
        SVM_score, SVM_params = _optimize_hypperparameter(
            svm.SVC(), data_opt, class_opt, cv, rt_dim, algorithm, cores
        )
        print('Best accuracy SVM: {}'.format(SVM_score))
    else:
        SVM_score = -math.inf
    # Optimize Hyperparamters for RF
    if RF:
        RF_score, RF_params = _optimize_hypperparameter(
            RandomForestClassifier(),
            data_opt, class_opt, cv, rt_dim, algorithm, cores
        )
        print('Best accuracy RF: {}'.format(RF_score))
    else:
        RF_score = -math.inf

    if SVM_score >= RF_score:
        SVM_params['probability'] = True
        clf = svm.SVC(**SVM_params)
    else:
        RF_params['probability'] = True
        clf = RandomForestClassifier(**RF_params)

    # Fit all test data to classifier
    clf.fit(data_opt, class_opt)
    # Final validation
    class_predict = clf.predict(data_val)
    accuracy = accuracy_score(class_val, class_predict)
    print(
        '\n{l}\nReal achieved accuracy: {a}\n{l}\n'.format(a=accuracy, l='-'*80)
    )

    # Store training and validation data
    clf_data = {
        'train_peaks': data_opt, 'train_classes': class_opt,
        'val_peaks': data_val, 'val_classes': class_val
    }
    # Fit all classified data to classifier
    all_peaks = data_opt + data_val
    all_classes = class_opt + class_val
    clf.fit(all_peaks, all_classes)

    # Save classifier
    clf_file = os.path.join(
        '01_Classifier', '{}__classifier.pkl'.format(algorithm)
    )
    joblib.dump(clf, clf_file, compress=3)
    # Safe data
    clf_data_file = os.path.join(
        '01_Classifier', '{}__training_data.pkl'.format(algorithm)
    )
    pickle.dump(clf_data, open(clf_data_file, 'wb'))
    # Safe retention time interpolation
    rt_file = os.path.join('01_Classifier', '{}__rt_dim.txt'.format(algorithm))
    with open(rt_file, 'w') as f:
        f.write(str(rt_dim))  
    


def _train_val_split(annot_file, data_dir, f_type, val_size):
    # Read data
    data = pd.read_csv(annot_file, sep='\t')
    data['class'] = data['class'].astype(int)
    data['sample'] = data['visualization_file'] \
        .apply(lambda x: os.path.basename(x).split('__')[0])

    y_train = []
    x_train = []
    avg_rt = None

    # Iterate over all training samples
    for g_file, g_data in data.groupby('sample'):
        print('\tExtracting peaks: {}'.format(g_file))
        sample = load_sampledata(
            os.path.join(data_dir, '{}.{}'.format(g_file, f_type))
        )

        if not avg_rt:
            avg_rt = round(data['peak_width'].mean() * sample.scans_per_sec)
        # Shuffle data
        g_data_shuffle = g_data.sample(frac=1)

        y_train.extend(g_data_shuffle['class'].tolist())
        x_sample = []
        for peak_idx, peak in g_data_shuffle.iterrows():
            x_sample.append(
                sample.get_processed_peak(peak[['rtmin', 'rtmax']], avg_rt)
            )
        x_train.extend(x_sample)

    x_val = []
    y_val = []
    # Pick random data for final validation (equal from all groups)
    for single_class in set(y_train):
        # Get number of peaks in this class for final validation
        val_no = int(np.round(y_train.count(single_class) * val_size))
        for i in range(val_no):
            old_idx = y_train.index(single_class)
            y_val.append(y_train.pop(old_idx))
            x_val.append(x_train.pop(old_idx))

    return (x_train, y_train, x_val, y_val, avg_rt)


def _optimize_hypperparameter(clf_in, data_opt, class_opt, cv_no, rt_dim,
        alg, cores):
    if isinstance(clf_in, svm.SVC):
        clf_type = 'SVM'
        param_grid = [
            {'C': np.logspace(-3, 3, 7), 'kernel': ['linear']},
            {'C': np.logspace(-3, 3, 7), 'kernel': ['rbf'],
                'gamma': np.logspace(-3, 3, 7)},
            # {'C': np.logspace(-3, 3, 7), 'kernel': ['poly'],
            #     'degree': np.arange(2,7), 'coef0': np.logspace(-3, 3, 7)},
            # {'C': np.logspace(-3, 3, 7), 'kernel': ['sigmoid'], 
            #     'coef0': np.logspace(-3, 3, 7)}
        ]
    elif isinstance(clf_in, RandomForestClassifier):
        clf_type = 'RF'
        est_array = np.append(np.linspace(10, 90, 5), np.linspace(100, 500, 5))
        param_grid = [{'n_estimators': [int(i) for i in est_array]}]

    print('{}\nOptimizing Hyperparameters for {}\n'.format('-'*80, clf_type))   

    # Redirect stdout to file
    original_stdout = sys.stdout
    tmp_stdout = os.path.join(
        '01_Classifier',
        'Hyperparameters__{}__{}_{}.txt'.format(alg, clf_type, rt_dim)
    )
    hyperparameter_file = open(tmp_stdout, 'w')
    sys.stdout = hyperparameter_file
    clf = GridSearchCV(
        clf_in, param_grid,
        n_jobs=cores, return_train_score=True, cv=cv_no, verbose=3
    )
    clf.fit(data_opt, class_opt)
    # Direct stdout back to console
    hyperparameter_file.close
    sys.stdout = original_stdout

    print('Best Score:\t{}'.format(clf.best_score_))
    par_str = '\n'.join(
        ['\t{}:\t\t{}'.format(i, j) for i,j in clf.best_params_.items()]
    )
    print('Parameters:\n{}'.format(par_str))
    return (clf.best_score_, clf.best_params_)


def load_classifier(clf_path):
    """ Loads classifier from file system
    
    Args:
        clf_path (str): Absolute or relative path to compressed classifier.

    Returns:
        sklearn.SVM/ sklearn.RF: Previously trained classifier.

    """
    return joblib.load(in_path)


def generate_classifier(algorithm, tr_data_dir, f_type,
            validation_size=1/6, cv=5, SVM=True, RF=False, cores=None):
    """ Generates classifier for the classificaion of peaks detected by a peak 
        picking algorithm and saves it to the file system.

    Args:
        algorithm (str): Algorithm description for file system.
        tr_data_dir (str): Absolute or relative path to the directory containing
            the training samples.
        f_type (str). File type of the training samples.
        validation_size (float): Ratio of peaks to keep aside as validation set
            (Default: 0.167).
        cv (int): Number of folds in the Stratified-KFold cross validation
            (Default: 5).
        cores (int): Number of available cores. (Default: maximal available core
            number - 2).

    """
    if not cores:
        cores = os.cpu_count() - 2

    # Check if classifier already exists
    clf_path = os.path.join(
        '01_Classifier', '{}__classifier.pkl'.format(algorithm)
    )
    if os.path.exists(clf_path):
        print('Classifier already existing')
    else:
        # Check if at least one classifier is set to True
        if not SVM and not RF:
            raise OSError('Set at least one classifier "True".')

        # Check if annotated data exists
        annot_data_file = os.path.join(
            '00_Training', '03_Annotation', '{}__annotated.csv'.format(algorithm)
        )
        if not os.path.exists(annot_data_file):
            raise OSError('No annotated peaks found.')

        print('Generating classifier for: {}'.format(algorithm))
        train_classifier(
            algorithm, annot_data_file, tr_data_dir, f_type,
            validation_size, cv, SVM, RF, cores
        )


def copy_classifier(clf_dir_path, alg):
    """ Copies existing classifier to current project directory.

    Args:
        clf_dir_path (str): Absolute or relative path to existing classifier 
            folder.
        alg (str): Algorithm description for file system.

    """
    # Copy pickled classifier
    o_clf_path = os.path.join(clf_dir_path, '{}__classifier.pkl'.format(alg))
    n_clf_path = os.path.join('01_Classifier', '{}__classifier.pkl'.format(alg))
    shutil.copy(o_clf_path, n_clf_path)
    # Copy rt file
    o_rt_file = os.path.join(clf_dir_path, '{}__rt_dim.txt'.format(alg))
    n_rt_file = os.path.join('01_Classifier', '{}__rt_dim.txt'.format(alg))
    shutil.copy(o_rt_file, n_rt_file)


def classify_XCMS_peaks(clf_file, res_file, out_file, sample_file):
    """ Classifies peaks detected by XCMS

    Args:
        clf_file (str): Absolute or relative path to a classifier.
        res_file (str): Absolute or relative path to a XCMS output file 
            containing detected peaks.
        out_file (str): Absolute or relative path to the prospective output file.
        sample_file (str): Absolute or relative path to the corresponding sample
            file.
        f_type (str): File type of the data samples.

    """
    algorithm = os.path.basename(clf_file).split('__')[0]
    peaks = DetectedXCMSPeaks(res_file, algorithm, 0, 1, 0)
    _classify_single_result(peaks, clf_file, res_file, out_file, sample_file)   


def _classify_single_result(peaks, clf_file, res_file, out_file, sample_file,
            counts_file=None):   
    clf = joblib.load(clf_file)
    rt_file = clf_file.replace('classifier.pkl', 'rt_dim.txt')
    rt_dim = float(open(rt_file, 'r').read())

    peaks.classify_peaks(clf, rt_dim, sample_file)
    peaks.save_data(out_file)
    if counts_file:
        _save_counts(
            peaks.get_data()['class'], clf.classes_, _get_params(res_file),
            os.path.basename(sample_path), counts_file
        )
    

def _save_counts(class_data, clf_classes, params, sample_name, out_file):
    pred_classes, pred_counts = np.unique(class_data, return_counts=True)

    # Fill missing classes with zero if necessary
    if len(pred_classes) != clf_classes.size:
        filled_counts = np.zeros(clf_classes.size)
        for idx, mis_class in enumerate(pred_classes):
            mis_class_idx  = np.where(clf_classes == mis_class)[0]
            filled_counts[mis_class_idx] = pred_counts[idx]
        pred_counts = filled_counts.astype(int)

    final = pd.Series(params, index=['par1', 'par2', 'par3', 'par4', 'par5'])
    final[sample_name] = ','.join([str(i) for i in pred_counts])
    final.to_frame().T.to_csv(out_file, sep='\t', index=False)


def _get_params(file):
    par0 = re.findall('_par0-(\d+\.\d+|\d+)', file)[0]
    par1 = re.findall('_par1-(\d+\.\d+|\d+)', file)[0]
    try:
        par2 = re.findall('_par2-(-?\d+\.\d+|\d+)', file)[0]
    except IndexError:
        par2 = None
    try:
        par3 = re.findall('_par3-(-?\d+\.\d+|\d+)', file)[0]
    except IndexError:
        par3 = None
    try:
        par4 = re.findall('_par4-(-?\d+\.\d+|\d+)', file)[0]
    except IndexError:
        par4 = None
    return [par0, par1, par2, par3, par4]


def evaluate_grid_search(res_files, out_file, raw_data_dir, f_type, scoring):
    """ Calculate a score for each parameter tuple based on the number of peaks
    and the classes of detected peaks.

    Args:
        res_files (list of str): Absolute or relative path to classified peaks
            detected in the optimization samples.
        out_file (str): Absolute or relative path to the prospective output file.
        raw_data_dir (str): Absolute or relative path to the data directory
            containing all samples.
        f_type (str): File type of the data samples.
        scoring (dict of str: str): Scoring values for the several classes. Key:
            class number, value: scoring value for the class.

    """

    # Load scoring
    str_to_class = {j: int(i) for i, j in valid_classes.items()}
    scoring = {str_to_class[i]: j for i, j in scoring.items()}
    # Group result files by samples
    sample_results = {}
    for res_file in res_files:
        sample = os.path.basename(res_file).split('__')[0]
        try:
            sample_results[sample].append(res_file)
        except KeyError:
            sample_results[sample] = [res_file]
    algorithm = utils.split_path(res_files[0])[1]
    # Iterate over detected peaks within one sample but detected by different
    # parameter tuples.
    for sample, sample_files in sample_results.items():
        sample_res = pd.DataFrame()
        for res_file in sample_files:     
            peaks = DetectedXCMSPeaks(res_file, algorithm, 0, 1, 0)
            score = peaks.get_score(scoring)
            idx = pd.MultiIndex.from_tuples(
                [tuple(peaks.params.values())], names=peaks.params.keys()
            )
            new_row = pd.DataFrame({sample: score}, index=idx)

            sample_res = sample_res.append(new_row, sort=False)    
        sample_res.sort_index(inplace=True)
        try:
            results = pd.concat([results, sample_res], axis=1)
        except NameError:
            results = sample_res

    # Get mean scores and highest scored parameters
    results['mean'] = results.mean(axis=1)
    par_best = results['mean'].idxmax()
    par_best_dict = {}
    with open(out_file, 'w') as f:
        for idx, par_str in enumerate(results.index.names):
            f.write('{}\t{}\n'.format(par_str, par_best[idx]))
            par_best_dict[par_str] = par_best[idx]

    # Save all results 
    overview_file = out_file.replace('00', '01') \
        .replace('best_parameters', 'all_results') 
    results.to_csv(overview_file, sep='\t')
    # Sleep 1 sec before continuing pipeline to prevent tangling
    return par_best_dict



def merge_detected_peaks(peak_files, out_file, raw_data_dir, f_type, tol,
            min_pw, min_mz):
    """ Merge peaks detected by different algorithms in the same sample

    Args:
        peak_files (list of str): Absolute or relative path to classified peaks
            detected by the different algorithms.
        out_file (str): Absolute or relative path to the prospective output file.
        raw_data_dir (str): Absolute or relative path to the data directory
            containing all samples.
        f_type (str): File type of the data samples.
        tol (int|float): RT window within which peaks are merged.
        min_pw (int|float): Minimum peak width necessary for considering a 
            detected peak a real peak.
        min_mz (int): Minimum number of detected m/z values necessary for
            considering a detected peak a real peak.

    """

    for peak_file in peak_files:
        algorithm = utils.split_path(peak_file)[1]
        new_peaks = DetectedXCMSPeaks(peak_file, algorithm, min_pw, min_mz, tol)
        print(new_peaks.get_data()['class'].isin([1,2,3]).sum())
        print(new_peaks.get_data()['class'].isin([5,6,7]).sum())
        print(new_peaks.get_data()['class'].isin([9]).sum())
        print(new_peaks.get_data().shape)
        try:
            peaks.join(new_peaks, tol)
        except NameError:
            peaks = new_peaks
    
    file_path_split = utils.split_path(peak_files[0])
    sample_name = '{}.{}'.format(file_path_split[-1].split('__')[0], f_type)
    sample_path = os.path.join(raw_data_dir, file_path_split[-2], sample_name)
    
    peaks.add_spectra(load_sampledata(sample_path))
    peaks.save_data(out_file)
    peaks.save_dropped(out_file.replace('merged', 'dropped'))


if __name__ == '__main__':
    print('There is nothing here...')