#!/usr/bin/env python3

""" Script for sampling and visualization of detected peaks as PDFs.

"""

import os
import re
import time
import multiprocessing
import numpy as np
import pandas as pd

from sampledata import load_sampledata
from detectedpeaks import DetectedXCMSPeaks, DetectedMergedPeaks


def merge_detected_peaks(files, algorithm, out_file, tol, min_pw, min_mz):
    """ Merge peaks detected by the same algorithms with different parameter
        tuples in the same sample.

    Args:
        files (list of str): Absolute or relative path to detected peaks.
        algorithm (str): Algorithm description for file system.
        out_file (str): Absolute or relative path to the prospective output file.
        raw_data_dir (str): Absolute or relative path to the data directory
            containing all samples.
        tol (int|float): RT window within which peaks are merged.
        min_pw (int|float): Minimum peak width necessary for considering a 
            detected peak a real peak.
        min_mz (int): Minimum number of detected m/z values necessary for
            considering a detected peak a real peak.

    """
    for file in files:
        new_peaks = DetectedXCMSPeaks(file, algorithm, min_pw, min_mz, tol)
        try:
            peaks.join(new_peaks, tol)
        except NameError:
            peaks = new_peaks
    peaks.save_data(out_file)
    peaks.save_statistics(out_file.replace('__merged', '__statistics'))


def visualize_peaks(merged_data, raw_data_dir, train_dir, f_type,
            peak_no, algorithm, sample, cores):
    """ Plot and save peaks chosen for manual annotation as PDFs.

    Args:
        merged_data (str): Absolute or relative path to peaks merged within the
            sample.
        raw_data_dir (str): Absolute or relative path to the data directory
            containing all samples.
        train_dir (str): Name of the directory containing the training samples.
        f_type (str). File type of the training samples.
        peak_no (int): Number of peaks to plot for each sample.
        algorithm (str): Algorithm description for file system.
        sample (str): Name of training samplw without file extension.
        cores (int): Number of available cores.

    """
    sample_path = os.path.join(
        raw_data_dir, train_dir, '{}.{}'.format(sample, f_type)
    )
    sample = load_sampledata(sample_path)
    
    peaks = DetectedMergedPeaks(merged_data, algorithm)
    peaks_to_plot = peaks.sample_peaks(peak_no)

    annot_name = '{}__{}__annotated.csv'.format(algorithm, sample.name)
    annot_file = os.path.join('00_Training', '03_Annotation', annot_name)
    if os.path.exists(annot_file):
        peaks_to_plot = pd.read_csv(annot_file, sep='\t')

    out_dir = os.path.join('00_Training', '02_Visualization', algorithm)
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass
  
    print(
        'Generating {} peaks from sample: {}\t({})' \
            .format(peak_no, sample.name, algorithm)
    )
    plotted_peaks = _plot_peaks(peaks_to_plot, sample, out_dir, max(cores, 1))

    out_name = '{}__{}__visualization_overview.csv'.format(algorithm, sample.name)
    out_path = os.path.join('00_Training', '02_Visualization', out_name)
    plotted_peaks.to_csv(out_path, sep='\t', index=False)


def _plot_peaks(peaks, sample, out_dir, cores):
    peaks['visualization_file'] = None
    processes = []
    
    for idx, peak_vals in peaks.iterrows():
        if ((idx+1) % 10) == 0:
            print('\t\t{:>4} / {}'.format(idx+1, len(peaks)))

        rt = peak_vals[['rtmin', 'rt', 'rtmax']].astype(float).round(1).tolist()
        out_file = '{}__{:.1f}-{:.1f}-{:.1f}.pdf' \
            .format(sample.name, rt[0], rt[1], rt[2])
        out_path = os.path.join(out_dir, out_file)

        # Check if visualization was created before
        if os.path.isfile(out_path):
            peaks.at[idx, 'visualization_file'] = out_path
            continue

        while True:
            processes = [i for i in processes if i.is_alive()]
            if len(processes) >= cores:
                time.sleep(0.5)
            else:
                break

        single_peak_process = multiprocessing.Process(
            target=sample.plot_peak, args=(rt, out_path,)
        )
        single_peak_process.start()
        processes.append(single_peak_process)

        # Update output csv
        peaks.at[idx, 'visualization_file'] = out_path

    # Wait till all processes are finished
    while True:
        processes = [i for i in processes if i.is_alive()]
        if processes:
            time.sleep(0.5)
        else:
            break

    return peaks 


if __name__ == '__main__':
    print('There is nothing here...')
