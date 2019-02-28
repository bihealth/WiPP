#!/usr/bin/env python3

""" Script for aligning peaks across samples based on RI if possible, on RT
otherwise.

"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from scipy import spatial

from sampledata import load_sampledata
from detectedpeaks import DetectedMergedPeaks
import utils


def calc_RI_with_wash(peak_file, out_file, wash_file, alkanes):
    """ Calculate Retention Index based on alkanes found in wash sample.

    Args:
        peak_file (str): Absolute or relative path to the file containing the
            merged peaks.
        out_file (str): Absolute or relative path to the prospective output file.
        wash_file (str): Absolute or relative path to a wash sample file.
        alkanes (list of str): Alkane names used for data generation (e.g. "c10").

    """
    wash_alkanes = _get_alkanes(
        wash_file, normalize=False, alk_no=len(alkanes)
    )
    sample_alkanes = _get_alkanes(
        peak_file, wash_rt=wash_alkanes, alk_no=len(alkanes)
    )
    _calc_RI(peak_file, out_file, sample_alkanes, alkanes)


def _calc_RI(data_path, out_file, alkanes_rt, alkanes_name):
    """ Calculate Retention Index based on alkanes found in wash sample

    Args:
        data_path (str): absolute or relative path to detected peak set
        out_file (str): Path to use for output generation
        alkanes_rt (list of float): Containing the alkanes retention time
        alkanes_name (list of str): Containing the alkanes name (e.g. "c10")

    Returns:
        True
    """
    # Map alkane RT to corresponding RI
    alkanes_RI = [int(i.lstrip('c')) * 100 for i in sorted(alkanes_name)]
    alkanes = {j: alkanes_RI[i] for i,j in enumerate(sorted(alkanes_rt))}

    peaks = DetectedMergedPeaks(data_path, 'Merged')
    peaks.add_RI(alkanes)
    peaks.save_data(out_file)


def _get_alkanes(data_path, wash_rt=pd.DataFrame(), wash_rt_tol=10, normalize=False,
            alk_masses=[71, 85, 99], alk_no=9):    
    if normalize:
        detect_col = 'mz_raw_norm'
    else:
        detect_col = 'mz_raw'

    data = DetectedMergedPeaks(data_path, 'Merged').get_data()

    def _get_mass(spectra, mass):
        int_pattern = '{:.0f}:(\d+),'.format(mass)
        try:
            intensity = re.search(int_pattern, spectra).group(1)
        except AttributeError:
            float_pattern = '{}:(\d+),'.format(mass)
            intensity = re.search(float_pattern, spectra).group(1)
        return int(intensity)

    mz_cols = []
    for a_mass in alk_masses:
        new_mz_col = 'spectra_{}'.format(a_mass)
        data[new_mz_col] = data[detect_col].apply(_get_mass, args=(a_mass,))
        mz_cols.append(new_mz_col)
    data['alk_masses_mean'] = data[mz_cols].mean(axis=1)

    if wash_rt.empty:
        alkanes_idx = _search_wash_alkanes(
            data, mz_cols, alk_no, wash_rt_tol
        )
        return data.iloc[alkanes_idx][['rt', 'mz_raw_norm']].sort_index()
    else:
        alkanes_rt = _search_sample_alkanes(data, wash_rt, wash_rt_tol, alk_no)
        return sorted([i[0] for i in alkanes_rt])


def _search_wash_alkanes(data, mass_cols, alk_no, wash_rt_tol):
    pos_alks = {}
    # Iterate over specific masses
    for col in mass_cols:
        # Iterate over nine highest intensity peaks
        added = 0
        highest_peaks = data.sort_values(by=col, ascending=False)
        for peak_idx, peak_data in highest_peaks.iterrows():
            peak = (peak_data['rt'], peak_idx)
            # Count how often peak is among highest nine intesities
            duplicate = False
            for added_peak in pos_alks:
                if added_peak[0] == peak_data['rt']:
                    break
                elif abs(added_peak[0] - peak_data['rt']) < wash_rt_tol:
                    duplicate = True
                    break
            if not duplicate:
                try:
                    pos_alks[peak].append(added)
                except KeyError:
                    pos_alks[peak] = [added]

                added += 1
                if added == alk_no:
                    break

    alks = sorted(
        pos_alks.items(), key=lambda x: (len(x[1]), 24 - sum(x[1])),
        reverse=True
    )[:alk_no]
    return [i[0][1] for i in alks]


def _search_sample_alkanes(data, wash_alks, wash_rt_tol, alk_no):
    def _get_sim_score(spec_1, spec_2):
        spec_1_str = spec_1['mz_raw_norm']
        spec_2_str = spec_2['mz_raw_norm']

        RT_diff = np.abs(spec_1['rt'] - spec_2['rt'])
        if RT_diff < 1.5:
            RT_pun = 1
        elif RT_diff < 3:
            RT_pun = 0.97
        elif RT_diff < 4:
            RT_pun = 0.95
        elif RT_diff < 5:
            RT_pun = 0.93
        else:
            RT_pun = 0.85

        spec_1 = [int(i.split(':')[1]) for i in spec_1_str.split(',')]
        spec_2 = [int(i.split(':')[1]) for i in spec_2_str.split(',')]
        return (RT_pun * (1 - spatial.distance.cosine(spec_1, spec_2)))

    alkanes = []
    for idx, wash_alk in wash_alks.iterrows():
        wash_alk_rt = wash_alk['rt']

        pos_peaks = data[
            (data['rt'] > wash_alk_rt - wash_rt_tol) \
                & (data['rt'] < wash_alk_rt + wash_rt_tol)
                & (data['rt'] < wash_alk_rt + wash_rt_tol) \
                & ((data['class'] == 2) | (data['class'] == 5))
        ].copy()
        
        if pos_peaks.empty:
            alkanes.append((wash_alk_rt, None))
            continue

        pos_peaks['similarity'] = pos_peaks \
            .apply(_get_sim_score, args=(wash_alk,), axis=1)
        pos_peaks['match_score'] = pos_peaks['similarity'] \
            * pos_peaks['alk_masses_mean']
        pos_peaks.sort_values(by='match_score', ascending=False, inplace=True)
        alkanes.append((pos_peaks.iloc[0]['rt'], pos_peaks.index[0]))
            
    return alkanes


class AlignedPeaks():
    """ A peak set to which other peaks sets are merged to.

    Args (DetectedPeaks): Peaks detected within a certain sample.

    """ 


    def __init__(self, peak_obj):
        self.data, _ = self._get_detected_peak_data(peak_obj)
        self._number_cols = ['rt', 'rtmin', 'rtmax', 'peak_width']
        if 'RI' in self.data.columns:
            self._number_cols.append('RI')


    def save_data(self, out_file, split_mz=False):
        """ Save aligned data to file system.

        Args:
            ut_file (str): Absolute or relative path to the prospective output file.

        """
        if split_mz:
            self._save_data_split(out_file)
        else:
            self._save_data_all(out_file)


    def _save_data_all(self, out_file):
        out_data = self.data
        cols = self._number_cols + ['mz_raw']
        cols.extend(sorted([i for i in out_data.columns if not i in cols]))
        out_data[cols].to_csv(out_file, sep='\t', index=False)


    def _save_data_split(self, out_file):
        out_data = self.data
        cols0 = self._number_cols + ['mz']
        cols0.extend(sorted(
            [i for i in out_data.columns if not i in cols0 and not 'mz_raw' in i]
        ))
        out_data[cols0].to_csv(out_file, sep='\t', index=False)

        cols1 = self._number_cols + ['mz_raw']
        cols1.extend(sorted(
            [i for i in out_data.columns if not i in cols1 and not 'mz' in i]
        ))
        out_data[cols1].to_csv(
            out_file.replace('final.csv', 'final_mz_raw.csv'), sep='\t', index=False
        )


    def round_floats(self, no):
        """ Round all float columns in the peak data.

        Args:
            no (int): Number of decimal to round to.

        """
        self.data[self._number_cols] = self.data[self._number_cols].round(1)


    def _get_detected_peak_data(self, peak_obj):
        sample_data = peak_obj.get_data()
        sample_name = peak_obj.get_sample_name()
        data = sample_data.drop(['parameters', 'class', 'mz_raw_norm'], axis=1)

        mz_col = '{}__mz'.format(sample_name)
        data[mz_col] = sample_data['mz']
        class_col = '{}__class'.format(sample_name)
        data[class_col] = sample_data['class']
        mz_spec_col = '{}__mz_raw'.format(sample_name)
        data[mz_spec_col] = sample_data['mz_raw']

        sample_cols = [mz_col, class_col, mz_spec_col]
        if 'RI' in data.columns:
            RI_col = '{}__RI'.format(sample_name)
            data[RI_col] = sample_data['RI']
            sample_cols.append(RI_col)
            data = data[~data['RI'].isnull()]
            data = data.sort_values('RI').reset_index(drop=True)

        return (data, sample_cols)


    def merge_samples(self, peak_obj, merge_col, tol, min_sim):
        """ Merge another peaks set to the existing one.

        Args:
            peak_obj (DetectedPeaks): Peaks detected within a certain sample.
            merge_col (str): Column name used for merging: either 'rt' or 'RI'.
            tol (int|float): RI (if calculable) or RT window used for merging.
            min_sim (float): Minimum mean similarity score across samples for 
                peaks to be reported.

        """
        new_data, new_cols = self._get_detected_peak_data(peak_obj)
        spectrum_col = [i for i in new_cols if 'mz_raw' in i][0]

        add_peaks = pd.DataFrame()
        merge_peaks = pd.DataFrame()
        for idx, peak in new_data.iterrows():
            match = np.argwhere(
                np.abs(self.data[merge_col] - peak[merge_col]) < tol
            ).flatten()
            # No match
            if match.size == 0:
                add_peaks = add_peaks.append(peak, sort=False)
                continue
            # matches exactly one peak
            elif match.size == 1:
                idx_match = match[0]
            # matches several peaks: merge with the most similar spectra
            else:
                sims = []
                for idx_pos in match:
                    sim = _get_similarity(
                        self.data.loc[idx_pos, 'mz_raw'],
                        peak['mz_raw']
                    )
                    sims.append((sim, idx_pos))
                idx_match = sorted(sims)[0][1]

            peak_sim = _get_similarity(
                self.data.loc[idx_match, 'mz_raw'], peak['mz_raw']
            )
            if peak_sim >= min_sim:
                self.data.loc[idx_match, self._number_cols] += \
                    peak[self._number_cols]
                self.data.loc[idx_match, self._number_cols] /= 2
                self.data.loc[idx_match, 'mz_raw'] = _merge_spectra(
                    self.data.loc[idx_match, 'mz_raw'], peak['mz_raw']
                )
                self.data.loc[idx_match, 'mz'] = _merge_spectra(
                    self.data.loc[idx_match, 'mz'], peak['mz']
                )

                to_merge = peak[new_cols]
                to_merge.name = idx_match
                merge_peaks = merge_peaks.append(to_merge, sort=False)
            else:
                add_peaks = add_peaks.append(peak, sort=False)

        merge_peaks.reset_index(inplace=True)
        dupl_match_idx = merge_peaks.duplicated(subset=['index'], keep=False)
        dupl_match = merge_peaks[dupl_match_idx]
        if not dupl_match.empty:
            for idx_grp, grp_data in dupl_match.groupby('index'):
                sims = []
                for idx_dupl_peak, dupl_peak in grp_data.iterrows():
                    sim = _get_similarity(
                        self.data.loc[idx_grp, 'mz_raw'],
                        dupl_peak[spectrum_col]
                    )
                    sims.append((sim, idx_dupl_peak))
                drop = [i[1] for i in sorted(sims)[:-1]]
                merge_peaks.drop(drop, inplace=True)
        
        self.data = self.data.merge(
            merge_peaks, how='outer', left_index=True, right_on='index'
        ).drop('index',axis=1)
        self.data = self.data.append(add_peaks, sort=False)
        self.data = self.data.sort_values(merge_col).reset_index(drop=True)
        

    def add_ratio_column(self, min_no):
        """ Add in how many samples a certain peak was found as a column.

        Args:
            min_no (int): Minimum number of samples in which a peak has to be
                present to be reported.

        """
        class_cols = [i for i in self.data.columns if i.endswith('__class')]
        ratio_col = 'sample_ratio'
        self.data[ratio_col] = self.data[class_cols] \
            .apply(lambda x: x.dropna().size, axis=1)

        self.data = self.data[
            self.data[ratio_col] >= np.ceil(len(class_cols) * min_no)
        ]
        self.data[ratio_col] = self.data[ratio_col] \
            .apply(lambda x: '{}/{}'.format(x, len(class_cols), axis=1))
        self._number_cols.append(ratio_col)


    def add_similarity_columns(self):
        """ Add the mean inter-sample similarity of each peak as a column.

        """
        spec_cols = [i for i in self.data.columns if i.endswith('__mz_raw')]

        def calc_similarity(peak, ref_col):
            ref_spec = peak[ref_col]
            if not isinstance(ref_spec, str):
                return np.nan
            sims = []
            for col_name, sample_spec in peak.items():
                if isinstance(sample_spec, str) and not col_name == ref_col:
                    sims.append(_get_similarity(ref_spec, sample_spec))
            if sims:
                return np.mean(sims).round(5)
            else:
                return np.nan

        # Calculate similarity: single sample vs. all samples
        for idx, sample_col in enumerate(spec_cols):
            sample_sim_col = '{}__similarity'.format(sample_col.split('__')[0])
            self.data[sample_sim_col] = self.data[spec_cols] \
                .apply(calc_similarity, axis=1, args=(sample_col,))


        # Calculate similarity: mean spec vs. all samples
        sim_col = 'similarity'
        self.data[sim_col] = self.data[['mz_raw'] + spec_cols] \
            .apply(calc_similarity, axis=1, args=('mz_raw',))
        self._number_cols.append(sim_col)


def _merge_spectra(spec1_str, spec2_str, norm=False):
    spec1 = _spec_str_to_s(spec1_str, norm)
    spec2 = _spec_str_to_s(spec2_str, norm)
    
    spec = (spec1 + spec2) / 2
    # If a mass is only present in one spectra, keep this intensity!
    # (instead of its half (treating not found as zero))
    only_spec1 = set(spec1.index).difference(spec2.index)
    only_spec2 = set(spec2.index).difference(spec1.index)
    try:
        spec[only_spec1] = spec1[only_spec1]
        spec[only_spec2] = spec2[only_spec2]
    except:
        import pdb; pdb.set_trace()
    return ','.join(['{}:{:.0f}'.format(i, j) for i, j in spec.items()])


def _spec_str_to_s(spec_str, norm=False):
    spec = pd.Series(
        np.array([int(i.split(':')[1]) for i in spec_str.split(',')]),
        index=[int(i.split(':')[0]) for i in spec_str.split(',')]
    )
    spec = spec.groupby(spec.index).first()
    if norm:
        spec = spec / spec.max() * 999
    return spec


def _get_similarity(spec1_str, spec2_str):
    spec1 = np.array([int(i.split(':')[1]) for i in spec1_str.split(',')])
    spec2 = np.array([int(i.split(':')[1]) for i in spec2_str.split(',')])
    spec1_norm = spec1 / spec1.max() * 999
    spec2_norm = spec2 / spec2.max() * 999
    return (1 - spatial.distance.cosine(spec1_norm, spec2_norm))


def align_across_samples(res_files, out_file, tol, min_no, min_sim, RI):
    """ Align detected peaks across samples to a final peak list.

    Args:
        res_files (list of str): Absolute or relative path to merged peaks
            detected in certain samples.
        out_file (str): Absolute or relative path to the prospective output file.
        tol (int|float): RI (if calculable) or RT window used for alignment.
        min_no (int): Minimum number of samples in which a peak has to be
            present to be reported.
        min_sim (float): Minimum mean similarity score across samples for 
            peaks to be reported.
        RI (bool): True if RI is calculated, False otherwise.

    """
    for res_file in res_files:
        peaks = DetectedMergedPeaks(res_file, 'Merged')
        try:
            if RI:
                final_df.merge_samples(peaks, 'RI', tol, min_sim)
            else:
                final_df.merge_samples(peaks, 'rt', tol, min_sim) 
        except UnboundLocalError:
            final_df = AlignedPeaks(peaks)
    final_df.round_floats(1)
    final_df.add_ratio_column(min_no)

    final_df.add_similarity_columns()

    final_df.save_data(out_file, split_mz=True)


if __name__ == '__main__':
    print('There is nothing here...')