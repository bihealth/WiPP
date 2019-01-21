#!/usr/bin/env python3

""" Script containing the DetectedPeaks class representing the result of a peak
detection algorithm.

This class can be used to merge different peak picking results.

"""

import os
import re
import pandas as pd
import numpy as np

from sampledata import load_sampledata


class DetectedPeaks():
    """ An abstract object containing the picked peaks within a sample by a 
    certain algorithm.

    Args:
        file (str): Absolute or relative path to the peaked peaks csv file.
        algorithm (str): Algorithm used for peak picking.
        min_pw (float): Minimum peak width to consider a peak as real.
        min_mz (int): Minimum number of detected m/z to consider a peak as real.
        rt_tol (float): RT window used for merging.

    """ 

    def __init__(self, file, algorithm, min_pw, min_mz, rt_tol):
        self.file = file
        self.algorithm = algorithm
        self._min_pw = min_pw
        self._min_mz = min_mz
        self._rt_tol = rt_tol

        self.sample_name = os.path.basename(file).split('__')[0]

        self.params = self._parse_params()
        self.data, self.dropped, self.concatenated = self._load_data()
        self.stats = self._load_stats()


    def _load_data(self):
        raise NotImplementedError()


    def get_file(self):
        """ Returns:
            str: the data file from which the peaks were loaded.

        """
        return self.file


    def get_algorithm(self):
        """ Returns:
            str: the algorithm used for peak detection.

        """
        return self.algorithm


    def get_sample_name(self):
        """ Returns:
            str: the name of the sample in which the peaks were detected.

        """
        return self.sample_name


    def get_params(self):
        """ Returns:
            dict: the algorithm parameters used for peak picking.
                key: parameter name, value: parameter value.

        """
        return self.params


    def get_data(self):
        """ Returns:
            pd.DataFrame: the detected peaks.

        """
        return self.data


    def get_dropped(self):
        """ Returns:
            pd.DataFrame: detected peaks that were dropped during processing.

        """
        return self.dropped


    def get_concatenated(self):
        """ Returns:
            pd.DataFrame: detected peaks that were concatenated during processing.

        """
        return self.concatenated


    def get_stats(self):
        """ Returns:
            pd.DataFrame: processing statistics (total, dropped and concatenated
                counts).

        """
        return self.stats


    def get_param_str(self):
        """ Returns:
                str: the algorithm parameters used for peak picking

        """
        if self.params:
            par_str = '_'.join(
                ['{}-{}'.format(i[0], i[1]) for i in sorted(self.params.items())]
            )
            return '{}__{}'.format(self.algorithm, par_str)
        else:
            return self.algorithm


    def get_score(self, scoring):
        """ Calculates a score for the detected peaks.

        Args:
            scoring (dict): Scoring values for each class:
                key: class, values: class values.

        Raises:
            RunTimeError: If peaks are not classified yet.

        Returns:
            int: Calculated score.

        """
        if not 'class' in self.data.columns:
            raise RuntimeError('Peaks need to be classified before scoring')

        score = 0
        for class_idx, class_count in self.data['class'].value_counts().items():
            score += class_count * scoring[class_idx]
        return score


    def save_data(self, out_file):
        """ Save the (modified) peaks to the file system.

        Args:
            out_file (str): Absolute or relative path to the prospective output
                file.

        """
        if self.concatenated.empty:
            out_data = self.data
        else:
            out_data = self.data.merge(
                self.concatenated, how='left', left_index=True, right_index=True,
                suffixes=('', '_1')
            )
        out_data = self._sort_and_idx_reset(out_data, ['rt', 'rtmin', 'rtmax'])
        out_data.to_csv(out_file, sep='\t', index=False)


    def save_dropped(self, out_file):
        """ Save the dropped peaks to the file system.

        Args:
            out_file (str): Absolute or relative path to the prospective output
                file.

        """
        dropped_out = self._sort_and_idx_reset(
            self.dropped, ['rt', 'rtmin', 'rtmax']
        )
        dropped_out.to_csv(out_file, sep='\t', index=False)


    def save_statistics(self, out_file):
        """ Save the processing statistics to the file system.

        Args:
            out_file (str): Absolute or relative path to the prospective output
                file.

        """
        self.stats.to_csv(out_file, sep='\t')


    def _parse_params(self):
        if '_par' in self.file:
            return self._parse_params_from_file()
        else:
            return {}


    def _parse_params_from_file(self):
        params = {}
        for i in range(5):
            par_str = 'par{}'.format(i)
            par_pattern = '_par{}-(\d+\.\d+|\d+)'.format(i)
            try:
                params[par_str] = float(re.findall(par_pattern, self.file)[0])
            except IndexError:
                pass
        return params


    def _load_stats(self):
        stats = pd.Series(
            {'peaks': self.data.shape[0],
            'pw_mean': self.data['peak_width'].mean(),
            'pw_std': self.data['peak_width'].std(),
            'pw_var': self.data['peak_width'].var(),
            'pw_min': self.data['peak_width'].min(),
            'pw_lower_quantil': self.data['peak_width'].quantile(0.25),
            'pw_median': self.data['peak_width'].median(),
            'pw_upper_quantil': self.data['peak_width'].quantile(0.75),
            'pw_max': self.data['peak_width'].max()},
            name=self.get_param_str()
        )
        return stats


    def add_spectra(self, sample):
        """ Get the spectra of each detected peak and add it to the corresponding
        peak.

        Args:
            sample (SampleData): SampleData object corresponding to the peaks. 

        """

        self.data['mz_spectrum'] = self._get_data_spectra(self.data, sample)
        self.data['mz_spectrum_norm'] = self._get_data_spectra_normalized(
            self.data
        )
        if not self.dropped.empty:
            self.dropped['mz_spectrum'] = self._get_data_spectra(
                self.dropped, sample
            )
            self.dropped['mz_spectrum_norm'] = \
                self._get_data_spectra_normalized(self.data)
        if not self.concatenated.empty:
            peak_no = len([i for i in self.concatenated.columns if 'class' in i])
            for idx in range(1, peak_no+1, 1):
                peak_cols = [
                    i for i in self.concatenated.columns if '_{}'.format(idx) in i
                ]
                peak_col_data = self.concatenated[peak_cols]
                peak_col_data.columns = [
                    i.replace('_{}'.format(idx), '') for i in peak_cols
                ]

                self.concatenated.loc[
                    peak_col_data.index, 'mz_spectrum_{}'.format(idx)
                ] = self._get_data_spectra(peak_col_data, sample)

                self.concatenated \
                    .loc[peak_col_data.index, 'mz_spectrum_norm_{}'.format(idx)] \
                        = self._get_data_spectra_normalized(
                            self.concatenated.loc[
                                peak_col_data.index, 'mz_spectrum_{}'.format(idx)
                            ]
                        )

            self.concatenated.columns = sorted(
                self.concatenated.columns, key= lambda x: x.split('_')[-1]
            )


    def _get_data_spectra(self, data, sample):
        if data.empty:
            return []

        mz_min = sample.mz[0]
        spectra = []
        for idx, peak in data.iterrows():
            peak_data = sample.get_processed_peak(
                peak[['rtmin', 'rtmax']], rt_dim=-1,
                ordered=False, normalize=False, flattened=False
            )
            peak_apex = sample._rt_idx(peak['rt']) - sample._rt_idx(peak['rtmin'])
            try:
                peak_class = peak['class']
            # Peaks not classified
            except NameError:
                peak_class = -1
            spectra.append(
                self._get_spectrum_str(peak_data, peak_apex, peak_class, mz_min)
            )
        return spectra


    def _get_data_spectra_normalized(self, data):
        def _normalize_spectra(spectra_raw):
            spectra = np.array([
                int(i.split(':')[1]) for i in spectra_raw.split(',') \
            ])
            if spectra.max() > 0:
                spectra_norm = spectra / spectra.max() * 999
            else:
                spectra_norm = spectra

            first_mz = int(re.match('(\d+)', spectra_raw).group(1))
            out_str = ''
            for idx, intensity in enumerate(spectra_norm):
                out_str += '{:.0f}:{:.0f},'.format(first_mz+idx, intensity)
            return out_str.rstrip(',')

        if isinstance(data, pd.DataFrame):
            return data['mz_spectrum'].apply(_normalize_spectra)
        else:
            return data.apply(_normalize_spectra)


    def _get_spectrum_str(self, peak, apex, pk_class, mz_min):
        scan_no = peak.shape[1]
        if pk_class == 1:
            start, end = (0, int(scan_no * 0.5))
        if pk_class == 2:
            start, end = (int(scan_no * 0.25), int(scan_no * 0.75))
        elif pk_class == 3:
            start, end = (int(scan_no * 0.5), None)
        else:
            start, end = (0, None)

        if (start and apex >= start) and (end and apex <= end):
            # Take masses at predicted apex RT
            spectrum = peak[:, apex]
        else:
            # Take max of each mass in 50% window around apex
            spectrum = peak[:, start:end].max(axis=1)
        
        spectrum_str = ','.join([
            '{:.0f}:{:.0f}'.format(i+mz_min, j) \
                for i, j in enumerate(spectrum)
        ])
        return spectrum_str 


    def add_RI(self, alkanes):
        """ Get the RI of each detected peak and add it to the corresponding
        peak.

        Args:
            alkanes (dict): Alkanes added during data aquisition:
                keys: alkane RT, values: alkane RI. 

        """
        def scale_range(s, min_RI, max_RI):
            s_norm = (s - min_RI[0]) / (max_RI[0] - min_RI[0]) \
                * (max_RI[1] - min_RI[1])
            return (s_norm + min_RI[1])

        old_alk = None
        RI = pd.Series()
        for alk in sorted(alkanes.items()):
            if not old_alk:
                old_alk = alk
                continue

            aff_peaks = self.data[
                (self.data['rt'] >= old_alk[0]) & (self.data['rt'] <= alk[0])
            ]
            new_RI = scale_range(aff_peaks['rt'], old_alk, alk)
            # Old pandas versions require arg sort=False or will throw an warning
            try:
                RI = RI.append(new_RI, sort=False)
            except TypeError:
                RI = RI.append(new_RI)
            old_alk = alk

        self.data['RI'] = RI.dropna().drop_duplicates()


    def join(self, peaks2, tol):
        """ Merge a second DetectedPeaks object to the detected peaks.
        peak.

        Args:
            peaks2 (DetectedPeaks): additional Peaks to be added.
            tol (float): RT window used for merging.

        """
        if 'class' in self.data.columns and 'class' in peaks2.get_data().columns:
            self.data, concatenated, dropped = \
                self._join_classified_data(peaks2, tol)

            self.concatenated = self.concatenated.append(concatenated, sort=False)
            self.dropped = self.dropped.append(dropped, sort=False)
        else:
            self.data, dropped = self._join_unclassified_data(peaks2, tol)
            self.dropped = self.dropped.append(dropped, sort=False)
            self.stats = pd.concat(
                [self.stats, peaks2.get_stats()], axis = 1
            )


    def _sort_and_idx_reset(self, df, cols):
        return df.sort_values(cols).reset_index(drop=True)


    def _join_unclassified_data(self, peaks2, tol):
        new_data = self.data.append(peaks2.get_data(), sort=false)
        match_cols = ['rt', 'rtmin', 'rtmax']
        # Remove duplicates
        duplicates = new_data.duplicated(subset=match_cols, keep=False)
        new_data.loc[duplicates, 'parameters'] = \
            new_data.loc[duplicates, 'parameters'] + peaks2.get_param_str()
        new_data.drop_duplicates(subset=match_cols, inplace=True)

        new_data = self._sort_and_idx_reset(new_data, match_cols)
        new_data, dropped = self._merge_close_RT_vals(new_data, tol)
        return (new_data, dropped)


    def _merge_close_RT_vals(self, data, tol):
        # Identify peaks with overlaps on apex, borderMin and borderMax
        apex_diff = self._get_diff_se_tol(data['rt'], tol)
        rtmin_diff = self._get_diff_se_tol(data['rtmin'], tol)
        rtmax_diff = self._get_diff_se_tol(data['rtmax'], tol)
        to_merge = apex_diff & rtmin_diff & rtmax_diff

        # Merge identified peaks
        drop = []
        for idx in to_merge:
            merged_peak = self._merge_peaks(
                data.loc[[idx[0], idx[1]]], data.index.max() + 1
            )
            data = data.append(merged_peak, ignore_index=False, sort=False)
            drop.extend(list(idx))
        return (data.drop(drop), data.loc[drop])




    def _join_classified_data(self, peaks2, tol):
        new_data_all = self.data.append(peaks2.get_data(), sort=False)
        match_cols = ['rt', 'rtmin', 'rtmax']
        # Save and remove peaks classified as noise
        fp = new_data_all['class'] == 9
        new_data = new_data_all[~fp]
        # Remove duplicates
        duplicates = new_data.duplicated(subset=match_cols, keep=False)
        new_data.loc[duplicates]['parameters'] += peaks2.get_param_str()
        if duplicates.any():
            print(
                'Possibly problematic: duplicates while merging\n{}\tand\t{}' \
                    .format(self.get_sample_name(), peaks2.get_sample_name())
            )
        new_data = new_data.drop_duplicates(subset=match_cols)
        # Actual merging, concatenating and dropping (see function)
        new_data = self._sort_and_idx_reset(new_data, match_cols)
        new_data, concatenated, dropped = self._merging_classified(new_data, tol)
        # Add peaks classified as boise to dropped ones
        dropped = dropped.append(new_data_all[fp], sort=False)
        dropped['reason'].fillna('noise', inplace=True)
        return (new_data, concatenated, dropped)
       

    def _merge_peaks(self, df, idx):
        merged = df[['rt', 'rtmin', 'rtmax', 'peak_width']].mean()
        merged['parameters'] = ';'.join(sorted(
            set(df.iloc[0]['parameters'].split(';') \
                + df.iloc[1]['parameters'].split(';')
            )
        ))
        merged['mz'] = self._merge_mz_strings(df['mz'])
        if 'class' in df.columns:
            merged['class'] = df.iloc[0]['class']
        merged.name = idx
        return merged


    def _concat_peaks(self, df):
        peak1 = df.iloc[0]
        peak2 = df.iloc[1]
        if peak2['class'] in [1, 2, 3] and peak1['class'] == 5:
            drop = peak1.name
            peak1.name = peak2.name
            concat = peak1
        else:
            drop = peak2.name
            peak2.name = peak1.name
            concat = peak2
        return (drop, concat)
        

    def _merging_classified(self, data, tol):
        # Merge peaks with overlapping RT apex
        apex_diff = self._get_diff_se_tol(data['rt'], tol)
        data1, add1, dropped1 = self._merge_peak_overlap(data, apex_diff)
        # Merge peaks with overlapping RT borders
        rtmin_diff = self._get_diff_se_tol(data1['rtmin'], tol)
        rtmax_diff = self._get_diff_se_tol(data1['rtmax'], tol)
        border_diff = rtmin_diff & rtmax_diff
        data2, add2, dropped2 = self._merge_peak_overlap(data1, border_diff)
        # Merge peaks enclosing other peaks
        enclosing = self._get_enclosing_peaks(data2)
        data3, add3, dropped3 = self._merge_peak_overlap(data2, enclosing)
        # Join data dropped and added in all three merging steps
        dropped_all = dropped1.append([dropped2, dropped3])
        add_all = add1.append([add2, add3])
        concat_dupl = add_all.index.duplicated()
        if concat_dupl.any():
            peak_rows = len([i for i in self.data.columns if 'rt_' in i])
            add_all = add_all[~concat_dupl].merge(
                add_all[concat_dupl], how='left', left_index=True,
                right_index=True,
                suffixes=('_{}'.format(peak_rows+1), '_{}'.format(peak_rows+2))
            )
        else:
            add_all.columns = ['{}_1'.format(i) for i in add_all.columns]
        return (data3, add_all, dropped_all)


    def _merge_peak_overlap(self, data, overlap_idx):
        # Merge peaks depending on classes
        drop_dict = {}
        add_df = pd.DataFrame()
        for idx in overlap_idx:
            if -1 in idx:
                continue

            peak_1 = data.loc[idx[0]]
            peak_2 = data.loc[idx[1]]

            # Keep merged peaks
            if peak_1['class'] == 5 or peak_2['class'] == 5:
                drop, concat = self._concat_peaks(data.loc[[idx[0], idx[1]]])
                drop_dict[drop] = 'concatenated'
                add_df = add_df.append(concat, sort=False)
            # Merge same peaks (if not classified as merged)
            elif peak_1['class'] == peak_2['class']:
                drop_dict[peak_1.name] = 'merged'
                drop_dict[peak_2.name] = 'merged'
                data = data.append(
                    self._merge_peaks(
                        data.loc[[idx[0], idx[1]]], data.index.max() + 1
                    ), sort=False
                )   
            # Keep real compound related peaks
            elif peak_1['class'] == 2:
                drop_dict[peak_2.name] = 'other_2'
            elif peak_2['class'] == 2:
                drop_dict[peak_1.name] = 'other_2'
            # One peaks apex is shifted, other one's "more bad": keep shifted one
            elif peak_1['class'] in [1, 3] and peak_2['class'] not in [1, 3]:
                drop_dict[peak_2.name] = 'other_1,3'
            elif peak_2['class'] in [1, 3] and peak_1['class'] not in [1, 3]:
                drop_dict[peak_1.name] = 'other_1,3'
            # Concatenate if two peaks are calssified as too wide and two narrow
            elif peak_1['class'] in [6, 7] and peak_2['class'] in [6, 7]:
                drop, concat = self._concat_peaks(data.loc[[idx[0], idx[1]]])
                drop_dict[drop] = 'concatenated'
                add_df = add_df.append(concat, sort=False)
            elif peak_1['class'] in [1, 3] and peak_2['class'] in [1, 3]:
                drop, concat = self._concat_peaks(data.loc[[idx[0], idx[1]]])
                drop_dict[drop] = 'concatenated'
                add_df = add_df.append(concat, sort=False)
            else:
                print(
                    'No rule defined for overlap of classes: {} vs. {}' \
                        .format(peak_1['class'], peak_2['class'] )
                )
        dropped_raw = pd.concat(
            [data.loc[drop_dict.keys()], pd.Series(drop_dict,name='reason')],
            axis=1
        )
        dropped = dropped_raw[
            ~dropped_raw['reason'].isin(['concatenated', 'merged'])
        ]
        if not add_df.empty:
            add_df.drop_duplicates(inplace=True)
            for dup_idx in add_df.index[add_df.index.duplicated()]:
                dup_merge = self._merge_peaks(add_df.loc[dup_idx], dup_idx)
                add_df.drop(dup_idx, inplace=True)
                add_df.append(dup_merge, sort=False)
        return (data.drop(drop_dict.keys()), add_df, dropped)


    def _merge_mz_strings(self, s):
        mz = pd.DataFrame(
            {'mz0': {float(i.split(':')[0]): float(i.split(':')[1]) \
                for i in s.iloc[0].split(',')},
            'mz1': {float(i.split(':')[0]): float(i.split(':')[1]) \
                for i in s.iloc[1].split(',')}}
        )
        mz_str = ','.join([
            '{:.0f}:{:.0f}'.format(i[0], i[1]) \
                for i in mz.sort_index().mean(axis=1).items()
        ])
        return mz_str


    def _get_diff_se_tol(self, col, tol):
        col_sorted = col.sort_values()
        diff = np.abs((col_sorted.values[1:] - col_sorted.values[:-1]).round(1)) \
            <= tol
        diff = np.insert(diff, 0, False)
        overlap_df = pd.DataFrame(
            {'overlap': diff, 'with': [-1] + col_sorted.index.tolist()[:-1]},
            index=col_sorted.index
        )
        overlap = [
            (i, j['with']) \
                for i, j in overlap_df[overlap_df['overlap']].iterrows()
        ]
        return set(overlap)


    def _get_enclosing_peaks(self, data):
        new_data = data.sort_values(by='rtmin')
        encl = new_data['rtmax'].values[:-1] - new_data['rtmax'].values[1:] > 0
        encl = np.insert(encl, -1, False)
        enclosing_df = pd.DataFrame(
            {'outer': encl, 'inner': new_data.index.tolist()[1:] + [-1]},
            index=new_data.index
        )
        enclosing = [
            (i, j['inner']) \
                for i, j in enclosing_df[enclosing_df['outer']].iterrows()
        ]
        return enclosing


    def sample_peaks(self, no, sections=10):
        """  Randomly sample a certain number of peaks from the detected ones

        Args:
            no (int): Total number of peaks to sample.
            sections (int): Split the peaks based on RT into sections containing
                a nearly equal number of peaks. Randomly sample from each section.
                (Default: 10).

        Returns:
            pd.DataFrame: Sampled peaks.

        """
        if self.data.shape[0] <= no:
            return self.data

        if no < sections:
            sections = no

        peaks_split = np.array_split(self.data, sections)
        peaks_per_interval = int(no / sections)
        for idx in range(sections):
            rdm_peak_idx = np.random.choice(
                peaks_split[idx].index, peaks_per_interval, replace=False
            )
            new_rdm_peaks = peaks_split[idx].loc[rdm_peak_idx]
            try:
                rdm_peaks = pd.concat(
                    [rdm_peaks, new_rdm_peaks], ignore_index=True
                )
            except NameError:
                rdm_peaks = new_rdm_peaks
        rdm_peaks.reset_index(drop=True, inplace=True)
        return rdm_peaks


    def classify_peaks(self, clf, rt_dim, sample_file):
        """ Classify the detected peaks

        Args:
            clf (sklearn.SVM|sklearn.RF): Trained scikit-learn classifier.
            rt_dim (int): Number of data points in x direction (RT) to
                interpolation to.
            sample_file (str): Absolute or relative path to the samples raw data.

        """
        if 'class' in self.data.columns:
            print('Peaks already classified')
            return

        sample = load_sampledata(sample_file)

        to_classify = []
        for p_idx, peak in self.data.iterrows():
            to_classify.append(
                sample.get_processed_peak(peak[['rtmin', 'rtmax']], rt_dim)
            )
        predicted = clf.predict(to_classify)
        self.data['class'] = predicted


class DetectedXCMSPeaks(DetectedPeaks):
    """ An object containing the peaks picked by an XCMS algorithm.

    Args:
        file (str): Absolute or relative path to the peaked peaks csv file.
        algorithm (str): Algorithm used for peak picking.
        min_pw (float): Minimum peak width to consider a peak as real.
        min_mz (int): Minimum number of detected m/z to consider a peak as real.
        rt_tol (float): RT window used for merging.

    """ 

    def __init__(self, file, algorithm, min_pw, min_mz, rt_tol):
        super().__init__(file, algorithm, min_pw, min_mz, rt_tol)


    def _load_data(self):
        data = pd.read_csv(self.file, sep='\t')
        # Round peaks to 1 decimal and drop duplicates
        data[['rt', 'rtmin', 'rtmax']] = \
            data[['rt', 'rtmin', 'rtmax']].round(1)
        dropped = data[data.duplicated(subset=['rt', 'rtmin', 'rtmax'])]
        data.drop_duplicates(subset=['rt', 'rtmin', 'rtmax'], inplace=True)
        # Drop peaks where the apex is placed on the left/right border
        apex_border = (data['rt'] == data['rtmin']) \
            | (data['rt'] == data['rtmax'])
        dropped = dropped.append(data[apex_border], sort=False)
        data = data[~apex_border]
        # Add peak width column
        data['peak_width'] = data['rtmax'] - data['rtmin']
        # Drop peaks with narrower than predefined minimal peak width
        too_narrow = data['peak_width'] < self._min_pw 
        dropped = dropped.append(data[too_narrow], sort=False)
        data = data[~too_narrow]
        # Add parameter columns
        data['parameters'] = self.get_param_str()
        # Merge nearly duplicated peaks
        data, dupl = self._merge_close_RT_vals(data, self._rt_tol)
        dropped.append(dupl, sort=False)
        # Drop peaks with less than predefined number of detected masses
        too_few_mz = data['mz'].str.count(',') < self._min_mz - 1 
        dropped = dropped.append(data[too_few_mz], sort=False)
        data = data[~too_few_mz]
        dropped['reason'] = 'internal'
        return (data, dropped, pd.DataFrame())

    
class DetectedChromaTOFPeaks():
    """ An object containing the peaks picked by ChromaTOF.

    Args:
        file (str): Absolute or relative path to the peaked peaks csv file.
        algorithm (str): Algorithm used for peak picking.
        min_pw (float): Minimum peak width to consider a peak as real.
        min_mz (int): Minimum number of detected m/z to consider a peak as real.
        rt_tol (float): RT window used for merging.

    """ 

    def __init__(self, file, algorithm, min_pw, min_mz, rt_tol):
        super().__init__(file, algorithm, min_pw, min_mz, rt_tol)


    def _load_data(self):
        data = pd.read_csv(
            self.file, sep=',', doublequote=False,
            usecols=['R.T. (s)', 'IntegrationBegin', 'IntegrationEnd', 'Spectra']
        )
        data.rename(
            {'R.T. (s)': 'rt', 'IntegrationBegin': 'rtmin', 
            'IntegrationEnd': 'rtmax', 'Spectra': 'mz'},
            axis=1, inplace=True
        )
        data['rtmin'] = peak_data['rtmin'] \
            .apply(lambda x: float(x.split(',')[0]))
        data['rtmax'] = peak_data['rtmax'] \
            .apply(lambda x: float(x.split(',')[0]))
        data['mz'] = peak_data['mz'].str.replace(' ', ',')

        # Round peaks to 1 decimal and drop duplicates
        data[['rt', 'rtmin', 'rtmax']] = \
            data[['rt', 'rtmin', 'rtmax']].round(1)
        data.drop_duplicates(subset=['rt', 'rtmin', 'rtmax'], inplace=True)
        # Drop peaks where the apex is placed on the left/right border
        data = data[~
            ((data['rt'] == data['rtmin']) | (data['rt'] == data['rtmax']))
        ]
        # Add peak width column
        data['peak_width'] = data['rtmax'] - data['rtmin']
        # Drop peaks with narrower than predefined minimal peak width
        data = data[data['peak_width'] >= self._min_pw]
        # Add parameter columns
        data['parameters'] = self.get_param_str()
        # Merge nearly duplicated peaks
        data, dropped = self._merge_close_RT_vals(data, self._rt_tol)
        dropped['reason'] = 'internal'
        # Drop peaks with less than predefined number of detected masses
        data = data[data['mz'].str.count(',') >= self._min_mz - 1]
        return (data, dropped, pd.DataFrame())


class DetectedMergedPeaks(DetectedPeaks):
    """ An object containing picked and merged peaks by either different
    algorithms or different parameters.

    Args:
        file (str): Absolute or relative path to the peaked peaks csv file.
        algorithm (str): Algorithms used for peak picking. (Default: 'Merged')

    """ 
    def __init__(self, file, algorithm='Merged'):
        super().__init__(file, algorithm, 0, 1, 0)


    def _load_data(self):
        data_raw = pd.read_csv(self.file, sep='\t')
        if any([i.endswith('_1') for i in data_raw.columns]):
            base_cols = [i for i in data_raw.columns if not re.search('_\d+$', i)]
            data = data_raw[base_cols]
            concat = data_raw[[i for i in data_raw.columns if not i in base_cols]]
        else:
            data = data_raw
            concat = pd.DataFrame()
        return (data, pd.DataFrame(), concat)


if __name__ == '__main__':
    print('There is nothing here...')