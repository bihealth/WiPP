#!/usr/bin/env python3

""" Script containing the SampleData class representing a sample.

This class can be used to do calculations or plotting on the raw sample data.

"""

import os
import pickle
import math
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter
import scipy.sparse as sparse
import scipy.linalg as LA
import pyopenms
from netCDF4 import Dataset


class SampleData():
    """ An object containing the processed data of a sample.

    Args:
        file (str): Absolute or relative path to the samples raw data.

    """ 

    def __init__(self, file):
        self._file = file
        self._data_format = file.split('.')[-1]

        self._group = file.split('/')[-2]
        self.name = os.path.basename(file) \
            .replace(self._data_format, '').rstrip('.')
        
        if self._data_format.lower() == 'cdf':
            self._read_cdf()
        elif self._data_format.lower() == 'mzml':
            self._read_mzML()
        elif self._data_format.lower() == 'mzdata':
            self._read_mzData()
        else:
            raise RuntimeError(
                'Unsupported input format: {}.'.format(self._data_format)
            )

        data_bl_corrected = np.apply_along_axis(
            self._global_baseline_correction, axis=1, arr=self.data_raw
        )

        # Smooth data
        window_size = self.scans_per_sec*2 + (self.scans_per_sec*2)%2 - 1
        data_smoothed = np.apply_along_axis(
            savgol_filter, axis=1, arr=data_bl_corrected,
            window_length=window_size, polyorder=3, mode='nearest'
        )
        # set values smaller than 0 to 0
        data_smoothed[data_smoothed < 0] = 0

        self.data_processed = data_smoothed
        self._dump_pickle()


    def _read_cdf(self):
        data_cdf = Dataset(self._file, 'r')

        self.rt = np.sort(np.array(data_cdf.variables['scan_acquisition_time']))
        if pd.Series(self.rt).value_counts().unique().size != 1:
            raise RuntimeError('Inconsistent number of scans.')

        # self.mz = np.sort(np.unique(data_cdf.variables['mass_values']))
        self.mz = np.arange(
            np.min(data_cdf.variables['mass_values']), 
            np.max(data_cdf.variables['mass_values']) + 1,
            1
        )

        mz_sorted = np.sort(
            np.unique(data_cdf.variables['mass_values']).round()
        )

        # All masses measured in every scan
        try:
            all_measured = (self.mz == mz_sorted).all()
        except AttributeError:
            all_measured = (self.mz == mz_sorted)
        if all_measured:
            intensity = data_cdf.variables['intensity_values']
            if len(intensity) / len(self.mz) != len(self.rt):
                raise RuntimeError('Dimension Error: .CDF data is corrupted.')
            self.data_raw = np.column_stack(np.split(intensity, len(self.rt)))
        # Only selected masses measured -> fill unmeasured masses with zeros
        else:
            scan_end_idx = np.argwhere(
                data_cdf.variables['mass_values'][1:] \
                    - data_cdf.variables['mass_values'][:-1] < 0
            ).flatten() +1

            # Check if dimensions fit
            if scan_end_idx.size + 1 != self.rt.size:
                raise RuntimeError(
                    'Dimension Error: netCDF file data is corrupt.'
                )

            scan_ints = np.split(
                data_cdf.variables['intensity_values'], scan_end_idx
            )
            scan_masses = np.split(
                data_cdf.variables['mass_values'], scan_end_idx
            )

            int_data_raw = np.stack((scan_ints, scan_masses), axis=1)

            def fill_mz(arr_in):
                new_scan = np.zeros(self.mz.size)
                new_scan[np.isin(self.mz, arr_in[1].round())] = arr_in[0]
                return new_scan

            self.data_raw = np.apply_along_axis(
                fill_mz , axis=1, arr=int_data_raw
            ).T
            # self.data_raw = np.column_stack(np.split(intensities, self.rt.size))

        scan_duration = np.unique(data_cdf.variables['scan_duration'])
        if len(scan_duration) != 1:
            raise RuntimeError('Inconsistent number of scans per second.')
        elif scan_duration[0] < 0:
            scan_rt, rt_count = np.unique(np.floor(self.rt), return_counts=True)
            self.scans_per_sec = rt_count.mean().round()
        else:
            self.scans_per_sec = 1 / scan_duration[0]
        

    def _read_mzML(self):
        self._read_openMS_file(pyopenms.MzMLFile())


    def _read_mzData(self):
        self._read_openMS_file(pyopenms.mzDataFile())


    def _read_openMS_file(self, file):               
        exp = pyopenms.MSExperiment()
        file.load(self._file, exp)
        file.store(self._file, exp)

        def bin_highres(df_in):
            df = df_in.groupby(df_in['mz'].round()).sum()
            I = df['I'].loc[self.mz].as_matrix()
            I[np.isnan(I)] = 0
            return I

        # TODO: change hardcoded min/max masses to be either taken from config 
        # or automatically derived from the data
        self.mz = np.arange(70, 601, 1)
        rt = []
        intensities = []
        spectra = exp.getSpectra()
     
        print('Binning:')
        for idx, spectrum in enumerate(spectra):
            if ((idx + 1) % 100) == 0:
                print('\t{:>4} / {}'.format(idx + 1, len(spectra)))

            rt.append(spectrum.getRT())
            spectrum_df = pd.DataFrame(
                list(spectrum.get_peaks()), index=['mz', 'I']
            ).T
            intensities.append(bin_highres(spectrum_df))
        self.data_raw = np.array(intensities).T
        self.rt = np.array(rt)
        self.scans_per_sec = pd.Series(self.rt.round()) \
            .value_counts().value_counts().index[0]


    def _dump_pickle(self, out_name=None):
        if not out_name:
            out_name = self._file.replace(self._data_format, 'pkl')
        with open(out_name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        

    def _global_baseline_correction(self, y, lam=1e5, p=0.01, n=10):
        m = len(y)
        D0 = sparse.eye(m)
        D1 = sparse.diags([np.ones(m - 1) * -2], [-1])
        D2 = sparse.diags([np.ones(m - 2) * 1], [-2])
        D = D0 + D2 + D1
        w = np.ones(m)
        for i in range(n):
            W = sparse.diags([w], [0])  
            C = W + lam * D.dot(D.transpose())
            z = sparse.linalg.spsolve(C, w * y)
            w = p * (y > z) + (1 - p) * (y < z)

        # Set negative values to zero
        bl_corrected = y - z 
        bl_corrected[bl_corrected < 0] = 0
       
        return bl_corrected 


    def _mz_idx(self, mz):
        # Convert actual m/z to index for accessing data
        if not isinstance(mz, (int, float, np.int64, np.float64)):
            raise TypeError('unsupported m/z type: {}'.format(type(mz)))
        elif (self.mz[0] > mz) or (mz > self.mz[-1]):
            raise IndexError('m/z {} not recorded'.format(mz))
        else:
            return int(mz - self.mz[0])


    def _rt_idx(self, rt):
        # Convert actual retention time to index for accessing data
        if not isinstance(rt, (int, float, np.int64, np.float64)):
            raise TypeError('unsupported RT type: {}'.format(type(rt)))
        elif rt < self.rt[0]:
            rt = self.rt[0]
        elif rt > self.rt[-1]:
            rt = self.rt[-1]
        return np.argmin(np.abs((self.rt - rt)))


    def _get_rt_range(self, rt_range):
        if isinstance(rt_range, list):
            rt_min = self._rt_idx(rt_range[0])
            rt_max = self._rt_idx(rt_range[-1])
        elif isinstance(rt_range, pd.Series):
            rt_min = self._rt_idx(rt_range.iat[0])
            rt_max = self._rt_idx(rt_range.iat[-1])
        else:
            raise TypeError(
                'Unsupported RT-range type: {}\nSupported: list | pd.Series' \
                    .format(type(rt_range))
            )
        return (rt_min, rt_max+1)


    def get_raw_data(self, rt_range):
        rt_min, rt_max = self._get_rt_range(rt_range)
        return self.data_raw[:, rt_min:rt_max]


    def get_processed_data(self, rt_range):
        rt_min, rt_max = self._get_rt_range(rt_range)
        return self.data_processed[:, rt_min:rt_max]


    def get_processed_peak(self, rt_range, rt_dim=-1,
            ordered=True, normalize=True, flattened=True, bl_naive=True):
        """ Process peak raw data to classifier fitting format.

        Args:
            rt_range (list of flaots): Peak min RT and peak max RT.
            rt_dim (int): Number of data points in x direction (RT) to
                interpolation to. If negative, no interpolation is done.
                (Default: -1).
            ordered (bool): Return peak data sorted by intensity (mass trace 
                with highest intensity first), sorted by m/z otherwise.
                (Default: True).
            flattened (bool): Return array flattened, as matrix otherwise.
                (Default: True).
            bl_naive (bool): Remove minimum of each spectra, not otherwise.
                (Default: True).

        Returns:
            np.array: processed peak data within given RT window.
        """

        peak = self.get_processed_data(rt_range)

        # Interpolate in RT direction
        if rt_dim > 0:
            def _interpolate_EIC(x, new_x):
                return np.interp(new_x, np.arange(x.size), x)
            peak = np.apply_along_axis(
                _interpolate_EIC, axis=1, arr=peak,
                new_x=np.linspace(0, peak.shape[1], num=rt_dim)
            )
        
        # Remove local baseline of each EIC
        if bl_naive:
            peak = np.apply_along_axis(
                lambda x: x - x.min(), axis=1, arr=peak
            )
                
        # Set masses without sign change to zero
        # (dropping baseline/ tailing/fronting)
        def _check_bl_sign(x):
            u = np.unique(np.sign(x[1:] - x[:-1]))
            if len(u) == 1 or (0 in u and len(u) == 2):
                return True
            return False

        no_sign_change = np.apply_along_axis(_check_bl_sign, axis=1, arr=peak)
        # Skip this step if less than 5 masses are remaining otherwise
        if np.where(no_sign_change == False)[0].size > 5:
            peak[np.argwhere(no_sign_change == True).flatten()] = 0

        # Normalize linear to [0, 1]
        if normalize:
            peak = np.divide(peak, peak.max())

        if ordered:
            mz_max = np.amax(peak, axis=1)
            mz_max_idx = np.argsort(mz_max)
            peak = np.flip(peak[mz_max_idx], axis=0)

        if flattened:
            return peak.flatten()
        else:
            return peak


    def get_AUC(self, rt, mz, rawdata=False):
        """ Get the Area Under the Curve (AUC) of a certain mass in a certain RT
            window.

        Args:
            rt (list of flaots): Peak min RT and peak max RT.
            mz (int): m/z to consider.
            rawdata (bool): If True: calculate the AUC from the raw data, 
                otherwise calculate the AUC from the processed data.
                (Default: False).

        Returns:
            int: calculated AUC.
        """
        if rawdata:
            data = self.data_raw
        else:
            data = self.data_processed
        AUC = data[self._mz_idx(mz), self._rt_idx(rt[0]):self._rt_idx(rt[-1])] \
            .sum().round()
        return AUC


    def get_maxI(self, rt, mz=None, rawdata=False):
        """ Get the maximum intensity in a certain RT window.

        Args:
            rt (list of flaots): Peak min RT and peak max RT.
            mz (int): m/z to consider. If not give, get the maximum over all m/z.
                (Default: None)
            rawdata (bool): If True: calculate the AUC from the raw data, 
                otherwise calculate the AUC from the processed data.
                (Default: False).

        Returns:
            int: maximum intensity.
        """
        if rawdata:
            data = self.data_raw
        else:
            data = self.data_processed

        if mz:
            maxI = data[
                self._mz_idx(mz), self._rt_idx(rt[0]):self._rt_idx(rt[-1])
            ].max()
        else:
            maxI = data[:, self._rt_idx(rt[0]):self._rt_idx(rt[-1])].max()
        return maxI


    def plot_peak(self, rt, out_name, s_borders=2):
        """ Plot a peak in the predefined 4 subplot schema:
        The first subplot shows the raw data as a surface plot.
        The second subplot shows the data baseline corrected as a surface plot.
        The third subplot shows the data baseline corrected as a contour plot.
        The fourth subplot shows the data processed for classification as a 
            surface plot

        Args:
            rt (list of flaots): Peak min RT, peak apex RT and peak max RT.
            out_name (str): Absolute or relative path to the prospective output
                file.
            s_borders (int/float): Additional RT to add to both peak borders in 
                plot 1,2 and 3. (Default: 2).

        """
        # Get x value: Retention time
        rt_idx_min, rt_idx_max = self._get_rt_range(rt)
        rt_extend = [rt[0]-s_borders, rt[-1]+s_borders]
        rt_extend_min_idx, rt_extend_max_idx = self._get_rt_range(rt_extend)
        x_vals = self.rt[rt_extend_min_idx:rt_extend_max_idx]

        # Get y value: m/z
        y_vals = self.mz[:]

        # Get z value: Intensity
        # Raws
        z_vals_raw = self.get_raw_data(rt_extend)
        # Smoothed and baseline correct (each EIC / mass trace)
        z_vals_processed = self.get_processed_data(rt_extend)
        # Smoothed, baseline corrected and only peak width
        z_vals_processed_peak = self.get_processed_data(rt)

        # Get RT for plotting borders and apex
        add_scans_left = rt_idx_min - rt_extend_min_idx
        add_scans_right = rt_extend_max_idx - rt_idx_max
        rt_apex_scan = self._rt_idx(rt[1]) - rt_extend_min_idx

        # Interpolate data if more than 80 scans to safe time and disk space
        max_x_size = 80
        if x_vals.size > max_x_size:
            int_factor = max_x_size / x_vals.size
            x_vals_interpol = np.linspace(x_vals[0], x_vals[-1], num=max_x_size)

            def _interpolate_EIC_plotting(y, new_x, old_x):
                return np.interp(new_x, old_x, y)

            z_vals_raw = np.apply_along_axis(
                _interpolate_EIC_plotting,
                axis=1, arr=z_vals_raw, new_x=x_vals_interpol, old_x=x_vals
            )
            z_vals_processed = np.apply_along_axis(
                _interpolate_EIC_plotting,
                axis=1, arr=z_vals_processed, new_x=x_vals_interpol, old_x=x_vals
            )
            x_vals = x_vals_interpol
            add_scans_left = int(round(add_scans_left * int_factor))
            add_scans_right = int(round(add_scans_right * int_factor))
            rt_apex_scan = int(round(rt_apex_scan * int_factor))

        min_int = 10
        z_vals_raw = np.log10(np.clip(z_vals_raw, min_int, z_vals_raw.max()))
        z_vals_processed = np.apply_along_axis(
            lambda x: x - x.min(), axis=1, arr=z_vals_processed
        )
        z_vals_processed = np.log10(
            np.clip(z_vals_processed, min_int, z_vals_processed.max())
        )

        # Baseline correct and interpolated
        interpolation_no = max(20, self.scans_per_sec * 4)
        z_vals_processed_interpol = self.get_processed_peak(
            rt, rt_dim=interpolation_no, ordered=False, flattened=False
        )

        # Create data dict for plotting borders and apex
        if add_scans_right == 0:
            rt_scans = {
                add_scans_left: ':',
                rt_apex_scan: '-',
                x_vals.size-1: ':'
            }
        else:
            rt_scans = {
                add_scans_left: ':',
                rt_apex_scan: '-',
                -add_scans_right: ':'
            }

        rt_scans_interpolated = {
            0: ':',
            math.floor(
                (self._rt_idx(rt[1]) - self._rt_idx(rt[0])) / \
                    z_vals_processed_peak.shape[1] * interpolation_no
            ): '-',
            -1: ':',
        }

        fig = plt.figure()
        if len(rt) == 3:
            plt.suptitle(
                'Detected peak @ Rt = {:.2f} ({:.2f}–{:.2f}) [s]' \
                    .format(rt[1], rt[0], rt[2]),
                fontsize=16
            )

        # Fix perspective angels
        angles = (100, 30)

        # Get max intensity
        max_I = self.get_maxI(rt, rawdata=True)

        # Add surface plot: raw_data
        self._plot_surface(
            fig, x_vals, y_vals, z_vals_raw, rt_scans, angles, max_I, n_plot=1
        )
        # Add surface plot: baseline corrected
        self._plot_surface(
            fig, x_vals, y_vals, z_vals_processed, rt_scans, angles, n_plot=2
        )
        # Add contour plot: baseline corrected
        self._plot_contour(
            fig, x_vals, y_vals, z_vals_processed, rt_scans, n_plot=3
        )
        # Add surface plot: baseline corrected and interpolated
        self._plot_surface(
            fig, np.arange(interpolation_no), y_vals, z_vals_processed_interpol,
            rt_scans_interpolated, angles, n_plot=4
        )
        self._save_plot(fig, out_name)


    def _plot_contour(self, fig, x, y, z, rt, flip=True, n_plot=1):
        # Create right subplot axis
        ax = fig.add_subplot(2, 2, n_plot)
        ax.set_title('Contour: baseline corrected')

        z[z <= 1] = np.nan
        z = np.ma.masked_invalid(z)

        # Lines for borders and apex
        borders = [i for i, j in rt.items() if j == ':']
        # Apex overlaps with one border
        if len(borders) == 1:
            borders = [i for i in rt.keys()]
        if -1 in np.sign(borders):
            left_border = max(borders)
            right_border = min(borders)
        else:
            left_border = min(borders)
            right_border = max(borders)

        cp = plt.contourf(
            x, y, z,
            100, # number of colors
            vmin=min(3, z[:,left_border:right_border].min()),
            vmax=z[:,left_border:right_border].max(),
            cmap=plt.get_cmap('viridis_r')
        )

        label_fontsize = 16
        plt.colorbar(cp, extend='max') \
            .set_label(label='Intensity [log10]', size=label_fontsize)
        ax.set_xlabel('RT [s]', fontsize=label_fontsize)

        plt.xticks(
            x[::int(x.size/10)], [round(i, 1) for i in x[::int(x.size/10)]]
        )
        ax.set_ylabel('m/z', fontsize=label_fontsize)
        ax.set_yticks(np.arange(50, y.max(), 25))
        ax.tick_params(labelsize=14)

        # Add borders and apex
        lw = 1
        lc = 'red'
        mzs = [self.mz[0], self.mz[-1]]        
        for scan, ls in rt.items():
            plt.plot(
                [x[scan], x[scan]], mzs, color=lc, linestyle=ls, linewidth=lw
            )

        # Highlight highest points per mass
        if right_border == -1:
            right_border = x.size -1

        mz_max_scans = np.argmax(z[:,left_border:right_border-1], axis=1) \
            + left_border
        mz_max = pd.DataFrame([mz_max_scans, z.max(axis=1).data]).T
        mz_max = mz_max[
            (mz_max[0] > left_border) & (mz_max[0] < z.shape[1] + right_border)
        ]

        sorted_data = mz_max.sort_values(by=1, ascending=False)[:25]
        for mz, mz_data in sorted_data.iterrows():
            plt.plot(x[int(mz_data[0])], mz+y.min(), 'b.')

        # Invert axis -> low masses at high y, high masses at low y
        if flip:
            ax.invert_yaxis()

        # Border lines for segments
        segment_x = np.percentile(
            np.linspace(x[left_border], x[right_border], z.shape[1]),
            [10, 37, 63, 90]
        )
        for seg_x in segment_x:
            plt.plot([seg_x, seg_x], mzs, color='grey', linewidth=lw, alpha=0.5)


    def _plot_surface(self, fig, x_in, y, z_in,
                peak_lines, angles, max_I=None, n_plot=1):
        # Create right subplot axis
        ax = fig.add_subplot(2, 2, n_plot, projection='3d')
        if n_plot == 1:
            subplot_title = 'Surface: raw data (I_max = {})'.format(max_I)
            z_label = 'Intensity [log10]'
            x_label = '\n\nRT [s]'
            z = z_in
        elif n_plot == 2:
            subplot_title = 'Surface: baseline corrected'
            z_label = 'Intensity [log10]'
            x_label = '\n\nRT [s]'
            z = np.copy(z_in)
            z[z < 1] = np.nan
            z[z == -np.inf] = np.nan
        else:
            subplot_title = 'Surface: baseline corrected, Rt interpolated, normalized'
            z_label = 'Intensity [normalized]'
            x_label = '\n\nscans [interpolated]'
            z = z_in
            x_in = x_in.astype(int)
        ax.set_title(subplot_title)

        x, y = np.meshgrid(x_in, y)

        mz_min = 60
        mz_max_color = 400
        mz_diff = mz_max_color - mz_min
        colors_norm = np.arange(0, mz_diff) / mz_diff * 255

        alpha = 0.6       
        col_arr = np.tile(
            plt.cm.plasma(range(0, z.shape[0])), (1, z.shape[1])
        ).reshape(z.shape[0], z.shape[1], 4)

        surf = ax.plot_surface(
            x, y, z,
            ccount=z.shape[1], rcount=z.shape[0],
            vmin=z.min(), vmax=z.max(),
            facecolors=col_arr, alpha=alpha,
            linewidth=0,
            antialiased=False
        )

        label_fontsize = 16
        ax.set_xlabel(x_label, fontsize=label_fontsize)
        x_ticks = x_in[::int(x_in.size/10)]
        tick_fontsize = 14
        plt.xticks(
            x_in[::int(x_in.size/10)],
            [round(i, 1) for i in x_in[::int(x_in.size/10)]],
            rotation=45, ha='right'
        )

        ax.set_ylabel('     m/z', fontsize=label_fontsize)
        ax.set_zlabel(z_label, fontsize=label_fontsize)
        ax.invert_xaxis()
        ax.xaxis.set_label_coords(1.05, -0.025)
        ax.xaxis.set_label_position('top')
        if n_plot in [1,2]:
            ax.set_xticklabels(
                x_in.round(1), rotation=30, va='baseline', ha='right'
            )

        ax.tick_params(labelsize=14)

        # Camera perspective
        angle1, angle2 = angles
        ax.view_init(azim=angle1, elev=angle2)

        # Colorbar for mzs
        m = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
        m.set_array(np.clip(y, y.min(), y.min()+255))
        cb = plt.colorbar(m, extend='max')
        cb.set_label('m/z', size=label_fontsize)
       
        # Add borders and apex
        lc = 'red'
        lw = 2
        # Borders
        border_z = max([np.nanmin(z[:, i]) for i in peak_lines.keys()] + [0])
        for line_scan, line_style in peak_lines.items():
            ax.plot(
                xs=x[:,line_scan],
                ys=y[:,1],
                zs=np.full(z.shape[0], border_z),
                color=lc, linestyle=line_style, linewidth=lw, alpha=alpha,
            )

        if n_plot == 4:
            add_surface_x = np.percentile(
                np.arange(z.shape[1]), [10, 37, 63, 90],
                interpolation='nearest'
            )
            for surface_x in add_surface_x:
                ax.plot_surface(
                    np.array([[surface_x, surface_x], [surface_x, surface_x]]),
                    np.array([[y.min(), y.min()], [y.max(), y.max()]]),
                    np.array([[0,1] ,[0,1]]),
                    ccount=1,
                    rcount=1,
                    color='grey',
                    linewidth=0,
                    alpha=0.5,
                )


    def _save_plot(self, fig, out_name):
        fig.set_frameon(False)
        fig.set_figwidth(fig.get_figwidth() * 3)
        fig.set_figheight(fig.get_figheight() * 2)
        plt.subplots_adjust(
            left=0.05, bottom=0.075, right=1, top=0.95, wspace=0.02, hspace=0.15
        )
        plt.savefig(out_name, dpi=36, orient='landscape')


    def plot_all_EICs(self, rt=[0, np.inf]):
        """ Plot all Extracted Ion Chromatograms (EIC).
        
        Args:
            rt (list of floats): min RT and max RT. (Default: [0, np.inf]).

        """
        for i in np.argwhere(self.data_raw.sum(axis=1).round() > 0).flatten():
            mz = i + int(self.mz.min())
            self.plot_EIC('EIC_{}.pdf'.format(mz), mz, rt)


    def plot_EIC(self, out_file, masses=[71, 85, 99], rt=[0, np.inf], apex=None,
                char_mass=None, add_rt=0):
        """ Plot the Extracted Ion Chromatograms (EIC) of certain masses.

        Args:
            out_file (str): Absolute or relative path to the prospective output
                file.
            masses (list of ints): The masses of which the correlating EICs are
                plotted. (Default: [71, 85, 99]).
            rt (list of floats): min RT and max RT. (Default: [0, np.inf]).
            apex (float): If given, plot the apex in the EIC. (Default: None).
            char_mass (int): If given, highlight this mass in the plots legend.
                (Default: None).
            add_rt (int/float): Additional RT to add to the peaks RT.
                (Default: 0).

        """
        if isinstance(masses, (int, float, np.int64, np.float64)):
            masses = [masses]

        tick_fontsize = 20
        axisLabels_fontsize = 25
        fig, ax = plt.subplots()

        colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
        for idx, mass in enumerate(masses):
            if mass == char_mass:
                label_str = 'm/z {:.0f} (characteristic)'.format(mass)
            else:
                label_str = 'm/z {:.0f}'.format(mass)
            plt.plot(
                self.rt[self._rt_idx(rt[0]-add_rt):self._rt_idx(rt[1]+add_rt)],
                self.data_processed[
                    self._mz_idx(mass),
                    self._rt_idx(rt[0]-add_rt):self._rt_idx(rt[1]+add_rt)
                ],
                color=colors[idx], label=label_str
            )
            if add_rt:
                plt.axvline(rt[0], color='#e41a1c', ls='-')
                plt.axvline(rt[1], color='#e41a1c', ls='-')
        if apex:
            plt.axvline(apex, color='#e41a1c', ls='--')

        plt.xlabel('Rt [s]', fontsize=axisLabels_fontsize)
        plt.ylabel('Intensity', fontsize=axisLabels_fontsize)
        plt.legend(fontsize=axisLabels_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.98)

        fig.set_figwidth(fig.get_figwidth() * 6)
        fig.set_figheight(fig.get_figheight() * 4)
        plt.savefig(out_file, dpi=36, orient='landscape')


    def plot_TIC(self):
        """ Show the Total Ion Chromatograms (TIC).

        """
        tick_fontsize = 20
        axisLabels_fontsize = 25
        fig, ax = plt.subplots()

        plt.plot(self.rt, self.data_raw.sum(axis=0), label='TIC')

        plt.xlabel('Rt [s]', fontsize=axisLabels_fontsize)
        plt.ylabel('Intensity', fontsize=axisLabels_fontsize)
        plt.legend(fontsize=axisLabels_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

        plt.show()


    def plot_spectra(self, RT, out_file, color='red', norm=False, spec2_str=''):
        """ Plot the mass spectra at a certain RT.

        Args:
            RT (float): The mass spectras RT.
            out_file (str): Absolute or relative path to the prospective output
                file.
            color (str): Legit color name for plotting the spectra.
                (Default: 'red').
            norm (bool): If True, normalize the intensity, otherwise not.
                (Default: False).
            spec2_str (str): If present, plot both spectra as a north-south plot.
                Format of the second spectra needs to be:
                    '60:15,61:101, 75:1500, ...'
                (Default: '').

        """
        tick_fontsize = 20
        axisLabels_fontsize = 25
        fig, ax = plt.subplots()
        data = self.data_processed[:,self._rt_idx(RT)]
        if norm:
            data = data / data.max() * 1000

        plt.bar(self.mz, data, color=color, label='RT {:.0f}'.format(RT))
        if spec2_str:
            spec2_mzs = pd.Series(
                {int(i.split(':')[0]): -int(i.split(':')[1]) \
                    for i in spec2_str.split(',')}
            )
            spec2 = pd.Series(0, index=self.mz)
            spec2.loc[spec2_mzs.index] = spec2_mzs.values
            plt.bar(self.mz, spec2, color='blue', label='Reference')
            plt.axhline(0, color='black')

        plt.xlabel('m/z', fontsize=axisLabels_fontsize)
        plt.ylabel('Intensity', fontsize=axisLabels_fontsize)

        plt.legend(fontsize=axisLabels_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

        ax.set_xlim([self.mz.min()-5, self.mz.max()+5])
        majorLocator = MultipleLocator(25)
        minorLocator = MultipleLocator(1)

        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)

        fig.set_figwidth(fig.get_figwidth() * 6)
        fig.set_figheight(fig.get_figheight() * 1)
        plt.subplots_adjust(left=0.05, bottom=0.15, right=0.99, top=0.99)

        fig.set_figwidth(fig.get_figwidth() * 3)
        fig.set_figheight(fig.get_figheight() * 2)
        plt.savefig(out_file, dpi=72, orient='landscape')



def load_sampledata(file):
    """ Load sample data from the file system. If sample data was processed
    before and saved as pickle, use pickle file.

    Args:
        file (str): Absolute or relative path to the sample file.

    Returns:
        SampleData: loaded SampleData object

    """

    file_type = file.split('.')[-1]
    pkl_file = file.replace(file_type, 'pkl')
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)
    else:
        return SampleData(file)


if __name__ == '__main__':
    print('There is nothing here...')