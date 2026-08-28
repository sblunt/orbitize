import numpy as np
import corner
import warnings
import itertools

import astropy.units as u
import astropy.constants as consts
from astropy.time import Time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
import matplotlib.transforms as transforms

from erfa import ErfaWarning

import orbitize
import orbitize.kepler as kepler


cmap = mpl.cm.Purples_r
cmap = colors.LinearSegmentedColormap.from_list(
    "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=0.0, b=0.7),
    cmap(np.linspace(0.0, 0.7, 1000)),
)

class Plotter(object):
    """
    A class to plot results

    Args:
        results (orbitize.results.Results): results of a fit to be plotted
        object_to_plot (int or list of int): which objects to plot (default: 1)
        start (float): time at which to start plotting orbits, in `time_format` (default: None,
            three years before the first data point)
        end (float): time at which to stop plotting orbits in `time_format` (default: None,
            three years after the last data point)
        time_format (string): time format for `start` and `end` such as "mjd" or others
            from astropy.time.Time.FORMATS (default: 'decimalyear')
        num_orbits_to_plot (int): number of orbits to plot (default: 100)
        num_epochs_to_plot (int): number of points to plot per orbit (default: 100)
        cbar_param (string): options are the following: 'Epoch [year]', 'sma1', 'ecc1', 'inc1', 'aop1',
            'pan1', 'tau1', 'plx', 'm0', 'm1'. Number can be switched out. (default: 'Epoch [year]').
        rv_time_series (bool): if fitting for secondary mass using MCMC for rv fitting,
            calculate rv_time_series of the primary (object 0) (default: False)
        rv_time_series2 (bool): if fitting for secondary mass using MCMC for rv fitting,
            calculate rv_time_series of the companion (object 1) (default: False)
        post_to_use (np.array of float): num_orbits_to_plot x num_params posterior to use instead of randomly draw from the posterior 

    (written): Henry Ngo, Sarah Blunt, 2018
    Additions by Malena Rice, 2019
    Additions by Dino Hsu, 2023
    Refactored to class by Eshel Dror, 2026

    """
    # Color Map
    CMAP = cmap
    # color bar
    CBAR_SPACING = 0.005
    CBAR_WIDTH = 0.02
    # colour/shape scheme scheme for data points
    ASTR_COLORS = ("#FF7F11", "#FF1919", "#7A11FF", "#11FFE3", "#14FF11")
    ASTR_SYMBOLS = (".", "*", "p", "s")
    MODEL_COLORS = ("#372554", "#0496FF", "#FF1053", "#3A7CA5", "#143109")
    RV_COLORS = ("#0496FF", "#372554", "#FF1053", "#3A7CA5", "#143109")
    RV_ERR_COLORS = ("#FF7F11", "#FF1919", "#7A11FF", "#11FFE3", "#14FF11")
    RV_SYMBOLS = ("o", "^", "v", "s")
    # Latex for rv error
    RV_ERR_MATH = {"offset" : "std(\\gamma)", "observation": "\\epsilon", "jitter": "med(\\sigma)"}
    # Labels for objects by object index used when plotting multiple objects (the object index is used if not in the dictionary)
    OBJECT_LABELS = {0: "*"}


    def __init__(
        self,
        results,
        object_to_plot=1,
        start=None,
        end=None,
        time_format="decimalyear",
        num_orbits_to_plot=100,
        num_epochs_to_plot=100,
        cbar_param="Epoch [year]",
        rv_time_series=False,
        rv_time_series2=False,
        post_to_use=None
    ):
        self.results = results
        self.system = results.system

        if start is None:
           start = getattr(Time(np.min(self.system.data_table['epoch'])-365*3, format="mjd"), time_format)
        if end is None:
           end = getattr(Time(np.max(self.system.data_table['epoch'])+365*3, format="mjd"), time_format)

        self.set_params(object_to_plot, start, end, time_format, num_orbits_to_plot, num_epochs_to_plot, cbar_param, rv_time_series, rv_time_series2, post_to_use)

    def set_params(
        self,
        object_to_plot=None,
        start=None,
        end=None,
        time_format=None, 
        num_orbits_to_plot=None,
        num_epochs_to_plot=None,
        cbar_param=None,
        rv_time_series=None,
        rv_time_series2=None,
        post_to_use=None,
    ):
        """
        Change parameters set when initializing `Plotter` and perform plotting precalculations
        """
        if object_to_plot is not None:
            if isinstance(object_to_plot, int):
                self.objects_to_plot = [object_to_plot]
            else:
                self.objects_to_plot = object_to_plot
        if time_format is not None:
            self.time_format = time_format
        if start is not None:
            self.start = Time(start, format=self.time_format)
        if end is not None:
            self.end = Time(end, format=self.time_format)
        if num_orbits_to_plot is not None:
            self.num_orbits_to_plot = num_orbits_to_plot
        if num_epochs_to_plot is not None:
            self.num_epochs_to_plot = num_epochs_to_plot
        if rv_time_series is not None:
            self.rv_time_series = rv_time_series
        if rv_time_series2 is not None:
            self.rv_time_series2 = rv_time_series2

        self.data = self.results.data

        if cbar_param is not None:
            # Check cbar_param
            if cbar_param in ["Epoch [year]", "Epoch (year)"]:
                pass
            elif cbar_param in self.results.param_idx or cbar_param in self.results.standard_param_idx:
                pass
            else:
                raise Exception(
                    "Invalid input; acceptable inputs include 'Epoch [year]', 'plx', 'sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'sma2', 'ecc2', ...)"
                )
            self.cbar_param = cbar_param

        if self.start >= self.end:
            raise ValueError(
                "start keyword date must be less than end keyword date."
            )

        if max(self.objects_to_plot) > self.results.num_secondary_bodies:
            raise ValueError(
                "Only {0} secondary bodies being fit. Requested to plot body {1} which is out of range".format(
                    self.results.num_secondary_bodies, self.objects_to_plot
                )
            )

        if 0 in self.objects_to_plot and not (self.system.use_rebound or self.system.track_planet_perturbs):
            raise ValueError(
                "Plotting the primary's orbit without rebound or tracking planetary perturbations (fitting for secondary masses) currently unsupported."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)

            if post_to_use is not None:
                self.post = post_to_use
                post = np.copy(post_to_use)
                self.standard_post = self.results.basis.to_standard_basis(post.T).T
            else:
                self.post, self.standard_post = self._get_standard_post(self.num_orbits_to_plot)
            self.period_raoffs, self.period_deoffs, self.period_vzs, self.period_epochss = self._calc_full_orbits(
                self.start, self.num_orbits_to_plot, self.num_epochs_to_plot, self.standard_post)
            self.fixed_raoffs, self.fixed_deoffs, self.fixed_vzs, self.fixed_brightnesses, self.fixed_epochs = self._calc_panel_orbits(
                self.start, self.num_epochs_to_plot, self.standard_post, self.end)
            self.cbar_param_arr, self.norm, self.norm_yr = self._create_cbar(self.cbar_param, self.period_epochss, self.post, self.standard_post)
            if self.rv_time_series:
                self.rv_data, self.gam_idx, self.rv_inst_inds, self.sig_idx = self._calc_rv(object_index=0)
                if len(self.rv_data) == 0:
                    warnings.warn("Unable to calculate primary radial velocity data.")
                    self.rv_time_series=False
            if self.rv_time_series2:
                self.rv_data2, self.gam_idx2, self.rv_inst_inds2, self.sig_idx2 = self._calc_rv(object_index=1)
                if len(self.rv_data2) == 0:
                    warnings.warn("Unable to calculate secondary radial velocity data.")
                    self.rv_time_series2=False
            (
                self.sep_datas, self.sep_errs, self.pa_datas, self.pa_errs, self.seppa_corrs, self.ra_datas, self.ra_errs, self.dec_datas, self.dec_errs, self.radec_corrs,
                self.bright_datas, self.bright_errs, self.bright_insts, self.bright_inst_inds, self.bright_epochs,
                self.astr_raoffs, self.astr_deoffs, self.astr_vzs, self.astr_epochs, self.astr_insts, self.astr_inst_inds
            ) = self._calc_astrometry(self.standard_post, self.data, self.objects_to_plot)

    def _calc_astrometry(self, standard_post, all_data, objects_to_plot):
        """
        Calculate seppa/radec/brightness data and model from ``self.start`` to ``self.end``
        
        Args:
            standard_post (np.array): num_orbits x num_params posterior of orbital parameters from Results.post
            all_data (astropy.table.Table): data on all objects
            objects_to_plot (list of int): indexes of objects for which to calculate values

        Return:
            19-tuple:
                sep_datas (list of np.array): separation data for any data point where seppa/radec was available for each object
                sep_errs (list of np.array): separation data error for any data point where seppa/radec was available for each object
                pa_datas (list of np.array): principal axis data for any data point where seppa/radec was available for each object
                pa_errs (list of np.array): principal axis data error for any data point where seppa/radec was available for each object
                seppa_corrs (list of np.array): correlation coefficient of sep/pa data error for any data point where seppa/radec was available for each object, may be nan
                ra_datas (list of np.array): right ascension data for any data point where seppa/radec was available for each object
                ra_errs (list of np.array): right ascension data error for any data point where seppa/radec was available for each object
                dec_datas (list of np.array): declination data for any data point where seppa/radec was available for each object
                dec_errs (list of np.array): declination data error for any data point where seppa/radec was available for each object
                radec_corrs (list of np.array): correlation coefficient of ra/dec data error for any data point where seppa/radec was available for each object, may be nan
                brightness_datas (list of np.array of float): brightness measurements for each object
                brithness_errs (list of np.array of float): brightness measurement errors for each object
                bright_insts (list of np.array of string): brightness instruments for each object
                bright_inst_inds (list of dictionary of string to np.array of int): indices of data points of each brightness instrument for each object
                brightness_epochs (list of np.array of float): epochs of brightness data for each object
                astr_raoffs (list of np.array of float): num_orbits x num_astr_epochs Right Ascension offset at astrometry epochs for each object
                astr_deoffs (list of np.array of float): num_orbits x num_astr_epochs Declination offset at astrometry epochs for each object
                astr_vzs (list of np.array of float): num_orbits x num_astr_epochs Radial velocities at astrometry epochs for each object
                astr_epochs (list of np.array of float): epochs of astrometry data for each object
                astr_insts (list of np.array of string): names of astrometry instruments for each object
                astr_inst_inds (list of dictionary of string to np.array of int): indices of data points of each astrometry instrument for each object  
        """
        sep_datas = []
        sep_errs = []
        pa_datas = []
        pa_errs = []
        seppa_corrs = []
        ra_datas = []
        ra_errs = []
        dec_datas = []
        dec_errs = []
        radec_corrs = []

        bright_datas = []
        bright_errs = []
        bright_insts = []
        bright_inst_inds = []
        bright_epochs = []

        astr_raoffs = []
        astr_deoffs = []
        astr_vzs = []
        astr_epochs = []
        astr_insts = []
        astr_inst_inds = []

        for i in objects_to_plot:
            object_data = all_data[all_data["object"] == i]
            sep_data, sep_err, pa_data, pa_err, seppa_corr, ra_data, ra_err, dec_data, dec_err, radec_corr = self._calc_seppa_radec(object_data)
            sep_datas.append(sep_data)
            sep_errs.append(sep_err)
            pa_datas.append(pa_data)
            pa_errs.append(pa_err)
            seppa_corrs.append(seppa_corr)
            ra_datas.append(ra_data)
            ra_errs.append(ra_err)
            dec_datas.append(dec_data)
            dec_errs.append(dec_err)
            radec_corrs.append(radec_corr)

            bright_data, bright_err, bright_inst, bright_inst_ind, bright_epoch = self._calc_brightness(object_data)
            bright_datas.append(bright_data)
            bright_errs.append(bright_err)
            bright_insts.append(bright_inst)
            bright_inst_inds.append(bright_inst_ind)
            bright_epochs.append(bright_epoch)

            astr_raoff, astr_deoff, astr_vz, astr_epoch, astr_inst, astr_inst_ind = self._calc_astr_orbits(standard_post, i, object_data)
            astr_raoffs.append(astr_raoff) 
            astr_deoffs.append(astr_deoff) 
            astr_vzs.append(astr_vz) 
            astr_epochs.append(astr_epoch) 
            astr_insts.append(astr_inst) 
            astr_inst_inds.append(astr_inst_ind) 


        return (sep_datas, sep_errs, pa_datas, pa_errs, seppa_corrs, ra_datas, ra_errs, dec_datas, dec_errs, radec_corrs,
                bright_datas, bright_errs, bright_insts, bright_inst_inds, bright_epochs,
                astr_raoffs, astr_deoffs, astr_vzs, astr_epochs, astr_insts, astr_inst_inds)

    def _calc_seppa_radec(self, object_data):
        """
        Calculate both Sepparation/PA and Right Ascension/Declination from the data of an object

        Arg:
            object_data (astropy.table.Table): data on an object
        Return:
            8-tuple:
                sep_data (np.array): separation data for any data point where seppa/radec was available
                sep_err (np.array): separation data error for any data point where seppa/radec was available
                pa_data (np.array): principal axis data for any data point where seppa/radec was available
                pa_err (np.array): principal axis data error for any data point where seppa/radec was available
                seppa_corr (np.array): correlation coefficient of sep/pa data error for any data point where seppa/radec was available, may be nan
                ra_data (np.array): right ascension data for any data point where seppa/radec was available
                ra_err (np.array): right ascension data error for any data point where seppa/radec was available
                dec_data (np.array): declination data for any data point where seppa/radec was available
                dec_err (np.array): declination data error for any data point where seppa/radec was available
                radec_corr (np.array): correlation coefficient of ra/dec data error for any data point where seppa/radec was available, may be nan
        """
        astr_inds = np.where((~np.isnan(object_data["quant1"])) & (~np.isnan(object_data["quant2"])))
        data = object_data[astr_inds]
        radec_inds = np.where(data["quant_type"] == "radec")
        seppa_inds = np.where(data["quant_type"] == "seppa")

        # transform RA/Dec points to Sep/PA
        sep_data = np.copy(data["quant1"])
        sep_err = np.copy(data["quant1_err"])
        pa_data = np.copy(data["quant2"])
        pa_err = np.copy(data["quant2_err"])
        seppa_corr = np.copy(data["quant12_corr"])

        if len(radec_inds[0] > 0):

            sep_from_ra_data, pa_from_dec_data = orbitize.system.radec2seppa(
                data["quant1"][radec_inds], data["quant2"][radec_inds]
            )

            num_radec_pts = len(radec_inds[0])
            sep_err_from_ra_data = np.empty(num_radec_pts)
            pa_err_from_dec_data = np.empty(num_radec_pts)
            seppa_corr_from_radec_data = np.empty(num_radec_pts)
            for j in np.arange(num_radec_pts):

                sep_err_from_ra_data[j], pa_err_from_dec_data[j], seppa_corr_from_radec_data[j] = (
                    orbitize.system.transform_errors(
                        np.array(data["quant1"][radec_inds][j]),
                        np.array(data["quant2"][radec_inds][j]),
                        np.array(data["quant1_err"][radec_inds][j]),
                        np.array(data["quant2_err"][radec_inds][j]),
                        np.array(data["quant12_corr"][radec_inds][j]),
                        orbitize.system.radec2seppa,
                    )
                )

            sep_data[radec_inds] = sep_from_ra_data
            sep_err[radec_inds] = sep_err_from_ra_data

            pa_data[radec_inds] = pa_from_dec_data
            pa_err[radec_inds] = pa_err_from_dec_data

            seppa_corr[radec_inds] = seppa_corr_from_radec_data

        # Transform Sep/PA points to RA/Dec
        ra_data = np.copy(data["quant1"])
        ra_err = np.copy(data["quant1_err"])
        dec_data = np.copy(data["quant2"])
        dec_err = np.copy(data["quant2_err"])
        radec_corr = np.copy(data["quant12_corr"])

        if len(seppa_inds[0] > 0):

            ra_from_seppa_data, dec_from_seppa_data = orbitize.system.seppa2radec(
                data["quant1"][seppa_inds], data["quant2"][seppa_inds]
            )

            num_seppa_pts = len(seppa_inds[0])
            ra_err_from_seppa_data = np.empty(num_seppa_pts)
            dec_err_from_seppa_data = np.empty(num_seppa_pts)
            radec_corr_from_seppa_data = np.empty(num_seppa_pts)
            for j in np.arange(num_seppa_pts):

                ra_err_from_seppa_data[j], dec_err_from_seppa_data[j], radec_corr_from_seppa_data[j] = (
                    orbitize.system.transform_errors(
                        np.array(data["quant1"][seppa_inds][j]),
                        np.array(data["quant2"][seppa_inds][j]),
                        np.array(data["quant1_err"][seppa_inds][j]),
                        np.array(data["quant2_err"][seppa_inds][j]),
                        np.array(data["quant12_corr"][seppa_inds][j]),
                        orbitize.system.seppa2radec,
                    )
                )

            ra_data[seppa_inds] = ra_from_seppa_data
            ra_err[seppa_inds] = ra_err_from_seppa_data

            dec_data[seppa_inds] = dec_from_seppa_data
            dec_err[seppa_inds] = dec_err_from_seppa_data

            radec_corr[seppa_inds] = radec_corr_from_seppa_data
        
        return sep_data, sep_err, pa_data, pa_err, seppa_corr, ra_data, ra_err, dec_data, dec_err, radec_corr

    def _calc_brightness(self, object_data):
        """
        Calculate brightness data for object
        
        Args:
            object_data (``astropy.table.Table``): data of object
        
        Return:
            5-tuple:
                brightness_data (np.array of float): brightness measurements
                brithness_err (np.array of float): brightness measurement errors
                bright_insts (np.array of string): brightness instruments
                bright_inst_inds (dictionary of string to np.array of int): indices of data points of each brightness instrument
                brightness_epochs (np.array of float): epochs of brightness data
        """
        brightness_indices = np.where(object_data["quant_type"] == "brightness")
        data = object_data[brightness_indices]
        brightness_data = data["quant1"]
        brightness_err = data["quant1_err"]

        bright_epochs = object_data["epoch"][brightness_indices]
        bright_insts = np.unique(data["instrument"])
        
        # Indices corresponding to each instrument in datafile
        bright_inst_inds = {}
        for i in range(len(bright_insts)):
            bright_inst_inds[bright_insts[i]] = np.where(
                (data["instrument"] == bright_insts[i].encode()) | (data["instrument"] ==  bright_insts[i])
            )[0]
        return brightness_data, brightness_err, bright_insts, bright_inst_inds, bright_epochs

    def _calc_astr_orbits(self, standard_post, object_to_plot, data):
        """
        Calculate position in orbit at epochs of astrometry data

        Args:
            standard_post (np.array): num_orbits x num_params posterior of orbital parameters from Results.post
            object_to_plot (int): index of object to plot
            data (astropy.table.Table): data of object
        
        Return:
            6-tuple:
                raoff (np.array of float): num_orbits x num_astr_epochs Right Ascension offset at astrometry epochs
                deoff (np.array of float): num_orbits x num_astr_epochs Declination offset at astrometry epochs
                vz (np.array of float): num_orbits x num_astr_epochs Radial velocities at astrometry epochs
                astr_epochs (np.array of float): epochs of astrometry data
                astr_insts (np.array of string): names of astrometry instruments
                astr_inst_inds (dictionary of string to np.array of int): indices of data points of each astrometry instrument
        """
        astr_inds = np.where((~np.isnan(data["quant1"])) & (~np.isnan(data["quant2"])))
        astr_epochs = data["epoch"][astr_inds]

        astr_data = data[astr_inds]
        astr_insts = np.unique(astr_data["instrument"])

        # Indices corresponding to each instrument in datafile
        astr_inst_inds = {}
        for i in range(len(astr_insts)):
            astr_inst_inds[astr_insts[i]] = np.where(
                (astr_data["instrument"] == astr_insts[i].encode()) | (astr_data["instrument"] ==  astr_insts[i])
            )[0]

        # Calculate ra/dec offsets for all epochs of this orbit
        raoff0, deoff0, vz0, _ = self.system.compute_all_orbits(
            standard_post.T,
            astr_epochs
        )

        raoff = np.transpose(raoff0, [1,2,0])[object_to_plot, :, :]
        deoff = np.transpose(deoff0, [1,2,0])[object_to_plot, :, :]
        vz = np.transpose(vz0, [1,2,0])[object_to_plot, :, :]
        
        return raoff, deoff, vz, astr_epochs, astr_insts, astr_inst_inds

    def _calc_rv(self, object_index):
        """
        Calculate information relevant to RV data

        Args:
            object_index (int): index of object for which to calculate RV information
        Return:
            4-tuple:
                rv_data (astropy.table.Table): RV data of object
                gam_idx (list of int): indices corresponding to each gamma within results.labels
                inds (list of int): indexes of each instrument in the datafile
                sig_idx (list of int): indices corresponding to each sigma within results.labels
        """
        rv_data = self.results.data[self.results.data["object"] == object_index]
        rv_data = rv_data[rv_data["quant_type"] == "rv"]

        # get list of rv instruments
        insts = np.unique(rv_data["instrument"])
        if len(insts) == 0:
            insts = ["defrv"]

        # get gamma/sigma labels and corresponding positions in the posterior
        gams = ["gamma_" + inst for inst in insts]
        sigs = ["sigma_" + inst for inst in insts]

        if isinstance(self.results.labels, list):
            labels = np.array(self.results.labels)
        else:
            labels = self.results.labels

        # get the indices corresponding to each gamma within results.labels
        gam_idx = [np.where(labels == inst_gamma)[0] for inst_gamma in gams]
        sig_idx = [np.where(labels == inst_sigma)[0] for inst_sigma in sigs]

        # indices corresponding to each instrument in the datafile
        inds = {}
        for i in range(len(insts)):
            inds[insts[i]] = np.where( # include encode for backwards compatibility
                (rv_data["instrument"] == insts[i].encode()) | (rv_data["instrument"] == insts[i])
            )[0]
        return rv_data, gam_idx, inds, sig_idx

    def _get_standard_post(self, num_orbits_to_plot):
        """
        Downsamples a posterior and calculates standard basis values

        Args:
            num_orbits_to_plot (int): Number of orbits to sample from `self.results`
        Return:
            2-tuple:
                ``np.array``: min(num_orbits_to_plot, post.shape[0]) x (num params) posterior
                ``np.array``: min(num_orbits_to_plot, post.shape[0]) x (num params) posterior converted to standard basis
        """
        # TODO: Replace random with results.downsample
        num_orbits = len(self.results.post[:, 0])
        if num_orbits_to_plot > num_orbits:
            self.num_orbits_to_plot = num_orbits
            num_orbits_to_plot = num_orbits
        choose = np.random.randint(0, high=num_orbits, size=num_orbits_to_plot)

        post = self.results.post[choose, :]
        standard_post = self.results.basis.to_standard_basis(np.copy(post).T).T
        return post, standard_post
    
    def _calc_full_orbits(self, start, num_orbits_to_plot, num_epochs_to_plot, standard_post, periods_to_plot=1):
        """
        Calculate position in orbit at equally spaced epochs over a number of orbital periods for all objects

        Args:
            start (astropy.time.Time): epoch at which to start calculating the orbits
            num_orbits_to_plot (int): number of orbits for which to calculate, no more than num_orbits in standard_post
            num_epochs_to_plot (int): number of equally spaced epochs at which to calculate positions
            standard_post (np.array): num_orbits x num_params posterior of orbital parameters from Results.post
            periods_to_plot (float): number of periods to calculate for each set of orbital parameters (default: 1)
        
        Return:
            4-tuple:
                raoffs (np.array of float): num_objects x num_orbits_to_plot x num_epochs_to_plot Right Ascension offset at epochs
                deoffs (np.array of float): num_objects x num_orbits_to_plot x num_epochs_to_plot Declination offset at epochs
                vzs (np.array of float): num_objects x num_orbits_to_plot x num_epochs_to_plot Radial velocities at epochs
                epochss (np.array of float): num_objects x num_orbits_to_plot x num_epochs_to_plot epochs for each orbit 
        """
        num_objects = self.results.num_secondary_bodies + 1
        raoffs = np.zeros((num_objects, num_orbits_to_plot, num_epochs_to_plot))
        deoffs = np.zeros((num_objects, num_orbits_to_plot, num_epochs_to_plot))
        vzs = np.zeros((num_objects, num_orbits_to_plot, num_epochs_to_plot))
        epochss = np.zeros((num_objects, num_orbits_to_plot, num_epochs_to_plot))

        for i in range(num_objects):
            raoff, deoff, vz, epochs = self._calc_object_full_orbits(start, num_orbits_to_plot, num_epochs_to_plot, i, standard_post, periods_to_plot)
            raoffs[i, :, :] = raoff
            deoffs[i, :, :] = deoff
            vzs[i, :, :] = vz
            epochss[i, :, :] = epochs
        return raoffs, deoffs, vzs, epochss

    def _calc_object_full_orbits(self, start, num_orbits_to_plot, num_epochs_to_plot, object_to_plot, standard_post, periods_to_plot=1):
        """
        Calculate position in orbit at equally spaced epochs over a number of orbital periods for an object

        Args:
            start (astropy.time.Time): epoch at which to start calculating the orbits
            num_orbits_to_plot (int): number of orbits for which to calculate, no more than num_orbits in standard_post
            num_epochs_to_plot (int): number of equally spaced epochs at which to calculate positions
            object_to_plot (int): index of object to plot
            standard_post (np.array): num_orbits x num_params posterior of orbital parameters from Results.post
            periods_to_plot (float): number of periods to calculate for each set of orbital parameters (default: 1)
        
        Return:
            4-tuple:
                raoff (np.array of float): num_orbits_to_plot x num_epochs_to_plot Right Ascension offset at epochs
                deoff (np.array of float): num_orbits_to_plot x num_epochs_to_plot Declination offset at epochs
                vz (np.array of float): num_orbits_to_plot x num_epochs_to_plot Radial velocities at epochs
                epochs (np.array of float): num_orbits_to_plot x num_epochs_to_plot epochs for each orbit 
        """
        raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        vz = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        
        # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
        # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
        # Compute period (from Kepler's third law)
        if object_to_plot == 0:
            # Use longest secondary period
            secondaries = np.arange(self.results.num_secondary_bodies) + 1
            sma_indexes = [self.results.standard_param_idx["sma{}".format(i)] for i in secondaries]
            smas = standard_post[:, sma_indexes]
            if "mtot" in self.results.labels:
                mtots = standard_post[:, self.results.standard_param_idx["mtot"]]
            elif "m0" in self.results.labels:
                m0 = standard_post[:, self.results.standard_param_idx["m0"]]
                m1_indexes = [self.results.standard_param_idx["m{}".format(i)] for i in secondaries]
                print(m1_indexes)
                m1s = standard_post[:, m1_indexes]
                mtots = (m0 + m1s.T).T
            periods = np.sqrt(
                4 * np.pi**2.0 * (smas * u.AU) ** 3 / (consts.G * (mtots * u.Msun))
            )
            periods = periods.to(u.day).value
            period = np.max(periods, axis=1)
        else:
            sma = standard_post[
                :, self.results.standard_param_idx["sma{}".format(object_to_plot)]
            ]
            if "mtot" in self.results.labels:
                mtot = standard_post[:, self.results.standard_param_idx["mtot"]]
            elif "m0" in self.results.labels:
                m0 = standard_post[:, self.results.standard_param_idx["m0"]]
                m1 = standard_post[
                    :, self.results.standard_param_idx["m{}".format(object_to_plot)]
                ]
                mtot = m0 + m1
            period = np.sqrt(
                4 * np.pi**2.0 * (sma * u.AU) ** 3 / (consts.G * (mtot * u.Msun))
            )
            period = period.to(u.day).value
        for i in np.arange(num_orbits_to_plot):
            # Create an epochs array to plot num_epochs_to_plot points over one orbital period
            epochs[i, :] = np.linspace(
                start.mjd, float(start.mjd + period[i]*periods_to_plot), num_epochs_to_plot
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            raoff0, deoff0, vz0, _ = self.system.compute_all_orbits(
                standard_post[i],
                epochs[i, :]
            )

            raoff[i, :] = raoff0[:, object_to_plot, 0]
            deoff[i, :] = deoff0[:, object_to_plot, 0]
            vz[i, :] = vz0[:, object_to_plot, 0]
        return raoff, deoff, vz, epochs

    def _calc_panel_orbits(self, start, num_epochs_to_plot, standard_post, end):
        """
        Calculate position in orbit at equally spaced epochs from a start to end epochs
            
        Args:
            start (``astropy.time.Time``): epoch at which to start calculating the orbits
            num_epochs_to_plot (int): number of equally spaced epochs at which to calculate positions
            standard_post (np.array): num_orbits x num_params posterior of orbital parameters from Results.post
            end (``astropy.time.Time``): Epoch at which to end calculating the orbits
        
        Return:
            5-tuple:
                raoff (np.array of float): num_objects x num_orbits_to_plot x num_epochs_to_plot Right Ascension offset at epochs
                deoff (np.array of float): num_objects x num_orbits_to_plot x num_epochs_to_plot Declination offset at epochs
                vz (np.array of float): num_objects x num_orbits_to_plot x num_epochs_to_plot Radial velocities at epochs
                brightness (np.array of float): num_objects x num_orbits_to_plot x num_epochs_to_plot relative brightness at epochs
                epochs (np.array of float): (num_epochs_to_plot) epochs
        """
        epochs = np.linspace(
                start.mjd, end.mjd, num_epochs_to_plot
            )
        
        raoff0, deoff0, vz0, brightness0 = self.system.compute_all_orbits(
            standard_post.T,
            epochs
        )

        # epochs x bodies x orbits -> bodies x orbits x epochs
        raoffs = np.transpose(raoff0, [1,2,0])
        deoffs = np.transpose(deoff0, [1,2,0])
        vzs = np.transpose(vz0, [1,2,0])
        brightnesses = np.transpose(brightness0, [1,2,0])
        return raoffs, deoffs, vzs, brightnesses, epochs

    def _get_object_label(self, object_index):
        """
        Gets the label of an object by its index

        Arg:
            object_index: int
        
        Return:
            (string): object label from `self.OBJECT_LABELS` or `str(object_index)` if no label exists
        """
        return self.OBJECT_LABELS.get(object_index, str(object_index))

    def _create_cbar(self, cbar_param, epochss, post, standard_post):
        """
        Create a linearly increasing colormap for the range of epochs

        Args:
            cbar_param (string): name of parameter ('Epoch [year]', 'Epoch (year)', or a parameter label followed by object index)
            epochss (np.array of float): num_objects x num_orbits x num_epochs epochs for each orbit in standard_post
            post (np.array of float): num_orbits x num_params posterior in system.basis
            standard_post (np.array of float): num_orbits x num_params posterior in standard basis
        
        Return:
            3-tuple:
                cbar_param_arr (np.array of float): (num_orbits) if ``cbar_param`` is not "Epoch [year]", otherwise (num_objects x num_orbits x num_epochs) ``epochs``.
                    The value of the cbar parameter for each orbit.
                norm (``matploblib.colors.Normalize``): linear normalization from the minimum to maximum values of the cbar param.
                    Maximum epoch is a maximum of 1000 years after the minimum epoch if cbar_param is epoch. 
                norm_yr (``matplotlib.colors.Normalize``): same as ``norm`` except in decimal year if ``cbar_param`` is epoch
        """

        if cbar_param not in ["Epoch [year]", "Epoch (year)"]:
            if cbar_param in self.results.param_idx:
                index = self.results.param_idx[cbar_param]
                cbar_param_arr = post[:, index]
            else:
                index = self.results.standard_param_idx[cbar_param]
                cbar_param_arr = standard_post[:, index]
            norm = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )
            norm_yr = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )

        elif cbar_param in ["Epoch [year]", "Epoch (year)"]:
            cbar_param_arr = epochss
            min_cbar_date = np.min(epochss[:, :, 0])
            max_cbar_date = np.max(epochss[:, :, -1])

            # if we're plotting orbital periods greater than 1,000 yrs, limit the colorbar dynamic range
            if max_cbar_date - min_cbar_date > 1000 * 365.25:
                max_cbar_date = min_cbar_date + 1000 * 365.25

            norm = mpl.colors.Normalize(vmin=min_cbar_date, vmax=max_cbar_date)

            norm_yr = mpl.colors.Normalize(
                vmin=Time(min_cbar_date, format="mjd").decimalyear,
                vmax=Time(max_cbar_date, format="mjd").decimalyear,
            )
        return cbar_param_arr, norm, norm_yr

    @staticmethod
    def _plot_correlated_errorbars(ax, x, y, xerr, yerr, pearson, n_std=1.0, **kwargs):
        """
        Plot errorbars with correlated x/y errors

        Args:
            ax (matplotlib.axes.Axes): Axes on which to plot
            x (np.array): x values at which to plot 
            y (np.array): y values at which to plot
            xerr (np.array): xerr of value
            yerr (np.array): yerr of value
            pearson (np.array): pearson coefficient of errors
            n_std (float): number of standard deviations by which to scale radii
            **kwargs: passed to `ax.errorbar`

        Eshel Dror (2026), based on "Plot a confidence ellipse of a two-dimensional dataset" matplotlib tutorial
        """
        if not (x.size == y.size == xerr.size == yerr.size == pearson.size):
            raise ValueError("x and y must be the same size")

        not_correlated = (np.isnan(pearson)) | (pearson == 0.0)
        for i in range(x.size):
            scale_x = xerr[i] * n_std
            scale_y = yerr[i] * n_std
            x0 = x[i]
            y0 = y[i]
            if not_correlated[i]:
                ax.errorbar(x[i], y[i], xerr=scale_x, yerr=scale_y, **kwargs)
            else:
                eigen_x = np.sqrt(1 + pearson[i])
                eigen_y = np.sqrt(1 - pearson[i])
                transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(x0, y0)
                ax.errorbar(0, 0, xerr=eigen_x, yerr=eigen_y, transform=(transf + ax.transData), **kwargs)

    def _plot_full_orbits(self, ax, plot_astrometry, full_plot, fontsize, cmap, plot_astrometry_insts, use_cmap, n_std, object_i, object_index, astr_color, astr_symbols, model_color, err_color):
        """
        Plot RAoff/Decoff orbits and astrometry

        Args:
            ax (matploblib.axes.Axes): Axes on which to plot
            plot_astrometry (bool): plot astrometry data
            full_plot (bool): aspect ratio is always equal, but if
                full_plot is True, then the axes will fill the available space,
                otherwise, white space padding is used
            fontsize (float): fontsize for labels
            cmap (``matplotlib.cm.ColorMap``): color map to use on orbits with ``self.norm, self.norm_yr, self.cbar_param_arr``
            plot_astrometry_insts (bool): plot each astrometry instrument separately
            use_cmap (bool): map a color map to orbits according to `self.cbar_param`,
                otherwise, use one color per planet from `self.MODEL_COLORS`
            n_std (float): number of standard deviations to plot with errorbars
            object_i (int): index of index of object to be plotted witin ``self.objects_to_plot``
            object_index (int): index of object to be plotted
            astr_color (string): matplotlib color string to use for astrometry if ``plot_astrometry``
            astr_symbols (iterable of strings): matplotlib symbol strings to use for astrometry if ``plot_astrometry``
            model_color (string): matploblib color string to use for orbits if not ``use_cmap``
            err_color (string): matplotlib color string to use for errorbars
        """
        # Plot each orbit (each segment between two points coloured using colormap)
        for i in np.arange(self.num_orbits_to_plot):
            points = np.array([self.period_raoffs[object_index, i, :], self.period_deoffs[object_index, i, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            if use_cmap:
                lc = LineCollection(segments, cmap=cmap, norm=self.norm, linewidth=1.0)
                if self.cbar_param not in ["Epoch [year]", "Epoch (year)"]:
                    lc.set_array(np.ones(self.num_epochs_to_plot) * self.cbar_param_arr[i])
                elif self.cbar_param in ["Epoch [year]", "Epoch (year)"]:
                    lc.set_array(self.period_epochss[object_index, i, :])
            else:
                lc = LineCollection(segments, colors=model_color, norm=self.norm, linewidth=1.0)
            ax.add_collection(lc)

        if plot_astrometry:
            # Plot astrometry along with instruments
            if plot_astrometry_insts:
                for astr_inst in self.astr_insts[object_i]:
                    ra = self.ra_datas[object_i][self.astr_inst_inds[object_i][astr_inst]]
                    dec = self.dec_datas[object_i][self.astr_inst_inds[object_i][astr_inst]]
                    ax.scatter(
                        ra,
                        dec,
                        marker=next(astr_symbols),
                        c=astr_color,
                        zorder=10,
                        s=60,
                        label=astr_inst,
                    )
                plt.sca(ax)
                plt.legend(fontsize=15, loc=1)
            else:
                ax.scatter(self.ra_datas[object_i], self.dec_datas[object_i], marker=next(astr_symbols), c=astr_color, zorder=10, s=60)
        if n_std != 0:
            self._plot_correlated_errorbars(
                ax, self.ra_datas[object_i], self.dec_datas[object_i],
                xerr=self.ra_errs[object_i], yerr=self.dec_errs[object_i],
                pearson=self.radec_corrs[object_i], n_std=n_std,
                ecolor=err_color, zorder=11
            )

        # modify the axes
        if full_plot:
            adjustable_param = "datalim"
        else:
            adjustable_param = "box"

        ax.set_aspect("equal", adjustable=adjustable_param)
        ax.set_xlabel("$\\Delta$RA (mas)", fontsize=fontsize)
        ax.set_ylabel("$\\Delta$Dec (mas)", fontsize=fontsize)
        ax.locator_params(axis="x", nbins=6)
        ax.locator_params(axis="y", nbins=6)
        ax.invert_xaxis()  # To go to a left-handed coordinate system

    def _add_colorbar(self, ax, fig, cmaps, num_cbars):
        """
        Adds a colorbar

        Args:
            ax (``matploblib.axes.Axes``): fxes besides which to put colorbar
            fig (``matplotlib.figure.Figure``): Figure which contains ``ax``
            cmaps (list of ``matplotlib.colors.Colormap``): color maps to use with ``self.cbar_param, self.norm_yr``
            num_cbars (int): number of colorsbars to plot
        """
        
        # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
        for i in range(num_cbars):
            cbar_ax = fig.add_axes(
                [
                    ax.get_position().x1 + self.CBAR_SPACING + (self.CBAR_SPACING + self.CBAR_WIDTH)*i,
                    ax.get_position().y0,
                    self.CBAR_WIDTH,
                    ax.get_position().height,
                ]
            )
            cbar = mpl.colorbar.ColorbarBase(
                cbar_ax,
                cmap=cmaps[i],
                norm=self.norm_yr,
                orientation="vertical",
            )
            if num_cbars > 1:
                cbar_objects = self.objects_to_plot[i::num_cbars]
                cbar.ax.set_xlabel(",".join(map(self._get_object_label,cbar_objects)), size=15)
            last = i == num_cbars - 1
            if not last:
                cbar.ax.set_yticklabels([])
        # Add label to last cbar
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(label=self.cbar_param, size=20)

    def _plot_sep_pa_model(self, ax1, ax2, mod180, sep_pa_color, object_i, object_index):
        """
        Plot Sep/PA vs time from model
        
        Args:
            ax1 (``matploblib.axes.Axes``): sep axes
            ax2 (``matploblib.axes.Axes``): pa axes
            mod180 (bool): output PA values will be given in range [180, 540)
                (useful for plotting short arcs with PAs that cross 360 during observations)
            sep_pa_color (string): matploblib color string of orbit tracks
            object_i (int): index of index of object to be plotted witin ``self.objects_to_plot``
            object_index (int): index of object to be plotted
        """
        for i in np.arange(self.num_orbits_to_plot):
            yr_epochs = Time(self.fixed_epochs, format="mjd").decimalyear

            seps, pas = orbitize.system.radec2seppa(
                self.fixed_raoffs[object_index, i, :], self.fixed_deoffs[object_index, i, :], mod180=mod180
            )

            plt.sca(ax1)
            plt.plot(yr_epochs, seps, color=sep_pa_color)

            plt.sca(ax2)
            plt.plot(yr_epochs, pas, color=sep_pa_color)

    def _plot_brightness_model(self, ax, sep_pa_color, object_i, object_index):
        """
        Plot relative brightness vs epoch from model
            
        Args:
            ax (``matploblib.axes.Axes``): axes on which to plot
            sep_pa_color (string): matploblib color string of orbit tracks
            object_i (int): index of index of object to be plotted witin ``self.objects_to_plot``
            object_index (int): index of object to be plotted
        """
        for i in np.arange(self.num_orbits_to_plot):
            yr_epochs = Time(self.fixed_epochs, format="mjd").decimalyear
            brightnesses = self.fixed_brightnesses[object_index, i, :]

            plt.sca(ax)
            plt.plot(yr_epochs, brightnesses, color=sep_pa_color)
    
    def _plot_rv_model(self, ax3, ax4, rv_time_series, rv_time_series2, sep_pa_color):
        """
        Plot primary/secondary rv vs epoch from model

        Args:
            ax3 (``matploblib.axes.Axes``): axes of primary rv time series if ``rv_time_series``,
                axes of secondary rv time series if ``rv_time_series2 and not rv_time_series``
            ax4 (``matploblib.axes.Axes``): axes of secondary rv time series if ``rv_time_series2 and rv_time_series``
            rv_time_series (bool): primary rv time series is being plotted
            rv_time_series2 (bool): secondary rv time series is being plotted
            sep_pa_color (string): matploblib color string of orbit tracks
        """
        for i in np.arange(self.num_orbits_to_plot):
            if rv_time_series:
                plt.sca(ax3)

                # scale back to primary RV semi amplitude
                vz0 = self.fixed_vzs[0, i, :]
                # vz0 = self.vz1[i] * (-(mtot[i] - m0[i]) / np.median(m0[i]))

                plt.plot(
                    Time(self.fixed_epochs, format="mjd").decimalyear,
                    vz0,
                    color=sep_pa_color,
                )

            if rv_time_series2:
                if rv_time_series:
                    plt.sca(ax4)
                else:
                    plt.sca(ax3)

                vz1 = self.fixed_vzs[self.objects_to_plot[0], i, :]
                plt.plot(
                    Time(self.fixed_epochs, format="mjd").decimalyear,
                    vz1,
                    color=sep_pa_color,
                )

    def _plot_sep_pa_data(self, ax1, ax2, plot_astrometry_insts, plot_errorbars, object_i, object_index):
        """
        Plot Sep/PA vs epoch of instrument data.
        
        Args:
            ax1 (``matploblib.axes.Axes``): sep axes
            ax2 (``matploblib.axes.Axes``): pa axes
            plot_astrometry_insts (bool): plot instruments separately
            plot_errorbars (bool): plot errorbars
            object_i (int): index of index of object to be plotted witin ``self.objects_to_plot``
            object_index (int): index of object to be plotted
        """
        # Plot sep/pa instruments
        if plot_astrometry_insts:
            colors = itertools.cycle(self.ASTR_COLORS)
            symbols = itertools.cycle(self.ASTR_SYMBOLS)

            for inst in self.astr_insts[object_i]:
                inst_inds = self.astr_inst_inds[object_i][inst]
                sep = self.sep_datas[object_i][inst_inds]
                pa = self.pa_datas[object_i][inst_inds]
                epochs = self.astr_epochs[object_i][inst_inds]

                serr = self.sep_errs[object_i][inst_inds]
                perr = self.pa_errs[object_i][inst_inds]

                plt.sca(ax1)
                color = next(colors)
                symbol = next(symbols)
                plt.scatter(
                    Time(epochs, format="mjd").decimalyear,
                    sep,
                    s=60,
                    marker=symbol,
                    c=color,
                    zorder=10,
                    label=inst,
                )
                if plot_errorbars:
                    plt.errorbar(
                        Time(epochs, format="mjd").decimalyear,
                        sep,
                        yerr=serr,
                        ms=5,
                        linestyle="",
                        ecolor=color,
                        zorder=10,
                        capsize=2,
                    )
                plt.sca(ax2)
                plt.scatter(
                    Time(epochs, format="mjd").decimalyear,
                    pa,
                    s=60,
                    marker=symbol,
                    c=color,
                    zorder=10,
                    label=inst,
                )
                if plot_errorbars:
                    plt.errorbar(
                        Time(epochs, format="mjd").decimalyear,
                        pa,
                        yerr=perr,
                        ms=5,
                        linestyle="",
                        ecolor=color,
                        zorder=10,
                        capsize=2,
                    )
            plt.sca(ax1)
            plt.legend(fontsize=15, loc=1)
            plt.sca(ax2)
            plt.legend(fontsize=15, loc=1)
        else:
            plt.sca(ax1)
            plt.scatter(
                Time(self.astr_epochs[object_i], format="mjd").decimalyear,
                self.sep_datas[object_i],
                s=60,
                marker=self.ASTR_SYMBOLS[0],
                c=self.ASTR_COLORS[0],
                zorder=10,
            )
            if plot_errorbars:
                plt.errorbar(
                    Time(self.astr_epochs[object_i], format="mjd").decimalyear,
                    self.sep_datas[object_i],
                    yerr=self.sep_errs[object_i],
                    ms=5,
                    linestyle="",
                    ecolor=self.ASTR_COLORS[0],
                    zorder=10,
                    capsize=2,
                )
            plt.sca(ax2)
            plt.scatter(
                Time(self.astr_epochs[object_i], format="mjd").decimalyear,
                self.pa_datas[object_i],
                s=60,
                marker=self.ASTR_SYMBOLS[0],
                c=self.ASTR_COLORS[0],
                zorder=10,
            )
            if plot_errorbars:
                plt.errorbar(
                    Time(self.astr_epochs[object_i], format="mjd").decimalyear,
                    self.pa_datas[object_i],
                    yerr=self.pa_errs[object_i],
                    ms=5,
                    linestyle="",
                    ecolor=self.ASTR_COLORS[0],
                    zorder=10,
                    capsize=2,
                )

    def _plot_brightness_data(self, ax, plot_astrometry_insts, plot_errorbars, object_i, object_index):
        """
        Plot relative brightness vs epoch of instrument data.
        
        Args:
            ax (``matploblib.axes.Axes``): axes on which to plot
            plot_astrometry_insts (bool): plot instruments separately
            plot_errorbars (bool): plot errorbars
            object_i (int): index of index of object to be plotted witin ``self.objects_to_plot``
            object_index (int): index of object to be plotted
        """
        # Plot brightness instruments
        plt.sca(ax)
        if plot_astrometry_insts:
            colors = itertools.cycle(self.ASTR_COLORS)
            symbols = itertools.cycle(self.ASTR_SYMBOLS)

            for inst in self.bright_insts[object_i]:
                inst_inds = self.bright_inst_inds[object_i][inst]
                brightness = self.bright_datas[object_i][inst_inds]
                bright_err = self.bright_errs[object_i][inst_inds]
                epochs = self.bright_epochs[object_i][inst_inds]

                color = next(colors)
                symbol = next(symbols)
                plt.scatter(
                    Time(epochs, format="mjd").decimalyear,
                    brightness,
                    s=60,
                    marker=symbol,
                    c=color,
                    zorder=10,
                    label=inst,
                )
                if plot_errorbars:
                    plt.errorbar(
                        Time(epochs, format="mjd").decimalyear,
                        brightness,
                        yerr=bright_err,
                        ms=5,
                        linestyle="",
                        ecolor=color,
                        zorder=10,
                        capsize=2,
                    )
            plt.legend(fontsize=15, loc=1)
        else:
            plt.scatter(
                Time(self.bright_epochs[object_i], format="mjd").decimalyear,
                self.bright_datas[object_i],
                s=60,
                marker=self.ASTR_SYMBOLS[0],
                c=self.ASTR_COLORS[0],
                zorder=10,
            )
            if plot_errorbars:
                plt.errorbar(
                    Time(self.bright_epochs[object_i], format="mjd").decimalyear,
                    self.bright_datas[object_i],
                    yerr=self.bright_errs[object_i],
                    ms=5,
                    linestyle="",
                    ecolor=self.ASTR_COLORS[0],
                    zorder=10,
                    capsize=2,
                )
    
    def _plot_rv_instruments(self, ax3, ax4, rv_time_series, rv_time_series2, rv_err_grouping, plot_errorbars):
        """
        Plot primary/secondary rv vs epoch of instrument data. The median rv offset (gamma) is subtracted from each instrument.

        Args:
            ax3 (``matploblib.axes.Axes``): axes of primary rv time series if ``rv_time_series``,
                axes of secondary rv time series if ``rv_time_series2 and not rv_time_series``
            ax4 (``matploblib.axes.Axes``): axes of secondary rv time series if ``rv_time_series2 and rv_time_series``
            rv_time_series (bool): plot primary rv time series
            rv_time_series2 (bool): plot secondary rv time series
            sep_pa_color (string): matploblib color string of orbit tracks
            rv_err_grouping (list of tuples of string literals ["observation", "offset", "jitter"]):
                determines how errors for rv time series are grouped. The strings within each tuple determine
                what types of error are included in that errorbar. For example [('offset'), ('observation', 'jitter')]
                would create one errorbar for the rv offset (gamma) and another for the combined observation (epsilon)
                and jitter (sigma) errors.
            plot_errorbars (bool): plot errorbars on data
        """
        ax3_colors = itertools.cycle(self.RV_COLORS)
        ax3_symbols = itertools.cycle(self.RV_SYMBOLS)
        if rv_time_series and len(self.rv_data) > 0:
            # switch current axis to rv panel
            plt.sca(ax3)

            med_ga = [np.median(self.results.post[:,i]) for i in self.gam_idx]
            stddev_ga = [np.std(self.results.post[:,i]) for i in self.gam_idx]
            med_sigma = [np.median(self.results.post[:,i]) for i in self.sig_idx]


            # get rvs and plot them
            for i, name in enumerate(self.rv_inst_inds.keys()):
                inst_data = self.rv_data[self.rv_inst_inds[name]]
                rvs = inst_data["quant1"]
                epochs = inst_data["epoch"]
                epochs = Time(epochs, format="mjd").decimalyear
                primary_obs_err = inst_data["quant1_err"]
                primary_offset_err = stddev_ga[i]
                primary_jitter = med_sigma[i]
                primary_errors = {'observation': primary_obs_err, 'offset': primary_offset_err, 'jitter': primary_jitter}
                name2 = "RV data" if name == "defrv" else name
                plt.scatter(
                    epochs,
                    rvs-med_ga[i],
                    s=30,
                    marker=next(ax3_symbols),
                    c=next(ax3_colors),
                    label=name2,
                    zorder=5,
                )
                if plot_errorbars:
                    bar_width = 1
                    for group_i, grouping in enumerate(rv_err_grouping):
                        if not isinstance(grouping, str):
                            primary_rv_err2 = 0
                            primary_rv_err_labels = []
                            for err_type in grouping:
                                primary_rv_err2 += np.square(primary_errors[err_type])
                                primary_rv_err_labels.append("{0}^2".format(self.RV_ERR_MATH[err_type]))
                            primary_rv_err_label = "$(" + "+".join(primary_rv_err_labels) + ")^\\frac{{1}}{{2}}$"
                            primary_rv_err = np.sqrt(primary_rv_err2)
                        else:
                            primary_rv_err = primary_errors[grouping]
                            primary_rv_err_label = "${0}$".format(self.RV_ERR_MATH[grouping])
                        plt.errorbar(
                            x=epochs,
                            y=rvs-med_ga[i],
                            yerr=primary_rv_err,
                            ecolor=self.RV_ERR_COLORS[group_i],
                            elinewidth=bar_width,
                            zorder=6,
                            ls="none",
                            label=primary_rv_err_label if i==0 else None
                        )
                        bar_width += 1

            if (rv_err_grouping == [("observation", "offset", "jitter")] or (not plot_errorbars)) and len(self.rv_inst_inds.keys()) == 1 and "defrv" in self.rv_inst_inds.keys():
                pass
            else:
                plt.legend(fontsize=20, loc=1)

        if rv_time_series2 and len(self.rv_data2) > 0:
            med_ga2 = [np.median(self.results.post[:,i]) for i in self.gam_idx2]
            stddev_ga2 = [np.std(self.results.post[:,i]) for i in self.gam_idx2]
            med_sigma2 = [np.median(self.results.post[:,i]) for i in self.sig_idx2]

            if rv_time_series:
                plt.sca(ax4)
            else:
                plt.sca(ax3)

            # get rvs and plot them
            for i, name in enumerate(self.rv_inst_inds2.keys()):
                name2 = name.replace("_", " ")
                if name2 == "defrv":
                    name2 = "RV Data"
                inst_data2 = self.rv_data2[self.rv_inst_inds2[name]]
                rvs2 = inst_data2["quant1"]
                epochs2 = inst_data2["epoch"]
                epochs2 = Time(epochs2, format="mjd").decimalyear
                secondary_obs_err = inst_data2["quant1_err"]
                secondary_offset_err = stddev_ga2[i]
                secondary_jitter = med_sigma2[i]
                secondary_errors = {'observation': secondary_obs_err, 'offset': secondary_offset_err, 'jitter': secondary_jitter}
                plt.scatter(
                    epochs2,
                    rvs2-med_ga2[i],
                    s=30,
                    marker=next(ax3_symbols),
                    c=next(ax3_colors),
                    label=name2,
                    zorder=5,
                )
                if plot_errorbars:
                    bar_width = 1
                    for group_i, grouping in enumerate(rv_err_grouping):
                        if not isinstance(grouping, str):
                            secondary_rv_err2 = 0
                            secondary_rv_err_labels = []
                            for err_type in grouping:
                                secondary_rv_err2 += np.square(secondary_errors[err_type])
                                secondary_rv_err_labels.append("{0}^2".format(self.RV_ERR_MATH[err_type]))
                            secondary_rv_err_label = "$(" + "+".join(secondary_rv_err_labels) + ")^\\frac{{1}}{{2}}$"
                            secondary_rv_err = np.sqrt(secondary_rv_err2)
                        else:
                            secondary_rv_err = secondary_errors[grouping]
                            secondary_rv_err_label = "${0}$".format(self.RV_ERR_MATH[grouping])
                        plt.errorbar(
                            x=epochs2,
                            y=rvs2-med_ga2[i],
                            yerr=secondary_rv_err,
                            ecolor=self.RV_ERR_COLORS[group_i],
                            elinewidth=bar_width,
                            zorder=6,
                            ls="none",
                            label=secondary_rv_err_label if i==0 else None
                        )
                        bar_width += 1

            if rv_err_grouping == [("observation", "offset", "jitter")] and len(self.rv_inst_inds2.keys()) == 1 and "defrv" in self.rv_inst_inds2.keys():
                pass
            else:
                plt.legend(fontsize=20, loc=1)

    def plot_orbits(
        self,
        full_plot=True,
        show_colorbar=True,
        use_cmap=True,
        cmap=None,
        sep_pa_color="lightgrey",
        mod180=False,
        plot_astrometry=True,
        plot_astrometry_insts=False,
        plot_brightness=False,
        plot_errorbars=True,
        n_std=1.0,
        rv_time_series=False,
        rv_time_series2=False,
        rv_err_grouping=[("observation", "offset", "jitter")],
        fontsize=20,
        fig=None,
    ):
        """
        Plots RA/Dec for one orbital period and orbit tracks in Sep/PA/RV/Brightness panels from `self.start` to `self.end`.

        Args:
            full_plot (bool): aspect ratio is always equal, but if
                full_plot is True, then the axes will fill the available space,
                otherwise, white space padding is used (deafult: True)
            show_colorbar (bool): displays colorbar to the right of the RA/Dec plot (default: True).
            use_cmap (bool): map a color map to orbits according to `self.cbar_param`,
                otherwise, use one color per planet from `self.MODEL_COLORS` (default: True)
            cmap (``matplotlib.colors.Colormap`` or list of ``matplotlib.colors.Colormap``): color maps
                to use for making orbit tracks (default: None (uses `self.CMAP`))
            sep_pa_color (string): any valid matplotlib color string, used to set the
                color of the orbit tracks in the Sep/PA/RV/Brightness panels (default: 'lightgrey').
            mod180 (bool): if True, PA will be plotted in range [180, 540]. Useful for plotting short
                arcs with PAs that cross 360 deg during observations (default: False)
            plot_astrometry (bool): plots the astrometric data in the RA/Dec panel (default: True)
            plot_astrometry_insts (bool): plots the astrometric data by instruments (default: False)
            plot_brightness (bool): plots the relative brightness if it exists (default: False)
            plot_errorbars (bool): plot errorbars on data (default: True)
            n_std (float): number of standard deviations to plot in the RA/Dec plot, 
                no errorbars are plotted if n_std=0 (default: 1.0)
            rv_time_series (bool): if fitting for secondary mass using MCMC for rv fitting,
                display rv time series of the primary (object 0) (default: False)
            rv_time_series2 (bool): if fitting for secondary mass using MCMC for rv fitting,
                display rv time series of the companion (object 1) (default: False)
            rv_err_grouping (list of tuples of string literals ["observation", "offset", "jitter"]):
                determines how errors for rv time series are grouped. The strings within each tuple determine
                what types of error are included in that errorbar. For example [('offset'), ('observation', 'jitter')]
                would create one errorbar for the rv offset (gamma) and another for the combined observation (epsilon)
                and jitter (sigma) errors. (default: [('observation', 'offset', 'jitter)]) 
            fontsize (int): font size of labels (default: 20)
            fig (``matplotlib.pyplot.Figure``): optionally include a predefined Figure object to plot the orbit on.
                Most users will not need this keyword.

        Return:
            ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


        (written): Henry Ngo, Sarah Blunt, 2018
        Additions by Malena Rice, 2019
        Additions by Dino Hsu, 2023
        Refactoring and additions by Eshel Dror, 2026 

        """
        
        if cmap is None:
            cmap = self.CMAP

        if isinstance(cmap, colors.Colormap):
            cmaps = [cmap]
        else:
            cmaps = cmap

        if not use_cmap:
            show_colorbar = False

        if show_colorbar:
            num_cbars = min(len(cmaps), len(self.objects_to_plot))
        else:
            num_cbars = 0

        if (rv_time_series or rv_time_series2) and "m0" not in self.results.labels:
            rv_time_series = False
            rv_time_series2 = False

            warnings.warn(
                "It seems that the stellar and companion mass "
                "have not been fitted separately. Setting "
                "rv_time_series=True is therefore not possible "
                "so the argument is set to False instead."
            )

        if plot_brightness and len(self.objects_to_plot) == 1 and len(self.bright_datas[0]) == 0:
            plot_brightness = False
            warnings.warn("No brightness data was provided so plot_brightness=True is not possible")

        if (rv_time_series and not self.rv_time_series) or (rv_time_series2 and not self.rv_time_series2):
            self.set_params(rv_time_series=rv_time_series, rv_time_series2=rv_time_series2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            # Create figure for orbit plots
            height = 2 + rv_time_series + rv_time_series2
            num_objects_to_plot = len(self.objects_to_plot)
            if num_objects_to_plot > 1:
                height += num_objects_to_plot + 2
            if num_objects_to_plot == 1 and plot_brightness:
                height += 1
            if fig is None:
                fig = plt.figure(figsize=(16, height * 4))
                plt.subplots_adjust(hspace=0.3)
            else:
                plt.figure(fig)

            shape = (height, 16)
            bright_shape = (height, 18)
            # Main Panel
            if num_objects_to_plot == 1:
                ax = plt.subplot2grid(shape, (0, 0), rowspan=2+plot_brightness, colspan=8 - show_colorbar)
            else:
                ax = plt.subplot2grid(shape, (0, 0), rowspan=4, colspan=16 - show_colorbar*2 - num_cbars//2)

            # sep/PA/bright panels
            sep_axes = []
            pa_axes = []
            bright_axes = []
            if num_objects_to_plot == 1:
                sep_ax = plt.subplot2grid(shape, (0, 10), rowspan=1, colspan=6)
                pa_ax = plt.subplot2grid(shape, (1, 10), rowspan=1, colspan=6)
                if plot_brightness:
                    bright_ax = plt.subplot2grid(shape, (2, 10), rowspan=1, colspan=6)
                    bright_ax.set_ylabel("Brightness", fontsize=fontsize)
                    bright_ax.set_xlabel("Epoch", fontsize=fontsize)
                else:
                    bright_ax = None
                sep_ax.set_ylabel("$\\rho$ (mas)", fontsize=fontsize)
                pa_ax.set_ylabel("PA ($^{{\\circ}}$)", fontsize=fontsize)
                sep_ax.set_xlabel("Epoch", fontsize=fontsize)
                pa_ax.set_xlabel("Epoch", fontsize=fontsize)
                sep_axes.append(sep_ax)
                pa_axes.append(pa_ax)
                bright_axes.append(bright_ax)
            else:
                for i, object_index in enumerate(self.objects_to_plot):
                    if plot_brightness and len(self.bright_datas[i]) > 0:
                        sep_ax = plt.subplot2grid(bright_shape, (4+i, 0), rowspan=1, colspan=4)
                        pa_ax = plt.subplot2grid(bright_shape, (4+i, 6), rowspan=1, colspan=4)
                        bright_ax = plt.subplot2grid(bright_shape, (4+i, 12), rowspan=1, colspan=4)
                        bright_ax.set_ylabel("$Brightness_{{{0}}}$".format(self._get_object_label(object_index)), fontsize=fontsize)
                        bright_ax.set_xlabel("Epoch", fontsize=fontsize)
                    else:
                        sep_ax = plt.subplot2grid(shape, (4+i, 0), rowspan=1, colspan=7)
                        pa_ax = plt.subplot2grid(shape, (4+i, 9), rowspan=1, colspan=7)
                        bright_ax = None
                    sep_ax.set_ylabel("$\\rho_{{{0}}}$ (mas)".format(self._get_object_label(object_index)), fontsize=fontsize)
                    pa_ax.set_ylabel("PA$_{{{0}}}$ ($^{{\\circ}}$)".format(self._get_object_label(object_index)), fontsize=fontsize)
                    sep_ax.set_xlabel("Epoch", fontsize=fontsize)
                    pa_ax.set_xlabel("Epoch", fontsize=fontsize)

                    sep_axes.append(sep_ax)
                    pa_axes.append(pa_ax)
                    bright_axes.append(bright_ax)

            # rv panels
            ax3 = ax4 = None
            if rv_time_series:
                ax3 = plt.subplot2grid(shape, (height - 1 - rv_time_series2, 0), rowspan=1, colspan=16)
                ax3.set_ylabel("Primary RV (km/s)", fontsize=fontsize)
                ax3.set_xlabel("Epoch", fontsize=fontsize)
            if rv_time_series2:
                ax4 = plt.subplot2grid(shape, (height - 1, 0), rowspan=1, colspan=16)
                ax4.set_ylabel("Companion Relative RV (km/s)", fontsize=fontsize)
                ax4.set_xlabel("Epoch", fontsize=fontsize)
                if not rv_time_series:
                    ax3 = ax4
                    ax4 = None

            astr_colors = itertools.cycle(self.ASTR_COLORS)
            astr_symbols = itertools.cycle(self.ASTR_SYMBOLS)
            model_colors = itertools.cycle(self.MODEL_COLORS)
            cmaps_iter = itertools.cycle(cmaps)
            for object_i, object_index in enumerate(self.objects_to_plot):
                astr_color = next(astr_colors)
                if plot_errorbars:
                    err_color = next(astr_colors)
                else:
                    err_color = None
                self._plot_full_orbits(ax, plot_astrometry, full_plot, fontsize, next(cmaps_iter), plot_astrometry_insts, use_cmap, n_std, object_i, object_index, astr_color, astr_symbols, next(model_colors), err_color)
                self._plot_sep_pa_model(sep_axes[object_i], pa_axes[object_i], mod180, sep_pa_color, object_i, object_index)
                self._plot_sep_pa_data(sep_axes[object_i], pa_axes[object_i], plot_astrometry_insts, plot_errorbars, object_i, object_index)

                bright_ax = bright_axes[object_i]
                if bright_ax is not None:
                    self._plot_brightness_model(bright_ax, sep_pa_color, object_i, object_index)
                    self._plot_brightness_data(bright_ax, plot_astrometry_insts, plot_errorbars, object_i, object_index)

            if rv_time_series or rv_time_series2:
                self._plot_rv_model(ax3, ax4, rv_time_series, rv_time_series2, sep_pa_color)
                self._plot_rv_instruments(ax3, ax4, rv_time_series, rv_time_series2, rv_err_grouping, plot_errorbars)

            # add colorbar
            if show_colorbar:
                self._add_colorbar(ax, fig, cmaps, num_cbars)
            
            for ax1 in fig.get_axes():
                ax1.tick_params(axis="both", which="both", labelsize=15, top=True, right=True)
                ax1.minorticks_on()

        fig.tight_layout()
        
        return fig

    def plot_residuals(
            self,
            sep_pa_color="lightgrey",
            mod180=False,
            separate_error=False
        ):
        """
        Plots sep/PA residuals for a set of orbits

        Args:
            sep_pa_color (string): any valid matplotlib color string, used to set the
                color of the orbit tracks in the Sep/PA panels (default: 'lightgrey').
            mod180 (bool): if True, PA will be plotted in range [180, 540]. Useful for plotting short
                arcs with PAs that cross 360 deg during observations (default: False)
            separate_error (bool): separate the data error and error due to the standard deviation
                in the model subtracted from the data (default: False)

        Return:
            ``matplotlib.pyplot.Figure``: the residual plots

        Refactored by Eshel Dror, 2026        
    
        """

        fig, axes = plt.subplots(len(self.objects_to_plot), 2, figsize=(8, 4*len(self.objects_to_plot)))
        for object_index, object_to_plot in enumerate(self.objects_to_plot):
            object_axes = axes[object_index] if len(self.objects_to_plot) > 1 else axes
            seps = []
            pas = []
            seps_100 = []
            pas_100 = []
            for i in np.arange(self.num_orbits_to_plot):

                raoff0, deoff0 = self.astr_raoffs[object_index][i], self.astr_deoffs[object_index][i]

                raoff2, deoff2 = self.fixed_raoffs[object_to_plot, i, :], self.fixed_deoffs[object_to_plot, i, :]

                seps1, pas1 = orbitize.system.radec2seppa(raoff0, deoff0, mod180=mod180)

                seps.append(seps1)
                pas.append(pas1)

                seps2, pas2 = orbitize.system.radec2seppa(raoff2, deoff2, mod180=mod180)

                seps_100.append(seps2)
                pas_100.append(pas2)

            yr_epochs = Time(self.astr_epochs[object_index], format="mjd").decimalyear
            yr_epochs2 = Time(self.fixed_epochs, format="mjd").decimalyear

            seps = np.array(seps)
            pas = np.array(pas)
            seps_100 = np.array(seps_100)
            pas_100 = np.array(pas_100)

            median_seps_100 = np.median(seps_100, axis=0)
            median_pas_100 = np.median(pas_100, axis=0)
            
            median_seps = np.median(seps, axis=0)
            median_pas = np.median(pas, axis=0)
            stddev_seps = np.std(seps, axis=0)
            stddev_pas = np.std(pas, axis=0)

            residual_seps = median_seps - self.sep_datas[object_index]
            residual_pas = median_pas - self.pa_datas[object_index]
            
            if separate_error:
                object_axes[0].errorbar(
                    yr_epochs,
                    residual_seps,
                    yerr=self.sep_errs[object_index],
                    xerr=None,
                    fmt="o",
                    ms=5,
                    linestyle="",
                    c=self.ASTR_COLORS[0],
                    zorder=10,
                    capsize=2,
                    label="Data Error"
                )
                object_axes[0].errorbar(
                    yr_epochs,
                    residual_seps,
                    yerr=stddev_seps,
                    xerr=None,
                    ms=5,
                    linestyle="",
                    c=self.ASTR_COLORS[1],
                    zorder=10,
                    capsize=2,
                    label="Model Standard Deviation"
                )
                object_axes[0].legend()
            else:
                residual_seps_err = np.sqrt(self.sep_errs[object_index] ** 2 + stddev_seps ** 2)
                object_axes[0].errorbar(
                    yr_epochs,
                    residual_seps,
                    yerr=residual_seps_err,
                    xerr=None,
                    fmt="o",
                    ms=5,
                    linestyle="",
                    c=self.ASTR_COLORS[0],
                    zorder=10,
                    capsize=2,
                )

            for i in range(self.num_orbits_to_plot):
                residual_seps_100 = median_seps_100 - seps_100[i]
                object_axes[0].plot(yr_epochs2, residual_seps_100, color=sep_pa_color, zorder=1)
            object_axes[0].axhline(y=0, color="black", linestyle="-")
            if len(self.objects_to_plot) > 1:
                object_axes[0].set_ylabel("Residual $\\rho_{{{0}}}$ [mas]".format(self._get_object_label(object_index)))
            else:
                object_axes[0].set_ylabel("Residual $\\rho$ [mas]")
            object_axes[0].set_xlabel("Epoch")
            object_axes[0].set_xlim(yr_epochs2[0], yr_epochs2[-1])

            if separate_error:
                object_axes[1].errorbar(
                    yr_epochs,
                    residual_pas,
                    yerr=self.pa_errs[object_index],
                    xerr=None,
                    fmt="o",
                    ms=5,
                    linestyle="",
                    c=self.ASTR_COLORS[0],
                    zorder=10,
                    capsize=2,
                    label="Data Error"
                )
                object_axes[1].errorbar(
                    yr_epochs,
                    residual_pas,
                    yerr=stddev_pas,
                    xerr=None,
                    ms=5,
                    linestyle="",
                    c=self.ASTR_COLORS[1],
                    zorder=10,
                    capsize=2,
                    label="Model Standard Deviation"
                )
                object_axes[1].legend()
            else:
                residual_pa_err = np.sqrt(self.pa_errs[object_index] ** 2 + stddev_pas ** 2)
                object_axes[1].errorbar(
                    yr_epochs,
                    residual_pas,
                    yerr=residual_pa_err,
                    xerr=None,
                    fmt="o",
                    ms=5,
                    linestyle="",
                    c=self.ASTR_COLORS[0],
                    zorder=10,
                    capsize=2,
                )
            for i in range(self.num_orbits_to_plot):
                residual_pas_100 = median_pas_100 - pas_100[i]
                object_axes[1].plot(yr_epochs2, residual_pas_100, color=sep_pa_color, zorder=1)
            object_axes[1].axhline(y=0, color="black", linestyle="-")
            if len(self.objects_to_plot) > 1:
                object_axes[1].set_ylabel("Residual PA$_{{{0}}}$ [$^{{\\circ}}$]".format(self._get_object_label(object_index)))
            else:
                object_axes[1].set_ylabel("Residual PA [$^{{\\circ}}$]")
            object_axes[1].set_xlabel("Epoch")
            object_axes[1].set_xlim(yr_epochs2[0], yr_epochs2[-1])
            for ax in object_axes:
                ax.tick_params(axis="both", which="both", top=True, right=True)
                ax.minorticks_on()
        plt.tight_layout()

        return fig

    def plot_propermotion(
        self,
        periods_to_plot=1,
        alpha=0.05,
        show_colorbar=True,
        cmap=None,
        tight_layout=False,
    ):
        """
        Plots the proper motion of a host star as induced by a companion for
        a number of orbital periods for a select number of fitted orbits
        for a given object, with line segments colored according to a given
        parameter (most informative is usually mass of companion)

        Important Note: These plotted trajectories aren't what are fitting in the
        likelihood evaluation for the HGCA runs. The implementation forward models
        the Hip/Gaia measurements per epoch and infers the differential proper motions.
        This plot is given only for the purposes of an approximate visualization.

        Note: The `orbitize.results.Results` object used when initializing the `Plotter` must have
            an orbitize.system object with a HGCALogProb passed to system.gaia

        Args:
            periods_to_plot (int): number of periods to plot (default: 1)
            alpha (float): transparency of lines (default: 0.05)
            show_colorbar (bool): Displays colorbar to the right of the plot (default: True)
            cmap (matplotlib.cm.ColorMap): color map to use for making orbit tracks
                (default: None (uses `self.CMAP`))
            tight_layout (bool): apply plt.tight_layout function (default: False)

        Return:
            ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


        (written): William Balmer (2023), based on plot_orbits by Sarah Blunt and Henry Ngo
        Refactored by Eshel Dror, 2026

        """
        if cmap is None:
            cmap = self.CMAP

        if not isinstance(cmap, colors.Colormap):
            cmap = cmap[0]

        object_to_plot = self.objects_to_plot[0] # TODO: All objects?

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)

            if periods_to_plot == 1:
                raoff = self.period_raoffs[object_to_plot]
                deoff = self.period_deoffs[object_to_plot]
                epochs = self.period_epochss[object_to_plot]
            else:
                raoff, deoff, _, epochs = self._calc_object_full_orbits(
                self.start, self.num_orbits_to_plot, self.num_epochs_to_plot, object_to_plot, self.standard_post, periods_to_plot)


            # Create figure for orbit plots
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="white")

            # Plot each orbit (each segment between two points coloured using colormap)
            for i in np.arange(self.num_orbits_to_plot):
                epoch_in_yr = Time(epochs[i, :], format="mjd").decimalyear
                # masses (in same units, solar)
                m_b = self.standard_post[:, self.results.param_idx["m1"]][i]
                m_a = self.standard_post[:, self.results.param_idx["m0"]][i]
                # dt
                timestep = epoch_in_yr[1] - epoch_in_yr[0]
                # dra/dt and ddec/dt
                ddec_b = np.gradient(deoff[i, :], timestep)  # in mas/yr
                dec_b_radian = (
                    deoff[i, :] * u.mas
                ).to(u.rad).value  # mas -> radian
                ra_b = raoff[i, :]
                rastar_b = ra_b * np.cos(dec_b_radian)  # in mas
                drastar_b = np.gradient(rastar_b, timestep)  # in mas/yr

                # convert to dRA^star_star (lol) and dDec_star
                mass_ratio_ = -1 * m_b / (m_a + m_b)
                ddec_a = ddec_b * mass_ratio_
                drastar_a = drastar_b * mass_ratio_

                if self.cbar_param is not None and self.cbar_param not in ["Epoch [year]", "Epoch (year)"]:
                    color = cmap(self.norm(self.standard_post[:, self.results.param_idx[self.cbar_param]][i]))
                else:
                    color = "k"

                axes[0].plot(
                    epoch_in_yr,
                    drastar_a + self.system.gaia.hg_pm[0],
                    color=color,
                    alpha=alpha,
                    zorder=0,
                )
                axes[1].plot(
                    epoch_in_yr,
                    ddec_a + self.system.gaia.hg_pm[1],
                    color=color,
                    alpha=alpha,
                    zorder=0,
                )

        axes[0].set_xlim(self.start.decimalyear, self.end.decimalyear)
        axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axes[1].set_xlabel("Epoch")

        axes[0].set_ylabel(r"$\mu_\alpha^*$ [mas/yr]")

        axes[0].errorbar(
            np.nanmedian(self.system.gaia.hipparcos_epoch),
            self.system.gaia.hip_pm[0],
            yerr=self.system.gaia.hip_pm_err[0],
            zorder=30,
            mec="k",
            fmt="s",
            color="cornflowerblue",
        )

        hgca_epoch = (
            self.system.gaia.gaia_epoch_ra + np.nanmedian(self.system.gaia.hipparcos_epoch)
        ) / 2
        hgca_epoch_err = (
            self.system.gaia.gaia_epoch_ra - np.nanmedian(self.system.gaia.hipparcos_epoch)
        ) / 2

        axes[0].errorbar(
            hgca_epoch,
            self.system.gaia.hg_pm[0],
            xerr=hgca_epoch_err,
            yerr=self.system.gaia.hg_pm_err[0],
            zorder=30,
            mec="k",
            fmt="^",
            color="#6280D6",
        )

        axes[0].errorbar(
            self.system.gaia.gaia_epoch_ra,
            self.system.gaia.gaia_pm[0],
            yerr=self.system.gaia.gaia_pm_err[0],
            zorder=30,
            mec="k",
            fmt="o",
            color="#5f61b4",
        )

        axes[1].set_xlim(self.start.decimalyear, self.end.decimalyear)
        axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        axes[1].errorbar(
            np.nanmedian(self.system.gaia.hipparcos_epoch),
            self.system.gaia.hip_pm[1],
            yerr=self.system.gaia.hip_pm_err[1],
            zorder=30,
            mec="k",
            fmt="s",
            color="cornflowerblue",
            label="Hip.",
        )

        axes[1].errorbar(
            hgca_epoch,
            self.system.gaia.hg_pm[1],
            xerr=hgca_epoch_err,
            yerr=self.system.gaia.hg_pm_err[1],
            zorder=30,
            mec="k",
            fmt="^",
            color="#6280D6",
            label="H-G",
        )

        axes[1].errorbar(
            self.system.gaia.gaia_epoch_ra,
            self.system.gaia.gaia_pm[1],
            yerr=self.system.gaia.gaia_pm_err[1],
            zorder=30,
            mec="k",
            fmt="o",
            color="#5f61b4",
            label="Gaia",
        )

        axes[1].set_ylabel(r"$\mu_\delta$ [mas/yr]")
        axes[1].set_xlabel("Epoch")
        axes[0].set_xlabel("Epoch")

        if show_colorbar:
            cbar_ax = fig.add_axes([1.03, 0.15, 0.03, 0.80])

            cbar = mpl.colorbar.ColorbarBase(
                cbar_ax, cmap=cmap, norm=self.norm, orientation="vertical", label=self.cbar_param
            )

        axes[0].set_rasterization_zorder(1)
        axes[1].set_rasterization_zorder(1)

        axes[1].legend()

        for ax in axes:
            ax.tick_params(axis="both", which="both", top=True, right=True)
            ax.minorticks_on()

        print(
            "Important Note of Caution: the orbitize! implementation of the HGCA \n",
            "fits for the time-averaged proper motions, and not the instantaneous proper \n",
            "motions that are being plotted here. This plot is provided only for the \n",
            "purpose of an approximate check on the fit.",
        )

        if tight_layout:
            plt.tight_layout()

        return fig

    def plot_corner(self, param_list=None, plot_priors=True, **corner_kwargs):
        """
        Wrapper for `orbitize.plot.plot_corner`
        """
        return plot_corner(self.results, param_list, plot_priors, **corner_kwargs)

def plot_corner(results, param_list=None, plot_priors=True, **corner_kwargs):
    """
    Make a corner plot of posterior on orbit fit from any sampler

    Args:
        param_list (list of strings): each entry is a name of a parameter to include.
            Valid strings::

                sma1: semimajor axis
                ecc1: eccentricity
                inc1: inclination
                aop1: argument of periastron
                pan1: position angle of nodes
                tau1: epoch of periastron passage, expressed as fraction of orbital period
                per1: period
                K1: stellar radial velocity semi-amplitude
                [repeat for 2, 3, 4, etc if multiple objects]
                plx:  parallax
                pm_ra: RA proper motion
                pm_dec: Dec proper motion
                alpha0: primary offset from reported Hipparcos RA @ alphadec0_epoch (generally 1991.25)
                delta0: primary offset from reported Hipparcos Dec @ alphadec0_epoch (generally 1991.25)
                gamma: rv offset
                sigma: rv jitter
                mi: mass of individual body i, for i = 0, 1, 2, ... (only if fit_secondary_mass)
                mtot: total mass (only if fit_secondary_mass == False)
        plot priors (bool): overplot prior probabilites on the 1d histograms (default: True)

        **corner_kwargs: any remaining keyword args are sent to ``corner.corner``.
                            See `here <https://corner.readthedocs.io/>`_.
                            Note: default axis labels used unless overwritten by user input.

    Return:
        ``matplotlib.pyplot.Figure``: corner plot

    .. Note:: **Example**: Use ``param_list = ['sma1,ecc1,inc1,sma2,ecc2,inc2']`` to only
        plot posteriors for semimajor axis, eccentricity and inclination
        of the first two companions

    Written: Henry Ngo, 2018
    Additions: Eshel Dror, 2026
    """

    # Define array of default axis labels (overwritten if user specifies list)
    default_labels = {

        "sma": "$a_{0}$ [au]",
        "ecc": "$ecc_{0}$",
        "inc": "$inc_{0}$ [$^\\circ$]",
        "aop": "$\\omega_{0}$ [$^\\circ$]",
        "pan": "$\\Omega_{0}$ [$^\\circ$]",
        "tau": "$\\tau_{0}$",
        "tp": "$T_{{\\mathrm{{P}}}}$",
        "plx": "$\\pi$ [mas]",
        "gam": "$\\gamma$ [km/s]",
        "sig": "$\\sigma$ [km/s]",
        "mtot": "$M_T$ [M$_{{\\odot}}$]",
        "m0": "$M_0$ [M$_{{\\odot}}$]",
        "m": "$M_{0}$ [M$_{{\\rm Jup}}$]",
        "pm_ra": "$\\mu_{{\\alpha}}$ [mas/yr]",
        "pm_dec": "$\\mu_{{\\delta}}$ [mas/yr]",
        "alpha0": "$\\alpha^{{*}}_{{0}}$ [mas]",
        "delta0": "$\\delta_0$ [mas]",
        "m": "$M_{0}$ [M$_{{\\rm Jup}}$]",
        "per": "$P_{0}$ [yr]",
        "K": "$K_{0}$ [km/s]",
        "x": "$X_{0}$ [AU]",
        "y": "$Y_{0}$ [AU]",
        "z": "$Z_{0}$ [AU]",
        "xdot": "$xdot_{0}$ [km/s]",
        "ydot": "$ydot_{0}$ [km/s]",
        "zdot": "$zdot_{0}$ [km/s]",
    }

    if param_list is None:
        param_list = results.labels

    param_indices = []
    angle_indices = []
    secondary_mass_indices = []
    fixed_indices = []
    for i, label_key in enumerate(param_list):
        index_num = results.param_idx[label_key]

        # only plot non-fixed parameters
        if not np.isclose(0.0, np.std(results.post[:, index_num])):
            param_indices.append(index_num)
            if (
                label_key.startswith("aop")
                or label_key.startswith("pan")
                or label_key.startswith("inc")
            ):
                angle_indices.append(i-len(fixed_indices))
            if label_key.startswith("m") and label_key != "m0" and label_key != "mtot":
                secondary_mass_indices.append(i-len(fixed_indices))
        else:
            fixed_indices.append(i)

    samples = np.copy(
        results.post[:, param_indices]
    )  # keep only chains for selected parameters
    samples[:, angle_indices] = np.degrees(
        samples[:, angle_indices]
    )  # convert angles from rad to deg
    samples[:, secondary_mass_indices] *= u.solMass.to(
        u.jupiterMass
    )  # convert to Jupiter masses for companions

    if (
        "labels" not in corner_kwargs
    ):  # use default labels if user didn't already supply them
        reduced_labels_list = []
        for i in range(len(param_indices)):
            label_key = param_list[i]
            if label_key.startswith("m") and label_key != "m0" and label_key != "mtot":
                body_num = label_key[1]
                label_key = "m"
            elif (
                label_key == "m0" or label_key == "mtot" or label_key.startswith("plx")
            ):
                body_num = ""
                # maintain original label key
            elif label_key in ["pm_ra", "pm_dec", "alpha0", "delta0"]:
                body_num = ""
            elif label_key.startswith("gamma") or label_key.startswith("sigma"):
                body_num = ""
                label_key = label_key[0:3]
            else:
                body_num = label_key[-1]
                label_key = label_key[0:-1]
            reduced_labels_list.append(default_labels[label_key].format(body_num))

        corner_kwargs["labels"] = reduced_labels_list

    if plot_priors:
        hist_kwargs = corner_kwargs.get("hist_kwargs", {})
        hist_kwargs["density"] = True
        corner_kwargs["hist_kwargs"] = hist_kwargs

    figure = corner.corner(samples, **corner_kwargs)

    if plot_priors:
        axes = figure.axes
        num_params = len(param_indices)
        for i, param_i in enumerate(param_indices):
            prior = results.system.sys_priors[param_i]
            if not hasattr(prior, "compute_lnprob"):
                continue
            ax = axes[i * (num_params+1)]
            if i in angle_indices:
                dmin, dmax = np.radians(ax.dataLim.intervalx)
                x = np.linspace(dmin, dmax, 10000)
                x_plot = np.degrees(x)
            elif i in secondary_mass_indices:
                dmin, dmax = ax.dataLim.intervalx / u.solMass.to(u.jupiterMass)
                x = np.linspace(dmin, dmax, 10000)
                x_plot = x * u.solMass.to(u.jupiterMass)
            else:
                dmin, dmax = ax.dataLim.intervalx
                x_plot = x = np.linspace(dmin, dmax, 10000)
            y = np.exp(prior.compute_lnprob(x))
            if y.shape != x.shape: # Some priors may require specific input procedures (ObsPrior)
                warnings.warn("Could not compute prior probability for {0}".format(prior))
                continue
            if i in angle_indices:
                y_plot = y / 180 * np.pi
            elif i in secondary_mass_indices:
                y_plot = y / u.solMass.to(u.jupiterMass)
            else:
                y_plot = y
            ax.plot(x_plot, y_plot, color="orange")

    return figure

def plot_orbits(
    results,
    object_to_plot=1,
    start_mjd=51544.0,
    num_orbits_to_plot=100,
    num_epochs_to_plot=100,
    square_plot=True,
    show_colorbar=True,
    cmap=None,
    sep_pa_color="lightgrey",
    sep_pa_end_year=2025.0,
    cbar_param="Epoch [year]",
    mod180=False,
    rv_time_series=False,
    plot_astrometry=True,
    plot_astrometry_insts=False,
    plot_errorbars=True,
    rv_time_series2=False,
    primary_instrument_name=None,
    fontsize=20,
    fig=None,
):
    warnings.warn("The orbitize.plot.plot_orbits function is deprecated. Instead, initialize an orbitize.plot.Plotter and use the plot_orbits method.")
    plotter = Plotter(
        results=results,
        object_to_plot=object_to_plot,
        start=Time(start_mjd, format="mjd").decimalyear,
        end=sep_pa_end_year,
        time_format="decimalyear",
        num_orbits_to_plot=num_orbits_to_plot,
        num_epochs_to_plot=num_epochs_to_plot,
        cbar_param=cbar_param,
        rv_time_series=rv_time_series,
        rv_time_series2=rv_time_series2
        )
    if cmap is not None:
        plotter.CMAP = cmap
    return plotter.plot_orbits(
        full_plot=square_plot,
        show_colorbar=show_colorbar,
        sep_pa_color=sep_pa_color,
        mod180=mod180,
        plot_astrometry=plot_astrometry,
        plot_astrometry_insts=plot_astrometry_insts,
        plot_errorbars=plot_errorbars,
        rv_time_series=rv_time_series,
        rv_time_series2=rv_time_series2,
        fontsize=fontsize,
        fig=fig
    )

def plot_residuals(
    my_results,
    object_to_plot=1,
    start_mjd=51544,
    num_orbits_to_plot=100,
    num_epochs_to_plot=100,
    sep_pa_color="lightgrey",
    sep_pa_end_year=2025.0,
    cbar_param="Epoch [year]",
    mod180=False,
):
    warnings.warn("The orbitize.plot.plot_residuals function is deprecated. Instead, initialize an orbitize.plot.Plotter and use the plot_residuals method.")
    plotter = Plotter(
        results=my_results,
        object_to_plot=object_to_plot,
        start=Time(start_mjd, format="mjd").decimalyear,
        end=sep_pa_end_year,
        time_format="decimalyear",
        num_orbits_to_plot=num_orbits_to_plot,
        num_epochs_to_plot=num_epochs_to_plot,
        cbar_param=cbar_param
    )
    return plotter.plot_residuals(
        sep_pa_color=sep_pa_color,
        mod180=mod180
    )

def plot_propermotion(
    results,
    system,
    object_to_plot=1,
    start_mjd=44239.0,
    periods_to_plot=1,
    end_year=2030.0,
    alpha=0.05,
    num_orbits_to_plot=100,
    num_epochs_to_plot=100,
    show_colorbar=True,
    cmap=None,
    cbar_param="m0",
    tight_layout=False,
):
    warnings.warn("The orbitize.plot.plot_propermotion function is deprecated. Instead, initialize an orbitize.plot.Plotter and use the plot_propermotion method.")
    plotter = Plotter(
        results=results,
        object_to_plot=object_to_plot,
        start=Time(start_mjd, format="mjd").decimalyear,
        end=end_year,
        time_format="decimalyear",
        num_orbits_to_plot=num_orbits_to_plot,
        num_epochs_to_plot=num_epochs_to_plot,
        cbar_param=cbar_param
    )
    if cmap is not None:
        plotter.CMAP = cmap
    return plotter.plot_propermotion(
        periods_to_plot=periods_to_plot,
        alpha=alpha,
        show_colorbar=show_colorbar,
        tight_layout=tight_layout
    )
