import numpy as np
import corner
import warnings
import itertools
import string

import astropy.units as u
import astropy.constants as consts
from astropy.time import Time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

from erfa import ErfaWarning

import orbitize
import orbitize.kepler as kepler
import orbitize.results


# TODO: deprecatation warning for plots in results

# define modified color map for default use in orbit plots
cmap = mpl.cm.Purples_r
cmap = colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.0, b=0.7),
    cmap(np.linspace(0.0, 0.7, 1000))
)

def plot_corner(results, param_list=None, **corner_kwargs):
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
                mi: mass of individual body i, for i = 0, 1, 2, ... (only if fit_secondary_mass == True)
                mtot: total mass (only if fit_secondary_mass == False)

        **corner_kwargs: any remaining keyword args are sent to ``corner.corner``.
                            See `here <https://corner.readthedocs.io/>`_.
                            Note: default axis labels used unless overwritten by user input.

    Return:
        ``matplotlib.pyplot.Figure``: corner plot

    .. Note:: **Example**: Use ``param_list = ['sma1,ecc1,inc1,sma2,ecc2,inc2']`` to only
        plot posteriors for semimajor axis, eccentricity and inclination
        of the first two companions

    Written: Henry Ngo, 2018
    """

    # Define array of default axis labels (overwritten if user specifies list)
    default_labels = {
        'sma': '$a_{0}$ [au]',
        'ecc': '$ecc_{0}$',
        'inc': '$inc_{0}$ [$^\\circ$]',
        'aop': '$\\omega_{0}$ [$^\\circ$]',
        'pan': '$\\Omega_{0}$ [$^\\circ$]',
        'tau': '$\\tau_{0}$',
        'plx': '$\\pi$ [mas]',
        'gam': '$\\gamma$ [km/s]',
        'sig': '$\\sigma$ [km/s]',
        'mtot': '$M_T$ [M$_{{\\odot}}$]',
        'm0': '$M_0$ [M$_{{\\odot}}$]',
        'm': '$M_{0}$ [M$_{{\\rm Jup}}$]',
        'pm_ra': '$\\mu_{{\\alpha}}$ [mas/yr]',
        'pm_dec': '$\\mu_{{\\delta}}$ [mas/yr]',
        'alpha0': '$\\alpha^{{*}}_{{0}}$ [mas]',
        'delta0': '$\\delta_0$ [mas]',
        'm': '$M_{0}$ [M$_{{\\rm Jup}}$]',
        'per' : '$P_{0}$ [yr]',
        'K' : '$K_{0}$ [km/s]',
        'x' : '$X_{0}$ [AU]',
        'y' : '$Y_{0}$ [AU]',
        'z' : '$Z_{0}$ [AU]',
        'xdot' : '$xdot_{0}$ [km/s]',
        'ydot' : '$ydot_{0}$ [km/s]',
        'zdot' : '$zdot_{0}$ [km/s]'
    }

    if param_list is None:
        param_list = results.labels

    param_indices = []
    angle_indices = []
    secondary_mass_indices = []
    for i, param in enumerate(param_list):
        index_num = results.param_idx[param]

        # only plot non-fixed parameters
        if np.std(results.post[:, index_num]) > 0:
            param_indices.append(index_num)
            label_key = param
            if label_key.startswith('aop') or label_key.startswith('pan') or label_key.startswith('inc'):
                angle_indices.append(i)
            if label_key.startswith('m') and label_key != 'm0' and label_key != 'mtot':
                secondary_mass_indices.append(i)

    samples = np.copy(results.post[:, param_indices])  # keep only chains for selected parameters
    samples[:, angle_indices] = np.degrees(
        samples[:, angle_indices])  # convert angles from rad to deg
    samples[:, secondary_mass_indices] *= u.solMass.to(u.jupiterMass) # convert to Jupiter masses for companions

    if 'labels' not in corner_kwargs:  # use default labels if user didn't already supply them
        reduced_labels_list = []
        for i in np.arange(len(param_indices)):
            label_key = param_list[i]
            if label_key.startswith("m") and label_key != 'm0' and label_key != 'mtot':
                body_num = label_key[1]
                label_key = "m"
            elif label_key == 'm0' or label_key == 'mtot' or label_key.startswith('plx'):
                body_num = ""
                # maintain original label key
            elif label_key in ['pm_ra', 'pm_dec', 'alpha0', 'delta0']:
                body_num = ""
            elif label_key.startswith("gamma") or label_key.startswith("sigma"):
                body_num = ""
                label_key = label_key[0:3]
            else:
                body_num = label_key[-1]
                label_key = label_key[0:-1]
            reduced_labels_list.append(default_labels[label_key].format(body_num))

        corner_kwargs['labels'] = reduced_labels_list

    figure = corner.corner(samples, **corner_kwargs)
    return figure

def plot_orbits(results, object_to_plot=1, start_mjd=51544.,
                num_orbits_to_plot=100, num_epochs_to_plot=100,
                square_plot=True, show_colorbar=True, cmap=cmap,
                sep_pa_color='lightgrey', sep_pa_end_year=2025.0,
                cbar_param='Epoch [year]', mod180=False, rv_time_series=False, plot_astrometry=True,
                plot_astrometry_insts=False, plot_errorbars=True, fig=None):
    """
    Plots one orbital period for a select number of fitted orbits
    for a given object, with line segments colored according to time

    Args:
        object_to_plot (int): which object to plot (default: 1)
        start_mjd (float): MJD in which to start plotting orbits (default: 51544,
            the year 2000)
        num_orbits_to_plot (int): number of orbits to plot (default: 100)
        num_epochs_to_plot (int): number of points to plot per orbit (default: 100)
        square_plot (Boolean): Aspect ratio is always equal, but if
            square_plot is True (default), then the axes will be square,
            otherwise, white space padding is used
        show_colorbar (Boolean): Displays colorbar to the right of the plot [True]
        cmap (matplotlib.cm.ColorMap): color map to use for making orbit tracks
            (default: modified Purples_r)
        sep_pa_color (string): any valid matplotlib color string, used to set the
            color of the orbit tracks in the Sep/PA panels (default: 'lightgrey').
        sep_pa_end_year (float): decimal year specifying when to stop plotting orbit
            tracks in the Sep/PA panels (default: 2025.0).
        cbar_param (string): options are the following: 'Epoch [year]', 'sma1', 'ecc1', 'inc1', 'aop1',
            'pan1', 'tau1', 'plx. Number can be switched out. Default is Epoch [year].
        mod180 (Bool): if True, PA will be plotted in range [180, 540]. Useful for plotting short
            arcs with PAs that cross 360 deg during observations (default: False)
        rv_time_series (Boolean): if fitting for secondary mass using MCMC for rv fitting and want to
            display time series, set to True.
        plot_astrometry (Boolean): set to True by default. Plots the astrometric data.
        plot_astrometry_insts (Boolean): set to False by default. Plots the astrometric data by instruments.
        plot_errorbars (Boolean): set to True by default. Plots error bars of measurements
        fig (matplotlib.pyplot.Figure): optionally include a predefined Figure object to plot the orbit on.
            Most users will not need this keyword. 

    Return:
        ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


    (written): Henry Ngo, Sarah Blunt, 2018
    Additions by Malena Rice, 2019

    """

    if Time(start_mjd, format='mjd').decimalyear >= sep_pa_end_year:
        raise ValueError('start_mjd keyword date must be less than sep_pa_end_year keyword date.')

    if object_to_plot > results.num_secondary_bodies:
        raise ValueError("Only {0} secondary bodies being fit. Requested to plot body {1} which is out of range".format(results.num_secondary_bodies, object_to_plot))

    if object_to_plot == 0:
        raise ValueError("Plotting the primary's orbit is currently unsupported. Stay tuned.")

    if rv_time_series and 'm0' not in results.labels:
        rv_time_series = False

        warnings.warn("It seems that the stellar and companion mass "
                      "have not been fitted separately. Setting "
                      "rv_time_series=True is therefore not possible "
                      "so the argument is set to False instead.")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ErfaWarning)

        data = results.data[results.data['object'] == object_to_plot]
        possible_cbar_params = [
            'sma',
            'ecc',
            'inc',
            'aop'
            'pan',
            'tau',
            'plx'
        ]

        if cbar_param == 'Epoch [year]':
            pass
        elif cbar_param[0:3] in possible_cbar_params:
            index = results.param_idx[cbar_param]
        else:
            raise Exception(
                "Invalid input; acceptable inputs include 'Epoch [year]', 'plx', 'sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'sma2', 'ecc2', ...)"
            )
        # Select random indices for plotted orbit
        num_orbits = len(results.post[:, 0])
        if num_orbits_to_plot > num_orbits:
            num_orbits_to_plot = num_orbits
        choose = np.random.randint(0, high=num_orbits, size=num_orbits_to_plot)

        # Get posteriors from random indices
        standard_post = []
        if results.sampler_name == 'MCMC':
            # Convert the randomly chosen posteriors to standard keplerian set
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                param_set = np.copy(results.post[orb_ind])
                standard_post.append(results.basis.to_standard_basis(param_set))
        else: # For OFTI, posteriors are already converted
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                standard_post.append(results.post[orb_ind])

        standard_post = np.array(standard_post)

        sma = standard_post[:, results.standard_param_idx['sma{}'.format(object_to_plot)]]
        ecc = standard_post[:, results.standard_param_idx['ecc{}'.format(object_to_plot)]]
        inc = standard_post[:, results.standard_param_idx['inc{}'.format(object_to_plot)]]
        aop = standard_post[:, results.standard_param_idx['aop{}'.format(object_to_plot)]]
        pan = standard_post[:, results.standard_param_idx['pan{}'.format(object_to_plot)]]
        tau = standard_post[:, results.standard_param_idx['tau{}'.format(object_to_plot)]]
        plx = standard_post[:, results.standard_param_idx['plx']]

        # Then, get the other parameters
        if 'mtot' in results.labels:
            mtot = standard_post[:, results.standard_param_idx['mtot']]
        elif 'm0' in results.labels:
            m0 = standard_post[:, results.standard_param_idx['m0']]
            m1 = standard_post[:, results.standard_param_idx['m{}'.format(object_to_plot)]]
            mtot = m0 + m1

        raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        vz_star = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
        # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
        for i in np.arange(num_orbits_to_plot):
            # Compute period (from Kepler's third law)
            period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
            period = period.to(u.day).value

            # Create an epochs array to plot num_epochs_to_plot points over one orbital period
            epochs[i, :] = np.linspace(start_mjd, float(
                start_mjd+period[i]), num_epochs_to_plot)

            # Calculate ra/dec offsets for all epochs of this orbit
            raoff0, deoff0, _ = kepler.calc_orbit(
                epochs[i, :], sma[i], ecc[i], inc[i], aop[i], pan[i],
                tau[i], plx[i], mtot[i], tau_ref_epoch=results.tau_ref_epoch
            )

            raoff[i, :] = raoff0
            deoff[i, :] = deoff0

        # Create a linearly increasing colormap for our range of epochs
        if cbar_param != 'Epoch [year]':
            cbar_param_arr = results.post[:, index]
            norm = mpl.colors.Normalize(vmin=np.min(cbar_param_arr),
                                        vmax=np.max(cbar_param_arr))
            norm_yr = mpl.colors.Normalize(vmin=np.min(
                cbar_param_arr), vmax=np.max(cbar_param_arr))

        elif cbar_param == 'Epoch [year]':

            min_cbar_date = np.min(epochs)
            max_cbar_date = np.max(epochs[-1, :])

            # if we're plotting orbital periods greater than 1,000 yrs, limit the colorbar dynamic range
            if max_cbar_date - min_cbar_date > 1000 * 365.25:
                max_cbar_date = min_cbar_date + 1000 * 365.25

            norm = mpl.colors.Normalize(vmin=min_cbar_date, vmax=max_cbar_date)

            norm_yr = mpl.colors.Normalize(
                vmin=Time(min_cbar_date, format='mjd').decimalyear,
                vmax=Time(max_cbar_date, format='mjd').decimalyear
            )

        # Before starting to plot rv data, make sure rv data exists:
        rv_indices = np.where(data['quant_type'] == 'rv')
        if rv_time_series and len(rv_indices) == 0:
            warnings.warn("Unable to plot radial velocity data.")
            rv_time_series = False

        # Create figure for orbit plots
        if fig is None:
            fig = plt.figure(figsize=(14, 6))
            if rv_time_series:
                fig = plt.figure(figsize=(14, 9))
                ax = plt.subplot2grid((3, 14), (0, 0), rowspan=2, colspan=6)
            else:
                fig = plt.figure(figsize=(14, 6))
                ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)
        else:
            plt.set_current_figure(fig)
            if rv_time_series:
                ax = plt.subplot2grid((3, 14), (0, 0), rowspan=2, colspan=6)
            else:
                ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)
        
        astr_inds=np.where((~np.isnan(data['quant1'])) & (~np.isnan(data['quant2'])))
        astr_epochs=data['epoch'][astr_inds]

        radec_inds = np.where(data['quant_type'] == 'radec')
        seppa_inds = np.where(data['quant_type'] == 'seppa')

        # transform RA/Dec points to Sep/PA
        sep_data = np.copy(data['quant1'])
        sep_err = np.copy(data['quant1_err'])
        pa_data = np.copy(data['quant2'])
        pa_err = np.copy(data['quant2_err'])

        if len(radec_inds[0] > 0):

            sep_from_ra_data, pa_from_dec_data = orbitize.system.radec2seppa(
                data['quant1'][radec_inds], data['quant2'][radec_inds]
            )

            num_radec_pts = len(radec_inds[0])
            sep_err_from_ra_data = np.empty(num_radec_pts)
            pa_err_from_dec_data = np.empty(num_radec_pts)
            for j in np.arange(num_radec_pts):

                sep_err_from_ra_data[j], pa_err_from_dec_data[j], _ = orbitize.system.transform_errors(
                    np.array(data['quant1'][radec_inds][j]), np.array(data['quant2'][radec_inds][j]), 
                    np.array(data['quant1_err'][radec_inds][j]), np.array(data['quant2_err'][radec_inds][j]), 
                    np.array(data['quant12_corr'][radec_inds][j]), orbitize.system.radec2seppa
                )

            sep_data[radec_inds] = sep_from_ra_data
            sep_err[radec_inds] = sep_err_from_ra_data

            pa_data[radec_inds] = pa_from_dec_data
            pa_err[radec_inds] = pa_err_from_dec_data

        # Transform Sep/PA points to RA/Dec
        ra_data = np.copy(data['quant1'])
        ra_err = np.copy(data['quant1_err'])
        dec_data = np.copy(data['quant2'])
        dec_err = np.copy(data['quant2_err'])

        if len(seppa_inds[0] > 0):

            ra_from_seppa_data, dec_from_seppa_data = orbitize.system.seppa2radec(
                data['quant1'][seppa_inds], data['quant2'][seppa_inds]
            )

            num_seppa_pts = len(seppa_inds[0])
            ra_err_from_seppa_data = np.empty(num_seppa_pts)
            dec_err_from_seppa_data = np.empty(num_seppa_pts)
            for j in np.arange(num_seppa_pts):

                ra_err_from_seppa_data[j], dec_err_from_seppa_data[j], _ = orbitize.system.transform_errors(
                    np.array(data['quant1'][seppa_inds][j]), np.array(data['quant2'][seppa_inds][j]), 
                    np.array(data['quant1_err'][seppa_inds][j]), np.array(data['quant2_err'][seppa_inds][j]), 
                    np.array(data['quant12_corr'][seppa_inds][j]), orbitize.system.seppa2radec
                )

            ra_data[seppa_inds] = ra_from_seppa_data
            ra_err[seppa_inds] = ra_err_from_seppa_data

            dec_data[seppa_inds] = dec_from_seppa_data
            dec_err[seppa_inds] = dec_err_from_seppa_data

        # For plotting different astrometry instruments
        if plot_astrometry_insts:
            astr_colors = ('purple','#FF7F11', '#11FFE3', '#14FF11', '#7A11FF', '#FF1919')
            astr_symbols = ( 'o', '*', 'p', 's')

            ax_colors = itertools.cycle(astr_colors)
            ax_symbols = itertools.cycle(astr_symbols)

            astr_data = data[astr_inds]
            astr_insts = np.unique(data[astr_inds]['instrument'])

            # Indices corresponding to each instrument in datafile
            astr_inst_inds = {}
            for i in range(len(astr_insts)):
                astr_inst_inds[astr_insts[i]]=np.where(astr_data['instrument']==astr_insts[i].encode())[0]

        # Plot each orbit (each segment between two points coloured using colormap)
        for i in np.arange(num_orbits_to_plot):
            points = np.array([raoff[i, :], deoff[i, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments, cmap=cmap, norm=norm, linewidth=1.0
            )
            if cbar_param != 'Epoch [year]':
                lc.set_array(np.ones(len(epochs[0]))*cbar_param_arr[i])
            elif cbar_param == 'Epoch [year]':
                lc.set_array(epochs[i, :])
            ax.add_collection(lc)

        # if plot_astrometry:

        #     # Plot astrometry along with instruments
        #     if plot_astrometry_insts:
        #         for i in range(len(astr_insts)):
        #             ra = ra_data[astr_inst_inds[astr_insts[i]]]
        #             dec = dec_data[astr_inst_inds[astr_insts[i]]]
        #             if plot_errorbars:
        #                 xerr = ra_err[astr_inst_inds[astr_insts[i]]]
        #                 yerr = dec_err[astr_inst_inds[astr_insts[i]]]
        #             else:
        #                 xerr = None
        #                 yerr = None

        #             ax.errorbar(ra, dec, xerr=xerr, yerr=yerr, marker=next(ax_symbols), c=next(ax_colors), zorder=10, label=astr_insts[i], linestyle='', ms=5, capsize=2)
        #     else:
        #         if plot_errorbars:
        #             xerr = ra_err
        #             yerr = dec_err
        #         else:
        #             xerr = None
        #             yerr = None

        #         ax.errorbar(ra_data, dec_data, xerr=xerr, yerr=yerr, marker='o', c='#FF7F11', zorder=10, linestyle='', capsize=2, ms=5)

        # modify the axes
        if square_plot:
            adjustable_param = 'datalim'
        else:
            adjustable_param = 'box'
        ax.set_aspect('equal', adjustable=adjustable_param)
        ax.set_xlabel('$\\Delta$RA [mas]')
        ax.set_ylabel('$\\Delta$Dec [mas]')
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=6)
        ax.invert_xaxis()  # To go to a left-handed coordinate system

        # plot sep/PA and/or rv zoom-in panels
        if rv_time_series:
            ax1 = plt.subplot2grid((3, 14), (0, 8), colspan=6)
            ax2 = plt.subplot2grid((3, 14), (1, 8), colspan=6)
            ax3 = plt.subplot2grid((3, 14), (2, 0), colspan=14, rowspan=1)
            ax2.set_ylabel('PA [$^{{\\circ}}$]')
            ax1.set_ylabel('$\\rho$ [mas]')
            ax3.set_ylabel('RV [km/s]')
            ax3.set_xlabel('Epoch')
            ax2.set_xlabel('Epoch')
            plt.subplots_adjust(hspace=0.3)
        else:
            ax1 = plt.subplot2grid((2, 14), (0, 9), colspan=6)
            ax2 = plt.subplot2grid((2, 14), (1, 9), colspan=6)
            ax2.set_ylabel('PA [$^{{\\circ}}$]')
            ax1.set_ylabel('$\\rho$ [mas]')
            ax2.set_xlabel('Epoch')

        if plot_astrometry_insts:
            ax1_colors = itertools.cycle(astr_colors)
            ax1_symbols = itertools.cycle(astr_symbols)

            ax2_colors = itertools.cycle(astr_colors)
            ax2_symbols = itertools.cycle(astr_symbols)

        epochs_seppa = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        for i in np.arange(num_orbits_to_plot):

            epochs_seppa[i, :] = np.linspace(
                start_mjd,
                Time(sep_pa_end_year, format='decimalyear').mjd,
                num_epochs_to_plot
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            if rv_time_series:
                raoff0, deoff0, _ = kepler.calc_orbit(
                    epochs_seppa[i, :], sma[i], ecc[i], inc[i], aop[i], pan[i],
                    tau[i], plx[i], mtot[i], tau_ref_epoch=results.tau_ref_epoch,
                    mass_for_Kamp=m0[i]
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0
            else:
                raoff0, deoff0, _ = kepler.calc_orbit(
                    epochs_seppa[i, :], sma[i], ecc[i], inc[i], aop[i], pan[i],
                    tau[i], plx[i], mtot[i], tau_ref_epoch=results.tau_ref_epoch
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0

            yr_epochs = Time(epochs_seppa[i, :], format='mjd').decimalyear

            seps, pas = orbitize.system.radec2seppa(raoff[i, :], deoff[i, :], mod180=mod180)

            plt.sca(ax1)
            plt.plot(yr_epochs, seps, color=sep_pa_color, zorder=1)

            plt.sca(ax2)
            plt.plot(yr_epochs, pas, color=sep_pa_color, zorder=1)

        # Plot sep/pa instruments
        if plot_astrometry_insts:
            for i in range(len(astr_insts)):
                sep = sep_data[astr_inst_inds[astr_insts[i]]]
                pa = pa_data[astr_inst_inds[astr_insts[i]]]
                epochs = astr_epochs[astr_inst_inds[astr_insts[i]]]
                if plot_errorbars:
                    serr = sep_err[astr_inst_inds[astr_insts[i]]]
                    perr = pa_err[astr_inst_inds[astr_insts[i]]]
                else:
                    yerr = None
                    perr = None

                plt.sca(ax1)
                plt.errorbar(Time(epochs,format='mjd').decimalyear,sep,yerr=serr,ms=5, linestyle='',marker=next(ax1_symbols),c=next(ax1_colors),zorder=10,label=astr_insts[i], capsize=2)
                plt.sca(ax2)
                plt.errorbar(Time(epochs,format='mjd').decimalyear,pa,yerr=perr,ms=5, linestyle='',marker=next(ax2_symbols),c=next(ax2_colors),zorder=10, capsize=2)
            plt.sca(ax1)
            plt.legend(title='Instruments', bbox_to_anchor=(1.3, 1), loc='upper right')
        else:
            if plot_errorbars:
                serr = sep_err
                perr = pa_err
            else:
                yerr = None
                perr = None

            plt.sca(ax1)
            plt.errorbar(Time(astr_epochs,format='mjd').decimalyear,sep_data,yerr=serr,ms=5, linestyle='',marker='o',c='purple',zorder=2, capsize=2)
            plt.sca(ax2)
            plt.errorbar(Time(astr_epochs,format='mjd').decimalyear,pa_data,yerr=perr,ms=5, linestyle='',marker='o',c='purple',zorder=2, capsize=2)

        if rv_time_series:

            rv_data = results.data[results.data['object'] == 0]
            rv_data = rv_data[rv_data['quant_type'] == 'rv']

            # switch current axis to rv panel
            plt.sca(ax3)
    
            # get list of rv instruments
            insts = np.unique(rv_data['instrument'])
            if len(insts) == 0:
                insts = ['defrv']

            # get gamma/sigma labels and corresponding positions in the posterior
            gams=['gamma_'+inst for inst in insts]
            sigs = ['sigma_'+inst for inst in insts]

            if isinstance(results.labels,list):
                labels=np.array(results.labels)
            else:
                labels=results.labels
            
            # get the indices corresponding to each gamma within results.labels
            gam_idx=[np.where(labels==inst_gamma)[0] for inst_gamma in gams]

            # indices corresponding to each instrument in the datafile
            inds={}
            for i in range(len(insts)):
                inds[insts[i]]=np.where(rv_data['instrument']==insts[i].encode())[0]

            # choose the orbit with the best log probability
            best_like=np.where(results.lnlike==np.amax(results.lnlike))[0][0] 

            # Get the posteriors for this index and convert to standard basis
            best_post = results.basis.to_standard_basis(results.post[best_like].copy())

            # Get the masses for the best posteriors:
            best_m0 = best_post[results.standard_param_idx['m0']]
            best_m1 = best_post[results.standard_param_idx['m{}'.format(object_to_plot)]]
            best_mtot = best_m0 + best_m1

            # colour/shape scheme scheme for rv data points
            clrs=('purple', '#0496FF','#372554','#FF1053','#3A7CA5','#143109')
            symbols=('o','^','v','s')

            ax3_colors = itertools.cycle(clrs)
            ax3_symbols = itertools.cycle(symbols)
            
            # get rvs and plot them
            for i,name in enumerate(inds.keys()):
                inst_data=rv_data[inds[name]]
                rvs=inst_data['quant1']
                epochs=inst_data['epoch']
                epochs=Time(epochs, format='mjd').decimalyear
                rvs -= best_post[results.param_idx[gams[i]]]
                if plot_errorbars:
                    yerr = inst_data['quant1_err']
                    yerr = np.sqrt(yerr**2 + best_post[results.param_idx[sigs[i]]]**2)
                plt.errorbar(epochs,rvs,yerr=yerr,ms=5, linestyle='',marker=next(ax3_symbols),c=next(ax3_colors),label=name,zorder=5,capsize=2)
            if len(inds.keys()) == 1 and 'defrv' in inds.keys():
                pass
            else:
                plt.legend()
            
            # calculate the predicted rv trend using the best orbit 
            _, _, vz = kepler.calc_orbit(
                epochs_seppa[0, :], 
                best_post[results.standard_param_idx['sma{}'.format(object_to_plot)]], 
                best_post[results.standard_param_idx['ecc{}'.format(object_to_plot)]], 
                best_post[results.standard_param_idx['inc{}'.format(object_to_plot)]], 
                best_post[results.standard_param_idx['aop{}'.format(object_to_plot)]], 
                best_post[results.standard_param_idx['pan{}'.format(object_to_plot)]], 
                best_post[results.standard_param_idx['tau{}'.format(object_to_plot)]], 
                best_post[results.standard_param_idx['plx']], best_mtot, 
                tau_ref_epoch=results.tau_ref_epoch, mass_for_Kamp=best_m0
            )
            
            vz=vz*-(best_m1)/np.median(best_m0)

            # plot rv trend
            plt.plot(Time(epochs_seppa[0, :],format='mjd').decimalyear, vz, color=sep_pa_color, zorder=1)


        # add colorbar
        if show_colorbar:
            if rv_time_series:
                # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
                # You can change x1.0.05 to adjust the distance between the main image and the colorbar.
                # You can change 0.02 to adjust the width of the colorbar.
                cbar_ax = fig.add_axes(
                    [ax.get_position().x1+0.005, ax.get_position().y0, 0.02, ax.get_position().height])
                cbar = mpl.colorbar.ColorbarBase(
                    cbar_ax, cmap=cmap, norm=norm_yr, orientation='vertical', label=cbar_param)
            else:
                # xpos, ypos, width, height, in fraction of figure size
                cbar_ax = fig.add_axes([0.47, 0.15, 0.015, 0.7])
                cbar = mpl.colorbar.ColorbarBase(
                    cbar_ax, cmap=cmap, norm=norm_yr, orientation='vertical', label=cbar_param)

        ax1.locator_params(axis='x', nbins=6)
        ax1.locator_params(axis='y', nbins=6)
        ax2.locator_params(axis='x', nbins=6)
        ax2.locator_params(axis='y', nbins=6)

    return fig



def plot_n_orbits(results, num_objects = 1, start_mjd=51544.,
                num_orbits_to_plot=100, num_epochs_to_plot=100,
                square_plot=True, show_colorbar=True, cmap_list = cmap,
                sep_pa_color='lightgrey', sep_pa_end_year=2025.0,
                cbar_param='Epoch [year]', mod180=False, rv_time_series=False, plot_astrometry=True,
                plot_astrometry_insts=False, plot_errorbars=True, figure=None):
    """
    Plots one orbital period for a select number of fitted orbits
    for a given object, with line segments colored according to time

    Args:
        num_objects (array): the total number of planets to plot
        object_to_plot (int): which object to plot (default: 1)
        start_mjd (float): MJD in which to start plotting orbits (default: 51544,
            the year 2000)
        num_orbits_to_plot (int): number of orbits to plot (default: 100)
        num_epochs_to_plot (int): number of points to plot per orbit (default: 100)
        square_plot (Boolean): Aspect ratio is always equal, but if
            square_plot is True (default), then the axes will be square,
            otherwise, white space padding is used
        show_colorbar (Boolean): Displays colorbar to the right of the plot [True]
        cmap_list (matplotlib.cm.ColorMap): array of color maps to use for making orbit tracks must be
            the same length as number of objects
            (default: modified Purples_r)
        sep_pa_color (string): any valid matplotlib color string, used to set the
            color of the orbit tracks in the Sep/PA panels (default: 'lightgrey').
        sep_pa_end_year (float): decimal year specifying when to stop plotting orbit
            tracks in the Sep/PA panels (default: 2025.0).
        cbar_param (string): options are the following: 'Epoch [year]', 'sma1', 'ecc1', 'inc1', 'aop1',
            'pan1', 'tau1', 'plx. Number can be switched out. Default is Epoch [year].
        mod180 (Bool): if True, PA will be plotted in range [180, 540]. Useful for plotting short
            arcs with PAs that cross 360 deg during observations (default: False)
        rv_time_series (Boolean): if fitting for secondary mass using MCMC for rv fitting and want to
            display time series, set to True.
        plot_astrometry (Boolean): set to True by default. Plots the astrometric data.
        plot_astrometry_insts (Boolean): set to False by default. Plots the astrometric data by instruments.
        plot_errorbars (Boolean): set to True by default. Plots error bars of measurements
        figure (matplotlib.pyplot.Figure): optionally include a predefined Figure object to plot the orbit on.
            Most users will not need this keyword. 

    Return:
        ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


    (written): Henry Ngo, Sarah Blunt, 2018
    Additions by Malena Rice, 2019

    """
    planets = ['HR8799e', 'HR8799d', 'HR8799c', 'HR8799b']
    sep_pa_figures = []

    for ind in range(num_objects):
        object_to_plot = int(ind +1)
        planet_name = planets[ind]
        cmap = cmap_list[ind]

        if Time(start_mjd, format='mjd').decimalyear >= sep_pa_end_year:
            raise ValueError('start_mjd keyword date must be less than sep_pa_end_year keyword date.')

        if object_to_plot > results.num_secondary_bodies:
            raise ValueError("Only {0} secondary bodies being fit. Requested to plot body {1} which is out of range".format(results.num_secondary_bodies, object_to_plot))

        if object_to_plot == 0:
            raise ValueError("Plotting the primary's orbit is currently unsupported. Stay tuned.")

        if rv_time_series and 'm0' not in results.labels:
            rv_time_series = False

            warnings.warn("It seems that the stellar and companion mass "
                        "have not been fitted separately. Setting "
                        "rv_time_series=True is therefore not possible "
                        "so the argument is set to False instead.")

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ErfaWarning)

            data = results.data[results.data['object'] == object_to_plot]
            possible_cbar_params = [
                'sma',
                'ecc',
                'inc',
                'aop'
                'pan',
                'tau',
                'plx'
            ]

            if cbar_param == 'Epoch [year]':
                pass
            elif cbar_param[0:3] in possible_cbar_params:
                index = results.param_idx[cbar_param]
            else:
                raise Exception(
                    "Invalid input; acceptable inputs include 'Epoch [year]', 'plx', 'sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'sma2', 'ecc2', ...)"
                )
            # Select random indices for plotted orbit
            num_orbits = len(results.post[:, 0])
            if num_orbits_to_plot > num_orbits:
                num_orbits_to_plot = num_orbits
            choose = np.random.randint(0, high=num_orbits, size=num_orbits_to_plot)

            # Get posteriors from random indices
            standard_post = []
            if results.sampler_name == 'MCMC':
                # Convert the randomly chosen posteriors to standard keplerian set
                for i in np.arange(num_orbits_to_plot):
                    orb_ind = choose[i]
                    param_set = np.copy(results.post[orb_ind])
                    standard_post.append(results.basis.to_standard_basis(param_set))
            else: # For OFTI, posteriors are already converted
                for i in np.arange(num_orbits_to_plot):
                    orb_ind = choose[i]
                    standard_post.append(results.post[orb_ind])

            standard_post = np.array(standard_post)

            sma = standard_post[:, results.standard_param_idx['sma{}'.format(object_to_plot)]]
            ecc = standard_post[:, results.standard_param_idx['ecc{}'.format(object_to_plot)]]
            inc = standard_post[:, results.standard_param_idx['inc{}'.format(object_to_plot)]]
            aop = standard_post[:, results.standard_param_idx['aop{}'.format(object_to_plot)]]
            pan = standard_post[:, results.standard_param_idx['pan{}'.format(object_to_plot)]]
            tau = standard_post[:, results.standard_param_idx['tau{}'.format(object_to_plot)]]
            plx = standard_post[:, results.standard_param_idx['plx']]

            # Then, get the other parameters
            if 'mtot' in results.labels:
                mtot = standard_post[:, results.standard_param_idx['mtot']]
            elif 'm0' in results.labels:
                m0 = standard_post[:, results.standard_param_idx['m0']]
                m1 = standard_post[:, results.standard_param_idx['m{}'.format(object_to_plot)]]
                mtot = m0 + m1

            raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
            deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
            vz_star = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
            epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

            # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
            # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
            for i in np.arange(num_orbits_to_plot):
                # Compute period (from Kepler's third law)
                period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
                period = period.to(u.day).value

                # Create an epochs array to plot num_epochs_to_plot points over one orbital period
                epochs[i, :] = np.linspace(start_mjd, float(
                    start_mjd+period[i]), num_epochs_to_plot)

                # Calculate ra/dec offsets for all epochs of this orbit

                #### HERE SHOULD BE THE MULTI PLANET SOLVER
                
                raoff0, deoff0, _ = kepler.calc_orbit(
                    epochs[i, :], sma[i], ecc[i], inc[i], aop[i], pan[i],
                    tau[i], plx[i], mtot[i], tau_ref_epoch=results.tau_ref_epoch
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0

            # Create a linearly increasing colormap for our range of epochs
            if cbar_param != 'Epoch [year]':
                cbar_param_arr = results.post[:, index]
                norm = mpl.colors.Normalize(vmin=np.min(cbar_param_arr),
                                            vmax=np.max(cbar_param_arr))
                norm_yr = mpl.colors.Normalize(vmin=np.min(
                    cbar_param_arr), vmax=np.max(cbar_param_arr))

            elif cbar_param == 'Epoch [year]':

                min_cbar_date = np.min(epochs)
                max_cbar_date = np.max(epochs[-1, :])

                # if we're plotting orbital periods greater than 1,000 yrs, limit the colorbar dynamic range
                if max_cbar_date - min_cbar_date > 1000 * 365.25:
                    max_cbar_date = min_cbar_date + 1000 * 365.25

                norm = mpl.colors.Normalize(vmin=min_cbar_date, vmax=max_cbar_date)

                norm_yr = mpl.colors.Normalize(
                    vmin=Time(min_cbar_date, format='mjd').decimalyear,
                    vmax=Time(max_cbar_date, format='mjd').decimalyear
                )

            # Create figure for orbit plots
            if figure is None:
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot()

            else:
                fig = plt.figure(figure)
                ax = mpl.axes.Axes(fig, (1,1,1,1))
            
            astr_inds=np.where((~np.isnan(data['quant1'])) & (~np.isnan(data['quant2'])))
            astr_epochs=data['epoch'][astr_inds]

            radec_inds = np.where(data['quant_type'] == 'radec')
            seppa_inds = np.where(data['quant_type'] == 'seppa')

            # transform RA/Dec points to Sep/PA
            sep_data = np.copy(data['quant1'])
            sep_err = np.copy(data['quant1_err'])
            pa_data = np.copy(data['quant2'])
            pa_err = np.copy(data['quant2_err'])

            if len(radec_inds[0] > 0):

                sep_from_ra_data, pa_from_dec_data = orbitize.system.radec2seppa(
                    data['quant1'][radec_inds], data['quant2'][radec_inds]
                )

                num_radec_pts = len(radec_inds[0])
                sep_err_from_ra_data = np.empty(num_radec_pts)
                pa_err_from_dec_data = np.empty(num_radec_pts)
                for j in np.arange(num_radec_pts):

                    sep_err_from_ra_data[j], pa_err_from_dec_data[j], _ = orbitize.system.transform_errors(
                        np.array(data['quant1'][radec_inds][j]), np.array(data['quant2'][radec_inds][j]), 
                        np.array(data['quant1_err'][radec_inds][j]), np.array(data['quant2_err'][radec_inds][j]), 
                        np.array(data['quant12_corr'][radec_inds][j]), orbitize.system.radec2seppa
                    )

                sep_data[radec_inds] = sep_from_ra_data
                sep_err[radec_inds] = sep_err_from_ra_data

                pa_data[radec_inds] = pa_from_dec_data
                pa_err[radec_inds] = pa_err_from_dec_data

            # Transform Sep/PA points to RA/Dec
            ra_data = np.copy(data['quant1'])
            ra_err = np.copy(data['quant1_err'])
            dec_data = np.copy(data['quant2'])
            dec_err = np.copy(data['quant2_err'])

            if len(seppa_inds[0] > 0):

                ra_from_seppa_data, dec_from_seppa_data = orbitize.system.seppa2radec(
                    data['quant1'][seppa_inds], data['quant2'][seppa_inds]
                )

                num_seppa_pts = len(seppa_inds[0])
                ra_err_from_seppa_data = np.empty(num_seppa_pts)
                dec_err_from_seppa_data = np.empty(num_seppa_pts)
                for j in np.arange(num_seppa_pts):

                    ra_err_from_seppa_data[j], dec_err_from_seppa_data[j], _ = orbitize.system.transform_errors(
                        np.array(data['quant1'][seppa_inds][j]), np.array(data['quant2'][seppa_inds][j]), 
                        np.array(data['quant1_err'][seppa_inds][j]), np.array(data['quant2_err'][seppa_inds][j]), 
                        np.array(data['quant12_corr'][seppa_inds][j]), orbitize.system.seppa2radec
                    )

                ra_data[seppa_inds] = ra_from_seppa_data
                ra_err[seppa_inds] = ra_err_from_seppa_data

                dec_data[seppa_inds] = dec_from_seppa_data
                dec_err[seppa_inds] = dec_err_from_seppa_data

            # Plot each orbit 
            for i in np.arange(num_orbits_to_plot):
                plt.plot(raoff[i, :], deoff[i, :], c= cmap(0.5), linewidth=0.5, alpha=0.2)

            # modify the axes
            if square_plot:
                adjustable_param = 'datalim'
            else:
                adjustable_param = 'box'
            plt.errorbar(ra_data, dec_data, xerr= ra_err, yerr=dec_err, c=cmap(0.9),  linestyle='', marker='o',ms=5)
            ax.set_aspect('equal', adjustable=adjustable_param)
            ax.set_xlabel('$\\Delta$RA [mas]')
            ax.set_ylabel('$\\Delta$Dec [mas]')
            ax.locator_params(axis='x', nbins=6)
            ax.locator_params(axis='y', nbins=6)
            ax.invert_xaxis()  # To go to a left-handed coordinate system
            plt.tight_layout()





            # Create new figure subplot with a sep/PA plot for each planet
            sep_pa_fig, (sep_ax, pa_ax) = plt.subplots(2,1, figsize = (5,10))

            # plot sep/PA 
            pa_ax.set_ylabel('PA [$^{{\\circ}}$]')
            sep_ax.set_ylabel('$\\rho$ [mas]')
            pa_ax.set_xlabel('Epoch')
            sep_ax.set_title(planet_name)


            epochs_seppa = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

            for i in np.arange(num_orbits_to_plot):

                epochs_seppa[i, :] = np.linspace(
                    start_mjd,
                    Time(sep_pa_end_year, format='decimalyear').mjd,
                    num_epochs_to_plot
                )

                # Calculate ra/dec offsets for all epochs of this orbit
                if rv_time_series:
                    raoff0, deoff0, _ = kepler.calc_orbit(
                        epochs_seppa[i, :], sma[i], ecc[i], inc[i], aop[i], pan[i],
                        tau[i], plx[i], mtot[i], tau_ref_epoch=results.tau_ref_epoch,
                        mass_for_Kamp=m0[i]
                    )

                    raoff[i, :] = raoff0
                    deoff[i, :] = deoff0
                else:
                    raoff0, deoff0, _ = kepler.calc_orbit(
                        epochs_seppa[i, :], sma[i], ecc[i], inc[i], aop[i], pan[i],
                        tau[i], plx[i], mtot[i], tau_ref_epoch=results.tau_ref_epoch
                    )

                    raoff[i, :] = raoff0
                    deoff[i, :] = deoff0

                yr_epochs = Time(epochs_seppa[i, :], format='mjd').decimalyear

                seps, pas = orbitize.system.radec2seppa(raoff[i, :], deoff[i, :], mod180=mod180)

                plt.sca(sep_ax)
                plt.plot(yr_epochs, seps, color=sep_pa_color, zorder=1)

                plt.sca(pa_ax)
                plt.plot(yr_epochs, pas, color=sep_pa_color, zorder=1)

            # Plot sep/pa data points
            if plot_errorbars:
                serr = sep_err
                perr = pa_err
            else:
                yerr = None
                perr = None

            plt.sca(sep_ax)
            plt.errorbar(Time(astr_epochs,format='mjd').decimalyear,sep_data,yerr=serr, linestyle='',marker='o',ms=5,c=cmap(0.5),zorder=2, capsize=2)
            plt.sca(pa_ax)
            plt.errorbar(Time(astr_epochs,format='mjd').decimalyear,pa_data,yerr=perr, linestyle='',marker='o',ms=5,c=cmap(0.5),zorder=2, capsize=2)



            # add colorbar
            # if show_colorbar:
            #         # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
            #         # You can change x1.0.05 to adjust the distance between the main image and the colorbar.
            #         # You can change 0.02 to adjust the width of the colorbar.

            #     # xpos, ypos, width, height, in fraction of figure size
            #     # cbar_ax = fig.add_axes([0.47, 0.15, 0.015, 0.7])
            #     # cbar = mpl.colorbar.ColorbarBase(
            #            # ax, cmap=cmap, norm=norm_yr, orientation='vertical', label=cbar_param)
            #     fig.colorbar(cm.ScalarMappable(norm=norm_yr, cmap=cmap), ax=ax)

            # ax1.locator_params(axis='x', nbins=6)
            # ax1.locator_params(axis='y', nbins=6)
            # ax2.locator_params(axis='x', nbins=6)
            # ax2.locator_params(axis='y', nbins=6)

        
        figure = fig
        sep_pa_figures.append(sep_pa_fig)

    return fig, sep_pa_figures

def plot_with_system(results, colors = mpl.cm.Purples, objects = 1, orbits = 1000, 
                     epochs=100, start_mjd = 51544, sep_pa_end_year = 2025.0, sep_pa_color='lightgrey',
                     figure=None):
    ''' inputs
    ** assumes astropy.constants is imported as consts and astropy.units is u **
    results (orbitize.results.Results): orbitize results object
    colors (matplotlib.cm.ColorMap): array of color maps the same length as objects
    objects (int): number of objects in the system, default=1
    orbits (int): number of orbits to plot, default=1000
    epochs (int): number of epochs to plot for each orbit, default=100
    start_mjd (int/ float): starting date for plotting orbits, default=51544
    sep_pa_end_year (float): end date (in decimal years) for plotting orbits in sep/pa plots only, default=2025.0

    returns:
    fig (matplotlib.pyplot.Figure): the projected orbits in RA vs Dec space
    sep_pa_figure (list): list of sep/pa plots, one for each object in the system
    '''

    # Get posteriors from random indices 
    num_orbits = len(results.post[:, 0])
    if orbits > num_orbits:
        orbits = num_orbits
    choose = np.random.randint(0, high= num_orbits, size=orbits)

    standard_post = []
    if results.sampler_name == 'MCMC':
        for i in np.arange(orbits):
            orb_ind = choose[i]
            param_set = np.copy(results.post[orb_ind])
            standard_post.append(results.basis.to_standard_basis(param_set))
    else:
        for i in np.arange(orbits):
            orb_ind = choose[i]
            standard_post.append(results.post[orb_ind])

    standard_post = np.array(standard_post)

    #posterior = results.post
    posterior = standard_post.transpose() # need for system.compute input format (params, walkers * steps)
        # transposed post.shape = (param, total_orbits)

    epoch_array = np.zeros((orbits, epochs))
    raoff = np.zeros((epochs, objects + 1, orbits))
    decoff = np.zeros((epochs, objects + 1, orbits))

    ## plot the ra, dec data points and orbital projections of each object
    # for obj in range(objects):
    
    sma = posterior[results.standard_param_idx['sma{}'.format(objects)]]

    all_smas = posterior[results.system.sma_indx]
    all_masses = posterior[results.system.secondary_mass_indx]
    within_orbit = np.where(all_smas <= sma)
    inside_masses = all_masses[within_orbit]
    if 'm0' not in results.standard_param_idx:
        mtot = posterior[results.standard_param_idx['mtot']]
    else:
        m0 = posterior[results.standard_param_idx['m0']]
        # m1 = 
        mtot = np.sum(inside_masses) + m0


    period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3 / (consts.G*(mtot * u.Msun)))
    period = period.to(u.day).value

    for orb in range(orbits):
        epoch_array[orb, :] = np.linspace(start_mjd, 
                        float(start_mjd + period[orb]), epochs)

        raoff0, decoff0, _ = results.system.compute_all_orbits(
            standard_post[orb, :], epoch_array[orb, :], comp_rebound=True
        )
        raoff[:, :, orb] = raoff0
        decoff[:, :, orb] = decoff0


    if figure is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()
    else:
        fig = plt.figure(figure)
        ax = mpl.axes.Axes(fig, (1,1,1,1))

    for obj in range(objects):
        object_to_plot = int(obj + 1)
        data = results.data[results.data['object'] == object_to_plot]

        c = colors[obj]
        for orb in range(orbits):
            plt.plot(raoff[:, obj + 1, orb], decoff[:, obj + 1, orb], color = c(0.5), linewidth = 0.5, alpha=0.2)

        # convert seppa points to ra/dec
        seppa_inds = np.where(data['quant_type'] == 'seppa')

        ra_data = np.copy(data['quant1'])
        ra_err = np.copy(data['quant1_err'])
        dec_data = np.copy(data['quant2'])
        dec_err = np.copy(data['quant2_err'])

        if len(seppa_inds[0] > 0):
            ra_from_seppa, dec_from_seppa = orbitize.system.seppa2radec(
                data['quant1'][seppa_inds], data['quant2'][seppa_inds]
            )

            num_seppa = len(seppa_inds[0])
            ra_err_from_seppa = np.empty(num_seppa)
            dec_err_from_seppa = np.empty(num_seppa)
            for i in np.arange(num_seppa):
                ra_err_from_seppa[i], dec_err_from_seppa[i], _ = orbitize.system.transform_errors(
                    np.array(data['quant1'][seppa_inds][i]), np.array(data['quant2'][seppa_inds][i]),
                    np.array(data['quant1_err'][seppa_inds][i]), np.array(data['quant2_err'][seppa_inds][i]),
                    np.array(data['quant12_corr'][seppa_inds][i]), orbitize.system.seppa2radec
                )
        
            ra_data[seppa_inds] = ra_from_seppa
            ra_err[seppa_inds] = ra_err_from_seppa
            dec_data[seppa_inds] = dec_from_seppa
            dec_err[seppa_inds] = dec_err_from_seppa
        
        # Plot data points on top of orbits
        plt.errorbar(ra_data, dec_data, xerr= ra_err, yerr = dec_err, color =c(0.001), marker='o', ms=5, linestyle='')
        ax.set_xlabel('$\\Delta$RA [mas]')
        ax.set_ylabel('$\\Delta$Dec [mas]')
        figure = fig

    plt.gca().invert_xaxis()

    # create different plots for each objects sep, pa
    sep_pa_figures = []
    epochs_seppa = np.linspace(start_mjd, Time(sep_pa_end_year, format='decimalyear').mjd,
                                        epochs)
    

    for obj in range(objects):
        object_to_plot = int(obj + 1)
        data = results.data[results.data['object'] == object_to_plot]

        #transform ra dec measurements to sep PA (to plot orbits against given data points)
        radec_inds = np.where(data['quant_type'] == 'radec')

        # transform RA/Dec data points to Sep/PA
        sep_data = np.copy(data['quant1'])
        sep_err = np.copy(data['quant1_err'])
        pa_data = np.copy(data['quant2'])
        pa_err = np.copy(data['quant2_err'])

        if len(radec_inds[0] > 0):

            sep_from_ra_data, pa_from_dec_data = orbitize.system.radec2seppa(
                data['quant1'][radec_inds], data['quant2'][radec_inds]
            )

            num_radec_pts = len(radec_inds[0])
            sep_err_from_ra_data = np.empty(num_radec_pts)
            pa_err_from_dec_data = np.empty(num_radec_pts)
            
            for j in np.arange(num_radec_pts):
                sep_err_from_ra_data[j], pa_err_from_dec_data[j], _ = orbitize.system.transform_errors(
                    np.array(data['quant1'][radec_inds][j]), np.array(data['quant2'][radec_inds][j]), 
                    np.array(data['quant1_err'][radec_inds][j]), np.array(data['quant2_err'][radec_inds][j]), 
                    np.array(data['quant12_corr'][radec_inds][j]), orbitize.system.radec2seppa
                )

            sep_data[radec_inds] = sep_from_ra_data
            sep_err[radec_inds] = sep_err_from_ra_data
            pa_data[radec_inds] = pa_from_dec_data
            pa_err[radec_inds] = pa_err_from_dec_data

        ob_epochs = data['epoch']

        # Create new figure subplot for each planet
        sep_pa_fig, (sep_ax, pa_ax) = plt.subplots(2, 1, figsize = (5,10))

        # plot sep/PA 
        pa_ax.set_ylabel('PA [$^{{\\circ}}$]')
        sep_ax.set_ylabel('$\\rho$ [mas]')
        pa_ax.set_xlabel('Epoch')
        sep_ax.set_title('Object {}'.format(obj +1))
        

        # Plotting the computed orbits from MCMC posterior in Sep/ PA over time
        for orb in range(orbits):
            # raoff0, decoff0, _ = results.system.compute_all_orbits(
            #     standard_post[orb], epochs_seppa, comp_rebound=True)
            
            # raoff[:,obj+1, orb] = raoff0[:,obj+1,0]
            # decoff[:,obj+1, orb] = decoff0[:,obj+1,0]
            epoch_array[orb, :] = np.linspace(start_mjd, 
                float(start_mjd + period[orb]), epochs)
            
            yr_epochs = Time(epoch_array[orb], format='mjd').decimalyear
            sep_pa_start_year = Time(start_mjd, format='mjd').decimalyear
            seps, pas = orbitize.system.radec2seppa(raoff[:, obj + 1, orb], decoff[:, obj + 1, orb], mod180=False)

            plt.sca(sep_ax)
            plt.plot(yr_epochs, seps, c=sep_pa_color, alpha = 0.3)
            plt.xlim((sep_pa_start_year, sep_pa_end_year))
            plt.sca(pa_ax)
            plt.plot(yr_epochs, pas, c=sep_pa_color, alpha = 0.3)
            plt.xlim((sep_pa_start_year, sep_pa_end_year))


        c = colors[obj]

        plt.sca(sep_ax)
        plt.errorbar(Time(ob_epochs, format='mjd').decimalyear, sep_data, yerr=sep_err, ms=5, linestyle='', marker='o', zorder=2, capsize=2, color=c(0.001))
        plt.ylim((np.min(sep_data) - 0.05*(np.max(sep_data)), np.max(sep_data) + 0.05*(np.max(sep_data))))
        plt.sca(pa_ax)
        plt.errorbar(Time(ob_epochs, format='mjd').decimalyear, pa_data, yerr=pa_err, ms=5, linestyle='', marker='o', zorder=2, capsize=2, color=c(0.001))
        plt.ylim((np.min(pa_data) - 0.1*(np.max(pa_data)), np.max(pa_data) + 0.1*(np.max(pa_data))))


        sep_pa_figures.append(sep_pa_fig)

    return fig, sep_pa_figures

def plot_period_ratios(results, num_objects, colors):

    posterior = results.post

    ratios = []
    for i in range(num_objects -1):
        smaller_sma = posterior[:, results.standard_param_idx['sma{}'.format(i+1)]]
        bigger_sma = posterior[:, results.standard_param_idx['sma{}'.format(i+2)]]
        
        if 'mtot' in results.labels:
            mtot = posterior[:, results.standard_param_idx['mtot']]
        elif 'm0' in results.labels:
            mtot = 0
            for i in range(num_objects +1):
                m = posterior[:, results.standard_param_idx['m{}'.format(i)]]
                mtot += m
        
        shorter_period = np.sqrt(4*np.pi**2.0*(smaller_sma*u.AU)**3/(consts.G*(mtot*u.Msun))).value
        longer_period = np.sqrt(4*np.pi**2.0*(bigger_sma*u.AU)**3/(consts.G*(mtot*u.Msun))).value
        
        period_ratio = longer_period / shorter_period
        ratios.append(period_ratio)
    
    ratio_fig = plt.figure(figsize=(8,8))

    for i in range(len(ratios)):
        plt.hist(ratios[i], histtype='step', label='{}-{} ratio'.format(i+1, i))

    plt.axvline(2, color='r', ls='--', label='2:1 Resonance')
    plt.xlabel('Period Ratio')
    plt.title('Consecutive Planet Period Ratios')
    plt.legend()

    return ratio_fig


def plot_n_orbits_new(results, num_objects=1, start_mjd=51544.,
                num_orbits_to_plot=100, num_epochs_to_plot=100,
                square_plot=True, cmap_list=None, nbody_solver=False,
                sep_pa_color='lightgrey', sep_pa_end_year=2025.0,
                cbar_param='Epoch [year]', mod180=False, rv_time_series=False, 
                plot_errorbars=True, figure=None, fig_titles=None, tau_ref_epoch=None):
    """
    Plots one orbital period for a select number of fitted orbits
    for a given object, with line segments colored according to time

    Args:
        results (orbitize.results.Results): an orbitize Results object with fit results
        num_objects (array): the total number of planets to plot
        start_mjd (float): MJD in which to start plotting orbits (default: 51544,
            the year 2000)
        num_orbits_to_plot (int): number of orbits to plot (default: 100)
        num_epochs_to_plot (int): number of points to plot per orbit (default: 100)
        square_plot (Boolean): Aspect ratio is always equal, but if
            square_plot is True (default), then the axes will be square,
            otherwise, white space padding is used
        cmap_list (matplotlib.cm.ColorMap): array of color maps to use for making orbit tracks must be
            the same length as number of objects
            (default: modified Purples_r)
        nbody_solver (boolean): Keyword to determine if n-body solver is used to compute orbits (default: False)
        sep_pa_color (string): any valid matplotlib color string, used to set the
            color of the orbit tracks in the Sep/PA panels (default: 'lightgrey').
        sep_pa_end_year (float): decimal year specifying when to stop plotting orbit
            tracks in the Sep/PA panels (default: 2025.0).
        cbar_param (string): options are the following: 'Epoch [year]', 'sma1', 'ecc1', 'inc1', 'aop1',
            'pan1', 'tau1', 'plx. Number can be switched out. Default is Epoch [year].
        mod180 (Bool): if True, PA will be plotted in range [180, 540]. Useful for plotting short
            arcs with PAs that cross 360 deg during observations (default: False)
        rv_time_series (Boolean): if fitting for secondary mass using MCMC for rv fitting and want to
            display time series, set to True.
        plot_errorbars (Boolean): set to True by default. Plots error bars of measurements
        figure (matplotlib.pyplot.Figure): optionally include a predefined Figure object to plot the orbit on.
        fig_titles (list): a list of strings to title the orbit, sep, and pa plots -should have length (num_planets + 1) (default: None)
        tau_ref_epoch (float):

    Return:
        ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


    (written): Henry Ngo, Sarah Blunt, 2018
    Additions by Malena Rice, 2019

    """
    sep_pa_figures = []

    for ind in range(num_objects):
        object_to_plot = int(ind +1)        
        cmap = cmap_list[ind]

        if Time(start_mjd, format='mjd').decimalyear >= sep_pa_end_year:
            raise ValueError('start_mjd keyword date must be less than sep_pa_end_year keyword date.')

        if object_to_plot > results.num_secondary_bodies:
            raise ValueError("Only {0} secondary bodies being fit. Requested to plot body {1} which is out of range".format(results.num_secondary_bodies, object_to_plot))

        if object_to_plot == 0:
            raise ValueError("Plotting the primary's orbit is currently unsupported. Stay tuned.")

        if rv_time_series and 'm0' not in results.labels:
            rv_time_series = False

            warnings.warn("It seems that the stellar and companion mass "
                        "have not been fitted separately. Setting "
                        "rv_time_series=True is therefore not possible "
                        "so the argument is set to False instead.")

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ErfaWarning)

            data = results.data[results.data['object'] == object_to_plot]
            possible_cbar_params = [
                'sma',
                'ecc',
                'inc',
                'aop'
                'pan',
                'tau',
                'plx'
            ]

            if cbar_param == 'Epoch [year]':
                pass
            elif cbar_param[0:3] in possible_cbar_params:
                index = results.param_idx[cbar_param]
            else:
                raise Exception(
                    "Invalid input; acceptable inputs include 'Epoch [year]', 'plx', 'sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'sma2', 'ecc2', ...)"
                )
            # Select random indices for plotted orbit
            num_orbits = len(results.post[:, 0])
            if num_orbits_to_plot > num_orbits:
                num_orbits_to_plot = num_orbits
            choose = np.random.randint(0, high=num_orbits, size=num_orbits_to_plot)

            # Get posteriors from random indices
            standard_post = []
            if results.sampler_name == 'MCMC':
                # Convert the randomly chosen posteriors to standard keplerian set
                for i in np.arange(num_orbits_to_plot):
                    orb_ind = choose[i]
                    param_set = np.copy(results.post[orb_ind])
                    standard_post.append(results.basis.to_standard_basis(param_set))
            else: # For OFTI, posteriors are already converted
                for i in np.arange(num_orbits_to_plot):
                    orb_ind = choose[i]
                    standard_post.append(results.post[orb_ind])
            standard_post = np.array(standard_post)

            if 'mtot' in results.labels:
                mtot = standard_post[:, results.standard_param_idx['mtot']]
            elif 'm0' in results.labels:
                m0 = standard_post[:, results.standard_param_idx['m0']]
                m1 = standard_post[:, results.standard_param_idx['m{}'.format(object_to_plot)]]
                mtot = m0 + m1

            raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
            deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
            epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

            # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
            ## need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
            for i in np.arange(num_orbits_to_plot):
                # Compute period (from Kepler's third law) based on the object being plotted
                sma = standard_post[:, results.standard_param_idx['sma{}'.format(object_to_plot)]]
                period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
                period = period.to(u.day).value

                # Create an epochs array to plot num_epochs_to_plot points over one orbital period
                epochs[i, :] = np.linspace(start_mjd, float(
                    start_mjd+period[i]), num_epochs_to_plot)

                # Calculate ra/dec offsets for all epochs of this orbit
                # compute_all_orbits automatically uses perturbation approx. when passed masses in param list
                raoff0, deoff0, _ = results.system.compute_all_orbits(
                    standard_post[i, :], epochs[i, :], comp_rebound=nbody_solver, tau_ref_epoch=tau_ref_epoch
                )
                
                raoff[i, :] = raoff0[:,object_to_plot,0]
                deoff[i, :] = deoff0[:,object_to_plot,0]

            # Create a linearly increasing colormap for our range of epochs
            if cbar_param != 'Epoch [year]':
                cbar_param_arr = results.post[:, index]
                norm = mpl.colors.Normalize(vmin=np.min(cbar_param_arr),
                                            vmax=np.max(cbar_param_arr))
                norm_yr = mpl.colors.Normalize(vmin=np.min(
                    cbar_param_arr), vmax=np.max(cbar_param_arr))

            elif cbar_param == 'Epoch [year]':

                min_cbar_date = np.min(epochs)
                max_cbar_date = np.max(epochs[-1, :])

                # if we're plotting orbital periods greater than 1,000 yrs, limit the colorbar dynamic range
                if max_cbar_date - min_cbar_date > 1000 * 365.25:
                    max_cbar_date = min_cbar_date + 1000 * 365.25

                norm = mpl.colors.Normalize(vmin=min_cbar_date, vmax=max_cbar_date)

                norm_yr = mpl.colors.Normalize(
                    vmin=Time(min_cbar_date, format='mjd').decimalyear,
                    vmax=Time(max_cbar_date, format='mjd').decimalyear
                )

            # Create figure for orbit plots
            if figure is None:
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot()

            else:
                fig = plt.figure(figure)
                ax = mpl.axes.Axes(fig, (1,1,1,1))
            
            astr_inds=np.where((~np.isnan(data['quant1'])) & (~np.isnan(data['quant2'])))
            astr_epochs=data['epoch'][astr_inds]

            radec_inds = np.where(data['quant_type'] == 'radec')
            seppa_inds = np.where(data['quant_type'] == 'seppa')

            # transform RA/Dec points to Sep/PA
            sep_data = np.copy(data['quant1'])
            sep_err = np.copy(data['quant1_err'])
            pa_data = np.copy(data['quant2'])
            pa_err = np.copy(data['quant2_err'])

            if len(radec_inds[0] > 0):

                sep_from_ra_data, pa_from_dec_data = orbitize.system.radec2seppa(
                    data['quant1'][radec_inds], data['quant2'][radec_inds]
                )

                num_radec_pts = len(radec_inds[0])
                sep_err_from_ra_data = np.empty(num_radec_pts)
                pa_err_from_dec_data = np.empty(num_radec_pts)
                for j in np.arange(num_radec_pts):

                    sep_err_from_ra_data[j], pa_err_from_dec_data[j], _ = orbitize.system.transform_errors(
                        np.array(data['quant1'][radec_inds][j]), np.array(data['quant2'][radec_inds][j]), 
                        np.array(data['quant1_err'][radec_inds][j]), np.array(data['quant2_err'][radec_inds][j]), 
                        np.array(data['quant12_corr'][radec_inds][j]), orbitize.system.radec2seppa
                    )

                sep_data[radec_inds] = sep_from_ra_data
                sep_err[radec_inds] = sep_err_from_ra_data

                pa_data[radec_inds] = pa_from_dec_data
                pa_err[radec_inds] = pa_err_from_dec_data

            # Transform Sep/PA points to RA/Dec
            ra_data = np.copy(data['quant1'])
            ra_err = np.copy(data['quant1_err'])
            dec_data = np.copy(data['quant2'])
            dec_err = np.copy(data['quant2_err'])

            if len(seppa_inds[0] > 0):

                ra_from_seppa_data, dec_from_seppa_data = orbitize.system.seppa2radec(
                    data['quant1'][seppa_inds], data['quant2'][seppa_inds]
                )

                num_seppa_pts = len(seppa_inds[0])
                ra_err_from_seppa_data = np.empty(num_seppa_pts)
                dec_err_from_seppa_data = np.empty(num_seppa_pts)
                for j in np.arange(num_seppa_pts):

                    ra_err_from_seppa_data[j], dec_err_from_seppa_data[j], _ = orbitize.system.transform_errors(
                        np.array(data['quant1'][seppa_inds][j]), np.array(data['quant2'][seppa_inds][j]), 
                        np.array(data['quant1_err'][seppa_inds][j]), np.array(data['quant2_err'][seppa_inds][j]), 
                        np.array(data['quant12_corr'][seppa_inds][j]), orbitize.system.seppa2radec
                    )

                ra_data[seppa_inds] = ra_from_seppa_data
                ra_err[seppa_inds] = ra_err_from_seppa_data

                dec_data[seppa_inds] = dec_from_seppa_data
                dec_err[seppa_inds] = dec_err_from_seppa_data

            # Plot each orbit (for each planet,, num_object_to_plot)
            for i in np.arange(num_orbits_to_plot):
                plt.plot(raoff[i, :], deoff[i, :], c= cmap(0.5), linewidth=0.5, alpha=0.2)

            # modify the axes
            if square_plot:
                adjustable_param = 'datalim'
            else:
                adjustable_param = 'box'
            plt.errorbar(ra_data, dec_data, xerr= ra_err, yerr=dec_err, c=cmap(0.9),  linestyle='', marker='o',ms=5)
            ax.set_aspect('equal', adjustable=adjustable_param)
            ax.set_xlabel('$\\Delta$RA [mas]')
            ax.set_ylabel('$\\Delta$Dec [mas]')
            ax.locator_params(axis='x', nbins=6)
            ax.locator_params(axis='y', nbins=6)
            ax.invert_xaxis()  # To go to a left-handed coordinate system
            # plt.tight_layout()


            # Create new figure subplot with a sep/PA plot for each planet
            sep_pa_fig, (sep_ax, pa_ax) = plt.subplots(2,1, figsize = (5,10))
            pa_ax.set_ylabel('PA [$^{{\\circ}}$]')
            sep_ax.set_ylabel('$\\rho$ [mas]')
            pa_ax.set_xlabel('Epoch')

            # Plot Sep/PA 
            epochs_seppa = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

            for i in np.arange(num_orbits_to_plot):

                epochs_seppa[i, :] = np.linspace(
                    start_mjd,
                    Time(sep_pa_end_year, format='decimalyear').mjd,
                    num_epochs_to_plot
                )

                # Calculate ra/dec offsets for all epochs of this orbit
                raoff0, deoff0, _ = results.system.compute_all_orbits(
                    standard_post[i, :], epochs_seppa[i, :], comp_rebound=nbody_solver, tau_ref_epoch=tau_ref_epoch
                )
    
                raoff[i, :] = raoff0[:,object_to_plot,0]
                deoff[i, :] = deoff0[:,object_to_plot,0]

                yr_epochs = Time(epochs_seppa[i, :], format='mjd').decimalyear

                seps, pas = orbitize.system.radec2seppa(raoff[i, :], deoff[i, :], mod180=mod180)

                plt.sca(sep_ax)
                plt.plot(yr_epochs, seps, color=sep_pa_color, zorder=1)

                plt.sca(pa_ax)
                plt.plot(yr_epochs, pas, color=sep_pa_color, zorder=1)

            # Plot sep/pa data points
            if plot_errorbars:
                serr = sep_err
                perr = pa_err
            else:
                serr = None
                perr = None

            plt.sca(sep_ax)
            plt.errorbar(Time(astr_epochs,format='mjd').decimalyear,sep_data,yerr=serr, linestyle='',marker='o',ms=5,c=cmap(0.5),zorder=2, capsize=2)
            plt.sca(pa_ax)
            plt.errorbar(Time(astr_epochs,format='mjd').decimalyear,pa_data,yerr=perr, linestyle='',marker='o',ms=5,c=cmap(0.5),zorder=2, capsize=2)

            # Add Figure titles
            if fig_titles is not None:
                seppa_titles = fig_titles[1:]
                ax.set_title(fig_titles[0])
                sep_ax.set_title(seppa_titles[ind])

        figure = fig
        sep_pa_figures.append(sep_pa_fig)
        plt.close(fig=sep_pa_fig)

    plt.close(fig=figure)
    return fig, sep_pa_figures


def plot_chains(file, num_planets, save_dir, which_params=['sma', 'ecc', 'inc'], walkers=1000, 
                colors=[mpl.cm.Purples, mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Oranges],
                ):
    '''
    inputs
    file (str): path to orbitize results.hdf5 file
    num_planets (int): number of planetary objects in system
    save_dir (str): path to directory where plots will be saved
    which_params (list of strings): which parameter chains to be examined, default:['sma', 'ecc', 'inc']
    walkers (int): number of walkers to plot, default:1000
    colors (list of matplotlib.colormaps): list of colors to distinguish planets

    outputs
    none: saves the chain plots to provided directory
    '''
    
    if 'loader' in locals():
        del loader
        
    loader = orbitize.results.Results()
    loader.load_results(file)
    posty = loader.post

    total_samples, n_params = posty.shape
    n_steps = int(total_samples/walkers)

    steps = n_steps

    params = n_params

    posty = posty.reshape(int(walkers), int(steps), int(params))

    # create stacked plot for each planet
    for planet in range(num_planets):
        planet_number = str(planet + 1)
        fig, ax = plt.subplots(len(which_params), figsize=(10, 2*(len(which_params))))
        color = colors[planet]

        for par in range(len(which_params)):
            ax[par].set(ylabel=which_params[par])
            for n in range(walkers):
                ax[par].plot(range(steps), posty[n, :, loader.standard_param_idx[which_params[par]+'{}'.format(planet_number)]], alpha=0.2, color=color(0.5))


        plt.xlabel('Step')

        fig.suptitle('Object {}'.format(planet_number))
        plt.savefig(save_dir + '/p'+planet_number+'_stacked_chain.png')

def plot_coplanarity(results, num_objects, fig=None):
    '''
    inputs
    results (orbitize.results.Results): orbitize results object
    num_objects (int): number of companions in the system
    fig (matplotlib.pyplot.figure): figure on which histograms are plotted default=None

    returns
    fig (matplotlib.pyplot.figure): the coplanarity figure
    '''
    for pl in range(num_objects -1):
        planet = pl + 1
        second = planet + 1

        post = results.post
        inc1 = post[:, results.standard_param_idx['inc{}'.format(planet)]]
        pan1 = post[:, results.standard_param_idx['pan{}'.format(planet)]]

        while second <= num_objects:
            if fig == None:
                fig = plt.figure()
            else:
                plt.figure(fig)

            inc2 = post[:, results.standard_param_idx['inc{}'.format(second)]]
            pan2 = post[:, results.standard_param_idx['pan{}'.format(second)]]

            cos_cop = (np.cos(inc1) * np.cos(inc2)) + (np.sin(inc1) * np.sin(inc2) * np.cos(pan1 - pan2))
            coplanarity_rad = np.arccos(cos_cop)
            coplanarity = np.rad2deg(coplanarity_rad)

            plt.hist(coplanarity, histtype='step', label="{} & {}".format(string.ascii_lowercase[2:num_objects+1][-planet], string.ascii_lowercase[1:num_objects+1][-second]))
            second += 1

    plt.title('Coplanarity')
    plt.xlabel("Mutual Inclination [deg]")
    plt.legend()

    return fig