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

from erfa import ErfaWarning

import orbitize
import orbitize.kepler as kepler


# TODO: deprecatation warning for plots in results

# define modified color map for default use in orbit plots
cmap = mpl.cm.Purples_r
cmap = colors.LinearSegmentedColormap.from_list(
    "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=0.0, b=0.7),
    cmap(np.linspace(0.0, 0.7, 1000)),
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
        if np.std(results.post[:, index_num]) > 0:
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
        for i in param_indices:
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

    figure = corner.corner(samples, **corner_kwargs)
    return figure


def plot_orbits(
    results,
    object_to_plot=1,
    start_mjd=51544.0,
    num_orbits_to_plot=100,
    num_epochs_to_plot=100,
    square_plot=True,
    show_colorbar=True,
    cmap=cmap,
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
        fig (matplotlib.pyplot.Figure): optionally include a predefined Figure object to plot the orbit on.
            Most users will not need this keyword.

    Return:
        ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


    (written): Henry Ngo, Sarah Blunt, 2018
    Additions by Malena Rice, 2019
    Additions by Dino Hsu, 2023

    """

    if Time(start_mjd, format="mjd").decimalyear >= sep_pa_end_year:
        raise ValueError(
            "start_mjd keyword date must be less than sep_pa_end_year keyword date."
        )

    if object_to_plot > results.num_secondary_bodies:
        raise ValueError(
            "Only {0} secondary bodies being fit. Requested to plot body {1} which is out of range".format(
                results.num_secondary_bodies, object_to_plot
            )
        )

    if object_to_plot == 0:
        raise ValueError(
            "Plotting the primary's orbit is currently unsupported. Stay tuned."
        )

    if rv_time_series and "m0" not in results.labels:
        rv_time_series = False

        warnings.warn(
            "It seems that the stellar and companion mass "
            "have not been fitted separately. Setting "
            "rv_time_series=True is therefore not possible "
            "so the argument is set to False instead."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)

        data = results.data[results.data["object"] == object_to_plot]
        possible_cbar_params = ["sma", "ecc", "inc", "aop" "pan", "tau", "plx"]

        if cbar_param in ["Epoch [year]", "Epoch (year)"]:
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
        if results.sampler_name == "MCMC":
            # Convert the randomly chosen posteriors to standard keplerian set
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                param_set = np.copy(results.post[orb_ind])
                standard_post.append(results.basis.to_standard_basis(param_set))
        else:  # For OFTI, posteriors are already converted
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                standard_post.append(results.post[orb_ind])

        standard_post = np.array(standard_post)

        sma = standard_post[
            :, results.standard_param_idx["sma{}".format(object_to_plot)]
        ]
        ecc = standard_post[
            :, results.standard_param_idx["ecc{}".format(object_to_plot)]
        ]
        inc = standard_post[
            :, results.standard_param_idx["inc{}".format(object_to_plot)]
        ]
        aop = standard_post[
            :, results.standard_param_idx["aop{}".format(object_to_plot)]
        ]
        pan = standard_post[
            :, results.standard_param_idx["pan{}".format(object_to_plot)]
        ]
        tau = standard_post[
            :, results.standard_param_idx["tau{}".format(object_to_plot)]
        ]
        plx = standard_post[:, results.standard_param_idx["plx"]]

        # test gamma 3
        if rv_time_series:
            # guess the instrument name if this is not specified
            if primary_instrument_name == None:
                primary_instrument_name = results.data[results.data["object"] == 0][
                    "instrument"
                ][0]
            gamma3 = standard_post[
                :, results.standard_param_idx["gamma_" + primary_instrument_name]
            ]

        if (rv_time_series == True) or (rv_time_series2 == True):
            rv_data = results.data[results.data["object"] == 0]
            rv_data = rv_data[rv_data["quant_type"] == "rv"]

            # get list of rv instruments
            insts = np.unique(rv_data["instrument"])
            if len(insts) == 0:
                insts = ["defrv"]

            # get gamma/sigma labels and corresponding positions in the posterior
            gams = ["gamma_" + inst for inst in insts]

            if isinstance(results.labels, list):
                labels = np.array(results.labels)
            else:
                labels = results.labels

            # get the indices corresponding to each gamma within results.labels
            gam_idx = [np.where(labels == inst_gamma)[0] for inst_gamma in gams]

            gamma = standard_post[:, gam_idx]

            if (rv_time_series == True) and (rv_time_series2 == True):
                gamma2 = gamma.reshape(sma.shape)

        # Then, get the other parameters
        if "mtot" in results.labels:
            mtot = standard_post[:, results.standard_param_idx["mtot"]]
        elif "m0" in results.labels:
            m0 = standard_post[:, results.standard_param_idx["m0"]]
            m1 = standard_post[
                :, results.standard_param_idx["m{}".format(object_to_plot)]
            ]
            mtot = m0 + m1

        raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        vz_star = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
        # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
        for i in np.arange(num_orbits_to_plot):
            # Compute period (from Kepler's third law)
            period = np.sqrt(
                4 * np.pi**2.0 * (sma * u.AU) ** 3 / (consts.G * (mtot * u.Msun))
            )
            period = period.to(u.day).value

            # Create an epochs array to plot num_epochs_to_plot points over one orbital period
            epochs[i, :] = np.linspace(
                start_mjd, float(start_mjd + period[i]), num_epochs_to_plot
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            raoff0, deoff0, _ = kepler.calc_orbit(
                epochs[i, :],
                sma[i],
                ecc[i],
                inc[i],
                aop[i],
                pan[i],
                tau[i],
                plx[i],
                mtot[i],
                tau_ref_epoch=results.tau_ref_epoch,
            )

            raoff[i, :] = raoff0
            deoff[i, :] = deoff0

        # Create a linearly increasing colormap for our range of epochs
        if cbar_param not in ["Epoch [year]", "Epoch (year)"]:
            cbar_param_arr = results.post[:, index]
            norm = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )
            norm_yr = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )

        elif cbar_param in ["Epoch [year]", "Epoch (year)"]:

            min_cbar_date = np.min(epochs)
            max_cbar_date = np.max(epochs[-1, :])

            # if we're plotting orbital periods greater than 1,000 yrs, limit the colorbar dynamic range
            if max_cbar_date - min_cbar_date > 1000 * 365.25:
                max_cbar_date = min_cbar_date + 1000 * 365.25

            norm = mpl.colors.Normalize(vmin=min_cbar_date, vmax=max_cbar_date)

            norm_yr = mpl.colors.Normalize(
                vmin=Time(min_cbar_date, format="mjd").decimalyear,
                vmax=Time(max_cbar_date, format="mjd").decimalyear,
            )

        # Before starting to plot rv data, make sure rv data exists:
        rv_indices = np.where(data["quant_type"] == "rv")
        if (rv_time_series == True) or (rv_time_series2 == True):
            if len(rv_indices) == 0:
                warnings.warn("Unable to plot radial velocity data.")
                rv_time_series = False

        # Create figure for orbit plots
        if fig is None:
            fig = plt.figure(figsize=(14, 6))
            if (rv_time_series == True) and (rv_time_series2 == True):
                fig = plt.figure(figsize=(18, 16))
                ax = plt.subplot2grid((4, 18), (0, 0), rowspan=2, colspan=6)
            elif (rv_time_series == False) and (rv_time_series2 == True):
                fig = plt.figure(figsize=(16, 12))
                ax = plt.subplot2grid((3, 16), (0, 0), rowspan=2, colspan=6)
            elif (rv_time_series == True) and (rv_time_series2 == False):
                fig = plt.figure(figsize=(16, 12))
                ax = plt.subplot2grid((3, 16), (0, 0), rowspan=2, colspan=6)
            else:
                fig = plt.figure(figsize=(14, 8))
                ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)
        else:
            plt.set_current_figure(fig)
            if (rv_time_series) and (rv_time_series2):
                ax = plt.subplot2grid((4, 16), (0, 0), rowspan=2, colspan=6)
            elif (rv_time_series) and (not rv_time_series2):
                ax = plt.subplot2grid((3, 16), (0, 0), rowspan=2, colspan=6)
            elif (not rv_time_series) and (rv_time_series):
                ax = plt.subplot2grid((3, 16), (0, 0), rowspan=2, colspan=6)
            else:
                ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)

        astr_inds = np.where((~np.isnan(data["quant1"])) & (~np.isnan(data["quant2"])))
        astr_epochs = data["epoch"][astr_inds]

        radec_inds = np.where(data["quant_type"] == "radec")
        seppa_inds = np.where(data["quant_type"] == "seppa")

        sep_data, sep_err = data["quant1"][seppa_inds], data["quant1_err"][seppa_inds]
        pa_data, pa_err = data["quant2"][seppa_inds], data["quant2_err"][seppa_inds]

        if len(radec_inds[0] > 0):

            sep_from_ra_data, pa_from_dec_data = orbitize.system.radec2seppa(
                data["quant1"][radec_inds], data["quant2"][radec_inds]
            )

            num_radec_pts = len(radec_inds[0])
            sep_err_from_ra_data = np.empty(num_radec_pts)
            pa_err_from_dec_data = np.empty(num_radec_pts)
            for j in np.arange(num_radec_pts):

                sep_err_from_ra_data[j], pa_err_from_dec_data[j], _ = (
                    orbitize.system.transform_errors(
                        np.array(data["quant1"][radec_inds][j]),
                        np.array(data["quant2"][radec_inds][j]),
                        np.array(data["quant1_err"][radec_inds][j]),
                        np.array(data["quant2_err"][radec_inds][j]),
                        np.array(data["quant12_corr"][radec_inds][j]),
                        orbitize.system.radec2seppa,
                    )
                )

            sep_data = np.append(sep_data, sep_from_ra_data)
            sep_err = np.append(sep_err, sep_err_from_ra_data)

            pa_data = np.append(pa_data, pa_from_dec_data)
            pa_err = np.append(pa_err, pa_err_from_dec_data)

        # For plotting different astrometry instruments
        if plot_astrometry_insts:
            astr_colors = ("#FF7F11", "#11FFE3", "#14FF11", "#7A11FF", "#FF1919")
            astr_symbols = ("*", "o", "p", "s")

            ax_colors = itertools.cycle(astr_colors)
            ax_symbols = itertools.cycle(astr_symbols)

            astr_data = data[astr_inds]
            astr_insts = np.unique(data[astr_inds]["instrument"])

            # Indices corresponding to each instrument in datafile
            astr_inst_inds = {}
            for i in range(len(astr_insts)):
                astr_inst_inds[astr_insts[i]] = np.where(
                    astr_data["instrument"] == astr_insts[i].encode()
                )[0]

        # Plot each orbit (each segment between two points coloured using colormap)
        for i in np.arange(num_orbits_to_plot):
            points = np.array([raoff[i, :], deoff[i, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.0)
            if cbar_param not in ["Epoch [year]", "Epoch (year)"]:
                lc.set_array(np.ones(len(epochs[0])) * cbar_param_arr[i])
            elif cbar_param in ["Epoch [year]", "Epoch (year)"]:
                lc.set_array(epochs[i, :])
            ax.add_collection(lc)

        if plot_astrometry:
            ra_data, dec_data = orbitize.system.seppa2radec(sep_data, pa_data)

            # Plot astrometry along with instruments
            if plot_astrometry_insts:
                for i in range(len(astr_insts)):
                    ra = ra_data[astr_inst_inds[astr_insts[i]]]
                    dec = dec_data[astr_inst_inds[astr_insts[i]]]
                    ax.scatter(
                        ra,
                        dec,
                        marker=next(ax_symbols),
                        c=next(ax_colors),
                        zorder=10,
                        s=60,
                        label=astr_insts[i],
                    )
            else:
                ax.scatter(ra_data, dec_data, marker="*", c="red", zorder=10, s=60)

        # modify the axes
        if square_plot:
            adjustable_param = "datalim"
        else:
            adjustable_param = "box"

        ax.set_aspect("equal", adjustable=adjustable_param)
        ax.set_xlabel("$\\Delta$RA (mas)", fontsize=fontsize)
        ax.set_ylabel("$\\Delta$Dec (mas)", fontsize=fontsize)
        ax.locator_params(axis="x", nbins=6)
        ax.locator_params(axis="y", nbins=6)
        ax.invert_xaxis()  # To go to a left-handed coordinate system

        # plot sep/PA and/or rv zoom-in panels
        if (rv_time_series == True) and (rv_time_series2 == True):
            ax1 = plt.subplot2grid((4, 16), (0, 8), colspan=8)
            ax2 = plt.subplot2grid((4, 16), (1, 8), colspan=8)
            ax3 = plt.subplot2grid((4, 16), (2, 0), colspan=16, rowspan=1)
            ax4 = plt.subplot2grid((4, 16), (3, 0), colspan=16, rowspan=1)
            ax2.set_ylabel("PA ($^{{\\circ}}$)", fontsize=fontsize)
            ax1.set_ylabel("$\\rho$ (mas)", fontsize=fontsize)
            ax3.set_ylabel("Primary RV (km/s)", fontsize=fontsize)
            ax3.set_xlabel("Epoch", fontsize=fontsize)
            ax2.set_xlabel("Epoch", fontsize=fontsize)

            ax4.set_ylabel("Companion RV (km/s)", fontsize=fontsize)
            ax4.set_xlabel("Epoch", fontsize=fontsize)
            plt.subplots_adjust(hspace=0.3)

        elif (rv_time_series == True) and (rv_time_series2 == False):
            ax1 = plt.subplot2grid((3, 14), (0, 8), colspan=6)
            ax2 = plt.subplot2grid((3, 14), (1, 8), colspan=6)
            ax3 = plt.subplot2grid((3, 14), (2, 0), colspan=14, rowspan=1)
            ax2.set_ylabel("PA ($^{{\\circ}}$)", fontsize=fontsize)
            ax1.set_ylabel("$\\rho$ (mas)", fontsize=fontsize)
            ax3.set_ylabel("Primary RV (km/s)", fontsize=fontsize)
            ax3.set_xlabel("Epoch", fontsize=fontsize)
            ax2.set_xlabel("Epoch", fontsize=fontsize)
            plt.subplots_adjust(hspace=0.3)

        elif (rv_time_series == False) and (rv_time_series2 == True):
            ax1 = plt.subplot2grid((3, 14), (0, 8), colspan=6)
            ax2 = plt.subplot2grid((3, 14), (1, 8), colspan=6)
            ax3 = plt.subplot2grid((3, 14), (2, 0), colspan=14, rowspan=1)
            ax2.set_ylabel("PA ($^{{\\circ}}$)", fontsize=fontsize)
            ax1.set_ylabel("$\\rho$ (mas)", fontsize=fontsize)
            ax3.set_ylabel("Companion RV (km/s)", fontsize=fontsize)
            ax3.set_xlabel("Epoch", fontsize=fontsize)
            ax2.set_xlabel("Epoch", fontsize=fontsize)
            plt.subplots_adjust(hspace=0.3)
        else:
            ax1 = plt.subplot2grid((2, 14), (0, 9), colspan=6)
            ax2 = plt.subplot2grid((2, 14), (1, 9), colspan=6)
            ax2.set_ylabel("PA [$^{{\\circ}}$]", fontsize=fontsize)
            ax1.set_ylabel("$\\rho$ (mas)", fontsize=fontsize)
            ax2.set_xlabel("Epoch", fontsize=fontsize)

        if plot_astrometry_insts:
            ax1_colors = itertools.cycle(astr_colors)
            ax1_symbols = itertools.cycle(astr_symbols)

            ax2_colors = itertools.cycle(astr_colors)
            ax2_symbols = itertools.cycle(astr_symbols)

        epochs_seppa = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        for i in np.arange(num_orbits_to_plot):

            epochs_seppa[i, :] = np.linspace(
                start_mjd,
                Time(sep_pa_end_year, format="decimalyear").mjd,
                num_epochs_to_plot,
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            if (rv_time_series == True) or (rv_time_series2 == True):
                raoff0, deoff0, vz = kepler.calc_orbit(
                    epochs_seppa[i, :],
                    sma[i],
                    ecc[i],
                    inc[i],
                    aop[i],
                    pan[i],
                    tau[i],
                    plx[i],
                    mtot[i],
                    tau_ref_epoch=results.tau_ref_epoch,
                    mass_for_Kamp=m0[i],
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0
            else:
                raoff0, deoff0, _ = kepler.calc_orbit(
                    epochs_seppa[i, :],
                    sma[i],
                    ecc[i],
                    inc[i],
                    aop[i],
                    pan[i],
                    tau[i],
                    plx[i],
                    mtot[i],
                    tau_ref_epoch=results.tau_ref_epoch,
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0

            yr_epochs = Time(epochs_seppa[i, :], format="mjd").decimalyear

            seps, pas = orbitize.system.radec2seppa(
                raoff[i, :], deoff[i, :], mod180=mod180
            )

            plt.sca(ax1)
            plt.plot(yr_epochs, seps, color=sep_pa_color)

            plt.sca(ax2)
            plt.plot(yr_epochs, pas, color=sep_pa_color)

            # plot RV orbits here
            if rv_time_series == True:
                plt.sca(ax3)

                # scale back to primary RV semi amplitude
                vz0 = vz * (-(mtot[i] - m0[i]) / np.median(m0[i]))

                epochs_rv = np.linspace(
                    rv_data["epoch"][0] - 3 * 365,
                    epochs_seppa[0, -1],
                    num_epochs_to_plot,
                )

                plt.plot(
                    Time(epochs_rv, format="mjd").decimalyear,
                    vz0 + gamma3[i],
                    color=sep_pa_color,
                )

            if rv_time_series2 == True:
                if rv_time_series == True:
                    plt.sca(ax4)
                else:
                    plt.sca(ax3)

                # scale back to primary RV semi amplitude
                if rv_time_series:
                    epochs_rv = np.linspace(
                        rv_data["epoch"][0] - 3 * 365,
                        epochs_seppa[0, -1],
                        num_epochs_to_plot,
                    )

                    plt.plot(
                        Time(epochs_rv, format="mjd").decimalyear,
                        vz,
                        color=sep_pa_color,
                    )
                else:
                    rv_data2 = results.data[results.data["object"] == 1]
                    rv_data2 = rv_data2[rv_data2["quant_type"] == "rv"]

                    epochs_rv2 = np.linspace(
                        rv_data2["epoch"][0] - 3 * 365,
                        epochs_seppa[0, -1],
                        num_epochs_to_plot,
                    )

                    plt.plot(
                        Time(epochs_rv2, format="mjd").decimalyear,
                        vz,
                        color=sep_pa_color,
                    )

        # Plot sep/pa instruments
        if plot_astrometry_insts:
            for i in range(len(astr_insts)):
                sep = sep_data[astr_inst_inds[astr_insts[i]]]
                pa = pa_data[astr_inst_inds[astr_insts[i]]]
                epochs = astr_epochs[astr_inst_inds[astr_insts[i]]]

                serr = sep_err[astr_inst_inds[astr_insts[i]]]
                perr = pa_err[astr_inst_inds[astr_insts[i]]]

                plt.sca(ax1)
                plt.scatter(
                    Time(epochs, format="mjd").decimalyear,
                    sep,
                    s=10,
                    marker=next(ax1_symbols),
                    c=next(ax1_colors),
                    zorder=10,
                    label=astr_insts[i],
                )
                plt.errorbar(
                    Time(epochs, format="mjd").decimalyear,
                    sep,
                    yerr=serr,
                    ms=5,
                    linestyle="",
                    ecolor=next(ax1_colors),
                    zorder=10,
                    capsize=2,
                )
                plt.sca(ax2)
                plt.scatter(
                    Time(epochs, format="mjd").decimalyear,
                    pa,
                    s=10,
                    marker=next(ax2_symbols),
                    c=next(ax2_colors),
                    zorder=10,
                )
                plt.errorbar(
                    Time(epochs, format="mjd").decimalyear,
                    pa,
                    yerr=perr,
                    ms=5,
                    linestyle="",
                    marker=next(ax2_symbols),
                    ecolor=next(ax2_colors),
                    zorder=10,
                    capsize=2,
                )
            plt.sca(ax1)
            plt.legend(fontsize=15, loc=1)
        else:
            plt.sca(ax1)
            plt.scatter(
                Time(astr_epochs, format="mjd").decimalyear,
                sep_data,
                s=60,
                marker="*",
                c="red",
                zorder=10,
            )
            plt.errorbar(
                Time(astr_epochs, format="mjd").decimalyear,
                sep_data,
                yerr=sep_err,
                ms=5,
                linestyle="",
                ecolor="red",
                zorder=10,
                capsize=2,
            )
            plt.sca(ax2)
            plt.scatter(
                Time(astr_epochs, format="mjd").decimalyear,
                pa_data,
                s=60,
                marker="*",
                c="red",
                zorder=10,
            )
            plt.errorbar(
                Time(astr_epochs, format="mjd").decimalyear,
                pa_data,
                yerr=pa_err,
                ms=5,
                linestyle="",
                ecolor="red",
                zorder=10,
                capsize=2,
            )

        if rv_time_series == True:

            rv_data = results.data[results.data["object"] == 0]
            rv_data = rv_data[rv_data["quant_type"] == "rv"]

            # switch current axis to rv panel
            plt.sca(ax3)

            # get list of rv instruments
            insts = np.unique(rv_data["instrument"])
            if len(insts) == 0:
                insts = ["defrv"]

            # get gamma/sigma labels and corresponding positions in the posterior
            gams = ["gamma_" + inst for inst in insts]

            if isinstance(results.labels, list):
                labels = np.array(results.labels)
            else:
                labels = results.labels

            # get the indices corresponding to each gamma within results.labels
            gam_idx = [np.where(labels == inst_gamma)[0] for inst_gamma in gams]

            # indices corresponding to each instrument in the datafile
            inds = {}
            for i in range(len(insts)):
                inds[insts[i]] = np.where(rv_data["instrument"] == insts[i].encode())[0]

            # choose the orbit with the best log probability
            best_like = np.where(results.lnlike == np.amax(results.lnlike))[0][0]

            med_ga = [results.post[best_like, i] for i in gam_idx]

            # Get the posteriors for this index and convert to standard basis
            best_post = results.basis.to_standard_basis(results.post[best_like].copy())

            # Get the masses for the best posteriors:
            best_m0 = best_post[results.standard_param_idx["m0"]]
            best_m1 = best_post[
                results.standard_param_idx["m{}".format(object_to_plot)]
            ]
            best_mtot = best_m0 + best_m1

            # colour/shape scheme scheme for rv data points
            clrs = ("#0496FF", "#372554", "#FF1053", "#3A7CA5", "#143109")
            symbols = ("o", "^", "v", "s")

            ax3_colors = itertools.cycle(clrs)
            ax3_symbols = itertools.cycle(symbols)

            # get rvs and plot them
            for i, name in enumerate(inds.keys()):
                inst_data = rv_data[inds[name]]
                rvs = inst_data["quant1"]
                epochs = inst_data["epoch"]
                epochs = Time(epochs, format="mjd").decimalyear
                # don't include this so we can plot more orbits
                # rvs -= med_ga[i]
                # rvs -= best_post[results.param_idx[gams[i]]]
                plt.scatter(
                    epochs,
                    rvs,
                    s=30,
                    marker=next(ax3_symbols),
                    c="blue",
                    label=name,
                    zorder=5,
                )
                plt.errorbar(
                    x=epochs,
                    y=rvs,
                    yerr=inst_data["quant1_err"],
                    ecolor="blue",
                    zorder=5,
                    ls="none",
                )
            if len(inds.keys()) == 1 and "defrv" in inds.keys():
                pass
            else:
                plt.legend(fontsize=20, loc=1)

            ## calculate the predicted rv trend using the best orbit
            # _, _, vz = kepler.calc_orbit(
            #    epochs_seppa[0, :],
            #    best_post[results.standard_param_idx['sma{}'.format(object_to_plot)]],
            #    best_post[results.standard_param_idx['ecc{}'.format(object_to_plot)]],
            #    best_post[results.standard_param_idx['inc{}'.format(object_to_plot)]],
            #    best_post[results.standard_param_idx['aop{}'.format(object_to_plot)]],
            #    best_post[results.standard_param_idx['pan{}'.format(object_to_plot)]],
            #    best_post[results.standard_param_idx['tau{}'.format(object_to_plot)]],
            #    best_post[results.standard_param_idx['plx']], best_mtot,
            #    tau_ref_epoch=results.tau_ref_epoch, mass_for_Kamp=best_m0
            # )
            #
            #
            ## scale to the RV semiampltude of primary
            # vz=vz*-(best_m1)/np.median(best_m0)
            #
            ## plot rv trend
            # plt.plot(Time(epochs_seppa[0, :],format='mjd').decimalyear, vz, color=sep_pa_color)

        if rv_time_series2 == True:
            if rv_time_series == False:
                # get list of rv instruments
                insts = np.unique(rv_data["instrument"])
                if len(insts) == 0:
                    insts = ["defrv"]

                # get gamma/sigma labels and corresponding positions in the posterior
                gams = ["gamma_" + inst for inst in insts]

                if isinstance(results.labels, list):
                    labels = np.array(results.labels)
                else:
                    labels = results.labels

                # get the indices corresponding to each gamma within results.labels
                gam_idx = [np.where(labels == inst_gamma)[0] for inst_gamma in gams]

                # indices corresponding to each instrument in the datafile
                inds = {}
                for i in range(len(insts)):
                    inds[insts[i]] = np.where(
                        rv_data["instrument"] == insts[i].encode()
                    )[0]

                # choose the orbit with the best log probability
                best_like = np.where(results.lnlike == np.amax(results.lnlike))[0][0]
                med_ga = [results.post[best_like, i] for i in gam_idx]

                # Get the posteriors for this index and convert to standard basis
                best_post = results.basis.to_standard_basis(
                    results.post[best_like].copy()
                )

                # Get the masses for the best posteriors:
                best_m0 = best_post[results.standard_param_idx["m0"]]
                best_m1 = best_post[
                    results.standard_param_idx["m{}".format(object_to_plot)]
                ]
                best_mtot = best_m0 + best_m1

                # colour/shape scheme scheme for rv data points
                clrs = ("#0496FF", "#372554", "#FF1053", "#3A7CA5", "#143109")
                symbols = ("o", "^", "v", "s")

                ax3_colors = itertools.cycle(clrs)
                ax3_symbols = itertools.cycle(symbols)

            rv_data2 = results.data[results.data["object"] == 1]
            rv_data2 = rv_data2[rv_data2["quant_type"] == "rv"]

            # get list of rv2 instruments
            insts2 = np.unique(rv_data2["instrument"])

            inds2 = {}
            for i in range(len(insts2)):
                inds2[insts2[i]] = np.where(
                    rv_data2["instrument"] == insts2[i].encode()
                )[0]

            if rv_time_series == True:
                plt.sca(ax4)
            else:
                plt.sca(ax3)

            # get rvs and plot them
            for i, name in enumerate(inds2.keys()):
                inst_data2 = rv_data2[inds2[name]]
                rvs2 = inst_data2["quant1"]
                epochs2 = inst_data2["epoch"]
                epochs2 = Time(epochs2, format="mjd").decimalyear
                # don't include this so we can plot more orbits
                # rvs -= med_ga[i]
                # rvs -= best_post[results.param_idx[gams[i]]]
                plt.scatter(
                    epochs2,
                    rvs2,
                    s=30,
                    marker=next(ax3_symbols),
                    c="blue",
                    label=name.replace("_", " "),
                    zorder=5,
                )
                plt.errorbar(
                    x=epochs2,
                    y=rvs2,
                    yerr=inst_data2["quant1_err"],
                    ecolor="blue",
                    zorder=5,
                    ls="none",
                )
            if len(inds.keys()) == 1 and "defrv" in inds.keys():
                pass
            else:
                plt.legend(fontsize=20, loc=1)

        # add colorbar
        if show_colorbar:
            if (rv_time_series == True) or (rv_time_series2 == True):
                # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
                # You can change x1.0.05 to adjust the distance between the main image and the colorbar.
                # You can change 0.02 to adjust the width of the colorbar.
                cbar_ax = fig.add_axes(
                    [
                        ax.get_position().x1 + 0.005,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height,
                    ]
                )
                cbar = mpl.colorbar.ColorbarBase(
                    cbar_ax,
                    cmap=cmap,
                    norm=norm_yr,
                    orientation="vertical",
                    label=cbar_param,
                )
                cbar.ax.tick_params(labelsize=15)
                cbar.set_label(label=cbar_param, size=20)
            else:
                # xpos, ypos, width, height, in fraction of figure size
                cbar_ax = fig.add_axes([0.47, 0.15, 0.015, 0.7])
                cbar = mpl.colorbar.ColorbarBase(
                    cbar_ax,
                    cmap=cmap,
                    norm=norm_yr,
                    orientation="vertical",
                    label=cbar_param,
                )
                cbar.ax.tick_params(labelsize=15)
                cbar.set_label(label=cbar_param, size=20)

        # hard code custom things
        # ax2.set_xlim(2000, 2025)
        # if rv_time_series:
        #    ax3.set_xlim(1995, 2025)
        # if rv_time_series2:
        #    ax4.set_xlim(1995, 2025)

        ax1.locator_params(axis="x", nbins=6)
        ax1.locator_params(axis="y", nbins=6)
        ax2.locator_params(axis="x", nbins=6)
        ax2.locator_params(axis="y", nbins=6)

        for ax1 in fig.get_axes():
            ax1.tick_params(axis="both", labelsize=15)
            ax1.minorticks_on()

        for ax2 in fig.get_axes():
            ax2.tick_params(axis="both", labelsize=15)
            ax2.minorticks_on()

    fig.tight_layout()

    # if (rv_time_series == True) and (rv_time_series2 == True):
    #    return fig, ax1, ax2, ax3, ax4
    # elif (rv_time_series == True) and (rv_time_series2 == False):
    #    return fig, ax1, ax2, ax3
    # elif (rv_time_series == False) and (rv_time_series2 == True):
    #    return fig, ax1, ax2, ax3
    # else:
    #    return fig, ax1, ax2

    return fig


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
    """
    Plots sep/PA residuals for a set of orbits

    Args:
        my_results (orbitiez.results.Results): results to plot
        object_to_plot (int): which object to plot (default: 1)
        start_mjd (float): MJD in which to start plotting orbits (default: 51544,
            the year 2000)
        num_orbits_to_plot (int): number of orbits to plot (default: 100)
        num_epochs_to_plot (int): number of points to plot per orbit (default: 100)
        sep_pa_color (string): any valid matplotlib color string, used to set the
            color of the orbit tracks in the Sep/PA panels (default: 'lightgrey').
        sep_pa_end_year (float): decimal year specifying when to stop plotting orbit
            tracks in the Sep/PA panels (default: 2025.0).
        cbar_param (string): options are the following: 'Epoch [year]', 'sma1', 'ecc1', 'inc1', 'aop1',
            'pan1', 'tau1', 'plx. Number can be switched out. Default is Epoch [year].
        mod180 (Bool): if True, PA will be plotted in range [180, 540]. Useful for plotting short
            arcs with PAs that cross 360 deg during observations (default: False)

    Return:
        ``matplotlib.pyplot.Figure``: the residual plots

    """
    data = my_results.data[my_results.data["object"] == object_to_plot]

    possible_cbar_params = ["sma", "ecc", "inc", "aop" "pan", "tau", "plx"]
    num_orbits = len(my_results.post[:, 0])
    if num_orbits_to_plot > num_orbits:
        num_orbits_to_plot = num_orbits
    choose = np.random.randint(0, high=num_orbits, size=num_orbits_to_plot)

    standard_post = []
    if my_results.sampler_name == "MCMC":
        # Convert the randomly chosen posteriors to standard keplerian set
        for i in np.arange(num_orbits_to_plot):
            orb_ind = choose[i]
            param_set = np.copy(my_results.post[orb_ind])
            standard_post.append(my_results.basis.to_standard_basis(param_set))
    else:  # For OFTI, posteriors are already converted
        for i in np.arange(num_orbits_to_plot):
            orb_ind = choose[i]
            standard_post.append(my_results.post[orb_ind])

    standard_post = np.array(standard_post)

    sma = standard_post[
        :, my_results.standard_param_idx["sma{}".format(object_to_plot)]
    ]
    ecc = standard_post[
        :, my_results.standard_param_idx["ecc{}".format(object_to_plot)]
    ]
    inc = standard_post[
        :, my_results.standard_param_idx["inc{}".format(object_to_plot)]
    ]
    aop = standard_post[
        :, my_results.standard_param_idx["aop{}".format(object_to_plot)]
    ]
    pan = standard_post[
        :, my_results.standard_param_idx["pan{}".format(object_to_plot)]
    ]
    tau = standard_post[
        :, my_results.standard_param_idx["tau{}".format(object_to_plot)]
    ]
    plx = standard_post[:, my_results.standard_param_idx["plx"]]

    if "mtot" in my_results.labels:
        mtot = standard_post[:, my_results.standard_param_idx["mtot"]]
    elif "m0" in my_results.labels:
        m0 = standard_post[:, my_results.standard_param_idx["m0"]]
        m1 = standard_post[
            :, my_results.standard_param_idx["m{}".format(object_to_plot)]
        ]
        mtot = m0 + m1

    raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
    deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
    vz_star = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
    epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

    for i in np.arange(num_orbits_to_plot):
        # Compute period (from Kepler's third law)
        period = np.sqrt(
            4 * np.pi**2.0 * (sma * u.AU) ** 3 / (consts.G * (mtot * u.Msun))
        )
        period = period.to(u.day).value

        # Create an epochs array to plot num_epochs_to_plot points over one orbital period
        epochs[i, :] = np.linspace(
            Time(start_mjd, format="mjd").mjd,
            float(Time(start_mjd, format="mjd").mjd + period[i]),
            num_epochs_to_plot,
        )

        # Calculate ra/dec offsets for all epochs of this orbit
        raoff0, deoff0, _ = kepler.calc_orbit(
            epochs[i, :],
            sma[i],
            ecc[i],
            inc[i],
            aop[i],
            pan[i],
            tau[i],
            plx[i],
            mtot[i],
            tau_ref_epoch=my_results.tau_ref_epoch,
        )

        raoff[i, :] = raoff0
        deoff[i, :] = deoff0

    astr_inds = np.where((~np.isnan(data["quant1"])) & (~np.isnan(data["quant2"])))
    astr_epochs = data["epoch"][astr_inds]

    radec_inds = np.where(data["quant_type"] == "radec")
    seppa_inds = np.where(data["quant_type"] == "seppa")

    # transform RA/Dec points to Sep/PA
    sep_data = np.copy(data["quant1"])
    sep_err = np.copy(data["quant1_err"])
    pa_data = np.copy(data["quant2"])
    pa_err = np.copy(data["quant2_err"])

    if len(radec_inds[0] > 0):

        sep_from_ra_data, pa_from_dec_data = orbitize.system.radec2seppa(
            data["quant1"][radec_inds], data["quant2"][radec_inds]
        )

        num_radec_pts = len(radec_inds[0])
        sep_err_from_ra_data = np.empty(num_radec_pts)
        pa_err_from_dec_data = np.empty(num_radec_pts)
        for j in np.arange(num_radec_pts):

            sep_err_from_ra_data[j], pa_err_from_dec_data[j], _ = (
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

    # Transform Sep/PA points to RA/Dec
    ra_data = np.copy(data["quant1"])
    ra_err = np.copy(data["quant1_err"])
    dec_data = np.copy(data["quant2"])
    dec_err = np.copy(data["quant2_err"])

    if len(seppa_inds[0] > 0):

        ra_from_seppa_data, dec_from_seppa_data = orbitize.system.seppa2radec(
            data["quant1"][seppa_inds], data["quant2"][seppa_inds]
        )

        num_seppa_pts = len(seppa_inds[0])
        ra_err_from_seppa_data = np.empty(num_seppa_pts)
        dec_err_from_seppa_data = np.empty(num_seppa_pts)
        for j in np.arange(num_seppa_pts):

            ra_err_from_seppa_data[j], dec_err_from_seppa_data[j], _ = (
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

        epochs_seppa = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

    raoff = []
    deoff = []
    seps = []
    pas = []
    raoff_100 = []
    deoff_100 = []
    seps_100 = []
    pas_100 = []
    for i in np.arange(num_orbits_to_plot):

        epochs_seppa[i, :] = np.linspace(
            Time(start_mjd, format="mjd").mjd,
            Time(sep_pa_end_year, format="decimalyear").mjd,
            num_epochs_to_plot,
        )

        raoff0, deoff0, _ = kepler.calc_orbit(
            astr_epochs,
            sma[i],
            ecc[i],
            inc[i],
            aop[i],
            pan[i],
            tau[i],
            plx[i],
            mtot[i],
            tau_ref_epoch=my_results.tau_ref_epoch,
        )

        raoff2, deoff2, _ = kepler.calc_orbit(
            epochs_seppa[0],
            sma[i],
            ecc[i],
            inc[i],
            aop[i],
            pan[i],
            tau[i],
            plx[i],
            mtot[i],
            tau_ref_epoch=my_results.tau_ref_epoch,
        )

        raoff.append(raoff0)
        deoff.append(deoff0)
        raoff_100.append(raoff2)
        deoff_100.append(deoff2)

        seps1, pas1 = orbitize.system.radec2seppa(raoff0, deoff0, mod180=mod180)
        # seps1 = []
        # pas1 = []

        # for j in range(len(astr_epochs)):

        #     seps0, pas0 = orbitize.system.radec2seppa(raoff[i][j], deoff[i][j], mod180=mod180)

        #     seps1.append(seps0)
        #     pas1.append(pas0)

        seps.append(seps1)
        pas.append(pas1)

        seps2, pas2 = orbitize.system.radec2seppa(raoff2, deoff2, mod180=mod180)
        # seps2 = []
        # pas2 = []

        # for j in range(len(epochs_seppa[0])):

        #     seps0_100, pas0_100 = orbitize.system.radec2seppa(raoff_100[i][j], deoff_100[i][j], mod180=mod180)

        #     seps2.append(seps0_100)
        #     pas2.append(pas0_100)

        seps_100.append(seps2)
        pas_100.append(pas2)

    yr_epochs = Time(astr_epochs, format="mjd").decimalyear
    yr_epochs2 = Time(epochs_seppa[i, :], format="mjd").decimalyear

    seps = np.array(seps)
    pas = np.array(pas)
    seps_100 = np.array(seps_100)
    pas_100 = np.array(pas_100)

    median_seps = []
    median_pas = []
    median_seps_100 = []
    median_pas_100 = []

    seps_T = np.transpose(seps)
    pas_T = np.transpose(pas)
    seps_100_T = np.transpose(seps_100)
    pas_100_T = np.transpose(pas_100)

    for j in range(len(epochs_seppa[0])):
        median_seps2 = np.median(seps_100_T[j])
        median_pas2 = np.median(pas_100_T[j])

        median_seps_100.append(median_seps2)
        median_pas_100.append(median_pas2)

    for j in range(len(astr_epochs)):
        median_seps1 = np.median(seps_T[j])
        median_pas1 = np.median(pas_T[j])

        median_seps.append(median_seps1)
        median_pas.append(median_pas1)

    orbits_to_plot = np.linspace(0, num_orbits_to_plot - 1, 100)

    residual_seps = median_seps - sep_data
    residual_pas = median_pas - pa_data

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].errorbar(
        yr_epochs,
        residual_seps,
        yerr=sep_err,
        xerr=None,
        fmt="o",
        ms=5,
        linestyle="",
        c="purple",
        zorder=10,
        capsize=2,
    )
    for i in range(len(orbits_to_plot)):
        residual_seps_100 = median_seps_100 - seps_100[int(orbits_to_plot[i])]
        axes[0].plot(yr_epochs2, residual_seps_100, color=sep_pa_color, zorder=1)
    axes[0].axhline(y=0, color="black", linestyle="-")
    axes[0].set_ylabel("Residual $\\rho$ [mas]")
    axes[0].set_xlabel("Epoch")
    axes[0].set_xlim(yr_epochs2[0], yr_epochs2[-1])

    axes[1].errorbar(
        yr_epochs,
        residual_pas,
        yerr=pa_err,
        xerr=None,
        fmt="o",
        ms=5,
        linestyle="",
        c="purple",
        zorder=10,
        capsize=2,
    )
    for i in range(len(orbits_to_plot)):
        residual_pas_100 = median_pas_100 - pas_100[int(orbits_to_plot[i])]
        axes[1].plot(yr_epochs2, residual_pas_100, color=sep_pa_color, zorder=1)
    axes[1].axhline(y=0, color="black", linestyle="-")
    axes[1].set_ylabel("Residual PA [$^{{\\circ}}$]")
    axes[1].set_xlabel("Epoch")
    axes[1].set_xlim(yr_epochs2[0], yr_epochs2[-1])

    plt.tight_layout()


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
    cmap=cmap,
    cbar_param=None,
    tight_layout=False,
    # fig=None
):
    """
    Plots the proper motion of a host star as induced by a companion for
    one orbital period for a select number of fitted orbits
    for a given object, with line segments colored according to a given
    parameter (most informative is usually mass of companion)

    Important Note: These plotted trajectories aren't what are fitting in the
    likelihood evaluation for the HGCA runs. The implementation forward models
    the Hip/Gaia measurements per epoch and infers the differential proper motions.
    This plot is given only for the purposes of an approximate visualization.

    Args:
        system (object): orbitize.system object with a HGCALogProb passed to system.gaia
        object_to_plot (int): which object to plot (default: 1)
        start_mjd (float): MJD in which to start plotting orbits (default: 51544,
            the year 2000)
        periods_to_plot (int): number of periods to plot (default: 1)
        end_year (float): decimal year specifying when to stop plotting orbit
            tracks in the Sep/PA panels (default: 2025.0).
        alpha (float): transparency of lines (default: 0.05)
        num_orbits_to_plot (int): number of orbits to plot (default: 100)
        num_epochs_to_plot (int): number of points to plot per orbit (default: 100)
        show_colorbar (Boolean): Displays colorbar to the right of the plot [True]
        cmap (matplotlib.cm.ColorMap): color map to use for making orbit tracks
            (default: modified Purples_r)
        cbar_param (string): options are the following: 'sma1', 'ecc1', 'inc1', 'aop1',
            'pan1', 'tau1', 'plx', 'm0', 'm1', etc. Number can be switched out. Default is None.
        tight_layout (bool): apply plt.tight_layout function?
        fig (matplotlib.pyplot.Figure): optionally include a predefined Figure object to plot the orbit on.
            Most users will not need this keyword.

    Return:
        ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


    (written): William Balmer (2023), based on plot_orbits by Sarah Blunt and Henry Ngo

    """

    if Time(start_mjd, format="mjd").decimalyear >= end_year:
        raise ValueError(
            "start_mjd keyword date must be less than end_year keyword date."
        )

    if object_to_plot > results.num_secondary_bodies:
        raise ValueError(
            "Only {0} secondary bodies being fit. Requested to plot body {1} which is out of range".format(
                results.num_secondary_bodies, object_to_plot
            )
        )

    if object_to_plot == 0:
        raise ValueError(
            "Plotting the primary's orbit is currently unsupported. Stay tuned."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)

        data = results.data[results.data["object"] == object_to_plot]
        possible_cbar_params = [
            "sma",
            "ecc",
            "inc",
            "aop" "pan",
            "tau",
            "plx",
            "m0",
            "m1",
        ]

        if cbar_param is None:
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
        if results.sampler_name == "MCMC":
            # Convert the randomly chosen posteriors to standard keplerian set
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                param_set = np.copy(results.post[orb_ind])
                standard_post.append(results.basis.to_standard_basis(param_set))
        else:  # For OFTI, posteriors are already converted
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                standard_post.append(results.post[orb_ind])

        standard_post = np.array(standard_post)

        sma = standard_post[
            :, results.standard_param_idx["sma{}".format(object_to_plot)]
        ]
        ecc = standard_post[
            :, results.standard_param_idx["ecc{}".format(object_to_plot)]
        ]
        inc = standard_post[
            :, results.standard_param_idx["inc{}".format(object_to_plot)]
        ]
        aop = standard_post[
            :, results.standard_param_idx["aop{}".format(object_to_plot)]
        ]
        pan = standard_post[
            :, results.standard_param_idx["pan{}".format(object_to_plot)]
        ]
        tau = standard_post[
            :, results.standard_param_idx["tau{}".format(object_to_plot)]
        ]
        plx = standard_post[:, results.standard_param_idx["plx"]]

        # Then, get the other parameters
        if "mtot" in results.labels:
            mtot = standard_post[:, results.standard_param_idx["mtot"]]
        elif "m0" in results.labels:
            m0 = standard_post[:, results.standard_param_idx["m0"]]
            m1 = standard_post[
                :, results.standard_param_idx["m{}".format(object_to_plot)]
            ]
            mtot = m0 + m1

        raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        vz_star = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
        # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
        for i in np.arange(num_orbits_to_plot):
            # Compute period (from Kepler's third law)
            period = np.sqrt(
                4 * np.pi**2.0 * (sma * u.AU) ** 3 / (consts.G * (mtot * u.Msun))
            )
            period = period.to(u.day).value

            # Create an epochs array to plot num_epochs_to_plot points over one orbital period
            epochs[i, :] = np.linspace(
                start_mjd,
                float(start_mjd + (period[i] * periods_to_plot)),
                num_epochs_to_plot,
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            raoff0, deoff0, _ = kepler.calc_orbit(
                epochs[i, :],
                sma[i],
                ecc[i],
                inc[i],
                aop[i],
                pan[i],
                tau[i],
                plx[i],
                mtot[i],
                tau_ref_epoch=results.tau_ref_epoch,
            )

            raoff[i, :] = raoff0
            deoff[i, :] = deoff0

        # Create a linearly increasing colormap for our range of epochs
        if cbar_param is not None:
            cbar_param_arr = results.post[:, index]
            norm = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )

        elif cbar_param is None:

            norm = mpl.colors.Normalize()

        # Create figure for orbit plots
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), facecolor="white")

        # Plot each orbit (each segment between two points coloured using colormap)
        for i in np.arange(num_orbits_to_plot):
            epoch_in_yr = Time(epochs[i, :], format="mjd").decimalyear
            # masses (in same units, solar)
            m_b = standard_post[:, results.param_idx["m1"]][i]
            m_a = standard_post[:, results.param_idx["m0"]][i]
            # dt
            timestep = epoch_in_yr[1] - epoch_in_yr[0]
            # dra/dt and ddec/dt
            ddec_b = np.gradient(deoff[i, :], timestep)  # in mas/yr
            dec_b_radian = (
                deoff[i, :] * (2.7777778e-7) * (0.017453293)
            )  # mas -> deg -> radian
            ra_b = raoff[i, :]
            rastar_b = ra_b * np.cos(dec_b_radian)  # in mas
            drastar_b = np.gradient(rastar_b, timestep)  # in mas/yr

            # convert to dRA^star_star (lol) and dDec_star
            mass_ratio_ = -1 * m_b / (m_a + m_b)
            ddec_a = ddec_b * mass_ratio_
            drastar_a = drastar_b * mass_ratio_

            if cbar_param is not None:
                color = cmap(norm(standard_post[:, results.param_idx[cbar_param]][i]))
            else:
                color = "k"

            axs[0].plot(
                epoch_in_yr,
                drastar_a + system.gaia.hg_pm[0],
                color=color,
                alpha=alpha,
                zorder=0,
            )
            axs[1].plot(
                epoch_in_yr,
                ddec_a + system.gaia.hg_pm[1],
                color=color,
                alpha=alpha,
                zorder=0,
            )

    axs[0].set_xlim(1980, 2030)
    axs[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axs[1].set_xlabel("Epoch")

    axs[0].set_ylabel(r"$\mu_\alpha^*$ [mas/yr]")

    axs[0].errorbar(
        np.nanmedian(system.gaia.hipparcos_epoch),
        system.gaia.hip_pm[0],
        yerr=system.gaia.hip_pm_err[0],
        zorder=30,
        mec="k",
        fmt="s",
        color="cornflowerblue",
    )

    hgca_epoch = (
        system.gaia.gaia_epoch_ra + np.nanmedian(system.gaia.hipparcos_epoch)
    ) / 2
    hgca_epoch_err = (
        system.gaia.gaia_epoch_ra - np.nanmedian(system.gaia.hipparcos_epoch)
    ) / 2

    axs[0].errorbar(
        hgca_epoch,
        system.gaia.hg_pm[0],
        xerr=hgca_epoch_err,
        yerr=system.gaia.hg_pm_err[0],
        zorder=30,
        mec="k",
        fmt="^",
        color="#6280D6",
    )

    axs[0].errorbar(
        system.gaia.gaia_epoch_ra,
        system.gaia.gaia_pm[0],
        yerr=system.gaia.gaia_pm_err[0],
        zorder=30,
        mec="k",
        fmt="o",
        color="#5f61b4",
    )

    axs[1].set_xlim(1980, 2030)
    axs[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    axs[1].errorbar(
        np.nanmedian(system.gaia.hipparcos_epoch),
        system.gaia.hip_pm[1],
        yerr=system.gaia.hip_pm_err[1],
        zorder=30,
        mec="k",
        fmt="s",
        color="cornflowerblue",
        label="Hip.",
    )

    axs[1].errorbar(
        hgca_epoch,
        system.gaia.hg_pm[1],
        xerr=hgca_epoch_err,
        yerr=system.gaia.hg_pm_err[1],
        zorder=30,
        mec="k",
        fmt="^",
        color="#6280D6",
        label="H-G",
    )

    axs[1].errorbar(
        system.gaia.gaia_epoch_ra,
        system.gaia.gaia_pm[1],
        yerr=system.gaia.gaia_pm_err[1],
        zorder=30,
        mec="k",
        fmt="o",
        color="#5f61b4",
        label="Gaia",
    )

    axs[1].set_ylabel(r"$\mu_\delta$ [mas/yr]")
    axs[1].set_xlabel("Epoch")
    axs[0].set_xlabel("Epoch")

    cbar_ax = fig.add_axes([1.03, 0.15, 0.03, 0.80])

    cbar = mpl.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm, orientation="vertical", label=cbar_param
    )

    axs[0].set_rasterization_zorder(1)
    axs[1].set_rasterization_zorder(1)

    axs[1].legend()

    print(
        "Important Note of Caution: the orbitize! implementation of the HGCA \n",
        "fits for the time-averaged proper motions, and not the instantaneous proper \n",
        "motions that are being plotted here. This plot is provided only for the \n",
        "purpose of an approximate check on the fit.",
    )

    if tight_layout:
        plt.tight_layout()

    return fig
