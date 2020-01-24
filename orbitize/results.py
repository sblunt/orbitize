import numpy as np
import warnings
import h5py
import copy

import astropy.units as u
import astropy.constants as consts
from astropy.io import fits
from astropy.time import Time
from astropy._erfa.core import ErfaWarning

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

import corner

import orbitize.kepler as kepler
import orbitize.system


# define modified color map for default use in orbit plots
cmap = mpl.cm.Purples_r
cmap = colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.0, b=0.7),
    cmap(np.linspace(0.0, 0.7, 1000))
)

class Results(object):
    """
    A class to store accepted orbital configurations from the sampler

    Args:
        sampler_name (string): name of sampler class that generated these results (default: None).
        post (np.array of float): MxN array of orbital parameters
            (posterior output from orbit-fitting process), where M is the
            number of orbits generated, and N is the number of varying orbital
            parameters in the fit (default: None).
        lnlike (np.array of float): M array of log-likelihoods corresponding to
            the orbits described in ``post`` (default: None).
        tau_ref_epoch (float): date (in days, typically MJD) that tau is defined relative to
        labels (list of str): parameter labels in same order as `post`

    The ``post`` array is in the following order::

        semimajor axis 1, eccentricity 1, inclination 1,
        argument of periastron 1, position angle of nodes 1,
        epoch of periastron passage 1,
        [semimajor axis 2, eccentricity 2, etc.],
        [parallax, masses (see docstring for orbitize.system.System)]

    where 1 corresponds to the first orbiting object, 2 corresponds
    to the second, etc.

    Written: Henry Ngo, Sarah Blunt, 2018
    """
    def __init__(self, sampler_name=None, post=None, lnlike=None, tau_ref_epoch=None, labels=None):
        self.sampler_name = sampler_name
        self.post = post
        self.lnlike = lnlike
        self.tau_ref_epoch = tau_ref_epoch
        self.labels=labels
        
    def add_samples(self, orbital_params, lnlikes, labels):
        """
        Add accepted orbits and their likelihoods to the results

        Args:
            orbital_params (np.array): add sets of orbital params (could be multiple) to results
            lnlike (np.array): add corresponding lnlike values to results
            labels (list of str): list of parameter labels specifying the order in ``orbital_params``

        Written: Henry Ngo, 2018
        """
        # If no exisiting results then it is easy
        if self.post is None:
            self.post = orbital_params
            self.lnlike = lnlikes
            self.labels = labels
        # Otherwise, need to append properly
        else:
            self.post = np.vstack((self.post,orbital_params))
            self.lnlike = np.append(self.lnlike,lnlikes)

    def _set_sampler_name(self, sampler_name):
        """
        internal method to set object's sampler_name attribute
        """
        self.sampler_name = sampler_name

    def save_results(self, filename):
        """
        Save results.Results object to an hdf5 file

        Args:
            filename (string): filepath to save to

        Save attributes from the ``results.Results`` object. 

        ``sampler_name``, ``tau_ref_epcoh`` are attributes of the root group.
        ``post``, ``lnlike``, and ``parameter_labels`` are datasets 
        that are members of the root group.

        Written: Henry Ngo, 2018
        """
        hf = h5py.File(filename,'w') # Creates h5py file object
        # Add sampler_name as attribute of the root group
        hf.attrs['sampler_name']=self.sampler_name
        hf.attrs['tau_ref_epoch'] = self.tau_ref_epoch
        # Now add post and lnlike from the results object as datasets
        hf.create_dataset('post', data=self.post)
        if self.lnlike is not None: 
            hf.create_dataset('lnlike', data=self.lnlike)
        if self.labels is not None:
            hf['col_names'] = np.array(self.labels).astype('S')
        hf.close() # Closes file object, which writes file to disk

    def load_results(self, filename, append=False):
        """
        Populate the ``results.Results`` object with data from a datafile

        Args:
            filename (string): filepath where data is saved
            append (boolean): if True, then new data is added to existing object.
                If False (default), new data overwrites existing object

        See the ``save_results()`` method in this module for information on how the
        data is structured.

        Written: Henry Ngo, 2018
        """
        hf = h5py.File(filename,'r') # Opens file for reading
        # Load up each dataset from hdf5 file
        sampler_name = np.str(hf.attrs['sampler_name'])
        post = np.array(hf.get('post'))
        lnlike = np.array(hf.get('lnlike'))
        # get the tau reference epoch
        try:
            tau_ref_epoch = float(hf.attrs['tau_ref_epoch'])
        except KeyError:
            # probably a old results file when reference epoch was fixed at MJD = 0
            tau_ref_epoch = 0
        try:
            labels = np.array([hf.attrs['parameter_labels']])
        except KeyError:
            # again, probably an old file without saved parameter labels
            labels = ['sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'plx', 'mtot']

        hf.close() # Closes file object

        # Adds loaded data to object as per append keyword
        if append:
            # if no sampler_name set, use the input file's value
            if self.sampler_name is None:
                self._set_sampler_name(sampler_name)
            # otherwise only proceed if the sampler_names match
            elif self.sampler_name != sampler_name:
                raise Exception('Unable to append file {} to Results object. sampler_name of object and file do not match'.format(filename))

            # if no tau reference epoch is set, use input file's value
            if self.tau_ref_epoch is None:
                self.tau_ref_epoch = tau_ref_epoch
            # otherwise, only proceed if they are identical
            elif self.tau_ref_epoch != tau_ref_epoch:
                raise ValueError("Loaded data has tau reference epoch of {0} while Results object has already been initialized to {1}".format(tau_ref_epoch, self.tau_ref_epoch))
            if self.labels is None:
                self.labels = labels
            elif self.labels != labels:
                raise ValueError("Loaded data has parameter labels {} while Results object has already been initialized to {}.".format(labels, self.labels))


            # Now append post and lnlike
            self.add_samples(post, lnlike, self.labels)
        else:
            # Only proceed if object is completely empty
            if self.sampler_name is None and self.post is None and self.lnlike is None and self.tau_ref_epoch is None:
                self._set_sampler_name(sampler_name)
                self.add_samples(post, lnlike, self.labels)
                self.tau_ref_epoch = tau_ref_epoch
                self.labels = labels
            else:
                raise Exception('Unable to load file {} to Results object. append is set to False but object is not empty'.format(filename))


    def plot_corner(self, param_list=None, **corner_kwargs):
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
                    [repeat for 2, 3, 4, etc if multiple objects]
                    plx:  parallax
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
            'sma':'a [au]',
            'ecc':'ecc',
            'inc':'inc [$^{\\circ}$]',
            'aop':'$\\omega$ [$^{\\circ}$]',
            'pan':'$\\Omega$ [$^{\\circ}$]',
            'tau':'$\\tau$',
            'plx':'$\\pi$ [mas]',
            'mtot':'$M_T$ [M$_{\\odot}$]',
            'm0':'$M_0$ [M$_{\\odot}$]',
            'm1':'$M_1$ [M$_{\\odot}$]',
        }

        if param_list is None:
            param_list = self.labels

        param_indices = []
        angle_indices = []
        for i, param in enumerate(param_list):
            index_num = np.where(np.array(self.labels) == param)[0][0]
            param_indices.append(index_num)
            label_key = param_list[i]
            if label_key.startswith('aop') or label_key.startswith('pan') or label_key.startswith('inc'):
                angle_indices.append(index_num)

        samples = copy.copy(self.post[:,param_indices]) # keep only chains for selected parameters
        samples[:,angle_indices] = np.degrees(self.post[:,angle_indices]) # convert angles from rad to deg


        if 'labels' not in corner_kwargs: # use default labels if user didn't already supply them
            reduced_labels_list = []
            for i in np.arange(len(param_indices)):
                label_key = param_list[i]
                if label_key.startswith('m') or label_key.startswith('plx'):
                    pass
                else:
                    label_key = label_key[0:3]
                reduced_labels_list.append(default_labels[label_key])
            corner_kwargs['labels'] = reduced_labels_list

        figure = corner.corner(samples, **corner_kwargs)
        return figure


    def plot_orbits(self, object_to_plot=1, start_mjd=51544.,
                    num_orbits_to_plot=100, num_epochs_to_plot=100,
                    square_plot=True, show_colorbar=True, cmap=cmap,
                    sep_pa_color='lightgrey', sep_pa_end_year=2025.0,
                    cbar_param='epochs', mod180=False):

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
            cbar_param (string): options are the following: epochs, sma1, ecc1, inc1, aop1,
                pan1, tau1. Number can be switched out. Default is epochs.
            mod180 (Bool): if True, PA will be plotted in range [180, 540]. Useful for plotting short
                arcs with PAs that cross 360 deg during observations (default: False)

        Return:
            ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


        (written): Henry Ngo, Sarah Blunt, 2018
        Additions by Malena Rice, 2019

        """

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ErfaWarning)

            dict_of_indices = {
                'sma': 0,
                'ecc': 1,
                'inc': 2,
                'aop': 3,
                'pan': 4,
                'tau': 5,
                'plx':6
            }

            if cbar_param == 'epochs':
                pass
            elif cbar_param[0:3] in dict_of_indices:
                try:
                    object_id = np.int(cbar_param[3:])
                except ValueError:
                    object_id = 1

                index = dict_of_indices[cbar_param[0:3]] + 6*(object_id-1)
            else:
                raise Exception('Invalid input; acceptable inputs include epochs, sma1, ecc1, inc1, aop1, pan1, tau1, sma2, ecc2, ...')


            # Split the 2-D post array into series of 1-D arrays for each orbital parameter
            num_objects, remainder = np.divmod(self.post.shape[1],6)
            if object_to_plot > num_objects:
                return None

            sma = self.post[:,dict_of_indices['sma']]
            ecc = self.post[:,dict_of_indices['ecc']]
            inc = self.post[:,dict_of_indices['inc']]
            aop = self.post[:,dict_of_indices['aop']]
            pan = self.post[:,dict_of_indices['pan']]
            tau = self.post[:,dict_of_indices['tau']]
            plx = self.post[:,dict_of_indices['plx']]

            # Then, get the other parameters
            if 'mtot' in self.labels:
                mtot = self.post[:,-1]
            elif 'm0' in self.labels:
                m0 = self.post[:,-1]
                mplanet = self.post[:,-2]
                mtot = m0 + mplanet

            # Select random indices for plotted orbit
            if num_orbits_to_plot > len(sma):
                num_orbits_to_plot = len(sma)
            choose = np.random.randint(0, high=len(sma), size=num_orbits_to_plot)

            raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
            deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
            epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

            # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
            # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                # Compute period (from Kepler's third law)
                period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
                period = period.to(u.day).value
                # Create an epochs array to plot num_epochs_to_plot points over one orbital period
                epochs[i,:] = np.linspace(start_mjd, float(start_mjd+period[orb_ind]), num_epochs_to_plot)

                # Calculate ra/dec offsets for all epochs of this orbit
                raoff0, deoff0, _ = kepler.calc_orbit(
                    epochs[i,:], sma[orb_ind], ecc[orb_ind], inc[orb_ind], aop[orb_ind], pan[orb_ind],
                    tau[orb_ind], plx[orb_ind], mtot[orb_ind], tau_ref_epoch=self.tau_ref_epoch
                )

                raoff[i,:] = raoff0
                deoff[i,:] = deoff0

            # Create a linearly increasing colormap for our range of epochs
            if cbar_param != 'epochs':
                cbar_param_arr = self.post[:,index]
                norm = mpl.colors.Normalize(vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr))
                norm_yr = mpl.colors.Normalize(vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr))

            elif cbar_param == 'epochs':
                norm = mpl.colors.Normalize(vmin=np.min(epochs), vmax=np.max(epochs[-1,:]))

                norm_yr = mpl.colors.Normalize(
                vmin=np.min(Time(epochs,format='mjd').decimalyear),
                vmax=np.max(Time(epochs,format='mjd').decimalyear)
                )


            # Create figure for orbit plots
            fig = plt.figure(figsize=(14,6))

            ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)

            # Plot each orbit (each segment between two points coloured using colormap)
            for i in np.arange(num_orbits_to_plot):
                points = np.array([raoff[i,:], deoff[i,:]]).T.reshape(-1,1,2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(
                    segments, cmap=cmap, norm=norm, linewidth=1.0
                )
                if cbar_param != 'epochs':
                    lc.set_array(np.ones(len(epochs[0]))*cbar_param_arr[i])
                elif cbar_param == 'epochs':
                    lc.set_array(epochs[i,:])
                ax.add_collection(lc)

            # modify the axes
            if square_plot:
                adjustable_param='datalim'
            else:
                adjustable_param='box'
            ax.set_aspect('equal', adjustable=adjustable_param)
            ax.set_xlabel('$\\Delta$RA [mas]')
            ax.set_ylabel('$\\Delta$Dec [mas]')
            ax.locator_params(axis='x', nbins=6)
            ax.locator_params(axis='y', nbins=6)

            # add colorbar
            if show_colorbar:
                cbar_ax = fig.add_axes([0.47, 0.15, 0.015, 0.7]) # xpos, ypos, width, height, in fraction of figure size
                cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm_yr, orientation='vertical', label=cbar_param)

            # plot sep/PA zoom-in panels
            ax1 = plt.subplot2grid((2, 14), (0, 9), colspan=6)
            ax2 = plt.subplot2grid((2, 14), (1, 9), colspan=6)
            ax2.set_ylabel('PA [$^{{\\circ}}$]')
            ax1.set_ylabel('$\\rho$ [mas]')
            ax2.set_xlabel('Epoch')

            epochs_seppa = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

            for i in np.arange(num_orbits_to_plot):

                orb_ind = choose[i]


                epochs_seppa[i,:] = np.linspace(
                    start_mjd,
                    Time(sep_pa_end_year, format='decimalyear').mjd,
                    num_epochs_to_plot
                )

                # Calculate ra/dec offsets for all epochs of this orbit
                raoff0, deoff0, _ = kepler.calc_orbit(
                    epochs_seppa[i,:], sma[orb_ind], ecc[orb_ind], inc[orb_ind], aop[orb_ind], pan[orb_ind],
                    tau[orb_ind], plx[orb_ind], mtot[orb_ind], tau_ref_epoch=self.tau_ref_epoch
                )

                raoff[i,:] = raoff0
                deoff[i,:] = deoff0

                yr_epochs = Time(epochs_seppa[i,:],format='mjd').decimalyear
                plot_epochs = np.where(yr_epochs <= sep_pa_end_year)[0]
                yr_epochs = yr_epochs[plot_epochs]

                seps, pas = orbitize.system.radec2seppa(raoff[i,:], deoff[i,:], mod180=mod180)

                plt.sca(ax1)
                plt.plot(yr_epochs, seps, color=sep_pa_color)

                plt.sca(ax2)
                plt.plot(yr_epochs, pas, color=sep_pa_color)

            ax1.locator_params(axis='x', nbins=6)
            ax1.locator_params(axis='y', nbins=6)
            ax2.locator_params(axis='x', nbins=6)
            ax2.locator_params(axis='y', nbins=6)

        return fig
