import numpy as np
import h5py
import os

import astropy.table as table

import orbitize.system
import orbitize.basis
import orbitize.plot
import orbitize.gaia, orbitize.hipparcos


class Results(object):
    """
    A class to store accepted orbital configurations from the sampler

    Args:
        system (orbitize.system.System): System object used to do the fit.
        sampler_name (string): name of sampler class that generated these results 
            (default: None).
        post (np.array of float): MxN array of orbital parameters
            (posterior output from orbit-fitting process), where M is the
            number of orbits generated, and N is the number of varying orbital
            parameters in the fit (default: None).
        lnlike (np.array of float): M array of log-likelihoods corresponding to
            the orbits described in ``post`` (default: None).
        version_number (str): version of orbitize that produced these results. 
        data (astropy.table.Table): output from ``orbitize.read_input.read_file()``
        curr_pos (np.array of float): for MCMC only. A multi-D array of the 
            current walker positions that is used for restarting a MCMC sampler. 

    Written: Henry Ngo, Sarah Blunt, 2018

    API Update: Sarah Blunt, 2021
    """

    def __init__(
        self, system=None, sampler_name=None, post=None, lnlike=None,
        version_number=None, curr_pos=None
    ):

        self.system = system
        self.sampler_name = sampler_name
        self.post = post
        self.lnlike = lnlike
        self.curr_pos = curr_pos
        self.version_number = version_number

        if self.system is not None:
            self.tau_ref_epoch = self.system.tau_ref_epoch
            self.labels = self.system.labels
            self.data = self.system.data_table
            self.num_secondary_bodies = self.system.num_secondary_bodies
            self.fitting_basis = self.system.fitting_basis
            self.basis = self.system.basis
            self.param_idx = self.system.param_idx
            self.standard_param_idx = self.system.basis.standard_basis_idx

    def add_samples(self, orbital_params, lnlikes, curr_pos=None): 
        """
        Add accepted orbits, their likelihoods, and the orbitize version number 
        to the results

        Args:
            orbital_params (np.array): add sets of orbital params (could be multiple) 
                to results
            lnlike (np.array): add corresponding lnlike values to results
            curr_pos (np.array of float): for MCMC only. A multi-D array of the 
                current walker positions

        Written: Henry Ngo, 2018

        API Update: Sarah Blunt, 2021
        """
        
        # Adding the orbitize version number to the results
        if self.version_number is None:
            self.version_number = orbitize.__version__

        # If no exisiting results then it is easy
        if self.post is None:
            self.post = orbital_params
            self.lnlike = lnlikes

        # Otherwise, need to append properly
        else:
            self.post = np.vstack((self.post, orbital_params))
            self.lnlike = np.append(self.lnlike, lnlikes)

        if curr_pos is not None:
            self.curr_pos = curr_pos

    def save_results(self, filename):
        """
        Save results.Results object to an hdf5 file

        Args:
            filename (string): filepath to save to

        Save attributes from the ``results.Results`` object.

        ``sampler_name``, ``tau_ref_epcoh``, ``version_number`` are attributes of the root group.
        ``post``, ``lnlike``, and ``parameter_labels`` are datasets
        that are members of the root group.

        Written: Henry Ngo, 2018

        API Update: Sarah Blunt, 2021
        """

        hf = h5py.File(filename, 'w')  # Creates h5py file object
        # Add sampler_name as attribute of the root group

        hf.attrs['sampler_name'] = self.sampler_name
        hf.attrs['version_number'] = self.version_number

        # Now add post and lnlike from the results object as datasets
        hf.create_dataset('post', data=self.post)
        # hf.create_dataset('data', data=self.data)
        if self.lnlike is not None:
            hf.create_dataset('lnlike', data=self.lnlike)

        if self.curr_pos is not None:
            hf.create_dataset("curr_pos", data=self.curr_pos)

        self.system.save(hf)

        hf.close()  # Closes file object, which writes file to disk

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
        
        API Update: Sarah Blunt, 2021
        """

        hf = h5py.File(filename, 'r')  # Opens file for reading
        # Load up each dataset from hdf5 file
        sampler_name = np.str(hf.attrs['sampler_name'])
        try:
            version_number = np.str(hf.attrs['version_number'])
        except KeyError:
            version_number = "<= 1.13"
        post = np.array(hf.get('post'))
        lnlike = np.array(hf.get('lnlike'))


        try:
            num_secondary_bodies = int(hf.attrs['num_secondary_bodies'])
        except KeyError:
            # old, has to be single planet fit
            num_secondary_bodies = 1  

        try:
            data_table = table.Table(np.array(hf.get('data')))
        except ValueError: # old version of results; add a dummy table
            data_table = table.Table(
                names = (
                    'epoch', 'object', 'quant1', 'quant1_err', 'quant2', 
                    'quant2_err', 'quant12_corr', 'quant_type', 'instrument'
                ),
                dtype=('<f8', '<i8', '<f8', '<f8', '<f8', '<f8', '<f8', 'S5', 'S5')
            )

        try: # these are all saved keywords introduced in v2
            restrict_angle_ranges = bool(hf.attrs['restrict_angle_ranges'])
            stellar_or_system_mass = float(hf.attrs['stellar_or_system_mass'])
            mass_err = float(hf.attrs['mass_err'])
            plx_err = float(hf.attrs['plx_err'])
            plx = float(hf.attrs['plx'])
            fit_secondary_mass = bool(hf.attrs['fit_secondary_mass'])
            use_rebound = bool(hf.attrs['use_rebound'])
        except KeyError:
            restrict_angle_ranges = False
            stellar_or_system_mass = np.nan
            plx = np.nan
            plx_err = 0
            mass_err = 0
            fit_secondary_mass = False
            use_rebound = False
        try:
            tau_ref_epoch = float(hf.attrs['tau_ref_epoch'])
        except KeyError:
            # probably an old results file when reference epoch was fixed at MJD = 0
            tau_ref_epoch = 0

        iad_data = hf.get("IAD_datafile")
        if iad_data is not None:
            
            tmpfile = 'thisisprettyhackysorrylmao'
            np.array(iad_data).tofile(tmpfile)

            hip_num = str(hf.attrs['hip_num'])
            alphadec0_epoch = float(hf.attrs['alphadec0_epoch'])
            renormalize_errors = bool(hf.attrs['renormalize_errors'])

            hipparcos_IAD = orbitize.hipparcos.HipparcosLogProb(
                tmpfile, hip_num, alphadec0_epoch, renormalize_errors
            )

            os.system('rm {}'.format(tmpfile))
            try:
                gaia_num = int(hf.attrs['gaia_num'])
                dr = str(hf.attrs['dr'])
                gaia = orbitize.gaia.GaiaLogProb(gaia_num, hipparcos_IAD, dr)
            except KeyError:
                gaia = None
        else:
            hipparcos_IAD = None
            gaia = None

        try:
            fitting_basis = np.str(hf.attrs['fitting_basis'])
        except KeyError:
            # if key does not exist, then it was fit in the standard basis
            fitting_basis = 'Standard'

        self.system = orbitize.system.System(
            num_secondary_bodies, data_table, stellar_or_system_mass,
            plx, mass_err, plx_err, restrict_angle_ranges,
            tau_ref_epoch, fit_secondary_mass,
            hipparcos_IAD, gaia, fitting_basis, use_rebound
        )

        self.tau_ref_epoch = self.system.tau_ref_epoch
        self.labels = self.system.labels
        self.data = self.system.data_table
        self.num_secondary_bodies = self.system.num_secondary_bodies
        self.fitting_basis = self.system.fitting_basis
        self.basis = self.system.basis
        self.param_idx = self.system.param_idx
        self.standard_param_idx = self.basis.standard_basis_idx

        try:
            curr_pos = np.array(hf.get('curr_pos'))
        except KeyError:
            curr_pos = None

        hf.close()  # Closes file object

        # doesn't matter if append or not. Overwrite curr_pos if not None
        if curr_pos is not None:
            self.curr_pos = curr_pos

        # Adds loaded data to object as per append keyword
        if append:
            # if no sampler_name set, use the input file's value
            if self.sampler_name is None:
                self.sampler_name = sampler_name
            # otherwise only proceed if the sampler_names match
            elif self.sampler_name != sampler_name:
                raise Exception(
                    'Unable to append file {} to Results object. sampler_name of object and file do not match'.format(filename))
            # if no version_number set, use the input file's value
            if self.version_number is None:
                self.version_number = version_number
            # otherwise only proceed if the version_numbers match
            elif self.version_number != version_number:
                raise Exception(
                    'Unable to append file {} to Results object. version_number of object and file do not match'.format(filename))

            # Now append post and lnlike
            self.add_samples(post, lnlike)#, self.labels)
        else:

            # Only proceed if object is completely empty
            if self.sampler_name is None and self.post is None and self.lnlike is None and self.version_number is None:# and self.tau_ref_epoch is None :
                self.sampler_name = sampler_name
                self.version_number = version_number
                self.add_samples(post, lnlike)#, self.labels)

            else:
                raise Exception(
                    'Unable to load file {} to Results object. append is set to False but object is not empty'.format(filename))

    def print_results(self):
        """
        Prints median and 68% credible intervals alongside fitting labels
        """

        print('\nparam: med [68% CI]')
        print('-------------------\n')
        for i, l in enumerate(self.system.labels):
            print(
                '{}: {:.3f} [{:.3f} - {:.3f}]'.format(
                    l, 
                    np.median(self.post[:,i]),
                    np.quantile(self.post[:,i], 0.16),
                    np.quantile(self.post[:,i], 0.84)
                )
            )
        print('-------------------\n')
    
    def plot_corner(self, param_list=None, **corner_kwargs):
        """
        Wrapper for orbitize.plot.plot_corner
        """
        return orbitize.plot.plot_corner(self, param_list, **corner_kwargs)

    def plot_orbits(self, object_to_plot=1, start_mjd=51544.,
        num_orbits_to_plot=100, num_epochs_to_plot=100,
        square_plot=True, show_colorbar=True, cmap=orbitize.plot.cmap,
        sep_pa_color='lightgrey', sep_pa_end_year=2025.0,
        cbar_param='Epoch [year]', mod180=False, rv_time_series=False, 
        plot_astrometry=True,
        plot_astrometry_insts=False,
        plot_errorbars=True, fig=None
    ):
        """
        Wrapper for orbitize.plot.plot_orbits
        """

        return orbitize.plot.plot_orbits(
            self, object_to_plot=object_to_plot, start_mjd=start_mjd,
            num_orbits_to_plot=num_orbits_to_plot, 
            num_epochs_to_plot=num_epochs_to_plot,
            square_plot=square_plot, show_colorbar=show_colorbar, cmap=cmap,
            sep_pa_color=sep_pa_color, sep_pa_end_year=sep_pa_end_year,
            cbar_param=cbar_param, mod180=mod180, rv_time_series=rv_time_series, 
            plot_astrometry=plot_astrometry,
            plot_astrometry_insts=plot_astrometry_insts, 
            plot_errorbars=plot_errorbars, fig=fig
        )



