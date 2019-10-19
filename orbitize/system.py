import numpy as np
from orbitize import priors, read_input, kepler

class System(object):
    """
    A class to store information about a system (data & priors)
    and calculate model predictions given a set of orbital
    parameters.

    Args:
        num_secondary_bodies (int): number of secondary bodies in the system.
            Should be at least 1.
        data_table (astropy.table.Table): output from ``orbitize.read_input.read_file()``
        stellar_mass (float): mean mass of the primary, in M_sol. See `fit_secondary_mass`
            docstring below.
        plx (float): mean parallax of the system, in mas
        mass_err (float, optional): uncertainty on ``stellar_mass``, in M_sol
        plx_err (float, optional): uncertainty on ``plx``, in mas
        restrict_angle_ranges (bool, optional): if True, restrict the ranges
            of the position angle of nodes to [0,180)
            to get rid of symmetric double-peaks for imaging-only datasets.
        tau_ref_epoch (float, optional): reference epoch for defining tau (MJD).
            Default is 58849 (Jan 1, 2020).
        fit_secondary_mass (bool, optional): if True, include the dynamical 
            mass of the orbiting body as a fitted parameter. If this is set to False, ``stellar_mass``
            is taken to be the total mass of the system. (default: False)
        results (list of orbitize.results.Results): results from an orbit-fit
            will be appended to this list as a Results class

    Users should initialize an instance of this class, then overwrite
    priors they wish to customize.

    Priors are initialized as a list of ``orbitize.priors.Prior`` objects,
    in the following order::

        semimajor axis 1, eccentricity 1, inclination 1,
        argument of periastron 1, position angle of nodes 1,
        epoch of periastron passage 1,
        [semimajor axis 2, eccentricity 2, etc.],
        [parallax, [mass1, mass2, ..], total_mass/m0]

    where 1 corresponds to the first orbiting object, 2 corresponds
    to the second, etc. Mass1, mass2, ... correspond to masses of secondary
    bodies. If `fit_secondary_mass` is set to True, the last element of this 
    list is initialized to the mass of the primary. If not, it is 
    initialized to the total system mass.

    Written: Sarah Blunt, Henry Ngo, Jason Wang, 2018
    """
    def __init__(self, num_secondary_bodies, data_table, stellar_mass,
                 plx, mass_err=0, plx_err=0, restrict_angle_ranges=None,
                 tau_ref_epoch=58849, fit_secondary_mass=False, results=None):

        self.num_secondary_bodies = num_secondary_bodies
        self.sys_priors = []
        self.labels = []
        self.results = []
        self.fit_secondary_mass = fit_secondary_mass
        self.tau_ref_epoch = tau_ref_epoch

        #
        # Group the data in some useful ways
        #

        self.data_table = data_table
        # Creates a copy of the input in case data_table needs to be modified
        self.input_table = self.data_table.copy()

        # List of arrays of indices corresponding to each body
        self.body_indices = []

        # List of arrays of indices corresponding to epochs in RA/Dec for each body
        self.radec = []

        # List of arrays of indices corresponding to epochs in SEP/PA for each body
        self.seppa = []

        radec_indices = np.where(self.data_table['quant_type']=='radec')
        seppa_indices = np.where(self.data_table['quant_type']=='seppa')

        for body_num in np.arange(self.num_secondary_bodies+1):

            self.body_indices.append(
                np.where(self.data_table['object']==body_num)
            )

            self.radec.append(
                np.intersect1d(self.body_indices[body_num], radec_indices)
            )
            self.seppa.append(
                np.intersect1d(self.body_indices[body_num], seppa_indices)
            )

        if restrict_angle_ranges:
            angle_upperlim = np.pi
        else:
            angle_upperlim = 2.*np.pi

        #
        # Set priors for each orbital element
        #

        for body in np.arange(num_secondary_bodies):
            # Add semimajor axis prior
            self.sys_priors.append(priors.LogUniformPrior(0.001, 1e7))
            self.labels.append('sma{}'.format(body+1))

            # Add eccentricity prior
            self.sys_priors.append(priors.UniformPrior(0.,1.))
            self.labels.append('ecc{}'.format(body+1))

            # Add inclination angle prior
            self.sys_priors.append(priors.SinPrior())
            self.labels.append('inc{}'.format(body+1))

            # Add argument of periastron prior
            self.sys_priors.append(priors.UniformPrior(0.,2.*np.pi))
            self.labels.append('aop{}'.format(body+1))

            # Add position angle of nodes prior
            self.sys_priors.append(priors.UniformPrior(0.,angle_upperlim))
            self.labels.append('pan{}'.format(body+1))

            # Add epoch of periastron prior.
            self.sys_priors.append(priors.UniformPrior(0., 1.))
            self.labels.append('tau{}'.format(body+1))

        #
        # Set priors on total mass and parallax
        #
        self.labels.append('plx')
        if plx_err > 0:
            self.sys_priors.append(priors.GaussianPrior(plx, plx_err))
        else:
            self.sys_priors.append(plx)
        
        if self.fit_secondary_mass:
            for body in np.arange(num_secondary_bodies):
                self.sys_priors.append(priors.LogUniformPrior(1e-6, 1)) # in Solar masses for now
                self.labels.append('m{}'.format(body))
            self.labels.append('m0')
        else:
            self.labels.append('mtot')

        # still need to append m0/mtot, even though labels are appended above
        if mass_err > 0:
            self.sys_priors.append(priors.GaussianPrior(stellar_mass, mass_err))
        else:
            self.sys_priors.append(stellar_mass)

        # add labels dictionary for parameter indexing
        self.param_idx = dict(zip(self.labels, np.arange(len(self.labels))))


    def compute_model(self, params_arr):
        """
        Compute model predictions for an array of fitting parameters.

        Args:
            params_arr (np.array of float): RxM array
                of fitting parameters, where R is the number of
                parameters being fit, and M is the number of orbits
                we need model predictions for. Must be in the same order
                documented in ``System()`` above. If M=1, this can be a 1d array.

        Returns:
            np.array of float: Nobsx2xM array model predictions. If M=1, this is
            a 2d array, otherwise it is a 3d array.
        """

        if len(params_arr.shape) == 1:
            model = np.zeros((len(self.data_table), 2))
        else:
            model = np.zeros((len(self.data_table), 2, params_arr.shape[1]))


        for body_num in np.arange(self.num_secondary_bodies)+1:

            epochs = self.data_table['epoch'][self.body_indices[body_num]]
            sma = params_arr[body_num-1]
            ecc = params_arr[body_num]
            inc = params_arr[body_num+1]
            argp = params_arr[body_num+2]
            lan = params_arr[body_num+3]
            tau = params_arr[body_num+4]
            plx = params_arr[6*self.num_secondary_bodies]
            if self.fit_secondary_mass:
                # mass of secondary bodies are in order from -1-num_bodies until -2 in order.
                mass = params_arr[-1-self.num_secondary_bodies+(body_num-1)]
                m0 = params_arr[-1]
                mtot = m0 + mass
            else:
                mass = None
                mtot = params_arr[-1]

            raoff, decoff, vz = kepler.calc_orbit(
                epochs, sma, ecc, inc, argp, lan, tau, plx, mtot, mass_for_Kamp=mass, tau_ref_epoch=self.tau_ref_epoch
            )

            if len(raoff[self.radec[body_num]]) > 0: # (prevent empty array dimension errors)
                model[self.radec[body_num], 0] = raoff[self.radec[body_num]]
                model[self.radec[body_num], 1] = decoff[self.radec[body_num]]

            if len(raoff[self.seppa[body_num]]) > 0:
                sep, pa = radec2seppa(
                    raoff[self.seppa[body_num]],
                    decoff[self.seppa[body_num]]
                )

                model[self.seppa[body_num], 0] = sep
                model[self.seppa[body_num], 1] = pa

        return model

    def convert_data_table_radec2seppa(self,body_num=1):
        """
        Converts rows of self.data_table given in radec to seppa.
        Note that self.input_table remains unchanged.

        Args:
            body_num (int): which object to convert (1 = first planet)
        """
        for i in self.radec[body_num]: # Loop through rows where input provided in radec
            # Get ra/dec values
            ra = self.data_table['quant1'][i]
            ra_err = self.data_table['quant1_err'][i]
            dec = self.data_table['quant2'][i]
            dec_err = self.data_table['quant2_err'][i]
            # Convert to sep/PA
            sep, pa = radec2seppa(ra,dec)
            sep_err = 0.5*(ra_err+dec_err)
            pa_err = np.degrees(sep_err/sep)
            # Update data_table
            self.data_table['quant1'][i]=sep
            self.data_table['quant1_err'][i]=sep_err
            self.data_table['quant2'][i]=pa
            self.data_table['quant2_err'][i]=pa_err
            self.data_table['quant_type'][i]='seppa'
            # Update self.radec and self.seppa arrays
            self.radec[body_num]=np.delete(self.radec[body_num],np.where(self.radec[body_num]==i)[0])
            self.seppa[body_num]=np.append(self.seppa[body_num],i)


    def add_results(self, results):
        """
        Adds an orbitize.results.Results object to the list in system.results

        Args:
            results (orbitize.results.Results object): add this object to list
        """
        self.results.append(results)

    def clear_results(self):
        """
        Removes all stored results
        """
        self.results = []

def radec2seppa(ra, dec, mod180=False):
    """
    Convenience function for converting from
    right ascension/declination to separation/
    position angle.

    Args:
        ra (np.array of float): array of RA values, in mas
        dec (np.array of float): array of Dec values, in mas
        mod180 (Bool): if True, output PA values will be given
            in range [180, 540) (useful for plotting short
            arcs with PAs that cross 360 during observations)
            (default: False)


    Returns:
        tulple of float: (separation [mas], position angle [deg])

    """
    sep = np.sqrt((ra**2) + (dec**2))
    pa = np.degrees(np.arctan2(ra, dec)) % 360.

    if mod180:
        pa[pa < 180] += 360

    return sep, pa
