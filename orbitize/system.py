import numpy as np
from orbitize import priors, read_input, kepler

deg2rad = 0.0174532925199433

class System(object):
    """
    A class to store information about a system (data & priors) 
    and calculate model predictions given a set of orbital 
    parameters.

    Args:
        num_secondary_bodies (int): number of secondary bodies in the system. 
            Should be at least 1.
        data_table (astropy.table.Table): output from either
            ``orbitize.read_input.read_formatted_file()`` or 
            ``orbitize.read_input.read_orbitize_input()``
        system_mass (float): mean total mass of the system, in M_sol
        plx (float): mean parallax of the system, in arcsec
        mass_err (float [optional]): uncertainty on ``system_mass``, in M_sol
        plx_err (float [optional]): uncertainty on ``plx``, in arcsec
        restrict_angle_ranges (bool [optional]): if True, restrict the ranges
            of PAN and AOP to [0,180) to get rid of symmetric double-peaks for
            imaging-only datasets.

    Users should initialize an instance of this class, then overwrite 
    priors they wish to customize. 

    Priors are initialized as a list of orbitize.priors.Prior objects,
    in the following order:

        semimajor axis b, eccentricity b, AOP b, PAN b, inclination b, EPP b, 
        [semimajor axis c, eccentricity c, etc.]
        mass, parallax

    where `b` corresponds to the first orbiting object, `c` corresponds
    to the second, etc. 

    (written): Sarah Blunt, 2018
    """
    def __init__(self, num_secondary_bodies, data_table, system_mass, 
                 plx, mass_err=0, plx_err=0, restrict_angle_ranges=False):

        self.num_secondary_bodies = num_secondary_bodies
        self.sys_priors = []

        if restrict_angle_ranges:
            angle_upperlim = np.pi
        else:
            angle_upperlim = 2.*np.pi

        # Set priors for each orbital element
        for body in np.arange(num_secondary_bodies):
            # Add semimajor axis prior
            self.sys_priors.append(priors.JeffreysPrior(0.1, 100.))

            # Add eccentricity prior
            self.sys_priors.append(priors.UniformPrior(0.,1.))

            # Add argument of periastron prior
            self.sys_priors.append(priors.UniformPrior(0.,angle_upperlim))

            # Add position angle of nodes prior
            self.sys_priors.append(priors.UniformPrior(0.,angle_upperlim))

            # Add inclination angle prior
            self.sys_priors.append(priors.SinPrior())

            # Add epoch of periastron prior. 
            self.sys_priors.append(priors.UniformPrior(0., 1.))

        # Set priors on system mass and parallax
        if mass_err > 0:
            self.sys_priors.append(priors.GaussianPrior(
                system_mass, mass_err)
            )
            self.abs_system_mass = None
            self.abs_system_mass = np.nan
        else:
            self.abs_system_mass = system_mass
        if plx_err > 0:
            self.sys_priors.append(priors.GaussianPrior(plx, plx_err))
            self.abs_system_mass = None
            self.abs_plx = np.nan
        else:
            self.abs_plx = plx



        # Group the data in some useful ways

        self.data_table = data_table

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


    def compute_model(self, params_arr):
        """
        Compute model predictions for an array of fitting parameters.

        Args:
            params_arr (np.array of float): RxM array 
                of fitting parameters, where R is the number of 
                parameters being fit, and M is the number of orbits
                we need model predictions for. Must be in the same order
                documented in System() above. If M=1, this can be a 1d array.

        Returns:
            np.array of float: Nobsx2xM array model predictions. If M=1, this is 
                a 2d array, otherwise it is a 3d array.
        """

        if len(params_arr.shape) == 1:
            model = np.zeros((len(self.data_table), 2))        
        else:
            model = np.zeros((len(self.data_table), 2, params_arr.shape[1]))

        if not np.isnan(self.abs_plx):
            plx = self.abs_plx
        else:
            plx = params_arr[-1]
        if not np.isnan(self.abs_system_mass):
            mtot = self.abs_system_mass
        else:
            mtot = params_arr[6*self.num_secondary_bodies]

        for body_num in np.arange(self.num_secondary_bodies)+1:

            epochs = self.data_table['epoch'][self.body_indices[body_num]]
            sma = params_arr[body_num-1]
            ecc = params_arr[body_num]
            argp = params_arr[body_num+1]
            lan = params_arr[body_num+2]
            inc = params_arr[body_num+3]
            tau = params_arr[body_num+4]

            raoff, decoff, vz = kepler.calc_orbit(
                epochs, sma, ecc, tau, argp, lan, inc, plx, mtot
            )
            # todo: hack to get this working for mcmc
            # if len(raoff.shape) == 1:
            #     raoff = raoff.reshape(1, raoff.shape[0])
            #     decoff = decoff.reshape(1, decoff.shape[0])
            #     vz = vz.reshape(1, vz.shape[0])

            model[self.radec[body_num], 0] = raoff[self.radec[body_num]]
            model[self.radec[body_num], 1] = decoff[self.radec[body_num]]

            sep, pa = radec2seppa(
                raoff[self.seppa[body_num]], 
                decoff[self.seppa[body_num]]
            )

            model[self.seppa[body_num], 0] = sep
            model[self.seppa[body_num], 1] = pa

        return model


def radec2seppa(ra, dec):
    """
    Convenience function for converting from 
    right ascension/declination to separation/
    position angle.

    Args:
        ra (np.array of float): array of RA values
        dec (np.array of float): array of Dec values

    Returns:
        tulple of float: (separation, position angle)

    """

    sep = np.sqrt((ra**2) + (dec**2))
    pa = (np.arctan2(ra, dec) / deg2rad) % 360.

    return sep, pa
