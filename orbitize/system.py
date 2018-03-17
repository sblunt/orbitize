import numpy as np
from orbitize import priors, read_input, kepler

class System(object):
    """
    A class containing information about all data, system parameters 
    (e.g. total mass, parallax), and priors relevant for a gravitationally-
    bound system.

    Args:
        num_secondary_bodies (int): number of secondary bodies in the system. 
            Should be at least 1.
        data_table (astropy.table.Table): output from either
            ``orbitize.read_input.read_formatted_file()`` or 
            ``orbitize.read_input.read_orbitize_input()``
        system_mass (float): mean total mass of the system, in M_sol
        plx (float): mean parallax of the system, in arcsec
        mass_err (float): uncertainty on ``system_mass``, in M_sol
        plx_err (float): uncertainty on ``plx``, in arcsec

    Users should initialize an instance of this class, then overwrite 
    priors they wish to customize. 

    Priors are initialized as a list of orbitize.priors.Prior objects,
    in the following order:

        semimajor axis b, eccentricity b, AOP b, PAN b, inclination b, EPP b, 
        [semimajor axis c, eccentricity c, etc.], 
        mass, parallax

    where `b` corresponds to the first orbiting object, `c` corresponds
    to the second, etc. 

    (written): Sarah Blunt, 2018
    """
    def __init__(self, num_secondary_bodies, data_table, system_mass, 
                 plx, mass_err=0, plx_err=0):

        self.num_secondary_bodies = num_secondary_bodies
        self.sys_priors = []

        # Set priors for each orbital element
        for body in np.arange(num_secondary_bodies):
            # Add semimajor axis prior
            self.sys_priors.append(priors.JeffreysPrior(0.1, 100.))

            # Add eccentricity prior
            self.sys_priors.append(priors.UniformPrior(0.,1.))

            # Add argument of periastron prior
            self.sys_priors.append(priors.UniformPrior(0.,2.*np.pi))

            # Add position angle of nodes prior
            self.sys_priors.append(priors.UniformPrior(0.,2.*np.pi))

            # Add inclination angle prior
            self.sys_priors.append(priors.CosPrior(0.,np.pi))

            # Add epoch of periastron prior. 
            self.sys_priors.append(priors.UniformPrior(0., 1.))

        # Set priors on system mass and parallax
        if mass_error > 0:
            self.sys_priors.append(priors.GaussianPrior(
                system_mass, mass_error)
            )
            self.abs_system_mass = None
        else:
            self.abs_system_mass = system_mass
        if plx_error > 0:
            self.sys_priors.append(priors.GaussianPrior(plx, plx_err))
            self.abs_system_mass = None
        else:
            self.abs_plx = plx

        # Group the data in some useful ways
        self.data_table = data_table

        self.body_indices = []
        self.radec = []
        self.seppa = []

        radec_indices = np.where(self.data_table['quant_type']=='radec')
        seppa_indices = np.where(self.data_table['quant_type']=='seppa')

        for body_num in np.arange(self.num_secondary_bodies+1):

            self.body_indices.append(
                np.where(self.data_table['object']==body_num)
            )

            self.radec.append(
                np.intersect1d(body_indices[body_num], radec_indices)
            )
            self.seppa.append(
                np.intersect1d(body_indices[body_num], seppa_indices)
            )


    def compute_model(self, params_arr):
    """
    params_arr (2d numpy array: num_orbits x num_params)

    # TODO: document this whole file better.
    # TODO: write tests:
    # 1. test that __init__ produces correct list of priors (both err=0 and not cases)
    # 2. self.seppa and self.radec are read in correctly for multi-planet data table
    # 3. test that model produces correct output.
    # 4. little unit tests for radec3seppa
    # TODO: upgrade lnlike to act on arrays.
    # TODO: update orbitize team.
    """

        model = np.zeros(params_arr.shape[0], len(self.data_table), 2.)

        if self.plx:
            plx = self.plx
        else:
            plx = params_arr[:, -1]
        if self.system_mass:
            mtot = self.system_mass
        else:
            mtot = params_arr[:, 6*self.num_secondary_bodies]

        for body_num in self.num_secondary_bodies:

            epochs = self.data_table['epoch'][self.body_indices[body_num]]
            sma = params_arr[:, body_num]
            ecc = params_arr[:, body_num+1]
            argp = params_arr[:, body_num+2]
            lan = params_arr[:, body_num+3]
            inc = params_arr[:, body_num+4]
            tau = params_arr[:, body_num+5]

            raoff, decoff, vz = kepler.calc_orbit(
                epochs, sma, ecc, tau, argp, lan, inc, plx, mtot
            )

            model[:, self.radec[body_num], 0] = raoff[self.radec[body_num]]
            model[:, self.radec[body_num], 1] = decoff[self.radec[body_num]]

            sep, pa = radec2seppa(
                raoff[self.seppa[body_num]], 
                decoff[self.seppa[body_num]]
            )
            model[:, self.seppa[body_num], 0] = sep
            model[:, self.seppa[body_num], 1] = pa

        return model


def radec2seppa(ra, dec):
"""

ra, dec must be np arrays

(written: Eric Nielsen, <2016)
(ported to Python: Sarah Blunt, 2018)
"""

    deg2rad = 0.0174532925199433
    sep = np.sqrt((ra**2) + (dec**2))
    pa = np.arctan(ra/dec) / deg2rad

    test1 = [i for i, dec in enumerate(dec) if dec<0]
    test2 = [i for i, dec in enumerate(dec) if dec>=0 and ra[i]<0]

    pa[test1] += 180.
    pa[test2] += 360.

    return sep, pa
