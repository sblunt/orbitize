import numpy as np
from orbitize import priors, read_input


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

        mass, parallax, semimajor axis b, eccentricity b, AOP b, PAN b, 
        inclination b, EPP b, [semimajor axis c, eccentricity c, etc.]

    where `b` corresponds to the first orbiting object, `c` corresponds
    to the second, etc.

    TODO: remove DeltaPriors, communicate better to MCMC (and OFTI) 
          not to fit mass or parallax if err=0

    (written): Sarah Blunt, 2018
    """
    def __init__(self, num_secondary_bodies, data_table, system_mass, 
                 plx, mass_err=0, plx_err=0):

        self.data_table = data_table
        self.num_secondary_bodies = num_secondary_bodies
        self.sys_priors = []

        # Set priors on system mass and parallax
        if mass_error > 0:
            self.sys_priors.append(priors.GaussianPrior(system_mass, mass_error))
        else:
            self.sys_priors.append(priors.DeltaPrior(system_mass))
        if plx_error > 0:
            self.sys_priors.append(priors.GaussianPrior(plx, plx_err))
        else:
            self.sys_priors.append(priors.DeltaPrior(plx))

        # Set priors for each orbital element
        for body in np.arange(num_secondary_bodies):
            # Add semimajor axis prior
            self.sys_priors.append(priors.Jeffreys(0.1, 100.))

            # Add eccentricity prior
            self.sys_priors.append(priors.Uniform(0.,1.))

            # Add argument of periastron prior
            self.sys_priors.append(priors.Uniform(0.,2.*np.pi))

            # Add position angle of nodes prior
            self.sys_priors.append(priors.Uniform(0.,2.*np.pi))

            # Add inclination angle prior
            self.sys_priors.append(priors.Sine(0.,np.pi))

            # Add epoch of periastron prior. Here, EPP is defined as the 
            # time fraction of the orbit that elapsed between 2000.00 and periastron
            self.sys_priors.append(priors.Uniform(0., 1.))
