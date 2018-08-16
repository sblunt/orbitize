from orbitize import read_input, system


class Driver(object):
    """
    Runs through ``orbitize`` methods in a standardized way.

    Args:
        filename (str): path to data file. See ``orbitize.read_input.py``
        sampler (str): algorithm to use for orbit computation. 'PTMCMC' for 
            Markov Chain Monte Carlo, 'OFTI' for Orbits for the Impatient.
        lnlike (str): name of function in ``orbitize.lnlike.py`` that will
            be used to compute likelihood. ["chi2_lnlike"]
        num_secondary_bodies (int): number of secondary bodies in the system. 
            Should be at least 1.
        system_mass (float): mean total mass of the system, in M_sol
        plx (float): mean parallax of the system, in mas
        mass_err (float): uncertainty on ``system_mass``, in M_sol
        plx_err (float): uncertainty on ``plx``, in mas

    (written): Sarah Blunt, 2018
    """
    def __init__(self, filename, sampler,
                 num_secondary_bodies, system_mass, plx, 
                 mass_err=0, plx_err=0, lnlike='chi2_lnlike'):

        # Read in data
        data_table = read_input.read_formatted_file(filename)

        # Initialize System object which stores data & sets priors
        self.system = system.System(
            num_secondary_bodies, data_table, system_mass, 
            plx, mass_err=mass_err, plx_err=plx_err
        )

        # Initialize Sampler object, which stores information about
        # the likelihood function & the algorithm used to generate
        # orbits, and has System object as an attribute.
        sampler_func = getattr(orbitize.sampler, sampler)
        self.sampler = sampler_func(lnlike, self.system)

    def compute_posteriors(self, total_orbits):

        accepted_orbits = self.sampler.run_sampler(total_orbits)
        return accepted_orbits
