class OFTI(object):
    """
    OFTI Sampler
    
    TEST I am editing this file

    Args:
        lnlike: likelihood object (TBD)
        system: system object that describes the star and planets in the system 
            (TBD)
    """
    def __init__(self, lnlike, system):
        pass

    def prepare_samples(self, num_samples):
        """
        Prepare some orbits for rejection sampling. This draws random orbits 
        from priors, and performs scale & rotate.

        Args:
            num_samples (int): number of orbits to prepare for OFTI to run 
                rejection sampling on

        Return:
            np.array: array of prepared samples. The first dimension has size of num_samples. This should be able to be passed into `reject()`
        """

        # draw an array of num_samples smas, eccs, etc. from prior objects: prior = (some object inhertiting from priors.Prior); samples = prior.draw_samples(#)
        elements = system.priors.keys() # -> this step should be done in __init__ so it doesn't slow performance

        #example creating array
        for element in elements:
            samples[i,j] = system.priors[element].draw_samples(num_samples)

    def reject(self, orbit_configs):
        """
        Runs rejection sampling on some prepared samples

        Args:
            orbit_configs (np.array): array of prepared samples. The first dimension has size `num_samples`. This should be the output of `prepare_samples()`

        Return:
            np.array: a subset of orbit_configs that are accepted based on the data.

        """
        pass

    def run_sampler(self, total_orbits):
        """
        Runs OFTI until we get the number of total accepted orbits we want. 

        Args:
            total_orbits (int): total number of accepted possible orbits that 
                are desired

        Return:
            np.array: array of accepted orbits. First dimension has size `total_orbits`.
        """
        # this function shold first check if we have reached enough orbits, and break when we do

        pass


class PTMCMC(object):
    """
    Parallel-Tempered MCMC Sampler using the emcee Affine-infariant sampler

    Args:
        lnlike: likelihood object (TBD)
        system: system object that describes the star and planets in the system 
            (TBD)
        num_temps (int): number of temperatures to run the sampler at
        num_walkers (int): number of walkers at each temperature
    """
    def __init__(self, lnlike, system, num_temps, num_walkers):
        pass

    def run_sampler(self, total_orbits, burn_steps=0, thin=1):
        """
        Runs PT MCMC sampler

        Args:
            total_orbits (int): total number of accepted possible orbits that are desired. This equals `num_steps_per_walker`x`num_walkers`
            burn_steps (int): optional paramter to tell sampler to discard certain number of steps at the beginning
            thin (int): factor to thin the steps of each walker by to remove correlations in the walker steps
        """