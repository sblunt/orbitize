import orbitize.lnlike
import orbitize.priors
import orbitize.results
import sys
import abc
import numpy as np
import emcee
import ptemcee

# Python 2 & 3 handle ABCs differently
if sys.version_info[0] < 3:
    ABC = abc.ABCMeta('ABC', (), {})
else:
    ABC = abc.ABC

class Sampler(ABC):
    """
    Abstract base class for sampler objects.
    All sampler objects should inherit from this class.

    (written): Sarah Blunt, 2018
    """

    def __init__(self, system, like='chi2_lnlike'):
        self.system = system

        # check if likliehood fuction is a string of a function
        if callable(like):
            self.lnlike = like
        else:
            self.lnlike = getattr(orbitize.lnlike, like)

    @abc.abstractmethod
    def run_sampler(self, total_orbits):
        pass


class OFTI(Sampler):
    """
    OFTI Sampler

    Args:
        lnlike (string): name of likelihood function in ``lnlike.py``
        system (system.System): system.System object
    """
    def __init__(self, system, like='chi2_lnlike'):
        super(OFTI, self).__init__(system, like=like)

    def prepare_samples(self, num_samples):
        """
        Prepare some orbits for rejection sampling. This draws random orbits
        from priors, and performs scale & rotate.

        Args:
            num_samples (int): number of orbits to prepare for OFTI to run
                rejection sampling on

        Return:
            np.array: array of prepared samples. The first dimension has size of num_samples. This should be passed into ``reject()``
        """
        pass
        # draw an array of num_samples smas, eccs, etc. from prior objects: prior = (some object inhertiting from priors.Prior); samples = prior.draw_samples(#)
      #  elements = system.priors # -> this step should be done in OFTI.__init__ so it doesn't slow performance

    #    for element in elements:
     #       samples[i,j] = system.priors[element].draw_samples(num_samples)

    def reject(self, orbit_configs):
        """
        Runs rejection sampling on some prepared samples

        Args:
            orbit_configs (np.array): array of prepared samples. The first dimension has size `num_samples`. This should be the output of ``prepare_samples()``

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
            np.array: array of accepted orbits. First dimension has size ``total_orbits``.
        """
        # this function shold first check if we have reached enough orbits, and break when we do

        # put outputs of calc_orbit into format specified by mask passed from System object. Feed these arrays of data, model, and errors into lnlike.py
        pass


class PTMCMC(Sampler):
    """
    Parallel-Tempered MCMC Sampler using ptemcee, a fork of the emcee Affine-infariant sampler

    Args:
        lnlike (string): name of likelihood function in ``lnlike.py``
        system (system.System): system.System object
        num_temps (int): number of temperatures to run the sampler at
        num_walkers (int): number of walkers at each temperature
        num_threads (int): number of threads to use for parallelization (default=1)

    (written): Jason Wang, Henry Ngo, 2018
    """
    def __init__(self, lnlike, system, num_temps, num_walkers, num_threads=1):
        super(PTMCMC, self).__init__(system, like=lnlike)
        self.num_temps = num_temps
        self.num_walkers = num_walkers
        self.num_threads = num_threads
        # Create an empty results object
        self.results = orbitize.results.Results(
            sampler_name = self.__class__.__name__,
            mass_err = system.mass_err,
            plx_err = system.plx_err
        )

        # get priors from the system class
        self.priors = system.sys_priors

        # initialize walkers initial postions
        self.num_params = len(self.priors)
        init_positions = []
        for prior in self.priors:
            # draw them uniformly becase we don't know any better right now
            # todo: be smarter in the future
            random_init = prior.draw_samples(num_walkers*num_temps).reshape([num_temps, num_walkers])

            init_positions.append(random_init)

        # make this an numpy array, but combine the parameters into a shape of (ntemps, nwalkers, nparams)
        # we currently have a list of [ntemps, nwalkers] with nparam arrays. We need to make nparams the third dimension
        # save this as the current position
        self.curr_pos = np.dstack(init_positions)

    def run_sampler(self, total_orbits, burn_steps=0, thin=1):
        """
        Runs PT MCMC sampler. Results are stored in self.chain, and self.lnlikes
        Results also added to orbitize.results.Results object (self.results)

        Can be run multiple times if you want to pause and inspect things.
        Each call will continue from the end state of the last execution

        Args:
            total_orbits (int): total number of accepted possible
                orbits that are desired. This equals
                ``num_steps_per_walker``x``num_walkers``
            burn_steps (int): optional paramter to tell sampler
                to discard certain number of steps at the beginning
            thin (int): factor to thin the steps of each walker
                by to remove correlations in the walker steps

        Returns:
            emcee.sampler object
        """
        sampler = ptemcee.Sampler(self.num_walkers, self.num_params, self._logl, orbitize.priors.all_lnpriors, ntemps=self.num_temps, threads=self.num_threads, logpargs=[self.priors,] )


        for pos, lnprob, lnlike in sampler.sample(self.curr_pos, iterations=burn_steps, thin=thin):
            pass

        sampler.reset()
        self.curr_pos = pos
        print('Burn in complete')

        for pos, lnprob, lnlike in sampler.sample(p0=pos, iterations=total_orbits, thin=thin):
            pass

        self.curr_pos = pos
        self.chain = sampler.chain
        self.lnlikes = sampler.logprobability
        self.results.add_orbits(self.chain,self.lnlikes)

        return sampler

    def _logl(self, params):
        """
        log likelihood function for emcee that interfaces with the orbitize objectts
        Comptues the sum of the log likelihoods of all the data given the input model

        Args:
            params (np.array): 1-D numpy array of size self.num_params

        Returns:
            lnlikes (float): sum of all log likelihoods of the data given input model

        """
        # compute the model based on system params
        model = self.system.compute_model(params)

        # fold data/errors to match model output shape. In particualr, quant1/quant2 are interleaved
        data = np.array([self.system.data_table['quant1'], self.system.data_table['quant2']]).T
        errs = np.array([self.system.data_table['quant1_err'], self.system.data_table['quant2_err']]).T

        # todo: THIS ONLY WORKS FOR 1 PLANET. Could in the future make this a for loop to work for multiple planets.
        seppa_indices = np.union1d(self.system.seppa[0], self.system.seppa[1])

        # compute lnlike now
        lnlikes =  self.lnlike(data, errs, model, seppa_indices)

        # return sum of lnlikes (aka product of likeliehoods)
        return np.nansum(lnlikes)

class EnsembleMCMC(Sampler):
    """
    Affine-Invariant Ensemble MCMC Sampler using emcee. Warning: may not work well for multi-modal distributions

    Args:
        lnlike (string): name of likelihood function in ``lnlike.py``
        system (system.System): system.System object
        num_walkers (int): number of walkers at each temperature
        num_threads (int): number of threads to use for parallelization (default=1)

    (written): Jason Wang, Henry Ngo, 2018
    """
    def __init__(self, lnlike, system, num_walkers, num_threads=1):
        super(EnsembleMCMC, self).__init__(system, like=lnlike)
        self.num_walkers = num_walkers
        self.num_threads = num_threads
        # Create an empty results object
        self.results = orbitize.results.Results(
            sampler_name = self.__class__.__name__,
            mass_err = system.mass_err,
            plx_err = system.plx_err
        )

        # get priors from the system class
        self.priors = system.sys_priors

        # initialize walkers initial postions
        self.num_params = len(self.priors)
        init_positions = []
        for prior in self.priors:
            # draw them uniformly becase we don't know any better right now
            # todo: be smarter in the future
            random_init = prior.draw_samples(num_walkers)

            init_positions.append(random_init)

        # make this an numpy array, but combine the parameters into a shape of (nwalkers, nparams)
        # we currently have a list of arrays where each entry is num_walkers prior draws for each parameter
        # We need to make nparams the second dimension, so we have to transpose the stacked array
        self.curr_pos = np.stack(init_positions).T

    def run_sampler(self, total_orbits, burn_steps=0, thin=1):
        """
        Runs the Affine-Invariant MCMC sampler. Results are stored in self.chain, and self.lnlikes
        Results also added to orbitize.results.Results object (self.results)

        Can be run multiple times if you want to pause and inspect things.
        Each call will continue from the end state of the last execution

        Args:
            total_orbits (int): total number of accepted possible
                orbits that are desired. This equals
                ``num_steps_per_walker``x``num_walkers``
            burn_steps (int): optional paramter to tell sampler
                to discard certain number of steps at the beginning
            thin (int): factor to thin the steps of each walker
                by to remove correlations in the walker steps

        Returns:
            emcee.sampler object
        """
        # sampler = emcee.EnsembleSampler(num_walkers, self.num_params, self._logl, orbitize.priors.all_lnpriors, threads=num_threads, logpargs=[self.priors,] )
        sampler = emcee.EnsembleSampler(self.num_walkers, self.num_params, self._logl, threads=self.num_threads)

        for pos, lnprob, lnlike in sampler.sample(self.curr_pos, iterations=burn_steps, thin=thin):
            pass

        sampler.reset()
        self.curr_pos = pos
        print('Burn in complete')

        for pos, lnprob, lnlike in sampler.sample(pos, lnprob0=lnprob, iterations=total_orbits, thin=thin):
            pass

        self.curr_pos = pos
        self.chain = sampler.chain
        self.lnlikes = sampler.lnprobability
        self.results.add_orbits(self.chain,self.lnlikes)

        return sampler

    def _logl(self, params):
        """
        log likelihood function for emcee that interfaces with the orbitize objectts
        Comptues the sum of the log likelihoods of all the data given the input model

        Args:
            params (np.array): 1-D numpy array of size self.num_params

        Returns:
            lnlikes (float): sum of all log likelihoods of the data given input model

        """
        # compute the model based on system params
        model = self.system.compute_model(params)

        # fold data/errors to match model output shape. In particualr, quant1/quant2 are interleaved
        data = np.array([self.system.data_table['quant1'], self.system.data_table['quant2']]).T
        errs = np.array([self.system.data_table['quant1_err'], self.system.data_table['quant2_err']]).T

        # todo: THIS ONLY WORKS FOR 1 PLANET. Could in the future make this a for loop to work for multiple planets.
        seppa_indices = np.union1d(self.system.seppa[0], self.system.seppa[1])

        # compute lnlike now
        lnlikes =  self.lnlike(data, errs, model, seppa_indices)

        # return sum of lnlikes (aka product of likeliehoods)
        return np.nansum(lnlikes)
