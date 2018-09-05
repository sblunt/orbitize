import numpy as np
import astropy.units as u
import astropy.constants as consts
import sys
import abc

import emcee
import ptemcee

import orbitize.lnlike
import orbitize.priors
import orbitize.kepler
from orbitize.system import radec2seppa
import orbitize.results

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

    (written): Isabel Angelo, Logan Pearce, Sarah Blunt 2018
    """
    def __init__(self, system, like='chi2_lnlike'):

        super(OFTI, self).__init__(system, like=like)
        
        self.priors = self.system.sys_priors
        self.tbl = self.system.data_table
        self.radec_idx = self.system.radec[1]
        self.seppa_idx = self.system.seppa[1]
            
        # these are of type astropy.table.column
        self.sep_observed = self.tbl[:]['quant1']
        self.pa_observed = self.tbl[:]['quant2']
        self.sep_err = self.tbl[:]['quant1_err']
        self.pa_err = self.tbl[:]['quant2_err']
    
        # convert RA/Dec rows to sep/PA
        for i in self.radec_idx:
            self.sep_observed[i], self.pa_observed[i] = radec2seppa(
                self.sep_observed[i], self.pa_observed[i]
            )
            self.sep_err[i], self.pa_err[i] = radec2seppa(
                self.sep_err[i], self.pa_err[i]
            )

        self.epochs = np.array(self.tbl['epoch'])
        
        # choose scale-and-rotate epoch
        self.epoch_idx = np.argmin(self.sep_err) # epoch with smallest error

        # format sep/PA observations for use with the lnlike code
        self.seppa_for_lnlike = np.column_stack((self.sep_observed, self.pa_observed))
        self.seppa_errs_for_lnlike = np.column_stack((self.sep_err, self.pa_err))

    def prepare_samples(self, num_samples):
        """
        Prepare some orbits for rejection sampling. This draws random orbits
        from priors, and performs scale & rotate.

        Args:
            num_samples (int): number of orbits to draw and scale & rotate for 
            OFTI to run rejection sampling on

        Return:
            np.array: array of prepared samples. The first dimension has size of 
            num_samples. This should be passed into ``reject()``
        """

        # TODO: modify to work for multi-planet systems
        
        # generate sample orbits
        samples = np.empty([len(self.priors), num_samples])
        for i in range(len(self.priors)): 
            samples[i, :] = self.priors[i].draw_samples(num_samples)

        # TODO: fix for the case where m_err and plx_err are nan
        sma, ecc, argp, lan, inc, tau, mtot, plx = [s for s in samples]

        period_prescale = np.sqrt(
            4*np.pi**2*(sma*u.AU)**3/(consts.G*(mtot*u.Msun))
        )
        period_prescale = period_prescale.to(u.day).value
        meananno = self.epochs[self.epoch_idx]/period_prescale - tau

        # compute sep/PA of generated orbits 
        ra, dec, vc = orbitize.kepler.calc_orbit(
            self.epochs[self.epoch_idx], sma, ecc, tau, argp, lan, inc, plx, mtot
        )
        sep, pa = orbitize.system.radec2seppa(ra, dec) # sep[mas], PA[deg]  
        
        # generate Gaussian offsets from observational uncertainties
        sep_offset = np.random.normal(
            0, self.sep_err[self.epoch_idx], size=num_samples
        )
        pa_offset =  np.random.normal(
            0, self.pa_err[self.epoch_idx], size=num_samples
        )
        
        # calculate correction factors
        sma_corr = (sep_offset + self.sep_observed[self.epoch_idx])/sep
        lan_corr = (pa_offset + self.pa_observed[self.epoch_idx] - pa)
        
        # perform scale-and-rotate
        sma *= sma_corr # [AU]
        lan += np.radians(lan_corr) # [rad] 
        lan = lan % (2*np.pi)

        period_new = np.sqrt(
            4*np.pi**2*(sma*u.AU)**3/(consts.G*(mtot*u.Msun))
        )
        period_new = period_new.to(u.day).value

        tau = (self.epochs[self.epoch_idx]/period_new - meananno)

        # updates samples with new values of sma, pan, tau
        samples[0,:] = sma
        samples[3,:] = lan
        samples[5,:] = tau
        
        return samples
        

    def reject(self, samples):
        """
        Runs rejection sampling on some prepared samples.

        Args:
            samples (np.array): array of prepared samples. The first dimension 
            has size `num_samples`. This should be the output of 
            `prepare_samples()`.

        Return:
            np.array: a subset of `samples` that are accepted based on the 
                data.
            
        """
        
        # generate seppa for all remaining epochs
        sma, ecc, argp, lan, inc, tau, mtot, plx = [s for s in samples]
        
        ra, dec, vc = orbitize.kepler.calc_orbit(
            self.epochs, sma, ecc,tau,argp,lan,inc,plx,mtot
        )
        sep, pa = orbitize.system.radec2seppa(ra, dec)

        seppa_model = np.vstack(zip(sep, pa))
        seppa_model = seppa_model.reshape((len(self.epochs), 2, len(sma)))

        # compute chi2 for each orbit
        chi2 = orbitize.lnlike.chi2_lnlike(
            self.seppa_for_lnlike, self.seppa_errs_for_lnlike, 
            seppa_model, self.seppa_idx
        )
        
        # convert to log(probability)
        chi2_sum = np.nansum(chi2, axis=(0,1))
        lnp = -chi2_sum/2.
               
        # reject orbits with probability less than a uniform random number
        random_samples = np.log(np.random.random(len(lnp)))
        saved_orbit_idx = np.where(lnp > random_samples)[0]
        saved_orbits = np.array([samples[:,i] for i in saved_orbit_idx])
        
        return saved_orbits
                

    def run_sampler(self, total_orbits, num_samples=10000):
        """
        Runs OFTI until we get the number of total accepted orbits we want. 

        Args:
            total_orbits (int): total number of accepted orbits desired by user
            num_samples (int): number of orbits to prepare for OFTI to run
            rejection sampling on

        Return:
            output_orbits (np.array): array of accepted orbits. First dimension 
            has size `total_orbits`.
        """

        n_orbits_saved = 0
        output_orbits = np.empty((total_orbits, len(self.priors)))
        
        # add orbits to `output_orbits` until `total_orbits` are saved
        while n_orbits_saved < total_orbits:
            samples = self.prepare_samples(num_samples)
            accepted_orbits = self.reject(samples)
            
            if len(accepted_orbits)==0:
                pass
            else:
                n_accepted = len(accepted_orbits)
                maxindex2save = np.min([n_accepted, total_orbits - n_orbits_saved])

                output_orbits[n_orbits_saved : n_orbits_saved+n_accepted] = accepted_orbits[0:maxindex2save]
                n_orbits_saved += maxindex2save
                
        return np.array(output_orbits)


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
            post = None,
            lnlike = None
        )

        # get priors from the system class
        self.priors = system.sys_priors

        # initialize walkers initial postions
        self.num_params = len(self.priors)
        init_positions = []
        for prior in self.priors:
            # draw them uniformly becase we don't know any better right now
            # TODO: be smarter in the future
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
        # TODO: Need something here to pick out temperatures, just using lowest one for now
        self.chain = sampler.chain
        self.post = sampler.flatchain[0,:]
        self.lnlikes = sampler.logprobability
        self.results.add_samples(self.post,self.lnlikes)

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
            post = None,
            lnlike = None
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
        self.post = sampler.flatchain
        self.lnlikes = sampler.lnprobability
        self.results.add_samples(self.post,self.lnlikes)

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
