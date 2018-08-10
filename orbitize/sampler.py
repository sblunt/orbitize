import orbitize.lnlike
import orbitize.priors
import orbitize.kepler
import numpy as np
import astropy.units as u
import astropy.constants as consts
import sys
import abc
import emcee

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
        
        self.tbl = self.system.data_table
        self.radec_idx = self.system.radec[1]
        self.seppa_idx = self.system.seppa[1]
            
        #these are in format astropy.table.column - change to list or array?
        self.sep_observed = self.tbl[:]['quant1']
        self.pa_observed = self.tbl[:]['quant2']
        self.sep_err = self.tbl[:]['quant1_err']
        self.pa_err = self.tbl[:]['quant2_err']
    
        #worth it to use np.where? or is it already fast enough?
        for i in self.radec_idx:
            self.sep_observed[i], self.pa_observed[i] = orbitize.system.radec2seppa(self.sep_observed[i], self.pa_observed[i])
            self.sep_observed[i] *= 1000 
            
            self.sep_err[i], self.pa_err[i] = orbitize.system.radec2seppa(self.sep_err[i], self.pa_err[i])
            self.sep_err[i] *= 1000


    def prepare_samples(self, num_samples):
        """
        Prepare some orbits for rejection sampling. This draws random orbits
        from priors, and performs scale & rotate.
        Args:
            num_samples (int): number of orbits to prepare for OFTI to run
                rejection sampling on
        Return:
            np.array: array of prepared samples. The first dimension has size of num_samples. 
            This should be passed into ``reject()``
        (written):Isabel Angelo & Sarah Blunt (2018)
        """
        #to do: modify to work for multi-planet systems
        
        pri = self.system.sys_priors
        
        #generate sample orbits
        samples = np.empty([len(pri), num_samples])
        for i in range(len(pri)): 
            samples[i, :] = pri[i].draw_samples(num_samples)
        
        samples[0][0:3] = 9.10812748e+00,  9.03372328e+00, 9.90227886e+00
        samples[5][0:3] = 7.09880659e-03, -2.42354374e-01, -2.27531267e-01
        samples[2][0:3] = -1.87270855e+00, -3.62485227e+00, 3.46368945e+00
        samples[3][0:3] = 5.52997248e-01, 5.52071231e-01, 5.52940093e-01
        samples[1][0:3] =  3.11670590e-02, 2.65838630e-03, 9.20449820e-02
        samples[4][0:3] = 1.54706921e+00, 1.54728477e+00, 1.54988731e+00
        samples[-2][0:3] = 1.74622216e+00, 1.80624161e+00, 1.80154866e+00
        print(samples)
               
        epochs = np.array([self.tbl[i][0] for i in range(len(self.tbl))])
        #m_err and plx_err only if they exist
        sma,ecc,argp,lan,inc,tau,mtot,plx = [s for s in samples]
        
        #compute seppa of generated orbits 
        ra, dec, vc = orbitize.kepler.calc_orbit(epochs, sma, ecc,tau,argp,lan,inc,plx,mtot)
        sep, pa = orbitize.system.radec2seppa(ra, dec) #sep["],pa[deg]   
        
        #determine scale-and-rotate epoch
        epoch_idx = np.argmin(self.sep_err) #epoch with smallest error
         
        #generate offsets from observational uncertainties
        sep_offset = np.random.normal(self.sep_observed[epoch_idx], self.sep_err[epoch_idx]) #sep [mas]
        pa_offset =  np.random.normal(self.pa_observed[epoch_idx], self.pa_err[epoch_idx]) #pa [deg]
        
        #import pdb; pdb.set_trace()
        #perform scale-and-rotate
        samples[0] *= sep_offset/(sep[epoch_idx]*1000) #sma [AU]
        samples[3] += np.radians((pa_offset - pa[epoch_idx]) % 360) #lan [rad] CONVERT PAs to DEGREES
        
        #scale tau with semi-major axis
        period = np.sqrt(4*np.pi**2.0*(samples[0]*u.AU)**3/(consts.G*(mtot*u.Msun)))
        period = period.to(u.day).value
        samples[5] = ((epochs[epoch_idx] - samples[5]*period)/period) % 1
        
        return samples
        

    def reject(self, orbit_configs):
        """
        Runs rejection sampling on some prepared samples
        Args:
            orbit_configs (np.array): array of prepared samples. The first dimension has size `num_samples`. This should be the output of ``prepare_samples()``
        Return:
            np.array: a subset of orbit_configs that are accepted based on the data.
            
        (written):Isabel Angelo (2018)    
        """
        
        #generate seppa for all remaining epochs
        epochs = np.array([self.tbl[i][0] for i in range(len(self.tbl))])
        sma,ecc,argp,lan,inc,tau,mtot,plx = [s for s in orbit_configs]
        
        #edit to calculate for all epochs
        ra, dec, vc = orbitize.kepler.calc_orbit(epochs, sma, ecc,tau,argp,lan,inc,plx,mtot)
        sep, pa = orbitize.system.radec2seppa(ra, dec)
        
        seppa_model = []
        for i in range(len(orbit_configs[0])):
            orbit_sep = [x[i] for x in sep] 
            orbit_pa = [x[i] for x in pa] 
            seppa_model.append(np.column_stack((orbit_sep,orbit_pa)))
        seppa_model = np.array(seppa_model)
        seppa_model = np.rollaxis(seppa_model, 0, 3) 
        
        #compute probability for each orbit
        seppa_data = np.column_stack((self.sep_observed, self.pa_observed))
        seppa_errs = np.column_stack((self.sep_err, self.pa_err))
        #import pdb; pdb.set_trace()
        chi2 = orbitize.lnlike.chi2_lnlike(seppa_data, seppa_errs, seppa_model, self.seppa_idx)
        #why only for seppa_idx?
        
        #convert to probability
        chi2_sum = np.nansum(chi2, axis=(0,1))
        p = np.e**(-chi2_sum/2.)
                 
        #reject orbits with p<randomly generate number until desired orbits reached
        max_orbits = 10000 #default, should be left to user
        saved_orbits = ([])
        
        #re-write using np.where
        for i in range(len(p)):
            if p[i] > np.random.random():
                np.append(saved_orbits,orbit_configs[i])
            if len(saved_orbits) >= max_orbits:
                break
#   
        return None
                

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
    Parallel-Tempered MCMC Sampler using the emcee Affine-infariant sampler

    NOTE: emcee will no longer support PTSampler in emcee v3.0. Would need to use ptemcee instead (or use the EnsembleSampler)

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

        Can be run multiple times if you want to pause and insepct things.
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
        sampler = emcee.PTSampler(self.num_temps, self.num_walkers, self.num_params, self._logl,orbitize.priors.all_lnpriors, threads=self.num_threads, logpargs=[self.priors,] )


        for pos, lnprob, lnlike in sampler.sample(self.curr_pos, iterations=burn_steps, thin=thin):
            pass

        sampler.reset()
        self.curr_pos = pos
        print('Burn in complete')

        for pos, lnprob, lnlike in sampler.sample(pos, lnprob0=lnprob, lnlike0=lnlike,
                                                        iterations=total_orbits, thin=thin):
            pass

        self.curr_pos = pos
        self.chain = sampler.chain
        self.lnlikes = sampler.lnprobability

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
    Affine-Invariant Ensemble MCMC Sampler using emcee

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
