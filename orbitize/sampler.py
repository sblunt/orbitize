import orbitize.lnlike
import orbitize.priors
import sys
import abc
import numpy as np
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
        
        #compute seppa of generated orbits   
        ra, dec, vc = map(kepler.calc_orbit, samples)
        sep, pa = self.system.radec2seppa(ra, dec)
        
        #compute observational uncertainties in seppa
        
        if len(self.system.seppa != 0): #check for seppa data
            #extract from data table
            seppa_index = self.system.seppa[0][0]
            
            sep_observed = self.system.data_table[seppa_index][1]
            sep_err = self.system.data_table[seppa_index][2]
            pa_observed = self.system.data_table[seppa_index][3]
            pa_err = self.system.data_table[seppa_index][4]
            
        else: #for if there are only radec datatypes
            #extract from data table and convert to seppa
            radec_index = self.system.radec[0][0]
            
            ra_observed = self.system.data_table[radec_index][1]
            ra_err = self.system.data_table[radec_index][2]
            dec_observed = self.system.data_table[radec_index][3]
            dec_err = self.system.data_table[radec_index][4]
            
            sep_observed, pa_observed = self.system.radec2seppa(ra_observed, dec_observed)
            sep_err, pa_err = self.system.radec2seppa(ra_err, dec_err)    
        
        #generate offsets from observational uncertainties
        sep_offsets = sep_err * np.random.randn(sep.size)
        pa_offsets = pa_err * np.random.randn(pa.size)  
        

        #perform scale-and-rotate
        samples[0] *= (sep_observed + sep_err)/sep
        samples[3] += ((pa_observed - pa + pa_err) % 360)
        
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
        ra, dec, vc = kepler.calc_orbit(orbit_configs) #how to edit this to generate for all epoch?
        sep, pa = self.system.radec2seppa(ra, dec)
        
        #compute probability for each orbit        
        seppa_obs = np.column_stack((self.system.data_table[seppa_index][1],self.system.data_table[seppa_index][3]))
        seppa_errs = np.column_stack((self.system.data_table[seppa_index][2],self.system.data_table[seppa_index][4]))

        chi2 = [lnlike.chi2_lnlike(data, errors, orbit) for orbit in orbit_configs]
        #CHECK: dimensions of chi2 should be n_orbits x n_epochs x 2
        
        #convert to probability
        chi2_sum = np.sum(chi2, axis = 1) #sum over all epochs
        p = np.e**(-chi2_sum/2.)
        
        #reject orbits with p<randomly generate number until desired orbits reached
        max_orbits = 10000 #default, should be left to user
        saved_orbits = ([])
        
        for prob in p:
            if prob < np.random.random():
                np.append(saved_orbits,samples[prob.index])
            if len(saved_orbits) >= max_orbits:
                break
        
        return saved_orbits
                

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

    NOTE: Does not currnetly support multithreading because orbitize classes are not yet pickleable. 

    Args:
        lnlike (string): name of likelihood function in ``lnlike.py``
        system (system.System): system.System object
        num_temps (int): number of temperatures to run the sampler at
        num_walkers (int): number of walkers at each temperature
    """
    def __init__(self, lnlike, system, num_temps, num_walkers):
        super(PTMCMC, self).__init__(system, like=lnlike)
        self.num_temps = num_temps
        self.num_walkers = num_walkers

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

        self.sampler = emcee.PTSampler(num_temps, num_walkers, self.num_params, self._logl, orbitize.priors.all_lnpriors, logpargs=[self.priors,] )


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
        """
        for pos, lnprob, lnlike in self.sampler.sample(self.curr_pos, iterations=burn_steps, thin=thin):
            pass

        self.sampler.reset()
        self.curr_pos = pos
        print('Burn in complete')

        for pos, lnprob, lnlike in self.sampler.sample(pos, lnprob0=lnprob, lnlike0=lnlike,
                                                        iterations=total_orbits, thin=thin):
            pass
        
        self.curr_pos = pos
        self.chain = self.sampler.chain
        self.lnlikes = self.sampler.lnprobability

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

