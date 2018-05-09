from orbitize import lnlike
from orbitize import system
from orbitize import kepler
import sys
import abc
import numpy as np


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

    def __init__(cls, system, like='chi2_lnlike'):
        cls.system = system
        cls.lnlike = getattr(lnlike, like)

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
        (written):Isabel Angelo & Sarah Blunt (2018)
        """
        #to do: modify to work for multi-planet systems
        
        #store priors -> this step should be done in OFTI.__init__ so it doesn't slow performance
        pri = self.system.sys_priors
        
        #generate sample orbits
        samples = np.empty([len(pri), num_samples])
        for i in range(len(pri)): 
            samples[i, :] = pri[i].draw_samples(num_samples)
        
        #compute seppa of generated orbits   
        ra, dec, vc = map(kepler.calc_orbit, samples)
        sep, pa = self.system.radec2seppa(ra, dec)
        
        #compute observational uncertainties in seppa
        sep_err = 0
        pa_err = 0

        if len(self.system.seppa != 0): #check for seppa data
            #extract from data table
            seppa_index = self.system.seppa[0][0]
            
            sep_err = self.system.data_table[seppa_index][2]
            pa_err = self.system.data_table[seppa_index][4]
            
        else: #for if there are only radec datatypes
            #extract from data table and convert to seppa
            radec_index = self.system.radec[0][0]
            
            ra_err = self.system.data_table[radec_index][2]
            dec_err = self.system.data_table[radec_index][4]
            
            sep_err, pa_err = self.system.radec2seppa(ra_err, dec_err)    
        
        #generate offsets from observational uncertainties
        sep_offsets = sep_err * np.random.randn(sep.size)
        pa_offsets = pa_err * np.random.randn(pa.size)  
        
        #perform scale-and-rotate
        samples[0] = sep * (sep/(sep + sep_offsets))
        samples[-1] = pa + pa_offsets
        

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
    Parallel-Tempered MCMC Sampler using the emcee Affine-infariant sampler
    Args:
        lnlike (string): name of likelihood function in ``lnlike.py``
        system (system.System): system.System object
        num_temps (int): number of temperatures to run the sampler at
        num_walkers (int): number of walkers at each temperature
    """
    def __init__(self, like, system, num_temps, num_walkers):
        super(OFTI, self).__init__(system, like=like)
        self.num_temps = num_temps
        self.num_walkers = num_walkers

    def run_sampler(self, total_orbits, burn_steps=0, thin=1):
        """
        Runs PT MCMC sampler
        Args:
            total_orbits (int): total number of accepted possible 
                orbits that are desired. This equals 
                ``num_steps_per_walker``x``num_walkers``
            burn_steps (int): optional paramter to tell sampler
                to discard certain number of steps at the beginning
            thin (int): factor to thin the steps of each walker 
                by to remove correlations in the walker steps
        """
        pass

