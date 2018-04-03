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
        """
         
        # generate sample orbits from priors
        elements = system.priors #I don't see priors defined in system.py???
        samples = np.array([])
            for element in elements:
                np.append(samples, system.priors[element].draw_samples(num_samples))
        samples.transpose()
         
        #compute separations    
        separations = map(kepler.calc_orbit, samples)
        
        #compute offsets and scale accordingly
        offsets = np.random.randn(separations.size)
        separations = separations/(separations + offsets)
        

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

