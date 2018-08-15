import numpy as np
import sys
import abc

# Python 2 & 3 handle ABCs differently
if sys.version_info[0] < 3:
    ABC = abc.ABCMeta('ABC', (), {})
else:
    ABC = abc.ABC

class Prior(ABC):
    """
    Abstract base class for prior objects.
    All prior objects should inherit from this class.

    (written): Sarah Blunt, 2018
    """

    @abc.abstractmethod
    def draw_samples(self, num_samples):
        pass

    @abc.abstractmethod
    def compute_lnprob(self, element_array):
        pass

class GaussianPrior(Prior):
    """Gaussian prior.

    .. math::

        log(p(x|\\sigma, \\mu)) \\propto \\frac{(x - \\mu)}{\\sigma}

    Args:
        mu (float): mean of the distribution
        sigma (float): standard deviation of the distribution

    (written) Sarah Blunt, 2018
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def draw_samples(self, num_samples):
        """
        Draw samples from a Gaussian distribution.

        Args:
            num_samples (float): the number of samples to generate

        Returns:
            numpy array of float: samples drawn from the appropriate
            Gaussian distribution. Array has length `num_samples`. 
        """
        samples = np.random.normal(
            loc=self.mu, scale=self.sigma, size=num_samples
            )
        return samples

    def compute_lnprob(self, element_array):
        """
        Compute log(probability) of an array of numbers wrt a Gaussian distibution.

        Args:
            element_array (numpy array of float): array of numbers. We want the 
                probability of drawing each of these from the appopriate Gaussian 
                distribution

        Returns:
            numpy array of float: array of log(probability) values, 
            corresponding to the probability of drawing each of the numbers 
            in the input `element_array`.
        """
        lnprob = (element_array - self.mu) / self.sigma
        return lnprob

class JeffreysPrior(Prior):
    """
    This is the probability distribution p(x) propto 1/x.

    The __init__ method should take in a "min" and "max" value
    of the distribution, which correspond to the domain of the prior. 
    (If this is not implemented, the prior has a singularity at 0 and infinite
    integrated probability).

    Args:
        minval (float): the lower bound of this distribution
        maxval (float): the upper bound of this distribution

    """
    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

        self.logmin = np.log(minval)
        self.logmax = np.log(maxval)

    def draw_samples(self, num_samples):
        """
        Draw samples from this 1/x distribution.

        Args:
            num_samples (float): the number of samples to generate

        Returns:
            samples (np.array):  samples ranging from [0, pi) as floats.
        """
        # sample from a uniform distribution in log space
        samples = np.random.uniform(self.logmin, self.logmax, num_samples)
        # convert from log space to linear space
        samples = np.exp(samples)

        return samples

    def compute_lnprob(self, element_array):
        """
        Compute the prior probability of each element given that its drawn from a Jeffreys prior

        Args:
            element_array (np.array): array of paramters to compute the prior probability of

        Returns:
            lnprob (np.array): array of prior probabilities
        """
        lnprob = np.zeros(np.size(element_array))
        
        outofbounds = np.where((element_array > self.maxval) | (element_array < self.minval))
        lnprob[outofbounds] = -np.inf

        return lnprob

class UniformPrior(Prior):
    """
    This is the probability distribution p(x) propto constant.

    Args:
        minval (float): the lower bound of the uniform prior
        maxval (float): the upper bound of the uniform prior

    """
    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

    def draw_samples(self, num_samples):
        """
        Draw samples from this uniform distribution.

        Args:
            num_samples (float): the number of samples to generate

        Returns:
            samples (np.array):  samples ranging from [0, pi) as floats.
        """
        # sample from a uniform distribution in log space
        samples = np.random.uniform(self.minval, self.maxval, num_samples)

        return samples

    def compute_lnprob(self, element_array):
        """
        Compute the prior probability of each element given that its drawn from this uniform prior

        Args:
            element_array (np.array): array of paramters to compute the prior probability of

        Returns:
            lnprob (np.array): array of prior probabilities
        """
        lnprob = np.zeros(np.size(element_array))
        
        outofbounds = np.where((element_array > self.maxval) | (element_array < self.minval))
        lnprob[outofbounds] = -np.inf

        return lnprob

class SinPrior(Prior):
    """
    This is the probability distribution p(x) propto sin(x).

    The domain of this prior should be [0,pi].

    Args:
        None
    """

    def __init__(self):
        pass

    def draw_samples(self, num_samples):
        """
        Draw samples from a Sine distribution.

        Args:
            num_samples (float): the number of samples to generate

        Returns:
            samples (np.array):  samples ranging from [0, pi) as floats.
        """
        # draw uniform from -1 to 1
        samples = np.random.uniform(-1, 1, num_samples)

        return np.arccos(samples)

    def compute_lnprob(self, element_array):
        return np.sin(element_array)

class LinearPrior(Prior):
    """
    Draw samples from the probability distribution:

    .. math::

        p(x) \\propto mx+b

    where m is negative, b is positive, and the 
    range is [0,-b/m].

    Args:
        m (float): slope of line. Must be negative.
        b (float): y intercept of line. Must be positive.

    """
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def draw_samples(self, num_samples):
        """
        Draw samples from a descending linear distribution.

        Args:
            num_samples (float): the number of samples to generate

        Returns:
            samples (np.array):  samples ranging from [0, -b/m) as floats.
        """
        norm = -0.5*self.b**2/self.m

        # draw uniform from 0 to 1
        samples = np.random.uniform(0, 1, num_samples)

        # generate samples following a linear distribution
        linear_samples = -np.sqrt(2.*norm*samples/self.m + (self.b/self.m)**2) - (self.b/self.m)

        return linear_samples

        

    def compute_lnprob(self, element_array):

        x_intercept = -self.b/self.m
        normalizer = -0.5*self.b**2/self.m

        prob = (self.m*element_array + self.b)/normalizer

        print(prob)

        prob[(element_array>x_intercept) | (element_array<0)] = 0.

        return prob




def all_lnpriors(params, priors):
    """
    Calculates log(prior probability) of a set of parameters and a list of priors

    Args:
        params (np.array): size of N parameters 
        priors (list): list of N prior objects corresponding to each parameter

    Returns:
        logp (float): prior probability of this set of parameters
    """
    logp = 0.
    for param, prior in zip(params, priors):
        logp += prior.compute_lnprob(param)
    
    return logp



if __name__ == '__main__':

    myPrior = LinearPrior(-1., 1.)
    mySamples = myPrior.draw_samples(1000)
    print(mySamples)
    myProbs = myPrior.compute_lnprob(mySamples)
    print(myProbs)

    myPrior = GaussianPrior(1.3, 0.2)
    mySamples = myPrior.draw_samples(1)
    print(mySamples)

    myProbs = myPrior.compute_lnprob(mySamples)
    print(myProbs)



