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

    Note: will need inverse transform sampling for this one.

    """
    def __init__(self, min, max):
        pass
    def draw_samples(self, num_samples):
        pass
    def compute_lnprob(self, element_array):
        pass

class UniformPrior(Prior):
    """
    This is the probability distribution p(x) propto constant.

    The __init__ method should take in a "min" and "max" value
    of the distribution, which correspond to the domain of the prior. 
    
    Note: can use numpy.random.random for this prior's ``draw_samples()``
    method.

    """
    def __init__(self, min, max):
        pass
    def draw_samples(self, num_samples):
        pass
    def compute_lnprob(self, element_array):
        pass

class SinPrior(Prior):
    """
    This is the probability distribution p(x) propto sin(x).

    The domain of this prior should be [0,pi].

    Note: will need inverse transform sampling for this one.

    """

    def __init__(self):
        pass
    def draw_samples(self, num_samples):
        pass
    def compute_lnprob(self, element_array):
        pass

class LinearPrior(Prior):
    """
    This is the probability distribution p(x) propto mx+b.

    The __init__ method should take in "m" and "b" values.

    Note: will need inverse transform sampling for this one.

    """
    def __init__(self, m, b):
        pass
    def draw_samples(self, num_samples):
        pass
    def compute_lnprob(self, element_array):
        pass


if __name__ == '__main__':

    myPrior = GaussianPrior(1.3, 0.2)
    mySamples = myPrior.draw_samples(1)
    print(mySamples)

    myProbs = myPrior.compute_lnprob(mySamples)
    print(myProbs)

"""

GENERAL TIPS: 

- calculate the inverse transform sampling 
stuff on paper before trying to code it all up.

- make sure to normalize the
probability distribution before performing 
inverse transform sampling.

- inverse transform sampling can be tricky. Let 
Sarah know if you run into problems.

"""



