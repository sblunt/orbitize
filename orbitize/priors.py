import numpy as np
import sys
import abc

"""
This module defines priors with methods to draw samples and compute log(probability)
"""

# Python 2 & 3 handle ABCs differently
if sys.version_info[0] < 3:
    ABC = abc.ABCMeta('ABC', (), {})
else:
    ABC = abc.ABC


class Prior(ABC):
    """
    Abstract base class for prior objects.
    All prior objects should inherit from this class.

    Written: Sarah Blunt, 2018
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
        no_negatives (bool): if True, only positive values will be drawn from
            this prior, and the probability of negative values will be 0 (default:True).

    (written) Sarah Blunt, 2018
    """

    def __init__(self, mu, sigma, no_negatives=True):
        self.mu = mu
        self.sigma = sigma
        self.no_negatives = no_negatives

    def __repr__(self):
        return "Gaussian"

    def draw_samples(self, num_samples):
        """
        Draw positive samples from a Gaussian distribution.
        Negative samples will not be returned.

        Args:
            num_samples (float): the number of samples to generate

        Returns:
            numpy array of float: samples drawn from the appropriate
            Gaussian distribution. Array has length `num_samples`.
        """

        samples = np.random.normal(
            loc=self.mu, scale=self.sigma, size=num_samples
        )
        bad = np.inf

        if self.no_negatives:

            while bad != 0:

                bad_samples = np.where(samples < 0)[0]
                bad = len(bad_samples)

                samples[bad_samples] = np.random.normal(
                    loc=self.mu, scale=self.sigma, size=bad
                )

        return samples

    def compute_lnprob(self, element_array):
        """
        Compute log(probability) of an array of numbers wrt a Gaussian distibution.
        Negative numbers return a probability of -inf.

        Args:
            element_array (float or np.array of float): array of numbers. We want the
                probability of drawing each of these from the appopriate Gaussian
                distribution

        Returns:
            numpy array of float: array of log(probability) values,
            corresponding to the probability of drawing each of the numbers
            in the input `element_array`.
        """
        lnprob = -0.5*np.log(2.*np.pi*self.sigma) - 0.5*((element_array - self.mu) / self.sigma)**2

        if self.no_negatives:

            bad_samples = np.where(element_array < 0)[0]
            lnprob[bad_samples] = -np.inf

        return lnprob


class LogUniformPrior(Prior):
    """
    This is the probability distribution :math:`p(x) \\propto 1/x`

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

    def __repr__(self):
        return "Log Uniform"

    def draw_samples(self, num_samples):
        """
        Draw samples from this 1/x distribution.

        Args:
            num_samples (float): the number of samples to generate

        Returns:
            np.array:  samples ranging from [``minval``, ``maxval``) as floats.
        """
        # sample from a uniform distribution in log space
        samples = np.random.uniform(self.logmin, self.logmax, num_samples)

        # convert from log space to linear space
        samples = np.exp(samples)

        return samples

    def compute_lnprob(self, element_array):
        """
        Compute the prior probability of each element given that its drawn from a Log-Uniofrm  prior

        Args:
            element_array (float or np.array of float): array of paramters to compute the prior probability of

        Returns:
            np.array: array of prior probabilities
        """
        normalizer = self.logmax - self.logmin

        lnprob = -np.log((element_array*normalizer))

        # account for scalar inputs
        if np.shape(lnprob) == ():
            if (element_array > self.maxval) or (element_array < self.minval):
                lnprob = -np.inf
        else:
            lnprob[(element_array > self.maxval) | (element_array < self.minval)] = -np.inf

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

    def __repr__(self):
        return "Uniform"

    def draw_samples(self, num_samples):
        """
        Draw samples from this uniform distribution.

        Args:
            num_samples (float): the number of samples to generate

        Returns:
            np.array:  samples ranging from [0, pi) as floats.
        """
        # sample from a uniform distribution in log space
        samples = np.random.uniform(self.minval, self.maxval, num_samples)

        return samples

    def compute_lnprob(self, element_array):
        """
        Compute the prior probability of each element given that its drawn from this uniform prior

        Args:
            element_array (float or np.array of float): array of paramters to compute the prior probability of

        Returns:
            np.array: array of prior probabilities
        """
        lnprob = np.log(np.ones(np.size(element_array))/(self.maxval - self.minval))

        # account for scalar inputs
        if np.shape(lnprob) == ():
            if (element_array > self.maxval) or (element_array < self.minval):
                lnprob = -np.inf
        else:
            lnprob[(element_array > self.maxval) | (element_array < self.minval)] = -np.inf

        return lnprob


class SinPrior(Prior):
    """
    This is the probability distribution :math:`p(x) \\propto sin(x)`

    The domain of this prior is [0,pi].
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "Sine"

    def draw_samples(self, num_samples):
        """
        Draw samples from a Sine distribution.

        Args:
            num_samples (float): the number of samples to generate

        Returns:
            np.array:  samples ranging from [0, pi) as floats.
        """

        # draw uniform from -1 to 1
        samples = np.random.uniform(-1, 1, num_samples)

        samples = np.arccos(samples) % np.pi

        return samples

    def compute_lnprob(self, element_array):
        """
        Compute the prior probability of each element given that its drawn from a sine prior

        Args:
            element_array (float or np.array of float): array of paramters to compute the prior probability of

        Returns:
            np.array: array of prior probabilities
        """
        normalization = 2.

        lnprob = np.log(np.sin(element_array)/normalization)

        # account for scalar inputs
        if np.shape(lnprob) == ():
            if (element_array >= np.pi) or (element_array <= 0):
                lnprob = -np.inf
        else:
            lnprob[(element_array >= np.pi) | (element_array <= 0)] = -np.inf

        return lnprob


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

    def __repr__(self):
        return "Linear"

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

        lnprob = np.log((self.m*element_array + self.b)/normalizer)

        # account for scalar inputs
        if np.shape(lnprob) == ():
            if (element_array >= x_intercept) or (element_array < 0):
                lnprob = -np.inf
        else:
            lnprob[(element_array >= x_intercept) | (element_array < 0)] = -np.inf

        return lnprob


def all_lnpriors(params, priors):
    """
    Calculates log(prior probability) of a set of parameters and a list of priors

    Args:
        params (np.array): size of N parameters
        priors (list): list of N prior objects corresponding to each parameter

    Returns:
        float: prior probability of this set of parameters
    """
    logp = 0.
    for param, prior in zip(params, priors):
        param = np.array([param])

        logp += prior.compute_lnprob(param)  # retrun a float

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
