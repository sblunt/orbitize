import abc
import numpy as np

"""
This module defines marginalizations to draw
samples of non-marginalized parameters and compute appropriate probability
"""

class Marginalizer(abc.ABC):

    QUANT_TYPES = []
    JITTER_PARAMS = []
    REQ_PARAMS = []
    MARG_PARAMS = []
    LNLIKE_TYPE = None
    MULTIBODY = False
    
    @abc.abstractmethod
    def apply(self, param_arr, jitter_arr=None, calculated_values={}):
        pass

class Marg1(Marginalizer):

    REQ_PARAMS = ["tau", "per"]
    MARG_PARAMS = ["sma", "pan"]

class Marg2(Marginalizer):

    REQ_PARAMS = ["tau", "sma", "pan"]
    MARG_PARAMS = ["gamma"]

class MargConstNoInput(Marginalizer):

    MARG_PARAMS = ["test"]

    def apply(self, param_arr, jitter_arr, calculated_values):
        shape = param_arr.shape
        if len(shape) > 1:
            n_orbits = shape[1]
            return 0, np.zeros((n_orbits)), np.ones((n_orbits)), calculated_values
        else:
            return 0, 0, 1, calculated_values
        
class MargPriorNoInput(Marginalizer):

    MARG_PARAMS = ["test"]

    def __init__(self, priors):
        self.dim = len(priors)
        self.MARG_PARAMS = ["test{}".format(i) for i in range(self.dim)]
        self.priors = priors

    def apply(self, param_arr, jitter_arr, calculated_values):
        shape = param_arr.shape
        if len(shape) > 1:
            n_orbits = shape[1]
            assert n_orbits > 1
            samples = np.array([[prior.draw_samples(1).item() for i in range(n_orbits)] for prior in self.priors])  
            assert samples.shape == (self.dim, n_orbits)  
            assert not np.isnan(samples).any(), samples
            return 0, np.zeros((n_orbits)), samples, calculated_values
        else:
            samples = np.array([prior.draw_samples(1).item() for prior in self.priors])
            assert samples.shape == (self.dim,), samples
            assert not np.isnan(samples).any(), samples
            return 0, 0, samples, calculated_values

def compatible_params(marginalizers, input_params):
    # TODO: Multibody, quant_type
    params = set(input_params)
    for marginalizer in marginalizers:
        req_params = set(marginalizer.REQ_PARAMS)
        marg_params = set(marginalizer.MARG_PARAMS)
        if len(req_params - params) == 0 and len(marg_params & params) == 0:
            params |= set(marginalizer.MARG_PARAMS)
        else:
            return False
    return True