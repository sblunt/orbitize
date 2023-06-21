import numpy as np
import pytest
from scipy.stats import norm as nm

import orbitize.priors as priors

threshold = 1e-1

initialization_inputs = {
    priors.GaussianPrior: [1000.0, 1.0],
    priors.LogUniformPrior: [1.0, 2.0],
    priors.UniformPrior: [0.0, 1.0],
    priors.SinPrior: [],
    priors.LinearPrior: [-2.0, 2.0],
}

expected_means_mins_maxes = {
    priors.GaussianPrior: (1000.0, 0.0, np.inf),
    priors.LogUniformPrior: (1 / np.log(2), 1.0, 2.0),
    priors.UniformPrior: (0.5, 0.0, 1.0),
    priors.SinPrior: (np.pi / 2.0, 0.0, np.pi),
    priors.LinearPrior: (1.0 / 3.0, 0.0, 1.0),
}

lnprob_inputs = {
    priors.GaussianPrior: np.array([-3.0, np.inf, 1000.0, 999.0]),
    priors.LogUniformPrior: np.array([-1.0, 0.0, 1.0, 1.5, 2.0, 2.5]),
    priors.UniformPrior: np.array([0.0, 0.5, 1.0, -1.0, 2.0]),
    priors.SinPrior: np.array([0.0, np.pi / 2.0, np.pi, 10.0, -1.0]),
    priors.LinearPrior: np.array([0.0, 0.5, 1.0, 2.0, -1.0]),
}

expected_probs = {
    priors.GaussianPrior: np.array(
        [
            0.0,
            0.0,
            nm(1000.0, 1.0).pdf(1000.0) * np.sqrt(2 * np.pi),
            nm(1000.0, 1.0).pdf(999.0) * np.sqrt(2 * np.pi),
        ]
    ),
    priors.LogUniformPrior: np.array([0.0, 0.0, 1.0, 2.0 / 3.0, 0.5, 0.0]) / np.log(2),
    priors.UniformPrior: np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
    priors.SinPrior: np.array([0.0, 0.5, 0.0, 0.0, 0.0]),
    priors.LinearPrior: np.array([2.0, 1.0, 0.0, 0.0, 0.0]),
}


def test_draw_samples():
    """
    Test basic functionality of `draw_samples()` method of each `Prior` class.
    """
    for Prior in initialization_inputs.keys():
        inputs = initialization_inputs[Prior]

        TestPrior = Prior(*inputs)
        samples = TestPrior.draw_samples(10000)

        exp_mean, exp_min, exp_max = expected_means_mins_maxes[Prior]
        assert np.mean(samples) == pytest.approx(exp_mean, abs=threshold)
        assert np.min(samples) > exp_min
        assert np.max(samples) < exp_max


def test_compute_lnprob():
    """
    Test basic functionality of `compute_lnprob()` method of each `Prior` class.
    """
    for Prior in initialization_inputs.keys():
        inputs = initialization_inputs[Prior]

        TestPrior = Prior(*inputs)
        values2test = lnprob_inputs[Prior]

        lnprobs = TestPrior.compute_lnprob(values2test)

        assert np.log(expected_probs[Prior]) == pytest.approx(lnprobs, abs=threshold)


if __name__ == "__main__":
    test_compute_lnprob()
    test_draw_samples()
