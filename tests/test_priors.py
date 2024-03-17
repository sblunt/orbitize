import numpy as np
import pytest
import os
from scipy.stats import norm as nm

import orbitize.priors as priors
from orbitize.system import System
from orbitize.read_input import read_file
from orbitize.sampler import MCMC
from orbitize import DATADIR

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


def test_obsprior():
    """
    Test API setup with obs prior and run it a few MCMC steps to make sure nothing
    breaks.
    """

    input_file = os.path.join(DATADIR, "xyz_test_data.csv")
    data_table = read_file(input_file)
    mtot = 1.0

    mySystem = System(
        1, data_table, mtot, 10.0, mass_err=0, plx_err=0, fitting_basis="ObsPriors"
    )

    # construct sampler
    n_walkers = 20
    num_temps = 1
    my_sampler = MCMC(mySystem, num_temps, n_walkers, num_threads=1)

    ra_err = mySystem.data_table["quant1_err"]
    dec_err = mySystem.data_table["quant2_err"]
    epochs = mySystem.data_table["epoch"]

    # define the `ObsPrior` object
    my_obsprior = priors.ObsPrior(ra_err, dec_err, epochs, mtot)

    # set the priors on `per`, `ecc`, `tp` to point to this object
    for i in [
        mySystem.param_idx["per1"],
        mySystem.param_idx["ecc1"],
        mySystem.param_idx["tp1"],
    ]:
        mySystem.sys_priors[i] = my_obsprior

    # run the mcmc a few steps to make sure nothing breaks
    my_sampler.run_sampler(5, burn_steps=0)


if __name__ == "__main__":
    # test_compute_lnprob()
    # test_draw_samples()
    test_obsprior()
