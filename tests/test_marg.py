import pytest

import os
import numpy as np
import orbitize
from orbitize.driver import Driver
import orbitize.sampler as sampler
import orbitize.marginalizer as marginalizer
import orbitize.priors as priors
import orbitize.system as system
import orbitize.read_input as read_input
import orbitize.results as results
from orbitize.plot import plot_corner
import matplotlib.pyplot as plt
import pytest

def test_MargConstNoInput():
    # use the test_csv dir
    input_file = os.path.join(orbitize.DATADIR, 'GJ504.csv')
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, 'test_marg.hdf5')

    mySystem = system.System(
        1,
        data_table,
        1,
        0.01,
    )
    marg_prior = 1
    marg = marginalizer.MargConstNoInput()
    mySystem.sys_priors.append(marg_prior)
    mySystem.param_idx["test"] = len(mySystem.sys_priors)-1
    mySystem.labels.append("test")
    mySampler = sampler.MCMC(
        mySystem,
        num_temps=0,
        num_walkers=100,
        num_threads=1,
        marginalizers=[marg]
    )
    mySampler.run_sampler(400, 10, output_filename=output_file, periodic_save_freq=2)
    myResults = mySampler.results
    assert (myResults.post[:, 8] == 1).all()

def test_MargPriorNoInput():
    # use the test_csv dir
    input_file = os.path.join(orbitize.DATADIR, 'GJ504.csv')
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, 'test_marg.hdf5')

    mySystem = system.System(
        1,
        data_table,
        1,
        0.01,
    )
    marg_priors = [priors.UniformPrior(0.5, 1.5), priors.GaussianPrior(10, 1)]
    marg = marginalizer.MargPriorNoInput(marg_priors)
    regular = len(mySystem.sys_priors)
    mySystem.sys_priors.extend(marg_priors)
    for i, label in enumerate(marg.MARG_PARAMS):
        mySystem.param_idx[label] = regular + i
    assert mySystem.param_idx["test0"] == 8
    assert mySystem.param_idx["test1"] == 9
    mySystem.labels.extend(marg.MARG_PARAMS)
    mySampler = sampler.MCMC(
        mySystem,
        num_temps=0,
        num_walkers=100,
        num_threads=1,
        marginalizers=[marg]
    )
    mySampler.run_sampler(400, 10, output_filename=output_file, periodic_save_freq=2)
    myResults = mySampler.results
    assert (0.5 <= myResults.post[:, 8]).all() and (myResults.post[:, 8] <= 1.5).all()
    assert np.mean(myResults.post[:, 8]) == pytest.approx(1, rel=0.1)
    assert np.std(myResults.post[:, 9]) == pytest.approx(1, rel=0.1)
    assert np.mean(myResults.post[:, 9]) == pytest.approx(10, rel=0.1)

if __name__ == "__main__":
    test_MargConstNoInput()