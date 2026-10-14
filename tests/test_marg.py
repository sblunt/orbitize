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
import multiprocessing as mp

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

def test_incompatible():
    input_file = os.path.join(orbitize.DATADIR, 'GJ504.csv')
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, 'test_marg.hdf5')

    mySystem = system.System(
        1,
        data_table,
        1,
        0.01,
    )
    ind = data_table["quant_type"] == "seppa"
    epochs = data_table["epoch"][ind]
    sep = data_table["quant1"][ind]
    pa = data_table["quant2"][ind]
    sep_err = data_table["quant1_err"][ind]
    pa_err = data_table["quant2_err"][ind]
    covar = np.zeros(len(epochs))
    marg = marginalizer.OFTeI_sma_pan(epochs, sep, pa, sep_err, pa_err, covar)
    try:
        mySampler = sampler.MCMC(
            mySystem,
            num_temps=0,
            num_walkers=100,
            num_threads=1,
            marginalizers=[marg]
        )
        mySampler.run_sampler(400, 10, output_filename=output_file, periodic_save_freq=2)
    except ValueError:
        assert True
    else:
        assert False, "Did not raise ValueError despite marginalizer incompatible with basis"

def test_OFTeI():
    input_file = os.path.join(orbitize.DATADIR, 'GJ504.csv')
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, 'test_marg.hdf5')

    mtot = 1.22  # Msol
    plx = 56.95  # mas
    mass_err = 0.08  # Msol
    plx_err = 0.26  # mas
    tau_ref_epoch = 0

    mySystem = system.System(
        1,
        data_table,
        mtot,
        plx,
        tau_ref_epoch=tau_ref_epoch,
        fitting_basis="OftEiBasis"
    )
    sma_prior_min = 0.1  # au
    sma_prior_max = 500  # au
    period_prior_min_yr = np.sqrt(sma_prior_min**3 / mtot)  # [yr]
    period_prior_max_yr = np.sqrt(sma_prior_max**3 / mtot)  # [yr]
    mySystem.sys_priors[mySystem.param_idx["sma1"]] = priors.LogUniformPrior(sma_prior_min, sma_prior_max)
    mySystem.sys_priors[mySystem.param_idx["per1"]] = priors.LogUniformPrior(period_prior_min_yr, period_prior_max_yr)
    ind = data_table["quant_type"] == "seppa"
    epochs = data_table["epoch"][ind]
    sep = data_table["quant1"][ind]
    pa = data_table["quant2"][ind]
    sep_err = data_table["quant1_err"][ind]
    pa_err = data_table["quant2_err"][ind]
    covar = np.zeros(len(epochs))
    sma_prior = mySystem.sys_priors[mySystem.param_idx["sma1"]]
    pan_prior = mySystem.sys_priors[mySystem.param_idx["pan1"]]
    mass_prior = priors.GaussianPrior(mtot, mass_err)
    per_prior = mySystem.sys_priors[mySystem.param_idx["per1"]]
    implicit_mass_prior = priors.OFTeIMassPrior(sma_prior.minval, sma_prior.maxval, per_prior.minval, per_prior.maxval)
    marg = marginalizer.OFTeI_sma_pan(epochs, sep, pa, sep_err, pa_err, covar, sma_prior, pan_prior, mass_prior, implicit_mass_prior)
    mySampler = sampler.MCMC(
        mySystem,
        num_temps=0,
        num_walkers=100,
        num_threads=mp.cpu_count(),
        marginalizers=[marg]
    )
    mySampler.run_sampler(4000, 10, output_filename=output_file, periodic_save_freq=2)
    myResults = mySampler.results
    print(myResults.post.shape)

def test_OFTeI_RV():
    input_file = os.path.join(orbitize.DATADIR, 'HD4747.csv')
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, 'test_marg.hdf5')

    mtot = 0.84  # [Msol]
    plx = 53.18  # [mas]
    mass_err = 0.04  # [Msol]
    plx_err = 0.12  # [mas]
    tau_ref_epoch = 0

    mySystem = system.System(
        1,
        data_table,
        mtot,
        plx,
        tau_ref_epoch=tau_ref_epoch,
        fitting_basis="OftEiBasis"
    )
    sma_prior_min = 0.1  # au
    sma_prior_max = 500  # au
    period_prior_min_yr = np.sqrt(sma_prior_min**3 / mtot)  # [yr]
    period_prior_max_yr = np.sqrt(sma_prior_max**3 / mtot)  # [yr]
    mySystem.sys_priors[mySystem.param_idx["sma1"]] = priors.LogUniformPrior(sma_prior_min, sma_prior_max)
    mySystem.sys_priors[mySystem.param_idx["per1"]] = priors.LogUniformPrior(period_prior_min_yr, period_prior_max_yr)
    ind = data_table["quant_type"] == "seppa"
    epochs = data_table["epoch"][ind]
    sep = data_table["quant1"][ind]
    pa = data_table["quant2"][ind]
    sep_err = data_table["quant1_err"][ind]
    pa_err = data_table["quant2_err"][ind]
    covar = np.zeros(len(epochs))
    sma_prior = mySystem.sys_priors[mySystem.param_idx["sma1"]]
    pan_prior = mySystem.sys_priors[mySystem.param_idx["pan1"]]
    mass_prior = priors.GaussianPrior(mtot, mass_err)
    per_prior = mySystem.sys_priors[mySystem.param_idx["per1"]]
    implicit_mass_prior = priors.OFTeIMassPrior(sma_prior.minval, sma_prior.maxval, per_prior.minval, per_prior.maxval)
    marg = marginalizer.OFTeI_sma_pan(epochs, sep, pa, sep_err, pa_err, covar, sma_prior, pan_prior, mass_prior, implicit_mass_prior)
    mySampler = sampler.MCMC(
        mySystem,
        num_temps=0,
        num_walkers=100,
        num_threads=mp.cpu_count(),
        marginalizers=[marg]
    )
    mySampler.run_sampler(4000, 10, output_filename=output_file, periodic_save_freq=2)
    myResults = mySampler.results
    assert mySampler.non_marg_quants == ["rv"]
    print(myResults.post.shape)

if __name__ == "__main__":
    # test_MargConstNoInput()
    test_OFTeI()