import pytest

import os
import numpy as np
import pandas as pd
import orbitize
from orbitize.driver import Driver
import orbitize.sampler as sampler
import orbitize.marginalizer as marginalizer
import orbitize.priors as priors
import orbitize.system as system
import orbitize.read_input as read_input
import orbitize.gaia as gaia
import orbitize.results as results
from orbitize.plot import plot_corner
import matplotlib.pyplot as plt
import pytest
import multiprocessing as mp



class Marg1(marginalizer.Marginalizer):
    REQ_PARAMS = ["tau", "per"]
    MARG_PARAMS = ["sma", "pan"]


class Marg2(marginalizer.Marginalizer):
    REQ_PARAMS = ["tau", "sma", "pan"]
    MARG_PARAMS = ["gamma"]


class MargConstNoInput(marginalizer.Marginalizer):
    MARG_PARAMS = ["test"]

    def apply(self, param_arr, calculated_values):
        shape = param_arr.shape
        if len(shape) > 1:
            n_orbits = shape[1]
            return 0, np.zeros((n_orbits)), np.ones((n_orbits)), calculated_values
        else:
            return 0, 0, 1, calculated_values


class MargPriorNoInput(marginalizer.Marginalizer):
    MARG_PARAMS = ["test"]

    def __init__(self, priors):
        self.dim = len(priors)
        self.MARG_PARAMS = ["test{}".format(i) for i in range(self.dim)]
        self.priors = priors

    def apply(self, param_arr, calculated_values):
        shape = param_arr.shape
        if len(shape) > 1:
            n_orbits = shape[1]
            assert n_orbits > 1
            samples = np.array(
                [
                    [prior.draw_samples(1).item() for i in range(n_orbits)]
                    for prior in self.priors
                ]
            )
            assert samples.shape == (self.dim, n_orbits)
            assert not np.isnan(samples).any(), samples
            return 0, np.zeros((n_orbits)), samples, calculated_values
        else:
            samples = np.array([prior.draw_samples(1).item() for prior in self.priors])
            assert samples.shape == (self.dim,), samples
            assert not np.isnan(samples).any(), samples
            return 0, 0, samples, calculated_values


def test_MargConstNoInput():
    # use the test_csv dir
    input_file = os.path.join(orbitize.DATADIR, "GJ504.csv")
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, "test_marg.hdf5")

    mySystem = system.System(
        1,
        data_table,
        1,
        0.01,
    )
    marg_prior = 1
    marg = MargConstNoInput()
    mySystem.sys_priors.append(marg_prior)
    mySystem.param_idx["test"] = len(mySystem.sys_priors) - 1
    mySystem.labels.append("test")
    mySampler = sampler.MCMC(
        mySystem, num_temps=1, num_walkers=100, num_threads=1, marginalizers=[marg]
    )
    mySampler.run_sampler(400, 10, output_filename=output_file, periodic_save_freq=2)
    myResults = mySampler.results
    assert (myResults.post[:, 8] == 1).all()


def test_MargPriorNoInput():
    # use the test_csv dir
    input_file = os.path.join(orbitize.DATADIR, "GJ504.csv")
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, "test_marg.hdf5")

    mySystem = system.System(
        1,
        data_table,
        1,
        0.01,
    )
    marg_priors = [priors.UniformPrior(0.5, 1.5), priors.GaussianPrior(10, 1)]
    marg = MargPriorNoInput(marg_priors)
    regular = len(mySystem.sys_priors)
    mySystem.sys_priors.extend(marg_priors)
    for i, label in enumerate(marg.MARG_PARAMS):
        mySystem.param_idx[label] = regular + i
    assert mySystem.param_idx["test0"] == 8
    assert mySystem.param_idx["test1"] == 9
    mySystem.labels.extend(marg.MARG_PARAMS)
    mySampler = sampler.MCMC(
        mySystem, num_temps=1, num_walkers=100, num_threads=1, marginalizers=[marg]
    )
    mySampler.run_sampler(400, 10, output_filename=output_file, periodic_save_freq=2)
    myResults = mySampler.results
    assert (0.5 <= myResults.post[:, 8]).all() and (myResults.post[:, 8] <= 1.5).all()
    assert np.mean(myResults.post[:, 8]) == pytest.approx(1, rel=0.1)
    assert np.std(myResults.post[:, 9]) == pytest.approx(1, rel=0.2)
    assert np.mean(myResults.post[:, 9]) == pytest.approx(10, rel=0.1)


def test_incompatible():
    input_file = os.path.join(orbitize.DATADIR, "GJ504.csv")
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, "test_marg.hdf5")

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
            mySystem, num_temps=1, num_walkers=100, num_threads=1, marginalizers=[marg]
        )
        mySampler.run_sampler(
            400, 10, output_filename=output_file, periodic_save_freq=2
        )
    except ValueError:
        assert True
    else:
        assert False, (
            "Did not raise ValueError despite marginalizer incompatible with basis"
        )


def test_OFTeI():
    input_file = os.path.join(orbitize.DATADIR, "GJ504.csv")
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, "test_marg.hdf5")

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
        fitting_basis="OFTeIBasis",
    )
    sma_prior_min = 0.1  # au
    sma_prior_max = 500  # au
    period_prior_min_yr = np.sqrt(sma_prior_min**3 / mtot)  # [yr]
    period_prior_max_yr = np.sqrt(sma_prior_max**3 / mtot)  # [yr]
    mySystem.sys_priors[mySystem.param_idx["sma1"]] = priors.LogUniformPrior(
        sma_prior_min, sma_prior_max
    )
    mySystem.sys_priors[mySystem.param_idx["per1"]] = priors.LogUniformPrior(
        period_prior_min_yr, period_prior_max_yr
    )
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
    implicit_mass_prior = priors.OFTeIMassPrior(
        sma_prior.minval, sma_prior.maxval, per_prior.minval, per_prior.maxval
    )
    marg = marginalizer.OFTeI_sma_pan(
        epochs,
        sep,
        pa,
        sep_err,
        pa_err,
        covar,
        tau_ref_epoch,
    )
    reweight_prior = priors.OFTeIMassReweightPrior(implicit_mass_prior, mass_prior, mySystem.basis)
    mySystem.reweight_priors.append(reweight_prior)
    mySampler = sampler.MCMC(
        mySystem,
        num_temps=5,
        num_walkers=10,
        num_threads=mp.cpu_count(),
        marginalizers=[marg],
    )
    mySampler.run_sampler(5000, 20, output_filename=output_file, periodic_save_freq=2)
    assert not mySampler.force_valid_marg
    assert len(mySystem.reweight_priors) == 1
    myResults = mySampler.results

    class TestSampler(sampler.Sampler):
        def run_sampler(): ...

    noMargSampler = TestSampler(mySystem, marginalizers=[])
    true_l = []
    for param in myResults.post:
        l = noMargSampler._logl(param)[0]
        true_l.append(l)
    diff = np.abs(true_l - myResults.lnlike)
    print((true_l - myResults.lnlike)[np.where(diff > 0.2)])
    n_small_diff = np.sum(diff < 0.2)
    n_tot = myResults.post.shape[0]
    assert n_small_diff / n_tot > 0.99, np.where(diff < 0.2)


def test_OFTeI_RV():
    # TODO: add secondary mass compat so RV can run
    input_file = os.path.join(orbitize.DATADIR, "HD4747.csv")
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, "test_marg_rv.hdf5")

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
        plx_err=plx_err,
        tau_ref_epoch=tau_ref_epoch,
        fitting_basis="OFTeIBasis",
    )
    assert len(mySystem.reweight_priors) == 0

    sma_prior_min = 0.1  # au
    sma_prior_max = 500  # au
    period_prior_min_yr = np.sqrt(sma_prior_min**3 / mtot)  # [yr]
    period_prior_max_yr = np.sqrt(sma_prior_max**3 / mtot)  # [yr]
    mySystem.sys_priors[mySystem.param_idx["sma1"]] = priors.LogUniformPrior(
        sma_prior_min, sma_prior_max
    )
    mySystem.sys_priors[mySystem.param_idx["per1"]] = priors.LogUniformPrior(
        period_prior_min_yr, period_prior_max_yr
    )
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
    implicit_mass_prior = priors.OFTeIMassPrior(
        sma_prior.minval, sma_prior.maxval, per_prior.minval, per_prior.maxval
    )
    marg = marginalizer.OFTeI_sma_pan(
        epochs,
        sep,
        pa,
        sep_err,
        pa_err,
        covar,
        tau_ref_epoch,
    )
    reweight_prior = priors.OFTeIMassReweightPrior(implicit_mass_prior, mass_prior, mySystem.basis)
    mySystem.reweight_priors.append(reweight_prior)
    mySampler = sampler.MCMC(
        mySystem,
        num_temps=2,
        num_walkers=100,
        num_threads=mp.cpu_count(),
        marginalizers=[marg],
    )
    mySampler.run_sampler(4000, 10, output_filename=output_file, periodic_save_freq=2)
    myResults = mySampler.results
    assert len(mySystem.reweight_priors) == 1
    assert mySampler.non_marg_quants == ["rv"]

    class TestSampler(sampler.Sampler):
        def run_sampler(): ...

    noMargSampler = TestSampler(mySystem, marginalizers=[])
    true_l = []
    for param in myResults.post:
        l = noMargSampler._logl(param)[0]
        true_l.append(l)
    diff = np.abs(true_l - myResults.lnlike)
    print((true_l - myResults.lnlike)[np.where(diff > 0.2)])
    n_small_diff = np.sum(diff < 0.2)
    n_tot = myResults.post.shape[0]
    assert n_small_diff / n_tot > 0.99, np.where(diff < 0.2)


def test_OFTeI_RV_auto():
    input_file = os.path.join(orbitize.DATADIR, "HD4747.csv")
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, "test_marg_rv.hdf5")

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
        plx_err=plx_err,
        mass_err=mass_err,
        tau_ref_epoch=tau_ref_epoch,
        fitting_basis="OFTeIBasis",
    )
    sma_prior_min = 0.1  # au
    sma_prior_max = 500  # au
    period_prior_min_yr = np.sqrt(sma_prior_min**3 / mtot)  # [yr]
    period_prior_max_yr = np.sqrt(sma_prior_max**3 / mtot)  # [yr]
    mySystem.sys_priors[mySystem.param_idx["sma1"]] = priors.LogUniformPrior(
        sma_prior_min, sma_prior_max
    )
    mySystem.sys_priors[mySystem.param_idx["per1"]] = priors.LogUniformPrior(
        period_prior_min_yr, period_prior_max_yr
    )
    mySampler = sampler.MCMC(
        mySystem,
        num_temps=2,
        num_walkers=100,
        num_threads=mp.cpu_count(),
        marginalizers=None,  # automatically construct marginalization
    )
    assert len(mySampler.marginalizers) == 1
    marg = mySampler.marginalizers[0]
    assert isinstance(marg, marginalizer.OFTeI_sma_pan)

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
    implicit_mass_prior = priors.OFTeIMassPrior(
        sma_prior.minval, sma_prior.maxval, per_prior.minval, per_prior.maxval
    )
    true_marg = marginalizer.OFTeI_sma_pan(
        epochs,
        sep,
        pa,
        sep_err,
        pa_err,
        covar,
        tau_ref_epoch,
    )

    assert len(mySystem.reweight_priors) == 1
    assert mass_prior.mu == mySystem.reweight_priors[0].new_prior.mu, mySystem.reweight_priors
    assert mass_prior.sigma == mySystem.reweight_priors[0].new_prior.sigma
    assert implicit_mass_prior.sma_min == mySystem.reweight_priors[0].implied_prior.sma_min
    assert implicit_mass_prior.sma_max == mySystem.reweight_priors[0].implied_prior.sma_max
    assert implicit_mass_prior.per_min == mySystem.reweight_priors[0].implied_prior.per_min
    assert implicit_mass_prior.per_max == mySystem.reweight_priors[0].implied_prior.per_max
    assert (true_marg.epochs == marg.epochs).all()
    assert (true_marg.seps == marg.seps).all()
    assert (true_marg.pas == marg.pas).all()
    assert (true_marg.sigma_seps == marg.sigma_seps).all()
    assert (true_marg.sigma_pas == marg.sigma_pas).all()
    assert (true_marg.covar_seps_pas == marg.covar_seps_pas).all()
    assert true_marg.tau_ref_epoch == marg.tau_ref_epoch
    assert true_marg.n_obs == marg.n_obs

    mySampler.run_sampler(4000, 10, output_filename=output_file, periodic_save_freq=2)
    myResults = mySampler.results
    assert mySampler.non_marg_quants == ["rv"]
    print(myResults.post.shape)


def test_OFTeI_radec():
    input_file = os.path.join(orbitize.DATADIR, "GJ504.csv")
    data_table = read_input.read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, "test_marg_radec.hdf5")

    # read in seppa data table and turn into raddec data table
    data_table = read_input.read_file(input_file)
    data_ra, data_dec = system.seppa2radec(data_table["quant1"], data_table["quant2"])
    data_raerr, data_decerr, data_radeccorr = [], [], []

    for row in data_table:
        raerr, decerr, radec_corr = system.transform_errors(
            row["quant1"],
            row["quant2"],
            row["quant1_err"],
            row["quant2_err"],
            0,
            system.seppa2radec,
            nsamps=10000000,
        )
        data_raerr.append(raerr)
        data_decerr.append(decerr)
        data_radeccorr.append(radec_corr)

    data_table["quant1"] = data_ra
    data_table["quant2"] = data_dec
    data_table["quant1_err"] = np.array(data_raerr)
    data_table["quant2_err"] = np.array(data_decerr)
    data_table["quant12_corr"] = np.array(data_radeccorr)
    data_table["quant_type"] = np.array(["radec" for _ in data_table])

    mtot = 1.22  # Msol
    plx = 56.95  # mas
    mass_err = 0.08  # Msol
    plx_err = 0.26  # mas
    tau_ref_epoch = 0

    try:
        mySystem = system.System(
            1,
            data_table,
            mtot,
            plx,
            plx_err=plx_err,
            tau_ref_epoch=tau_ref_epoch,
            fitting_basis="OFTeIBasis",
        )
        sma_prior_min = 0.1  # au
        sma_prior_max = 500  # au
        period_prior_min_yr = np.sqrt(sma_prior_min**3 / mtot)  # [yr]
        period_prior_max_yr = np.sqrt(sma_prior_max**3 / mtot)  # [yr]
        mySystem.sys_priors[mySystem.param_idx["sma1"]] = priors.LogUniformPrior(
            sma_prior_min, sma_prior_max
        )
        mySystem.sys_priors[mySystem.param_idx["per1"]] = priors.LogUniformPrior(
            period_prior_min_yr, period_prior_max_yr
        )
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
        implicit_mass_prior = priors.OFTeIMassPrior(
            sma_prior.minval, sma_prior.maxval, per_prior.minval, per_prior.maxval
        )
        marg = marginalizer.OFTeI_sma_pan(
            epochs,
            sep,
            pa,
            sep_err,
            pa_err,
            covar,
            tau_ref_epoch,
        )
        reweight_prior = priors.OFTeIMassReweightPrior(implicit_mass_prior, mass_prior, mySystem.basis)
        mySystem.reweight_priors.append(reweight_prior)
        mySampler = sampler.MCMC(
            mySystem,
            num_temps=5,
            num_walkers=100,
            num_threads=mp.cpu_count(),
            marginalizers=[marg],
        )
    except ValueError:
        assert True
    else:
        assert False
    mySystem = system.System(
        1,
        data_table,
        mtot,
        plx,
        plx_err=plx_err,
        tau_ref_epoch=tau_ref_epoch,
        fitting_basis="OFTeIBasis",
    )
    mySystem.convert_data_table_radec2seppa()
    sma_prior_min = 0.1  # au
    sma_prior_max = 500  # au
    period_prior_min_yr = np.sqrt(sma_prior_min**3 / mtot)  # [yr]
    period_prior_max_yr = np.sqrt(sma_prior_max**3 / mtot)  # [yr]
    mySystem.sys_priors[mySystem.param_idx["sma1"]] = priors.LogUniformPrior(
        sma_prior_min, sma_prior_max
    )
    mySystem.sys_priors[mySystem.param_idx["per1"]] = priors.LogUniformPrior(
        period_prior_min_yr, period_prior_max_yr
    )
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
    implicit_mass_prior = priors.OFTeIMassPrior(
        sma_prior.minval, sma_prior.maxval, per_prior.minval, per_prior.maxval
    )
    marg = marginalizer.OFTeI_sma_pan(
        epochs,
        sep,
        pa,
        sep_err,
        pa_err,
        covar,
        tau_ref_epoch,
    )
    reweight_prior = priors.OFTeIMassReweightPrior(implicit_mass_prior, mass_prior, mySystem.basis)
    mySystem.reweight_priors.append(reweight_prior)
    mySampler = sampler.MCMC(
        mySystem,
        num_temps=5,
        num_walkers=100,
        num_threads=mp.cpu_count(),
        marginalizers=[marg],
    )
    mySampler.run_sampler(4000, 10, output_filename=output_file, periodic_save_freq=2)
    assert not mySampler.force_valid_marg
    myResults = mySampler.results
    print(myResults.post.shape)

    class TestSampler(sampler.Sampler):
        def run_sampler(): ...

    noMargSampler = TestSampler(mySystem, marginalizers=[])
    true_l = []
    for param in myResults.post:
        l = noMargSampler._logl(param)[0]
        true_l.append(l)
    diff = np.abs(true_l - myResults.lnlike)
    print((true_l - myResults.lnlike)[np.where(diff > 0.2)])
    n_small_diff = np.sum(diff < 0.2)
    n_tot = myResults.post.shape[0]
    assert n_small_diff / n_tot > 0.99, np.where(diff < 0.2)


# TODO: test_OFTeI_radec_auto

def test_OFTeI_gaia():
    # read solutions to pull pmra, pmdec, ra, dec that will be provided in the catalog.
    solutions = pd.read_csv(os.path.join(orbitize.DATADIR, "injected_solutions.csv"))
    target_file = "target_1.csv"
    row = solutions[solutions["filename"] == target_file].iloc[0]
    dr4_filepath = os.path.join(orbitize.DATADIR, target_file)
    plx_guess = row["parallax_mas"]
    plx_err = 0.5

    # set up prior ranges, sampler params, ref epochs
    m0_guess, m0_err = 1.0, 0.1
    m1_lo, m1_hi = 1e-4, 0.1
    sma_lo, sma_hi = 0.05, 50.0
    ecc_lo, ecc_hi = 0.01, 0.99
    GAIA_REF_MJD = 2457936.875-2400000.5
    TAU_REF_EPOCH = 58849

    dr4 = gaia.DR4LogProb(
        gaia_num=0,
        dr4_filepath=dr4_filepath,
        ref_epoch_mjd=GAIA_REF_MJD,
    )

    mySystem = system.System(
        num_secondary_bodies=1, data_table=None,
        stellar_or_system_mass=m0_guess, plx=plx_guess,
        mass_err=m0_err, plx_err=plx_err,
        fit_secondary_mass=True, gaia=dr4, tau_ref_epoch=TAU_REF_EPOCH, fitting_basis="OFTeIGaiaBasis"
    )

    #set up prior pdf in system
    pi = mySystem.param_idx
    sma_prior = priors.LogUniformPrior(sma_lo, sma_hi)
    mySystem.sys_priors[pi["ecc1"]] = priors.UniformPrior(ecc_lo, ecc_hi)
    mySystem.sys_priors[pi["inc1"]] = priors.SinPrior()
    mySystem.sys_priors[pi["aop1"]] = priors.UniformPrior(0, 2*np.pi)
    sma_photo_hi = sma_hi * m1_hi / (m0_guess - 2 * m0_err)
    u_hi = sma_photo_hi * (plx_guess + 2 * plx_err)
    mySystem.sys_priors[pi["mfrac1"]] = priors.UniformPrior(m1_lo/(m0_guess + 2 * m0_err), m1_hi/(m0_guess - 2 * m0_err))
    mySystem.sys_priors[pi["u"]] = priors.UniformPrior(-u_hi, u_hi)
    mySystem.sys_priors[pi["v"]] = priors.UniformPrior(-u_hi, u_hi)
    pan_prior = priors.UniformPrior(0, 2*np.pi)
    mySystem.sys_priors[pi["tau1"]] = priors.UniformPrior(0., 1.)
    mySystem.sys_priors[pi["plx"]] = priors.GaussianPrior(plx_guess, plx_err, no_negatives=True)
    m0_prior = priors.GaussianPrior(m0_guess, m0_err, no_negatives=True)
    # mySystem.sys_priors[pi["m1"]] = priors.LogUniformPrior(m1_lo, m1_hi) TODO: reweight
    plx_prior = mySystem.sys_priors[pi["plx"]]
    # TODO: mtot intead of m0
    period_prior_min_yr = np.sqrt(sma_lo**3 / m0_guess)
    period_prior_max_yr = np.sqrt(sma_hi**3 / m0_guess)
    implicit_mass_prior = priors.OFTeIMassPrior(sma_lo, sma_hi, period_prior_min_yr, period_prior_max_yr)
    mySystem.sys_priors[pi["per1"]] = priors.LogUniformPrior(period_prior_min_yr, period_prior_max_yr)

    myMarginalizer = marginalizer.OFTeI_DR4_gaia(dr4, TAU_REF_EPOCH)

    mySystem.reweight_priors.append(priors.OFTeIMassReweightPrior(implicit_mass_prior, m0_prior, mySystem.basis))
    mySystem.reweight_priors.append(priors.OFTeIUVReweightPrior(sma_prior, pan_prior, mySystem.basis))

    #run mcmc sampler
    num_walkers, num_temps, num_threads = 20, 5, 1
    total_steps, burn_steps, thin = 200, 200, 5
    mySampler = sampler.MCMC(mySystem, num_temps=num_temps,
                        num_walkers=num_walkers, num_threads=num_threads, marginalizers=[myMarginalizer])

    mySampler.run_sampler(num_walkers * total_steps, burn_steps=burn_steps, thin=thin)

    myResults = mySampler.results
    post = myResults.post

    assert post.shape[0] == num_walkers * total_steps / thin

    class TestSampler(sampler.Sampler):
            def run_sampler(): ...
    
    noMargSampler = TestSampler(mySystem, marginalizers=[])
    assert noMargSampler.system.gaia
    assert noMargSampler.marginalizers == []
    assert not noMargSampler.gaia_marg
    true_l = []
    for param in myResults.post:
        l = noMargSampler._logl(param)[0]
        true_l.append(l)
    diff = np.abs(true_l - myResults.lnlike)
    print(f"{true_l=}, {myResults.lnlike=}")
    n_small_diff = np.sum(diff < 0.2)
    n_tot = myResults.post.shape[0]
    assert n_small_diff / n_tot > 0.99, np.where(diff < 0.2)
    

if __name__ == "__main__":
    # test_MargConstNoInput()
    # test_OFTeI()
    test_OFTeI_gaia()