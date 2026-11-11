import pandas as pd
import numpy as np, os
from orbitize import system as osys, priors, sampler, gaia, DATADIR, marginalizer

# read solutions to pull pmra, pmdec, ra, dec that will be provided in the catalog.
solutions = pd.read_csv(os.path.join(DATADIR, "injected_solutions.csv"))
target_file = "target_1.csv"
row = solutions[solutions["filename"] == target_file].iloc[0]
dr4_filepath = os.path.join(DATADIR, target_file)
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

mySystem = osys.System(
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

myMarginalizer = marginalizer.OFTeI_DR4_gaia(dr4)

mySystem.reweight_priors.append(priors.OFTeIMassReweightPrior(implicit_mass_prior, m0_prior, mySystem.basis))
mySystem.reweight_priors.append(priors.OFTeIUVReweightPrior(sma_prior, pan_prior, mySystem.basis))

#run mcmc sampler
num_walkers, num_temps, num_threads = 100, 5, 1
total_steps, burn_steps, thin = 5000, 1000, 5
mySampler = sampler.MCMC(mySystem, num_temps=num_temps,
                     num_walkers=num_walkers, num_threads=num_threads, marginalizers=[myMarginalizer])

mySampler.run_sampler(num_walkers * total_steps, burn_steps=burn_steps, thin=thin)

myResults = mySampler.results

myResults.save_results("target_1_oftei_gaia.hdf5")