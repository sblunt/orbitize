import matplotlib.pyplot as plt

import orbitize

# Based on driver.py

import numpy as np
from orbitize import read_input, system, sampler, priors, marginalizer, plot
import multiprocessing as mp
import orbitize
import time

savedir = ""
savefile = f"{savedir}GJ504_oftei.hdf5"

start = time.time()

# System parameters
datafile = "GJ504.csv"
num_secondary_bodies = 1
system_mass = 1.22  # Msol
plx = 56.95  # mas
mass_err = 0.08  # Msol
plx_err = 0.26  # mas

# run MCMC
num_threads = mp.cpu_count()
num_temps = 20
num_walkers = 1000
num_steps = 1000000  # n_walkers x n_steps_per_walker
burn_steps = 1000
thin = 100
tau_ref_epoch = 0

# Read in data
data_table = read_input.read_file(orbitize.DATADIR + datafile)

# Initialize System object which stores data & sets priors
mySystem = system.System(
    1,
    data_table,
    system_mass,
    plx,
    tau_ref_epoch=tau_ref_epoch,
    fitting_basis="OftEiBasis"
)

sma_prior_min = 0.1  # au
sma_prior_max = 500  # au
period_prior_min_yr = np.sqrt(sma_prior_min**3 / system_mass)  # [yr]
period_prior_max_yr = np.sqrt(sma_prior_max**3 / system_mass)  # [yr]
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
mass_prior = priors.GaussianPrior(system_mass, mass_err)
per_prior = mySystem.sys_priors[mySystem.param_idx["per1"]]
implicit_mass_prior = priors.OFTeIMassPrior(sma_prior.minval, sma_prior.maxval, per_prior.minval, per_prior.maxval)
marg = marginalizer.OFTeI_sma_pan(epochs, sep, pa, sep_err, pa_err, covar, sma_prior, pan_prior, mass_prior, implicit_mass_prior)
mySampler = sampler.MCMC(
    mySystem,
    num_temps=num_temps,
    num_walkers=num_walkers,
    num_threads=num_threads,
    marginalizers=[marg]
)

mySampler.run_sampler(num_steps, burn_steps, output_filename=savefile, periodic_save_freq=None)
myResults = mySampler.results

end = time.time()
print(f"Processing time: {end} - {start} = {(end-start) / 60} minutes")

# make corner plot
p = plot.Plotter(myResults, num_orbits_to_plot=10000)
fig = p.plot_corner()
fig.savefig(f"{savedir}GJ504_oftei_corner.png")
fig2 = p.plot_orbits()
fig2.savefig(f"{savedir}GJ504_oftei_orbits.png")