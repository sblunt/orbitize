import os
import matplotlib.pyplot as plt

import orbitize
from orbitize import system, read_input, priors, sampler

"""
Attempts to reproduce the 1 planet orbit fit in GRAVITY Collaboration et al. (2019)
"""

# End to end example with beta Pic astrometry
# Based on driver.py

import numpy as np
from orbitize import read_input, system, sampler, priors
import multiprocessing as mp
import orbitize


# System parameters
datafile='hr8799e_1epochgravity.csv'
num_secondary_bodies=1
system_mass=1.52 # Msol
plx=24.2175 #mas
mass_err=0.15 # Msol
plx_err=0.0881 #mas

# Sampler parameters
likelihood_func_name='chi2_lnlike'
n_temps=20
n_walkers=1000
n_threads=mp.cpu_count()
total_orbits=10000000 # n_walkers x num_steps_per_walker
burn_steps=50000

tau_ref_epoch = 50000


# Read in data
data_table = read_input.read_file(datafile)

# Initialize System object which stores data & sets priors
my_system = system.System(
    num_secondary_bodies, data_table, system_mass,
    plx, mass_err=mass_err, plx_err=plx_err, tau_ref_epoch=tau_ref_epoch
)

my_sampler = sampler.MCMC(my_system, n_temps, n_walkers, n_threads)

# Run the sampler to compute some orbits, yeah!
# Results stored in bP_sampler.chain and bP_sampler.lnlikes
my_sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=10)

my_sampler.results.save_results("hr8799e_gravity_chains.hdf5")

#import orbitize.results as results
#my_sampler.results = results.Results()
#my_sampler.results.load_results("hr8799e_gravity_chains.hdf5")

# make corner plot
fig = my_sampler.results.plot_corner()
plt.savefig('corner_hr8799e_gravity.png', dpi=250)

# print SMA, ecc, inc
labels = ["sma", "ecc", "inc"]
paper_vals = ["16.4 (+2.1/-1.1)", "0.15 +/- 0.08", "25 +/- 8"]
for i in range(len(labels)):
    med_val = np.median(my_sampler.results.post[:,i])
    ci_vals = np.percentile(my_sampler.results.post[:,i], [84, 16]) - med_val
    if labels[i] == 'inc':
        med_val = np.degrees(med_val)
        ci_vals = np.degrees(ci_vals)
    print("{0}: paper value is {1}".format(labels[i], paper_vals[i]))
    print("{3}: this fit obtained {0:.2f} (+{1:.2f}/-{2:.2f})".format(med_val, ci_vals[0], ci_vals[1], labels[i]))
