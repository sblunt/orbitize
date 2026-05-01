import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import matplotlib.pyplot as plt

import orbitize

# Based on driver.py

import numpy as np
from orbitize import read_input, system, sampler, priors
import multiprocessing as mp
import orbitize
import time

# System parameters
datafile = "GJ504.csv"
num_secondary_bodies = 1
system_mass = 1.22  # Msol
plx = 56.95  # mas
mass_err = 0.08  # Msol
plx_err = 0.26  # mas

# Sampler parameters
likelihood_func_name = "chi2_lnlike"
n_threads = mp.cpu_count()
total_orbits = 10000000
prepared_samples = 10000

I = 1

results_file = f"GJ504_OFTI_{I}.hdf5"

tau_ref_epoch = 50000


# Read in data
data_table = read_input.read_file(orbitize.DATADIR + datafile)

# Initialize System object which stores data & sets priors
my_system = system.System(
    num_secondary_bodies,
    data_table,
    system_mass,
    plx,
    mass_err=mass_err,
    plx_err=plx_err,
    tau_ref_epoch=tau_ref_epoch,
)

my_sampler = sampler.OFTI(my_system, likelihood_func_name)

my_sampler.results.load_results(results_file)

start = time.time()

print(f"Running OFTI sampler with {n_threads} {total_orbits} {prepared_samples}")
# Run the sampler to compute some orbits, yeah!
my_sampler.run_sampler(total_orbits, prepared_samples, n_threads)

end = time.time()
print(f"Processing time: {end} - {start} = {(end-start) / 60} minutes")


my_sampler.results.save_results(results_file)

# import orbitize.results as results
# my_sampler.results = results.Results()
# my_sampler.results.load_results("hr8799e_gravity_chains.hdf5")

# make corner plot
# fig = my_sampler.results.plot_corner()
# plt.savefig("", dpi=250)

my_sampler.results.print_results()