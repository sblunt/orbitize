import matplotlib.pyplot as plt

import orbitize

# Based on driver.py

import numpy as np
from orbitize import read_input, system, sampler, priors
import multiprocessing as mp
import orbitize
import time

savedir = ""

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
n_live = 2000
n_update = None
sampler_args = {"n_networks": 4}
run_args = {"f_live": 0.01, "n_eff": 10000}

naut_file = f"{savedir}GJ504_naut.hdf5"

results_file = f"{savedir}GJ504_results.hdf5"

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

my_sampler = sampler.NautilusSampler(my_system, like=likelihood_func_name)

print(f"Running {datafile} sampler with {n_threads} {n_live} {n_update} {sampler_args} {run_args}")
# Run the sampler to compute some orbits, yeah!
my_sampler.run_sampler(n_live,
                       n_update,
                       verbose=False,
                       num_threads=n_threads,
                       savefile=naut_file,
                       sampler_kwargs=sampler_args,
                       run_kwargs=run_args)

end = time.time()
print(f"Processing time: {end} - {my_sampler.start} = {(end-my_sampler.start) / 60} minutes")


my_sampler.results.save_results(results_file)

# make corner plot
fig = my_sampler.results.plot_corner(downsample=10000)
plt.savefig(f"{savedir}GJ504_naut_corner.png", dpi=250)

my_sampler.results.print_results()