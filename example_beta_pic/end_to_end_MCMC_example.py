# End to end example with beta Pic astrometry
# Based on driver.py

import numpy as np
from orbitize import read_input, system, sampler, priors
import multiprocessing as mp

# System parameters
datafile='betapic_astrometry.csv'
num_secondary_bodies=1
system_mass=1.75 # Msol
plx=51.44 #mas
mass_err=0.05 # Msol
plx_err=0.12 #mas

# Sampler parameters
likelihood_func_name='chi2_lnlike'
n_temps=20
n_walkers=1000
n_threads=mp.cpu_count()
total_orbits=10000 # n_walkers x num_steps_per_walker
burn_steps=1

# Read in data
data_table = read_input.read_formatted_file(datafile)
# convert to mas
data_table['quant1'] *= 1000
data_table['quant1_err'] *= 1000
data_table['quant2'] *= 1000
data_table['quant2_err'] *= 1000

data_table['epoch'] -= 50000

# Initialize System object which stores data & sets priors
bP_system = system.System(
    num_secondary_bodies, data_table, system_mass,
    plx, mass_err=mass_err, plx_err=plx_err
)

# We could overwrite any priors we want to here.
# Using defaults for now.
bP_system.sys_priors[3] = priors.UniformPrior(np.pi/10, np.pi/2)

# Initialize Sampler object, which stores information about
# the likelihood function & the algorithm used to generate
# orbits, and has System object as an attribute.
bP_sampler = sampler.PTMCMC(likelihood_func_name,bP_system,n_temps,n_walkers,n_threads)

# Run the sampler to compute some orbits, yeah!
# Results stored in bP_sampler.chain and bP_sampler.lnlikes
bP_sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=10)

import h5py
import ctypes
fin = h5py.File("/indirect/big_scr6/jwang/demo.hdf5", "w")
chain = fin.create_dataset("chain", bP_sampler.chain.shape, dtype=ctypes.c_float)
chain[...] = bP_sampler.chain
fin.close()


import pdb; pdb.set_trace()
