# End to end example with beta Pic astrometry
# Based on driver.py

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
n_temps=5
n_walkers=1000
n_threads=mp.cpu_count()
total_orbits=1000 # n_walkers x num_steps_per_walker
burn_steps=1000

# Read in data
data_table = read_input.read_formatted_file(datafile)

# Initialize System object which stores data & sets priors
bP_system = system.System(
    num_secondary_bodies, data_table, system_mass,
    plx, mass_err=mass_err, plx_err=plx_err
)

# We could overwrite any priors we want to here.
# Using defaults for now.
#bP_system.sys_priors[3] = priors.UniformPrior(0, np.pi/3)

# Initialize Sampler object, which stores information about
# the likelihood function & the algorithm used to generate
# orbits, and has System object as an attribute.
bP_sampler = sampler.PTMCMC(likelihood_func_name,bP_system,n_temps,n_walkers,n_threads)

import pdb; pdb.set_trace()
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
