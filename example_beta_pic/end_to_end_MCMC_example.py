# End to end example with beta Pic astrometry
# Based on driver.py

from orbitize import read_input, system, sampler

# System parameters
datafile='betapic_astrometry.csv'
num_secondary_bodies=1
system_mass=1.75 # Msol
plx=51.44 #mas
mass_err=0.05 # Msol
plx_err=0.12 #mas

# Sampler parameters
likelihood_func_name='chi2_lnlike'
n_temps=1
n_walkers=20
n_threads=2
total_orbits=100 # n_walkers x num_steps_per_walker
burn_steps=1

# Read in data
data_table = read_input.read_formatted_file(datafile)

# Initialize System object which stores data & sets priors
bP_system = system.System(
    num_secondary_bodies, data_table, system_mass,
    plx, mass_err=mass_err, plx_err=plx_err
)

# We could overwrite any priors we want to here.
# Using defaults for now.

# Initialize Sampler object, which stores information about
# the likelihood function & the algorithm used to generate
# orbits, and has System object as an attribute.
bP_sampler = sampler.PTMCMC(likelihood_func_name,bP_system,n_temps,n_walkers,n_threads)

# Run the sampler to compute some orbits, yeah!
# Results stored in bP_sampler.chain and bP_sampler.lnlikes
bP_sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=1)

import pdb; pdb.set_trace()
