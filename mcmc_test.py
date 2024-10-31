import numpy as np

import orbitize
from orbitize import driver
import multiprocessing as mp

filename = "{}/simulated_ra_dec_data.csv".format(orbitize.DATADIR)

# system parameters
num_secondary_bodies = 1
total_mass = 1.75  # [Msol]
plx = 51.44  # [mas]
mass_err = 0.05  # [Msol]
plx_err = 0.12  # [mas]

# MCMC parameters
num_temps = 5
num_walkers = 20
num_threads = 2  # or a different number if you prefer, mp.cpu_count() for example


my_driver = driver.Driver(
    filename,
    "MCMC",
    num_secondary_bodies,
    total_mass,
    plx,
    mass_err=mass_err,
    plx_err=plx_err,
    mcmc_kwargs={
        "num_temps": num_temps,
        "num_walkers": num_walkers,
        "num_threads": num_threads,
    },
)
if __name__ == '__main__':

    total_orbits = 6000  # number of steps x number of walkers (at lowest temperature)
    burn_steps = 10  # steps to burn in per walker
    thin = 2  # only save every 2nd step

    my_driver.sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)


if my_driver.sampler.results is not None:
    print("Sampler results are available.")
    print("Number of orbits:", len(my_driver.sampler.results.post))
else:
    print("No sampler results available. Sampler may not have run correctly.")


