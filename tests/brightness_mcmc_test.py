import numpy as np

import orbitize
from orbitize import driver
import multiprocessing as mp

filename = "{}/orbital_data_with_id.csv".format(orbitize.DATADIR)

# system parameters
num_secondary_bodies = 1
total_mass = 1.75  # [Msol]
plx = 51.44  # [mas]
mass_err = 0.05  # [Msol]
plx_err = 0.12  # [mas]

# MCMC parameters
num_temps = 20
num_walkers = 1000
num_threads = 20  # or a different number if you prefer, mp.cpu_count() for example


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
    }, system_kwargs={"restrict_angle_ranges": True},
)


if __name__ == '__main__':

    total_orbits = 100000000  # number of steps x number of walkers (at lowest temperature)
    burn_steps = 50000  # steps to burn in per walker
    thin = 10  # only save every 2nd step

    my_driver.sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)

    corner_plot_fig = (
    my_driver.sampler.results.plot_corner()
    )  # Creates a corner plot and returns Figure object
    corner_plot_fig.savefig(
    "my_corner_plot.png"
    )  # This is matplotlib.figure.Figure.savefig()

    epochs = my_driver.system.data_table["epoch"]

    orbit_plot_fig = my_driver.sampler.results.plot_orbits(
    object_to_plot=1,  # Plot orbits for the first (and only, in this case) companion
    num_orbits_to_plot=100,  # Will plot 100 randomly selected orbits of this companion
    start_mjd=epochs[0],  # Minimum MJD for colorbar (here we choose first data epoch)
    sep_pa_end_year=1870.0
    )
    orbit_plot_fig.savefig(
    "my_orbit_plot.png"
    )  # This is matplotlib.figure.Figure.savefig()

    sma_chains, ecc_chains = my_driver.sampler.examine_chains(
    param_list=["sma1", "ecc1"], n_walkers=5
    )

    from orbitize import results


    hdf5_filename = "my_posterior.hdf5"
    import os

# To avoid weird behaviours, delete saved file if it already exists from a previous run of this notebook
    if os.path.isfile(hdf5_filename):
        os.remove(hdf5_filename)

    my_driver.sampler.results.save_results(hdf5_filename)
    loaded_results = results.Results()  # Create blank results object for loading
    loaded_results.load_results(hdf5_filename)

