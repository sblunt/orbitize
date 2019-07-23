import numpy as np
import sampler
import driver
import kepler
import system
import multiprocessing as mp

# MCMC parameters
num_temps = 5
num_walkers = 30
num_threads = mp.cpu_count()  # or a different number if you prefer

myDriver = driver.Driver('/Users/Helios/orbitize/tests/testdata0.csv',  # path to data file
                         sampler_str='MCMC',  # name of algorithm for orbit-fitting
                         num_secondary_bodies=1,  # number of secondary bodies in system
                         system_mass=3.0,  # total system mass [M_sun]
                         plx=61,  # total parallax of system [mas]
                         mass_err=0.5,  # mass error [M_sun]
                         plx_err=1.0,  # parallax error [mas]
                         system_kwargs={'fit_secondary_mass': True,
                                        'tau_ref_epoch': 0, 'gamma_bounds': (-100, 100), 'jitter_bounds': (1e-3, 20)},
                                  mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers,
                                               'num_threads': num_threads})

total_orbits = 1000  # number of steps x number of walkers (at lowest temperature)
burn_steps = 5  # steps to burn in per walker
thin = 21  # only save every 2nd step

myDriver.sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)

# Creates a corner plot and returns Figure object
corner_plot_fig = myDriver.sampler.results.plot_corner()
# This is matplotlib.figure.Figure.savefig()
corner_plot_fig.savefig(
    '/Users/Helios/Desktop/jupyter_notebooks/orbitize_radvel_stanford/testdata0_cornerplot.png')

epochs = myDriver.system.data_table['epoch']

orbit_plot_fig = myDriver.sampler.results.plot_orbits(
    object_to_plot=1,  # Plot orbits for the first (and only, in this case) companion
    num_orbits_to_plot=100,  # Will plot 100 randomly selected orbits of this companion
    start_mjd=epochs[0]  # Minimum MJD for colorbar (here we choose first data epoch)
)
# This is matplotlib.figure.Figure.savefig()
orbit_plot_fig.savefig(
    '/Users/Helios/Desktop/jupyter_notebooks/orbitize_radvel_stanford/testdata0_orbitalplot.png')
