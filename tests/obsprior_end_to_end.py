import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time

import orbitize.read_input
import orbitize.sampler
import orbitize.system
from orbitize.priors import ObsPrior

n_temps = 20
n_walkers = 1000
n_threads = 20
num_steps_per_walker = 50000
total_orbits = n_walkers * num_steps_per_walker
burn_steps = 10000
thin = 10

data_table = orbitize.read_input.read_file('{}/GJ504.csv'.format(orbitize.DATADIR))
epochs = data_table['epoch']

# convert input sep/PA measurements to RA/decl
sep = np.array(data_table['quant1'])
sep_err = np.array(data_table['quant1_err'])
pa = np.radians(np.array(data_table['quant2']))
pa_err = np.radians(np.array(data_table['quant2_err']))

ra_err = np.sqrt(
    (np.cos(pa) * sep_err)**2 + 
    (sep * np.sin(pa) * pa_err)**2
)
dec_err = np.sqrt(
    (np.sin(pa) * sep_err)**2 + 
    (sep * np.cos(pa) * pa_err)**2
)

# number of secondary bodies in system
num_planets = 1

# total mass [msol]
mtot = 1.22

# parallax [mas]
plx = 56.95

sys = orbitize.system.System(
    num_planets, data_table, mtot, plx
)

# alias for convenience
lab = sys.param_idx

# place ObsPrior on sma, ecc, and tau
my_obsprior = ObsPrior(epochs, ra_err, dec_err, mtot, plx)
sys.sys_priors[lab['sma1']] = my_obsprior
sys.sys_priors[lab['ecc1']] = my_obsprior
sys.sys_priors[lab['tau1']] = my_obsprior

obsprior = sys.sys_priors[0]

# initialize & run MCMC
sampler = orbitize.sampler.MCMC(
    sys, num_temps=n_temps, num_walkers=n_walkers, num_threads=n_threads
)
sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
obj_results = sampler.results
obj_results.save_results('obsprior_test.hdf5')

# make corner plot
corner_fig = obj_results.plot_corner(['sma1','ecc1','aop1','tau1'], bins=50, show_titles=True, plot_datapoints=False, quantiles=[0.05,0.5,0.95])
plt.savefig('obsprior_corner.png', dpi=250)