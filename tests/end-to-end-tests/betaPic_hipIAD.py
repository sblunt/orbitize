import os
import matplotlib.pyplot as plt

import orbitize
from orbitize import system, read_input, priors, sampler
from orbitize.hipparcos import HipparcosLogProb

"""
Attempts to reproduce case 3 (see table 3) of Nielsen+ 2020 (orbit fits of beta Pic b).
"""

fit_IAD = False
if fit_IAD:
    savedir = 'betaPic_hipIAD'
else:
    savedir = 'betaPic_noIAD'

if not os.path.exists(savedir):
    os.mkdir(savedir)

input_file = os.path.join(orbitize.DATADIR, 'betaPic.csv')
plx = 51.5

num_secondary_bodies = 1
data_table = read_input.read_file(input_file)

if fit_IAD:
    hipparcos_number='027321'
    fit_secondary_mass=True
    hipparcos_filename=os.path.join(orbitize.DATADIR, 'HIP027321.d')
    betaPic_Hip = HipparcosLogProb(hipparcos_filename, hipparcos_number, num_secondary_bodies)
else:
    fit_secondary_mass=False
    betaPic_Hip = None

betaPic_system = system.System(
    num_secondary_bodies, data_table, 1.75, plx, hipparcos_IAD=betaPic_Hip, 
    fit_secondary_mass=fit_secondary_mass, mass_err=0.01, plx_err=0.01
)

if fit_IAD:
    assert betaPic_system.fit_secondary_mass
    assert betaPic_system.track_planet_perturbs
else:
    assert not betaPic_system.fit_secondary_mass
    assert not betaPic_system.track_planet_perturbs

# set uniform m0 prior
betaPic_system.sys_priors[-1] = priors.UniformPrior(1.5, 2.0)

# set uniform parallax prior
betaPic_system.sys_priors[6] = priors.UniformPrior(plx - 1.0, plx + 1.0)

# run MCMC
num_threads = 50
num_temps = 20
num_walkers = 1000
num_steps = 100000 # 10000000 # n_walkers x n_steps_per_walker
burn_steps = 10000
thin = 100

betaPic_sampler = sampler.MCMC(betaPic_system, num_threads=num_threads, num_temps=num_temps, num_walkers=num_walkers)
betaPic_sampler.run_sampler(num_steps, burn_steps=burn_steps, thin=thin)

# save chains
betaPic_sampler.results.save_results('{}/betaPic_IAD{}.hdf5'.format(savedir, fit_IAD))

# make corner plot
fig = betaPic_sampler.results.plot_corner()
plt.savefig('{}/corner_IAD{}.png'.format(savedir, fit_IAD), dpi=250)