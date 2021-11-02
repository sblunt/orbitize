import numpy as np
import os
import matplotlib.pyplot as plt

import orbitize
from orbitize import system, read_input, priors, sampler
from orbitize.hipparcos import HipparcosLogProb
from orbitize.gaia import GaiaLogProb

"""
Attempts to reproduce case 3 (see table 3) of Nielsen+ 2020 (orbit fits of beta 
Pic b).

This is a publishable orbit fit that will take several hours-days to run. It
uses relative astrometry, Hipparcos intermediate astrometric data (IAD),
Gaia astrometric data, and a radial velocity measurement of beta Pic b.

Set these "keywords:"

- `fit_IAD` to True if you want to include the Hipparcos IAD. If False,
just fits the relative astrometry.
- `savedir` to where you want the fit outputs to be saved


Begin keywords <<
"""
fit_IAD = True 

if fit_IAD:
    savedir = '/data/user/{}/betaPic/hipIAD'.format(os.getlogin())
else:
    savedir = '/data/user/{}/betaPic/noIAD'.format(os.getlogin())
"""
>> End keywords
"""

if not os.path.exists(savedir):
    os.mkdir(savedir)

input_file = os.path.join(orbitize.DATADIR, 'betaPic.csv')
plx = 51.5

num_secondary_bodies = 1
data_table = read_input.read_file(input_file)

if fit_IAD:
    hipparcos_number='027321'
    gaia_edr3_number = 4792774797545800832
    gaia_dr2_number = 4792774797545105664
    fit_secondary_mass=True
    hipparcos_filename=os.path.join(orbitize.DATADIR, 'HIP027321.d')
    betaPic_Hip = HipparcosLogProb(
        hipparcos_filename, hipparcos_number, num_secondary_bodies
    )
    betaPic_gaia = GaiaLogProb(
        gaia_dr2_number, betaPic_Hip, dr='dr2'
    )
    # betaPic_gaia = GaiaLogProb(
    #     gaia_edr3_number, betaPic_Hip, dr='edr3'
    # )
else:
    fit_secondary_mass=False
    betaPic_Hip = None
    betaPic_gaia = None

betaPic_system = system.System(
    num_secondary_bodies, data_table, 1.75, plx, hipparcos_IAD=betaPic_Hip, 
    gaia=betaPic_gaia, fit_secondary_mass=fit_secondary_mass, mass_err=0.01, 
    plx_err=0.01
)

m0_or_mtot_prior = priors.UniformPrior(1.5, 2.0)

# set uniform parallax prior
plx_index = betaPic_system.param_idx['plx']
betaPic_system.sys_priors[plx_index] = priors.UniformPrior(plx - 1.0, plx + 1.0)

# set prior on Omega, since we know that know direction of orbital motion from RV
pan_index = betaPic_system.param_idx['pan1']
betaPic_system.sys_priors[pan_index] = priors.UniformPrior(0, np.pi)

# set uniform prior on m1 as Nielsen+ 2020 do
m1_index = betaPic_system.param_idx['m1']
betaPic_system.sys_priors[m1_index] = priors.UniformPrior(0, 0.1)

if fit_IAD:
    assert betaPic_system.fit_secondary_mass
    assert betaPic_system.track_planet_perturbs

    # set uniform m0 prior
    m0_index = betaPic_system.param_idx['m0']
    betaPic_system.sys_priors[m0_index] = m0_or_mtot_prior

else:
    assert not betaPic_system.fit_secondary_mass
    assert not betaPic_system.track_planet_perturbs

    # set uniform mtot prior
    mtot_index = betaPic_system.param_idx['mtot']    
    betaPic_system.sys_priors[mtot_index] = m0_or_mtot_prior

# run MCMC
num_threads = 100
num_temps = 20
num_walkers = 1000
num_steps = 1000000 # n_walkers x n_steps_per_walker
burn_steps = 1000
thin = 100

betaPic_sampler = sampler.MCMC(
    betaPic_system, num_threads=num_threads, num_temps=num_temps, 
    num_walkers=num_walkers
)
betaPic_sampler.run_sampler(num_steps, burn_steps=burn_steps, thin=thin)

# save chains
betaPic_sampler.results.save_results(
    '{}/betaPic_IAD{}.hdf5'.format(savedir, fit_IAD)
)

# make corner plot
fig = betaPic_sampler.results.plot_corner()
plt.savefig('{}/corner_IAD{}.png'.format(savedir, fit_IAD), dpi=250)

# make orbit plot
fig = betaPic_sampler.results.plot_orbits()
plt.savefig('{}/orbit_IAD{}.png'.format(savedir, fit_IAD), dpi=250)