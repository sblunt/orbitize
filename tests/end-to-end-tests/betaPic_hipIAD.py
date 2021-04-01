import os
import matplotlib.pyplot as plt

import orbitize
from orbitize import system, read_input, priors, sampler

"""
Attempts to reproduce case 3 (see table 3) of Nielsen+ 2020 (orbit fits of beta Pic b).
"""

fit_IAD = True

input_file = os.path.join(orbitize.DATADIR, 'betaPic.csv')
plx = 19.44

num_secondary_bodies = 1
data_table = read_input.read_file(input_file)

if fit_IAD:
    hipparcos_number='027321'
    fit_secondary_mass=True
    hipparcos_filename=os.path.join(orbitize.DATADIR, 'HIP027321.d')
else:
    hipparcos_number=None
    fit_secondary_mass=False
    hipparcos_filename=None

betaPic_system = system.System(
    num_secondary_bodies, data_table, 1, plx, hipparcos_number=hipparcos_number,
    hipparcos_filename=hipparcos_filename,
    fit_secondary_mass=fit_secondary_mass
)

# set uniform total mass prior
betaPic_system.sys_priors[-1] = priors.UniformPrior(1.5, 2.0)

# run MCMC
betaPic_sampler = sampler.MCMC(betaPic_system, num_threads=1, num_temps=10, num_walkers=100)
betaPic_sampler.run_sampler(500, burn_steps=0)

# save chains
betaPic_sampler.results.save_results('betaPic_IAD{}.hdf5'.format(fit_IAD))

# make corner plot
fig = betaPic_sampler.results.plot_corner()
plt.savefig('corner_IAD{}.png'.format(fit_IAD), dpi=250)