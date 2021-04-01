import os
import matplotlib.pyplot as plt

import orbitize
from orbitize import system, read_input, priors, sampler

"""
Attempts to reproduce case 3 (see table 3) of Nielsen+ 2020 (orbit fits of beta Pic b).
"""

input_file = os.path.join(orbitize.DATADIR, 'betaPic.csv')
plx = 19.44

num_secondary_bodies = 1
data_table = read_input.read_file(input_file)

betaPic_system = system.System(
    num_secondary_bodies, data_table, 1, plx, hipparcos_number='027321',
    hipparcos_filename=os.path.join(orbitize.DATADIR, 'HIP027321.d'),
    fit_secondary_mass=True
)

# set uniform total mass prior
betaPic_system.sys_priors[-1] = priors.UniformPrior(1.5, 2.0)

betaPic_sampler = sampler.MCMC(betaPic_system, num_threads=1, num_temps=10, num_walkers=100)

betaPic_sampler.run_sampler(500, burn_steps=0)
ax_list = betaPic_sampler.examine_chains()

fig = betaPic_sampler.results.plot_corner()
plt.savefig('corner.png', dpi=250)