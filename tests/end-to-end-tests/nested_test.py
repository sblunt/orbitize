import orbitize
from orbitize import read_input, system, sampler, priors
import matplotlib.pyplot as plt
import os


savedir = "."

"""
Runs the GJ504 fit (from the quickstart tutorial) using dynesty as a backend

Written: Thea McKenna, 2023
"""

data_table = read_input.read_file("{}/GJ504.csv".format(orbitize.DATADIR))

# number of secondary bodies in system
num_planets = 1

# total mass & error [msol]
total_mass = 1.22
mass_err = 0  # 0.08

# parallax & error[mas]
plx = 56.95
plx_err = 0  # 0.26

sys = system.System(
    num_planets,
    data_table,
    total_mass,
    plx,
    mass_err=mass_err,
    plx_err=plx_err,
)
# alias for convenience
lab = sys.param_idx

# set prior on semimajor axis
sys.sys_priors[lab["sma1"]] = priors.LogUniformPrior(10, 400)

nested_sampler = sampler.NestedSampler(sys)


samples, num_iter = nested_sampler.run_sampler(num_threads=4, static=True, dlogz=200)
nested_sampler.results.save_results("{}/nested_sampler_test.hdf5".format(savedir))
# print("execution time (min) is: " + str(exec_time))
print("iteration number is: " + str(num_iter))

fig, ax = plt.subplots(2, 1)
accepted_eccentricities = nested_sampler.results.post[:, lab["ecc1"]]
accepted_inclinations = nested_sampler.results.post[:, lab["inc1"]]
ax[0].hist(accepted_eccentricities, bins=50)
ax[1].hist(accepted_inclinations, bins=50)
ax[0].set_xlabel("ecc")
ax[1].set_xlabel("inc")
plt.tight_layout()
plt.savefig("{}/nested_sampler_test.png".format(savedir))
