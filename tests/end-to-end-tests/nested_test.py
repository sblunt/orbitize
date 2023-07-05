import orbitize
from orbitize import read_input, system, sampler
import matplotlib.pyplot as plt
import os


savedir = "/data/user/{}/nested_sampling_test".format(os.getlogin())

"""
Runs the GJ504 fit (from the quickstart tutorial) using dynesty as a backend

Written: Thea McKenna, 2023
"""

data_table = read_input.read_file("{}/GJ504.csv".format(orbitize.DATADIR))

# number of secondary bodies in system
num_planets = 1
# total mass & error [msol]
total_mass = 1.22
mass_err = 0.08
# parallax & error[mas]
plx = 56.95
plx_err = 0
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

nested_sampler = sampler.NestedSampler(sys)

samples, exec_time, num_iter = nested_sampler.run_sampler(
    static=True, bound="multi", num_threads=8
)
nested_sampler.results.save_results("{}/nested_sampler_test.hdf5".format(savedir))
print("execution time (min) is: " + str(exec_time))
print("iteration number is: " + str(num_iter))

fig, ax = plt.subplots(2, 1)
accepted_eccentricities = nested_sampler.results.post[:, lab["ecc1"]]
accepted_inclinations = nested_sampler.results.post[:, lab["inc1"]]
ax[0].hist(accepted_eccentricities)
ax[1].hist(accepted_inclinations)
ax[0].set_xlabel("ecc")
ax[1].set_xlabel("inc")
plt.savefig("{}/nested_sampler_test.png".format(savedir))
