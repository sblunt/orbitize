import orbitize
from orbitize import read_input, system, priors, sampler
import matplotlib.pyplot as plt


data_table = read_input.read_file('{}/GJ504.csv'.format(orbitize.DATADIR))
print(data_table)

# number of secondary bodies in system
num_planets = 1
# total mass & error [msol]
total_mass = 1.22
mass_err = 0.08
# parallax & error[mas]
plx = 56.95
plx_err = 0
sys = system.System(
    num_planets, data_table, total_mass,
    plx, mass_err=mass_err, plx_err=plx_err
)
# alias for convenience
lab = sys.param_idx

mu = 0.2
sigma = 0.05

sys.sys_priors[lab['ecc1']] = priors.GaussianPrior(mu, sigma)
sys.sys_priors[lab['inc1']] = 2.5
nested_sampler = sampler.NestedSampler(sys)

samples, exec_time, num_iter = nested_sampler.run_sampler(static = True, bound = 'multi')
nested_sampler.results.save_results('test34.hdf5')
print("execution time (min) is: " + str(exec_time))
print("iteration number is: " + str(num_iter))

plt.figure()
accepted_eccentricities = nested_sampler.results.post[:, lab['ecc1']]
plt.hist(accepted_eccentricities)
plt.xlabel('ecc'); plt.ylabel('number of orbits')
plt.savefig('../../results/ecc_test34.png')

plt.figure()
accepted_inclinations = nested_sampler.results.post[:, lab['inc1']]
plt.hist(accepted_inclinations)
plt.xlabel('inc'); plt.ylabel('number of orbits')
plt.savefig('../../results/inc_test34.png')