import orbitize
from orbitize import read_input, system, priors, sampler
from orbitize.kepler import calc_orbit
import matplotlib.pyplot as plt
import numpy as np
import astropy.table
import time
from orbitize.read_input import read_file



def generate_synthetic_data(filepath, sma=30., num_obs=4, unc=10):
    """ Generate an orbitize-table of synethic data

    Args:
        sma (float): semimajor axis (au)
        num_obs (int): number of observations to generate
        unc (float): uncertainty on all simulated RA & Dec measurements (mas)

    Returns:
        2-tuple:
            - `astropy.table.Table`: data table of generated synthetic data
            - float: the orbit fraction of the generated data
    """

    # assumed ground truth for non-input orbital parameters
    ecc = 0. # eccentricity
    inc = np.pi/4 # inclination [rad]
    argp = 0.
    lan = 0.
    tau = 0.8

    # calculate RA/Dec at three observation epochs
    observation_epochs = np.linspace(51550., 52650., num_obs) # `num_obs` epochs between ~2000 and ~2003 [MJD]
    num_obs = len(observation_epochs)
    ra, dec, _ = calc_orbit(observation_epochs, sma, ecc, inc, argp, lan, tau, plx, mtot)

    # add Gaussian noise to simulate measurement
    ra += np.random.normal(scale=unc, size=num_obs)
    dec += np.random.normal(scale=unc, size=num_obs)

    # define observational uncertainties
    ra_err = dec_err = np.ones(num_obs)*unc

    # make a plot of the data
    plt.figure()
    plt.errorbar(ra, dec, xerr=ra_err, yerr=dec_err, linestyle='')
    plt.xlabel('$\\Delta$ RA'); plt.ylabel('$\\Delta$ Dec')
    plt.savefig(filepath)

    # calculate the orbital fraction
    period = np.sqrt((sma**3)/mtot)
    orbit_coverage = (max(observation_epochs) - min(observation_epochs))/365.25 # [yr]
    orbit_fraction = 100*orbit_coverage/period

    data_table = astropy.table.Table(
        [observation_epochs, [1]*num_obs, ra, ra_err, dec, dec_err],
        names=('epoch', 'object', 'raoff', 'raoff_err', 'decoff', 'decoff_err')
    )
    # read into orbitize format
    data_table = read_file(data_table)

    return data_table, orbit_fraction


# generate data
mtot = 1.2 # total system mass [M_sol]
plx = 60.0 # parallax [mas]
n_orbs = 500
sma = 2.3
data_table, orbit_fraction = generate_synthetic_data('/home/tmckenna/orbitize/fake_data_test.png', sma=sma, num_obs=30)
print('The orbit fraction is {}%'.format(np.round(orbit_fraction),1))

#plot orbit coverage with fake data
plt.errorbar(data_table['quant1'], data_table['quant2'], yerr = data_table['quant1_err'], xerr = data_table['quant2_err'])
plt.savefig('/home/tmckenna/orbitize/results/fake_orbit_data.png')

# initialize orbitize `System` object
sys = system.System(1, data_table, mtot, plx)
print(data_table)
lab = sys.param_idx

#set all parameters except eccentricity to fixed values for testing
sys.sys_priors[lab['inc1']] = np.pi/4
sys.sys_priors[lab['sma1']] = sma
sys.sys_priors[lab['aop1']] = 0.
sys.sys_priors[lab['pan1']] = 0.
sys.sys_priors[lab['tau1']] = 0.8
sys.sys_priors[lab['plx']] = plx
sys.sys_priors[lab['mtot']] = mtot


# run nested sampler
nested_sampler = sampler.NestedSampler(sys)
_ = nested_sampler.run_sampler(n_orbs, static = True, bound = 'multi')

# save results
nested_sampler.results.save_results('test1.hdf5')
plt.figure()
accepted_eccentricities = nested_sampler.results.post[:, lab['ecc1']]
plt.hist(accepted_eccentricities, bins=50)
plt.xlabel('ecc'); plt.ylabel('number of orbits')
plt.savefig('/home/tmckenna/orbitize/results/ecc_test1.png')

plt.figure()
accepted_inclinations = nested_sampler.results.post[:, lab['inc1']]
plt.hist(accepted_inclinations, bins=50)
plt.xlabel('inc'); plt.ylabel('number of orbits')
plt.savefig('/home/tmckenna/orbitize/results/inc_test1.png')