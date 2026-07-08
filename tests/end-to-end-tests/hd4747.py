"""
Stellar RVs + astrometry, using data for HD 4747.
"""
import matplotlib.pyplot as plt
from orbitize import DATADIR, driver, plot, results
import numpy as np
from astropy import units as u, constants as cst

# Initialize Driver to Run MCMC
filename = "{}/HD4747.csv".format(DATADIR)

num_secondary_bodies = 1
system_mass = 0.84  # [Msol]
plx = 53.18  # [mas]
mass_err = 0.04  # [Msol]
plx_err = 0.12  # [mas]

num_temps = 5
num_walkers = 50
num_threads = 1  # or a different number if you prefer

my_driver = driver.Driver(
    filename,
    "MCMC",
    num_secondary_bodies,
    system_mass,
    plx,
    mass_err=mass_err,
    plx_err=plx_err,
    system_kwargs={"fit_secondary_mass": True, "tau_ref_epoch": 0},
    mcmc_kwargs={
        "num_temps": num_temps,
        "num_walkers": num_walkers,
        "num_threads": num_threads,
    },
)

# values from Xuan et al 2022
sma_cen = 10.0
sma_sig = 0.2
ecc_cen = .7317
ecc_sig = .0014
inc_cen = np.radians(48)
inc_sig = np.radians(1.1)
aop_cen = np.radians(267.2)
aop_sig = np.radians(0.5)
pan_cen = np.radians(89.4)
pan_sig = np.radians(1.1)
tau_cen = ((2462615 - my_driver.system.tau_ref_epoch) / (33.2 * 365.25)) % 1
tau_sig = 0.01

assert my_driver.system.param_idx['sma1'] == 0
assert my_driver.system.param_idx['ecc1'] == 1
assert my_driver.system.param_idx['inc1'] == 2
assert my_driver.system.param_idx['aop1'] == 3
assert my_driver.system.param_idx['pan1'] == 4
assert my_driver.system.param_idx['tau1'] == 5

walker_centres = np.array(
    [sma_cen, ecc_cen, inc_cen, aop_cen, pan_cen, tau_cen]
)
walker_1sigmas = np.array(
    [sma_sig, ecc_sig, inc_sig, aop_sig, pan_sig, tau_sig]
)

new_pos = np.random.standard_normal((num_temps, num_walkers, 6)) * walker_1sigmas + walker_centres

my_driver.sampler.curr_pos[:,:,:6] = np.copy(new_pos)
my_driver.sampler.curr_pos[:,:,my_driver.system.param_idx['m1']] = np.random.normal(loc=(67.2*u.M_jup/u.M_sun).to('').value, scale=0.01, size=(num_temps, num_walkers))

my_driver.sampler.check_prior_support()

assert len(my_driver.system.rv[0]) == 56

total_orbits = 10_00_000
burn_steps = 10_000
thin = 10

if __name__ == '__main__':

    # Run Sampler
    # m = my_driver.sampler
    # m.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
    # my_driver.sampler.results.save_results('hd4747_orbits.hdf5')

    my_results = results.Results()
    my_results.load_results('hd4747_orbits.hdf5')

    # plot.plot_corner(my_results)
    # plt.savefig('hd4747_corner.png')

    plot.plot_orbits(my_results, rv_time_series=True, start_mjd=51044.0)
    plt.savefig('hd4747_orbit.png')


