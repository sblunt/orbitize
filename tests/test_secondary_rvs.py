import numpy as np
import os
from astropy.time import Time
from pandas import DataFrame

from orbitize.kepler import calc_orbit
from orbitize import read_input, system, sampler

def test_secondary_rv_lnlike_calc():
    """
    Generates fake secondary RV data and asserts that
    the log(likelihood) of the true parameters is what we expect.
    Also tests that the primary and secondary RV orbits are related by
    -m/mtot
    """

    # define an orbit & generate secondary RVs
    a = 10
    e = 0
    i = np.pi / 4
    omega  = 0
    Omega = 0
    tau = 0.3
    m0 = 1
    m1 = 0.1
    plx = 10
    orbitize_params_list = np.array([a, e, i, omega, Omega, tau, plx, m1, m0])

    epochs = Time(np.linspace(2005, 2025, int(1e3)), format='decimalyear').mjd

    _, _, rv_p = calc_orbit(epochs, a, e, i, omega, Omega, tau, plx, m0+m1, mass_for_Kamp=m0)

    data_file = DataFrame(columns=['epoch', 'object','rv', 'rv_err'])
    data_file.epoch = epochs
    data_file.object = np.ones(len(epochs), dtype=int)
    data_file.rv = rv_p
    data_file.rv_err = np.ones(len(epochs)) * 0.01

    data_file.to_csv('tmp.csv', index=False)

    # set up a fit using the simulated data
    data_table = read_input.read_file('tmp.csv')
    mySys = system.System(1, data_table, m0, plx, mass_err=0.1, plx_err=0.1, fit_secondary_mass=True)
    mySamp = sampler.MCMC(mySys)
    computed_lnlike = mySamp._logl(orbitize_params_list)

    # residuals should be 0
    assert computed_lnlike == np.sum(-np.log(np.sqrt(2 * np.pi * data_file.rv_err.values**2)))

    # clean up
    os.system('rm tmp.csv')

    # assert that the secondary orbit is the primary orbit scaled
    _, _, rv = mySys.compute_all_orbits(orbitize_params_list)
    rv0 = rv[:,0]
    rv1 = rv[:,1]

    assert np.all(rv0 == -m1 / m0 * rv1)

if __name__ == '__main__':
    test_secondary_rv_lnlike_calc()
