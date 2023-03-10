"""
Tests the NestedSampler class by fixing all parameters except for eccentricity.
"""

import orbitize
from orbitize import read_input, system, priors, sampler
from orbitize.kepler import calc_orbit
import numpy as np
import astropy.table
import pytest
import time
from orbitize.read_input import read_file
from orbitize.system import generate_synthetic_data


def test_nested_sampler():
    # generate data
    mtot = 1.2 # total system mass [M_sol]
    plx = 60.0 # parallax [mas]
    n_orbs = 500
    sma = 2.3
    data_table, orbit_fraction = generate_synthetic_data(mtot, plx, sma=sma, 
    num_obs=30)
    print('The orbit fraction is {}%'.format(np.round(orbit_fraction),1))

    # assumed ecc and uncertainty values
    ecc = 0.5
    unc = 0.1

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
    _ = nested_sampler.run_sampler(n_orbs, bound = 'multi')

    # save results
    nested_sampler.results.save_results('test1.hdf5')
    accepted_eccentricities = nested_sampler.results.post[:, lab['ecc1']]

    assert accepted_eccentricities == pytest.approx(ecc, abs=unc)





#TO DO: add in a test that asserts that the recovered posterior is reasonable

if __name__ == "__main__":
    test_nested_sampler()
    print('Finished!')