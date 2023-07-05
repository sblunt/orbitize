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
    mtot = 1.2  # total system mass [M_sol]
    plx = 60.0  # parallax [mas]
    n_orbs = 500
    orbit_frac = 95
    data_table, sma = generate_synthetic_data(
        orbit_frac,
        mtot,
        plx,
        num_obs=30,
    )

    # assumed ecc value
    ecc = 0.5

    # initialize orbitize `System` object
    sys = system.System(1, data_table, mtot, plx)
    lab = sys.param_idx

    # set all parameters except eccentricity to fixed values for testing
    sys.sys_priors[lab["inc1"]] = np.pi / 4
    sys.sys_priors[lab["sma1"]] = sma
    sys.sys_priors[lab["aop1"]] = 0.0
    sys.sys_priors[lab["pan1"]] = 0.0
    sys.sys_priors[lab["tau1"]] = 0.8
    sys.sys_priors[lab["plx"]] = plx
    sys.sys_priors[lab["mtot"]] = mtot

    # run both static & dynamic nested samplers
    static_sampler = sampler.NestedSampler(sys)
    _ = static_sampler.run_sampler(n_orbs, bound="multi")

    dynamic_sampler = sampler.NestedSampler(sys)
    _ = static_sampler.run_sampler(n_orbs, bound="multi", pfrac=0.5, static=False)

    static_eccentricities = static_sampler.results.post[:, lab["ecc1"]]
    dynamic_eccentricities = dynamic_sampler.results.post[:, lab["ecc1"]]

    assert static_eccentricities == pytest.approx(ecc, abs=0.1)
    assert dynamic_eccentricities == pytest.approx(ecc, abs=0.1)


if __name__ == "__main__":
    test_nested_sampler()
