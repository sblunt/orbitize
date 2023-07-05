"""
Tests the NestedSampler class by fixing all parameters except for eccentricity.
"""

from orbitize import system, sampler
import numpy as np
import pytest
from orbitize.system import generate_synthetic_data


def test_nested_sampler():
    # generate data
    mtot = 1.2  # total system mass [M_sol]
    plx = 60.0  # parallax [mas]
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

    ecc = 0.5  # eccentricity

    # set all parameters except eccentricity to fixed values (same as used to generate data)
    sys.sys_priors[lab["inc1"]] = np.pi / 4
    sys.sys_priors[lab["sma1"]] = sma
    sys.sys_priors[lab["aop1"]] = np.pi / 4
    sys.sys_priors[lab["pan1"]] = np.pi / 4
    sys.sys_priors[lab["tau1"]] = 0.8
    sys.sys_priors[lab["plx"]] = plx
    sys.sys_priors[lab["mtot"]] = mtot

    # run both static & dynamic nested samplers
    dynamic_sampler = sampler.NestedSampler(sys)
    _ = dynamic_sampler.run_sampler(bound="multi", pfrac=0.95, static=False)

    dynamic_eccentricities = dynamic_sampler.results.post[:, lab["ecc1"]]
    assert np.median(dynamic_eccentricities) == pytest.approx(ecc, abs=0.1)

    static_sampler = sampler.NestedSampler(sys)
    _ = static_sampler.run_sampler(bound="multi")

    static_eccentricities = static_sampler.results.post[:, lab["ecc1"]]
    assert np.median(static_eccentricities) == pytest.approx(ecc, abs=0.1)

    # check that the static sampler raises an error when user tries to set pfrac
    try:
        static_sampler.run_sampler(pfrac=0.1)
    except ValueError:
        pass


if __name__ == "__main__":
    test_nested_sampler()
