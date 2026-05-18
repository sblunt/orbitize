"""
Tests the NestedSampler class by fixing all parameters except for eccentricity.
"""

from orbitize import system, sampler
import numpy as np
import pytest
from orbitize.system import generate_synthetic_data
import sys


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
    mySys = system.System(1, data_table, mtot, plx)
    lab = mySys.param_idx

    ecc = 0.5  # eccentricity

    # set all parameters except eccentricity to fixed values (same as used to generate data)
    mySys.sys_priors[lab["inc1"]] = np.pi / 4
    mySys.sys_priors[lab["sma1"]] = sma
    mySys.sys_priors[lab["aop1"]] = np.pi / 4
    mySys.sys_priors[lab["pan1"]] = np.pi / 4
    mySys.sys_priors[lab["tau1"]] = 0.8
    mySys.sys_priors[lab["plx"]] = plx
    mySys.sys_priors[lab["mtot"]] = mtot

    start_method="fork"
    if sys.platform == "darwin":
        start_method="spawn"

    # run both static & dynamic nested samplers
    mysampler = sampler.NestedSampler(mySys)
    _ = mysampler.run_sampler(bound="multi", pfrac=0.95, static=False, start_method=start_method, num_threads=8)
    print("Finished first run!")

    dynamic_eccentricities = mysampler.results.post[:, lab["ecc1"]]
    assert np.median(dynamic_eccentricities) == pytest.approx(ecc, abs=0.1)


    _ = mysampler.run_sampler(bound="multi", static=True, start_method=start_method, num_threads=8)
    print("Finished second run!")

    static_eccentricities = mysampler.results.post[:, lab["ecc1"]]
    assert np.median(static_eccentricities) == pytest.approx(ecc, abs=0.1)

    # check that the static sampler raises an error when user tries to set pfrac
    # for static sampler
    try:
        mysampler.run_sampler(pfrac=0.1, static=True)
    except ValueError:
        pass


if __name__ == "__main__":
    test_nested_sampler()
