"""
Tests the NestedSampler and MultiNest classes by fixing all parameters except for eccentricity.
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
    mysampler = sampler.NestedSampler(sys)
    _ = mysampler.run_sampler(bound="multi", pfrac=0.95, static=False, num_threads=8, run_nested_kwargs={})
    print("Finished first run!")

    dynamic_eccentricities = mysampler.results.post[:, lab["ecc1"]]
    assert np.median(dynamic_eccentricities) == pytest.approx(ecc, abs=0.1)

    _ = mysampler.run_sampler(bound="multi", static=True, num_threads=8, run_nested_kwargs={})
    print("Finished second run!")

    static_eccentricities = mysampler.results.post[:, lab["ecc1"]]
    assert np.median(static_eccentricities) == pytest.approx(ecc, abs=0.1)


def test_multinest():
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

    # running the actual sampler is not possible without compiling MultiNest
    mysampler = sampler.MultiNest(sys)
    assert hasattr(mysampler, "run_sampler")
    assert hasattr(mysampler, "results")
    assert hasattr(mysampler, "system")


if __name__ == "__main__":
    test_nested_sampler()
    test_multinest()
