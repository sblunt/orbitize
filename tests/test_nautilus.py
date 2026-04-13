import pytest
from orbitize import sampler, system
import numpy as np
from orbitize.system import generate_synthetic_data

# This is top part of the test is a direct translation of test_nested_sampler.py 

def test_nautilus_general():
    # generate synthetic data
    mtot = 1.2
    plx = 60.0
    orbit_frac = 95
    data_table, sma = generate_synthetic_data(
        orbit_frac,
        mtot,
        plx,
        num_obs=30 
    )

 #assume eccentricity 

    ecc = 0.5

#initlialize the orbit

    mySys = system.System(1, data_table, mtot, plx)
    lab = mySys.param_idx

    #set all paremeters except eccentricity

    mySys.sys_priors[lab["inc1"]] = np.pi / 4
    mySys.sys_priors[lab["sma1"]] = sma
    mySys.sys_priors[lab["aop1"]] = np.pi / 4
    mySys.sys_priors[lab["pan1"]] = np.pi / 4
    mySys.sys_priors[lab["tau1"]] = 0.8
    mySys.sys_priors[lab["plx"]] = plx
    mySys.sys_priors[lab["mtot"]] = mtot
    
    my_sampler = sampler.NautilusSampler(mySys)
    _ = my_sampler.run_sampler(n_live=1000, n_update=False, verbose=True)
    print("Finished 1st Run!")

    nautilus_eccentricities = my_sampler.results.post[:, lab["ecc1"]]
    assert np.median(nautilus_eccentricities) == pytest.approx(ecc, abs=0.1)

if __name__ ==  "__main__":
    test_nautilus_general() 
    
