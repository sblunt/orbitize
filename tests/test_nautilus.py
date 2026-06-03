import pytest
from orbitize import sampler, system, results
import numpy as np
from orbitize.system import generate_synthetic_data
import matplotlib.pyplot as plt
""" This is pytest for Nautilus_Sampler, it assumes values for all the priors except 

    eccentricity and trys to calculate the true eccentricity """

def test_nautilus_general(make_plot=False):
    # generate synthetic data
    mtot = 1.2
    plx = 60.0
    orbit_frac = 95
    ecc = 0.1
    inc = np.pi/4
    data_table, sma = generate_synthetic_data(
        orbit_frac,
        mtot,
        plx,
        num_obs=30,
        ecc = ecc,
        inc = inc
    )

    #initlialize the orbit

    mySys = system.System(1, data_table, mtot, plx)
    lab = mySys.param_idx

    #set all paremeters except eccentricity

    #mySys.sys_priors[lab["inc1"]] = np.pi / 4,
    mySys.sys_priors[lab["sma1"]] = sma 
    mySys.sys_priors[lab["aop1"]] = np.pi / 4
    mySys.sys_priors[lab["pan1"]] = np.pi / 4
    mySys.sys_priors[lab["tau1"]] = 0.8
    mySys.sys_priors[lab["plx"]] = plx
    mySys.sys_priors[lab["mtot"]] = mtot
        
    my_sampler = sampler.NautilusSampler(mySys)
    my_sampler.run_sampler(n_live=800, n_update=None, verbose=True)
    
    print("Finished 1st Run!")
    
    nautilus_eccentricities = my_sampler.results.post[:, lab["ecc1"]]
    assert np.mean(nautilus_eccentricities) == pytest.approx(0.1, abs=0.1)

    nautilus_inclination = my_sampler.results.post[:, lab["inc1"]]
    assert np.median(nautilus_inclination) == pytest.approx(inc, abs=0.1)

    if make_plot:
        myResults = my_sampler.results
        myResults.plot_corner(param_list=["ecc1","inc1"]).savefig('nautilus_test_3.png')
        print("Made A Plot")

if __name__ ==  "__main__":
    test_nautilus_general()


# TO DO
# fewer live points
# verbose = false 
# Test < 1min     
