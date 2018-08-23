"""
Test the routines in the orbitize.Results module
"""
# Based on driver.py

from orbitize import results
import numpy as np
import matplotlib.pyplot as plt

def test_init():
    """
    Tests results.Results initialization.
    Returns results.Results object
    """
    return results.Results(sampler_name='testing')

def simulate_orbit_sampling(n_sim_orbits):
    """
    Returns posterior array with n_sim_orbit samples for testing
    """
    # Parameters are based on beta Pic b from Wang+ 2016
    # Orbit parameters
    n_params = 8
    sma = 9.660
    sma_err = 1.1  # not real
    ecc = 0.08
    ecc_err = 0.03 # not real
    inc = 88.81
    inc_err = 0.12
    aop = 205.8
    aop_err = 20.0 # not real
    pan = 31.76
    pan_err = 0.09
    epp = 0.73
    epp_err = 0.20 # not real
    system_mass = 1.80
    system_mass_err = 0.04
    plx=51.44 #mas
    plx_err=0.12 #mas
    # Create some simulated orbit draws
    sim_post = np.zeros((n_sim_orbits,n_params))
    sim_post[:,0]=np.random.normal(sma,sma_err,n_sim_orbits)
    sim_post[:,1]=np.random.normal(ecc,ecc_err,n_sim_orbits)
    sim_post[:,2]=np.random.normal(inc,inc_err,n_sim_orbits)
    sim_post[:,3]=np.random.normal(aop,aop_err,n_sim_orbits)
    sim_post[:,4]=np.random.normal(pan,pan_err,n_sim_orbits)
    sim_post[:,5]=np.random.normal(epp,epp_err,n_sim_orbits)
    sim_post[:,6]=np.random.normal(system_mass,system_mass_err,n_sim_orbits)
    sim_post[:,7]=np.random.normal(plx,plx_err,n_sim_orbits)

    return sim_post

def test_add_orbits(results):
    """
    Tests add_orbits() with some simulated posterior samples
    Returns results.Results object
    """
    # Simulate some orbit draws, assign random likelihoods
    n_orbit_draws1 = 1000
    sim_post = simulate_orbit_sampling(n_orbit_draws1)
    sim_lnlike = np.random.uniform(size=n_orbit_draws1)
    # Test adding orbits
    results.add_orbits(sim_post,sim_lnlike)
    # Simulate some more orbit draws
    n_orbit_draws2 = 2000
    sim_post = simulate_orbit_sampling(n_orbit_draws2)
    sim_lnlike = np.random.uniform(size=n_orbit_draws2)
    # Test adding more orbits
    results.add_orbits(sim_post,sim_lnlike)
    # Check shape of results.post
    expected_length = n_orbit_draws1 + n_orbit_draws2
    assert results.post.shape == (expected_length,8)
    assert results.lnlike.shape == (expected_length,)

    return results

def test_plot_corner(results):
    Figure = results.plot_corner()
    return Figure


if __name__ == "__main__":
    test_results = test_init()
    test_results = test_add_orbits(test_results)
    test_corner_fig = test_plot_corner(test_results)
