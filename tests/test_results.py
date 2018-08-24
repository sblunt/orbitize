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

def test_add_samples(results):
    """
    Tests add_samples() with some simulated posterior samples
    Returns results.Results object
    """
    # Simulate some sample draws, assign random likelihoods
    n_orbit_draws1 = 1000
    sim_post = simulate_orbit_sampling(n_orbit_draws1)
    sim_lnlike = np.random.uniform(size=n_orbit_draws1)
    # Test adding samples
    results.add_samples(sim_post,sim_lnlike)
    # Simulate some more sample draws
    n_orbit_draws2 = 2000
    sim_post = simulate_orbit_sampling(n_orbit_draws2)
    sim_lnlike = np.random.uniform(size=n_orbit_draws2)
    # Test adding more samples
    results.add_samples(sim_post,sim_lnlike)
    # Check shape of results.post
    expected_length = n_orbit_draws1 + n_orbit_draws2
    assert results.post.shape == (expected_length,8)
    assert results.lnlike.shape == (expected_length,)

    return results

def test_plot_corner(results):
    """
    Tests plot_corner() with plotting simulated posterior samples
    for all 8 parameters and for just four selected parameters
    """
    Figure1 = results.plot_corner()
    assert Figure1 is not None
    Figure2 = results.plot_corner(param_list=['sma1','ecc1','inc1','mtot'])
    assert Figure2 is not None
    return Figure1, Figure2

def test_plot_orbits(results):
    """
    Tests plot_orbits() with simulated posterior samples
    """
    Figure1 = results.plot_orbits(num_orbits_to_plot=1,square_plot=True,show_colorbar=True)
    assert Figure1 is not None
    Figure2 = results.plot_orbits(num_orbits_to_plot=1,square_plot=True,show_colorbar=False)
    assert Figure2 is not None
    Figure3 = results.plot_orbits(num_orbits_to_plot=1,square_plot=False,show_colorbar=True)
    assert Figure3 is not None
    Figure4 = results.plot_orbits(num_orbits_to_plot=1,square_plot=False,show_colorbar=False)
    assert Figure4 is not None
    return (Figure1, Figure2, Figure3, Figure4)

if __name__ == "__main__":
    test_results = test_init()
    test_results = test_add_samples(test_results)
    test_corner_fig1, test_corner_fig2 = test_plot_corner(test_results)
    test_orbit_figs = test_plot_orbits(test_results)
    # test_corner_fig1.savefig('test_corner1.png')
    # test_corner_fig2.savefig('test_corner2.png')
    # test_orbit_figs[0].savefig('test_orbit1.png')
    # test_orbit_figs[1].savefig('test_orbit2.png')
    # test_orbit_figs[2].savefig('test_orbit3.png')
    # test_orbit_figs[3].savefig('test_orbit4.png')
