"""
Test the routines in the orbitize.Results module
"""
# Based on driver.py

from orbitize import results
import numpy as np
import matplotlib.pyplot as plt
import pytest
import os

std_labels = ['sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'plx', 'mtot']


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
    ecc_err = 0.03  # not real
    inc = np.radians(88.81)
    inc_err = np.radians(0.12)
    aop = np.radians(205.8)
    aop_err = np.radians(20.0)  # not real
    pan = np.radians(31.76)
    pan_err = np.radians(0.09)
    epp = 0.73
    epp_err = 0.20  # not real
    system_mass = 1.80
    system_mass_err = 0.04
    plx = 51.44  # mas
    plx_err = 0.12  # mas
    # Create some simulated orbit draws
    sim_post = np.zeros((n_sim_orbits, n_params))
    sim_post[:, 0] = np.random.normal(sma, sma_err, n_sim_orbits)
    sim_post[:, 1] = np.random.normal(ecc, ecc_err, n_sim_orbits)
    sim_post[:, 2] = np.random.normal(inc, inc_err, n_sim_orbits)
    sim_post[:, 3] = np.random.normal(aop, aop_err, n_sim_orbits)
    sim_post[:, 4] = np.random.normal(pan, pan_err, n_sim_orbits)
    sim_post[:, 5] = np.random.normal(epp, epp_err, n_sim_orbits)
    sim_post[:, 6] = np.random.normal(plx, plx_err, n_sim_orbits)
    sim_post[:, 7] = np.random.normal(system_mass, system_mass_err, n_sim_orbits)

    return sim_post


def test_init_and_add_samples():
    """
    Tests object creation and add_samples() with some simulated posterior samples
    Returns results.Results object
    """
    # Create object
    results_obj = results.Results(
        sampler_name='testing', tau_ref_epoch=50000,
        labels=std_labels, num_secondary_bodies=1
    )
    # Simulate some sample draws, assign random likelihoods
    n_orbit_draws1 = 1000
    sim_post = simulate_orbit_sampling(n_orbit_draws1)
    sim_lnlike = np.random.uniform(size=n_orbit_draws1)
    # Test adding samples
    results_obj.add_samples(sim_post, sim_lnlike, labels=std_labels)
    # Simulate some more sample draws
    n_orbit_draws2 = 2000
    sim_post = simulate_orbit_sampling(n_orbit_draws2)
    sim_lnlike = np.random.uniform(size=n_orbit_draws2)
    # Test adding more samples
    results_obj.add_samples(sim_post, sim_lnlike, labels=std_labels)
    # Check shape of results.post
    expected_length = n_orbit_draws1 + n_orbit_draws2
    assert results_obj.post.shape == (expected_length, 8)
    assert results_obj.lnlike.shape == (expected_length,)
    assert results_obj.tau_ref_epoch == 50000
    assert results_obj.labels == std_labels

    return results_obj


@pytest.fixture()
def results_to_test():
    results_obj = results.Results(
        sampler_name='testing', tau_ref_epoch=50000,
        labels=std_labels, num_secondary_bodies=1
    )
    # Simulate some sample draws, assign random likelihoods
    n_orbit_draws1 = 1000
    sim_post = simulate_orbit_sampling(n_orbit_draws1)
    sim_lnlike = np.random.uniform(size=n_orbit_draws1)
    # Test adding samples
    results_obj.add_samples(sim_post, sim_lnlike, labels=std_labels)
    # Simulate some more sample draws
    n_orbit_draws2 = 2000
    sim_post = simulate_orbit_sampling(n_orbit_draws2)
    sim_lnlike = np.random.uniform(size=n_orbit_draws2)
    # Test adding more samples
    results_obj.add_samples(sim_post, sim_lnlike, labels=std_labels)
    # Return object for testing
    return results_obj


def test_save_and_load_results(results_to_test, has_lnlike=True):
    """
    Tests saving and reloading of a results object
        has_lnlike: allows for tests with and without lnlike values
            (e.g. OFTI doesn't output lnlike)
    """
    results_to_save = results_to_test
    if not has_lnlike:  # manipulate object to remove lnlike (as in OFTI)
        results_to_save.lnlike = None
    save_filename = 'test_results.h5'
    # Save to file
    results_to_save.save_results(save_filename)
    # Create new blank results object and load from file
    loaded_results = results.Results()
    loaded_results.load_results(save_filename, append=False)
    # Check if loaded results equal saved results
    assert results_to_save.sampler_name == loaded_results.sampler_name
    assert np.array_equal(results_to_save.post, loaded_results.post)
    if has_lnlike:
        assert np.array_equal(results_to_save.lnlike, loaded_results.lnlike)
    # Try to load the saved results again, this time appending
    loaded_results.load_results(save_filename, append=True)
    # Now check that the loaded results object has the expected size
    original_length = results_to_save.post.shape[0]
    expected_length = original_length * 2
    assert loaded_results.post.shape == (expected_length, 8)
    assert loaded_results.labels.tolist() == std_labels
    if has_lnlike:
        assert loaded_results.lnlike.shape == (expected_length,)

    # check tau reference epoch is stored
    assert loaded_results.tau_ref_epoch == 50000

    # Clean up: Remove save file
    os.remove(save_filename)


def test_plot_corner(results_to_test):
    """
    Tests plot_corner() with plotting simulated posterior samples
    for all 8 parameters and for just four selected parameters
    """
    Figure1 = results_to_test.plot_corner()
    assert Figure1 is not None
    Figure2 = results_to_test.plot_corner(param_list=['sma1', 'ecc1', 'inc1', 'mtot'])
    assert Figure2 is not None
    return Figure1, Figure2


def test_plot_orbits(results_to_test):
    """
    Tests plot_orbits() with simulated posterior samples
    """
    Figure1 = results_to_test.plot_orbits(
        num_orbits_to_plot=1, square_plot=True, show_colorbar=True)
    assert Figure1 is not None
    Figure2 = results_to_test.plot_orbits(
        num_orbits_to_plot=1, square_plot=True, show_colorbar=False)
    assert Figure2 is not None
    Figure3 = results_to_test.plot_orbits(
        num_orbits_to_plot=1, square_plot=False, show_colorbar=True)
    assert Figure3 is not None
    Figure4 = results_to_test.plot_orbits(
        num_orbits_to_plot=1, square_plot=False, show_colorbar=False)
    assert Figure4 is not None
    Figure5 = results_to_test.plot_orbits(num_orbits_to_plot=1, square_plot=False, cbar_param='ecc')
    assert Figure5 is not None
    return (Figure1, Figure2, Figure3, Figure4, Figure5)


if __name__ == "__main__":
    test_results = test_init_and_add_samples()
    test_save_and_load_results(test_results, has_lnlike=True)
    test_save_and_load_results(test_results, has_lnlike=True)
    test_save_and_load_results(test_results, has_lnlike=False)
    test_save_and_load_results(test_results, has_lnlike=False)
    test_corner_fig1, test_corner_fig2 = test_plot_corner(test_results)
    test_orbit_figs = test_plot_orbits(test_results)
    test_corner_fig1.savefig('test_corner1.png')
    test_corner_fig2.savefig('test_corner2.png')
    test_orbit_figs[0].savefig('test_orbit1.png')
    test_orbit_figs[1].savefig('test_orbit2.png')
    test_orbit_figs[2].savefig('test_orbit3.png')
    test_orbit_figs[3].savefig('test_orbit4.png')
    test_orbit_figs[4].savefig('test_orbit5.png')
