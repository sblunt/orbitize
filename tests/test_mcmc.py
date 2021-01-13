import pytest

import os
import orbitize
from orbitize.driver import Driver
import orbitize.sampler as sampler
import orbitize.system as system
import orbitize.read_input as read_input
import matplotlib.pyplot as plt


def test_mcmc_runs(num_temps=0, num_threads=1):
    """
    Tests the MCMC sampler by making sure it even runs
    Args:
        num_temps: Number of temperatures to use
            Uses Parallel Tempering MCMC (ptemcee) if > 1,
            otherwises, uses Affine-Invariant Ensemble Sampler (emcee)
        num_threads: number of threads to run
    """

    # use the test_csv dir
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
    data_table = read_input.read_formatted_file(input_file)
    # Manually set 'object' column of data table
    data_table['object'] = 1

    # construct Driver
    n_walkers = 100
    myDriver = Driver(input_file, 'MCMC', 1, 1, 0.01,
                      mcmc_kwargs={'num_temps': num_temps, 'num_threads': num_threads,
                                   'num_walkers': n_walkers}
                      )

    # run it a little (tests 0 burn-in steps)
    myDriver.sampler.run_sampler(100)

    # run it a little more
    myDriver.sampler.run_sampler(1000, burn_steps=1)

    # run it a little more (tests adding to results object)
    myDriver.sampler.run_sampler(1000, burn_steps=1)

    # test that lnlikes being saved are correct
    returned_lnlike_test = myDriver.sampler.results.lnlike[0]
    computed_lnlike_test = myDriver.sampler._logl(myDriver.sampler.results.post[0])

    assert returned_lnlike_test == pytest.approx(computed_lnlike_test, abs=0.01)


def test_examine_chop_chains(num_temps=0, num_threads=1):
    """
    Tests the MCMC sampler's examine_chains and chop_chains methods
    Args:
        num_temps: Number of temperatures to use
            Uses Parallel Tempering MCMC (ptemcee) if > 1,
            otherwises, uses Affine-Invariant Ensemble Sampler (emcee)
        num_threads: number of threads to run
    """

    # use the test_csv dir
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
    data_table = read_input.read_formatted_file(input_file)
    # Manually set 'object' column of data table
    data_table['object'] = 1

    # construct the system
    orbit = system.System(1, data_table, 1, 0.01)

    # construct Driver
    n_walkers = 20
    mcmc = sampler.MCMC(orbit, num_temps, n_walkers, num_threads=num_threads)

    # run it a little
    n_samples1 = 2000  # 100 steps for each of 20 walkers
    n_samples2 = 2000  # 100 steps for each of 20 walkers
    n_samples = n_samples1+n_samples2
    mcmc.run_sampler(n_samples1)
    # run it a little more (tries examine_chains within run_sampler)
    mcmc.run_sampler(n_samples2, examine_chains=True)
    # (4000 orbit samples = 20 walkers x 200 steps)

    # Try all variants of examine_chains
    mcmc.examine_chains()
    plt.close('all')  # Close figures generated
    fig_list = mcmc.examine_chains(param_list=['sma1', 'ecc1', 'inc1'])
    # Should only get 3 figures
    assert len(fig_list) == 3
    plt.close('all')  # Close figures generated
    mcmc.examine_chains(walker_list=[10, 12])
    plt.close('all')  # Close figures generated
    mcmc.examine_chains(n_walkers=5)
    plt.close('all')  # Close figures generated
    mcmc.examine_chains(step_range=[50, 100])
    plt.close('all')  # Close figures generated

    # Now try chopping the chains
    # Chop off first 50 steps
    chop1 = 50
    mcmc.chop_chains(chop1)
    # Calculate expected number of orbits now
    expected_total_orbits = n_samples - chop1*n_walkers
    # Check lengths of arrays in results object
    assert len(mcmc.results.lnlike) == expected_total_orbits
    assert mcmc.results.post.shape[0] == expected_total_orbits

    # With 150 steps left, now try to trim 25 steps off each end
    chop2 = 25
    trim2 = 25
    mcmc.chop_chains(chop2, trim=trim2)
    # Calculated expected number of orbits now
    samples_removed = (chop1 + chop2 + trim2)*n_walkers
    expected_total_orbits = n_samples - samples_removed
    # Check lengths of arrays in results object
    assert len(mcmc.results.lnlike) == expected_total_orbits
    assert mcmc.results.post.shape[0] == expected_total_orbits


if __name__ == "__main__":
    # Parallel Tempering tests
    test_mcmc_runs(num_temps=2, num_threads=1)
    test_mcmc_runs(num_temps=2, num_threads=4)
    # Ensemble MCMC tests
    test_mcmc_runs(num_temps=0, num_threads=1)
    test_mcmc_runs(num_temps=0, num_threads=8)
    # Test examine/chop chains
    test_examine_chop_chains(num_temps=5)  # PT
    test_examine_chop_chains(num_temps=0)  # Ensemble
