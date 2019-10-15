import pytest
import numpy as np
import os
from orbitize.driver import Driver
import orbitize.sampler as sampler
import orbitize.system as system
import orbitize.read_input as read_input
import pdb


def test_pt_mcmc_runs(num_threads=1):
    """
    Tests the PTMCMC sampler by making sure it even runs
    """

    # use the test_csv dir
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')

    myDriver = Driver(input_file, 'MCMC', 1, 1, 0.01, mass_err=0.05, plx_err=0.01,
                      system_kwargs={'fit_secondary_mass': True, 'tau_ref_epoch': 0},
                      mcmc_kwargs={'num_temps': 2, 'num_threads': num_threads, 'num_walkers': 100}
                      )

    # run it a little (tests 0 burn-in steps)
    myDriver.sampler.run_sampler(100)

    # run it a little more
    #myDriver.sampler.run_sampler(1000, burn_steps=1)

    # run it a little more (tests adding to results object)
    #myDriver.sampler.run_sampler(1000, burn_steps=1)

    s = myDriver.sampler
    print(np.shape(s.chain))
    print(np.shape(s.results.post))

    # test that lnlikes being saved are correct
    returned_lnlike_test = myDriver.sampler.results.lnlike[0]
    computed_lnlike_test = myDriver.sampler._logl(myDriver.sampler.results.post[0])
    pdb.set_trace()
    print('returned lnlike test:', returned_lnlike_test)
    print('computed lnlike test:', computed_lnlike_test)

    assert returned_lnlike_test == pytest.approx(computed_lnlike_test, abs=0.01)  # was 0.01 for abs


def test_ensemble_mcmc_runs(num_threads=1):
    """
    Tests the EnsembleMCMC sampler by making sure it even runs
    """

    # use the test_csv dir
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')

    myDriver = Driver(input_file, 'MCMC', 1, 1, 0.01,
                      system_kwargs={'fit_secondary_mass': True,
                                     'tau_ref_epoch': 0},
                      mcmc_kwargs={'num_temps': 1, 'num_threads': num_threads, 'num_walkers': 100}
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


if __name__ == "__main__":
    test_pt_mcmc_runs(num_threads=1)
    test_pt_mcmc_runs(num_threads=4)
    test_ensemble_mcmc_runs(num_threads=1)
test_ensemble_mcmc_runs(num_threads=8)
