import os
import orbitize.sampler as sampler
import orbitize.system as system
import orbitize.read_input as read_input

def test_pt_mcmc_runs(num_threads=1):
    """
    Tests the PTMCMC sampler by making sure it even runs
    """
    # use the test_csv dir
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    data_table = read_input.read_file(input_file)
    # Manually set 'object' column of data table
    data_table['object'] = 1

    # construct the system
    orbit = system.System(1, data_table, 1, 0.01)

    # construct sampler
    n_temps=2
    n_walkers=100
    mcmc = sampler.MCMC(orbit, n_temps, n_walkers, num_threads=num_threads)

    # run it a little (tests 0 burn-in steps)
    mcmc.run_sampler(100)
    # run it a little more
    mcmc.run_sampler(1000, 1)
    # run it a little more (tests adding to results object)
    mcmc.run_sampler(1000, 1)

def test_ensemble_mcmc_runs(num_threads=1):
    """
    Tests the EnsembleMCMC sampler by making sure it even runs
    """
    # use the test_csv dir
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    data_table = read_input.read_file(input_file)
    # Manually set 'object' column of data table
    data_table['object'] = 1

    # construct the system
    orbit = system.System(1, data_table, 1, 0.01)

    # construct sampler
    n_walkers=100
    mcmc = sampler.MCMC(orbit, 0, n_walkers, num_threads=num_threads)


    # run it a little (tests 0 burn-in steps)
    mcmc.run_sampler(100)
    # run it a little more
    mcmc.run_sampler(1000, burn_steps=1)
    # run it a little more (tests adding to results object)
    mcmc.run_sampler(1000, burn_steps=1)

if __name__ == "__main__":
    test_pt_mcmc_runs(num_threads=1)
    test_pt_mcmc_runs(num_threads=4)
    test_ensemble_mcmc_runs(num_threads=1)
    test_ensemble_mcmc_runs(num_threads=8)
