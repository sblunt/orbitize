import os
import orbitize.sampler as sampler
import orbitize.system as system
import orbitize.read_input as read_input

def test_mcmc_runs(num_temps=0, num_threads=1):
    """
    Tests the MCMC sampler by making sure it even runs
    Args:
        num_temps: Number of temperatures to use
            Uses Parallel Tempering MCMC (ptemcee) if > 1, 
            otherwises, uses Affine-Invariant Ensemble Sampler (emcee)
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
    mcmc = sampler.MCMC(orbit, num_temps, n_walkers, num_threads=num_threads)

    # run it a little (tests 0 burn-in steps)
    mcmc.run_sampler(100)
    # run it a little more (tests adding to results object)
    mcmc.run_sampler(500, burn_steps=10)
    # run it a little more (tries examine_chains within run_sampler)
    mcmc.run_sampler(500, burn_steps=10, examine_chains=True)

if __name__ == "__main__":
    # Parallel Tempering tests
    test_mcmc_runs(num_temps=2, num_threads=1)
    test_mcmc_runs(num_temps=2, num_threads=4)
    # Ensemble MCMC tests
    test_mcmc_runs(num_temps=0, num_threads=1)
    test_mcmc_runs(num_temps=0, num_threads=8)
