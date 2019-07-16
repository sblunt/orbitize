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
        num_threads: number of threads to run
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
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    data_table = read_input.read_file(input_file)
    # Manually set 'object' column of data table
    data_table['object'] = 1

    # construct the system
    orbit = system.System(1, data_table, 1, 0.01)

    # construct sampler
    n_walkers = 100
    mcmc = sampler.MCMC(orbit, num_temps, n_walkers, num_threads=num_threads)

    # run it a little 
    mcmc.run_sampler(100)
    # run it a little more (tries examine_chains within run_sampler)
    mcmc.run_sampler(100, examine_chains=True)
    
    # Try all variants of examine_chains
    fig_list = mcmc.examine_chains()
    fig_list = mcmc.examine_chains(param_list=['sma1','ecc1','inc1'])
    if num_temps > 1:
        fig_list = mcmc.examine_chains(temp=1)
    fig_list = mcmc.examine_chains(walker_list=[10, 20])
    fig_list = mcmc.examine_chains(n_walkers=5)
    fig_list = mcmc.examine_chains(step_range=[50,100])
    examine_chains(self, param_list=None, temp=0, walker_list=None, n_walkers=None, step_range=None):

if __name__ == "__main__":
    # Parallel Tempering tests
    test_mcmc_runs(num_temps=2, num_threads=1)
    test_mcmc_runs(num_temps=2, num_threads=4)
    # Ensemble MCMC tests
    test_mcmc_runs(num_temps=0, num_threads=1)
    test_mcmc_runs(num_temps=0, num_threads=8)
    # Test examine/chop chains
    test_examine_chop_chains(num_temps=5) # PT
    test_examine_chop_chains(num_temps=0) # Ensemble
