import os
import orbitize.sampler as sampler
import orbitize.system as system
import orbitize.read_input as read_input
from orbitize.lnlike import chi2_lnlike

def test_pt_mcmc_runs(num_threads=1):
    """
    Tests the PTMCMC sampler by making sure it even runs
    """
    # use the test_csv dir
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    data_table = read_input.read_formatted_file(input_file)
    # Manually set 'object' column of data table
    data_table['object'] = 1

    # construct the system
    orbit = system.System(1, data_table, 1, 0.01)

    # construct sampler
    mcmc = sampler.PTMCMC(chi2_lnlike, orbit, 2, 100, num_threads=num_threads)

    # run it a little
    mcmc.run_sampler(10, 1)

    print(mcmc.sampler.chain[0, 0])

def test_ensemble_mcmc_runs(num_threads=1):
    """
    Tests the EnsembleMCMC sampler by making sure it even runs
    """
    # use the test_csv dir
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    data_table = read_input.read_formatted_file(input_file)
    # Manually set 'object' column of data table
    data_table['object'] = 1

    # construct the system
    orbit = system.System(1, data_table, 1, 0.01)

    # construct sampler
    mcmc = sampler.EnsembleMCMC(chi2_lnlike, orbit, 100, num_threads=num_threads)

    # run it a little
    mcmc.run_sampler(10, 1)

    print(mcmc.sampler.chain[0, 0])

if __name__ == "__main__":
    test_pt_mcmc_runs(num_threads=1)
    test_pt_mcmc_runs(num_threads=4)
    test_ensemble_mcmc_runs(num_threads=1)
    test_ensemble_mcmc_runs(num_threads=4)
