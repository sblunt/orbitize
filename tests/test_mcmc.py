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
<<<<<<< .merge_file_Hq2y1x
    n_temps=2
    n_walkers=100
    mcmc = sampler.PTMCMC(chi2_lnlike, orbit, n_temps, n_walkers, num_threads=num_threads)

    # run it a little
    mcmc.run_sampler(10, 1)
    # run it a little more (tests adding to results object
    mcmc.run_sampler(10, 1)
=======
    mcmc = sampler.PTMCMC(chi2_lnlike, orbit, 2, 100, num_threads=num_threads)

    # run it a little
    emcee_sampler_obj = mcmc.run_sampler(10, 1)

    print(emcee_sampler_obj.chain[0, 0])
>>>>>>> .merge_file_k3A3gQ

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
<<<<<<< .merge_file_Hq2y1x
    n_walkers=100
    mcmc = sampler.EnsembleMCMC(chi2_lnlike, orbit, n_walkers, num_threads=num_threads)

    # run it a little
    mcmc.run_sampler(10, burn_steps=1)
    # run it a little more (tests adding to results object)
    mcmc.run_sampler(10, burn_steps=1)
=======
    mcmc = sampler.EnsembleMCMC(chi2_lnlike, orbit, 100, num_threads=num_threads)

    # run it a little
    emcee_sampler_obj = mcmc.run_sampler(10, burn_steps=1)

    print(emcee_sampler_obj.chain[0, 0])
>>>>>>> .merge_file_k3A3gQ

if __name__ == "__main__":
    test_pt_mcmc_runs(num_threads=1)
    test_pt_mcmc_runs(num_threads=4)
    test_ensemble_mcmc_runs(num_threads=1)
    test_ensemble_mcmc_runs(num_threads=8)
