import numpy as np
import os

import matplotlib.pyplot as plt

from orbitize import DATADIR, read_input, system, sampler, results
from orbitize.gaia import GaiaLogProb
from orbitize.hipparcos import HipparcosLogProb, nielsen_iad_refitting_test

def test_hipparcos_api():
    """
    Check that error is caught for a star with solution type != 5 param, 
    and that doing an RV + Hipparcos IAD fit produces the expected list of 
    Prior objects.
    """

    # check sol type != 5 error message
    hip_num = '000025'
    num_secondary_bodies = 1
    path_to_iad_file = '{}H{}.d'.format(DATADIR, hip_num)

    try:
        _ = HipparcosLogProb(path_to_iad_file, hip_num, num_secondary_bodies)
        assert False, 'Test failed.'
    except ValueError: 
        pass

    # check that RV + Hip gives correct prior array labels
    hip_num = '027321' # beta Pic
    num_secondary_bodies = 1
    path_to_iad_file = '{}HIP{}.d'.format(DATADIR, hip_num)

    myHip = HipparcosLogProb(path_to_iad_file, hip_num, num_secondary_bodies)

    input_file = os.path.join(DATADIR, 'HD4747.csv')
    data_table_with_rvs = read_input.read_file(input_file)
    mySys = system.System(
        1, data_table_with_rvs, 1.22, 56.95, mass_err=0.08, plx_err=0.26, 
        hipparcos_IAD=myHip, fit_secondary_mass=True
    )

    # test that `fit_secondary_mass` and `track_planet_perturbs` keywords are set appropriately
    assert mySys.fit_secondary_mass
    assert mySys.track_planet_perturbs

    assert len(mySys.sys_priors) == 15 # 7 orbital params + 2 mass params + 
                                       # 4 Hip nuisance params + 
                                       # 2 RV nuisance params

    assert mySys.labels == [
       'sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'plx', 'pm_ra', 'pm_dec', 
       'alpha0', 'delta0', 'gamma_defrv', 'sigma_defrv', 'm1', 'm0'
   ]

    # test that `fit_secondary_mass` and `track_planet_perturbs` keywords are 
    # set appropriately for non-Hipparcos system
    noHip_system = system.System(
        num_secondary_bodies, data_table_with_rvs, 1.0, 1.0, hipparcos_IAD=None, 
        fit_secondary_mass=False, mass_err=0.01, plx_err=0.01
    )

    assert not noHip_system.fit_secondary_mass
    assert not noHip_system.track_planet_perturbs

    # check that negative residuals are rejected properly
    hip_num = '000026' # contains one negative residual
    num_secondary_bodies = 1
    path_to_iad_file = '{}H{}.d'.format(DATADIR, hip_num)

    raw_iad_data = np.transpose(np.loadtxt(path_to_iad_file))

    rejected_scansHip = HipparcosLogProb(path_to_iad_file, hip_num, num_secondary_bodies)
    assert len(rejected_scansHip.cos_phi) == len(raw_iad_data[0]) - 1

def test_dvd_vs_2021catalog():
    """
    Test code's ability to parse both a DVD data file and a 2021
    data file. Assert that these two files (for beta Pic) give the
    same best-fit astrometric solution and the same IAD.
    """

    hip_num = '027321'
    num_secondary_bodies = 1
    iad_file_2021 = '{}H{}.d'.format(DATADIR, hip_num)
    iad_file_dvd = '{}HIP{}.d'.format(DATADIR, hip_num)

    # first, test reading of 2021 catalog
    new_iadHipLogProb = HipparcosLogProb(
        iad_file_2021, hip_num, num_secondary_bodies
    )

    # next, test reading of a DVD file
    old_iadHipLogProb = HipparcosLogProb(
        iad_file_dvd, hip_num, num_secondary_bodies
    )

    # test that these give the same data file for beta Pic (which has no rejected scans)
    assert np.abs(new_iadHipLogProb.plx0 - old_iadHipLogProb.plx0) < 1e-3 # (plx precise to 0.01)
    assert np.abs(new_iadHipLogProb.plx0_err - old_iadHipLogProb.plx0_err) < 1e-3
    assert np.abs(new_iadHipLogProb.pm_ra0 - old_iadHipLogProb.pm_ra0) < 1e-3
    assert np.abs(new_iadHipLogProb.pm_ra0_err - old_iadHipLogProb.pm_ra0_err) < 1e-3
    assert np.abs(new_iadHipLogProb.pm_dec0 - old_iadHipLogProb.pm_dec0) < 1e-3
    assert np.abs(new_iadHipLogProb.pm_dec0_err - old_iadHipLogProb.pm_dec0_err) < 1e-3
    assert np.abs(new_iadHipLogProb.alpha0 - old_iadHipLogProb.alpha0) < 1e-8
    assert np.abs(new_iadHipLogProb.alpha0_err - old_iadHipLogProb.alpha0_err) < 1e-3
    assert np.abs(new_iadHipLogProb.delta0 - old_iadHipLogProb.delta0) < 1e-8
    assert np.abs(new_iadHipLogProb.delta0_err - old_iadHipLogProb.delta0_err) < 1e-3

    # this also asserts that they're the same length, i.e. no rejected scans
    assert np.all(np.isclose(new_iadHipLogProb.cos_phi, old_iadHipLogProb.cos_phi, atol=1e-2))
    assert np.all(np.isclose(new_iadHipLogProb.sin_phi, old_iadHipLogProb.sin_phi, atol=1e-2))
    assert np.all(np.isclose(new_iadHipLogProb.epochs, old_iadHipLogProb.epochs, atol=1e-2))

def test_iad_refitting():
    """
    Check that refitting the IAD gives posteriors that approximately match
    the official Hipparcos values. Only run the MCMC for a few steps because 
    this is a unit test. 
    """

    post, myHipLogProb = nielsen_iad_refitting_test(
        '{}/HIP027321.d'.format(DATADIR), burn_steps=10, mcmc_steps=200, 
        saveplot=None
    )

    # check that we get reasonable values for the posteriors of the refit IAD
    # (we're only running the MCMC for a few steps, so these are not strict)
    assert np.isclose(0, np.median(post[:, -1]), atol=0.1)
    assert np.isclose(myHipLogProb.plx0, np.median(post[:, 0]), atol=0.1)

def test_save_load():
    """
    Set up a Hip IAD + Gaia fit, save the results, and load them.
    """

    hip_num = '027321' # beta Pic

    num_secondary_bodies = 1
    path_to_iad_file = '{}HIP{}.d'.format(DATADIR, hip_num)

    myHip = HipparcosLogProb(path_to_iad_file, hip_num, num_secondary_bodies)
    myGaia = GaiaLogProb(4792774797545800832, myHip, dr='edr3')

    input_file = os.path.join(DATADIR, 'HD4747.csv')
    data_table_with_rvs = read_input.read_file(input_file)
    mySys = system.System(
        1, data_table_with_rvs, 1.22, 56.95, mass_err=0.08, plx_err=0.26, 
        hipparcos_IAD=myHip, fit_secondary_mass=True, gaia=myGaia
    )
    n_walkers = 50
    mySamp = sampler.MCMC(mySys, num_walkers=n_walkers)
    mySamp.run_sampler(n_walkers, burn_steps=0)
    filename = 'tmp.hdf5'
    mySamp.results.save_results(filename)

    myResults = results.Results()
    myResults.load_results(filename)

    os.system('rm tmp.hdf5')



if __name__ == '__main__':
    test_save_load()
    # test_hipparcos_api()
    # test_iad_refitting()
    # test_dvd_vs_2021catalog()

