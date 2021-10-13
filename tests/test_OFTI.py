#!/usr/bin/env python

"""
Test the orbitize.sampler OFTI class which performs OFTI on astrometric data
"""
import numpy as np
import os
import pytest
import time
import orbitize
import orbitize.sampler as sampler
import orbitize.driver
import orbitize.priors as priors
import orbitize.system as system
import orbitize.system
from orbitize.hipparcos import HipparcosLogProb

input_file = os.path.join(orbitize.DATADIR, 'GJ504.csv')
input_file_1epoch = os.path.join(orbitize.DATADIR, 'GJ504_1epoch.csv')
input_file_rvs = os.path.join(orbitize.DATADIR, 'HD4747.csv')

def test_scale_and_rotate():

    # perform scale-and-rotate
    myDriver = orbitize.driver.Driver(
        input_file, 'OFTI', 1, 1.22, 56.95, mass_err=0.08, plx_err=0.26
    )

    s = myDriver.sampler

    samples = s.prepare_samples(100)


    sma, ecc, inc, argp, lan, tau, plx, mtot = [samp for samp in samples]

    ra, dec, vc = orbitize.kepler.calc_orbit(
        s.epochs, sma, ecc, inc, argp, lan, tau, plx, mtot, tau_ref_epoch=0
    )
    sep, pa = orbitize.system.radec2seppa(ra, dec)
    sep_sar, pa_sar = np.median(sep[s.epoch_idx]), np.median(pa[s.epoch_idx])

    # test to make sure sep and pa scaled to scale-and-rotate epoch
    sar_epoch = s.system.data_table[s.epoch_idx]
    assert sep_sar == pytest.approx(sar_epoch['quant1'], abs=sar_epoch['quant1_err'])
    assert pa_sar == pytest.approx(sar_epoch['quant2'], abs=sar_epoch['quant2_err'])

    # test scale-and-rotate for orbits run all the way through OFTI
    s.run_sampler(100)

    # test orbit plot generation
    s.results.plot_orbits(start_mjd=s.epochs[0])

    samples = s.results.post
    sma = samples[:, 0]
    ecc = samples[:, 1]
    inc = samples[:, 2]
    argp = samples[:, 3]
    lan = samples[:, 4]
    tau = samples[:, 5]
    plx = samples[:, 6]
    mtot = samples[:, 7]

    ra, dec, vc = orbitize.kepler.calc_orbit(
        s.epochs, sma, ecc, inc, argp, lan, tau, plx, mtot, tau_ref_epoch=0
    )
    assert np.max(lan) > np.pi
    sep, pa = orbitize.system.radec2seppa(ra, dec)
    sep_sar, pa_sar = np.median(sep[s.epoch_idx]), np.median(pa[s.epoch_idx])

    # test to make sure sep and pa scaled to scale-and-rotate epoch
    assert sep_sar == pytest.approx(sar_epoch['quant1'], abs=sar_epoch['quant1_err'])
    assert pa_sar == pytest.approx(sar_epoch['quant2'], abs=sar_epoch['quant2_err'])

    # test scale-and-rotate with restricted upper limits on PAN
    myDriver = orbitize.driver.Driver(input_file, 'OFTI',
                                      1, 1.22, 56.95, mass_err=0.08, plx_err=0.26, system_kwargs={'restrict_angle_ranges':True})
    s = myDriver.sampler
    samples = s.prepare_samples(100)

    sma, ecc, inc, argp, lan, tau, plx, mtot = [samp for samp in samples]

    assert np.max(lan) < np.pi
    assert np.max(argp) > np.pi and np.max(argp) < 2 * np.pi

    ra, dec, vc = orbitize.kepler.calc_orbit(
        s.epochs, sma, ecc, inc, argp, lan, tau, plx, mtot, tau_ref_epoch=0
    )
    sep, pa = orbitize.system.radec2seppa(ra, dec)
    sep_sar, pa_sar = np.median(sep[s.epoch_idx]), np.median(pa[s.epoch_idx])

    sar_epoch = s.system.data_table[s.epoch_idx]
    assert sep_sar == pytest.approx(sar_epoch['quant1'], abs=sar_epoch['quant1_err'])
    assert pa_sar == pytest.approx(sar_epoch['quant2'], abs=sar_epoch['quant2_err'])


sma_seppa = 0
seppa_lnprob_compare = None
def test_run_sampler():
    global sma_seppa, seppa_lnprob_compare # use for covariances test

    # initialize sampler
    myDriver = orbitize.driver.Driver(input_file, 'OFTI',
    1, 1.22, 56.95, mass_err=0.08, plx_err=0.26)

    s = myDriver.sampler

    # change eccentricity prior
    myDriver.system.sys_priors[1] = priors.LinearPrior(-2.18, 2.01)

    # test num_samples=1
    s.run_sampler(0, num_samples=1)

    # test to make sure outputs are reasonable
    start=time.time()
    orbits = s.run_sampler(1000, num_cores=4)
    end=time.time()

    print()
    print("Runtime: " + str(end-start) +" s")
    print()
    print(orbits[0])

    # test that lnlikes being saved are correct
    returned_lnlike_test = s.results.lnlike[0]
    computed_lnlike_test = s._logl(orbits[0])
    assert returned_lnlike_test == pytest.approx(computed_lnlike_test, abs=0.01)

    seppa_lnprob_compare = (orbits[0], computed_lnlike_test) # one set of params and associated lnlike saved. 

    print()
    idx = s.system.param_idx
    sma = np.median([x[idx['sma1']] for x in orbits])
    ecc = np.median([x[idx['ecc1']] for x in orbits])
    inc = np.median([x[idx['inc1']] for x in orbits])

    # expected values from Blunt et al. (2017)
    sma_exp = 48.
    ecc_exp = 0.19
    inc_exp = np.radians(140)

    # test to make sure OFTI values are within 20% of expectations
    assert sma == pytest.approx(sma_exp, abs=0.2*sma_exp)
    assert ecc == pytest.approx(ecc_exp, abs=0.2*ecc_exp)
    assert inc == pytest.approx(inc_exp, abs=0.2*inc_exp)

    sma_seppa = sma # use for covarinaces test

    # test with only one core
    orbits = s.run_sampler(100, num_cores=1)

    # test with only one epoch
    myDriver = orbitize.driver.Driver(input_file_1epoch, 'OFTI',
                                      1, 1.22, 56.95, mass_err=0.08, plx_err=0.26)
    s = myDriver.sampler
    s.run_sampler(1)
    print()

def test_not_implemented():
    """
    Check that not implemented errors for RVs & Hipparcos IAD + OFTI work
    """

    data_table = orbitize.read_input.read_file(input_file)

    # test that if the `hipparcosIAD` attribute is set, OFTI won't work
    try:

        hip_num = '027321'
        num_secondary_bodies = 1
        iad_file = '{}/HIP{}.d'.format(orbitize.DATADIR, hip_num)
        myHip = HipparcosLogProb(iad_file, hip_num, num_secondary_bodies)
        mySystem = system.System(
            1, data_table, 1.22, 56.95, mass_err=0.08, plx_err=0.26, 
            hipparcos_IAD=myHip
        )
        _ = sampler.OFTI(mySystem)
        assert False, 'test failed'
    except NotImplementedError:
        pass

    # test that if there are RVs in the data file, OFTI won't work
    data_table_with_rvs = orbitize.read_input.read_file(input_file_rvs)
    try:
        _ = system.System(
            1, data_table_with_rvs, 1.22, 56.95, mass_err=0.08, plx_err=0.26
        )
        _ = sampler.OFTI(mySystem)
        assert False, 'test failed'
    except NotImplementedError:
        pass


def test_fixed_sys_params_sampling():
    # test in case of fixed mass and parallax
    myDriver = orbitize.driver.Driver(input_file, 'OFTI',
                                      1, 1.22, 56.95)

    s = myDriver.sampler
    samples = s.prepare_samples(100)
    assert np.all(samples[-1] == s.priors[-1])
    assert isinstance(samples[-3], np.ndarray)


def profile_system():
    import pycuda.driver
    # pycuda.driver.initialize_profiler()

    # initialize sampler
    myDriver = orbitize.driver.Driver(input_file, 'OFTI',
    1, 1.22, 56.95, mass_err=0.08, plx_err=0.26)

    s = myDriver.sampler

    # change eccentricity prior
    myDriver.system.sys_priors[1] = priors.LinearPrior(-2.18, 2.01)

    # test num_samples=1
    s.run_sampler(0, num_samples=1)

    # test to make sure outputs are reasonable
    pycuda.driver.start_profiler()
    start=time.time()
    orbitize.cext = True
    orbitize.cuda_ext = True
    orbits = s.run_sampler(30000, num_samples=10000, num_cores = 1)
    end=time.time()
    
    print()
    print("CUDA Runtime: " + str(end-start) +" s")
    print()
    print(orbits[0])

    start=time.time()
    orbitize.cext = True
    orbitize.cuda_ext = False
    orbits = s.run_sampler(30000)
    end=time.time()
    pycuda.driver.stop_profiler()
    
    print()
    print("MULTIPROCESSING Runtime: " + str(end-start) +" s")
    print()
    print(orbits[0])

    start=time.time()
    orbitize.cext = True
    orbitize.cuda_ext = False
    orbits = s.run_sampler(30000, num_cores = 1)
    end=time.time()
    pycuda.driver.stop_profiler()
    pycuda.autoinit.context.detach()
    
    print()
    print("single threaded Runtime: " + str(end-start) +" s")
    print()
    print(orbits[0])

def test_OFTI_multiplanet():
    # initialize sampler
    input_file = os.path.join(orbitize.DATADIR, "test_val_multi.csv")
    myDriver = orbitize.driver.Driver(input_file, 'OFTI',
                                      2, 1.52, 24.76, mass_err=0.15, plx_err=0.64)

    s = myDriver.sampler
    # change eccentricity prior for b
    myDriver.system.sys_priors[1] = priors.UniformPrior(0.0, 0.1)
    # change eccentricity prior for c
    myDriver.system.sys_priors[7] = priors.UniformPrior(0.0, 0.1)

    orbits = s.run_sampler(500)

    idx = s.system.param_idx
    sma1 = np.median(orbits[:,idx['sma1']])
    sma2 = np.median(orbits[:,idx['sma2']])

    sma1_exp = 66
    sma2_exp = 40
    print(sma1, sma2)
    assert sma1 == pytest.approx(sma1_exp, abs=0.3*sma1_exp)
    assert sma2 == pytest.approx(sma2_exp, abs=0.3*sma2_exp)
    assert np.all(orbits[:, idx['ecc1']] < 0.1)
    assert np.all(orbits[:, idx['ecc2']] < 0.1)

@pytest.hookimpl(trylast=True)
def test_OFTI_covariances():
    """
    Test OFTI fits by turning sep/pa measurements to RA/Dec measurements with covariances

    Needs to be run after test_run_sampler()!!
    """
    # only run if these variables are set. 
    if sma_seppa == 0 or seppa_lnprob_compare is None:
        print("Skipping OFTI covariances test because reference data not initalized. Please make sure test_run_sampler is run first.")
        return

    # read in seppa data table and turn into raddec data table
    data_table = orbitize.read_input.read_file(input_file)
    data_ra, data_dec = system.seppa2radec(data_table['quant1'], data_table['quant2'])
    data_raerr, data_decerr, data_radeccorr = [], [], []

    for row in data_table:
        
        raerr, decerr, radec_corr = system.transform_errors(row['quant1'], row['quant2'], 
                                                            row['quant1_err'], row['quant2_err'],
                                                            0, system.seppa2radec, nsamps=10000000)
        data_raerr.append(raerr)
        data_decerr.append(decerr)
        data_radeccorr.append(radec_corr)

    data_table['quant1'] = data_ra
    data_table['quant2'] = data_dec
    data_table['quant1_err'] = np.array(data_raerr)
    data_table['quant2_err'] = np.array(data_decerr)
    data_table['quant12_corr'] = np.array(data_radeccorr)
    data_table['quant_type'] = np.array(['radec' for _ in data_table])

    # initialize system
    my_sys = system.System(1, data_table, 1.22, 56.95, mass_err=0.08, plx_err=0.26)
    # initialize sampler
    s = sampler.OFTI(my_sys)

    # change eccentricity prior
    my_sys.sys_priors[1] = priors.LinearPrior(-2.18, 2.01)

    # test num_samples=1
    s.run_sampler(0, num_samples=1)

    # test to make sure outputs are reasonable
    orbits = s.run_sampler(1000, num_cores=4)

    # test that lnlikes being saved are correct
    returned_lnlike_test = s.results.lnlike[0]
    computed_lnlike_test = s._logl(orbits[0])
    assert returned_lnlike_test == pytest.approx(computed_lnlike_test, abs=0.01)

    # test that the lnlike is very similar to the values computed in seppa space
    ref_params, ref_lnlike = seppa_lnprob_compare
    computed_lnlike_ref = s._logl(ref_params)
    assert ref_lnlike == pytest.approx(computed_lnlike_ref, abs=0.05) # 5% differencesin lnprob is allowable. 

    idx = s.system.param_idx
    sma = np.median([x[idx['sma1']] for x in orbits])
    ecc = np.median([x[idx['ecc1']] for x in orbits])
    inc = np.median([x[idx['inc1']] for x in orbits])

    # test against seppa fits to see they are similar
    assert sma_seppa == pytest.approx(sma, abs=0.2 * sma_seppa)

def test_OFTI_pan_priors():

    # initialize sampler
    myDriver = orbitize.driver.Driver(
        input_file, 'OFTI', 1, 1.22, 56.95, mass_err=0.08, plx_err=0.26)

    s = myDriver.sampler

    # change PAN prior
    new_min = 0.05
    new_max = np.pi - 0.05
    myDriver.system.sys_priors[4] = priors.UniformPrior(new_min, new_max)

    # run sampler
    orbits = s.run_sampler(100)

    # check that bounds were applied correctly
    assert np.max(orbits[:,4]) < new_max
    assert np.min(orbits[:,4]) > new_min

    # change PAN prior again
    mu = np.pi / 2
    sigma = 0.05
    myDriver.system.sys_priors[4] = priors.GaussianPrior(mu, sigma = sigma)

    # run sampler again
    orbits = s.run_sampler(250)

    # check that bounds were applied correctly
    assert mu == pytest.approx(np.mean(orbits[:,4]), abs=0.01) 
    assert sigma == pytest.approx(np.std(orbits[:,4]), abs=0.01)

if __name__ == "__main__":

    test_scale_and_rotate()
    test_run_sampler()
    test_OFTI_covariances()
    test_OFTI_multiplanet()
    test_not_implemented()
    test_fixed_sys_params_sampling()
    test_OFTI_pan_priors()
    # profile_system()
    print("Done!")
