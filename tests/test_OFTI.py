"""
Test the orbitize.sampler OFTI class which performs OFTI on astrometric data
"""
import numpy as np
import os
import pytest
import matplotlib.pyplot as plt

import orbitize.sampler as sampler
import orbitize.driver
import orbitize.priors as priors

testdir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(testdir, 'GJ504.csv')
input_file_1epoch = os.path.join(testdir, 'GJ504_1epoch.csv')

def test_scale_and_rotate():
    
    # perform scale-and-rotate
    myDriver = orbitize.driver.Driver(input_file, 'OFTI',
    1, 1.22, 56.95,mass_err=0.08, plx_err=0.26)
    
    s = myDriver.sampler
    samples = s.prepare_samples(100)

    sma,ecc,inc,argp,lan,tau,plx,mtot = [samp for samp in samples]
    
    ra, dec, vc = orbitize.kepler.calc_orbit(s.epochs, sma, ecc, inc, argp, lan, tau, plx, mtot)
    sep, pa = orbitize.system.radec2seppa(ra, dec)
    sep_sar, pa_sar = np.median(sep[s.epoch_idx]), np.median(pa[s.epoch_idx])
    
    # test to make sure sep and pa scaled to scale-and-rotate epoch
    sar_epoch = s.data_table[s.epoch_idx]
    assert sep_sar == pytest.approx(sar_epoch['quant1'], abs=sar_epoch['quant1_err'])
    assert pa_sar == pytest.approx(sar_epoch['quant2'], abs=sar_epoch['quant2_err'])

    # test scale-and-rotate for orbits run all the way through OFTI
    s.run_sampler(100)

    # test orbit plot generation
    s.results.plot_orbits(start_mjd=s.epochs[0])
    
    samples = s.results.post
    sma = samples[:,0]
    ecc = samples[:,1]
    inc = samples[:,2]
    argp = samples[:,3]
    lan = samples[:,4]
    tau = samples[:,5]
    plx = samples[:,6]
    mtot = samples[:,7]

    ra, dec, vc = orbitize.kepler.calc_orbit(s.epochs, sma, ecc, inc, argp, lan, tau, plx, mtot)
    sep, pa = orbitize.system.radec2seppa(ra, dec)
    sep_sar, pa_sar = np.median(sep[s.epoch_idx]), np.median(pa[s.epoch_idx])
    
    # test to make sure sep and pa scaled to scale-and-rotate epoch
    assert sep_sar == pytest.approx(sar_epoch['quant1'], abs=sar_epoch['quant1_err'])
    assert pa_sar == pytest.approx(sar_epoch['quant2'], abs=sar_epoch['quant2_err'])


def test_run_sampler():
    
    # initialize sampler
    myDriver = orbitize.driver.Driver(input_file, 'OFTI',
    1, 1.22, 56.95,mass_err=0.08, plx_err=0.26)
    
    s = myDriver.sampler

    # change eccentricity prior
    myDriver.system.sys_priors[1] = priors.LinearPrior(-2.18, 2.01)
    
    # test num_samples=1
    s.run_sampler(0,num_samples=1)    

    # test to make sure outputs are reasonable
    orbits = s.run_sampler(1000)

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
        
    # test with only one epoch
    myDriver = orbitize.driver.Driver(input_file_1epoch, 'OFTI',
    1, 1.22, 56.95,mass_err=0.08, plx_err=0.26)
    s = myDriver.sampler
    s.run_sampler(1)
    print()
    
def test_fixed_sys_params_sampling():
    # test in case of fixed mass and parallax
    myDriver = orbitize.driver.Driver(input_file, 'OFTI',
    1, 1.22, 56.95)
    
    s = myDriver.sampler
    samples = s.prepare_samples(100)
    assert np.all(samples[-1] == s.priors[-1])
    assert isinstance(samples[-3], np.ndarray)


if __name__ == "__main__":
    test_scale_and_rotate()
    test_run_sampler()
    print("Done!")