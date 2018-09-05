"""
Test the orbitize.sampler OFTI class which performs OFTI on astrometric data
"""
import numpy as np
import os
import pytest

import orbitize.sampler as sampler
import orbitize.driver
import orbitize.priors as priors

testdir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(testdir, 'GJ504.csv')
input_file_1epoch = os.path.join(testdir, 'GJ504_1epoch.csv')

def test_scale_and_rotate():
    
    myDriver = orbitize.driver.Driver(input_file, 'OFTI',
    1, 1.22, 56.95,mass_err=0.08, plx_err=0.26)
    
    s = myDriver.sampler
    samples = s.prepare_samples(100)
    
    #these have been moved to init
    epochs = np.array(s.tbl['epoch']) # may move to init
    sma,ecc,argp,lan,inc,tau,mtot,plx = [samp for samp in samples]
    epoch_idx = np.argmin(s.sep_err) # may move to init
    
    ra, dec, vc = orbitize.kepler.calc_orbit(epochs, sma, ecc,tau,argp,lan,inc,plx,mtot)
    sep, pa = orbitize.system.radec2seppa(ra, dec)
    sep_sar, pa_sar = np.median(sep[epoch_idx]), np.median(pa[epoch_idx])
    
    assert sep_sar == pytest.approx(s.tbl[epoch_idx]['quant1'], abs=s.tbl[epoch_idx]['quant1_err'])
    assert pa_sar == pytest.approx(s.tbl[epoch_idx]['quant2'], abs=s.tbl[epoch_idx]['quant2_err'])
    
def test_run_sampler():

    myDriver = orbitize.driver.Driver(input_file, 'OFTI',
    1, 1.22, 56.95,mass_err=0.08, plx_err=0.26)
    
    s = myDriver.sampler

    # change eccentricity prior
    myDriver.system.sys_priors[1] = priors.LinearPrior(-2.18, 2.01)
    
    # test num_samples=1
    s.run_sampler(0,num_samples=1)
    
    # test to make sure outputs are reasonable
    orbits = s.run_sampler(1000)
    # should we use s.system.labels for idx??
    sma = np.median([x[0] for x in orbits])
    ecc = np.median([x[1] for x in orbits])
    inc = np.median([x[4] for x in orbits])
    
    sma_exp = 48.
    ecc_exp = 0.19
    inc_exp = np.radians(140)
    
    assert sma == pytest.approx(sma_exp, abs=0.2*sma_exp)
    assert ecc == pytest.approx(ecc_exp, abs=0.2*ecc_exp)
    assert inc == pytest.approx(inc_exp, abs=0.2*inc_exp)
        
    # test with only one epoch
    myDriver = orbitize.driver.Driver(input_file_1epoch, 'OFTI',
    1, 1.22, 56.95,mass_err=0.08, plx_err=0.26)
    s = myDriver.sampler
    s.run_sampler(1)
    

if __name__ == "__main__":
    test_scale_and_rotate()
    test_run_sampler()
    print("Done!")