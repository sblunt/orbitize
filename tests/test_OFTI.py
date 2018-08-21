"""
Test the orbitize.sampler OFTI class which performs OFTI on astrometric data
"""
import pytest
import numpy as np
import orbitize.sampler as sampler
import orbitize.driver

def test_scale_and_rotate():
    
#     myDriver = orbitize.driver.Driver('gsc6214_astrometry.csv', 'OFTI',
#     1, 0.8, 9.19,mass_err=0.1, plx_err=0.043)
    myDriver = orbitize.driver.Driver('GJ504.csv', 'OFTI',
    1, 1.22, 56.95,mass_err=0.08, plx_err=0.26)
    
    s = myDriver.sampler
    samples = s.prepare_samples(100)
    
    #these have been moved to init
    epochs = np.array(s.tbl['epoch']) # may move to init
    sma,ecc,argp,lan,inc,tau,mtot,plx = [s for s in samples]
    epoch_idx = np.argmin(s.sep_err) # may move to init
    
    ra, dec, vc = orbitize.kepler.calc_orbit(epochs, sma, ecc,tau,argp,lan,inc,plx,mtot)
    sep, pa = orbitize.system.radec2seppa(ra, dec)
    sep_sar, pa_sar = np.median(sep[epoch_idx]), np.median(pa[epoch_idx])
    
    assert sep_sar == pytest.approx(s.tbl[epoch_idx]['quant1'], abs=s.tbl[epoch_idx]['quant1_err'])
    assert pa_sar == pytest.approx(s.tbl[epoch_idx]['quant2'], abs=s.tbl[epoch_idx]['quant2_err'])
    
def test_run_sampler():

#     myDriver = orbitize.driver.Driver('gsc6214_astrometry.csv', 'OFTI',
#     1, 0.8, 9.19,mass_err=0.1, plx_err=0.043)
    myDriver = orbitize.driver.Driver('GJ504.csv', 'OFTI',
    1, 1.22, 56.95,mass_err=0.08, plx_err=0.26)
    
    s = myDriver.sampler
    
    #test num_samples=1
    s.run_sampler(0,num_samples=1)
    
    #test to make sure outputs are reasonable
    #look at posteriors in paper, hard code expected values , within 20%?
    #test which uncertainty seems reasonable
    orbits = s.run_sampler(100)
    sma = np.median([x[0] for x in orbits]) # should we use s.system.labels for idx??
    ecc = np.median([x[1] for x in orbits])
    inc = np.median([x[4] for x in orbits])
    
        
    #test with only one epoch
    myDriver = orbitize.driver.Driver('gsc6214_1epoch.csv', 'OFTI',
    1, 0.8, 9.19,mass_err=0.1, plx_err=0.043)
    s = myDriver.sampler
    s.run_sampler(1)
    

if __name__ == "__main__":
    test_scale_and_rotate()
    test_run_sampler()
    print("Done!")