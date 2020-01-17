import numpy as np
import os
import pytest
import matplotlib.pyplot as plt
import time

import orbitize.sampler as sampler
import orbitize.driver
import orbitize.priors as priors
from orbitize.lnlike import chi2_lnlike
from orbitize.kepler import calc_orbit
import orbitize.system
import pdb

testdir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(testdir, 'HD4747.csv')

# perform scale-and-rotate
myDriver = orbitize.driver.Driver(input_file, 'OFTI',
                                  1, 0.84, 53.1836,
                                  mass_err=0.04, plx_err=0.1264,
                                  system_kwargs={'fit_secondary_mass': True, 'tau_ref_epoch': 0}
                                  )

s = myDriver.sampler
samples = s.prepare_samples(10)

if myDriver.system.fit_secondary_mass:
    sma, ecc, inc, argp, lan, tau, plx, gamma, sigma, m1, m0 = [samp for samp in samples]
    ra, dec, vc = orbitize.kepler.calc_orbit(
        s.epochs, sma, ecc, inc, argp, lan, tau, plx, mtot=m1 + m0,
        mass_for_Kamp=m0)
    v_star = vc*-(m1/m0)
else:
    sma, ecc, inc, argp, lan, tau, plx, mtot = [samp for samp in samples]
    ra, dec, vc = orbitize.kepler.calc_orbit(s.epochs, sma, ecc, inc, argp, lan, tau, plx, mtot)

sep, pa = orbitize.system.radec2seppa(ra, dec)
sep_sar, pa_sar = np.median(sep[s.epoch_idx]), np.median(pa[s.epoch_idx])

rv_sar = np.median(v_star[s.epoch_rv_idx[1]])


# test to make sure sep and pa scaled to scale-and-rotate epoch
sar_epoch = s.system.data_table[np.where(
    s.system.data_table['quant_type'] == 'seppa')][s.epoch_idx]
rv_sar_epoch = s.system.data_table[np.where(
    s.system.data_table['quant_type'] == 'rv')][s.epoch_rv_idx[1]]
pdb.set_trace()
# assert sep_sar == pytest.approx(sar_epoch['quant1'], abs=sar_epoch['quant1_err'])  # issue here
#assert pa_sar == pytest.approx(sar_epoch['quant2'], abs=sar_epoch['quant2_err'])

#print('SEP/PA assert tests completed')

#assert rv_sar == pytest.approx(rv_sar_epoch['quant1'], abs=rv_sar_epoch['quant1_err'])

rv_data = myDriver.system.data_table[np.where(
    myDriver.system.data_table['quant_type'] == 'rv')]['quant1'].copy()

rv_data_epochs = myDriver.system.data_table[np.where(
    myDriver.system.data_table['quant_type'] == 'rv')]['epoch'].copy()

# Rob: note-- for HD4747 b, we expect 57Mjup which is about 0.05Msol

data_K = abs(rv_data[s.epoch_rv_idx[1]] - rv_data[s.epoch_rv_idx[0]])
model_K = abs(v_star[s.epoch_rv_idx[1], :] - v_star[s.epoch_rv_idx[0], :])

pdb.set_trace()

# iterating for the plots:
plt.plot(rv_data_epochs, rv_data, 'k.', label='RV data')
for i in range(10):
    plt.plot(s.epochs, v_star[:, i]+samples[i, 7], '.', label='Sample: {}'.format(i))
plt.legend()
plt.ylabel('RV (km/s)')
plt.xlabel('epoch')
plt.show()

# test scale-and-rotate for orbits run all the way through OFTI
# s.run_sampler(10)
