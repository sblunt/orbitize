import sampler
import system
import numpy as np
import lnlike
import read_input
import pandas as pd

data2 = read_input.read_file('/Users/Helios/orbitize/tests/mytestdata2.csv')
data = read_input.read_file('/Users/Helios/orbitize/tests/mytestdata.csv')
output = system.System(1, data, 1.0, 50, mass_err=0.01, plx_err=0.05, fit_secondary_mass=True,
                       tau_ref_epoch=0, gamma_bounds=(-100, 100), jitter_bounds=(1e-3, 20))

mysampler = sampler.MCMC(output, like=lnlike.chi2_lnlike)

params_arr = np.array([10.0, 0.5, np.pi/3, np.pi/6, np.pi/2, 0.0,
                       #20.0, 0.25, np.pi/6, np.pi/12, np.pi, 0.0,
                       #5.0, 0.1, np.pi/4, np.pi/5, np.pi/2, 0.0,
                       50, 0.0, 1.5,
                       0.08,  # m1
                       # 0.5, # m2
                       # 0.2, #m3
                       1.0])

print(mysampler._logl(params_arr))
