import numpy as np
import system as sys
import read_input
import pandas as pd

# data = pd.DataFrame({'epoch': np.arange(0, 200, 10), 'object': np.ones(20),
# 'raoff': np.ones(20), 'raoff_err': np.ones(20),
# 'decoff': np.ones(20), 'decoff_err': np.ones(20)},
# columns=['epoch', 'object', 'raoff', 'raoff_err', 'decoff', 'decoff_err', 'rv', 'rv_err'])
#data['object'] = data['object'].astype(int)
#data.to_csv('/Users/Helios/orbitize/tests/mytestdata.csv', index=False)
#data = read_input.read_file('/Users/Helios/orbitize/tests/test_val.csv')
data = read_input.read_file('/Users/Helios/orbitize/tests/mytestdata.csv')
# print(data)  # read_input will only read 'rv','rvß_err' named columns

output_1 = sys.System(2, data, 1.0, 40, mass_err=0.01, fit_secondary_mass=True,
                      tau_ref_epoch=0, gamma_bounds=(-100, 100), jitter_bounds=(0, 20))

print("The radial velocity indices are:", output_1.rv)
#print("The companion radial velocity indices are:", output_1.rv1)
#print("The ra and dec indices are:", output_1.radec)
#print("The separation indices are:", output_1.seppa)

"""semimajor axis 1, eccentricity 1, inclination 1,
argument of periastron 1, position angle of nodes 1,
epoch of periastron passage 1,"""

params_arr = np.array([1.0, 0.2, np.pi/4, np.pi/4, np.pi, 0.0,
                       2.0, 0.1, np.pi/4, np.pi/4, np.pi, 0.0,
                       40, 0.0, 2,
                       0.1, 0.2, 1.0])
# params_arr_2 = np.array([2.0, 0.2, np.pi/4, np.pi/4, np.pi, 0.0, 60, 0.0, 2,
# 0.2, 1.0])

# print(params_arr.shape)

model = output_1.compute_model(params_arr)
# print(model[15:20])  # m2 astrometry
print('m1 rv: ', model[28:])  # m1 rv
print('m2 rv: ', model[11:15])  # m2 rv
print('stellar rv: ', model[20:28])  # stellar rv
# print(model[0:11]) # m1 astrometry
