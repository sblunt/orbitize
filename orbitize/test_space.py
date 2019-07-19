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
data2 = read_input.read_file('/Users/Helios/orbitize/tests/mytestdata2.csv')
data3 = read_input.read_file('/Users/Helios/orbitize/tests/mytestdata3.csv')
# print(data)  # read_input will only read 'rv','rvß_err' named columns

output = sys.System(2, data2, 3.0, 50, mass_err=0.01, fit_secondary_mass=True,
                    tau_ref_epoch=0, gamma_bounds=(-100, 100), jitter_bounds=(0, 20))

#print("The radial velocity indices are:", output.rv)
#print("The ra and dec indices are:", output.radec)
#print("The separation indices are:", output.seppa)


"""semimajor axis 1, eccentricity 1, inclination 1,
argument of periastron 1, position angle of nodes 1,
epoch of periastron passage 1,"""

params_arr = np.array([10.0, 0.5, np.pi/3, np.pi/6, np.pi/2, 0.0,
                       20.0, 0.25, np.pi/6, np.pi/12, np.pi, 0.0,
                       #5.0, 0.1, np.pi/4, np.pi/5, np.pi/2, 0.0,
                       50, 0.0, 1.5,
                       0.08, 0.5,  # m1,m2
                       # 0.2, #m3
                       3.0])

model, jitter = output.compute_model(params_arr)
print(jitter)

#print('Stellar rv:', model[output.rv[0]])
#print('m1 rv:', model[output.rv[1]])
#print('m2 rv:', model[output.rv[2]])
#print('m3 rv:', model[output.rv[3]])

#print('Stellar astr:', model[output.radec[0]])
#print('m1 astr:', model[output.radec[1]])
#print('m2 astr:', model[output.radec[2]])
#print('m3 astr:', model[output.radec[3]])
