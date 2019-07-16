import numpy as np
import system as sys
import read_input

data = read_input.read_file('/Users/Helios/orbitize/tests/test_val.csv')
print(data)  # read_input will only read 'rv','rv_err' named columns

output_1 = sys.System(1, data, 1.0, 50, mass_err=0.01, fit_secondary_mass=True,
                      gamma_bounds=(-100, 100), jitter_bounds=(0, 20))

print("The stellar radial velocity indices are:", output_1.rv0)
print("The companion radial velocity indices are:", output_1.rv1)
print("The ra and dec indices are:", output_1.radec)
print("The separation indices are:", output_1.seppa)

"""semimajor axis 1, eccentricity 1, inclination 1,
argument of periastron 1, position angle of nodes 1,
epoch of periastron passage 1,"""

params_arr = np.array([100, 0.2, np.pi/4, np.pi, 0.0, 100, 40, 10, 2, 0.1, 1.0])
print(params_arr.shape)

model = output_1.compute_model(params_arr)

# print(model)
